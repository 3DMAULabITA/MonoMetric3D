"""Command line interface.

monometric detect     foto.jpg [--square-mm 40] [-o board.json]
monometric calibrate  cartella_foto/ --board board.json [-o camera.json]
monometric run        foto.jpg --board board.json [--camera camera.json] --models moge2-vitl ...
monometric batch      cartella_foto/ --board board.json --camera camera.json ...
monometric make-board --board board.json --out stampa/          (A3 @ 300 dpi, una pagina per board)
monometric summarize  cartella_report/ -o paper/          (tabelle CSV/DOCX/LaTeX)
monometric figures    cartella_report/ -o paper/figures/  (figure PDF/PNG per l'articolo)
monometric doctor     (verifica GPU/CUDA e profilo hardware)
monometric app        (interfaccia web locale)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def cmd_detect(a):
    from . import charuco
    from .imageio_exif import load_image

    rgb, _ = load_image(a.image)
    cfg, diag = charuco.auto_board_config(rgb, a.square_mm)
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"\nMarker rilevati: {diag['n_markers']}  ratio grezzo: {diag['ratio_raw']:.3f}")
    print(f"Suggerimento lato quadrato per A3 (DA VERIFICARE con calibro): {diag['suggested_square_mm_A3']} mm")
    if a.square_mm is None:
        print("ATTENZIONE: 'square_mm' non impostato. Misura la stampa e passa --square-mm.")
    cfg.save(a.out)
    print(f"Salvato {a.out}")


def cmd_calibrate(a):
    from . import calib, charuco
    from .imageio_exif import focal_prior_from_exif, load_image

    cfg = charuco.BoardConfig.load(a.board)
    if a.square_mm:
        cfg.square_mm = a.square_mm
    groups = {}
    for p in _images(a.folder):
        rgb, ex = load_image(p)
        H, W = rgb.shape[:2]
        key = (ex.get("Make"), ex.get("Model"), ex.get("FocalLength"), max(W, H), min(W, H))
        dets = charuco.detect_charuco(rgb, cfg)
        print(f"{p.name}: {len(dets)} board, corner {[len(d.corner_ids) for d in dets]}")
        groups.setdefault(key, []).append((dets, (W, H), ex))
    outs = {}
    for key, items in groups.items():
        ex, (W, H) = items[0][2], items[0][1]
        prior = focal_prior_from_exif(ex, W, H)
        try:
            c = calib.calibrate_from_set([(d, s) for d, s, _ in items], prior, free_principal_point=a.free_pp)
        except RuntimeError as e:
            print(f"{key}: {e}")
            continue
        c["camera"] = dict(make=key[0], model=key[1], focal_mm=key[2], n_images=len(items))
        name = f"{key[0]}_{key[1]}_{key[2]}mm".replace(" ", "_")
        outs[name] = c
        print(f"\n[{name}] f={c['f']:.2f}px ±{100 * c['sigma_f_rel']:.2f}%  k1={c['k1']:.5f} k2={c['k2']:.5f} "
              f"modello={c['model']} RMS={c['rms_px']:.3f}px viste={c['n_views']} (prior EXIF {c['prior_f']:.1f})")
    out = Path(a.out)
    if len(outs) == 1:
        out.write_text(json.dumps(next(iter(outs.values())), indent=2))
    else:
        out.write_text(json.dumps(outs, indent=2))
        print("Più camere/focali trovate: il file contiene un dizionario, estrai la voce desiderata.")
    print(f"Salvato {out}")


def _run_one(a, image, out_dir):
    from . import charuco, pipeline

    cfg = charuco.BoardConfig.load(a.board) if a.board else None
    return pipeline.run(image, cfg, square_mm=a.square_mm, models=a.models, families=a.align, fuse=not a.no_fuse,
                        stride=a.stride, edge_thr=a.edge, max_depth=a.max_depth, distortion=a.distortion,
                        camera_calib=a.camera, frame=a.frame, gcp_csv=a.gcp, formats=a.formats, out_dir=out_dir,
                        profile=a.profile, infer_max_mp=a.max_mp, site=a.site, tag=a.tag,
                        focal_sweep=[float(x) / 100 for x in a.focal_sweep] if a.focal_sweep else None)


def cmd_run(a):
    S = _run_one(a, a.image, a.out)
    print(json.dumps(dict(chosen=S["chosen"], cv=S["chosen_cv"], files=S["files"]), indent=2))


def cmd_batch(a):
    rows = []
    for p in _images(a.folder):
        try:
            S = _run_one(a, p, Path(a.out) / p.stem)
            c = S["chosen_cv"] or {}
            rows.append((p.name, S["chosen"], c.get("rmse3d_mm"), c.get("rel_depth_err_pct"), S["n_points"]))
        except Exception as e:  # keep going
            rows.append((p.name, f"ERRORE: {e}", None, None, 0))
    print("\nfoto | soluzione | LOBO RMSE3D mm | err rel Z % | punti")
    for r in rows:
        print(" | ".join("–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r))


def cmd_make_board(a):
    import cv2

    from . import charuco

    cfg = charuco.BoardConfig.load(a.board)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for off in cfg.offsets:
        page = charuco.generate_board_image(cfg, off, dpi=a.dpi)
        p = out / f"charuco_{cfg.dictionary}_{cfg.squares_x}x{cfg.squares_y}_ids{off}.png"
        cv2.imwrite(str(p), page)
        print(p)
    print("Stampare al 100% (nessun adattamento alla pagina) e MISURARE il lato reale dei quadrati.")


def cmd_app(a):
    from . import device
    from .app import launch

    device.set_profile(a.profile)
    launch(a.host, a.port, a.share, a.auth)


def cmd_summarize(a):
    from .paper import summarize

    for k, v in summarize(a.folder, a.out).items():
        print(f"{k}: {v}")


def cmd_figures(a):
    from .paper import figures

    r = figures(a.folder, a.out, a.run, a.condition)
    print(f"Figure create in {r['folder']} (font: {r['font']}; condizione per le figure singole: {r['condition']}):")
    for f in r["figures"]:
        print("  ", f + ".pdf / .png")
    print("Didascalie suggerite: captions.txt")


def cmd_doctor(a):
    """Hardware / installation check."""
    import platform

    from . import backends, device

    print(f"Python {platform.python_version()} su {platform.system()} {platform.release()}")
    try:
        import torch

        print(f"PyTorch {torch.__version__}  CUDA build {torch.version.cuda}  CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU {p.name}  {p.total_memory / 2**30:.1f} GB  compute capability {p.major}.{p.minor}")
            arch = f"sm_{p.major}{p.minor}"
            if arch not in torch.cuda.get_arch_list():
                print(f"ATTENZIONE: {arch} non è tra le architetture di questa build ({torch.cuda.get_arch_list()}): "
                      "reinstalla PyTorch cu128.")
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            print("Test calcolo fp16 su GPU: OK", float((x @ x).float().abs().mean()) > 0)
        else:
            print("Nessuna GPU CUDA: inferenza su CPU (lenta). Su notebook verifica che la GPU NVIDIA sia attiva "
                  "e che PyTorch sia la build cu128.")
    except ImportError:
        print("PyTorch NON installato.")
    print("Profilo:", device.set_profile(a.profile).describe())
    print("Backend installati:", backends.available())


def main(argv=None):
    from . import CREDITS, __version__

    ap = argparse.ArgumentParser("monometric", description="Nuvola di punti metrica da singola foto + ChArUco",
                                 epilog=CREDITS)
    ap.add_argument("--version", action="version", version=f"MonoMetric 3D {__version__} — {CREDITS}")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("detect", help="inferisce automaticamente la configurazione delle board")
    p.add_argument("image")
    p.add_argument("--square-mm", type=float)
    p.add_argument("-o", "--out", default="board.json")
    p.set_defaults(fn=cmd_detect)

    p = sp.add_parser("calibrate", help="calibrazione camera dal set di foto del rilievo")
    p.add_argument("folder")
    p.add_argument("--board", required=True)
    p.add_argument("--square-mm", type=float)
    p.add_argument("--free-pp", action="store_true", help="stima anche il punto principale (foto tutte con stesso orientamento)")
    p.add_argument("-o", "--out", default="camera.json")
    p.set_defaults(fn=cmd_calibrate)

    def common(p):
        p.add_argument("--board", help="board.json (se assente: rilevamento automatico)")
        p.add_argument("--square-mm", type=float)
        p.add_argument("--camera", help="camera.json da 'calibrate' (fortemente consigliato)")
        p.add_argument("--models", nargs="+", help="default dal profilo hardware (moge2-vitl)")
        p.add_argument("--profile", help="auto | workstation | laptop | lowmem | cpu (default auto)")
        p.add_argument("--max-mp", type=float, help="megapixel massimi in ingresso alla rete (default dal profilo)")
        p.add_argument("--align", default="auto", help="auto|scale|affine|disp_affine|field1|field2")
        p.add_argument("--no-fuse", action="store_true")
        p.add_argument("--stride", type=int, help="passo di campionamento della nuvola (default dal profilo)")
        p.add_argument("--edge", type=float, default=0.02, help="soglia relativa discontinuità (0 = off)")
        p.add_argument("--max-depth", type=float)
        p.add_argument("--distortion", default="auto", help="auto|none|k1|k1k2 (solo senza --camera)")
        p.add_argument("--frame", default="camera", help="camera | board:<n>")
        p.add_argument("--gcp", help="CSV punti rilevati (type,name,board,corner,u,v,X,Y,Z)")
        p.add_argument("--formats", nargs="+", default=["ply"], help="ply las e57")
        p.add_argument("--site", help="etichetta del sito (default: nome della cartella della foto)")
        p.add_argument("--tag", help="etichetta della condizione sperimentale (es. 'MoGe-2 anchored')")
        p.add_argument("--focal-sweep", nargs="+", metavar="PCT",
                       help="analisi di sensibilità: errori di focale in %%, es. -5 -2 -1 1 2 5")
        p.add_argument("-o", "--out", default="out")

    p = sp.add_parser("run", help="ricostruzione da una foto")
    p.add_argument("image")
    common(p)
    p.set_defaults(fn=cmd_run)

    p = sp.add_parser("batch", help="ricostruzione di tutte le foto di una cartella (una nuvola per foto)")
    p.add_argument("folder")
    common(p)
    p.set_defaults(fn=cmd_batch)

    p = sp.add_parser("make-board", help="genera le tavole A3 da stampare")
    p.add_argument("--board", required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", default="boards_print")
    p.set_defaults(fn=cmd_make_board)

    p = sp.add_parser("summarize", help="tabelle per l'articolo (CSV, DOCX, LaTeX) da tutti i report di una cartella")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="paper")
    p.set_defaults(fn=cmd_summarize)

    p = sp.add_parser("figures", help="figure per l'articolo (PDF + PNG 300 dpi, colonne ISPRS 82/170 mm)")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="paper/figures")
    p.add_argument("--run", help="nome della foto per il pannello qualitativo (default: la prima)")
    p.add_argument("--condition", help="condizione per le figure a condizione singola (default: la più accurata)")
    p.set_defaults(fn=cmd_figures)

    p = sp.add_parser("doctor", help="verifica hardware, PyTorch/CUDA e profilo scelto")
    p.add_argument("--profile")
    p.set_defaults(fn=cmd_doctor)

    p = sp.add_parser("app", help="interfaccia web locale (Gradio)")
    p.add_argument("--profile", help="auto | workstation | laptop | lowmem | cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="link pubblico temporaneo *.gradio.live (72 ore)")
    p.add_argument("--auth", help="utente:password (più utenti separati da virgola); obbligatorio se esposta in rete")
    p.set_defaults(fn=cmd_app)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
