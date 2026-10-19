"""Local web GUI (Gradio). Launch with:  monometric app   ->  http://127.0.0.1:7860

Tabs
  1. Ricostruzione        one photo, step by step (detect boards -> reconstruct), 3D preview
  2. Campagna             many photos with the same settings (+ optional zero-shot baseline, focal sweep)
  3. Calibrazione         camera.json from the survey photos
  4. Articolo             tables (CSV/DOCX/LaTeX) and figures (PDF/PNG, ISPRS columns) from the results folder

Results are written to a persistent folder (default ./risultati/<sito>/<condizione>/<foto>/) together with a
copy of the photo, so that `Articolo` (or `monometric summarize/figures`) can collect them later.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from . import AUTHORS, PROJECT, __version__, backends, calib, charuco, device, pipeline
from .imageio_exif import focal_prior_from_exif, load_image

DICTS = [n for fam in charuco.DICT_FAMILIES.values() for n in fam]
FAMS = ["auto", "none (zero-shot)", "scale", "affine", "disp_affine", "field1", "field2"]
DEFAULT_ROOT = "risultati"


def _cfg_from_fields(dictionary, sx, sy, ratio, legacy, offsets, square_mm):
    offs = [int(x) for x in str(offsets).replace(";", ",").split(",") if str(x).strip() != ""]
    return charuco.BoardConfig(dictionary=dictionary, squares_x=int(sx), squares_y=int(sy), marker_ratio=float(ratio),
                               legacy_pattern=bool(legacy), square_mm=float(square_mm) if square_mm else None,
                               markers_per_board=(int(sx) * int(sy)) // 2, offsets=offs)


def _slug(s, default):
    s = re.sub(r"[^\w\-. ]+", "_", str(s or "").strip()).strip().replace(" ", "_")
    return s or default


def _family(f):
    return "none" if str(f).startswith("none") else f


def _sweep(text):
    if not text or not str(text).strip():
        return None
    return [float(x) / 100 for x in re.split(r"[ ,;]+", str(text).strip()) if x]


def _tag(tag, family, models):
    if tag and str(tag).strip():
        return str(tag).strip()
    m = models[0] if models else "model"
    return f"{m} {'zero-shot' if _family(family) == 'none' else 'anchored'}"


def _out_dir(root, site, tag, photo):
    stem = Path(photo).stem
    d = Path(root or DEFAULT_ROOT) / _slug(site, "sito") / _slug(tag, "condizione") / stem
    d.mkdir(parents=True, exist_ok=True)
    dst = d / Path(photo).name
    if not dst.exists():
        shutil.copy2(photo, dst)  # keep the original photo (with EXIF) next to its results
    return d, dst


# ----------------------------------------------------------------------------- 1. single photo
def ui_detect(photo, square_mm):
    import gradio as gr

    if photo is None:
        raise gr.Error("Carica una foto.")
    rgb, ex = load_image(photo)
    try:
        cfg, diag = charuco.auto_board_config(rgb, square_mm or None)
    except RuntimeError as e:
        raise gr.Error(str(e))
    sq = square_mm or diag["suggested_square_mm_A3"]
    cfg.square_mm = sq
    dets = charuco.detect_charuco(rgb, cfg)
    vis = charuco.draw_detections(rgb, dets)
    H, W = rgb.shape[:2]
    pr = focal_prior_from_exif(ex, W, H)
    msg = (f"**{len(dets)} board** rilevate · corner {[len(d.corner_ids) for d in dets]} · "
           f"{ex.get('Make', '')} {ex.get('Model', '')} {W}×{H} · prior focale {pr.fx:.0f} px ({pr.source})\n\n"
           + ("" if square_mm else f"⚠️ Lato quadrato **ipotizzato {sq} mm**: misura la stampa con un calibro e "
                                   "correggi il valore.")
           + ("\n\n⚠️ Nessun EXIF di focale: carica la foto originale (non ridotta o inviata da chat) "
              "oppure una calibrazione camera.json." if pr.source == "generic_guess" else ""))
    return (vis, cfg.dictionary, cfg.squares_x, cfg.squares_y, cfg.marker_ratio, cfg.legacy_pattern,
            ",".join(map(str, cfg.offsets)), sq, msg)


def _run_one(photo, cfg, square_mm, camera_file, models, family, fuse, stride, edge, frame, formats, gcp_file,
             root, site, tag, sweep, logs):
    tagv = _tag(tag, family, models)
    out, local = _out_dir(root, site, tagv, photo)
    return pipeline.run(str(local), cfg, square_mm=square_mm, models=models, families=_family(family), fuse=fuse,
                        stride=int(stride), edge_thr=float(edge), camera_calib=camera_file, frame=frame,
                        gcp_csv=gcp_file, formats=tuple(formats or ["ply"]), out_dir=out, progress=logs.append,
                        site=_slug(site, "sito"), tag=tagv, focal_sweep=sweep)


def ui_run(photo, dictionary, sx, sy, ratio, legacy, offsets, square_mm, camera_file, models, family, fuse, stride,
           edge, frame, formats, gcp_file, root, site, tag, sweep_txt):
    import gradio as gr

    if photo is None:
        raise gr.Error("Carica una foto.")
    if not square_mm:
        raise gr.Error("Inserisci il lato fisico del quadrato (mm).")
    if not models:
        raise gr.Error("Seleziona almeno un modello.")
    cfg = _cfg_from_fields(dictionary, sx, sy, ratio, legacy, offsets, square_mm)
    logs = []
    try:
        S = _run_one(photo, cfg, square_mm, camera_file, models, family, fuse, stride, edge, frame, formats, gcp_file,
                     root, site, tag, _sweep(sweep_txt), logs)
    except Exception as e:  # show the reason in the page instead of a generic error
        raise gr.Error(f"{type(e).__name__}: {e}")
    rows = []
    for k, m in S["models"].items():
        for fam, v in (m["cv"] or {}).items():
            s = (v or {}).get("summary") or {}
            rows.append([k, fam + (" ✔" if fam == m["family"] else ""), _r(s.get("rmse3d_mm")), _r(s.get("mae3d_mm")),
                         _r(s.get("radial_rmse_mm")), _r(s.get("lateral_rmse_mm")), _r(s.get("rel_depth_err_pct")),
                         _r(s.get("dist_scale_err_pct"))])
    c = S["chosen_cv"] or {}
    cam = S["camera"]
    md = (f"### Soluzione: `{S['chosen']}` · allineamento `{S['models'][S['chosen']]['family']}` · condizione *{S['tag']}*\n"
          f"- RMSE 3D su board escluse (LOBO): **{_r(c.get('rmse3d_mm'))} mm** (radiale {_r(c.get('radial_rmse_mm'))}, "
          f"laterale {_r(c.get('lateral_rmse_mm'))}) · errore relativo Z **{_r(c.get('rel_depth_err_pct'))} %**\n"
          f"- Errore di scala sulle distanze: **{_r(c.get('dist_scale_err_pct'))} %**\n"
          f"- Focale {cam['fx']:.1f} px ± {100 * cam['sigma_f_rel']:.2f} % ({cam.get('source')}) · distorsione {cam['distortion_model']}\n"
          f"- σ stimata per punto: **{100 * S['sigma_rel_point']:.2f} % della distanza** · {S['n_points']:,} punti · {S['runtime_s']} s")
    if S.get("rejected_boards"):
        md += f"\n- Board scartate (riproiezione anomala): {[b['board'] for b in S['rejected_boards']]}"
    if cam["sigma_f_rel"] > 0.01:
        md += ("\n\n⚠️ Intrinseci stimati dalla sola foto: errore sistematico in profondità dell'ordine "
               f"del {100 * cam['sigma_f_rel']:.1f} %. Usa la scheda *Calibrazione*.")
    if S.get("gcp"):
        g = S["gcp"]
        md += (f"\n\n**Check point indipendenti** ({g.get('n_check', 0)}): MAE {_r(g.get('check_mae_mm'))} mm · "
               f"RMSE {_r(g.get('check_rmse_mm'))} mm · D90 {_r(g.get('check_d90_mm'))} mm · radiale "
               f"{_r(g.get('check_radial_rmse_mm'))} / laterale {_r(g.get('check_lateral_rmse_mm'))} mm · "
               f"controlli RMSE {g['control_rmse_mm']:.1f} mm")
    if S.get("focal_sweep"):
        md += "\n\n**Sensibilità alla focale**: " + " · ".join(
            f"ε {100 * s['eps']:+.0f}% → ΔZ/Z {100 * s['axial_rel_median']:+.2f}%" for s in S["focal_sweep"])
    out_dir = str(Path(S["files"]["report_json"]).parent.resolve())
    md += f"\n\nRisultati salvati in `{out_dir}`"
    files = [v for k, v in S["files"].items() if isinstance(v, str) and Path(v).exists() and k != "glb"]
    glb = S["files"].get("glb")
    return (glb if glb and Path(glb).exists() else None, md, rows, files, "\n".join(logs))


# ----------------------------------------------------------------------------- 2. campaign
def ui_campaign(photos, dictionary, sx, sy, ratio, legacy, offsets, square_mm, camera_file, models, family, fuse,
                stride, edge, frame, formats, root, site, tag, also_zero, sweep_txt, sweep_first, gcp_files,
                progress=None):
    import gradio as gr

    if not photos:
        raise gr.Error("Carica le foto della campagna.")
    if not square_mm:
        raise gr.Error("Inserisci il lato fisico del quadrato (mm).")
    if not models:
        raise gr.Error("Seleziona almeno un modello.")
    cfg = _cfg_from_fields(dictionary, sx, sy, ratio, legacy, offsets, square_mm)
    gcp_by_stem = {}
    for g in gcp_files or []:  # GCP file named like the photo: IMG_0012.csv or IMG_0012_gcp.csv
        st = Path(g).stem
        gcp_by_stem[st[:-4] if st.lower().endswith("_gcp") else st] = g
    conds = [(family, tag)]
    if also_zero and _family(family) != "none":
        conds.append(("none (zero-shot)", f"{models[0]} zero-shot"))
    sweep = _sweep(sweep_txt)
    logs, rows = [], []
    track = progress or (lambda *a, **k: None)
    n = len(photos) * len(conds)
    k = 0
    for p in photos:
        gcp = gcp_by_stem.get(Path(p).stem)
        for fam, tg in conds:
            k += 1
            track(k / n, desc=f"{Path(p).name} · {_tag(tg, fam, models)}")
            lg = []
            try:
                S = _run_one(p, cfg, square_mm, camera_file, models, fam, fuse, stride, edge, frame, formats, gcp,
                             root, site, tg, sweep if (sweep and (not sweep_first or p == photos[0])) else None, lg)
                c, g = S["chosen_cv"] or {}, S.get("gcp") or {}
                rows.append([Path(p).name, S["tag"], len(S["detections"]), _r(c.get("rmse3d_mm")),
                             _r(g.get("check_rmse_mm")), _r(100 * S["sigma_rel_point"]), S["runtime_s"]])
            except Exception as e:
                lg.append(f"ERRORE: {type(e).__name__}: {e}")
                rows.append([Path(p).name, _tag(tg, fam, models), 0, None, None, None, None])
            logs.append(f"--- {Path(p).name} · {_tag(tg, fam, models)}\n" + "\n".join(lg))
    folder = str((Path(root or DEFAULT_ROOT) / _slug(site, "sito")).resolve())
    return rows, "\n".join(logs), f"Risultati in `{folder}`. Ora puoi generare tabelle e figure nella scheda *Articolo*."


# ----------------------------------------------------------------------------- 3. calibration
def ui_calibrate(photos, dictionary, sx, sy, ratio, legacy, offsets, square_mm, root):
    import gradio as gr

    if not photos:
        raise gr.Error("Carica le foto del rilievo (stessa camera e focale).")
    cfg = _cfg_from_fields(dictionary, sx, sy, ratio, legacy, offsets, square_mm)
    if not cfg.square_mm:
        raise gr.Error("Inserisci il lato fisico del quadrato (mm).")
    items, log, ex0 = [], [], None
    for p in photos:
        rgb, ex = load_image(p)
        ex0 = ex0 or (ex, rgb.shape[1], rgb.shape[0])
        d = charuco.detect_charuco(rgb, cfg)
        items.append((d, (rgb.shape[1], rgb.shape[0])))
        log.append(f"{Path(p).name}: {len(d)} board {[len(x.corner_ids) for x in d]}")
    try:
        c = calib.calibrate_from_set(items, focal_prior_from_exif(*ex0))
    except RuntimeError as e:
        raise gr.Error(str(e))
    ex = ex0[0]
    name = _slug(f"camera_{ex.get('Make', '')}_{ex.get('Model', '')}_{ex.get('FocalLength', '')}mm", "camera")
    out = Path(root or DEFAULT_ROOT) / "calibrazioni" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=2))
    md = (f"**f = {c['f']:.2f} px ± {100 * c['sigma_f_rel']:.2f} %** · k1 = {c['k1']:.5f} · k2 = {c['k2']:.5f} · "
          f"modello `{c['model']}` · RMS {c['rms_px']:.3f} px · {c['n_views']} viste · prior EXIF {c['prior_f']:.1f} px"
          f"\n\nSalvata in `{out.resolve()}` e già collegata alle schede di ricostruzione.")
    return str(out), md, "\n".join(log), str(out)


# ----------------------------------------------------------------------------- 4. paper
def ui_paper(root, out, cond, run):
    import gradio as gr

    from . import paper

    src = Path(root or DEFAULT_ROOT)
    if not src.exists():
        raise gr.Error(f"Cartella risultati non trovata: {src.resolve()}")
    out = Path(out or "paper")
    try:
        tabs = paper.summarize(src, out)
        figs = paper.figures(src, out / "figures", run or None, cond or None)
    except RuntimeError as e:
        raise gr.Error(str(e))
    pngs = [str(out / "figures" / f"{f}.png") for f in figs["figures"]]
    files = [v for v in tabs.values() if Path(str(v)).exists()]
    files += [str(out / "figures" / f"{f}.pdf") for f in figs["figures"]]
    caps = (out / "figures" / "captions.txt").read_text(encoding="utf-8")
    md = (f"**{len(figs['figures'])} figure** e **tabelle** create in `{out.resolve()}` · font {figs['font']} · "
          f"condizione per le figure singole: *{figs['condition']}*.\n\n"
          "Inserisci i PDF nel template al 100 % (larghezza 82 mm = una colonna, 170 mm = pagina).")
    return [(p, Path(p).stem) for p in pngs], files, caps, md


def _conditions(root):
    from . import paper

    try:
        return sorted({paper.condition(r) for r in paper.load_reports(Path(root or DEFAULT_ROOT))})
    except Exception:
        return []


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


# ----------------------------------------------------------------------------- layout
def build():
    import gradio as gr

    model_choices = list(backends.REGISTRY)
    installed = backends.available()
    prof = device.active()
    default_models = [m for m in ("moge3-vitl",) + tuple(prof.default_models) if m in installed][:1] or installed[:1]

    with gr.Blocks(title="MonoMetric 3D") as demo:
        gr.Markdown("# MonoMetric 3D\nNuvola di punti metrica da **singola foto** con target ChArUco "
                    "· modello monoculare + ancoraggio ChArUco + validazione Leave-One-Board-Out\n\n"
                    f"<small>**{PROJECT}** · Authors: {AUTHORS} · 2026 – Beta Version {__version__}<br>"
                    f"Hardware: {prof.describe()}</small>")
        cam_state = gr.State(None)
        with gr.Accordion("Progetto e configurazione board", open=True):
            with gr.Row():
                root = gr.Textbox(value=DEFAULT_ROOT, label="Cartella risultati")
                site = gr.Textbox(value="sito1", label="Sito (es. laboratorio, cortile, facciata)")
                camera_file = gr.File(label="Calibrazione camera.json (consigliata)", type="filepath")
            with gr.Row():
                dictionary = gr.Dropdown(DICTS, value="DICT_4X4_250", label="Dizionario")
                square_mm = gr.Number(label="Lato quadrato misurato [mm]", value=None)
                sx = gr.Number(label="Quadrati X", value=7, precision=0)
                sy = gr.Number(label="Quadrati Y", value=5, precision=0)
                ratio = gr.Number(label="Marker/quadrato", value=0.75)
                legacy = gr.Checkbox(label="Legacy pattern", value=False)
                offsets = gr.Textbox(label="ID iniziale di ogni board", value="0")
        with gr.Accordion("Elaborazione", open=False):
            with gr.Row():
                models = gr.CheckboxGroup(model_choices, value=default_models,
                                          label=f"Modelli (installati: {', '.join(installed) or 'nessuno'})")
            with gr.Row():
                family = gr.Dropdown(FAMS, value="auto", label="Allineamento ('none' = baseline zero-shot)")
                tag = gr.Textbox(value="", label="Etichetta condizione (vuota = automatica)")
                fuse = gr.Checkbox(value=True, label="Fusione multi-modello")
            with gr.Row():
                stride = gr.Slider(1, 8, value=prof.default_stride, step=1, label="Passo campionamento [px]")
                edge = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Soglia discontinuità")
                frame = gr.Textbox(value="camera", label="Sistema di riferimento (camera | board:N)")
                formats = gr.CheckboxGroup(["ply", "las", "e57"], value=["ply"], label="Formati")
            sweep_txt = gr.Textbox(value="", label="Sensibilità focale: errori in % (es. -5 -2 -1 1 2 5; vuoto = no)")
        settings = [dictionary, sx, sy, ratio, legacy, offsets, square_mm, camera_file, models, family, fuse, stride,
                    edge, frame, formats]

        with gr.Tabs():
            # ------------------------------------------------------------------ 1
            with gr.Tab("1 · Ricostruzione da singola foto"):
                with gr.Row():
                    with gr.Column(scale=1):
                        photo = gr.File(label="Foto originale (JPG/HEIC/TIFF, EXIF preservati)", type="filepath")
                        btn_det = gr.Button("Rileva board", variant="secondary")
                        gcp = gr.File(label="GCP / check point CSV (opzionale)", type="filepath")
                        btn_run = gr.Button("Ricostruisci", variant="primary")
                    with gr.Column(scale=2):
                        det_msg = gr.Markdown()
                        overlay = gr.Image(label="Rilevamento ChArUco", type="numpy")
                        viewer = gr.Model3D(label="Anteprima nuvola (decimata)", clear_color=[0.1, 0.1, 0.1, 1])
                        metrics = gr.Markdown()
                        table = gr.Dataframe(headers=["Modello", "Famiglia", "RMSE3D mm", "MAE mm", "Radiale mm",
                                                      "Laterale mm", "Err Z %", "Err scala %"], label="Validazione LOBO")
                        files = gr.File(label="Download", file_count="multiple")
                        log = gr.Textbox(label="Log", lines=6)
                btn_det.click(ui_detect, [photo, square_mm],
                              [overlay, dictionary, sx, sy, ratio, legacy, offsets, square_mm, det_msg])
                btn_run.click(ui_run, [photo] + settings + [gcp, root, site, tag, sweep_txt],
                              [viewer, metrics, table, files, log])

            # ------------------------------------------------------------------ 2
            with gr.Tab("2 · Campagna (più foto)"):
                gr.Markdown("Elabora tutte le foto di un sito con le impostazioni sopra. Ogni foto è ricostruita "
                            "**da sola**. I file GCP vanno chiamati come la foto (`IMG_0012.csv` o `IMG_0012_gcp.csv`). "
                            "Prima usa *Rileva board* nella scheda 1 su una foto, oppure compila la configurazione.")
                c_photos = gr.File(label="Foto", file_count="multiple", type="filepath")
                c_gcps = gr.File(label="File GCP (opzionali, uno per foto)", file_count="multiple", type="filepath")
                with gr.Row():
                    also_zero = gr.Checkbox(value=True, label="Esegui anche la baseline zero-shot (stesso modello)")
                    sweep_first = gr.Checkbox(value=True, label="Sensibilità focale solo sulla prima foto")
                btn_camp = gr.Button("Elabora campagna", variant="primary")
                c_msg = gr.Markdown()
                c_table = gr.Dataframe(headers=["Foto", "Condizione", "Board", "LOBO RMSE3D mm", "Check RMSE mm",
                                                "σ %", "Tempo s"], label="Riepilogo")
                c_log = gr.Textbox(label="Log", lines=10)
                def _camp(*args, progress=gr.Progress()):
                    return ui_campaign(*args, progress=progress)

                btn_camp.click(_camp, [c_photos] + settings + [root, site, tag, also_zero, sweep_txt, sweep_first,
                                                                     c_gcps],
                               [c_table, c_log, c_msg])

            # ------------------------------------------------------------------ 3
            with gr.Tab("3 · Calibrazione dal set di foto"):
                gr.Markdown("Carica **tutte le foto fatte con la stessa camera e la stessa focale** "
                            "(iPhone 12 e reflex vanno calibrate separatamente). Ogni board in ogni foto è una vista "
                            "di calibrazione: f, k1, k2 risultano molto più affidabili della stima da singola immagine.")
                cal_photos = gr.File(label="Foto", file_count="multiple", type="filepath")
                btn_cal = gr.Button("Calibra", variant="primary")
                cal_md = gr.Markdown()
                cal_file = gr.File(label="camera.json")
                cal_log = gr.Textbox(label="Log", lines=8)
                btn_cal.click(ui_calibrate, [cal_photos, dictionary, sx, sy, ratio, legacy, offsets, square_mm, root],
                              [cal_file, cal_md, cal_log, cam_state]).then(lambda p: p, cam_state, camera_file)

            # ------------------------------------------------------------------ 4
            with gr.Tab("4 · Articolo: tabelle e figure"):
                gr.Markdown("Raccoglie **tutti i report** nella cartella risultati e crea le tabelle (CSV, DOCX, LaTeX) "
                            "e le figure (PDF vettoriale + PNG 300 dpi) dimensionate sulle colonne ISPRS "
                            "(82 / 170 mm, Times New Roman 8 pt, etichette in inglese, senza didascalie interne).")
                with gr.Row():
                    p_out = gr.Textbox(value="paper", label="Cartella di uscita")
                    p_cond = gr.Dropdown([], value=None, allow_custom_value=True,
                                         label="Condizione per le figure singole (vuoto = la più accurata)")
                    p_run = gr.Textbox(value="", label="Foto per il pannello qualitativo (nome senza estensione)")
                with gr.Row():
                    btn_refresh = gr.Button("Aggiorna elenco condizioni")
                    btn_paper = gr.Button("Genera tabelle e figure", variant="primary")
                p_msg = gr.Markdown()
                gallery = gr.Gallery(label="Figure", columns=3, height="auto", object_fit="contain")
                p_files = gr.File(label="Tabelle e PDF", file_count="multiple")
                p_caps = gr.Textbox(label="Didascalie suggerite (captions.txt)", lines=8)
                btn_refresh.click(lambda r: gr.Dropdown(choices=_conditions(r)), root, p_cond)
                btn_paper.click(ui_paper, [root, p_out, p_cond, p_run], [gallery, p_files, p_caps, p_msg])
        gr.Markdown(f"<div style='text-align:center;opacity:.75;font-size:0.85em;margin-top:1.5em'>"
                    f"<b>{PROJECT}</b><br>Authors: {AUTHORS}<br>2026 – Beta Version {__version__}</div>")
    return demo


def launch(host="127.0.0.1", port=7860, share=False, auth=None):
    """auth: 'utente:password' (or env MONOMETRIC_AUTH). Always set it when the app is reachable from the
    internet (share link, tunnel, server): the app writes files and uses the GPU of this computer."""
    import os

    auth = auth or os.environ.get("MONOMETRIC_AUTH")
    creds = None
    if auth:
        creds = [tuple(a.split(":", 1)) for a in auth.split(",") if ":" in a]
    elif share or host not in ("127.0.0.1", "localhost"):
        print("ATTENZIONE: app raggiungibile dalla rete senza password. Usa --auth utente:password.")
    build().queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share, auth=creds,
                                                       auth_message=f"MonoMetric 3D · {AUTHORS}")


if __name__ == "__main__":
    launch()
