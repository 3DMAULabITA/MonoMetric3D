"""Self-contained HTML accuracy report."""
from __future__ import annotations

import base64
import html

import cv2
import numpy as np


def _img_b64(rgb, max_side=1400, colormap=False):
    if colormap:
        Z = rgb
        v = Z[Z > 0]
        lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0, 1)
        n = np.clip((Z - lo) / max(hi - lo, 1e-6), 0, 1)
        im = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
        im[Z <= 0] = 0
    else:
        im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s = min(1.0, max_side / max(im.shape[:2]))
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def _table(rows, cols, heads):
    h = "".join(f"<th>{x}</th>" for x in heads)
    b = "".join("<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e4e3de;--acc:#1f6f8b;--good:#2e7d32;--warn:#b26a00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--acc:#6fb7d1;--good:#7bc47f;--warn:#e0a84a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th{color:var(--mut);font-weight:600}
td:first-child,th:first-child{text-align:left}.kpi{display:flex;gap:12px;flex-wrap:wrap}.k{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}
.k b{display:block;font-size:22px}.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}img{width:100%;border-radius:6px}
.note{border-left:3px solid var(--warn);padding:6px 10px;margin:10px 0}
"""


def write_html(path, S, overlay_rgb, Zbest, results):
    cam = S["camera"]
    cv = S.get("chosen_cv") or {}
    k = []
    k.append(("RMSE 3D LOBO", f"{cv.get('rmse3d_mm', float('nan')):.1f} mm" if cv else "n.d."))
    k.append(("Errore rel. profondità", f"{cv.get('rel_depth_err_pct', float('nan')):.2f} %" if cv else "n.d."))
    k.append(("Errore scala distanze", f"{cv.get('dist_scale_err_pct', float('nan')):.2f} %" if cv else "n.d."))
    k.append(("σ focale", f"{100 * cam['sigma_f_rel']:.2f} %"))
    if S.get("sigma_rel_point") is not None:
        k.append(("σ stimata punti (1σ)", f"{100 * S['sigma_rel_point']:.2f} % · Z"))
    k.append(("Punti", f"{S['n_points']:,}".replace(",", ".")))
    kpi = "".join(f"<div class='k'><span class='mut'>{a}</span><b>{b}</b></div>" for a, b in k)

    notes = []
    if not cv:
        notes.append("Una sola board utilizzabile: nessuna validazione indipendente possibile. "
                     "Collocare almeno 3–4 target a distanze diverse.")
    if cam["sigma_f_rel"] > 0.01:
        notes.append(f"Incertezza sulla focale {100 * cam['sigma_f_rel']:.1f}%: si traduce in un errore sistematico "
                     "della stessa entità lungo la profondità, NON visibile nella validazione LOBO (che misura la coerenza "
                     "con gli ancoraggi). Usare la calibrazione multi-foto del set di immagini (comando 'calibrate').")
    if cam.get("f_boards_only") and S["focal_prior"]["source"] != "generic_guess":
        z = (cam["f_boards_only"] / S["focal_prior"]["fx"] - 1)
        if abs(z) > 3 * S["focal_prior"]["sigma_rel"]:
            notes.append(f"La focale stimata dalle sole board differisce del {100 * z:.1f}% dal valore EXIF: "
                         "verificare il lato del quadrato inserito e il ritaglio/ridimensionamento della foto.")

    mrows = []
    for key, m in S["models"].items():
        for fam, v in (m["cv"] or {}).items():
            s = (v or {}).get("summary") or {}
            mrows.append(dict(model=key, family=fam + (" ✔" if fam == m["family"] else ""), **s))
    per_board = []
    ch = S["models"][S["chosen"]]
    pb = (ch["cv"].get(ch["family"]) or {}).get("per_board") or []
    for r in pb:
        per_board.append(r)

    det_tab = _table(S["detections"], ["board", "offset", "n_markers", "n_corners", "reproj_rms_px", "distance_m", "anchor_sigma_pct"],
                     ["Board", "ID iniziale", "Marker", "Corner", "Reproj. RMS [px]", "Distanza [m]", "σ ancoraggio Z [%]"])
    cam_rows = [dict(p="Origine intrinseci", v=cam.get("source", "")), dict(p="fx = fy [px]", v=cam["fx"]), dict(p="cx, cy [px]", v=f"{cam['cx']:.1f}, {cam['cy']:.1f}"),
                dict(p="FoV orizzontale [°]", v=cam["fov_x_deg"]),
                dict(p="Prior EXIF", v=f"{S['focal_prior']['fx']:.1f} px ({S['focal_prior']['source']}, ±{100 * S['focal_prior']['sigma_rel']:.0f}%)"),
                dict(p="Focale dalle sole board", v=f"{cam.get('f_boards_only') or 0:.1f} px (±{100 * (cam.get('sigma_f_boards_only_rel') or 0):.2f}%)"),
                dict(p="Distorsione", v=f"{cam['distortion_model']}  k={', '.join(f'{x:.4f}' for x in cam['dist'][:2])}"),
                dict(p="RMS riproiezione [px]", v=cam["rms_px"])]
    gcp = ""
    if S.get("gcp"):
        g = S["gcp"]
        gcp = (f"<h2>Validazione con coordinate rilevate (GCP)</h2><p>Controlli: {g['n_control']} corner, RMSE "
               f"{g['control_rmse_mm']:.1f} mm, fattore di scala Helmert {g['helmert_scale_ppm']:.0f} ppm. "
               f"Check point: {g['n_check']}, RMSE {_fmt(g['check_rmse_mm'], 1)} mm.</p>" +
               _table(g["checks"], ["name", "range_m", "err_mm"], ["Punto", "Distanza [m]", "Errore 3D [mm]"]))
    body = f"""<main><h1>Report ricostruzione metrica monoculare</h1>
<p class="mut">Soluzione: <b>{html.escape(S['chosen'])}</b> · allineamento <b>{html.escape(ch['family'])}</b> ·
{S['image']['width']}×{S['image']['height']} px · {S['runtime_s']} s</p>
<div class="kpi">{kpi}</div>
{''.join(f"<div class='note'>{html.escape(n)}</div>" for n in notes)}
<h2>Target ChArUco</h2><p class="mut">{html.escape(str(S['board_config']))}</p>{det_tab}
<h2>Camera (stima da singola immagine)</h2>{_table(cam_rows, ['p', 'v'], ['Parametro', 'Valore'])}
<h2>Confronto modelli × famiglie di allineamento (Leave-One-Board-Out)</h2>
<p class="mut">Ogni riga misura l'errore su board escluse dal fit (non include l'errore sistematico degli intrinseci). RMSE 3D: corner predetti vs corner da PnP.
Errore distanze: distanze tra corner predetti vs geometria stampata (indipendente dalla PnP).</p>
{_table(mrows, ['model', 'family', 'rmse3d_mm', 'rmse3d_mm_max', 'rel_depth_err_pct', 'abs_bias_pct', 'dist_rmse_mm', 'dist_scale_err_pct', 'flatness_rms_mm'],
        ['Modello', 'Famiglia', 'RMSE3D [mm]', 'max [mm]', 'Err. rel. Z [%]', '|bias| Z [%]', 'RMSE dist. [mm]', 'Err. scala [%]', 'Planarità [mm]'])}
<h2>Dettaglio per board (soluzione scelta, LOBO)</h2>
{_table(per_board, ['board', 'n', 'z_med_m', 'rmse3d_mm', 'rel_depth_err_pct', 'bias_depth_pct', 'dist_rmse_mm', 'dist_scale_err_pct', 'flatness_rms_mm'],
        ['Board', 'Corner', 'Distanza [m]', 'RMSE3D [mm]', 'Err. rel. Z [%]', 'Bias Z [%]', 'RMSE dist. [mm]', 'Err. scala [%]', 'Planarità [mm]'])}
{gcp}
<h2>Immagini</h2><div class="imgs"><img src="{_img_b64(overlay_rgb)}" alt="Rilevamento ChArUco">
<img src="{_img_b64(Zbest, colormap=True)}" alt="Profondità metrica"></div>
</main>"""
    doc = f"<!doctype html><html lang='it'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Report MonoMetric</title><style>{CSS}</style></head><body>{body}</body></html>"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
