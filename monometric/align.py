"""Metric alignment of a monocular depth map to ChArUco anchors + honest validation.

Anchors: ChArUco corners in the camera frame from the joint PnP (calib.py).
Alignment families (from rigid to flexible):
    scale        z = s d
    affine       z = a d + b
    disp_affine  1/z = a/d + b
    field1       z = d exp(c0 + c1 x + c2 y)                 (x,y normalised image coords)
    field2       z = d exp(quadratic polynomial in x,y)
    none         z = d   (zero-shot: the network's own metric scale, no target information;
                         only when requested explicitly, e.g. as baseline)
Each board weighs 1 regardless of its number of corners. Robust IRLS (Huber).
The family is selected by Leave-One-Board-Out (LOBO) cross-validation, so the
reported accuracy is always measured on boards NOT used for the fit.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

FAMILIES = ["scale", "affine", "disp_affine", "field1", "field2"]
_COMPLEXITY = {"none": -1, "scale": 0, "affine": 1, "disp_affine": 1, "field1": 2, "field2": 3}
_MIN_BOARDS = {"none": 0, "scale": 1, "affine": 2, "disp_affine": 2, "field1": 3, "field2": 6}


# ----------------------------------------------------------------------------- sampling
def sample_depth(depth, mask, uv, win=2):
    """Robust depth at sub-pixel positions: median of a (2win+1)^2 bilinear neighbourhood."""
    H, W = depth.shape
    d = np.where(mask, depth, np.nan).astype(np.float32)
    offs = np.arange(-win, win + 1, dtype=np.float32)
    vals = []
    for dx, dy in itertools.product(offs, offs):
        mx = (uv[:, 0] + dx).astype(np.float32).reshape(-1, 1)
        my = (uv[:, 1] + dy).astype(np.float32).reshape(-1, 1)
        vals.append(cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan).ravel())
    return np.nanmedian(np.stack(vals, 1), axis=1)


def norm_coords(uv, W, H):
    return np.c_[(2 * uv[:, 0] / (W - 1)) - 1, (2 * uv[:, 1] / (H - 1)) - 1]


def _phi(xy, family):
    x, y = xy[:, 0], xy[:, 1]
    if family == "field1":
        return np.c_[np.ones_like(x), x, y]
    return np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]


# ----------------------------------------------------------------------------- fitting
def _wls(A, b, w, ridge=0.0):
    sw = np.sqrt(w)[:, None]
    M = A * sw
    reg = ridge * np.eye(A.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(M.T @ M + reg, M.T @ (b * sw.ravel()))


def fit(family, d, z, xy, w, iters=10):
    if family == "none":
        return np.array([1.0])
    w0 = w / w.sum()
    ww = w0.copy()
    p = None
    for _ in range(iters):
        if family == "scale":
            q = d / z
            p = np.array([np.sum(ww * q) / np.sum(ww * q * q)])
        elif family == "affine":
            p = _wls(np.c_[d / z, 1 / z], np.ones_like(z), ww)
        elif family == "disp_affine":
            p = _wls(np.c_[z / d, z], np.ones_like(z), ww)
        else:
            p = _wls(_phi(xy, family), np.log(z) - np.log(d), ww, ridge=1e-4)
        r = predict(family, p, d, xy) / z - 1
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        c = 2.5 * s
        ww = w0 * np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
    return p


def predict(family, p, d, xy):
    if family == "none":
        return d
    if family == "scale":
        return p[0] * d
    if family == "affine":
        return p[0] * d + p[1]
    if family == "disp_affine":
        return 1.0 / (p[0] / d + p[1])
    return d * np.exp(_phi(xy, family) @ p)


def apply_to_map(family, p, depth, mask, rows=512):
    """Apply the alignment to the whole depth map, in row blocks (bounded RAM on laptops)."""
    H, W = depth.shape
    p = np.asarray(p, np.float64)
    out = np.zeros((H, W), np.float32)
    x = ((2 * np.arange(W, dtype=np.float32) / (W - 1)) - 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            d = depth[r0:r1].astype(np.float64)
            m = mask[r0:r1]
            if family in ("none", "scale", "affine", "disp_affine"):
                z = predict(family, p, np.where(m, d, 1.0), None)
            else:
                y = ((2 * np.arange(r0, r1, dtype=np.float32) / (H - 1)) - 1)
                xy = np.c_[np.tile(x, r1 - r0), np.repeat(y, W)]
                z = (np.where(m, d, 1.0).ravel() * np.exp(_phi(xy, family) @ p)).reshape(r1 - r0, W)
            out[r0:r1] = np.where(m & np.isfinite(z) & (z > 0), z, 0)
    return out


# ----------------------------------------------------------------------------- data assembly
class Anchors:
    """Per-board arrays: uv (undistorted px), Xc (camera frame m), obj (board frame m), d (model depth)."""

    def __init__(self, uv_list, Xc_list, obj_list, d_list, W, H, K, sigma_list=None, tau=0.01, ids=None):
        """sigma_list: relative 1-sigma of each board's PnP depth; board weight ~ 1/(sigma^2 + tau^2)
        where tau is the expected model error after alignment."""
        self.W, self.H, self.K = W, H, K
        self.boards = []
        sigma_list = sigma_list or [0.0] * len(uv_list)
        for b, (uv, Xc, obj, d, sg) in enumerate(zip(uv_list, Xc_list, obj_list, d_list, sigma_list)):
            ok = np.isfinite(d) & (d > 0)
            if ok.sum() < 4:
                continue
            sg = float(sg) if np.isfinite(sg) else 0.05
            self.boards.append(dict(b=(ids[b] if ids is not None else b), uv=uv[ok], Xc=Xc[ok], obj=obj[ok], d=d[ok], z=Xc[ok, 2],
                                    xy=norm_coords(uv[ok], W, H), sigma=sg, wb=1.0 / (sg ** 2 + tau ** 2)))

    def stack(self, idx):
        B = [self.boards[i] for i in idx]
        d = np.concatenate([b["d"] for b in B])
        z = np.concatenate([b["z"] for b in B])
        xy = np.concatenate([b["xy"] for b in B])
        w = np.concatenate([np.full(len(b["d"]), b["wb"] / len(b["d"])) for b in B])
        return d, z, xy, w


def _family_ok(family, anchors, idx):
    if len(idx) < _MIN_BOARDS[family]:
        return False
    B = [anchors.boards[i] for i in idx]
    if family in ("affine", "disp_affine"):
        zall = np.concatenate([b["z"] for b in B])
        return (np.percentile(zall, 95) - np.percentile(zall, 5)) / np.median(zall) > 0.15
    if family in ("field1", "field2"):
        c = np.array([b["xy"].mean(0) for b in B])
        if len(c) < 3:
            return False
        A = np.c_[c, np.ones(len(c))]
        return np.linalg.matrix_rank(A, tol=0.05) == 3
    return True


def backproject(uv, z, K):
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    return np.c_[x * z, y * z, z]


def board_metrics(b, z_hat, K):
    """Errors on one board for predicted depths z_hat at its corners."""
    X = backproject(b["uv"], z_hat, K)
    e3 = np.linalg.norm(X - b["Xc"], axis=1)
    # intra-board distances vs. the printed geometry (independent of PnP)
    n = len(X)
    i, j = np.triu_indices(n, 1)
    if len(i) > 4000:
        sel = np.random.default_rng(0).choice(len(i), 4000, replace=False)
        i, j = i[sel], j[sel]
    dh = np.linalg.norm(X[i] - X[j], axis=1)
    dt = np.linalg.norm(b["obj"][i] - b["obj"][j], axis=1)
    long = dt > 0.5 * dt.max()
    # flatness of predicted corners
    Xm = X - X.mean(0)
    flat = np.linalg.svd(Xm, full_matrices=False)[1][-1] / np.sqrt(n)
    # radial (along the viewing ray) / lateral decomposition of the 3D error
    ray = b["Xc"] / np.linalg.norm(b["Xc"], axis=1, keepdims=True)
    e = X - b["Xc"]
    er = np.sum(e * ray, 1)
    el = np.linalg.norm(e - er[:, None] * ray, axis=1)
    # incidence angle between the board normal and the line of sight
    C = b["Xc"] - b["Xc"].mean(0)
    nrm = np.linalg.svd(C, full_matrices=False)[2][-1]
    vc = b["Xc"].mean(0) / np.linalg.norm(b["Xc"].mean(0))
    inc = float(np.degrees(np.arccos(min(1.0, abs(float(nrm @ vc))))))
    return dict(board=int(b["b"]), n=int(n), z_med_m=float(np.median(b["z"])),
                range_m=float(np.median(np.linalg.norm(b["Xc"], axis=1))), incidence_deg=inc,
                anchor_sigma_pct=float(100 * b.get("sigma", 0.0)),
                mae3d_mm=float(1000 * np.mean(e3)), d90_mm=float(1000 * np.percentile(e3, 90)),
                radial_rmse_mm=float(1000 * np.sqrt(np.mean(er ** 2))),
                radial_bias_mm=float(1000 * np.mean(er)),
                lateral_rmse_mm=float(1000 * np.sqrt(np.mean(el ** 2))),
                rel_depth_err_pct=float(100 * np.median(np.abs(z_hat / b["z"] - 1))),
                bias_depth_pct=float(100 * np.median(z_hat / b["z"] - 1)),
                rmse3d_mm=float(1000 * np.sqrt(np.mean(e3 ** 2))),
                dist_rmse_mm=float(1000 * np.sqrt(np.mean((dh - dt) ** 2))),
                dist_scale_err_pct=float(100 * (np.median(dh[long] / dt[long]) - 1)),
                flatness_rms_mm=float(1000 * flat))


def lobo(anchors, family):
    """Leave-one-board-out predictions and metrics. Returns (list of metrics, dict b->z_hat)."""
    nb = len(anchors.boards)
    res, preds = [], {}
    for k in range(nb):
        tr = [i for i in range(nb) if i != k]
        if not _family_ok(family, anchors, tr):
            return None, None
        p = fit(family, *anchors.stack(tr))
        b = anchors.boards[k]
        zh = predict(family, p, b["d"], b["xy"])
        preds[k] = zh
        res.append(board_metrics(b, zh, anchors.K))
    return res, preds


def summarize(metrics):
    if not metrics:
        return None
    a = lambda k: np.array([m[k] for m in metrics])
    return dict(n_boards=len(metrics),
                rmse3d_mm=float(np.sqrt(np.mean(a("rmse3d_mm") ** 2))),
                rmse3d_mm_max=float(a("rmse3d_mm").max()),
                rel_depth_err_pct=float(np.mean(a("rel_depth_err_pct"))),
                abs_bias_pct=float(np.mean(np.abs(a("bias_depth_pct")))),
                dist_rmse_mm=float(np.sqrt(np.mean(a("dist_rmse_mm") ** 2))),
                dist_scale_err_pct=float(np.mean(np.abs(a("dist_scale_err_pct")))),
                flatness_rms_mm=float(np.mean(a("flatness_rms_mm"))),
                mae3d_mm=float(np.mean(a("mae3d_mm"))),
                d90_mm=float(np.median(a("d90_mm"))),
                radial_rmse_mm=float(np.sqrt(np.mean(a("radial_rmse_mm") ** 2))),
                lateral_rmse_mm=float(np.sqrt(np.mean(a("lateral_rmse_mm") ** 2))))


def select_and_fit(anchors, families="auto", tol=1.05):
    """Evaluate every admissible family with LOBO, choose the simplest one within `tol`
    of the best CV RMSE, refit on all boards. Returns a result dict."""
    fams = FAMILIES if families == "auto" else [families] if isinstance(families, str) else list(families)
    nb = len(anchors.boards)
    if nb == 0:
        raise RuntimeError("Nessun ancoraggio valido (depth del modello non disponibile sui target).")
    allidx = list(range(nb))
    cv = {}
    for f in fams:
        if f == "none":  # nothing is fitted: every board is an independent test
            m = [board_metrics(b, b["d"], anchors.K) for b in anchors.boards]
            cv[f] = dict(per_board=m, summary=summarize(m), preds={k: b["d"] for k, b in enumerate(anchors.boards)})
            continue
        if not _family_ok(f, anchors, allidx):
            continue
        if nb >= 2:
            m, preds = lobo(anchors, f)
            if m is None:
                continue
            cv[f] = dict(per_board=m, summary=summarize(m), preds=preds)
        else:
            cv[f] = dict(per_board=None, summary=None, preds=None)
    if not cv:
        raise RuntimeError("Nessuna famiglia di allineamento applicabile.")
    with_cv = {f: v for f, v in cv.items() if v["summary"] is not None}
    if with_cv:
        best = min(v["summary"]["rmse3d_mm"] for v in with_cv.values())
        chosen = min((f for f, v in with_cv.items() if v["summary"]["rmse3d_mm"] <= tol * best),
                     key=lambda f: (_COMPLEXITY[f], with_cv[f]["summary"]["rmse3d_mm"]))
    else:
        chosen = fams[0] if len(fams) == 1 else "scale"
    p = fit(chosen, *anchors.stack(allidx))
    insample = [board_metrics(b, predict(chosen, p, b["d"], b["xy"]), anchors.K) for b in anchors.boards]
    return dict(family=chosen, params=p.tolist(), cv={f: {"summary": v["summary"], "per_board": v["per_board"]}
                                                      for f, v in cv.items()},
                cv_preds=cv[chosen]["preds"], insample=insample, insample_summary=summarize(insample),
                validated=bool(with_cv))
