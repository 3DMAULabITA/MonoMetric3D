"""Single-image camera estimation from multiple ChArUco boards + EXIF prior.

Joint MAP estimate (Levenberg-Marquardt, robust loss) of
    log f (square pixels), optional principal point, optional radial k1[,k2],
    and the 6-DoF pose of every board,
with a Gaussian prior on log f from EXIF. The radial model is chosen by BIC.
Covariances come from the Gauss-Newton approximation (J^T J)^-1 * s^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class CameraSolution:
    K: np.ndarray
    dist: np.ndarray                 # OpenCV 5-vector (k1,k2,p1,p2,k3)
    rvecs: list
    tvecs: list
    sigma_f_rel: float
    rms_px: float
    per_board_rms: list
    model: str
    f_boards_only: float | None = None
    sigma_f_boards_only_rel: float | None = None
    bic: dict = field(default_factory=dict)
    image_size: tuple = (0, 0)

    def summary(self):
        return dict(fx=float(self.K[0, 0]), fy=float(self.K[1, 1]), cx=float(self.K[0, 2]), cy=float(self.K[1, 2]),
                    dist=[float(x) for x in self.dist.ravel()], sigma_f_rel=self.sigma_f_rel,
                    rms_px=self.rms_px, per_board_rms_px=self.per_board_rms, distortion_model=self.model,
                    f_boards_only=self.f_boards_only, sigma_f_boards_only_rel=self.sigma_f_boards_only_rel,
                    bic=self.bic, fov_x_deg=float(np.degrees(2 * np.arctan(self.image_size[0] / (2 * self.K[0, 0])))))


def _K(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _fit(dets, size, f0, sf_rel, model, free_pp, sigma_px=0.5, fixed=None):
    W, H = size
    nb = len(dets)
    nk = {"none": 0, "k1": 1, "k1k2": 2}[model]
    npp = 2 if free_pp else 0
    # ---- init poses with IPPE
    K0 = _K(f0, (W - 1) / 2, (H - 1) / 2)
    poses = []
    for d in dets:
        ok, r, t = cv2.solvePnP(d.obj_points.astype(np.float64), d.corners_px.astype(np.float64), K0, None,
                                flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            ok, r, t = cv2.solvePnP(d.obj_points, d.corners_px, K0, None, flags=cv2.SOLVEPNP_SQPNP)
        poses.append(np.r_[r.ravel(), t.ravel()])
    x0 = np.r_[np.log(f0), [(W - 1) / 2, (H - 1) / 2][:npp], np.zeros(nk), np.concatenate(poses)]

    def unpack(x):
        i = 1
        f = np.exp(x[0])
        if free_pp:
            cx, cy = x[1:3]
            i = 3
        else:
            cx, cy = (W - 1) / 2, (H - 1) / 2
        k = np.zeros(5)
        k[:nk] = x[i:i + nk]
        i += nk
        P = x[i:].reshape(nb, 6)
        return f, cx, cy, k, P

    def resid(x):
        f, cx, cy, k, P = unpack(x)
        K = _K(f, cx, cy)
        r = []
        for d, p in zip(dets, P):
            proj, _ = cv2.projectPoints(d.obj_points, p[:3], p[3:], K, k)
            r.append(((proj.reshape(-1, 2) - d.corners_px) / sigma_px).ravel())
        r = np.concatenate(r)
        pri = [(x[0] - np.log(f0)) / sf_rel]
        if free_pp:
            pri += [(cx - (W - 1) / 2) / (0.02 * W), (cy - (H - 1) / 2) / (0.02 * W)]
        if nk:
            pri += list(k[:nk] / 0.3)
        return np.r_[r, pri]

    res = least_squares(resid, x0, method="trf", loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=2000)
    f, cx, cy, k, P = unpack(res.x)
    n_obs = sum(2 * len(d.corner_ids) for d in dets)
    r_obs = res.fun[:n_obs] * sigma_px
    rss = float(np.sum(r_obs ** 2))
    n_par = 1 + npp + nk + 6 * nb
    s2 = max(rss / sigma_px ** 2 / max(n_obs - n_par, 1), 1.0)
    try:
        J = res.jac
        cov = np.linalg.pinv(J.T @ J) * s2
        sf = float(np.sqrt(max(cov[0, 0], 0)))
    except Exception:
        sf = float("nan")
    per_board, ofs = [], 0
    for d in dets:
        m = 2 * len(d.corner_ids)
        per_board.append(float(np.sqrt(np.sum(r_obs[ofs:ofs + m] ** 2) / (m / 2))))
        ofs += m
    bic = n_obs * np.log(rss / n_obs + 1e-12) + n_par * np.log(n_obs)
    return dict(f=f, cx=cx, cy=cy, k=k, P=P, sigma_f_rel=sf, rms=float(np.sqrt(rss / (n_obs / 2))),
                per_board=per_board, bic=float(bic))


def estimate_camera(dets, image_size, focal_prior, distortion="auto", free_principal_point=False,
                    fixed_K=None, fixed_dist=None) -> CameraSolution:
    """dets: list[BoardDetection]; image_size: (W, H); focal_prior: FocalPrior."""
    W, H = image_size
    if not dets:
        raise ValueError("Nessuna board rilevata: impossibile stimare la camera.")
    if fixed_K is not None:  # user-supplied calibration: only poses
        K = np.asarray(fixed_K, float)
        dist = np.zeros(5) if fixed_dist is None else np.r_[np.asarray(fixed_dist, float).ravel(), np.zeros(5)][:5]
        rv, tv, pb, allr = [], [], [], []
        for d in dets:
            ok, r, t = cv2.solvePnP(d.obj_points, d.corners_px, K, dist, flags=cv2.SOLVEPNP_IPPE)
            r, t = cv2.solvePnPRefineLM(d.obj_points, d.corners_px, K, dist, r, t)
            pr, _ = cv2.projectPoints(d.obj_points, r, t, K, dist)
            e = np.linalg.norm(pr.reshape(-1, 2) - d.corners_px, axis=1)
            rv.append(r.ravel()); tv.append(t.ravel()); pb.append(float(np.sqrt(np.mean(e ** 2)))); allr.append(e)
        return CameraSolution(K, dist, rv, tv, 0.0, float(np.sqrt(np.mean(np.concatenate(allr) ** 2))), pb,
                              "fixed", image_size=(W, H))

    f0, sf = focal_prior.fx, focal_prior.sigma_rel
    # coverage check for radial distortion
    pts = np.concatenate([d.corners_px for d in dets])
    rnorm = np.linalg.norm(pts - [(W - 1) / 2, (H - 1) / 2], axis=1) / (0.5 * np.hypot(W, H))
    n_pts = len(pts)
    models = ["none"]
    if distortion == "auto":
        if rnorm.max() > 0.45 and n_pts >= 40:
            models.append("k1")
        if rnorm.max() > 0.6 and n_pts >= 120 and len(dets) >= 3:
            models.append("k1k2")
    elif distortion in ("k1", "k1k2"):
        models = [distortion]

    fits = {m: _fit(dets, (W, H), f0, sf, m, free_principal_point) for m in models}
    best = min(fits, key=lambda m: fits[m]["bic"])
    fb = fits[best]
    # boards-only focal (very weak prior) for diagnostics / consistency check with EXIF
    fo = _fit(dets, (W, H), f0, 10.0, best, free_principal_point)
    K = _K(fb["f"], fb["cx"], fb["cy"])
    return CameraSolution(K=K, dist=fb["k"], rvecs=[p[:3] for p in fb["P"]], tvecs=[p[3:] for p in fb["P"]],
                          sigma_f_rel=fb["sigma_f_rel"], rms_px=fb["rms"], per_board_rms=fb["per_board"],
                          model=best, f_boards_only=float(fo["f"]), sigma_f_boards_only_rel=fo["sigma_f_rel"],
                          bic={m: v["bic"] for m, v in fits.items()}, image_size=(W, H))


def undistort(rgb, cam: CameraSolution):
    """Undistort image (same K) and return (image, valid_mask)."""
    if not np.any(cam.dist):
        return rgb, np.ones(rgb.shape[:2], bool)
    und = cv2.undistort(rgb, cam.K, cam.dist, None, cam.K)
    m = cv2.undistort(np.full(rgb.shape[:2], 255, np.uint8), cam.K, cam.dist, None, cam.K) > 250
    return und, m


def undistort_points(px, cam: CameraSolution):
    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if not np.any(cam.dist):
        return px.reshape(-1, 2)
    return cv2.undistortPoints(px, cam.K, cam.dist, P=cam.K).reshape(-1, 2)


def corners_camera_frame(dets, cam: CameraSolution):
    """3D corner coordinates in the camera frame (OpenCV: X right, Y down, Z forward) [m]."""
    out = []
    for d, r, t in zip(dets, cam.rvecs, cam.tvecs):
        R, _ = cv2.Rodrigues(np.asarray(r, float))
        out.append((R @ d.obj_points.T).T + np.asarray(t, float).ravel())
    return out


# =============================================================================== per-board anchor quality
def board_depth_sigma(dets, cam: CameraSolution, sigma_px=None):
    """Relative 1-sigma of each board's centroid depth from PnP (corner noise only,
    intrinsics treated as exact)."""
    out = []
    s_px = sigma_px or max(cam.rms_px, 0.1)
    for d, r, t in zip(dets, cam.rvecs, cam.tvecs):
        R, _ = cv2.Rodrigues(np.asarray(r, float))
        c = d.obj_points.mean(0)
        obj = d.obj_points - c
        tc = R @ c + np.asarray(t, float).ravel()
        _, J = cv2.projectPoints(obj, np.asarray(r, float), tc, cam.K, cam.dist)
        J = J[:, :6]
        try:
            cov = np.linalg.inv(J.T @ J) * s_px ** 2
            out.append(float(np.sqrt(max(cov[5, 5], 0)) / tc[2]))
        except np.linalg.LinAlgError:
            out.append(float("nan"))
    return out


# =============================================================================== multi-photo calibration
def _to_landscape(px, W, H):
    """Rotate pixel coords of a portrait image by 90 deg about the centre into a landscape
    frame (valid because the principal point is fixed at the centre and the distortion is radial)."""
    if W >= H:
        return px, (W, H)
    p = np.asarray(px, float)
    return np.c_[p[:, 1], (W - 1) - p[:, 0]], (H, W)


def calibrate_from_set(items, focal_prior, min_corners=10, free_principal_point=False, model="auto"):
    """See _calibrate_set. model='auto' picks none/k1/k1k2 by BIC."""
    if model != "auto":
        return _calibrate_set(items, focal_prior, min_corners, free_principal_point, model)
    res = {m: _calibrate_set(items, focal_prior, min_corners, free_principal_point, m) for m in ("none", "k1", "k1k2")}
    best = min(res, key=lambda m: res[m]["bic"])
    out = res[best]
    out["bic_all"] = {m: r["bic"] for m, r in res.items()}
    out["f_by_model"] = {m: r["f"] for m, r in res.items()}
    return out


def _calibrate_set(items, focal_prior, min_corners=10, free_principal_point=False, model="k1k2"):
    """Joint calibration from several photos taken with the SAME camera/focal.

    items: list of (detections, (W, H)). Every board in every photo is one planar view.
    Returns a dict (JSON-serialisable) with f, k1, k2 and uncertainties.
    """
    objp, imgp, size = [], [], None
    for dets, (W, H) in items:
        for d in dets:
            if len(d.corner_ids) < min_corners:
                continue
            px, sz = (d.corners_px, (W, H)) if free_principal_point else _to_landscape(d.corners_px, W, H)
            if size is None:
                size = sz
            if sz != size:
                continue
            objp.append(d.obj_points.astype(np.float32))
            imgp.append(px.astype(np.float32).reshape(-1, 1, 2))
    if len(objp) < 3:
        raise RuntimeError("Servono almeno 3 viste di board (con >= %d corner) per la calibrazione." % min_corners)
    W, H = size
    f0 = focal_prior.fx * (np.hypot(W, H) / np.hypot(*items[0][1]))
    K0 = _K(f0, (W - 1) / 2, (H - 1) / 2)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if not free_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if model == "k1":
        flags |= cv2.CALIB_FIX_K2
    elif model == "none":
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    rms, K, dist, rv, tv, sdi, sde, pve = cv2.calibrateCameraExtended(objp, imgp, (W, H), K0, np.zeros(5), flags=flags,
                                                                      criteria=crit)
    sdi = sdi.ravel()
    # drop outlier views (> 3x median error) and recalibrate once
    pve = pve.ravel()
    keep = pve < max(3 * np.median(pve), 1.0)
    if keep.sum() >= 3 and (~keep).any():
        objp = [o for o, k in zip(objp, keep) if k]
        imgp = [i for i, k in zip(imgp, keep) if k]
        rms, K, dist, rv, tv, sdi, sde, pve = cv2.calibrateCameraExtended(objp, imgp, (W, H), K, dist, flags=flags,
                                                                          criteria=crit)
        sdi = sdi.ravel()
    n_obs = 2 * sum(len(i) for i in imgp)
    n_par = 1 + (2 if free_principal_point else 0) + {"none": 0, "k1": 1, "k1k2": 2}[model] + 6 * len(objp)
    bic = float(n_obs * np.log(rms ** 2 / 2 + 1e-12) + n_par * np.log(n_obs))
    return dict(f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=int(W), height=int(H),
                k1=float(dist.ravel()[0]), k2=float(dist.ravel()[1]), sigma_f_rel=float(max(sdi[0], sdi[1]) / K[0, 0]),
                bic=bic,
                sigma_k1=float(sdi[4]), sigma_k2=float(sdi[5]), rms_px=float(rms), n_views=len(objp),
                principal_point="free" if free_principal_point else "centre", model=model,
                prior_f=float(f0))


def camera_for_image(calib_dict, W, H):
    """K, dist for an image of size (W,H) from a calibration dict (handles portrait/landscape)."""
    c = calib_dict
    s = np.hypot(W, H) / np.hypot(c["width"], c["height"])  # tolerate a uniformly resized photo
    f = c["f"] * s
    if c.get("principal_point") == "free" and (W, H) == (c["width"], c["height"]):
        cx, cy = c["cx"], c["cy"]
    else:
        cx, cy = (W - 1) / 2, (H - 1) / 2
    return _K(f, cx, cy), np.array([c["k1"], c["k2"], 0, 0, 0], float)
