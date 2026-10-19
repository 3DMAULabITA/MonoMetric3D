"""Point cloud generation, filtering, reference frames and export (PLY/LAS/E57/GLB)."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def depth_to_points(Z, K):
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u - K[0, 2]) / K[0, 0] * Z
    Y = (v - K[1, 2]) / K[1, 1] * Z
    return np.dstack([X, Y, Z]).astype(np.float32)


def normals_from_points(P):
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    flip = np.sum(n * P, axis=2) > 0          # orient towards the camera
    n[flip] *= -1
    return n.astype(np.float32)


def edge_mask(Z, rel_thr=0.02, dilate=1):
    """True where the pixel is NOT on a depth discontinuity (flying pixels removal)."""
    Zs = np.where(Z > 0, Z, np.nan)
    m = np.ones(Z.shape, bool)
    for sh in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        Zn = np.roll(Zs, sh, axis=(0, 1))
        r = np.abs(Zn - Zs) / np.fmin(Zn, Zs)
        bad = r > rel_thr
        m &= ~np.nan_to_num(bad, nan=False).astype(bool)
    if dilate:
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0
    return m


def grazing_mask(P, N, min_cos=0.08):
    v = -P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-12)
    return np.sum(v * N, axis=2) > min_cos


def build_cloud(Z, K, rgb, valid, stride=1, edge_thr=0.02, max_depth=None, sigma_rel=None):
    """Point cloud on the stride-subsampled pixel grid (memory ~ 1/stride^2 of full resolution).
    Normals and discontinuity filtering are computed on the same grid."""
    s = max(1, int(stride))
    Zs = np.ascontiguousarray(Z[::s, ::s])
    vs, us = np.arange(0, Z.shape[0], s, dtype=np.float32), np.arange(0, Z.shape[1], s, dtype=np.float32)
    u, v = np.meshgrid(us, vs)
    P = np.dstack([(u - K[0, 2]) / K[0, 0] * Zs, (v - K[1, 2]) / K[1, 1] * Zs, Zs]).astype(np.float32)
    del u, v
    N = normals_from_points(P)
    m = valid[::s, ::s] & (Zs > 0)
    if edge_thr:
        m &= edge_mask(Zs, edge_thr * s)
        m &= grazing_mask(P, N)
    if max_depth:
        m &= Zs < max_depth
    cloud = dict(xyz=P[m].astype(np.float64), rgb=rgb[::s, ::s][m].astype(np.uint8), normal=N[m],
                 depth=Zs[m].astype(np.float32))
    vi, ui = np.nonzero(m)
    cloud["uv"] = np.c_[ui * s, vi * s].astype(np.int32)
    if sigma_rel is not None:
        cloud["sigma_m"] = (Zs[m] * sigma_rel).astype(np.float32)
    return cloud


# ----------------------------------------------------------------------------- frames
def rigid_transform(R, t, cloud):
    c = dict(cloud)
    c["xyz"] = cloud["xyz"] @ R.T + t
    c["normal"] = (cloud["normal"] @ R.T).astype(np.float32)
    return c


def board_frame(rvec, tvec):
    """Transform camera -> board frame (board X along squares_x, Y along squares_y, Z = board normal)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    t = np.asarray(tvec, float).ravel()
    return R.T, -R.T @ t


def fit_rigid(A, B, with_scale=False):
    """Least-squares B ~ s R A + t (Umeyama). Returns s, R, t."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()) if with_scale else 1.0
    t = mb - s * R @ ma
    return s, R, t


def read_gcp_csv(path):
    """CSV header: type,name,board,corner,u,v,X,Y,Z   (coordinates in metres, survey Z axis up)
    type=board -> ChArUco corner (board id as shown in the overlay "B<id>", corner id) with surveyed X,Y,Z (control)
    type=pixel -> image pixel (u,v in the photo as displayed upright) with surveyed X,Y,Z (check point)
    Accepts ',' or ';' as separator and decimal comma (Excel with Italian settings)."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    first = raw.splitlines()[0] if raw.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else "\t" if first.count("\t") > first.count(",") else ","
    rows = []
    for r in csv.DictReader(raw.splitlines(), delimiter=delim):
        row = {}
        for k, v in r.items():
            if k is None:
                continue
            k = k.strip().lower()
            v = (v or "").strip()
            if k in ("u", "v", "x", "y", "z") and delim != "," and v:
                v = v.replace(".", "").replace(",", ".") if v.count(",") == 1 and v.count(".") >= 1 else v.replace(",", ".")
            row[k] = v
        row["type"] = row.get("type", "").lower()
        if row["type"] in ("board", "pixel"):
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- export
def write_ply(path, cloud):
    n = len(cloud["xyz"])
    props = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("depth", "f4")]
    if "sigma_m" in cloud:
        props.append(("sigma_m", "f4"))
    arr = np.empty(n, dtype=[(k, "<" + t) for k, t in props])
    arr["x"], arr["y"], arr["z"] = cloud["xyz"].T
    arr["nx"], arr["ny"], arr["nz"] = cloud["normal"].T
    arr["red"], arr["green"], arr["blue"] = cloud["rgb"].T
    arr["depth"] = cloud["depth"]
    if "sigma_m" in cloud:
        arr["sigma_m"] = cloud["sigma_m"]
    tmap = {"f8": "double", "f4": "float", "u1": "uchar"}
    head = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(
        f"property {tmap[t]} {k}\n" for k, t in props) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(arr.tobytes())
    return path


def write_las(path, cloud):
    import laspy

    hdr = laspy.LasHeader(point_format=2, version="1.2")
    hdr.scales = np.array([0.0001, 0.0001, 0.0001])
    hdr.offsets = cloud["xyz"].min(0)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = cloud["xyz"].T
    c = cloud["rgb"].astype(np.uint16) * 257
    las.red, las.green, las.blue = c.T
    las.write(str(path))
    return path


def write_e57(path, cloud):
    import pye57

    e = pye57.E57(str(path), mode="w")
    e.write_scan_raw({"cartesianX": cloud["xyz"][:, 0], "cartesianY": cloud["xyz"][:, 1],
                      "cartesianZ": cloud["xyz"][:, 2], "colorRed": cloud["rgb"][:, 0],
                      "colorGreen": cloud["rgb"][:, 1], "colorBlue": cloud["rgb"][:, 2]})
    e.close()
    return path


def write_glb_preview(path, cloud, max_points=400_000, yup=True):
    """Decimated GLB for the web viewer (OpenCV camera frame converted to Y-up)."""
    import trimesh

    n = len(cloud["xyz"])
    idx = np.random.default_rng(0).choice(n, min(n, max_points), replace=False)
    xyz = cloud["xyz"][idx].copy()
    xyz -= np.median(xyz, 0)
    if yup:
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1
    col = np.c_[cloud["rgb"][idx], np.full(len(idx), 255, np.uint8)]
    pc = trimesh.PointCloud(xyz, colors=col)
    trimesh.Scene(pc).export(str(path))
    return path


def export_all(cloud, out_dir, stem, formats=("ply",)):
    out_dir = Path(out_dir)
    paths = {}
    for f in formats:
        try:
            p = out_dir / f"{stem}.{f}"
            {"ply": write_ply, "las": write_las, "e57": write_e57}[f](p, cloud)
            paths[f] = str(p)
        except ImportError as e:
            paths[f] = f"skipped ({e})"
    return paths
