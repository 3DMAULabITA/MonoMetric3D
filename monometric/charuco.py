"""ChArUco handling: automatic board-layout inference and corner detection.

The layout (dictionary, squares X/Y, legacy pattern, marker/square ratio, ID
offsets of multiple boards) is inferred from the image alone. The PHYSICAL
square length cannot be inferred from an image and must be supplied (measure
the printed A3 sheet with a caliper over as many squares as possible).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

aruco = cv2.aruco

DICT_FAMILIES = {
    "4X4": ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"],
    "5X5": ["DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000"],
    "6X6": ["DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000"],
    "7X7": ["DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000"],
    "ARUCO_ORIGINAL": ["DICT_ARUCO_ORIGINAL"],
    "APRILTAG_36h11": ["DICT_APRILTAG_36h11"],
    "APRILTAG_25h9": ["DICT_APRILTAG_25h9"],
}
_DICT_SIZE = {"50": 50, "100": 100, "250": 250, "1000": 1000}
_STD_RATIOS = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])


# --------------------------------------------------------------------------- config
@dataclass
class BoardConfig:
    dictionary: str = "DICT_4X4_250"
    squares_x: int = 7
    squares_y: int = 5
    marker_ratio: float = 0.75          # marker_length / square_length
    legacy_pattern: bool = False
    square_mm: float | None = None      # PHYSICAL square length (must be measured)
    markers_per_board: int = 17
    offsets: list = field(default_factory=lambda: [0])   # first marker id of each board

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def make_board(self, offset: int, square_m: float | None = None):
        sq = square_m if square_m is not None else (self.square_mm or 1000.0) / 1000.0
        d = aruco.getPredefinedDictionary(getattr(aruco, self.dictionary))
        ids = np.arange(offset, offset + self.markers_per_board, dtype=np.int32)
        b = aruco.CharucoBoard((self.squares_x, self.squares_y), sq, sq * self.marker_ratio, d, ids)
        if self.legacy_pattern:
            b.setLegacyPattern(True)
        return b


@dataclass
class BoardDetection:
    board_index: int
    offset: int
    corner_ids: np.ndarray        # (N,)
    corners_px: np.ndarray        # (N,2) in the ORIGINAL (distorted) image
    obj_points: np.ndarray        # (N,3) in board frame [m]
    n_markers: int


# --------------------------------------------------------------------------- detection helpers
def _detector_params():
    p = aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 0.005
    p.maxMarkerPerimeterRate = 4.0
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    p.errorCorrectionRate = 0.6
    return p


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _detect_once(gray, d):
    det = aruco.ArucoDetector(d, _detector_params())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return [], np.zeros(0, int)
    return [c.reshape(4, 2).astype(np.float64) for c in corners], ids.ravel().astype(int)


def _default_scales(gray):
    m = max(gray.shape)
    return (1.0, 1.5, 2.0) if m <= 3000 else (1.0, 1.5) if m <= 6500 else (1.0,)


def detect_markers(gray, dict_name, scales=None):
    """ArUco detection at several scales (upsampling helps a lot when markers are < 15 px),
    merged by id and position; corners of the same marker found at several scales are averaged."""
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    scales = scales or _default_scales(gray)
    found = []  # (id, corners)
    H, W = gray.shape
    for s in scales:
        if s == 1.0:
            tiles = [(0, 0, W, H)]
        else:  # upsample in overlapping tiles: bounded memory on 20-50 MP photos
            T, ov = 2048, 256
            xs = list(range(0, max(W - ov, 1), T - ov))
            ys = list(range(0, max(H - ov, 1), T - ov))
            tiles = [(x, y, min(W, x + T), min(H, y + T)) for y in ys for x in xs]
        for x0, y0, x1, y1 in tiles:
            crop = gray[y0:y1, x0:x1]
            g = crop if s == 1.0 else cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            cs, ids = _detect_once(g, d)
            for c, i in zip(cs, ids):
                # pixel-centre convention: x_orig = (x_scaled + 0.5) / s - 0.5 + tile origin
                found.append((int(i), (c + 0.5) / s - 0.5 + [x0, y0]))
    merged = []
    for i, c in found:
        side = np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1))
        for m in merged:
            if m[0] == i and np.linalg.norm(m[1].mean(0) - c.mean(0)) < 0.5 * side:
                m[2].append(c)
                break
        else:
            merged.append([i, c, [c]])
    if not merged:
        return [], np.zeros(0, int)
    return [np.mean(m[2], 0) for m in merged], np.array([m[0] for m in merged], int)


def cluster_markers(marker_corners, k=3.5):
    """Group markers into boards by spatial proximity (connected components)."""
    n = len(marker_corners)
    if n == 0:
        return np.zeros(0, int)
    c = np.array([m.mean(0) for m in marker_corners])
    side = np.array([np.mean(np.linalg.norm(m - np.roll(m, 1, 0), axis=1)) for m in marker_corners])
    lab = -np.ones(n, int)
    cur = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = cur
        while stack:
            j = stack.pop()
            dd = np.linalg.norm(c - c[j], axis=1)
            nb = np.where((lab < 0) & (dd < k * np.maximum(side, side[j])))[0]
            lab[nb] = cur
            stack.extend(nb.tolist())
        cur += 1
    return lab


def identify_dictionary(gray, max_side=2400):
    """Return (dictionary_name, n_markers). Isolated detections (likely false positives
    of large dictionaries) are ignored when choosing the dictionary size."""
    s = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    scales = (1.5,) if max(small.shape) <= 3000 else (1.0,)
    best = (None, 0, None, None)
    for fam, names in DICT_FAMILIES.items():
        mc, ids = detect_markers(small, names[-1], scales=scales)
        if len(ids) == 0:
            continue
        lab = cluster_markers(mc)
        sizes = np.bincount(lab)
        keep = sizes[lab] >= 2
        score = int(keep.sum())
        if score > best[1] or (best[0] is None and len(ids)):
            best = (fam, score, ids[keep] if keep.any() else ids, ids)
    fam, n, ids, _ = best
    if fam is None:
        raise RuntimeError("Nessun marker ArUco rilevato nell'immagine.")
    names = DICT_FAMILIES[fam]
    for nm in names:  # smallest dictionary that contains every (non-isolated) detected id
        size = _DICT_SIZE.get(nm.rsplit("_", 1)[-1], 10**9)
        if ids.max() < size:
            return nm, n
    return names[-1], n


# --------------------------------------------------------------------------- layout inference
def _marker_centers_board_units(cx, cy, legacy, dict_name, ratio=0.5):
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    n = (cx * cy) // 2
    b = aruco.CharucoBoard((cx, cy), 1.0, ratio, d, np.arange(n, dtype=np.int32))
    if legacy:
        b.setLegacyPattern(True)
    obj = b.getObjPoints()
    return np.array([np.asarray(o).reshape(4, 3)[:, :2].mean(0) for o in obj]), np.array(
        [np.asarray(o).reshape(4, 3)[:, :2] for o in obj])


def _fit_cluster(ctr, quads, centers, mcorners, ids, side_px, offset):
    local = ids - offset
    src = ctr[local].astype(np.float32)
    dst = centers.astype(np.float32)
    if np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
        return None
    if len(src) >= 4:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.15 * float(np.median(side_px)))
    else:  # 3 markers: affine (orientation check below still discriminates layouts)
        A, _ = cv2.estimateAffine2D(src, dst)
        H = None if A is None else np.vstack([A, [0, 0, 1]])
    if H is None:
        return None
    proj = cv2.perspectiveTransform(src[None], H)[0]
    res = np.linalg.norm(proj - dst, axis=1) / side_px
    ok = res < 0.15
    q = cv2.perspectiveTransform(quads[local].reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 4, 2)
    v_pred = q[:, 0] - q.mean(1)
    v_det = mcorners[:, 0] - mcorners.mean(1)
    cosang = np.sum(v_pred * v_det, 1) / (np.linalg.norm(v_pred, axis=1) * np.linalg.norm(v_det, axis=1) + 1e-9)
    ok &= cosang > 0.9
    return int(ok.sum()), float(res[ok].sum()), ok


def infer_layout(marker_corners, marker_ids, dict_name, max_squares=14, min_markers=3):
    """Brute-force search of (squares_x, squares_y, legacy) explaining the marker layout.

    Markers are first grouped into boards by spatial proximity; for every candidate layout
    and every board the first marker id (offset) is searched too, so boards whose ids do not
    start at multiples of the markers-per-board (e.g. 17-24, 25-32) are handled. Each board is
    fitted with a homography board-plane -> image on marker centres; the true layout explains
    all markers with small residuals and the right marker orientation.
    """
    centers = np.array([c.mean(0) for c in marker_corners])
    mcs = np.array(marker_corners)
    side_px = np.array([np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)) for c in marker_corners])
    lab = cluster_markers(marker_corners)
    clusters = [np.where(lab == l)[0] for l in np.unique(lab)]
    big = [cl for cl in clusters if len(cl) >= min_markers]
    n_small = sum(len(cl) for cl in clusters if len(cl) < min_markers)
    results = []
    for sx in range(3, max_squares + 1):
        for sy in range(3, max_squares + 1):
            n = (sx * sy) // 2
            for legacy in ((False, True) if sy % 2 == 0 else (False,)):
                ctr, quads = _marker_centers_board_units(sx, sy, legacy, dict_name)
                bad = n_small
                per_cluster = []  # list of {offset: (k, err)}
                for cl in big:
                    ids = marker_ids[cl]
                    lo, hi = ids.min(), ids.max()
                    if hi - lo >= n:
                        bad += len(cl)
                        continue
                    fits = {}
                    for o in range(max(0, hi - n + 1), lo + 1):
                        r = _fit_cluster(ctr, quads, centers[cl], mcs[cl], ids, side_px[cl], o)
                        if r and r[0] > 0:
                            fits[o] = (r[0], r[1], len(cl))
                    if not fits:
                        bad += len(cl)
                        continue
                    per_cluster.append(fits)
                # boards are printed as consecutive id blocks: choose ONE residue (offset mod n)
                # for all boards, the one that explains most markers
                best_r, best_tot = None, (-1, 0.0)
                for r in {o % n for f in per_cluster for o in f}:
                    tot, err = 0, 0.0
                    for f in per_cluster:
                        cand = [v for o, v in f.items() if o % n == r]
                        if cand:
                            k, e, _ = max(cand, key=lambda v: (v[0], -v[1]))
                            tot += k
                            err += e
                    if (tot, -err) > (best_tot[0], -best_tot[1]):
                        best_r, best_tot = r, (tot, err)
                explained, err_sum, offsets = 0, 0.0, []
                for f in per_cluster:
                    cand = [(o, v) for o, v in f.items() if o % n == best_r]
                    if not cand:
                        bad += max(v[2] for v in f.values())
                        continue
                    o, (k, e, m) = max(cand, key=lambda t: (t[1][0], -t[1][1]))
                    explained += k
                    bad += m - k
                    err_sum += e
                    if k >= min_markers:
                        offsets.append(int(o))
                if explained == 0:
                    continue
                results.append(dict(squares_x=sx, squares_y=sy, legacy=legacy, n=n, explained=explained, bad=bad,
                                    mean_res=err_sum / explained, offsets=sorted(offsets), consistent=True))
    if not results:
        raise RuntimeError("Impossibile inferire il layout della board ChArUco (servono almeno "
                           f"{min_markers} marker decodificati su una board).")
    # most markers explained, consistent id blocks, fewest unexplained, smallest board
    results.sort(key=lambda r: (-r["explained"], not r["consistent"], r["bad"], r["n"], r["mean_res"]))
    return results[0], results[:5]


def estimate_marker_ratio(marker_corners, marker_ids, dict_name, layout):
    """Estimate marker/square length ratio through the board->image homography."""
    sx, sy, legacy, n = layout["squares_x"], layout["squares_y"], layout["legacy"], layout["n"]
    ctr, _ = _marker_centers_board_units(sx, sy, legacy, dict_name)
    centers = np.array([c.mean(0) for c in marker_corners])
    ratios = []
    for off in layout["offsets"]:
        sel = np.where((marker_ids >= off) & (marker_ids < off + n))[0]
        if len(sel) < 4:
            continue
        H, _ = cv2.findHomography(ctr[marker_ids[sel] - off].astype(np.float32),
                                  centers[sel].astype(np.float32), cv2.RANSAC, 3.0)
        if H is None:
            continue
        Hi = np.linalg.inv(H)
        for k in sel:
            q = cv2.perspectiveTransform(marker_corners[k].reshape(-1, 1, 2).astype(np.float32), Hi).reshape(4, 2)
            ratios.append(np.mean(np.linalg.norm(q - np.roll(q, 1, 0), axis=1)))
    r = float(np.median(ratios))
    snap = _STD_RATIOS[np.argmin(np.abs(_STD_RATIOS - r))]
    return float(snap) if abs(snap - r) < 0.025 else round(r, 3), r


def auto_board_config(rgb, square_mm=None):
    """Infer the full BoardConfig from an image. Returns (config, diagnostics)."""
    gray = _gray(rgb)
    dict_name, _ = identify_dictionary(gray)
    mc, mi = detect_markers(gray, dict_name)
    best, top = infer_layout(mc, mi, dict_name)
    if not best["offsets"]:
        raise RuntimeError("Nessuna board con abbastanza marker decodificati per inferire il layout.")
    # smallest dictionary of the family containing the ids of the recognised boards
    # (fewer false positives than the large dictionary used for the search)
    fam = next(f for f, names in DICT_FAMILIES.items() if dict_name in names)
    need = max(best["offsets"]) + best["n"]
    for nm in DICT_FAMILIES[fam]:
        if need <= _DICT_SIZE.get(nm.rsplit("_", 1)[-1], 10**9):
            if nm != dict_name:
                dict_name = nm
                mc, mi = detect_markers(gray, dict_name)
            break
    ratio, ratio_raw = estimate_marker_ratio(mc, mi, dict_name, best)
    cfg = BoardConfig(dictionary=dict_name, squares_x=best["squares_x"], squares_y=best["squares_y"],
                      marker_ratio=ratio, legacy_pattern=best["legacy"], square_mm=square_mm,
                      markers_per_board=best["n"], offsets=best["offsets"])
    diag = dict(n_markers=len(mi), marker_ids=mi.tolist(), ratio_raw=ratio_raw, candidates=top,
                suggested_square_mm_A3=suggest_square_mm_a3(cfg))
    return cfg, diag


def suggest_square_mm_a3(cfg, margin_mm=10.0):
    """Largest integer square length that fits an A3 sheet (only a HINT: measure the print!)."""
    long_, short_ = 420.0 - 2 * margin_mm, 297.0 - 2 * margin_mm
    a, b = max(cfg.squares_x, cfg.squares_y), min(cfg.squares_x, cfg.squares_y)
    return float(np.floor(min(long_ / a, short_ / b)))


# --------------------------------------------------------------------------- corner detection
def detect_charuco(rgb, cfg: BoardConfig, min_corners=6, complete=True):
    """Detect ChArUco corners of every board. Corners are in the input image pixel frame.

    Besides OpenCV's CharucoDetector, a homography-guided completion step predicts every
    chessboard corner of a board from its decoded markers and refines it as a saddle point:
    on distant A3 boards most markers are too small to decode (< 2 px per bit) while the
    chessboard X-junctions are still perfectly measurable.
    """
    if not cfg.square_mm:
        raise ValueError("square_mm (lato fisico del quadrato, misurato sulla stampa) è obbligatorio.")
    gray = _gray(rgb)
    mc, mi = detect_markers(gray, cfg.dictionary)
    sq = cfg.square_mm / 1000.0
    out = []
    cp = aruco.CharucoParameters()
    try:
        cp.tryRefineMarkers = True
    except Exception:
        pass
    n = cfg.markers_per_board
    r0 = (cfg.offsets[0] % n) if cfg.offsets else 0  # ids blocks: [k*n + r0, k*n + r0 + n)
    # boards absent from the config: every decoded marker votes for its block
    offsets = sorted(set(cfg.offsets) | {int((m - r0) // n) * n + r0 for m in mi.tolist() if m >= r0})
    for off in offsets:
        bi = off // n  # stable board id = first marker id / markers per board
        board = cfg.make_board(off, sq)
        sel = [k for k, m in enumerate(mi) if off <= m < off + cfg.markers_per_board]
        if len(sel) < 1:
            continue
        det = aruco.CharucoDetector(board, cp, _detector_params())
        mcs = tuple(mc[k].reshape(1, 4, 2).astype(np.float32) for k in sel)
        mis = mi[sel].reshape(-1, 1).astype(np.int32)
        ch_c, ch_i, _, _ = det.detectBoard(gray, markerCorners=mcs, markerIds=mis)
        if ch_i is None:
            ch_c, ch_i = np.zeros((0, 2)), np.zeros(0, int)
        ch_c = ch_c.reshape(-1, 2).astype(np.float64)
        ch_i = ch_i.ravel().astype(int)
        if complete and len(sel) >= 1:
            ch_c, ch_i = _complete_corners(gray, board, [mc[k] for k in sel], mi[sel], ch_c, ch_i)
        if len(ch_i) < min_corners:
            continue
        obj = np.asarray(board.getChessboardCorners()).reshape(-1, 3)[ch_i]
        out.append(BoardDetection(bi, off, ch_i, ch_c, obj.astype(np.float64), len(sel)))
    return out


def _saddle_ok(gray_f, p, sigma):
    """Negative Hessian determinant (X-junction) at p on a Gaussian-smoothed patch."""
    r = int(np.ceil(3 * sigma)) + 2
    x, y = int(round(p[0])), int(round(p[1]))
    H, W = gray_f.shape
    if x - r < 0 or y - r < 0 or x + r + 1 > W or y + r + 1 > H:
        return False, 0.0
    patch = cv2.GaussianBlur(gray_f[y - r:y + r + 1, x - r:x + r + 1], (0, 0), sigma)
    c = r
    h = max(1, int(round(sigma)))
    ixx = patch[c, c + h] + patch[c, c - h] - 2 * patch[c, c]
    iyy = patch[c + h, c] + patch[c - h, c] - 2 * patch[c, c]
    ixy = (patch[c + h, c + h] + patch[c - h, c - h] - patch[c + h, c - h] - patch[c - h, c + h]) / 4
    det = ixx * iyy - ixy * ixy
    contrast = float(patch.max() - patch.min())
    return det < -(0.002 * contrast ** 2), contrast


def _complete_corners(gray, board, mcorners, mids, ch_c, ch_i, iters=6):
    objc = np.asarray(board.getChessboardCorners()).reshape(-1, 3)[:, :2].astype(np.float64)
    ids_all = np.asarray(board.getIds()).ravel()
    objm = board.getObjPoints()
    src = [np.asarray(objm[int(np.where(ids_all == m)[0][0])]).reshape(4, 3)[:, :2] for m in mids]
    src = np.concatenate(src + [objc[ch_i]]) if len(ch_i) else np.concatenate(src)
    dst = np.concatenate([np.asarray(c).reshape(4, 2) for c in mcorners] + ([ch_c] if len(ch_i) else []))
    Hm, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, 2.0)
    if Hm is None:
        return ch_c, ch_i
    gray_f = gray.astype(np.float32)
    sqm = board.getSquareLength()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
    acc_c, acc_i = ch_c, ch_i
    for _ in range(iters):
        pred = cv2.perspectiveTransform(objc[None].astype(np.float32), Hm)[0].astype(np.float64)
        # local square size in px for every corner
        px = cv2.perspectiveTransform((objc + [sqm, 0])[None].astype(np.float32), Hm)[0]
        py = cv2.perspectiveTransform((objc + [0, sqm])[None].astype(np.float32), Hm)[0]
        sq_px = np.minimum(np.linalg.norm(px - pred, axis=1), np.linalg.norm(py - pred, axis=1))
        new_c, new_i = [], []
        Hh, Ww = gray.shape
        for k, (p, s) in enumerate(zip(pred, sq_px)):
            if s < 5 or not (2 <= p[0] < Ww - 3 and 2 <= p[1] < Hh - 3):
                continue
            w = int(np.clip(0.25 * s, 2, 11))
            q = cv2.cornerSubPix(gray, p.astype(np.float32).reshape(1, 1, 2), (w, w), (-1, -1), crit).reshape(2)
            if np.linalg.norm(q - p) > 0.2 * s:
                continue
            ok, _ = _saddle_ok(gray_f, q, max(0.7, 0.12 * s))
            if ok:
                new_c.append(q)
                new_i.append(k)
        if len(new_i) < 4:
            break
        acc_c, acc_i = np.array(new_c, np.float64), np.array(new_i, int)
        Hn, inl = cv2.findHomography(objc[acc_i].astype(np.float32), acc_c.astype(np.float32), cv2.RANSAC, 1.0)
        if Hn is None:
            break
        inl = inl.ravel().astype(bool)
        acc_c, acc_i = acc_c[inl], acc_i[inl]
        Hm = Hn
    if len(acc_i) < len(ch_i):  # never return fewer corners than OpenCV found
        acc_c, acc_i = ch_c, ch_i
    order = np.argsort(acc_i)
    return acc_c[order], acc_i[order]


def _square_px(c, ids, board):
    obj = np.asarray(board.getChessboardCorners()).reshape(-1, 3)[ids, :2]
    if len(c) < 4:
        return 30.0
    H, _ = cv2.findHomography(obj.astype(np.float32), c.astype(np.float32))
    if H is None:
        return 30.0
    sq = board.getSquareLength()
    o = obj.mean(0)
    pts = np.array([[o, o + [sq, 0], o + [0, sq]]], np.float32)
    p = cv2.perspectiveTransform(pts, H)[0]
    return float(min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])))


def draw_detections(rgb, detections, cfg=None):
    vis = rgb.copy()
    rad = max(2, int(max(rgb.shape[:2]) / 800))
    for d in detections:
        col = tuple(int(x) for x in np.array(cv2.applyColorMap(np.uint8([[40 * d.board_index % 255]]), cv2.COLORMAP_TURBO)[0, 0]))
        if len(d.corners_px) >= 2:
            dd = np.linalg.norm(d.corners_px[:, None] - d.corners_px[None], axis=2)
            dd[dd == 0] = np.inf
            sq_px = float(np.median(dd.min(1)))
        else:
            sq_px = 0.0
        for p, cid in zip(d.corners_px, d.corner_ids):
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), rad, col, -1, cv2.LINE_AA)
            if len(d.corner_ids) <= 60 and sq_px >= 18:  # corner ids (to write the GCP file), if legible
                fs = 0.30 * sq_px / 22.0
                th = max(1, int(round(fs * 1.6)))
                q = (int(round(p[0])) + 2, int(round(p[1])) - 2)
                cv2.putText(vis, str(cid), q, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(vis, str(cid), q, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        c = d.corners_px.mean(0)
        org = (int(d.corners_px[:, 0].max()) + 3 * rad, int(c[1]))
        txt = f"B{d.board_index} ({len(d.corner_ids)})"
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, rad * 0.6, (0, 0, 0), max(2, rad), cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, rad * 0.6, (255, 230, 0), max(1, rad // 2), cv2.LINE_AA)
    return vis


def generate_board_image(cfg: BoardConfig, offset=0, dpi=300, paper_mm=(420, 297)):
    """Printable board at given dpi centred on an A3 page (landscape)."""
    px_per_mm = dpi / 25.4
    board = cfg.make_board(offset, (cfg.square_mm or 40) / 1000)
    bw = int(round(cfg.squares_x * cfg.square_mm * px_per_mm))
    bh = int(round(cfg.squares_y * cfg.square_mm * px_per_mm))
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    W, H = int(round(paper_mm[0] * px_per_mm)), int(round(paper_mm[1] * px_per_mm))
    page = np.full((H, W), 255, np.uint8)
    y0, x0 = (H - bh) // 2, (W - bw) // 2
    page[y0:y0 + bh, x0:x0 + bw] = img
    txt = f"{cfg.dictionary} {cfg.squares_x}x{cfg.squares_y} sq={cfg.square_mm}mm ratio={cfg.marker_ratio} ids {offset}-{offset + cfg.markers_per_board - 1}"
    cv2.putText(page, txt, (x0, max(30, y0 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    return page
