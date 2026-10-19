"""End-to-end pipeline: photo + ChArUco -> metric, validated point cloud."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import align, backends, calib, charuco, device, pointcloud, report
from .imageio_exif import focal_prior_from_exif, load_image


def _log(cb, msg):
    print(msg, flush=True)
    if cb:
        cb(msg)


def run(image, board_cfg: charuco.BoardConfig | None = None, square_mm=None, models=None,
        families="auto", fuse=True, stride=None, edge_thr=0.02, max_depth=None, distortion="auto",
        free_principal_point=False, K_override=None, dist_override=None, camera_calib=None, frame="camera", gcp_csv=None,
        formats=("ply",), out_dir="out", preview=True, progress=None, profile=None, infer_max_mp=None,
        focal_sweep=None, site=None, tag=None):
    """focal_sweep: list of relative focal errors (e.g. [-0.05, -0.02, 0.02, 0.05]) for the sensitivity
    analysis (re-inference + re-alignment with f(1+eps)); site/tag: labels used by `summarize`/`figures`."""
    t0 = time.time()
    prof = device.set_profile(profile) if profile else device.active()
    models = tuple(models) if models else prof.default_models
    stride = int(stride or prof.default_stride)
    infer_max_mp = infer_max_mp or prof.infer_max_mp
    _log(progress, "Hardware: " + prof.describe())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. image
    image_path = None
    if isinstance(image, (str, Path)):
        rgb, exif = load_image(image)
        stem = Path(image).stem
        image_path = str(Path(image).resolve())
        site = site or Path(image).resolve().parent.name
    else:
        rgb, exif = image
        stem = "image"
    H, W = rgb.shape[:2]
    _log(progress, f"Immagine {W}x{H}  {exif.get('Make', '')} {exif.get('Model', '')}")

    # ---------------------------------------------------------------- 2. boards
    diag = None
    if board_cfg is None:
        board_cfg, diag = charuco.auto_board_config(rgb, square_mm)
        _log(progress, f"Board auto: {board_cfg.dictionary} {board_cfg.squares_x}x{board_cfg.squares_y} "
                       f"ratio={board_cfg.marker_ratio} legacy={board_cfg.legacy_pattern} offsets={board_cfg.offsets}")
    if square_mm:
        board_cfg.square_mm = float(square_mm)
    dets = charuco.detect_charuco(rgb, board_cfg)
    if not dets:
        raise RuntimeError("Nessuna board ChArUco rilevata con la configurazione fornita.")
    _log(progress, f"Board rilevate: {len(dets)}  corner: {[len(d.corner_ids) for d in dets]}")

    # ---------------------------------------------------------------- 3. camera
    prior = focal_prior_from_exif(exif, W, H)
    calib_src = "single_image"
    if camera_calib is not None:  # multi-photo calibration (dict or JSON path)
        cc = json.loads(Path(camera_calib).read_text()) if isinstance(camera_calib, (str, Path)) else camera_calib
        K_override, dist_override = calib.camera_for_image(cc, W, H)
        calib_src = f"calibrazione multi-foto ({cc.get('n_views')} viste)"
    elif K_override is not None:
        calib_src = "fornita dall'utente"
    cam = calib.estimate_camera(dets, (W, H), prior, distortion, free_principal_point, K_override, dist_override)
    # reject boards whose corners are inconsistent with the camera model (e.g. a grazing board
    # completed on the wrong lattice): reprojection RMS far above the others
    rejected = []
    while len(dets) > 2:
        rms = np.array(cam.per_board_rms)
        lim = max(1.5, 4 * float(np.median(rms)))
        worst = int(np.argmax(rms))
        if rms[worst] <= lim:
            break
        rejected.append(dict(board=dets[worst].board_index, reproj_rms_px=float(rms[worst])))
        _log(progress, f"Board {dets[worst].board_index} scartata: RMS di riproiezione {rms[worst]:.2f} px (limite {lim:.2f})")
        dets = [d for i, d in enumerate(dets) if i != worst]
        cam = calib.estimate_camera(dets, (W, H), prior, distortion, free_principal_point, K_override, dist_override)
    if camera_calib is not None:
        cam.sigma_f_rel = float(cc.get("sigma_f_rel", 0.0))
        cam.model = f"calib:{cc.get('model')}"
    sig_board = calib.board_depth_sigma(dets, cam)
    cs = cam.summary()
    _log(progress, f"Camera: f={cs['fx']:.1f}px (±{100 * cam.sigma_f_rel:.2f}%), prior {prior.source} "
                   f"f0={prior.fx:.1f}, dist={cam.model}, RMS reproj={cam.rms_px:.3f}px")
    rgb_u, valid_u = calib.undistort(rgb, cam)
    Xc = calib.corners_camera_frame(dets, cam)
    uv_u = [calib.undistort_points(d.corners_px, cam) for d in dets]
    obj = [d.obj_points for d in dets]

    # ---------------------------------------------------------------- 4. models + alignment
    results = {}
    for key in models:
        _log(progress, f"Inferenza {key} ...")
        tb = time.time()
        be = backends.get_backend(key)
        o = backends.infer_scaled(be, rgb_u, cam.K, infer_max_mp, log=lambda m: _log(progress, m))
        dep, msk = o["depth"], o["mask"] & valid_u & (o["depth"] > 0)
        infer_size = o["infer_size"]
        del o
        dvals = [align.sample_depth(dep, msk, uv) for uv in uv_u]
        anc = align.Anchors(uv_u, Xc, obj, dvals, W, H, cam.K, sig_board, ids=[d.board_index for d in dets])
        al = align.select_and_fit(anc, families)
        Z = align.apply_to_map(al["family"], np.array(al["params"]), dep, msk)
        s = al["cv"][al["family"]]["summary"]
        _log(progress, f"  {key}: famiglia={al['family']}  LOBO RMSE3D="
                       f"{s['rmse3d_mm']:.1f} mm  ({time.time() - tb:.1f}s)" if s else
             f"  {key}: famiglia={al['family']} (nessuna validazione indipendente: 1 board)")
        del dep
        results[key] = dict(align=al, Z=Z, mask=msk, anchors=anc, backend=be.name,
                            license=getattr(be, "license", ""), infer_size=infer_size)

    # ---------------------------------------------------------------- 5. fusion
    if fuse and len(results) >= 2:
        keys = list(results)
        acc = np.zeros(results[keys[0]]["Z"].shape, np.float32)
        mf = np.ones(acc.shape, bool)
        for k in keys:  # running geometric mean (no stacked copies: bounded RAM)
            Zk = results[k]["Z"]
            mf &= Zk > 0
            acc += np.log(np.where(Zk > 0, Zk, 1.0)).astype(np.float32)
        Zf = np.where(mf, np.exp(acc / len(keys)), 0).astype(np.float32)
        del acc
        anc0 = results[keys[0]]["anchors"]
        fused = None
        same = all([(b["b"], len(b["d"])) for b in results[k]["anchors"].boards] ==
                   [(b["b"], len(b["d"])) for b in anc0.boards] for k in keys)
        if same and all(results[k]["align"]["cv_preds"] for k in keys):
            per = []
            for bi, b in enumerate(anc0.boards):
                zh = np.exp(np.mean([np.log(results[k]["align"]["cv_preds"][bi]) for k in keys], 0))
                per.append(align.board_metrics(b, zh, cam.K))
            fused = dict(per_board=per, summary=align.summarize(per))
        results["fusion"] = dict(align=dict(family="geo-mean", params=[], cv={"geo-mean": fused} if fused else {},
                                            validated=fused is not None, insample_summary=None),
                                 Z=Zf, mask=mf, anchors=anc0, backend="+".join(keys), license="")

    def cv_rmse(k):
        a = results[k]["align"]
        s = a["cv"].get(a["family"], {}) or {}
        s = s.get("summary") if isinstance(s, dict) else None
        return s["rmse3d_mm"] if s else np.inf

    best = min(results, key=cv_rmse)
    R = results[best]
    best_cv = (R["align"]["cv"].get(R["align"]["family"]) or {}).get("summary")
    _log(progress, f"Soluzione scelta: {best}")

    # ---------------------------------------------------------------- 6. cloud
    # 1-sigma relative depth uncertainty per point: LOBO model error (median|e| -> sigma) + focal
    if best_cv:
        sig = float(np.hypot(1.4826 * best_cv["rel_depth_err_pct"] / 100, cam.sigma_f_rel))
    else:
        sig = float(np.hypot(0.03, cam.sigma_f_rel))  # no validation possible: conservative 3 %
    cloud = pointcloud.build_cloud(R["Z"], cam.K, rgb_u, R["mask"], stride, edge_thr, max_depth, sig)

    # ---------------------------------------------------------------- 7. reference frame / GCP
    gcp_report = None
    if frame.startswith("board:"):
        bid = int(frame.split(":")[1])
        idx = [i for i, d in enumerate(dets) if d.board_index == bid]
        if not idx:
            raise RuntimeError(f"Board {bid} non rilevata: impossibile usarla come sistema di riferimento.")
        Rb, tb_ = pointcloud.board_frame(cam.rvecs[idx[0]], cam.tvecs[idx[0]])
        cloud = pointcloud.rigid_transform(Rb, tb_, cloud)
    elif gcp_csv:
        gcp_rows = pointcloud.read_gcp_csv(gcp_csv)
        reg = _gcp_registration(gcp_rows, dets, Xc)
        gcp_report = dict(reg["report"], **_check_points(gcp_rows, cam, cam.K, R["Z"], reg, sig))
        cloud = pointcloud.rigid_transform(reg["R"], reg["t"], cloud)
        _log(progress, f"GCP: controlli RMSE {gcp_report['control_rmse_mm']:.1f} mm, check point "
                       f"RMSE {gcp_report.get('check_rmse_mm') or float('nan'):.1f} mm ({gcp_report['n_check']})")

    # ---------------------------------------------------------------- 7b. focal sensitivity
    sweep = None
    if focal_sweep:
        sweep = _focal_sweep(focal_sweep, best, results, R, cam, dets, prior, rgb_u, valid_u, uv_u, obj, sig_board,
                             W, H, infer_max_mp, gcp_csv, progress)

    # ---------------------------------------------------------------- 8. export
    files = pointcloud.export_all(cloud, out_dir, stem, formats)
    if preview:
        try:
            files["glb"] = str(pointcloud.write_glb_preview(out_dir / f"{stem}_preview.glb", cloud))
        except Exception as e:
            files["glb"] = f"skipped ({e})"
    np.save(out_dir / f"{stem}_depth_m.npy", R["Z"])
    files["depth_npy"] = str(out_dir / f"{stem}_depth_m.npy")
    vis = charuco.draw_detections(rgb, dets)
    cv2.imwrite(str(out_dir / f"{stem}_charuco.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    files["overlay"] = str(out_dir / f"{stem}_charuco.jpg")

    summary = dict(
        image_path=image_path, site=site, tag=tag, focal_sweep=sweep,
        image=dict(width=W, height=H, exif={k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in exif.items()}),
        board_config=board_cfg.to_dict(), board_autodetect=diag, rejected_boards=rejected,
        detections=[dict(board=d.board_index, offset=d.offset, n_corners=len(d.corner_ids), n_markers=d.n_markers,
                         reproj_rms_px=cam.per_board_rms[i], distance_m=float(np.median(Xc[i][:, 2])),
                         anchor_sigma_pct=100 * sig_board[i],
                         centroid_px=d.corners_px.mean(0).tolist(),
                         bbox_px=[*d.corners_px.min(0).tolist(), *d.corners_px.max(0).tolist()])
                    for i, d in enumerate(dets)],
        hardware=dict(prof.to_dict(), stride=stride, infer_max_mp=infer_max_mp),
        focal_prior=dict(fx=prior.fx, sigma_rel=prior.sigma_rel, source=prior.source),
        camera=dict(cam.summary(), source=calib_src),
        models={k: dict(backend=v["backend"], license=v["license"], family=v["align"]["family"],
                        infer_size=v.get("infer_size"),
                        params=v["align"].get("params"), validated=v["align"]["validated"],
                        cv=v["align"]["cv"], insample=v["align"].get("insample_summary"))
                for k, v in results.items()},
        chosen=best, chosen_cv=best_cv, sigma_rel_point=sig, n_points=int(len(cloud["xyz"])), frame=frame,
        gcp=gcp_report, files=files, runtime_s=round(time.time() - t0, 1))
    (out_dir / f"{stem}_report.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    files["report_json"] = str(out_dir / f"{stem}_report.json")
    files["report_html"] = str(report.write_html(out_dir / f"{stem}_report.html", summary, vis, R["Z"], results))
    _log(progress, f"Fatto in {summary['runtime_s']} s -> {out_dir}")
    return summary


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _gcp_registration(rows, dets, Xc):
    """Rigid (scale = 1) transform camera -> survey frame from surveyed ChArUco corners.
    It uses only the PnP corners, never the depth network, so the network output is not fitted
    to the control data."""
    A, B = [], []
    for r in rows:
        if r.get("type") != "board":
            continue
        bi, ci = int(r["board"]), int(r["corner"])
        for d, X in zip(dets, Xc):
            if d.board_index == bi and ci in d.corner_ids:
                A.append(X[list(d.corner_ids).index(ci)])
                B.append([float(r["x"]), float(r["y"]), float(r["z"])])
    if len(A) < 3:
        raise RuntimeError("GCP: servono almeno 3 corner ChArUco rilevati con coordinate rilevate.")
    A, B = np.array(A), np.array(B)
    keep = np.ones(len(A), bool)
    for _ in range(5):  # iterative rejection of control blunders (> 3 x median residual, at least 3 cm)
        _, Rr, tr = pointcloud.fit_rigid(A[keep], B[keep], with_scale=False)
        res = np.linalg.norm(A @ Rr.T + tr - B, axis=1)
        new = res <= max(0.03, 3 * float(np.median(res[keep])))
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    s_h, _, _ = pointcloud.fit_rigid(A[keep], B[keep], with_scale=True)
    ctrl = res[keep]
    return dict(R=Rr, t=tr, report=dict(n_control=int(keep.sum()), n_control_rejected=int((~keep).sum()),
                                         control_rmse_mm=float(1000 * np.sqrt(np.mean(ctrl ** 2))),
                                         helmert_scale_ppm=float((s_h - 1) * 1e6)))


def _local_normal(Z, K, uv, win=4):
    u, v = int(round(uv[0])), int(round(uv[1]))
    H, W = Z.shape
    y0, y1, x0, x1 = max(0, v - win), min(H, v + win + 1), max(0, u - win), min(W, u + win + 1)
    zz = Z[y0:y1, x0:x1]
    vv, uu = np.mgrid[y0:y1, x0:x1]
    m = zz > 0
    if m.sum() < 6:
        return None
    P = align.backproject(np.c_[uu[m], vv[m]].astype(float), zz[m].astype(float), K)
    n = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][-1]
    return n


def _check_points(rows, cam, K, Z, reg, sig_rel):
    """Errors at independent check points (type=pixel rows), in the survey frame.
    radial = component along the viewing ray, lateral = the rest. Survey Z axis assumed up."""
    Rr, tr = reg["R"], reg["t"]
    checks = []
    for r in rows:
        if r.get("type") != "pixel":
            continue
        uv = calib.undistort_points([[float(r["u"]), float(r["v"])]], cam)
        z = align.sample_depth(Z, Z > 0, uv, win=1)[0]
        if not np.isfinite(z):
            continue
        Xcam = align.backproject(uv, np.array([z]), K)[0]
        X = Xcam @ Rr.T + tr
        T = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        e = X - T
        ray = Rr @ (Xcam / np.linalg.norm(Xcam))
        er = float(e @ ray)
        el = float(np.linalg.norm(e - er * ray))
        n = _local_normal(Z, K, uv[0])
        orient, tilt = None, None
        if n is not None:
            tilt = float(np.degrees(np.arccos(min(1.0, abs(float((Rr @ n)[2]))))))  # 0 = horizontal surface
            orient = "orizzontale" if tilt < 30 else "verticale" if tilt > 60 else "inclinata"
        rng = float(np.linalg.norm(Xcam))
        checks.append(dict(name=r.get("name"), err_mm=float(1000 * np.linalg.norm(e)), dxyz_mm=(1000 * e).tolist(),
                           radial_mm=1000 * er, lateral_mm=1000 * el, range_m=rng, depth_m=float(z),
                           sigma_pred_mm=float(1000 * sig_rel * z), orientation=orient, tilt_deg=tilt))
    out = dict(n_check=len(checks), checks=checks)
    if checks:
        e = np.array([c["err_mm"] for c in checks])
        sp = np.array([c["sigma_pred_mm"] for c in checks])
        # 3D error of an isotropic-ish 1-sigma depth error: compare |e| with sigma (1D) and 2 sigma
        out.update(check_mae_mm=float(e.mean()), check_rmse_mm=float(np.sqrt(np.mean(e ** 2))),
                   check_d90_mm=float(np.percentile(e, 90)), check_max_mm=float(e.max()),
                   check_radial_rmse_mm=float(np.sqrt(np.mean([c["radial_mm"] ** 2 for c in checks]))),
                   check_lateral_rmse_mm=float(np.sqrt(np.mean([c["lateral_mm"] ** 2 for c in checks]))),
                   coverage_1s=float(np.mean(np.abs([c["radial_mm"] for c in checks]) <= sp)),
                   coverage_2s=float(np.mean(np.abs([c["radial_mm"] for c in checks]) <= 2 * sp)))
    return out


def _focal_sweep(eps_list, best, results, R, cam, dets, prior, rgb_u, valid_u, uv_u, obj, sig_board, W, H,
                 infer_max_mp, gcp_csv, progress):
    """Sensitivity of the reconstruction to a relative focal error eps: the whole chain is re-run
    with f(1+eps) (PnP anchors, network conditioning, alignment with the same family).
    Displacements w.r.t. the nominal solution are split into the component along the optical axis
    and the one in the image plane; first-order theory: dZ/Z = eps, dX = dY = 0."""
    key = best if best in backends.REGISTRY or best in backends._CACHE else next(k for k in results if k != "fusion")
    fam = results[key]["align"]["family"]
    Z0 = results[key]["Z"]
    rows = pointcloud.read_gcp_csv(gcp_csv) if gcp_csv else None
    rng = np.random.default_rng(0)
    vv, uu = np.nonzero(Z0 > 0)
    pick = rng.choice(len(vv), min(200_000, len(vv)), replace=False)
    uvs = np.c_[uu[pick], vv[pick]].astype(float)
    P0 = align.backproject(uvs, Z0[vv[pick], uu[pick]].astype(float), cam.K)
    r0 = np.linalg.norm(P0, axis=1)
    ray = P0 / r0[:, None]
    out = []
    for eps in [0.0] + [float(e) for e in eps_list if abs(e) > 1e-9]:
        K1 = cam.K.copy()
        K1[0, 0] *= 1 + eps
        K1[1, 1] *= 1 + eps
        c1 = calib.estimate_camera(dets, (W, H), prior, fixed_K=K1, fixed_dist=cam.dist)
        X1 = calib.corners_camera_frame(dets, c1)
        be = backends.get_backend(key)
        o = backends.infer_scaled(be, rgb_u, K1, infer_max_mp)
        msk = o["mask"] & valid_u & (o["depth"] > 0)
        dv = [align.sample_depth(o["depth"], msk, uv) for uv in uv_u]
        anc = align.Anchors(uv_u, X1, obj, dv, W, H, K1, sig_board, ids=[d.board_index for d in dets])
        p = align.fit(fam, *anc.stack(list(range(len(anc.boards)))))
        Z1 = align.apply_to_map(fam, p, o["depth"], msk)
        del o
        z1 = Z1[vv[pick], uu[pick]].astype(float)
        ok = z1 > 0
        P1 = align.backproject(uvs[ok], z1[ok], K1)
        dP = P1 - P0[ok]
        z0 = P0[ok][:, 2]
        da = dP[:, 2] / z0                               # along the optical axis
        dt = np.linalg.norm(dP[:, :2], axis=1) / z0      # in the image plane (X, Y)
        dr = np.sum(dP * ray[ok], 1) / r0[ok]            # along the viewing ray (for reference)
        row = dict(eps=eps, f=float(K1[0, 0]), model=key, family=fam,
                   axial_rel_median=float(np.median(da)), axial_rel_p90=float(np.percentile(np.abs(da), 90)),
                   transverse_rel_median=float(np.median(dt)), transverse_rel_p90=float(np.percentile(dt, 90)),
                   radial_rel_median=float(np.median(dr)))
        if rows:
            reg = _gcp_registration(rows, dets, X1)
            cp = _check_points(rows, c1, K1, Z1, reg, 0.0)
            row.update({k: v for k, v in cp.items() if k != "checks" and not k.startswith("coverage")})
        out.append(row)
        _log(progress, f"  sensibilità focale eps={100 * eps:+.1f}%: spostamento lungo l'asse ottico "
                       f"{100 * row['axial_rel_median']:+.2f}% di Z, nel piano immagine {100 * row['transverse_rel_median']:.2f}%")
    return out
