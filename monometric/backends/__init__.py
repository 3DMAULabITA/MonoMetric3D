"""Monocular geometry backends. Every backend returns metric-ish depth for the
UNDISTORTED image, conditioned on the estimated intrinsics when the model allows it.

    out = backend.infer(rgb_uint8_HxWx3, K_3x3)
    out["depth"]  (H,W) float32  [m, model scale]
    out["mask"]   (H,W) bool
    out["normal"] (H,W,3) optional,  out["conf"] (H,W) optional
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_CACHE = {}


def _device():
    import torch

    from ..device import active

    return torch.device(active().device)


def _is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def infer_scaled(be, rgb, K, max_mp=None, log=None):
    """Run a backend on a copy of the image limited to `max_mp` megapixels (intrinsics scaled
    accordingly), then upsample depth/mask to the full image. On CUDA out-of-memory the
    resolution is reduced and the inference retried (laptop GPUs)."""
    from ..device import active, free_gpu

    H, W = rgb.shape[:2]
    max_mp = max_mp or active().infer_max_mp
    s = min(1.0, math.sqrt(max_mp * 1e6 / (H * W)))
    last = None
    for _ in range(4):
        if s < 0.999:
            w, h = max(64, round(W * s)), max(64, round(H * s))
            sx, sy = w / W, h / H
            img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            Ks = K.astype(float).copy()
            Ks[0, 0] *= sx
            Ks[1, 1] *= sy
            Ks[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
            Ks[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        else:
            img, Ks = rgb, K
        try:
            o = be.infer(img, Ks)
            break
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            last = e
            free_gpu()
            s *= 0.7
            if log:
                log(f"  memoria GPU insufficiente: riprovo a {100 * s:.0f}% della risoluzione")
    else:
        raise RuntimeError(f"Memoria GPU insufficiente anche a risoluzione ridotta: {last}")
    out = {"depth": _resize_to(np.ascontiguousarray(o["depth"], np.float32), H, W),
           "mask": _resize_to(o["mask"].astype(np.uint8), H, W, nearest=True).astype(bool)}
    if "conf" in o:
        out["conf"] = _resize_to(o["conf"].astype(np.float32), H, W)
    out["infer_size"] = (img.shape[1], img.shape[0])
    del o
    free_gpu()
    return out


def _resize_to(a, H, W, nearest=False):
    if a.shape[:2] == (H, W):
        return a
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


class Backend:
    name = "base"
    license = ""
    accepts = ""

    def load(self):
        raise NotImplementedError

    def infer(self, rgb, K):
        raise NotImplementedError


# ----------------------------------------------------------------------------- MoGe-2 / MoGe-3
class MoGe(Backend):
    """Microsoft MoGe-2 (Wang et al., 2025) / MoGe-3 (Kong et al., 2026). Conditioned on horizontal FoV.
    MoGe-3 adds a sparse 3D refiner (FlexGEMM/Triton -> Linux or WSL2)."""
    accepts = "fov_x"
    license = "MIT"

    def __init__(self, hf_id="Ruicheng/moge-2-vitl-normal", version=2, fp16=True, resolution_level=9, refine_steps=3):
        self.hf_id, self.version, self.fp16, self.res_level = hf_id, version, fp16, resolution_level
        self.refine_steps = refine_steps
        self.name = f"moge{version}:{hf_id.split('/')[-1]}"
        self.model = None

    def load(self):
        import importlib

        mod = importlib.import_module(f"moge.model.v{self.version}")
        self.model = mod.MoGeModel.from_pretrained(self.hf_id).to(_device()).eval()
        return self

    def infer(self, rgb, K):
        import torch

        H, W = rgb.shape[:2]
        fov_x = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
        t = torch.tensor(rgb, dtype=torch.float32, device=_device()).permute(2, 0, 1) / 255.0
        from ..device import active

        use_fp16 = self.fp16 and active().fp16
        with torch.inference_mode():
            kw = dict(fov_x=fov_x, resolution_level=self.res_level, use_fp16=use_fp16)
            if self.version >= 3:
                kw["refine_steps"] = self.refine_steps
            o = self.model.infer(t, **kw)
        # only depth and mask are used downstream: copy them and drop the rest (RAM/VRAM)
        out = {"depth": o["depth"].float().cpu().numpy(), "mask": o["mask"].cpu().numpy().astype(bool)}
        del o, t
        out["depth"][~np.isfinite(out["depth"])] = 0
        return out


# ----------------------------------------------------------------------------- UniDepthV2
class UniDepthV2(Backend):
    """UniDepthV2 (Piccinelli et al., 2025). Conditioned on full K."""
    accepts = "K"
    license = "CC BY-NC 4.0"

    def __init__(self, hf_id="lpiccinelli/unidepth-v2-vitl14", resolution_level=9):
        self.hf_id, self.res_level = hf_id, resolution_level
        self.name = f"unidepthv2:{hf_id.split('/')[-1]}"

    def load(self):
        from unidepth.models import UniDepthV2 as M

        self.model = M.from_pretrained(self.hf_id).to(_device()).eval()
        try:
            self.model.resolution_level = self.res_level
        except Exception:
            pass
        return self

    def infer(self, rgb, K):
        import torch
        from unidepth.utils.camera import Pinhole

        H, W = rgb.shape[:2]
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(_device())
        cam = Pinhole(K=torch.tensor(K, dtype=torch.float32))  # 3x3 as in the official README
        with torch.inference_mode():
            o = self.model.infer(t, cam)
        d = o["depth"].squeeze().float().cpu().numpy()
        d = _resize_to(d, H, W)
        out = {"depth": d, "mask": np.isfinite(d) & (d > 0)}
        if "confidence" in o:
            out["conf"] = _resize_to(o["confidence"].squeeze().float().cpu().numpy(), H, W)
        return out


# ----------------------------------------------------------------------------- Depth Pro
class DepthPro(Backend):
    """Apple Depth Pro (Bochkovskii et al., 2025). Conditioned on focal length."""
    accepts = "f_px"
    license = "Apple sample code license"

    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.name = "depthpro"

    def load(self):
        import depth_pro
        import torch

        kw = {}
        if self.checkpoint:
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT as C

            C.checkpoint_uri = self.checkpoint
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=_device(), precision=torch.half if _device().type == "cuda" else torch.float32, **kw)
        self.model.eval()
        return self

    def infer(self, rgb, K):
        import torch

        H, W = rgb.shape[:2]
        x = self.transform(rgb)
        with torch.inference_mode():
            o = self.model.infer(x, f_px=float(K[0, 0]))
        d = _resize_to(o["depth"].squeeze().float().cpu().numpy(), H, W)
        return {"depth": d, "mask": np.isfinite(d) & (d > 0)}


# ----------------------------------------------------------------------------- Depth Anything 3
class DA3(Backend):
    """Depth Anything 3 (Lin et al., 2025). Intrinsics cannot be passed as input.
    DA3METRIC: metric = focal_px(processed res) * net / 300.  DA3NESTED: metres."""
    accepts = "none (focal used for metric conversion)"

    def __init__(self, hf_id="depth-anything/DA3METRIC-LARGE", process_res=1008):
        self.hf_id, self.process_res = hf_id, process_res
        self.name = f"da3:{hf_id.split('/')[-1]}"
        self.license = "Apache 2.0" if "METRIC" in hf_id or "MONO" in hf_id else "CC BY-NC 4.0"

    def load(self):
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(self.hf_id).to(_device())
        return self

    def infer(self, rgb, K):
        H, W = rgb.shape[:2]
        try:
            p = self.model.inference([rgb], process_res=self.process_res)
        except TypeError:
            p = self.model.inference([rgb])
        d = np.asarray(p.depth[0], np.float32)
        h, w = d.shape
        if "METRIC" in self.hf_id.upper():
            f_proc = K[0, 0] * (w / W)
            d = f_proc * d / 300.0
        out = {"depth": _resize_to(d, H, W)}
        out["mask"] = np.isfinite(out["depth"]) & (out["depth"] > 0)
        if getattr(p, "conf", None) is not None:
            out["conf"] = _resize_to(np.asarray(p.conf[0], np.float32), H, W)
        return out


# ----------------------------------------------------------------------------- registry
REGISTRY = {
    "moge2-vitl": lambda: MoGe("Ruicheng/moge-2-vitl-normal", 2),
    "moge2-vitb": lambda: MoGe("Ruicheng/moge-2-vitb-normal", 2),
    "moge2-vits": lambda: MoGe("Ruicheng/moge-2-vits-normal", 2),
    # MoGe-3 (Kong et al., 2026, released 2026-08-18): needs flex_gemm (Triton) -> Linux / WSL2
    "moge3-vitl": lambda: MoGe("Ruicheng/moge-3-vitl", 3),
    "moge3-vitg": lambda: MoGe("Ruicheng/moge-3-vitg", 3),
    "unidepthv2-vitl": lambda: UniDepthV2(),
    "depthpro": lambda: DepthPro(),
    "da3-metric": lambda: DA3("depth-anything/DA3METRIC-LARGE"),
    "da3-nested": lambda: DA3("depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
}


def get_backend(key):
    from ..device import active, free_gpu

    if key in _CACHE:
        return _CACHE[key]
    if key not in REGISTRY:
        raise KeyError(f"Backend sconosciuto: {key}. Disponibili: {list(REGISTRY)}")
    if active().single_model:  # laptop GPUs: keep a single network in VRAM
        for k in list(_CACHE):
            if getattr(_CACHE[k], "model", None) is not None:
                _CACHE.pop(k)
        free_gpu()
    b = REGISTRY[key]().load()
    _CACHE[key] = b
    return b


def register(key, backend_obj):
    """Register an already-loaded backend (used by tests or custom models)."""
    _CACHE[key] = backend_obj
    REGISTRY.setdefault(key, lambda: backend_obj)


def available():
    import importlib.util as u

    mods = {"moge2-vitl": "moge", "moge2-vitb": "moge", "moge2-vits": "moge", "moge3-vitl": "flex_gemm",
            "moge3-vitg": "flex_gemm",
            "unidepthv2-vitl": "unidepth", "depthpro": "depth_pro", "da3-metric": "depth_anything_3",
            "da3-nested": "depth_anything_3"}
    return [k for k, m in mods.items() if u.find_spec(m) is not None] + [k for k in _CACHE if k not in mods]
