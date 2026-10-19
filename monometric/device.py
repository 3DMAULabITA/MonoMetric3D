"""Hardware detection and execution profiles (workstation / laptop / lowmem / cpu).

The profile bounds GPU and RAM usage:
    infer_max_mp   megapixels fed to the depth network (output is upsampled to full size)
    single_model   keep only one network in memory at a time (VRAM <= 12 GB)
    default_stride point-cloud sampling step (pixels)
    fp16           half precision on GPU
Override with the environment variable MONOMETRIC_PROFILE or --profile.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class Profile:
    name: str
    device: str
    gpu: str
    vram_gb: float
    ram_gb: float
    infer_max_mp: float
    single_model: bool
    default_stride: int
    fp16: bool
    default_models: tuple

    def to_dict(self):
        return asdict(self)

    def describe(self):
        g = f"{self.gpu} ({self.vram_gb:.0f} GB)" if self.device == "cuda" else self.device.upper()
        return (f"profilo '{self.name}': {g}, RAM {self.ram_gb:.0f} GB, inferenza fino a {self.infer_max_mp:.0f} MP, "
                f"{'un modello alla volta' if self.single_model else 'modelli in cache'}, passo nuvola {self.default_stride} px")


PRESETS = {
    #            infer_max_mp single  stride fp16  default models
    "workstation": (24.0, False, 2, True, ("moge2-vitl",)),
    "laptop": (12.0, True, 2, True, ("moge2-vitl",)),
    "lowmem": (6.0, True, 3, True, ("moge2-vitb",)),
    "cpu": (4.0, True, 3, False, ("moge2-vitb",)),
}


def _ram_gb():
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except Exception:
            return 16.0


def detect(name: str | None = None) -> Profile:
    name = name or os.environ.get("MONOMETRIC_PROFILE") or "auto"
    ram = _ram_gb()
    device, gpu, vram = "cpu", "", 0.0
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            p = torch.cuda.get_device_properties(0)
            gpu, vram = p.name, p.total_memory / 2**30
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, gpu = "mps", "Apple GPU"
    except Exception:
        pass
    if name == "auto":
        if device == "cpu":
            name = "cpu"
        elif device == "mps":
            name = "laptop" if ram >= 16 else "lowmem"
        elif vram >= 16 and ram >= 24:
            name = "workstation"
        elif vram >= 7:
            name = "laptop"
        else:
            name = "lowmem"
    if name not in PRESETS:
        raise ValueError(f"Profilo sconosciuto: {name}. Disponibili: auto, {', '.join(PRESETS)}")
    mp, single, stride, fp16, models = PRESETS[name]
    # RAM guard: full-resolution maps of 24 MP need ~3-4 GB of RAM per model
    if ram < 12:
        mp, stride = min(mp, 6.0), max(stride, 3)
    return Profile(name, device, gpu, vram, ram, mp, single, stride, fp16 and device == "cuda", models)


_ACTIVE: Profile | None = None


def active() -> Profile:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = detect()
    return _ACTIVE


def set_profile(name: str | None):
    global _ACTIVE
    _ACTIVE = detect(name)
    return _ACTIVE


def free_gpu():
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
