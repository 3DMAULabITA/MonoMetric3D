"""Image loading (EXIF orientation, HEIC) and focal-length prior from EXIF."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC support
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

# EXIF tag ids
_EXIF_IFD = 0x8769
_TAGS = {
    "Make": 0x010F,
    "Model": 0x0110,
    "FocalLength": 0x920A,
    "FocalLengthIn35mmFilm": 0xA405,
    "FocalPlaneXResolution": 0xA20E,
    "FocalPlaneYResolution": 0xA20F,
    "FocalPlaneResolutionUnit": 0xA210,
    "PixelXDimension": 0xA002,
    "PixelYDimension": 0xA003,
    "LensModel": 0xA434,
    "LensMake": 0xA433,
}
_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}

# Minimal camera database (diagonal-based 35 mm equivalent focal lengths).
# Used only when EXIF lacks FocalLengthIn35mmFilm.
_PHONE_DB = {
    ("apple", "iphone 12"): {4.2: 26.0, 1.55: 13.0, 2.71: 23.0},
}


@dataclass
class FocalPrior:
    fx: float
    sigma_rel: float
    source: str
    exif: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        try:
            return float(v[0]) / float(v[1])
        except Exception:
            return None


def read_exif(img: Image.Image) -> dict:
    out = {}
    try:
        ex = img.getexif()
    except Exception:
        return out
    if not ex:
        return out
    sub = {}
    try:
        sub = ex.get_ifd(_EXIF_IFD)
    except Exception:
        pass
    for name, tag in _TAGS.items():
        v = sub.get(tag, ex.get(tag))
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode(errors="ignore").strip("\x00 ")
        out[name] = v if isinstance(v, str) else _num(v)
    try:
        out["Orientation"] = int(ex.get(0x0112, 1))
    except Exception:
        out["Orientation"] = 1
    return out


def load_image(path: str | Path):
    """Return (rgb uint8 HxWx3 with EXIF orientation applied, exif dict)."""
    img = Image.open(path)
    exif = read_exif(img)
    img = ImageOps.exif_transpose(img)
    rgb = np.asarray(img.convert("RGB")).copy()
    return rgb, exif


def focal_prior_from_exif(exif: dict, width: int, height: int) -> FocalPrior:
    """Focal length in pixels (for the image as loaded) + relative uncertainty.

    Priority:
      1. FocalLength [mm] x FocalPlaneXResolution (DSLR/mirrorless): sigma ~2 %
      2. FocalLengthIn35mmFilm (diagonal convention, CIPA DC-008): sigma ~3 %
      3. Small built-in phone DB
      4. Generic guess (fx = 1.2 * max(W, H)): sigma 30 %
    """
    diag_px = math.hypot(width, height)
    f_mm = exif.get("FocalLength")
    rot = exif.get("Orientation", 1) in (5, 6, 7, 8)

    # 1. focal plane resolution (resolution refers to the sensor/original image)
    fpx, fpu = exif.get("FocalPlaneXResolution"), exif.get("FocalPlaneResolutionUnit")
    if f_mm and fpx and fpu in _UNIT_MM:
        px_per_mm = fpx / _UNIT_MM[int(fpu)]
        f_px_orig = f_mm * px_per_mm
        # rescale if the stored image is smaller than the native sensor image
        native_w = exif.get("PixelXDimension")
        cur_w = height if rot else width
        scale = (cur_w / native_w) if native_w else 1.0
        # sanity: sensor width implied must be plausible (5-60 mm)
        sensor_w_mm = (native_w or cur_w) / px_per_mm
        if 4.0 < sensor_w_mm < 60.0:
            return FocalPrior(f_px_orig * scale, 0.02, "exif_focal_plane", exif)

    # 2. 35 mm equivalent (diagonal convention)
    f35 = exif.get("FocalLengthIn35mmFilm")
    if f35 and f35 > 0:
        return FocalPrior(f35 * diag_px / math.hypot(36.0, 24.0), 0.03, "exif_35mm_equiv", exif)

    # 3. phone DB
    make, model = str(exif.get("Make", "")).lower(), str(exif.get("Model", "")).lower()
    for (mk, md), table in _PHONE_DB.items():
        if mk in make and md in model and f_mm:
            best = min(table, key=lambda k: abs(k - f_mm))
            if abs(best - f_mm) < 0.3:
                return FocalPrior(table[best] * diag_px / math.hypot(36.0, 24.0), 0.04, "phone_db", exif)

    return FocalPrior(1.2 * max(width, height), 0.30, "generic_guess", exif)
