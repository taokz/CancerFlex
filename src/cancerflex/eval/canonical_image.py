"""Canonical image loader + preprocessing cache (single source of truth).

Why this module exists
----------------------
Two image loaders had diverged: `train_v2/collator.py` applied DICOM VOI LUT +
MONOCHROME1 inversion (clinically correct), while `core_code/eval_iterator.py`
used raw min-max — so the same breast CMMD `.dcm` rendered differently in train
vs eval (3744 `test_ood` rows affected). This module is the one decoder both
import, making that drift structurally impossible.

It also provides an offline preprocessing cache: large source images (cervix BMP,
breast DICOM, blood TIFF) are decoded once, resized to the model's native size,
and stored as lossless WebP. At train/eval time we read the small cached image
instead of re-decoding multi-MB originals every epoch / every run.

Parity
------
The cache resizes to 896x896 with **bilinear** — empirically the resize
Gemma3ImageProcessorFast uses internally (verified in tmp/imgcache_parity). With
a bilinear-896 cache the medgemma processor's resize is a no-op, so its
pixel_values are bit-exact vs decoding the original. gemma-3n (768 target) does
one extra 896->768 downscale; the deviation is metric-negligible (mean|Δ|~7e-4,
bf16-drift order). A single 896 cache therefore serves both models.

`load_cached` falls back to `load_image_rgb` on any cache miss, so wiring it in
is always safe — an unbuilt or partial cache just means the original is decoded
(current behavior). Deleting the cache directory fully reverts behavior.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)

# Cache version tag encodes the loader contract: bump it (new dir) whenever the
# decode or resize changes, so a stale cache is never silently mixed with new.
CACHE_VERSION = "v1_896_bilinear_voilut"
CACHE_ROOT = ROOT / "Experiment" / "datasets" / "_image_cache" / CACHE_VERSION
READY_ROOT = ROOT / "Experiment" / "datasets" / "_image_ready" / CACHE_VERSION / "images"
CACHE_SIZE = 896                       # square; serves medgemma (896) + gemma-3n (768)
_RESAMPLE = Image.Resampling.BILINEAR  # matches Gemma3ImageProcessorFast internal resize

# Extensions PIL handles directly; anything else (incl. no extension) is sniffed
# for the DICOM magic. Breast Mammo-MX stores JPEG-Lossless DICOM with NO
# extension — detecting by content (not suffix) recovers ~13.6k breast images
# that were silently dropped. Decoding JPEG-Lossless needs pylibjpeg-libjpeg.
_PIL_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _is_dicom(path: Path) -> bool:
    """True if the file carries the DICOM magic ('DICM' at byte offset 128)."""
    try:
        with open(path, "rb") as f:
            return f.read(132)[128:132] == b"DICM"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Canonical decode (the SSOT)
# ---------------------------------------------------------------------------

def load_image_rgb(rel_path: str, root: Path = ROOT, local_root: str | None = None) -> Image.Image:
    """Decode a manifest-relative image path to an RGB PIL.Image.

    DICOM is rendered clinically: VOI LUT windowing, MONOCHROME1 inversion, then
    min-max to uint8. All other formats go through PIL. Raises on unreadable
    input (callers decide whether to skip).

    `local_root`, if set, points at a byte-identical node-local mirror (NVMe /
    /dev/shm) used to dodge Lustre small-file latency; decode is identical.
    """
    if local_root:
        lp = Path(local_root) / rel_path
        abs_path = lp if lp.exists() else (root / rel_path).resolve()
    else:
        abs_path = (root / rel_path).resolve()

    suffix = abs_path.suffix.lower()
    # Route to pydicom for .dcm, or for ambiguous (extensionless / unknown-ext)
    # files that actually carry the DICOM magic. Known image extensions skip the
    # sniff (no extra I/O on the common path).
    is_dicom = suffix == ".dcm" or (suffix not in _PIL_EXTS and _is_dicom(abs_path))
    if is_dicom:
        import pydicom
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except Exception:
            apply_voi_lut = None
        ds = pydicom.dcmread(str(abs_path), force=True)
        arr = ds.pixel_array
        if apply_voi_lut is not None:
            arr = apply_voi_lut(arr, ds)
        arr = arr.astype("float32")
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-6)
        arr = (arr * 255.0).astype("uint8")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr).convert("RGB")

    return Image.open(str(abs_path)).convert("RGB")


# ---------------------------------------------------------------------------
# Cache layout + build
# ---------------------------------------------------------------------------

def cache_path(rel_path: str) -> Path:
    """Cache file for a source path: mirror the relative path under CACHE_ROOT,
    append .webp. Mirroring (vs hashing) keeps the cache human-debuggable."""
    return CACHE_ROOT / (rel_path + ".webp")


def ready_path(rel_path: str) -> Path:
    """Model-ready image location for a source path.

    The ready layer keeps the original relative path as the key, usually as a
    symlink to a preprocessed WebP for heavy images or to the original small
    image. Keeping the key stable avoids manifest/corpus churn while making the
    runtime path cache-first and easy to relocate.
    """
    return READY_ROOT / rel_path


def build_cache_entry(rel_path: str, overwrite: bool = False, root: Path = ROOT) -> str:
    """Decode -> bilinear 896x896 -> lossless WebP. Idempotent.

    Returns one of: "built", "skipped" (already present), "failed".
    """
    out = cache_path(rel_path)
    if out.exists() and not overwrite:
        return "skipped"
    try:
        img = load_image_rgb(rel_path, root=root)
        small = img.resize((CACHE_SIZE, CACHE_SIZE), _RESAMPLE)
        out.parent.mkdir(parents=True, exist_ok=True)
        # write to a temp then rename, so a killed job never leaves a partial file
        tmp = out.with_suffix(".webp.tmp")
        small.save(str(tmp), format="WEBP", lossless=True, quality=100, method=4)
        os.replace(tmp, out)
        return "built"
    except Exception:
        return "failed"


# ---------------------------------------------------------------------------
# Runtime read (cache-aware, always-safe fallback)
# ---------------------------------------------------------------------------

def _degrade(img: "Image.Image") -> "Image.Image":
    """OPT-IN acquisition degradation, off unless EDGE_IMAGE_DEGRADE is set. Diagnostic only.

    `EDGE_IMAGE_DEGRADE=WxH` downsamples to W x H and immediately resamples back, so the image
    keeps its original pixel dimensions but carries only W x H worth of real detail. Nothing else
    about the pipeline changes.

    WHY IT EXISTS. oral U4's external set is MODID, and MODID is not another clinic's phone: it is
    a 16-band IMEC multispectral camera, 460-600 nm, whose demosaiced per-band spatial resolution
    is 510x270 by construction (2048x1088 sensor under a 4x4 spectral filter array). Every training
    image is a 2.4-13 MP smartphone intraoral photo. Both are then resized to 896x896, so at the
    input the model sees training detail DOWNscaled 0.07x and MODID UPscaled 5.83x -- measured edge
    energy at 896 is 0.00102 for Piyarathne against 0.00016 for MODID, and high-frequency share
    0.256 against 0.140. The deployed model predicts OSMF 44 times internally (gold 66) and ZERO
    times on MODID (gold 86), so the capability exists and the source change suppresses it.
    Resolution is one candidate cause and colour is another (MODID's R/B is 1.25 against 1.47 for
    both phone sources, consistent with a sensor that stops at 600 nm and never sees red).
    Degrading the INTERNAL set to MODID's effective resolution separates them: if OSMF collapses
    internally under degradation alone, resolution is the mechanism and appearance augmentation on
    the training side is worth building. If it survives, resolution is exonerated and the cause is
    spectral, which no amount of phone data can fix.

    Set `EDGE_IMAGE_DEGRADE_COLOR=1` alongside to ALSO match MODID's channel statistics, as a
    second, separable arm.
    """
    spec = os.environ.get("EDGE_IMAGE_DEGRADE")
    colour = os.environ.get("EDGE_IMAGE_DEGRADE_COLOR") == "1"
    if not spec and not colour:
        return img
    out = img
    # The two arms are independent ON PURPOSE. A colour-only arm must not be faked by asking for a
    # spatial resize to the image's own size: a cache MISS returns the original at native
    # resolution, so "896x896" would silently become a real 5x downsample for that row and the two
    # causes would be confounded in exactly the run meant to separate them.
    if spec and spec.lower() != "none":
        w, h = (int(x) for x in spec.lower().split("x"))
        out = out.resize((w, h), _RESAMPLE).resize(img.size, _RESAMPLE)
    if colour:
        # MODID / phone channel means measured on the OSMF+leukoplakia rows of each set, 2026-09-20
        # (scripts/analysis/oral_source_domain_distance_20260920.py): MODID 0.377/0.307/0.302,
        # Piyarathne 0.524/0.379/0.356. Per-channel gain only -- no spatial change -- so this arm
        # is separable from the resolution arm above.
        a = np.asarray(out, dtype=np.float32)
        gain = np.array([0.377 / 0.524, 0.307 / 0.379, 0.302 / 0.356], dtype=np.float32)
        out = Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))
    return out


_CROP_MAP: dict | None = None
_CROP_MAP_SRC: str | None = None


def _crop_box(rel_path: str) -> "tuple[int, int, int, int, int, int] | None":
    """OPT-IN eval-time crop, off unless EDGE_IMAGE_CROP_JSON is set. Diagnostic only.

    `EDGE_IMAGE_CROP_JSON=<file>` points at a JSON object whose `boxes` maps a manifest-relative
    image_path to `[x1, y1, x2, y2, W, H]`: an integer PIXEL box on the ORIGINAL image
    (x = column, y = row; x2/y2 exclusive, i.e. PIL `Image.crop` semantics) plus the original
    width/height it was computed on. For a listed path, `load_cached` decodes the ORIGINAL (not
    the 896 cache, so the crop keeps native resolution), asserts the size matches W x H (fails
    loudly on a coordinate-frame mismatch), and returns the crop. Unlisted paths and an unset
    variable are the exact identity: this function returns None and nothing else changes.

    WHY IT EXISTS (2026-09-23, thyroid nodule-crop re-evaluation). Dedicated thyroid classifiers
    crop to the nodule using the boxes/masks that ship with TN5000 / TN3K / DDTI / ThyroidXL /
    Stanford; we feed the whole frame resized to 896. This hook tests the preprocessing alone,
    on a model trained on whole images (built by
    scripts/eval/thyroid_nodulecrop_20260923/build_crop_boxes.py).

    The frozen-vision embedding cache is keyed by path and would silently bypass the crop, so
    run_closed refuses EVAL_VISION_CACHE=1 together with this variable.
    """
    global _CROP_MAP, _CROP_MAP_SRC
    src = os.environ.get("EDGE_IMAGE_CROP_JSON")
    if not src:
        return None
    if _CROP_MAP is None or _CROP_MAP_SRC != src:
        import json
        with open(src) as fh:
            _CROP_MAP = json.load(fh)["boxes"]
        _CROP_MAP_SRC = src
    b = _CROP_MAP.get(rel_path)
    return None if b is None else tuple(int(v) for v in b)


def load_cached(rel_path: str, root: Path = ROOT, local_root: str | None = None) -> Image.Image:
    """Return the model-ready/cache image if present, else decode the original.

    If EDGE_IMAGE_CROP_JSON lists `rel_path`, the original is decoded and cropped instead
    (see `_crop_box`); otherwise behavior is unchanged.

    Resolution order:
    1. `_image_ready/<version>/images/<rel_path>` (unified runtime layer).
    2. `_image_cache/<version>/<rel_path>.webp` (legacy heavy-image cache).
    3. `local_root/<rel_path>` if provided.
    4. original `root/<rel_path>`.

    A miss transparently decodes the original. Raises only if every path is
    unreadable.
    """
    box = _crop_box(rel_path)
    if box is not None:
        x1, y1, x2, y2, w, h = box
        img = load_image_rgb(rel_path, root=root, local_root=local_root)
        if img.size != (w, h):
            raise ValueError(f"EDGE_IMAGE_CROP_JSON box for {rel_path} was computed on {w}x{h} "
                             f"but the original decodes as {img.size[0]}x{img.size[1]}")
        return _degrade(img.crop((x1, y1, x2, y2)))
    if os.environ.get("EDGE_IMAGE_PAD_SQUARE"):
        # Opt-in letterbox (see _pad_square). Unset -> this branch is skipped entirely.
        return _degrade(_pad_square(load_image_rgb(rel_path, root=root, local_root=local_root)))
    rp = ready_path(rel_path)
    if rp.exists():
        try:
            return _degrade(Image.open(str(rp)).convert("RGB"))
        except Exception:
            pass  # corrupt ready link/file -> fall through
    cp = cache_path(rel_path)
    if cp.exists():
        try:
            return _degrade(Image.open(str(cp)).convert("RGB"))
        except Exception:
            pass  # corrupt/partial cache file -> fall through to original
    return _degrade(load_image_rgb(rel_path, root=root, local_root=local_root))


def _pad_square(img: "Image.Image") -> "Image.Image":
    """OPT-IN pad-to-square, off unless EDGE_IMAGE_PAD_SQUARE is set. Diagnostic only.

    `EDGE_IMAGE_PAD_SQUARE=black` letterboxes the ORIGINAL image (decoded at native resolution,
    never the 896 cache, which is already stretched to a square) onto a black square canvas of
    side max(W, H), centred, then resizes bilinear to CACHE_SIZE -- exactly the resize the cache
    applies -- so the only difference from the default path is the aspect ratio (padding instead
    of anisotropic stretch). Unset = exact identity: load_cached never calls this.

    WHY IT EXISTS (2026-09-23, weak-cell audit RESULTS section 22): AnnoCerv colposcopy frames
    are 3:2 (e.g. 2976x1984) and the default path stretches them to 896x896, compressing the
    horizontal axis by 1.5x. Published colposcopy pipelines keep aspect. This tests whether the
    stretch costs AUROC on the external cervix U2 cell, eval-only (model trained on stretched
    images, so it measures preprocessing alone).

    The frozen-vision embedding cache is keyed by path and would silently return the stretched
    features, so this hook refuses to run with EVAL_VISION_CACHE=1.
    """
    mode = os.environ.get("EDGE_IMAGE_PAD_SQUARE", "").lower()
    if mode != "black":
        raise ValueError(f"EDGE_IMAGE_PAD_SQUARE={mode!r}: only 'black' is implemented")
    if os.environ.get("EVAL_VISION_CACHE") == "1":
        raise ValueError("EDGE_IMAGE_PAD_SQUARE is incompatible with EVAL_VISION_CACHE=1 "
                         "(path-keyed embedding cache would bypass the padding); unset the cache")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((CACHE_SIZE, CACHE_SIZE), _RESAMPLE)
