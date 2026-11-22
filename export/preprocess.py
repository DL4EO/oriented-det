"""Image → ONNX input tensor (training-style resize + RGB mean/std).

The ONNX graph expects ``[1, 3, H, W]`` float32 RGB, already normalized:
``(x / 255 - mean) / std``. Resize is bilinear (PIL), matching oriented-det
``resize_mode=fixed``.

Images larger than the sliding-window slice are tiled here. The slice is
``window_size`` in source pixels when set, otherwise ``target_size``. Each
window is resized onto ``target_size`` before normalization. ``pad`` and
``keep_ratio`` stay a single full-image resize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
from PIL import Image

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow < 9.1
    _RESAMPLE = Image.BILINEAR


def _as_float3(values: Sequence[float] | None, default: Sequence[float]) -> np.ndarray:
    src = default if values is None else values
    arr = np.asarray(list(src), dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"expected 3-channel mean/std, got {arr.size}")
    return arr


def canvas_size_from_meta(meta: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(height, width)`` from export meta."""
    prep = meta.get("preprocess") or {}
    ts = prep.get("target_size")
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return int(ts[0]), int(ts[1])
    shape = (meta.get("input") or {}).get("shape") or []
    if len(shape) >= 4:
        return int(shape[2]), int(shape[3])
    size = int(meta.get("image_size") or 1024)
    return size, size


def mean_std_from_meta(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prep = meta.get("preprocess") or {}
    mean = prep.get("normalize_mean") or meta.get("normalize_mean") or _DEFAULT_MEAN
    std = prep.get("normalize_std") or meta.get("normalize_std") or _DEFAULT_STD
    return _as_float3(mean, _DEFAULT_MEAN), _as_float3(std, _DEFAULT_STD)


def to_rgb_uint8(image: Union[Image.Image, np.ndarray], *, bgr: bool = False) -> np.ndarray:
    """Return HWC uint8 RGB."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image with 3 or 4 channels, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if bgr:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def preprocess_rgb(
    image: Union[Image.Image, np.ndarray],
    height: int,
    width: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    *,
    bgr: bool = False,
) -> tuple[np.ndarray, int, int]:
    """RGB (or BGR) image → ``[1, 3, H, W]`` float32 plus original ``(h, w)``."""
    rgb = to_rgb_uint8(image, bgr=bgr)
    orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])
    pil = Image.fromarray(rgb, mode="RGB")
    if pil.size != (int(width), int(height)):
        pil = pil.resize((int(width), int(height)), _RESAMPLE)
    chw = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean_a = _as_float3(mean, _DEFAULT_MEAN).reshape(3, 1, 1)
    std_a = _as_float3(std, _DEFAULT_STD).reshape(3, 1, 1)
    out = (chw - mean_a) / std_a
    return out[np.newaxis, ...], orig_h, orig_w


def preprocess_from_meta(
    image: Union[Image.Image, np.ndarray],
    meta: Mapping[str, Any],
    *,
    bgr: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Preprocess using ``normalize_mean`` / ``target_size`` from export meta."""
    height, width = canvas_size_from_meta(meta)
    mean, std = mean_std_from_meta(meta)
    return preprocess_rgb(image, height, width, mean, std, bgr=bgr)


def preprocess_path(
    path: Union[str, Path],
    meta: Mapping[str, Any],
) -> tuple[np.ndarray, int, int]:
    """Load a file as RGB and preprocess."""
    img = Image.open(path).convert("RGB")
    return preprocess_from_meta(img, meta)


def scale_obb_to_original(
    boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Scale ``(cx, cy, w, h, angle)`` from canvas pixels to original image pixels."""
    if boxes.size == 0:
        return boxes.reshape(0, 5)
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[:, 0] *= float(orig_w) / float(canvas_w)
    out[:, 1] *= float(orig_h) / float(canvas_h)
    out[:, 2] *= float(orig_w) / float(canvas_w)
    out[:, 3] *= float(orig_h) / float(canvas_h)
    return out


def _prep(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    prep = meta.get("preprocess")
    return prep if isinstance(prep, Mapping) else {}


def resize_mode_from_meta(meta: Mapping[str, Any]) -> str:
    """Return the sidecar resize mode (default ``fixed``)."""
    return str(_prep(meta).get("resize_mode") or "fixed").strip().lower()


def _size_pair(value: Any) -> tuple[int, int] | None:
    """Return ``(h, w)`` from a scalar or a 1- or 2-length sequence."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return (n, n)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            n = int(value[0])
            return (n, n)
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
    return None


def window_size_from_meta(meta: Mapping[str, Any]) -> tuple[int, int] | None:
    """Source-pixel slice from ``preprocess.window_size``, or None.

    ``pad`` and ``keep_ratio`` ignore ``window_size``.
    """
    if resize_mode_from_meta(meta) in {"pad", "keep_ratio"}:
        return None
    return _size_pair(_prep(meta).get("window_size"))


def slice_size_from_meta(meta: Mapping[str, Any]) -> tuple[int, int]:
    """Sliding-window slice ``(h, w)`` in source pixels.

    ``window_size`` when set, otherwise the ONNX canvas (``target_size``).
    """
    window = window_size_from_meta(meta)
    if window is not None:
        return window
    return canvas_size_from_meta(meta)


def overlap_pixels_from_meta(meta: Mapping[str, Any]) -> int:
    """Per-axis overlap in source pixels (default 200)."""
    prep = _prep(meta)
    raw = prep.get("overlap_pixels")
    if raw is None:
        prod = meta.get("production")
        if isinstance(prod, Mapping):
            raw = prod.get("overlap_pixels")
    if raw is None:
        return 200
    return max(0, int(raw))


def uses_sliding_window(image_height: int, image_width: int, meta: Mapping[str, Any]) -> bool:
    """True when ``fixed`` / ``crop`` should tile an image larger than the slice.

    The slice is ``window_size`` when set, otherwise ``target_size``. An image
    that fits in the slice is one forward. ``pad`` and ``keep_ratio`` never tile.
    """
    if resize_mode_from_meta(meta) in {"pad", "keep_ratio"}:
        return False
    slice_h, slice_w = slice_size_from_meta(meta)
    return int(image_height) > int(slice_h) or int(image_width) > int(slice_w)


def _sliding_window_axis_origins(length: int, slice_size: int, stride: int) -> list[int]:
    """Top-left origins along one axis. The last window flushes to the image edge."""
    if length <= slice_size:
        return [0]
    origins: list[int] = []
    pos = 0
    while pos < length:
        origins.append(pos)
        pos += stride
        if pos >= length:
            break
    origins[-1] = max(0, length - slice_size)
    out: list[int] = []
    seen: set[int] = set()
    for origin in origins:
        if origin not in seen:
            out.append(origin)
            seen.add(origin)
    return out


def sliding_window_grid(
    image_height: int,
    image_width: int,
    slice_h: int,
    slice_w: int,
    overlap_pixels: int | None = None,
) -> list[tuple[int, int]]:
    """``(x0, y0)`` windows. Stride is ``slice - overlap``. Default overlap is 200 px.

    The last row and column flush to the image edge so that window is a full
    slice of real pixels. Images that fit in one slice return ``[(0, 0)]``.
    """
    if image_height <= slice_h and image_width <= slice_w:
        return [(0, 0)]
    op = 200 if overlap_pixels is None else int(overlap_pixels)
    if op < 0:
        raise ValueError("overlap_pixels must be >= 0")
    oh = min(op, max(0, int(slice_h) - 1))
    ow = min(op, max(0, int(slice_w) - 1))
    stride_y = max(1, int(round(slice_h - oh)))
    stride_x = max(1, int(round(slice_w - ow)))
    xs = _sliding_window_axis_origins(int(image_width), int(slice_w), stride_x)
    ys = _sliding_window_axis_origins(int(image_height), int(slice_h), stride_y)
    return [(x0, y0) for y0 in ys for x0 in xs]


def crop_or_pad(
    rgb: np.ndarray,
    x0: int,
    y0: int,
    slice_w: int,
    slice_h: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Copy a source window onto a ``slice_h × slice_w`` uint8 canvas.

    Regions past the image border stay zero. ``rgb`` is HWC RGB.
    """
    x1 = min(int(x0) + int(slice_w), int(image_width))
    y1 = min(int(y0) + int(slice_h), int(image_height))
    src_x0 = max(0, int(x0))
    src_y0 = max(0, int(y0))
    crop_w = x1 - src_x0
    crop_h = y1 - src_y0
    canvas = np.zeros((int(slice_h), int(slice_w), 3), dtype=np.uint8)
    if crop_w > 0 and crop_h > 0:
        dx = src_x0 - int(x0)
        dy = src_y0 - int(y0)
        canvas[dy : dy + crop_h, dx : dx + crop_w, :] = rgb[src_y0:y1, src_x0:x1]
    return canvas
