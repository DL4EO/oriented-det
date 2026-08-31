"""Spatial image preprocessing for training and inference.

Modes (``resize_mode`` in config):

- **fixed** — stretch to ``target_size`` (H, W); aspect ratio may change.
- **pad** — uniform zoom/dezoom so the **larger** image side equals ``max(H, W)`` of
  ``target_size``, then zero-pad the remaining canvas; aspect ratio is always preserved.
- **keep_ratio** — same uniform scale as **pad** (long edge → ``max(H, W)``), but **no**
  square canvas pad; training/inference then apply ``pad_size_divisor`` (MMRotate
  ``RResize`` + ``Pad(size_divisor=…)``). Aspect ratio is preserved.
- **crop** — extract a ``target_size`` window at **native resolution** (no resize):
  pad when the image is too small in one or both dimensions, randomly crop the excess
  when it is too large (center crop when ``random_crop=False``); aspect ratio is always
  preserved because pixels are never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import random

from PIL import Image

from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox

TargetSize = Union[int, Tuple[int, int], List[int]]


@dataclass(frozen=True)
class SpatialPreprocessMeta:
    """Metadata to map coordinates between preprocessed and original image space."""

    mode: str
    orig_size: Tuple[int, int]  # (width, height)
    canvas_size: Tuple[int, int]  # (height, width)
    scale: float = 1.0  # uniform scale (pad) or sx (fixed uses scale_x/scale_y)
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_left: int = 0
    pad_top: int = 0
    crop_left: int = 0
    crop_top: int = 0
    content_size: Optional[Tuple[int, int]] = None  # (height, width) before divisor pad


@dataclass
class SpatialPreprocessResult:
    image: Image.Image
    rboxes: List[RBox]
    meta: SpatialPreprocessMeta


def parse_canvas_size(mode: str, target_size: TargetSize) -> Tuple[int, int]:
    """Return canvas (height, width) for the given mode and target_size."""
    if isinstance(target_size, (int, float)):
        side = int(target_size)
        return (side, side)
    if isinstance(target_size, (list, tuple)):
        if len(target_size) >= 2:
            return (int(target_size[0]), int(target_size[1]))
        if len(target_size) == 1:
            side = int(target_size[0])
            return (side, side)
    return (1024, 1024)


def large_edge_from_target(target_size: TargetSize) -> int:
    """Longer canvas edge used as the pad-mode scaling reference."""
    h, w = parse_canvas_size("pad", target_size)
    return max(h, w)


def get_model_canvas_size(mode: str, target_size: TargetSize) -> Tuple[int, int]:
    """Return (slice_h, slice_w) for sliding-window / model input canvas."""
    return parse_canvas_size(mode, target_size)


def _pad_image(
    image: Image.Image,
    canvas_h: int,
    canvas_w: int,
    pad_left: int,
    pad_top: int,
) -> Image.Image:
    if image.size == (canvas_w, canvas_h) and pad_left == 0 and pad_top == 0:
        return image
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(image, (pad_left, pad_top))
    return canvas


def apply_pad_mode(
    image: Image.Image,
    rboxes: Sequence[RBox],
    target_size: TargetSize,
) -> SpatialPreprocessResult:
    """Scale by large_edge (uniform), then pad to canvas; preserve aspect ratio."""
    orig_w, orig_h = image.size
    canvas_h, canvas_w = parse_canvas_size("pad", target_size)
    large_edge = max(canvas_h, canvas_w)

    scale = large_edge / max(orig_h, orig_w, 1)
    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))

    if (scaled_w, scaled_h) != (orig_w, orig_h):
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)

    pad_left = max(0, (canvas_w - scaled_w) // 2)
    pad_top = max(0, (canvas_h - scaled_h) // 2)
    image = _pad_image(image, canvas_h, canvas_w, pad_left, pad_top)

    out_boxes: List[RBox] = []
    for rb in rboxes:
        scaled = geom_transforms.scale(rb, scale_x=scale, scale_y=scale)
        out_boxes.append(geom_transforms.translate(scaled, pad_left, pad_top))

    content_size = (scaled_h, scaled_w) if (scaled_h, scaled_w) != (canvas_h, canvas_w) else None
    meta = SpatialPreprocessMeta(
        mode="pad",
        orig_size=(orig_w, orig_h),
        canvas_size=(canvas_h, canvas_w),
        scale=scale,
        scale_x=scale,
        scale_y=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        content_size=content_size,
    )
    return SpatialPreprocessResult(image=image, rboxes=out_boxes, meta=meta)


def apply_keep_ratio_mode(
    image: Image.Image,
    rboxes: Sequence[RBox],
    target_size: TargetSize,
) -> SpatialPreprocessResult:
    """Scale by large_edge (uniform); leave short edge unpadded (MMRotate RResize).

    ``pad_size_divisor`` is applied afterward in the collate / inference path
    (bottom-right pad), matching MMRotate ``Pad(size_divisor=…)``.
    """
    orig_w, orig_h = image.size
    ref_h, ref_w = parse_canvas_size("keep_ratio", target_size)
    large_edge = max(ref_h, ref_w)

    scale = large_edge / max(orig_h, orig_w, 1)
    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))

    if (scaled_w, scaled_h) != (orig_w, orig_h):
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)

    out_boxes = [geom_transforms.scale(rb, scale_x=scale, scale_y=scale) for rb in rboxes]
    meta = SpatialPreprocessMeta(
        mode="keep_ratio",
        orig_size=(orig_w, orig_h),
        canvas_size=(scaled_h, scaled_w),
        scale=scale,
        scale_x=scale,
        scale_y=scale,
        pad_left=0,
        pad_top=0,
        content_size=(scaled_h, scaled_w),
    )
    return SpatialPreprocessResult(image=image, rboxes=out_boxes, meta=meta)


def apply_crop_mode(
    image: Image.Image,
    rboxes: Sequence[RBox],
    target_size: TargetSize,
    *,
    random_crop: bool = True,
) -> SpatialPreprocessResult:
    """Crop or pad to target_size at native resolution (no scaling)."""
    orig_w, orig_h = image.size
    canvas_h, canvas_w = parse_canvas_size("crop", target_size)

    if orig_w >= canvas_w:
        max_x0 = orig_w - canvas_w
        crop_left = random.randint(0, max_x0) if random_crop and max_x0 > 0 else max_x0 // 2
    else:
        crop_left = 0

    if orig_h >= canvas_h:
        max_y0 = orig_h - canvas_h
        crop_top = random.randint(0, max_y0) if random_crop and max_y0 > 0 else max_y0 // 2
    else:
        crop_top = 0

    crop_right = min(crop_left + canvas_w, orig_w)
    crop_bottom = min(crop_top + canvas_h, orig_h)
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_w, crop_h = cropped.size

    pad_left = (canvas_w - crop_w) // 2 if crop_w < canvas_w else 0
    pad_top = (canvas_h - crop_h) // 2 if crop_h < canvas_h else 0
    out_image = _pad_image(cropped, canvas_h, canvas_w, pad_left, pad_top)

    out_boxes: List[RBox] = []
    for rb in rboxes:
        shifted = geom_transforms.translate(rb, -crop_left + pad_left, -crop_top + pad_top)
        out_boxes.append(shifted)

    meta = SpatialPreprocessMeta(
        mode="crop",
        orig_size=(orig_w, orig_h),
        canvas_size=(canvas_h, canvas_w),
        crop_left=crop_left,
        crop_top=crop_top,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return SpatialPreprocessResult(image=out_image, rboxes=out_boxes, meta=meta)


def apply_fixed_mode(
    image: Image.Image,
    rboxes: Sequence[RBox],
    target_size: TargetSize,
) -> SpatialPreprocessResult:
    """Stretch resize to exact (H, W); aspect ratio may change."""
    orig_w, orig_h = image.size
    canvas_h, canvas_w = parse_canvas_size("fixed", target_size)

    if (orig_w, orig_h) != (canvas_w, canvas_h):
        image = image.resize((canvas_w, canvas_h), Image.BILINEAR)

    scale_x = canvas_w / max(orig_w, 1)
    scale_y = canvas_h / max(orig_h, 1)
    out_boxes = [geom_transforms.scale(rb, scale_x=scale_x, scale_y=scale_y) for rb in rboxes]

    meta = SpatialPreprocessMeta(
        mode="fixed",
        orig_size=(orig_w, orig_h),
        canvas_size=(canvas_h, canvas_w),
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return SpatialPreprocessResult(image=image, rboxes=out_boxes, meta=meta)


def apply_spatial_preprocess(
    image: Image.Image,
    rboxes: Sequence[RBox],
    mode: str,
    target_size: TargetSize,
    *,
    random_crop: bool = True,
) -> SpatialPreprocessResult:
    """Apply configured spatial preprocessing."""
    m = (mode or "fixed").strip().lower()
    if m == "pad":
        return apply_pad_mode(image, rboxes, target_size)
    if m == "keep_ratio":
        return apply_keep_ratio_mode(image, rboxes, target_size)
    if m == "crop":
        return apply_crop_mode(image, rboxes, target_size, random_crop=random_crop)
    if m == "fixed":
        return apply_fixed_mode(image, rboxes, target_size)
    raise ValueError(
        f"Unknown resize_mode={mode!r}; expected 'fixed', 'pad', 'keep_ratio', or 'crop'."
    )


def remap_detections_to_original(
    detections: Sequence[dict],
    meta: SpatialPreprocessMeta,
) -> List[dict]:
    """Map model-space detections back to original image pixel coordinates."""
    out: List[dict] = []
    for d in detections:
        r = d["rbox"]
        if meta.mode in {"pad", "keep_ratio"}:
            cx = (r.cx - meta.pad_left) / max(meta.scale, 1e-8)
            cy = (r.cy - meta.pad_top) / max(meta.scale, 1e-8)
            rw = r.width / max(meta.scale, 1e-8)
            rh = r.height / max(meta.scale, 1e-8)
        elif meta.mode == "crop":
            cx = r.cx - meta.pad_left + meta.crop_left
            cy = r.cy - meta.pad_top + meta.crop_top
            rw, rh = r.width, r.height
        else:  # fixed
            cx = r.cx / max(meta.scale_x, 1e-8)
            cy = r.cy / max(meta.scale_y, 1e-8)
            rw = r.width / max(meta.scale_x, 1e-8)
            rh = r.height / max(meta.scale_y, 1e-8)
        out.append({
            **d,
            "rbox": RBox(cx, cy, rw, rh, r.angle),
        })
    return out


def build_spatial_meta_from_dims(
    mode: str,
    orig_width: int,
    orig_height: int,
    target_size: TargetSize,
) -> SpatialPreprocessMeta:
    """Infer metadata for inference remap without running the full transform."""
    m = (mode or "fixed").strip().lower()
    canvas_h, canvas_w = parse_canvas_size(m, target_size)
    orig_w, orig_h = orig_width, orig_height

    if m == "pad":
        large_edge = max(canvas_h, canvas_w)
        scale = large_edge / max(orig_h, orig_w, 1)
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))
        pad_left = max(0, (canvas_w - scaled_w) // 2)
        pad_top = max(0, (canvas_h - scaled_h) // 2)
        content_size = (scaled_h, scaled_w) if (scaled_h, scaled_w) != (canvas_h, canvas_w) else None
        return SpatialPreprocessMeta(
            mode="pad",
            orig_size=(orig_w, orig_h),
            canvas_size=(canvas_h, canvas_w),
            scale=scale,
            scale_x=scale,
            scale_y=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            content_size=content_size,
        )

    if m == "keep_ratio":
        large_edge = max(canvas_h, canvas_w)
        scale = large_edge / max(orig_h, orig_w, 1)
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))
        return SpatialPreprocessMeta(
            mode="keep_ratio",
            orig_size=(orig_w, orig_h),
            canvas_size=(scaled_h, scaled_w),
            scale=scale,
            scale_x=scale,
            scale_y=scale,
            pad_left=0,
            pad_top=0,
            content_size=(scaled_h, scaled_w),
        )

    if m == "crop":
        if orig_w >= canvas_w:
            crop_left = (orig_w - canvas_w) // 2
        else:
            crop_left = 0
        if orig_h >= canvas_h:
            crop_top = (orig_h - canvas_h) // 2
        else:
            crop_top = 0
        crop_w = min(canvas_w, orig_w)
        crop_h = min(canvas_h, orig_h)
        pad_left = (canvas_w - crop_w) // 2 if crop_w < canvas_w else 0
        pad_top = (canvas_h - crop_h) // 2 if crop_h < canvas_h else 0
        return SpatialPreprocessMeta(
            mode="crop",
            orig_size=(orig_w, orig_h),
            canvas_size=(canvas_h, canvas_w),
            crop_left=crop_left,
            crop_top=crop_top,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    scale_x = canvas_w / max(orig_w, 1)
    scale_y = canvas_h / max(orig_h, 1)
    return SpatialPreprocessMeta(
        mode="fixed",
        orig_size=(orig_w, orig_h),
        canvas_size=(canvas_h, canvas_w),
        scale_x=scale_x,
        scale_y=scale_y,
    )


__all__ = [
    "SpatialPreprocessMeta",
    "SpatialPreprocessResult",
    "TargetSize",
    "apply_crop_mode",
    "apply_fixed_mode",
    "apply_keep_ratio_mode",
    "apply_pad_mode",
    "apply_spatial_preprocess",
    "build_spatial_meta_from_dims",
    "get_model_canvas_size",
    "large_edge_from_target",
    "parse_canvas_size",
    "remap_detections_to_original",
]
