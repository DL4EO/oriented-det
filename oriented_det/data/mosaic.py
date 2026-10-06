"""Cached mosaic for oriented boxes (MMRotate 1.x / mmdet CachedMosaic).

Train-only. Four images are keep-ratio fitted into cells on a 2× canvas, then
(typically) random-resized and cropped back to ``target_size``. Boxes keep ``w``,
``h``, and ``θ``; a box is kept when its center lies inside the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math
import random

from PIL import Image

from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox
from .preprocessing import apply_crop_mode, parse_canvas_size
from .rotates import apply_rotate_to_image, apply_rotate_to_rboxes, sample_rotate_degrees

Loc = Tuple[str, ...]
_LOCS: Loc = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass
class MosaicSample:
    """One image plus box groups used by mosaic / cache."""

    image: Image.Image
    rboxes: List[RBox]
    labels: List[int]
    rboxes_ignore: List[RBox] = field(default_factory=list)
    labels_ignore: List[int] = field(default_factory=list)
    rboxes_lookalike: List[RBox] = field(default_factory=list)


class MosaicCache:
    """Worker-local results cache (mmdet CachedMosaic)."""

    def __init__(self, max_cached_images: int = 40, random_pop: bool = True):
        if max_cached_images < 5:
            raise ValueError("max_cached_images must be >= 5 (need 4 mix sources).")
        self.max_cached_images = int(max_cached_images)
        self.random_pop = bool(random_pop)
        self._items: List[MosaicSample] = []

    def __len__(self) -> int:
        return len(self._items)

    def ready(self) -> bool:
        return len(self._items) > 4

    def push(self, sample: MosaicSample) -> None:
        self._items.append(sample)
        if len(self._items) > self.max_cached_images:
            index = random.randint(0, len(self._items) - 1) if self.random_pop else 0
            self._items.pop(index)

    def sample(self, n: int = 3) -> List[MosaicSample]:
        last = len(self._items) - 1
        return [self._items[random.randint(0, last)] for _ in range(n)]


def resolve_mosaic_disable_last_n(num_epochs: int, override: Optional[int] = None) -> int:
    """Last-N epochs with mosaic off. ``None`` → 10% of the schedule (at least 1). ``0`` keeps mosaic on."""
    if override is not None:
        return max(0, int(override))
    return max(1, int(round(0.1 * int(num_epochs))))


def mosaic_combine(
    loc: str,
    center_xy: Tuple[int, int],
    img_wh: Tuple[int, int],
    canvas_wh: Tuple[int, int],
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Paste and crop rectangles for one mosaic quadrant (mmdet ``_mosaic_combine``).

    Args:
        loc: ``top_left`` / ``top_right`` / ``bottom_left`` / ``bottom_right``.
        center_xy: Mosaic center (x, y) on the 2× canvas.
        img_wh: Fitted patch size (width, height).
        canvas_wh: Full mosaic size (width, height).

    Returns:
        ``(paste_xyxy, crop_xyxy)`` in integer pixel coords.
    """
    if loc not in _LOCS:
        raise ValueError(f"Unknown mosaic loc: {loc!r}")
    cx, cy = int(center_xy[0]), int(center_xy[1])
    w, h = int(img_wh[0]), int(img_wh[1])
    canvas_w, canvas_h = int(canvas_wh[0]), int(canvas_wh[1])
    if loc == "top_left":
        x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
        crop = (w - (x2 - x1), h - (y2 - y1), w, h)
    elif loc == "top_right":
        x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w, canvas_w), cy
        crop = (0, h - (y2 - y1), min(w, x2 - x1), h)
    elif loc == "bottom_left":
        x1, y1, x2, y2 = max(cx - w, 0), cy, cx, min(cy + h, canvas_h)
        crop = (w - (x2 - x1), 0, w, min(y2 - y1, h))
    else:
        x1, y1, x2, y2 = cx, cy, min(cx + w, canvas_w), min(cy + h, canvas_h)
        crop = (0, 0, min(w, x2 - x1), min(y2 - y1, h))
    return (x1, y1, x2, y2), crop


def clip_and_filter_centers(
    rboxes: Sequence[RBox],
    labels: Optional[Sequence[int]],
    width: float,
    height: float,
    *,
    clip_border: bool,
) -> Tuple[List[RBox], List[int]]:
    """Clamp centers onto the canvas when ``clip_border``, then keep in-bounds centers."""
    kept_boxes: List[RBox] = []
    kept_labels: List[int] = []
    w = float(width)
    h = float(height)
    for i, rb in enumerate(rboxes):
        cx, cy = rb.cx, rb.cy
        if clip_border:
            cx = min(max(cx, 0.0), w)
            cy = min(max(cy, 0.0), h)
            rb = RBox(cx, cy, rb.width, rb.height, rb.angle)
        if 0.0 <= rb.cx < w and 0.0 <= rb.cy < h:
            kept_boxes.append(rb)
            if labels is not None:
                kept_labels.append(int(labels[i]))
    return kept_boxes, kept_labels


def _map_boxes(rboxes: Sequence[RBox], scale: float, padw: float, padh: float) -> List[RBox]:
    out: List[RBox] = []
    for rb in rboxes:
        scaled = geom_transforms.scale(rb, scale_x=scale, scale_y=scale)
        out.append(geom_transforms.translate(scaled, padw, padh))
    return out


def _paste_patch(
    canvas: Image.Image,
    patch: Image.Image,
    loc: str,
    center_xy: Tuple[int, int],
    cell_hw: Tuple[int, int],
) -> Tuple[float, float, float]:
    """Fit ``patch`` into one cell, paste it, return ``(scale, padw, padh)``."""
    orig_w, orig_h = patch.size
    cell_h, cell_w = int(cell_hw[0]), int(cell_hw[1])
    scale = min(cell_h / max(orig_h, 1), cell_w / max(orig_w, 1))
    fitted_w = max(1, int(orig_w * scale))
    fitted_h = max(1, int(orig_h * scale))
    if (fitted_w, fitted_h) != (orig_w, orig_h):
        patch = patch.resize((fitted_w, fitted_h), Image.BILINEAR)
    canvas_w, canvas_h = canvas.size
    paste, crop = mosaic_combine(loc, center_xy, (fitted_w, fitted_h), (canvas_w, canvas_h))
    x1_p, y1_p, x2_p, y2_p = paste
    x1_c, y1_c, x2_c, y2_c = crop
    if x2_p > x1_p and y2_p > y1_p and x2_c > x1_c and y2_c > y1_c:
        cropped = patch.crop((x1_c, y1_c, x2_c, y2_c))
        canvas.paste(cropped, (x1_p, y1_p))
    return scale, float(x1_p - x1_c), float(y1_p - y1_c)


def _rotate_sample_copy(
    sample: MosaicSample,
    *,
    prob: float,
    angle_range_deg: float,
    pad_val: int,
    rect_label_ids: Optional[Sequence[int]] = None,
) -> MosaicSample:
    """Independent copy of ``sample`` rotated in place. The cache object is not mutated."""
    degrees = sample_rotate_degrees(
        prob=prob,
        angle_range_deg=angle_range_deg,
        labels=list(sample.labels) + list(sample.labels_ignore),
        rect_label_ids=rect_label_ids,
    )
    if degrees is None:
        return sample
    width, height = sample.image.size
    fill = int(pad_val)
    image = apply_rotate_to_image(sample.image, degrees, fillcolor=(fill, fill, fill))
    radians = math.radians(degrees)

    def _rot(boxes: Sequence[RBox]) -> List[RBox]:
        return apply_rotate_to_rboxes(
            boxes,
            radians,
            image_width=float(width),
            image_height=float(height),
        )

    return MosaicSample(
        image=image,
        rboxes=_rot(sample.rboxes),
        labels=list(sample.labels),
        rboxes_ignore=_rot(sample.rboxes_ignore),
        labels_ignore=list(sample.labels_ignore),
        rboxes_lookalike=_rot(sample.rboxes_lookalike),
    )


def apply_mosaic(
    current: MosaicSample,
    mix: Sequence[MosaicSample],
    *,
    img_scale: Tuple[int, int],
    center_ratio_range: Tuple[float, float] = (0.5, 1.5),
    pad_val: int = 114,
    bbox_clip_border: bool = True,
    center_xy: Optional[Tuple[int, int]] = None,
    rotate_prob: float = 0.0,
    rotate_angle_range: float = 180.0,
    rotate_rect_label_ids: Optional[Sequence[int]] = None,
) -> MosaicSample:
    """Compose four samples onto a 2× canvas. ``img_scale`` is (H, W) of one cell."""
    if len(mix) < 3:
        raise ValueError(f"Mosaic needs 3 mix images, got {len(mix)}")
    cell_h, cell_w = parse_canvas_size("fixed", img_scale)
    canvas_w, canvas_h = cell_w * 2, cell_h * 2
    fill = int(pad_val)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (fill, fill, fill))
    if center_xy is None:
        cx = int(random.uniform(*center_ratio_range) * cell_w)
        cy = int(random.uniform(*center_ratio_range) * cell_h)
        center_xy = (cx, cy)

    sources = (current, mix[0], mix[1], mix[2])
    if float(rotate_prob) > 0.0:
        sources = tuple(
            _rotate_sample_copy(
                sample,
                prob=float(rotate_prob),
                angle_range_deg=float(rotate_angle_range),
                pad_val=int(pad_val),
                rect_label_ids=rotate_rect_label_ids,
            )
            for sample in sources
        )
    patches = sources
    all_real: List[RBox] = []
    all_labels: List[int] = []
    all_ign: List[RBox] = []
    all_ign_labels: List[int] = []
    all_look: List[RBox] = []
    for loc, sample in zip(_LOCS, patches):
        scale, padw, padh = _paste_patch(
            canvas, sample.image, loc, center_xy, (cell_h, cell_w)
        )
        all_real.extend(_map_boxes(sample.rboxes, scale, padw, padh))
        all_labels.extend(sample.labels)
        all_ign.extend(_map_boxes(sample.rboxes_ignore, scale, padw, padh))
        all_ign_labels.extend(sample.labels_ignore)
        all_look.extend(_map_boxes(sample.rboxes_lookalike, scale, padw, padh))

    all_real, all_labels = clip_and_filter_centers(
        all_real, all_labels, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    all_ign, all_ign_labels = clip_and_filter_centers(
        all_ign, all_ign_labels, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    all_look, _ = clip_and_filter_centers(
        all_look, None, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    return MosaicSample(
        image=canvas,
        rboxes=all_real,
        labels=all_labels,
        rboxes_ignore=all_ign,
        labels_ignore=all_ign_labels,
        rboxes_lookalike=all_look,
    )


def apply_cached_mosaic(
    current: MosaicSample,
    cache: MosaicCache,
    *,
    img_scale: Tuple[int, int],
    prob: float = 1.0,
    center_ratio_range: Tuple[float, float] = (0.5, 1.5),
    pad_val: int = 114,
    bbox_clip_border: bool = True,
    center_xy: Optional[Tuple[int, int]] = None,
    rotate_prob: float = 0.0,
    rotate_angle_range: float = 180.0,
    rotate_rect_label_ids: Optional[Sequence[int]] = None,
) -> MosaicSample:
    """Push ``current`` into ``cache``. Mosaic when the cache is warm and ``prob`` hits."""
    cache.push(current)
    if not cache.ready() or random.random() >= float(prob):
        return current
    mix = cache.sample(3)
    return apply_mosaic(
        current,
        mix,
        img_scale=img_scale,
        center_ratio_range=center_ratio_range,
        pad_val=pad_val,
        bbox_clip_border=bbox_clip_border,
        center_xy=center_xy,
        rotate_prob=rotate_prob,
        rotate_angle_range=rotate_angle_range,
        rotate_rect_label_ids=rotate_rect_label_ids,
    )


def apply_mosaic_resize_crop(
    sample: MosaicSample,
    target_size: Tuple[int, int],
    *,
    ratio_range: Tuple[float, float] = (0.5, 1.5),
    random_crop: bool = True,
    bbox_clip_border: bool = True,
) -> MosaicSample:
    """RandomResize then crop/pad to ``target_size`` (H, W).

    ``ratio`` multiplies the mosaic canvas (2× cell), matching MMRotate
    ``RandomResize(scale=(2H, 2W), ratio_range=...)``. Cropping to
    ``target_size`` is a window, not an extra 0.5×: ratio 1.0 keeps native
    tile object scale; ratio 0.5 fits the whole mosaic into 1024.
    """
    orig_w, orig_h = sample.image.size
    ratio = random.uniform(float(ratio_range[0]), float(ratio_range[1]))
    scale_h = max(1, int(orig_h * ratio))
    scale_w = max(1, int(orig_w * ratio))
    fit = min(scale_h / max(orig_h, 1), scale_w / max(orig_w, 1))
    new_w = max(1, int(orig_w * fit))
    new_h = max(1, int(orig_h * fit))
    image = sample.image
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.BILINEAR)
    sx = new_w / max(orig_w, 1)
    sy = new_h / max(orig_h, 1)

    def _scale_list(boxes: Sequence[RBox]) -> List[RBox]:
        return [geom_transforms.scale(rb, scale_x=sx, scale_y=sy) for rb in boxes]

    rboxes = _scale_list(sample.rboxes)
    rboxes_ignore = _scale_list(sample.rboxes_ignore)
    rboxes_lookalike = _scale_list(sample.rboxes_lookalike)

    tagged = rboxes + rboxes_ignore + rboxes_lookalike
    n_real = len(rboxes)
    n_ign = len(rboxes_ignore)
    cropped = apply_crop_mode(image, tagged, target_size, random_crop=random_crop)
    canvas_h, canvas_w = cropped.meta.canvas_size
    split = cropped.rboxes
    rboxes = split[:n_real]
    rboxes_ignore = split[n_real : n_real + n_ign]
    rboxes_lookalike = split[n_real + n_ign :]

    rboxes, labels = clip_and_filter_centers(
        rboxes, sample.labels, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    rboxes_ignore, labels_ignore = clip_and_filter_centers(
        rboxes_ignore, sample.labels_ignore, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    rboxes_lookalike, _ = clip_and_filter_centers(
        rboxes_lookalike, None, canvas_w, canvas_h, clip_border=bbox_clip_border
    )
    return MosaicSample(
        image=cropped.image,
        rboxes=rboxes,
        labels=labels,
        rboxes_ignore=rboxes_ignore,
        labels_ignore=labels_ignore,
        rboxes_lookalike=rboxes_lookalike,
    )


__all__ = [
    "MosaicCache",
    "MosaicSample",
    "apply_cached_mosaic",
    "apply_mosaic",
    "apply_mosaic_resize_crop",
    "clip_and_filter_centers",
    "mosaic_combine",
    "resolve_mosaic_disable_last_n",
]
