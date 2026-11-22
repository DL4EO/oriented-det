"""Cached mosaic for oriented boxes.

Train-only. Each tile is scaled so its long side matches the cell, with no
square pad, and pasted onto a gray canvas the way Ultralytics does. A following
affine scales and shifts that canvas down to ``target_size``. Boxes keep ``w``,
``h``, and ``θ`` up to that uniform scale. A box is dropped when less than
``min_visible`` of its area remains inside the canvas (default 0.2: more than
80% outside).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math
import random

from PIL import Image

from ..geometry import Polygon
from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox, normalize_le90
from ..ops.utils import polygon_intersection_area
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


def visible_fraction(rb: RBox, width: float, height: float) -> float:
    """Share of ``rb`` that lies inside the axis-aligned canvas ``[0, width) × [0, height)``."""
    area = float(rb.width) * float(rb.height)
    if area <= 0.0 or width <= 0.0 or height <= 0.0:
        return 0.0
    xs = [float(p[0]) for p in rb.corners()]
    ys = [float(p[1]) for p in rb.corners()]
    if max(xs) <= 0.0 or max(ys) <= 0.0 or min(xs) >= width or min(ys) >= height:
        return 0.0
    if min(xs) >= 0.0 and max(xs) <= width and min(ys) >= 0.0 and max(ys) <= height:
        return 1.0
    canvas = Polygon(
        [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    )
    return polygon_intersection_area(rb.to_polygon(), canvas) / area


def _is_axis_aligned(rb: RBox) -> bool:
    return abs(math.sin(2.0 * float(rb.angle))) < 1e-6


def clip_axis_aligned_box(rb: RBox, width: float, height: float) -> RBox:
    """Cut an axis-aligned box to ``[0, width] × [0, height]``.

    Angle 0 or ±90° only; the result is le90-normalized like a dataset box.
    """
    xs = [float(p[0]) for p in rb.corners()]
    ys = [float(p[1]) for p in rb.corners()]
    x1, x2 = max(min(xs), 0.0), min(max(xs), float(width))
    y1, y2 = max(min(ys), 0.0), min(max(ys), float(height))
    return normalize_le90(RBox((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, 0.0))


def filter_by_visible_fraction(
    rboxes: Sequence[RBox],
    labels: Optional[Sequence[int]],
    width: float,
    height: float,
    min_visible: float,
    clip_axis_aligned: bool = False,
) -> Tuple[List[RBox], List[int]]:
    """Drop boxes with less than ``min_visible`` of their area on the canvas.

    Width, height, and angle of a kept box are unchanged, unless
    ``clip_axis_aligned`` is set and the box is axis-aligned: then the kept
    box is the part inside the canvas (Ultralytics ``box_candidates``).
    Rotated boxes are never clipped.
    """
    if not 0.0 <= float(min_visible) <= 1.0:
        raise ValueError(f"min_visible must be in [0, 1], got {min_visible!r}")
    threshold = float(min_visible)
    kept_boxes: List[RBox] = []
    kept_labels: List[int] = []
    w = float(width)
    h = float(height)
    for i, rb in enumerate(rboxes):
        visible = visible_fraction(rb, w, h)
        if visible + 1e-6 >= threshold:
            if clip_axis_aligned and visible < 1.0 and _is_axis_aligned(rb):
                rb = clip_axis_aligned_box(rb, w, h)
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
    center_xy: Optional[Tuple[int, int]] = None,
    rotate_prob: float = 0.0,
    rotate_angle_range: float = 180.0,
    rotate_rect_label_ids: Optional[Sequence[int]] = None,
    min_visible: float = 0.2,
    clip_boxes: bool = False,
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

    all_real, all_labels = filter_by_visible_fraction(
        all_real, all_labels, canvas_w, canvas_h, min_visible, clip_boxes
    )
    all_ign, all_ign_labels = filter_by_visible_fraction(
        all_ign, all_ign_labels, canvas_w, canvas_h, min_visible, clip_boxes
    )
    all_look, _ = filter_by_visible_fraction(
        all_look, None, canvas_w, canvas_h, min_visible, clip_boxes
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
    center_xy: Optional[Tuple[int, int]] = None,
    rotate_prob: float = 0.0,
    rotate_angle_range: float = 180.0,
    rotate_rect_label_ids: Optional[Sequence[int]] = None,
    min_visible: float = 0.2,
    clip_boxes: bool = False,
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
        center_xy=center_xy,
        rotate_prob=rotate_prob,
        rotate_angle_range=rotate_angle_range,
        rotate_rect_label_ids=rotate_rect_label_ids,
        min_visible=min_visible,
        clip_boxes=clip_boxes,
    )


def resize_long_side(
    image: Image.Image,
    rboxes: Sequence[RBox],
    long_side: int,
) -> Tuple[Image.Image, List[RBox]]:
    """Scale so the longer side equals ``long_side``. No padding.

    One scale is applied to both axes, then each side is rounded. A square pad
    is not added: the short side stays short so mosaic can paste the rectangle.
    """
    side = max(int(long_side), 1)
    orig_w, orig_h = image.size
    scale = float(side) / max(orig_h, orig_w, 1)
    new_w = min(side, max(1, int(round(orig_w * scale))))
    new_h = min(side, max(1, int(round(orig_h * scale))))
    sx = new_w / max(orig_w, 1)
    sy = new_h / max(orig_h, 1)
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.BILINEAR)
    return image, [geom_transforms.scale(rb, sx, sy) for rb in rboxes]


def center_pad(
    image: Image.Image,
    rboxes: Sequence[RBox],
    canvas_w: int,
    canvas_h: int,
    pad_val: int,
) -> Tuple[Image.Image, List[RBox]]:
    """Center the image on a ``canvas_w`` × ``canvas_h`` canvas. Fill is ``pad_val``."""
    width, height = image.size
    pad_left = max(0, (int(canvas_w) - width) // 2)
    pad_top = max(0, (int(canvas_h) - height) // 2)
    if pad_left == 0 and pad_top == 0 and (width, height) == (int(canvas_w), int(canvas_h)):
        return image, list(rboxes)
    fill = int(pad_val)
    canvas = Image.new("RGB", (int(canvas_w), int(canvas_h)), (fill, fill, fill))
    canvas.paste(image, (pad_left, pad_top))
    moved = [geom_transforms.translate(rb, float(pad_left), float(pad_top)) for rb in rboxes]
    return canvas, moved


def sample_mosaic_affine(
    out_w: int,
    out_h: int,
    scale_gain: float,
    translate_gain: float,
) -> Tuple[float, float, float]:
    """Return ``(scale, tx, ty)`` for the Ultralytics mosaic affine.

    ``scale_gain`` 0.5 draws a uniform factor in [0.5, 1.5]. ``tx`` and ``ty``
    are output-pixel translations. A translate gain of 0.1 keeps them inside
    40%–60% of the output side, which is the center when the gain is 0.
    """
    scale = 1.0
    if float(scale_gain) > 0.0:
        low = max(1e-3, 1.0 - float(scale_gain))
        scale = random.uniform(low, 1.0 + float(scale_gain))
    tx = float(out_w) / 2.0
    ty = float(out_h) / 2.0
    if float(translate_gain) > 0.0:
        tx = random.uniform(0.5 - float(translate_gain), 0.5 + float(translate_gain)) * float(out_w)
        ty = random.uniform(0.5 - float(translate_gain), 0.5 + float(translate_gain)) * float(out_h)
    return scale, tx, ty


def apply_mosaic_affine(
    sample: MosaicSample,
    target_size: Tuple[int, int],
    scale: float,
    tx: float,
    ty: float,
    *,
    pad_val: int = 114,
    min_visible: float = 0.2,
    clip_boxes: bool = False,
) -> MosaicSample:
    """Warp a mosaic canvas to ``target_size`` (H, W).

    The forward map matches Ultralytics ``RandomPerspective`` with no rotation,
    shear, or perspective: ``x_out = scale * (x_in - W/2) + tx``. ``scale`` 1
    and ``tx, ty`` at the output center is the center window of the mosaic.
    New pixels are ``pad_val``.
    """
    target_h, target_w = int(target_size[0]), int(target_size[1])
    width, height = sample.image.size
    factor = max(float(scale), 1e-3)
    coeff = (
        1.0 / factor,
        0.0,
        width / 2.0 - float(tx) / factor,
        0.0,
        1.0 / factor,
        height / 2.0 - float(ty) / factor,
    )
    fill = int(pad_val)
    image = sample.image.transform(
        (target_w, target_h),
        Image.AFFINE,
        coeff,
        resample=Image.BILINEAR,
        fillcolor=(fill, fill, fill),
    )

    def _map(boxes: Sequence[RBox], labels: Optional[Sequence[int]]):
        mapped = [
            RBox(
                factor * (rb.cx - width / 2.0) + float(tx),
                factor * (rb.cy - height / 2.0) + float(ty),
                rb.width * factor,
                rb.height * factor,
                rb.angle,
            )
            for rb in boxes
        ]
        return filter_by_visible_fraction(
            mapped, labels, float(target_w), float(target_h), min_visible, clip_boxes
        )

    rboxes, labels = _map(sample.rboxes, sample.labels)
    rboxes_ignore, labels_ignore = _map(sample.rboxes_ignore, sample.labels_ignore)
    rboxes_lookalike, _ = _map(sample.rboxes_lookalike, None)
    return MosaicSample(
        image=image,
        rboxes=rboxes,
        labels=labels,
        rboxes_ignore=rboxes_ignore,
        labels_ignore=labels_ignore,
        rboxes_lookalike=rboxes_lookalike,
    )


def affine_mosaic_to_canvas(
    sample: MosaicSample,
    target_size: Tuple[int, int],
    *,
    scale_gain: float,
    translate_gain: float,
    pad_val: int = 114,
    min_visible: float = 0.2,
    clip_boxes: bool = False,
) -> MosaicSample:
    """Sample one Ultralytics-style affine and apply it."""
    target_h, target_w = int(target_size[0]), int(target_size[1])
    scale, tx, ty = sample_mosaic_affine(target_w, target_h, scale_gain, translate_gain)
    return apply_mosaic_affine(
        sample,
        (target_h, target_w),
        scale,
        tx,
        ty,
        pad_val=pad_val,
        min_visible=min_visible,
        clip_boxes=clip_boxes,
    )


def apply_mosaic_resize_crop(
    sample: MosaicSample,
    target_size: Tuple[int, int],
    *,
    ratio_range: Tuple[float, float] = (0.5, 1.5),
    random_crop: bool = True,
    min_visible: float = 0.2,
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

    rboxes, labels = filter_by_visible_fraction(
        rboxes, sample.labels, canvas_w, canvas_h, min_visible
    )
    rboxes_ignore, labels_ignore = filter_by_visible_fraction(
        rboxes_ignore, sample.labels_ignore, canvas_w, canvas_h, min_visible
    )
    rboxes_lookalike, _ = filter_by_visible_fraction(
        rboxes_lookalike, None, canvas_w, canvas_h, min_visible
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
    "affine_mosaic_to_canvas",
    "apply_cached_mosaic",
    "apply_mosaic",
    "apply_mosaic_affine",
    "apply_mosaic_resize_crop",
    "center_pad",
    "filter_by_visible_fraction",
    "resize_long_side",
    "sample_mosaic_affine",
    "visible_fraction",
    "mosaic_combine",
    "resolve_mosaic_disable_last_n",
]
