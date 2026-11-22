"""Train-only scale about the canvas center, then a shift, then crop or pad back.

Boxes move with ``oriented_det.geometry.transforms`` and are dropped with the
mosaic visibility filter when less than ``min_visible`` of the area remains.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from oriented_det.data.mosaic import filter_by_visible_fraction
from oriented_det.geometry.rbox import RBox
from oriented_det.geometry import transforms as geom


def sample_scale_translate(
    width: int,
    height: int,
    scale_gain: float,
    translate_gain: float,
) -> Tuple[float, float, float]:
    """Return ``(scale, dx, dy)``.

    ``scale_gain`` 0.5 draws a uniform factor in [0.5, 1.5]. ``translate_gain``
    0.1 shifts by up to 10% of the canvas on each axis. A gain of 0 is identity
    on that term.
    """
    scale = 1.0
    if float(scale_gain) > 0.0:
        low = max(1e-3, 1.0 - float(scale_gain))
        scale = random.uniform(low, 1.0 + float(scale_gain))
    dx = 0.0
    dy = 0.0
    if float(translate_gain) > 0.0:
        dx = random.uniform(-float(translate_gain), float(translate_gain)) * float(width)
        dy = random.uniform(-float(translate_gain), float(translate_gain)) * float(height)
    return scale, dx, dy


def pixel_scales(width: int, height: int, scale: float) -> Tuple[float, float]:
    """Scale factors after rounding the resized canvas to whole pixels."""
    new_w = max(1, int(round(float(width) * float(scale))))
    new_h = max(1, int(round(float(height) * float(scale))))
    return new_w / float(width), new_h / float(height)


def warp_image(
    image: Image.Image,
    scale: float,
    dx: float,
    dy: float,
    pad_val: int = 114,
) -> Image.Image:
    """Scale about the center, shift by ``(dx, dy)``, and fill back to the same size."""
    width, height = image.size
    sx, sy = pixel_scales(width, height, scale)
    ox = width / 2.0
    oy = height / 2.0
    coeff = (
        1.0 / sx,
        0.0,
        ox - (ox + float(dx)) / sx,
        0.0,
        1.0 / sy,
        oy - (oy + float(dy)) / sy,
    )
    fill = int(pad_val)
    fillcolor = (fill, fill, fill) if image.mode == "RGB" else fill
    return image.transform(
        (width, height),
        Image.AFFINE,
        coeff,
        resample=Image.BILINEAR,
        fillcolor=fillcolor,
    )


def warp_boxes(
    rboxes: Sequence[RBox],
    width: int,
    height: int,
    scale: float,
    dx: float,
    dy: float,
    min_visible: float,
    labels: Optional[Sequence[int]] = None,
    clip_axis_aligned: bool = False,
) -> Tuple[List[RBox], Optional[List[int]]]:
    """Move boxes with the same center-scale and shift as ``warp_image``, then drop."""
    sx, sy = pixel_scales(width, height, scale)
    ox = width / 2.0
    oy = height / 2.0
    mapped: List[RBox] = []
    for rb in rboxes:
        centered = geom.translate(rb, -ox, -oy)
        scaled = geom.scale(centered, sx, sy)
        mapped.append(geom.translate(scaled, ox + float(dx), oy + float(dy)))
    kept, kept_labels = filter_by_visible_fraction(
        mapped, labels, float(width), float(height), float(min_visible), clip_axis_aligned
    )
    if labels is None:
        return kept, None
    return kept, kept_labels
