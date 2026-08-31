"""Training-time random rotate for oriented boxes (MMRotate PolyRandomRotate parity)."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox, normalize_le90

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def apply_rotate_to_image(image, degrees: float):
    """Rotate a PIL image visually counter-clockwise around its center.

    Canvas size is unchanged (``expand=False``); corners fill black — MMRotate
    ``PolyRandomRotate(auto_bound=False)``.
    """
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for image rotation.")
    return image.rotate(
        degrees,
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0),
    )


def apply_rotate_to_rboxes(
    rboxes: Sequence[RBox],
    radians: float,
    *,
    image_width: float,
    image_height: float,
) -> List[RBox]:
    """Rotate RBoxes to match ``PIL.Image.rotate(+degrees)`` and re-normalize to le90.

    ``geometry.transforms.rotate`` uses math/y-up CCW. Image coordinates are
    y-down, so a visual CCW image rotate of ``+radians`` is ``rotate(-radians)``.
    """
    origin_x = image_width / 2.0
    origin_y = image_height / 2.0
    out: List[RBox] = []
    for rb in rboxes:
        rotated = geom_transforms.rotate(
            rb, -radians, origin_x=origin_x, origin_y=origin_y
        )
        out.append(normalize_le90(rotated))
    return out


def apply_random_train_rotate(
    image,
    rboxes: Sequence[RBox],
    *,
    image_width: float,
    image_height: float,
    prob: float = 0.5,
    angle_range_deg: float = 180.0,
) -> Tuple[any, List[RBox]]:
    """Apply MMRotate-style ``PolyRandomRotate`` (range mode, ``auto_bound=False``).

    With probability ``prob``, sample a uniform angle in
    ``[-angle_range_deg, +angle_range_deg]`` and rotate image + boxes.
    """
    rboxes_list = list(rboxes)
    if prob <= 0.0:
        return image, rboxes_list
    if random.random() >= prob:
        return image, rboxes_list
    degrees = random.uniform(-angle_range_deg, angle_range_deg)
    if degrees == 0.0:
        return image, rboxes_list
    radians = math.radians(degrees)
    return (
        apply_rotate_to_image(image, degrees),
        apply_rotate_to_rboxes(
            rboxes_list,
            radians,
            image_width=image_width,
            image_height=image_height,
        ),
    )


__all__ = [
    "apply_random_train_rotate",
    "apply_rotate_to_image",
    "apply_rotate_to_rboxes",
]
