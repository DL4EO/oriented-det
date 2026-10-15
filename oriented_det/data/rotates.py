"""Training-time random rotate for oriented boxes (MMRotate PolyRandomRotate parity)."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox, normalize_le90

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def apply_rotate_to_image(image, degrees: float, fillcolor=(0, 0, 0)):
    """Rotate a PIL image visually counter-clockwise around its center.

    Canvas size is unchanged (``expand=False``). Corners use ``fillcolor``
    (black for a lone tile; mosaic uses the pad value). MMRotate
    ``PolyRandomRotate(auto_bound=False)``.
    """
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for image rotation.")
    return image.rotate(
        degrees,
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=fillcolor,
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


_RIGHT_ANGLE_DEGREES = (90.0, 180.0, -90.0, -180.0)


def sample_rotate_degrees(
    *,
    prob: float,
    angle_range_deg: float,
    labels: Optional[Sequence[int]] = None,
    rect_label_ids: Optional[Sequence[int]] = None,
) -> Optional[float]:
    """Sample one rotate angle, or ``None`` when the rotate is skipped.

    Tiles that contain a ``rect_label_ids`` class use only 90°, 180°, −90°,
    or −180°. Other tiles draw uniformly from ``[-angle_range_deg, +angle_range_deg]``.
    """
    if float(prob) <= 0.0:
        return None
    if random.random() >= float(prob):
        return None
    rect_ids = {int(label_id) for label_id in (rect_label_ids or [])}
    if rect_ids and labels is not None and any(int(label) in rect_ids for label in labels):
        return float(random.choice(_RIGHT_ANGLE_DEGREES))
    return float(random.uniform(-float(angle_range_deg), float(angle_range_deg)))


def apply_random_train_rotate(
    image,
    rboxes: Sequence[RBox],
    *,
    image_width: float,
    image_height: float,
    prob: float = 0.5,
    angle_range_deg: float = 180.0,
    labels: Optional[Sequence[int]] = None,
    rect_label_ids: Optional[Sequence[int]] = None,
    fillcolor=(0, 0, 0),
) -> Tuple[any, List[RBox]]:
    """Apply MMRotate-style ``PolyRandomRotate`` (range mode, ``auto_bound=False``).

    With probability ``prob``, sample an angle and rotate image + boxes.
    ``rect_label_ids`` forces a right-angle tile rotate when one of those
    class ids is present.
    """
    rboxes_list = list(rboxes)
    degrees = sample_rotate_degrees(
        prob=prob,
        angle_range_deg=angle_range_deg,
        labels=labels,
        rect_label_ids=rect_label_ids,
    )
    if degrees is None or degrees == 0.0:
        return image, rboxes_list
    radians = math.radians(degrees)
    return (
        apply_rotate_to_image(image, degrees, fillcolor=fillcolor),
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
    "sample_rotate_degrees",
]
