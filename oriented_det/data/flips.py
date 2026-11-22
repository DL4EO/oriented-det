"""Training-time random flips for oriented boxes (MMRotate RRandomFlip parity)."""

from __future__ import annotations

import random
from typing import List, Literal, Sequence, Tuple

from ..geometry import transforms as geom_transforms
from ..geometry.rbox import RBox, normalize_le90

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

FlipMode = Literal["horizontal", "vertical", "diagonal"]


def apply_flip_to_image(image, mode: FlipMode):
    """Apply a single flip mode to a PIL image."""
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for image flips.")
    if mode == "horizontal":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if mode == "vertical":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if mode == "diagonal":
        # MMRotate/mmcv diagonal: flip both axes (point reflection through center).
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
    raise ValueError(f"Unknown flip mode: {mode!r}")


def apply_flip_to_rboxes(
    rboxes: Sequence[RBox],
    mode: FlipMode,
    *,
    image_width: float,
    image_height: float,
) -> List[RBox]:
    """Apply flip to RBoxes and re-normalize to le90."""
    out: List[RBox] = []
    for rb in rboxes:
        if mode == "horizontal":
            flipped = geom_transforms.flip_horizontal(rb, image_width)
        elif mode == "vertical":
            flipped = geom_transforms.flip_vertical(rb, image_height)
        elif mode == "diagonal":
            flipped = geom_transforms.flip_diagonal(rb, image_width, image_height)
        else:
            raise ValueError(f"Unknown flip mode: {mode!r}")
        out.append(normalize_le90(flipped))
    return out


def apply_random_train_flips(
    image,
    rboxes: Sequence[RBox],
    *,
    image_width: float,
    image_height: float,
    enable_horizontal: bool = True,
    enable_vertical: bool = True,
    enable_diagonal: bool = False,
    flip_prob: float = 0.5,
) -> Tuple[any, List[RBox]]:
    """Apply each enabled flip independently with probability ``flip_prob``.

    With horizontal and vertical enabled and ``flip_prob=0.5``, the four outcomes
    (none, horizontal, vertical, both) each occur 25% of the time. Applying both
    is a point reflection, the same geometry as one diagonal flip. Diagonal, when
    enabled, is a third independent trial at the same probability.
    """
    modes: List[FlipMode] = []
    if enable_horizontal and random.random() < flip_prob:
        modes.append("horizontal")
    if enable_vertical and random.random() < flip_prob:
        modes.append("vertical")
    if enable_diagonal and random.random() < flip_prob:
        modes.append("diagonal")
    if not modes:
        return image, list(rboxes)

    out_image = image
    out_boxes = list(rboxes)
    for mode in modes:
        out_image = apply_flip_to_image(out_image, mode)
        out_boxes = apply_flip_to_rboxes(
            out_boxes, mode, image_width=image_width, image_height=image_height
        )
    return out_image, out_boxes


__all__ = [
    "FlipMode",
    "apply_flip_to_image",
    "apply_flip_to_rboxes",
    "apply_random_train_flips",
]
