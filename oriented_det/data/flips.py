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
) -> Tuple[any, List[RBox]]:
    """Pick at most one random flip (MMRotate DOTA-style 25% per enabled mode).

    When horizontal, vertical, and diagonal are all enabled, each has probability
  0.25 and there is a 0.25 chance of no flip — matching
  ``RRandomFlip(flip_ratio=[0.25, 0.25, 0.25],
  direction=['horizontal', 'vertical', 'diagonal'])``.

    With only horizontal and vertical enabled (no diagonal), uses 0.5 per enabled
  mode and no no-flip bucket (legacy two-flip behavior).
    """
    modes: List[FlipMode] = []
    if enable_horizontal:
        modes.append("horizontal")
    if enable_vertical:
        modes.append("vertical")
    if enable_diagonal:
        modes.append("diagonal")
    if not modes:
        return image, list(rboxes)

    rboxes_list = list(rboxes)
    if len(modes) == 1:
        if random.random() >= 0.5:
            return image, rboxes_list
        mode = modes[0]
    elif enable_diagonal and enable_horizontal and enable_vertical:
        # MMRotate: three flip types + implicit no-flip (25% each).
        bucket = random.random()
        if bucket >= 0.75:
            return image, rboxes_list
        mode = modes[int(bucket / 0.25)]
    else:
        # Two modes only (e.g. H+V without diagonal): 50% each.
        mode = modes[0] if random.random() < 0.5 else modes[1]

    return (
        apply_flip_to_image(image, mode),
        apply_flip_to_rboxes(
            rboxes_list, mode, image_width=image_width, image_height=image_height
        ),
    )


__all__ = [
    "FlipMode",
    "apply_flip_to_image",
    "apply_flip_to_rboxes",
    "apply_random_train_flips",
]
