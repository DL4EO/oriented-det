"""Visualization helpers for geometric primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import itertools
import math
import random

try:  # Optional dependency; only required when drawing actual images.
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - exercised when pillow is not installed.
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

from ..geometry import Polygon, QBox, RBox

Point = Tuple[float, float]
Color = Tuple[int, int, int]

DEFAULT_PALETTE = [
    (253, 128, 93),
    (44, 160, 44),
    (31, 119, 180),
    (148, 103, 189),
    (214, 39, 40),
    (140, 86, 75),
    (188, 189, 34),
    (23, 190, 207),
]


def as_polygon(obj) -> Polygon:
    if isinstance(obj, Polygon):
        return obj
    if isinstance(obj, QBox):
        return obj.to_polygon()
    if isinstance(obj, RBox):
        return obj.to_polygon()
    return Polygon(obj)


def cycle_palette(n: int, *, palette: Sequence[Color] | None = None) -> List[Color]:
    palette = palette or DEFAULT_PALETTE
    if not palette:
        raise ValueError("Palette cannot be empty.")
    repeated = list(itertools.islice(itertools.cycle(palette), n))
    return repeated


def random_palette(n: int, seed: int | None = None) -> List[Color]:
    rng = random.Random(seed)
    return [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(n)]


def format_label(label: str, score: float | None = None) -> str:
    return f"{label} ({score:.2f})" if score is not None else label


@dataclass
class DrawingSpec:
    """Parameters controlling drawing appearance."""

    outline: Color = (255, 255, 255)
    fill: Color | None = None
    width: int = 2


def _ensure_pil_image(image):
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for drawing visualizations.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype("uint8"), "RGB")
    except Exception:  # pragma: no cover - optional dependency
        pass
    raise TypeError("image must be a PIL.Image or a numpy array.")


def _polygon_centroid(poly_pts: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Return (cx, cy) centroid of polygon points."""
    pts = [tuple(map(float, pt)) for pt in poly_pts]
    if not pts:
        return (0.0, 0.0)
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    return (cx, cy)


def _polygon_top_anchor(poly_pts: Sequence[Sequence[float]], offset: float = 2.0) -> Tuple[float, float]:
    """Return (x, y) position for label at the top of the polygon (smallest y = top in image coords).
    Uses the midpoint of the top edge so the label sits near the top without obscuring the box content.
    """
    pts = [tuple(map(float, pt)) for pt in poly_pts]
    if not pts:
        return (0.0, 0.0)
    sorted_by_y = sorted(pts, key=lambda p: p[1])
    top_two = sorted_by_y[:2]
    mx = (top_two[0][0] + top_two[1][0]) / 2
    my = (top_two[0][1] + top_two[1][1]) / 2
    return (mx, my - offset)


def draw_polygons(
    image,
    polygons: Iterable[Sequence[Sequence[float]]],
    specs: Sequence[DrawingSpec] | None = None,
    labels: Iterable[str] | None = None,
    label_color: Color = (255, 255, 255),
):
    """Draw polygons on image, optionally with text labels near the top of each polygon.

    Args:
        image: PIL Image or numpy array (RGB).
        polygons: Iterable of polygon point lists.
        specs: Optional drawing specs (outline, fill, width) per polygon.
        labels: Optional list of text labels; if provided, length must match polygons.
        label_color: RGB color for label text (default white).

    Returns:
        PIL Image with polygons and labels drawn.
    """
    pil_img = _ensure_pil_image(image)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    polygons = list(polygons)
    specs = specs or [DrawingSpec(outline=color) for color in cycle_palette(len(polygons) or 1)]
    if len(specs) != len(polygons):
        raise ValueError("specs length must match polygons length.")
    labels_list = list(labels) if labels is not None else None
    if labels_list is not None and len(labels_list) != len(polygons):
        raise ValueError("labels length must match polygons length when provided.")
    font = None
    if ImageFont is not None:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            try:
                font = ImageFont.truetype(path, 10)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                pass
    for i, (poly_pts, spec) in enumerate(zip(polygons, specs)):
        pts = [tuple(map(float, pt)) for pt in poly_pts]
        if spec.fill:
            draw.polygon(pts, fill=spec.fill + (64,))
        draw.line(pts + [pts[0]], fill=spec.outline, width=spec.width)
        if labels_list is not None and i < len(labels_list) and labels_list[i]:
            x, y = _polygon_top_anchor(poly_pts)
            kwargs = {"fill": label_color}
            if font is not None:
                kwargs["font"] = font
            draw.text((x, y), labels_list[i], **kwargs)
    return pil_img


def draw_boxes(image, boxes: Iterable[RBox | QBox | Sequence[float]], **kwargs):
    polygons = [as_polygon(box).points for box in boxes]
    return draw_polygons(image, polygons, **kwargs)


__all__ = [
    "cycle_palette",
    "random_palette",
    "format_label",
    "DrawingSpec",
    "draw_polygons",
    "draw_boxes",
]
