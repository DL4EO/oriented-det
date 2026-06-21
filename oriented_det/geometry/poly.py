"""Fundamental polygon utilities used across the geometry package.

The implementation intentionally avoids heavy geometry dependencies.
Only the Python standard library is required which keeps the module
easy to vendor inside training or inference environments where wheels
are difficult to install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple
import math

Point = Tuple[float, float]


def _to_point(point: Sequence[float]) -> Point:
    if len(point) != 2:
        raise ValueError("Each point must have exactly two coordinates.")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Point coordinates must be finite numbers.")
    return (x, y)


def _remove_redundant_points(points: Iterable[Point]) -> Tuple[Point, ...]:
    cleaned: List[Point] = []
    for pt in points:
        if not cleaned or cleaned[-1] != pt:
            cleaned.append(pt)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned)


@dataclass(frozen=True)
class Polygon:
    """Immutable polygon helper with a small but focused API."""

    points: Tuple[Point, ...]

    def __init__(self, points: Iterable[Sequence[float]]):
        processed = _remove_redundant_points(_to_point(p) for p in points)
        if len(processed) < 3:
            raise ValueError("A polygon requires at least three distinct points.")
        object.__setattr__(self, "points", processed)
        if math.isclose(self.area, 0.0, abs_tol=1e-9):
            raise ValueError("Degenerate polygon detected (zero area).")

    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def signed_area(self) -> float:
        area = 0.0
        pts = self.points + (self.points[0],)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            area += x0 * y1 - x1 * y0
        return area / 2.0

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    @property
    def is_clockwise(self) -> bool:
        return self.signed_area < 0.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = zip(*self.points)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def centroid(self) -> Point:
        area = self.signed_area
        if math.isclose(area, 0.0, abs_tol=1e-12):
            raise ValueError("Cannot compute centroid of degenerate polygon.")
        cx = cy = 0.0
        pts = self.points + (self.points[0],)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            factor = x0 * y1 - x1 * y0
            cx += (x0 + x1) * factor
            cy += (y0 + y1) * factor
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        return (cx, cy)

    def translate(self, dx: float, dy: float) -> "Polygon":
        return Polygon((x + dx, y + dy) for x, y in self.points)

    def rotate(self, radians: float, origin: Point | None = None) -> "Polygon":
        if origin is None:
            origin = (0.0, 0.0)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        ox, oy = origin
        rotated = []
        for x, y in self.points:
            x0, y0 = x - ox, y - oy
            rotated.append((ox + x0 * cos_a - y0 * sin_a, oy + x0 * sin_a + y0 * cos_a))
        return Polygon(rotated)

    def ensure_orientation(self, clockwise: bool) -> "Polygon":
        if self.is_clockwise == clockwise:
            return self
        return Polygon(reversed(self.points))

    @classmethod
    def rectangle(cls, cx: float, cy: float, width: float, height: float) -> "Polygon":
        if width <= 0 or height <= 0:
            raise ValueError("Rectangle width and height must be positive.")
        w2, h2 = width / 2.0, height / 2.0
        pts = (
            (cx - w2, cy - h2),
            (cx + w2, cy - h2),
            (cx + w2, cy + h2),
            (cx - w2, cy + h2),
        )
        return cls(pts)

    def to_list(self) -> List[Point]:
        return list(self.points)
