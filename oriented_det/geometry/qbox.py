"""Quadrilateral bounding box helper structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import math

from .poly import Point, Polygon, _to_point


def _normalize_points(points: Iterable[Sequence[float]]) -> Tuple[Point, ...]:
    pts = tuple(_to_point(pt) for pt in points)
    if len(pts) != 4:
        raise ValueError("A quadrilateral box requires exactly four points.")
    poly = Polygon(pts).ensure_orientation(clockwise=False)
    ordered = poly.points
    top_idx = min(range(4), key=lambda idx: (ordered[idx][1], ordered[idx][0]))
    return tuple(ordered[(top_idx + i) % 4] for i in range(4))


@dataclass(frozen=True)
class QBox:
    """Simple quadrilateral box with geometry helpers."""

    points: Tuple[Point, Point, Point, Point]

    def __init__(self, points: Iterable[Sequence[float]]):
        object.__setattr__(self, "points", _normalize_points(points))

    def to_polygon(self) -> Polygon:
        return Polygon(self.points)

    @property
    def center(self) -> Point:
        xs, ys = zip(*self.points)
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    @property
    def edges(self) -> Tuple[float, float, float, float]:
        lengths = []
        pts = self.points + (self.points[0],)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            lengths.append(math.hypot(x1 - x0, y1 - y0))
        return tuple(lengths)  # type: ignore[return-value]

    @property
    def width(self) -> float:
        e0, _, e2, _ = self.edges
        return (e0 + e2) / 2.0

    @property
    def height(self) -> float:
        _, e1, _, e3 = self.edges
        return (e1 + e3) / 2.0

    @property
    def angle(self) -> float:
        (x0, y0), (x1, y1) = self.points[0], self.points[1]
        return math.atan2(y1 - y0, x1 - x0)

    def as_tuple(self) -> Tuple[Point, Point, Point, Point]:
        return self.points
