"""Rotated rectangle helper utilities.

Angle Convention:
-----------------
The angle convention follows the standard image coordinate system:
- Origin: Top-left corner of the image
- X-axis: Points to the right (positive direction)
- Y-axis: Points downward (positive direction)
- Angle: Measured in radians, positive angles rotate counter-clockwise
  (mathematical convention, opposite to clock direction)

By default, angles are normalized to the interval (-π, π].
For the "le90" convention (long edge 90°), use normalize_le90() to
normalize to [-π/2, π/2).

Example:
    An angle of π/4 (45°) means the box is rotated 45° counter-clockwise
    from the horizontal axis. The first edge (width) points in the direction
    of the angle, and the second edge (height) is perpendicular to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import math

from .poly import Point, Polygon
from .qbox import QBox


def _normalize_angle(angle: float) -> float:
    """Normalize angle to (-π, π] interval.
    
    This is the default normalization used by RBox.
    For le90 convention, use normalize_le90() instead.
    """
    tau = math.tau if hasattr(math, "tau") else 2.0 * math.pi
    angle = math.fmod(angle, tau)
    if angle <= -math.pi:
        angle += tau
    elif angle > math.pi:
        angle -= tau
    return angle


def normalize_le90(rbox: RBox) -> RBox:
    """Normalize RBox angle to [-π/2, π/2) using le90 convention.
    
    The "le90" (long edge 90°) convention ensures that:
    - The angle is in [-π/2, π/2)
    - Width is always >= height (swaps dimensions if needed)
    - This makes boxes with the same orientation have the same representation
    
    The algorithm:
    1. First normalize angle to [-π/2, π/2) by adding/subtracting integer multiples of π
       (without swapping width/height).
    2. If width < height, swap dimensions and adjust angle by π/2.
    3. Normalize angle again while preserving width >= height.
    
    Args:
        rbox: Input RBox to normalize
    
    Returns:
        Normalized RBox with angle in [-π/2, π/2) and width >= height
    
    Example:
        >>> rbox = RBox(100, 100, 50, 100, math.pi)  # width < height, angle = π
        >>> normalized = normalize_le90(rbox)
        >>> # normalized.width = 100, normalized.height = 50, normalized.angle ≈ -π/2
    """
    angle = rbox.angle
    width = rbox.width
    height = rbox.height
    
    # Step 1: Normalize angle to [-π/2, π/2) via π-period wrapping.
    # A π rotation does NOT swap rectangle side assignment.
    pi = math.pi
    pi_half = pi / 2
    k = math.floor((angle + pi_half) / pi)
    angle = angle - k * pi
    
    # Step 2: Ensure width >= height by rotating by π/2 if needed
    # When we swap w/h, we need to adjust the angle to represent the new long edge
    if width < height:
        width, height = height, width
        angle += pi_half
        # Normalize to [-π/2, π/2) while preserving width >= height.
        angle = math.fmod(angle + pi_half, pi) - pi_half
    
    return RBox(rbox.cx, rbox.cy, width, height, angle)


@dataclass(frozen=True)
class RBox:
    """Immutable representation of a rotated rectangle."""

    cx: float
    cy: float
    width: float
    height: float
    angle: float = 0.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RBox width and height must be positive.")
        object.__setattr__(self, "cx", float(self.cx))
        object.__setattr__(self, "cy", float(self.cy))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "angle", _normalize_angle(float(self.angle)))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def is_valid(self, min_size: float = 1.0) -> bool:
        """Check if this RBox is valid (not degenerated).
        
        Args:
            min_size: Minimum width and height in pixels (default: 1.0)
        
        Returns:
            True if RBox is valid, False if degenerated
        """
        try:
            # Check for minimum size (width and height must be >= min_size)
            if self.width < min_size or self.height < min_size:
                return False
            
            # Check for NaN or inf values
            if (not math.isfinite(self.cx) or not math.isfinite(self.cy) or
                not math.isfinite(self.width) or not math.isfinite(self.height) or
                not math.isfinite(self.angle)):
                return False
            
            # Check for positive area
            if self.area <= 0:
                return False
            
            # Try to compute corners to catch any other issues
            corners = self.corners()
            if len(corners) != 4:
                return False
            
            return True
        except (ValueError, AttributeError, TypeError):
            return False

    def with_angle(self, radians: float) -> "RBox":
        return RBox(self.cx, self.cy, self.width, self.height, radians)

    def corners(self) -> Tuple[Point, Point, Point, Point]:
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        w2, h2 = self.width / 2.0, self.height / 2.0
        local = (
            (-w2, -h2),
            (w2, -h2),
            (w2, h2),
            (-w2, h2),
        )
        pts = []
        for x, y in local:
            rx = self.cx + x * cos_a - y * sin_a
            ry = self.cy + x * sin_a + y * cos_a
            pts.append((rx, ry))
        return tuple(pts)  # type: ignore[return-value]

    def to_polygon(self) -> Polygon:
        return Polygon(self.corners())

    def to_qbox(self) -> QBox:
        return QBox(self.corners())

    def axis_aligned_bounds(self) -> Tuple[float, float, float, float]:
        return self.to_polygon().bounds

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]]) -> "RBox":
        qbox = QBox(points)
        return cls.from_qbox(qbox)

    @classmethod
    def from_qbox(cls, qbox: QBox) -> "RBox":
        cx, cy = qbox.center
        width = qbox.width
        height = qbox.height
        angle = qbox.angle
        return cls(cx, cy, width, height, angle)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> "RBox":
        """Create RBox from a quadrilateral polygon.
        
        Args:
            polygon: Polygon with exactly 4 points
        
        Returns:
            RBox representing the polygon
        
        Raises:
            ValueError: If polygon doesn't have exactly 4 points
        """
        if len(polygon) != 4:
            raise ValueError("Only quadrilateral polygons can be converted to RBox.")
        return cls.from_points(polygon.points)
    
    @classmethod
    def from_xyxytheta(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        theta: float,
    ) -> "RBox":
        """Create RBox from axis-aligned bounding box and rotation angle.
        
        This is useful when you have an axis-aligned box (x1, y1, x2, y2)
        and a rotation angle theta.
        
        Args:
            x1: X coordinate of the top-left corner of the axis-aligned box.
            y1: Y coordinate of the top-left corner of the axis-aligned box.
            x2: X coordinate of the bottom-right corner of the axis-aligned box.
            y2: Y coordinate of the bottom-right corner of the axis-aligned box.
            theta: Rotation angle in radians.
        
        Returns:
            RBox with center at the center of the bounding box,
            dimensions from the bounding box, and the specified angle
        
        Example:
            >>> # Axis-aligned box from (10, 20) to (50, 60) rotated 30°
            >>> rbox = RBox.from_xyxytheta(10, 20, 50, 60, math.radians(30))
            >>> # rbox.cx = 30, rbox.cy = 40, rbox.width = 40, rbox.height = 40
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return cls(cx, cy, width, height, theta)


__all__ = ["RBox", "normalize_le90"]
