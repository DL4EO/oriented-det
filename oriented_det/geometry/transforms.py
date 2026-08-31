"""Conversion helpers and geometric transformations for oriented detection.

This module provides:
1. Conversion functions between polygon, qbox, and rbox representations
2. Geometric transformations (flip, rotate) that preserve oriented box semantics
"""

from __future__ import annotations

from typing import Iterable, Sequence
import math

from .poly import Polygon
from .qbox import QBox
from .rbox import RBox, normalize_le90


def rbox_to_qbox(rbox: RBox) -> QBox:
    return rbox.to_qbox()


def rbox_to_polygon(rbox: RBox) -> Polygon:
    return rbox.to_polygon()


def qbox_to_rbox(qbox: QBox) -> RBox:
    return RBox.from_qbox(qbox)


def qbox_to_polygon(qbox: QBox) -> Polygon:
    return qbox.to_polygon()


def polygon_to_qbox(polygon: Polygon) -> QBox:
    if len(polygon) != 4:
        raise ValueError("Only quadrilateral polygons can be converted to QBox.")
    return QBox(polygon.points)


def polygon_to_rbox(polygon: Polygon) -> RBox:
    return RBox.from_polygon(polygon)


def points_to_rbox(points: Iterable[Sequence[float]]) -> RBox:
    return RBox.from_points(points)


def rbox_from_xyxytheta(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    theta: float,
) -> RBox:
    """Create RBox from axis-aligned bounding box and rotation angle.
    
    Convenience function that calls RBox.from_xyxytheta().
    See RBox.from_xyxytheta() for details.
    """
    return RBox.from_xyxytheta(x1, y1, x2, y2, theta)


def rbox_from_polygon(polygon: Polygon) -> RBox:
    """Create RBox from a quadrilateral polygon.
    
    Convenience function that calls RBox.from_polygon().
    See RBox.from_polygon() for details.
    """
    return RBox.from_polygon(polygon)


# Geometric Transformations
# --------------------------

def flip_horizontal(rbox: RBox, image_width: float) -> RBox:
    """Flip RBox horizontally (mirror across vertical axis).
    
    Args:
        rbox: Input RBox
        image_width: Width of the image (for mirroring)
    
    Returns:
        Horizontally flipped RBox
    
    Note:
        The angle is negated to maintain the correct orientation.
    """
    new_cx = image_width - rbox.cx
    new_angle = -rbox.angle
    return RBox(new_cx, rbox.cy, rbox.width, rbox.height, new_angle)


def flip_vertical(rbox: RBox, image_height: float) -> RBox:
    """Flip RBox vertically (mirror across horizontal axis).
    
    Args:
        rbox: Input RBox
        image_height: Height of the image (for mirroring)
    
    Returns:
        Vertically flipped RBox
    
    Note:
        The angle is transformed as π - angle to maintain correct orientation.
    """
    new_cy = image_height - rbox.cy
    new_angle = math.pi - rbox.angle
    return RBox(rbox.cx, new_cy, rbox.width, rbox.height, new_angle)


def flip_diagonal(rbox: RBox, image_width: float, image_height: float) -> RBox:
    """Flip RBox for MMRotate ``diagonal`` direction (le90: center mirror, angle unchanged).

    Image uses both horizontal and vertical flips (opencv/mmcv ``flipCode=-1``).
    MMRotate ``RRandomFlip.bbox_flip`` for ``direction='diagonal'`` mirrors ``(cx, cy)``
    and returns early without changing θ (unlike H/V, which use ``norm_angle(π − θ)``).
    """
    new_cx = image_width - rbox.cx
    new_cy = image_height - rbox.cy
    return RBox(new_cx, new_cy, rbox.width, rbox.height, rbox.angle)


def rotate_90(rbox: RBox, image_width: float, image_height: float, k: int = 1) -> RBox:
    """Rotate RBox by 90° increments around image center.
    
    Args:
        rbox: Input RBox
        image_width: Width of the image
        image_height: Height of the image
        k: Number of 90° rotations (1 = 90°, 2 = 180°, 3 = 270°, -1 = -90°, etc.)
    
    Returns:
        Rotated RBox
    
    Note:
        k=1 rotates counter-clockwise by 90°, k=-1 rotates clockwise by 90°.
    """
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    
    # Translate to origin
    dx = rbox.cx - center_x
    dy = rbox.cy - center_y
    
    # Rotate k times by 90°
    angle_90 = math.pi / 2.0
    cos_k = math.cos(k * angle_90)
    sin_k = math.sin(k * angle_90)
    
    # Apply rotation
    new_dx = dx * cos_k - dy * sin_k
    new_dy = dx * sin_k + dy * cos_k
    
    # Translate back
    new_cx = center_x + new_dx
    new_cy = center_y + new_dy
    
    # Update angle
    new_angle = rbox.angle + k * angle_90
    
    # Swap width/height for odd k
    if k % 2 != 0:
        width, height = rbox.height, rbox.width
    else:
        width, height = rbox.width, rbox.height
    
    return RBox(new_cx, new_cy, width, height, new_angle)


def rotate(rbox: RBox, radians: float, origin_x: float = 0.0, origin_y: float = 0.0) -> RBox:
    """Rotate RBox by arbitrary angle around a point.
    
    Args:
        rbox: Input RBox
        radians: Rotation angle in radians (positive = counter-clockwise)
        origin_x: X coordinate of rotation center (default: 0, 0)
        origin_y: Y coordinate of rotation center (default: 0, 0)
    
    Returns:
        Rotated RBox
    """
    # Translate to origin
    dx = rbox.cx - origin_x
    dy = rbox.cy - origin_y
    
    # Rotate
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    new_dx = dx * cos_a - dy * sin_a
    new_dy = dx * sin_a + dy * cos_a
    
    # Translate back
    new_cx = origin_x + new_dx
    new_cy = origin_y + new_dy
    
    # Update angle
    new_angle = rbox.angle + radians
    
    return RBox(new_cx, new_cy, rbox.width, rbox.height, new_angle)


def scale(rbox: RBox, scale_x: float, scale_y: float = None) -> RBox:
    """Scale RBox center and dimensions (e.g. when resizing the image).
    
    Args:
        rbox: Input RBox
        scale_x: Scale factor for x (center and width)
        scale_y: Scale factor for y (center and height); defaults to scale_x if None
    
    Returns:
        Scaled RBox (center and dimensions scaled; angle unchanged)
    """
    if scale_y is None:
        scale_y = scale_x
    return RBox(
        rbox.cx * scale_x,
        rbox.cy * scale_y,
        rbox.width * scale_x,
        rbox.height * scale_y,
        rbox.angle,
    )


def translate(rbox: RBox, dx: float, dy: float) -> RBox:
    """Translate RBox by offset.
    
    Args:
        rbox: Input RBox
        dx: Translation in x direction
        dy: Translation in y direction
    
    Returns:
        Translated RBox (dimensions and angle unchanged)
    """
    return RBox(
        rbox.cx + dx,
        rbox.cy + dy,
        rbox.width,
        rbox.height,
        rbox.angle,
    )


__all__ = [
    # Conversions
    "rbox_to_qbox",
    "rbox_to_polygon",
    "qbox_to_rbox",
    "qbox_to_polygon",
    "polygon_to_qbox",
    "polygon_to_rbox",
    "points_to_rbox",
    "rbox_from_xyxytheta",
    "rbox_from_polygon",
    # Transformations
    "flip_horizontal",
    "flip_vertical",
    "flip_diagonal",
    "rotate_90",
    "rotate",
    "scale",
    "translate",
    # Normalization
    "normalize_le90",
]
