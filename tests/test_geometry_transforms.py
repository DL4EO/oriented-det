"""Tests for geometry transformations and utilities."""

import math
import pytest

from oriented_det.geometry import RBox, Polygon
from oriented_det.geometry.transforms import (
    rbox_from_xyxytheta,
    rbox_from_polygon,
    normalize_le90,
    flip_diagonal,
    flip_horizontal,
    flip_vertical,
    rotate_90,
    rotate,
    scale,
    translate,
)


def test_rbox_from_xyxytheta():
    """Test creating RBox from axis-aligned box and angle."""
    rbox = rbox_from_xyxytheta(10, 20, 50, 60, math.radians(30))
    
    assert math.isclose(rbox.cx, 30.0)
    assert math.isclose(rbox.cy, 40.0)
    assert math.isclose(rbox.width, 40.0)
    assert math.isclose(rbox.height, 40.0)
    assert math.isclose(rbox.angle, math.radians(30))


def test_rbox_from_polygon():
    """Test creating RBox from polygon."""
    poly = Polygon.rectangle(100, 100, 50, 30)
    rbox = rbox_from_polygon(poly)
    
    assert math.isclose(rbox.cx, 100.0)
    assert math.isclose(rbox.cy, 100.0)
    assert math.isclose(rbox.width, 50.0)
    assert math.isclose(rbox.height, 30.0)


def test_normalize_le90_basic():
    """Test le90 normalization with basic cases."""
    # Box with angle in [-π/2, π/2) and width >= height should be unchanged
    rbox = RBox(100, 100, 80, 40, math.radians(30))
    normalized = normalize_le90(rbox)
    
    assert math.isclose(normalized.width, 80.0)
    assert math.isclose(normalized.height, 40.0)
    assert -math.pi / 2 <= normalized.angle < math.pi / 2


def test_normalize_le90_swap_dimensions():
    """Test le90 normalization swaps dimensions when width < height."""
    # Box with width < height should swap
    rbox = RBox(100, 100, 40, 80, math.radians(30))
    normalized = normalize_le90(rbox)
    
    assert normalized.width >= normalized.height
    assert math.isclose(normalized.width, 80.0)
    assert math.isclose(normalized.height, 40.0)


def test_normalize_le90_angle_range():
    """Test le90 normalization keeps angle in [-π/2, π/2)."""
    # Test various angles
    for angle_deg in [-180, -90, -45, 0, 45, 90, 180]:
        rbox = RBox(100, 100, 80, 40, math.radians(angle_deg))
        normalized = normalize_le90(rbox)
        
        assert -math.pi / 2 <= normalized.angle < math.pi / 2
        assert normalized.width >= normalized.height


def test_flip_horizontal():
    """Test horizontal flip transformation."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(30))
    image_width = 400
    
    flipped = flip_horizontal(rbox, image_width)
    
    # X should be mirrored
    assert math.isclose(flipped.cx, image_width - rbox.cx)
    assert math.isclose(flipped.cy, rbox.cy)
    # Angle should be negated
    assert math.isclose(flipped.angle, -rbox.angle, abs_tol=1e-6)
    # Dimensions unchanged
    assert math.isclose(flipped.width, rbox.width)
    assert math.isclose(flipped.height, rbox.height)


def test_flip_vertical():
    """Test vertical flip transformation."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(30))
    image_height = 600
    
    flipped = flip_vertical(rbox, image_height)
    
    # Y should be mirrored
    assert math.isclose(flipped.cx, rbox.cx)
    assert math.isclose(flipped.cy, image_height - rbox.cy)
    # Angle should be π - angle
    assert math.isclose(flipped.angle, math.pi - rbox.angle, abs_tol=1e-6)
    # Dimensions unchanged
    assert math.isclose(flipped.width, rbox.width)
    assert math.isclose(flipped.height, rbox.height)


def test_flip_diagonal():
    """Test diagonal flip (MMRotate le90)."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(30))
    image_width, image_height = 400, 600

    flipped = flip_diagonal(rbox, image_width, image_height)

    assert math.isclose(flipped.cx, image_width - rbox.cx)
    assert math.isclose(flipped.cy, image_height - rbox.cy)
    assert math.isclose(flipped.angle, math.pi - rbox.angle, abs_tol=1e-6)


def test_rotate_90():
    """Test 90° rotation."""
    rbox = RBox(cx=200, cy=300, width=80, height=40, angle=math.radians(30))
    image_width, image_height = 400, 600
    
    rotated = rotate_90(rbox, image_width, image_height, k=1)
    
    # Angle should increase by 90°
    assert math.isclose(rotated.angle, rbox.angle + math.pi / 2, abs_tol=1e-6)
    # Width and height should swap
    assert math.isclose(rotated.width, rbox.height)
    assert math.isclose(rotated.height, rbox.width)


def test_rotate_90_180():
    """Test 180° rotation (two 90° rotations)."""
    rbox = RBox(cx=200, cy=300, width=80, height=40, angle=math.radians(30))
    image_width, image_height = 400, 600
    
    rotated = rotate_90(rbox, image_width, image_height, k=2)
    
    # Angle should increase by 180° (mod 2π; result may be normalized to le90)
    d = (rotated.angle - rbox.angle - math.pi) % (2 * math.pi)
    assert d < 1e-6 or (2 * math.pi - d) < 1e-6
    # Width and height should be back to original (even number of swaps)
    assert math.isclose(rotated.width, rbox.width)
    assert math.isclose(rotated.height, rbox.height)


def test_rotate_arbitrary():
    """Test arbitrary angle rotation."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(20))
    
    rotated = rotate(rbox, math.radians(45), origin_x=0, origin_y=0)
    
    # Angle should increase by rotation amount
    assert math.isclose(rotated.angle, rbox.angle + math.radians(45), abs_tol=1e-6)
    # Dimensions unchanged
    assert math.isclose(rotated.width, rbox.width)
    assert math.isclose(rotated.height, rbox.height)


def test_scale():
    """Test scaling transformation (center and dimensions scale, e.g. when resizing image)."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(20))
    
    scaled = scale(rbox, 2.0)
    
    assert math.isclose(scaled.width, 100.0)
    assert math.isclose(scaled.height, 60.0)
    assert math.isclose(scaled.cx, 200.0)  # center scales with factor
    assert math.isclose(scaled.cy, 400.0)
    assert math.isclose(scaled.angle, rbox.angle)


def test_scale_xy():
    """Test scaling with different x and y factors."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(20))
    
    scaled = scale(rbox, 2.0, 3.0)
    
    assert math.isclose(scaled.width, 100.0)
    assert math.isclose(scaled.height, 90.0)


def test_translate():
    """Test translation transformation."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(20))
    
    translated = translate(rbox, 10, 20)
    
    assert math.isclose(translated.cx, 110.0)
    assert math.isclose(translated.cy, 220.0)
    assert math.isclose(translated.width, rbox.width)
    assert math.isclose(translated.height, rbox.height)
    assert math.isclose(translated.angle, rbox.angle)


def test_flip_horizontal_round_trip():
    """Test that double horizontal flip returns to original."""
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(30))
    image_width = 400
    
    flipped_once = flip_horizontal(rbox, image_width)
    flipped_twice = flip_horizontal(flipped_once, image_width)
    
    assert math.isclose(flipped_twice.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(flipped_twice.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(flipped_twice.angle, rbox.angle, abs_tol=1e-6)


def test_flip_vertical_round_trip():
    """Test that double vertical flip returns to original."""
    rbox = RBox(cx=140, cy=220, width=60, height=25, angle=math.radians(20))
    image_height = 500
    flipped_once = flip_vertical(rbox, image_height)
    flipped_twice = flip_vertical(flipped_once, image_height)
    assert math.isclose(flipped_twice.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(flipped_twice.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(flipped_twice.width, rbox.width, abs_tol=1e-6)
    assert math.isclose(flipped_twice.height, rbox.height, abs_tol=1e-6)
    assert math.isclose(flipped_twice.angle, rbox.angle, abs_tol=1e-6)


def test_rotate_then_inverse_rotate_returns_original():
    """Rotate by theta then -theta should recover original box."""
    rbox = RBox(cx=33.0, cy=-17.0, width=18.0, height=7.0, angle=math.radians(-40))
    theta = math.radians(67)
    origin_x, origin_y = 12.0, -4.0
    rotated = rotate(rbox, theta, origin_x=origin_x, origin_y=origin_y)
    recovered = rotate(rotated, -theta, origin_x=origin_x, origin_y=origin_y)
    assert math.isclose(recovered.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(recovered.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(recovered.width, rbox.width, abs_tol=1e-6)
    assert math.isclose(recovered.height, rbox.height, abs_tol=1e-6)
    assert math.isclose(recovered.angle, rbox.angle, abs_tol=1e-6)


def test_scale_then_inverse_scale_returns_original():
    """Scale by factors then inverse factors should recover original."""
    rbox = RBox(cx=90, cy=45, width=40, height=16, angle=math.radians(15))
    sx, sy = 2.5, 0.4
    scaled = scale(rbox, sx, sy)
    recovered = scale(scaled, 1.0 / sx, 1.0 / sy)
    assert math.isclose(recovered.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(recovered.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(recovered.width, rbox.width, abs_tol=1e-6)
    assert math.isclose(recovered.height, rbox.height, abs_tol=1e-6)
    assert math.isclose(recovered.angle, rbox.angle, abs_tol=1e-6)


def test_translate_then_inverse_translate_returns_original():
    """Translate by (dx,dy) then (-dx,-dy) should recover original."""
    rbox = RBox(cx=-10, cy=80, width=22, height=11, angle=math.radians(-5))
    dx, dy = 13.5, -91.0
    moved = translate(rbox, dx, dy)
    recovered = translate(moved, -dx, -dy)
    assert math.isclose(recovered.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(recovered.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(recovered.width, rbox.width, abs_tol=1e-6)
    assert math.isclose(recovered.height, rbox.height, abs_tol=1e-6)
    assert math.isclose(recovered.angle, rbox.angle, abs_tol=1e-6)


def test_rotate_90_round_trip():
    """Test that four 90° rotations return to original."""
    rbox = RBox(cx=200, cy=300, width=80, height=40, angle=math.radians(30))
    image_width, image_height = 400, 600
    
    rotated = rbox
    for _ in range(4):
        rotated = rotate_90(rotated, image_width, image_height, k=1)
    
    # Should be back to original (within floating point precision)
    assert math.isclose(rotated.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(rotated.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(rotated.width, rbox.width, abs_tol=1e-6)
    assert math.isclose(rotated.height, rbox.height, abs_tol=1e-6)
    assert math.isclose(rotated.angle, rbox.angle, abs_tol=1e-6)

