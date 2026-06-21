"""Tests for data augmentation transforms."""

import math

from oriented_det.data import DiagonalFlip, HorizontalFlip, Rotate, VerticalFlip
from oriented_det.geometry import RBox


def test_horizontal_flip():
    """Test horizontal flip transform."""
    flip = HorizontalFlip(p=1.0)
    
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=0)
    image_width, image_height = 400, 600
    
    flipped = flip.apply_to_rbox(rbox, image_width, image_height)
    
    # After horizontal flip, x should be mirrored
    assert math.isclose(flipped.cx, image_width - rbox.cx, abs_tol=1e-6)
    assert math.isclose(flipped.cy, rbox.cy, abs_tol=1e-6)
    # Angle should be negated
    assert math.isclose(flipped.angle, -rbox.angle, abs_tol=1e-6)


def test_vertical_flip():
    """Test vertical flip transform."""
    flip = VerticalFlip(p=1.0)
    
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=0)
    image_width, image_height = 400, 600
    
    flipped = flip.apply_to_rbox(rbox, image_width, image_height)
    
    # After vertical flip, y should be mirrored
    assert math.isclose(flipped.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(flipped.cy, image_height - rbox.cy, abs_tol=1e-6)


def test_diagonal_flip():
    """Test diagonal flip transform (MMRotate le90)."""
    flip = DiagonalFlip(p=1.0)

    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=math.radians(15))
    image_width, image_height = 400, 600

    flipped = flip.apply_to_rbox(rbox, image_width, image_height)

    assert math.isclose(flipped.cx, image_width - rbox.cx, abs_tol=1e-6)
    assert math.isclose(flipped.cy, image_height - rbox.cy, abs_tol=1e-6)
    assert math.isclose(flipped.angle, math.pi - rbox.angle, abs_tol=1e-6)


def test_rotate():
    """Test rotation transform."""
    rotate = Rotate(degrees=90, p=1.0)
    
    rbox = RBox(cx=200, cy=300, width=50, height=30, angle=0)
    image_width, image_height = 400, 600
    
    rotated = rotate.apply_to_rbox(rbox, image_width, image_height)
    
    # Angle should be increased by rotation
    assert math.isclose(rotated.angle, rbox.angle + math.radians(90), abs_tol=1e-6)
    # Center should be rotated around image center
    assert isinstance(rotated, RBox)


def test_compose_transforms():
    """Test composing multiple transforms."""
    from oriented_det.data import Compose
    
    flip = HorizontalFlip(p=1.0)
    rotate = Rotate(degrees=90, p=1.0)
    
    compose = Compose([flip, rotate])
    
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=0)
    image_width, image_height = 400, 600
    
    # Compose doesn't have apply_to_rbox, it uses __call__ which needs image
    # So we test that it can be instantiated and has the transforms
    assert len(compose.transforms) == 2
    assert isinstance(compose.transforms[0], HorizontalFlip)
    assert isinstance(compose.transforms[1], Rotate)


def test_transform_probability():
    """Test that transforms respect probability parameter."""
    import random
    
    random.seed(42)
    flip = HorizontalFlip(p=0.0)  # Never flip
    
    rbox = RBox(cx=100, cy=200, width=50, height=30, angle=0)
    image_width, image_height = 400, 600
    
    # With p=0, transform __call__ should not apply geometric changes.
    _, transformed_boxes = flip(
        image=None,
        rboxes=[rbox],
        image_width=image_width,
        image_height=image_height,
    )
    flipped = transformed_boxes[0]
    # Original box should be unchanged (cx should be same)
    assert math.isclose(flipped.cx, rbox.cx, abs_tol=1e-6)

