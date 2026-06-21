"""Tests for visualization utilities."""

import pytest

try:
    from PIL import Image
except ImportError:
    Image = None

from oriented_det.geometry import Polygon, QBox, RBox
from oriented_det.utils import viz


@pytest.mark.skipif(Image is None, reason="PIL/Pillow not available")
def test_draw_polygons():
    """Test drawing polygons on an image."""
    img = Image.new("RGB", (100, 100), color="white")
    poly = Polygon.rectangle(10, 10, 20, 30)
    
    result = viz.draw_polygons(img, [poly])
    assert isinstance(result, Image.Image)
    assert result.size == img.size


@pytest.mark.skipif(Image is None, reason="PIL/Pillow not available")
def test_draw_boxes():
    """Test drawing boxes (RBox/QBox) on an image."""
    img = Image.new("RGB", (100, 100), color="white")
    rbox = RBox(50, 50, 20, 30, 0)
    
    result = viz.draw_boxes(img, [rbox])
    assert isinstance(result, Image.Image)


def test_cycle_palette():
    """Test palette cycling."""
    colors = viz.cycle_palette(5)
    assert len(colors) == 5
    assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)


def test_random_palette():
    """Test random palette generation."""
    colors = viz.random_palette(5, seed=42)
    assert len(colors) == 5
    # Same seed should produce same colors
    colors2 = viz.random_palette(5, seed=42)
    assert colors == colors2


def test_format_label():
    """Test label formatting."""
    label = viz.format_label("plane", score=0.95)
    assert "plane" in label
    assert "0.95" in label or "95" in label


def test_drawing_spec():
    """Test DrawingSpec dataclass."""
    spec = viz.DrawingSpec(outline=(255, 0, 0), fill=(10, 20, 30), width=2)
    assert spec.outline == (255, 0, 0)
    assert spec.fill == (10, 20, 30)
    assert spec.width == 2

