"""Tests for MMRotate-style training random rotate."""

import math

from PIL import Image

from oriented_det.data.rotates import (
    apply_random_train_rotate,
    apply_rotate_to_image,
    apply_rotate_to_rboxes,
)
from oriented_det.geometry.rbox import RBox
import oriented_det.data.rotates as rotates_mod


def test_rotate_90_image_space_center_and_angle():
    """PIL +90° is visual CCW: (75, 50) angle 0 → (50, 25) angle −π/2."""
    rbox = RBox(75.0, 50.0, 40.0, 20.0, 0.0)
    out = apply_rotate_to_rboxes([rbox], math.pi / 2.0, image_width=100.0, image_height=100.0)
    assert len(out) == 1
    assert math.isclose(out[0].cx, 50.0, abs_tol=1e-5)
    assert math.isclose(out[0].cy, 25.0, abs_tol=1e-5)
    assert math.isclose(out[0].angle, -math.pi / 2.0, abs_tol=1e-5)


def test_rotate_zero_is_identity():
    rbox = RBox(30.0, 40.0, 16.0, 8.0, 0.2)
    out = apply_rotate_to_rboxes([rbox], 0.0, image_width=80.0, image_height=60.0)
    assert math.isclose(out[0].cx, rbox.cx)
    assert math.isclose(out[0].cy, rbox.cy)
    assert math.isclose(out[0].width, rbox.width)
    assert math.isclose(out[0].height, rbox.height)
    assert math.isclose(out[0].angle, rbox.angle, abs_tol=1e-6)


def test_rotate_output_is_le90():
    rbox = RBox(40.0, 50.0, 10.0, 24.0, 0.3)
    out = apply_rotate_to_rboxes([rbox], math.radians(35.0), image_width=100.0, image_height=80.0)
    assert out[0].width >= out[0].height
    assert -math.pi / 2.0 <= out[0].angle < math.pi / 2.0


def test_rotate_keeps_image_size():
    image = Image.new("RGB", (120, 80), (10, 20, 30))
    out = apply_rotate_to_image(image, 30.0)
    assert out.size == (120, 80)


def test_random_rotate_prob_zero_never_applies(monkeypatch):
    called = []

    def fail_uniform(*_args, **_kwargs):
        called.append(True)
        return 45.0

    monkeypatch.setattr(rotates_mod.random, "uniform", fail_uniform)
    image = Image.new("RGB", (32, 32), (0, 0, 0))
    rbox = RBox(16.0, 16.0, 8.0, 4.0, 0.0)
    out_img, out_boxes = apply_random_train_rotate(
        image,
        [rbox],
        image_width=32.0,
        image_height=32.0,
        prob=0.0,
        angle_range_deg=180.0,
    )
    assert not called
    assert out_boxes[0].cx == rbox.cx
    assert out_img.size == image.size


def test_random_rotate_prob_one_always_applies(monkeypatch):
    monkeypatch.setattr(rotates_mod.random, "random", lambda: 0.0)
    monkeypatch.setattr(rotates_mod.random, "uniform", lambda *_args, **_kwargs: 90.0)
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    rbox = RBox(75.0, 50.0, 40.0, 20.0, 0.0)
    _, out_boxes = apply_random_train_rotate(
        image,
        [rbox],
        image_width=100.0,
        image_height=100.0,
        prob=1.0,
        angle_range_deg=180.0,
    )
    assert math.isclose(out_boxes[0].cx, 50.0, abs_tol=1e-5)
    assert math.isclose(out_boxes[0].cy, 25.0, abs_tol=1e-5)
