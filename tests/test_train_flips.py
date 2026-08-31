"""Tests for MMRotate-style training flips."""

import math

from oriented_det.data.flips import apply_flip_to_rboxes, apply_random_train_flips
from oriented_det.geometry import transforms as geom_transforms
from oriented_det.geometry.rbox import RBox, normalize_le90


def test_flip_diagonal_center_keeps_angle():
    """MMRotate diagonal mirrors center only; angle is unchanged."""
    rbox = RBox(100.0, 200.0, 80.0, 40.0, math.radians(20.0))
    w, h = 400.0, 600.0
    flipped = geom_transforms.flip_diagonal(rbox, w, h)
    assert math.isclose(flipped.cx, w - rbox.cx)
    assert math.isclose(flipped.cy, h - rbox.cy)
    assert math.isclose(flipped.angle, rbox.angle, abs_tol=1e-6)


def test_flip_diagonal_matches_horizontal_then_vertical_le90():
    """Diagonal equals H then V after le90 (point reflection; angle preserved)."""
    rbox = normalize_le90(RBox(100.0, 200.0, 80.0, 40.0, math.radians(20.0)))
    w, h = 400.0, 600.0
    diag = normalize_le90(geom_transforms.flip_diagonal(rbox, w, h))
    hv = normalize_le90(
        geom_transforms.flip_vertical(
            geom_transforms.flip_horizontal(rbox, w),
            h,
        )
    )
    assert math.isclose(diag.cx, hv.cx, abs_tol=1e-5)
    assert math.isclose(diag.cy, hv.cy, abs_tol=1e-5)
    assert math.isclose(diag.angle, hv.angle, abs_tol=1e-5)
    assert math.isclose(diag.width, hv.width, abs_tol=1e-5)
    assert math.isclose(diag.height, hv.height, abs_tol=1e-5)


def test_apply_flip_to_rboxes_diagonal():
    rbox = RBox(50.0, 60.0, 30.0, 20.0, 0.1)
    out = apply_flip_to_rboxes([rbox], "diagonal", image_width=200.0, image_height=300.0)
    assert len(out) == 1
    assert out[0].width >= out[0].height
    assert math.isclose(out[0].angle, normalize_le90(rbox).angle, abs_tol=1e-5)


def test_random_flips_three_way_distribution(monkeypatch):
    """With H+V+D, only one flip applies per call (25% buckets)."""
    import oriented_det.data.flips as flips_mod

    calls = []

    def fake_random():
        # Cycle through bucket starts: 0.1 -> h, 0.3 -> v, 0.6 -> d, 0.8 -> none
        vals = [0.1, 0.3, 0.6, 0.8]
        return vals[len(calls) % len(vals)]

    monkeypatch.setattr(flips_mod.random, "random", fake_random)

    rbox = RBox(10.0, 10.0, 4.0, 2.0, 0.0)

    class _MockImage:
        def transpose(self, *_args, **_kwargs):
            return self

    image = _MockImage()

    for _ in range(4):
        img_out, boxes_out = apply_random_train_flips(
            image,
            [rbox],
            image_width=100.0,
            image_height=100.0,
            enable_horizontal=True,
            enable_vertical=True,
            enable_diagonal=True,
        )
        changed = boxes_out[0].cx != rbox.cx or boxes_out[0].cy != rbox.cy
        calls.append(changed)

    assert calls == [True, True, True, False]
