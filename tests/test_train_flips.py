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


def test_random_flips_hv_four_combinations(monkeypatch):
    """H and V at p=0.5 are independent: none, H, V, and both."""
    import oriented_det.data.flips as flips_mod

    draws = iter([
        0.1, 0.1,  # both
        0.1, 0.9,  # horizontal
        0.9, 0.1,  # vertical
        0.9, 0.9,  # none
    ])
    monkeypatch.setattr(flips_mod.random, "random", lambda: next(draws))

    rbox = RBox(10.0, 20.0, 4.0, 2.0, 0.2)
    w, h = 100.0, 80.0

    class _MockImage:
        def transpose(self, *_args, **_kwargs):
            return self

    image = _MockImage()
    outcomes = []
    for _ in range(4):
        _, boxes = apply_random_train_flips(
            image,
            [rbox],
            image_width=w,
            image_height=h,
            enable_horizontal=True,
            enable_vertical=True,
            enable_diagonal=False,
            flip_prob=0.5,
        )
        outcomes.append(boxes[0])

    both, horiz, vert, none = outcomes
    expected_h = apply_flip_to_rboxes([rbox], "horizontal", image_width=w, image_height=h)[0]
    expected_v = apply_flip_to_rboxes([rbox], "vertical", image_width=w, image_height=h)[0]
    expected_both = apply_flip_to_rboxes([expected_h], "vertical", image_width=w, image_height=h)[0]

    def _close(a, b):
        assert math.isclose(a.cx, b.cx, abs_tol=1e-5)
        assert math.isclose(a.cy, b.cy, abs_tol=1e-5)
        assert math.isclose(a.angle, b.angle, abs_tol=1e-5)

    _close(both, expected_both)
    _close(horiz, expected_h)
    _close(vert, expected_v)
    _close(none, rbox)


def test_flip_prob_zero_skips(monkeypatch):
    import oriented_det.data.flips as flips_mod

    monkeypatch.setattr(flips_mod.random, "random", lambda: 0.0)
    rbox = RBox(10.0, 20.0, 4.0, 2.0, 0.2)

    class _MockImage:
        def transpose(self, *_args, **_kwargs):
            raise AssertionError("flip_prob=0 must not flip")

    _, boxes = apply_random_train_flips(
        _MockImage(),
        [rbox],
        image_width=100.0,
        image_height=80.0,
        enable_horizontal=True,
        enable_vertical=True,
        enable_diagonal=True,
        flip_prob=0.0,
    )
    assert boxes[0] is rbox


def test_diagonal_flip_is_independent(monkeypatch):
    """Diagonal is its own coin, not a mutually exclusive bucket with H and V."""
    import oriented_det.data.flips as flips_mod

    draws = iter([0.9, 0.9, 0.1])  # skip H, skip V, take diagonal
    monkeypatch.setattr(flips_mod.random, "random", lambda: next(draws))
    rbox = RBox(10.0, 20.0, 4.0, 2.0, 0.2)
    w, h = 100.0, 80.0

    class _MockImage:
        def transpose(self, *_args, **_kwargs):
            return self

    _, boxes = apply_random_train_flips(
        _MockImage(),
        [rbox],
        image_width=w,
        image_height=h,
        enable_horizontal=True,
        enable_vertical=True,
        enable_diagonal=True,
        flip_prob=0.5,
    )
    expected = apply_flip_to_rboxes([rbox], "diagonal", image_width=w, image_height=h)[0]
    assert math.isclose(boxes[0].cx, expected.cx, abs_tol=1e-5)
    assert math.isclose(boxes[0].cy, expected.cy, abs_tol=1e-5)
    assert math.isclose(boxes[0].angle, expected.angle, abs_tol=1e-5)
