"""Cached mosaic (MMRotate 1.x / mmdet CachedMosaic) for oriented boxes."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from PIL import Image

from oriented_det.data.dota import DOTAAnnotation, DOTASample
from oriented_det.data.mosaic import (
    MosaicCache,
    MosaicSample,
    apply_cached_mosaic,
    apply_mosaic,
    apply_mosaic_resize_crop,
    clip_and_filter_centers,
    resolve_mosaic_disable_last_n,
)
from oriented_det.geometry.rbox import RBox
from oriented_det.runtime.collate import create_collate_fn


def _solid(size: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _sample(image: Image.Image, rbox: RBox, label: int = 1) -> MosaicSample:
    return MosaicSample(image=image, rboxes=[rbox], labels=[label])


def test_mosaic_four_quadrants_and_box_translation():
    cell = 64
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    rbox = RBox(32.0, 32.0, 8.0, 4.0, 0.2)
    current = _sample(_solid(cell, colors[0]), rbox)
    mix = [_sample(_solid(cell, c), rbox) for c in colors[1:]]
    out = apply_mosaic(
        current,
        mix,
        img_scale=(cell, cell),
        pad_val=114,
        center_xy=(cell, cell),
    )
    assert out.image.size == (128, 128)
    px = out.image.load()
    assert px[16, 16][:3] == colors[0]
    assert px[80, 16][:3] == colors[1]
    assert px[16, 80][:3] == colors[2]
    assert px[80, 80][:3] == colors[3]
    assert px[0, 0][:3] == (114, 114, 114) or px[0, 0][:3] == colors[0]
    assert len(out.rboxes) == 4
    assert math.isclose(out.rboxes[0].cx, 32.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[0].cy, 32.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[1].cx, 96.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[1].cy, 32.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[2].cx, 32.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[2].cy, 96.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[3].cx, 96.0, abs_tol=1e-4)
    assert math.isclose(out.rboxes[3].cy, 96.0, abs_tol=1e-4)
    for rb in out.rboxes:
        assert math.isclose(rb.angle, 0.2, abs_tol=1e-6)
        assert math.isclose(rb.width, 8.0, abs_tol=1e-4)
        assert math.isclose(rb.height, 4.0, abs_tol=1e-4)


def test_mosaic_drops_center_outside_without_clip():
    cell = 64
    inside = RBox(32.0, 32.0, 8.0, 4.0, 0.1)
    outside = RBox(-40.0, 32.0, 8.0, 4.0, 0.3)
    hanging = RBox(2.0, 32.0, 40.0, 6.0, 0.4)
    current = MosaicSample(
        image=_solid(cell, (10, 10, 10)),
        rboxes=[inside, outside, hanging],
        labels=[1, 2, 3],
    )
    mix = [
        _sample(_solid(cell, (20, 20, 20)), RBox(32.0, 32.0, 4.0, 2.0, 0.0), label=9)
        for _ in range(3)
    ]
    out = apply_mosaic(
        current,
        mix,
        img_scale=(cell, cell),
        bbox_clip_border=False,
        center_xy=(cell, cell),
    )
    labels = set(out.labels)
    assert 1 in labels
    assert 2 not in labels
    assert 3 in labels
    assert labels == {1, 3, 9}
    hanging_out = next(rb for rb, lab in zip(out.rboxes, out.labels) if lab == 3)
    assert math.isclose(hanging_out.angle, 0.4, abs_tol=1e-6)
    assert math.isclose(hanging_out.width, 40.0, abs_tol=1e-4)


def test_mosaic_clip_border_keeps_clamped_center():
    cell = 64
    outside = RBox(-40.0, 32.0, 8.0, 4.0, 0.3)
    current = _sample(_solid(cell, (10, 10, 10)), outside, label=1)
    mix = [
        _sample(_solid(cell, (20, 20, 20)), RBox(32.0, 32.0, 4.0, 2.0, 0.0), label=9)
        for _ in range(3)
    ]
    out = apply_mosaic(
        current,
        mix,
        img_scale=(cell, cell),
        bbox_clip_border=True,
        center_xy=(cell, cell),
    )
    assert 1 in out.labels
    kept = next(rb for rb, lab in zip(out.rboxes, out.labels) if lab == 1)
    assert kept.cx == 0.0
    assert math.isclose(kept.angle, 0.3, abs_tol=1e-6)


def test_mosaic_resize_crop_returns_target_size(monkeypatch):
    import oriented_det.data.mosaic as mosaic_mod
    import oriented_det.data.preprocessing as prep_mod

    monkeypatch.setattr(mosaic_mod.random, "uniform", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(prep_mod.random, "randint", lambda *_a, **_k: 0)

    cell = 64
    rbox = RBox(32.0, 32.0, 8.0, 4.0, 0.0)
    current = _sample(_solid(cell, (255, 0, 0)), rbox)
    mix = [_sample(_solid(cell, (0, 255, 0)), rbox) for _ in range(3)]
    mosaiced = apply_mosaic(current, mix, img_scale=(cell, cell), center_xy=(cell, cell))
    cropped = apply_mosaic_resize_crop(
        mosaiced,
        (cell, cell),
        ratio_range=(1.0, 1.0),
        random_crop=True,
    )
    assert cropped.image.size == (cell, cell)
    assert all(0.0 <= rb.cx < cell and 0.0 <= rb.cy < cell for rb in cropped.rboxes)


def test_mosaic_ratio_is_object_scale_crop_does_not_halve(monkeypatch):
    """ratio multiplies the 2× canvas; crop to cell is a window, not ×0.5."""
    import oriented_det.data.preprocessing as prep_mod

    monkeypatch.setattr(prep_mod.random, "randint", lambda *_a, **_k: 0)

    cell = 64
    box_w, box_h = 16.0, 8.0
    rbox = RBox(32.0, 32.0, box_w, box_h, 0.0)
    current = _sample(_solid(cell, (255, 0, 0)), rbox)
    mix = [_sample(_solid(cell, (0, 255, 0)), rbox) for _ in range(3)]
    mosaiced = apply_mosaic(current, mix, img_scale=(cell, cell), center_xy=(cell, cell))
    assert mosaiced.image.size == (cell * 2, cell * 2)
    native = next(rb for rb in mosaiced.rboxes if math.isclose(rb.width, box_w))
    assert math.isclose(native.width, box_w)

    at_1x = apply_mosaic_resize_crop(
        mosaiced, (cell, cell), ratio_range=(1.0, 1.0), random_crop=True
    )
    kept_1x = next(rb for rb in at_1x.rboxes if math.isclose(rb.width, box_w, abs_tol=0.5))
    assert math.isclose(kept_1x.width, box_w, abs_tol=0.5)

    at_half = apply_mosaic_resize_crop(
        mosaiced, (cell, cell), ratio_range=(0.5, 0.5), random_crop=True
    )
    kept_half = next(rb for rb in at_half.rboxes if math.isclose(rb.width, box_w * 0.5, abs_tol=0.5))
    assert math.isclose(kept_half.width, box_w * 0.5, abs_tol=0.5)


def test_cached_mosaic_cold_cache_is_identity():
    cache = MosaicCache(max_cached_images=8)
    current = _sample(_solid(32, (1, 2, 3)), RBox(8.0, 8.0, 4.0, 2.0, 0.0))
    out = apply_cached_mosaic(current, cache, img_scale=(32, 32), prob=1.0)
    assert out is current
    assert len(cache) == 1


def test_cached_mosaic_prob_zero_is_identity(monkeypatch):
    import oriented_det.data.mosaic as mosaic_mod

    cache = MosaicCache(max_cached_images=8)
    samples = [
        _sample(_solid(32, (i, i, i)), RBox(8.0, 8.0, 4.0, 2.0, 0.0)) for i in range(6)
    ]
    for s in samples:
        cache.push(s)
    assert cache.ready()
    monkeypatch.setattr(mosaic_mod.random, "random", lambda: 0.99)
    current = samples[0]
    out = apply_cached_mosaic(current, cache, img_scale=(32, 32), prob=0.0)
    assert out is current


def test_resolve_mosaic_disable_last_n():
    assert resolve_mosaic_disable_last_n(12) == 1
    assert resolve_mosaic_disable_last_n(100) == 10
    assert resolve_mosaic_disable_last_n(12, override=0) == 0
    assert resolve_mosaic_disable_last_n(12, override=2) == 2


def test_clip_and_filter_centers_empty_labels():
    boxes = [RBox(5.0, 5.0, 2.0, 1.0, 0.0), RBox(50.0, 5.0, 2.0, 1.0, 0.0)]
    kept, labels = clip_and_filter_centers(boxes, None, 10.0, 10.0, clip_border=False)
    assert len(kept) == 1
    assert labels == []


def _write_color_png(path: Path, color: tuple[int, int, int], size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _solid(size, color).save(path)


def _dota_sample(path: Path, cx: float = 16.0, cy: float = 16.0) -> DOTASample:
    rbox = RBox(cx, cy, 8.0, 4.0, 0.0)
    ann = DOTAAnnotation("ship", 0, rbox.to_polygon(), rbox)
    return DOTASample(image_path=path, width=32, height=32, annotations=(ann,))


def test_collate_accepts_mosaic_keys_and_epoch_flag(tmp_path, monkeypatch):
    import oriented_det.runtime.collate as collate_mod

    samples = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, color in enumerate(colors):
        p = tmp_path / f"{i}.png"
        _write_color_png(p, color)
        samples.append(_dota_sample(p))

    collate = create_collate_fn(
        {"ship": 1},
        normalize=False,
        resize_mode="fixed",
        resize_to=(32, 32),
        pad_size_divisor=1,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_mosaic=True,
        mosaic_prob=1.0,
        mosaic_img_scale=(32, 32),
        mosaic_ratio_range=(1.0, 1.0),
        mosaic_disable_last_n_epochs=1,
        mosaic_total_epochs=12,
        mosaic_max_cached_images=8,
    )
    assert collate.enable_mosaic is True
    assert collate.mosaic_is_enabled() is True
    assert collate.update_mosaic_for_epoch(10) is True
    assert collate.update_mosaic_for_epoch(11) is False
    assert collate.mosaic_is_enabled() is False

    collate.set_mosaic_enabled(True)
    calls = {"n": 0}

    def fake_mosaic(current, cache, **kwargs):
        calls["n"] += 1
        return apply_cached_mosaic(current, cache, **kwargs)

    monkeypatch.setattr(collate_mod, "apply_cached_mosaic", fake_mosaic)
    for sample in samples:
        collate([sample])
    assert calls["n"] == 5
    assert len(collate._mosaic_cache) == 5

    collate.set_mosaic_enabled(False)
    calls["n"] = 0
    collate([samples[0]])
    assert calls["n"] == 0


def test_rect_labels_snap_tile_to_right_angle(tmp_path, monkeypatch):
    from oriented_det.data import rotates as rotates_mod

    seen = []
    real = rotates_mod.apply_rotate_to_image

    def _spy(image, degrees, fillcolor=(0, 0, 0)):
        seen.append(float(degrees))
        return real(image, degrees, fillcolor=fillcolor)

    monkeypatch.setattr("oriented_det.data.rotates.apply_rotate_to_image", _spy)
    monkeypatch.setattr("oriented_det.data.rotates.random.random", lambda: 0.0)
    monkeypatch.setattr("oriented_det.data.rotates.random.choice", lambda seq: seq[0])
    monkeypatch.setattr("oriented_det.data.rotates.random.uniform", lambda _a, _b: 37.0)
    path = tmp_path / "tile.png"
    _solid(32, (255, 255, 255)).save(path)
    ship = RBox(10.0, 12.0, 8.0, 4.0, 0.4)
    tank = RBox(20.0, 18.0, 6.0, 6.0, 0.7)

    def _sample(*anns: DOTAAnnotation) -> DOTASample:
        return DOTASample(image_path=path, width=32, height=32, annotations=anns)

    collate = create_collate_fn(
        {"ship": 1, "storage-tank": 2},
        normalize=False,
        resize_mode="fixed",
        resize_to=(32, 32),
        pad_size_divisor=1,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=True,
        random_rotate_prob=1.0,
        random_rotate_angle_range=180.0,
        random_rotate_rect_labels=["storage-tank"],
    )
    collate([
        _sample(
            DOTAAnnotation("ship", 0, ship.to_polygon(), ship),
            DOTAAnnotation("storage-tank", 0, tank.to_polygon(), tank),
        )
    ])
    assert seen == [90.0]
    seen.clear()
    collate([_sample(DOTAAnnotation("ship", 0, ship.to_polygon(), ship))])
    assert seen == [37.0]

    still = create_collate_fn(
        {"ship": 1, "storage-tank": 2},
        normalize=False,
        resize_mode="fixed",
        resize_to=(32, 32),
        pad_size_divisor=1,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=False,
        random_rotate_rect_labels=["storage-tank"],
    )
    _, held = still([
        _sample(DOTAAnnotation("storage-tank", 0, tank.to_polygon(), tank))
    ])
    assert held[0]["rboxes"][0, 4].item() == pytest.approx(0.7)


def test_mosaic_tile_with_rect_label_uses_right_angle(monkeypatch):
    from oriented_det.data import rotates as rotates_mod
    from oriented_det.data.mosaic import _rotate_sample_copy

    seen = []
    real = rotates_mod.apply_rotate_to_image

    def _spy(image, degrees, fillcolor=(0, 0, 0)):
        seen.append(float(degrees))
        return real(image, degrees, fillcolor=fillcolor)

    monkeypatch.setattr("oriented_det.data.mosaic.apply_rotate_to_image", _spy)
    monkeypatch.setattr("oriented_det.data.rotates.random.random", lambda: 0.0)
    monkeypatch.setattr("oriented_det.data.rotates.random.choice", lambda seq: seq[0])
    monkeypatch.setattr("oriented_det.data.rotates.random.uniform", lambda _a, _b: 37.0)
    image = _solid(32, (255, 255, 255))
    box = RBox(16.0, 16.0, 8.0, 8.0, 0.0)
    sample = _sample(image, box)
    sample.labels = [13]
    _rotate_sample_copy(
        sample,
        prob=1.0,
        angle_range_deg=180.0,
        pad_val=114,
        rect_label_ids=[13],
    )
    assert seen == [90.0]
    seen.clear()
    sample.labels = [1]
    _rotate_sample_copy(
        sample,
        prob=1.0,
        angle_range_deg=180.0,
        pad_val=114,
        rect_label_ids=[13],
    )
    assert seen == [37.0]


def test_mosaic_rotates_each_tile_before_paste(monkeypatch):
    from oriented_det.data import rotates as rotates_mod
    from oriented_det.data.mosaic import _rotate_sample_copy

    calls = []
    real_rotate = rotates_mod.apply_rotate_to_image

    def _spy(image, degrees, fillcolor=(0, 0, 0)):
        calls.append((float(degrees), tuple(fillcolor), image.size))
        return real_rotate(image, degrees, fillcolor=fillcolor)

    monkeypatch.setattr("oriented_det.data.mosaic.apply_rotate_to_image", _spy)
    monkeypatch.setattr("oriented_det.data.rotates.random.uniform", lambda _a, _b: 45.0)
    image = _solid(32, (255, 255, 255))
    box = RBox(16.0, 8.0, 6.0, 3.0, 0.0)
    samples = [_sample(image.copy(), box) for _ in range(4)]
    rotated = _rotate_sample_copy(samples[0], prob=1.0, angle_range_deg=180.0, pad_val=114)
    assert rotated.image.getpixel((0, 0)) == (114, 114, 114)
    moved = rotated.rboxes[0]
    assert moved.cx != pytest.approx(box.cx) or moved.cy != pytest.approx(box.cy)

    calls.clear()
    apply_mosaic(
        samples[0],
        samples[1:],
        img_scale=(32, 32),
        center_xy=(32, 32),
        pad_val=114,
        rotate_prob=1.0,
        rotate_angle_range=180.0,
    )
    assert len(calls) == 4
    assert all(fill == (114, 114, 114) for _deg, fill, _size in calls)
    assert all(size == (32, 32) for _deg, _fill, size in calls)


def test_collate_skips_canvas_rotate_after_mosaic(tmp_path, monkeypatch):
    import oriented_det.runtime.collate as collate_mod

    samples = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        _solid(32, (255, 255, 255)).save(path)
        samples.append(_dota_sample(path))

    collate = create_collate_fn(
        {"ship": 1},
        normalize=False,
        resize_mode="fixed",
        resize_to=(32, 32),
        pad_size_divisor=1,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=True,
        random_rotate_prob=1.0,
        random_rotate_angle_range=180.0,
        enable_mosaic=True,
        mosaic_prob=1.0,
        mosaic_img_scale=(32, 32),
        mosaic_ratio_range=(1.0, 1.0),
        mosaic_max_cached_images=8,
    )
    canvas_rotates = {"n": 0}
    real = collate_mod.apply_random_train_rotate

    def _spy(image, rboxes, **kwargs):
        canvas_rotates["n"] += 1
        return real(image, rboxes, **kwargs)

    monkeypatch.setattr(collate_mod, "apply_random_train_rotate", _spy)
    for sample in samples[:4]:
        collate([sample])
    canvas_rotates["n"] = 0
    collate([samples[4]])
    assert collate._mosaic_cache.ready()
    assert canvas_rotates["n"] == 0
