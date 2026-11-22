"""Letterbox, mosaic, and train-only scale/translate for the COCO YOLO recipe."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from PIL import Image

from oriented_det.data.dota import DOTAAnnotation, DOTASample
from oriented_det.data.scale_translate import warp_boxes
from oriented_det.geometry.rbox import RBox
from oriented_det.runtime.collate import create_collate_fn


def _box(cx: float, cy: float, width: float, height: float, class_name: str = "ship") -> DOTAAnnotation:
    x1, y1 = cx - width / 2.0, cy - height / 2.0
    x2, y2 = cx + width / 2.0, cy + height / 2.0
    return DOTAAnnotation.from_corners(
        [x1, y1, x2, y1, x2, y2, x1, y2],
        class_name,
    )


def _sample(path: Path, color: tuple[int, int, int], size: tuple[int, int], boxes) -> DOTASample:
    image = Image.new("RGB", size, color)
    image.save(path)
    return DOTASample(
        image_path=path,
        width=size[0],
        height=size[1],
        annotations=tuple(boxes),
    )


def _collate(**kwargs):
    params = dict(
        class_map={"ship": 1},
        augmentation=None,
        normalize=False,
        resize_mode="pad",
        resize_to=(640, 640),
        pad_size_divisor=32,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=False,
        enable_mosaic=False,
    )
    params.update(kwargs)
    return create_collate_fn(**params)


def test_pad_keeps_aspect_and_canvas(tmp_path: Path):
    sample = _sample(
        tmp_path / "wide.png",
        (128, 128, 128),
        (640, 480),
        [_box(200, 100, 100, 40)],
    )
    images, targets = _collate()([sample])
    assert images[0].shape == (3, 640, 640)
    pad = 114.0 / 255.0
    assert images[0][:, 0, 0].tolist() == pytest.approx([pad, pad, pad], abs=1e-4)
    row_mean = images[0].mean(dim=(0, 2))
    content_rows = int((row_mean > pad + 0.01).sum().item())
    assert content_rows == 480
    box = targets[0]["rboxes"][0]
    assert float(box[2]) == pytest.approx(100.0)
    assert float(box[3]) == pytest.approx(40.0)
    assert float(box[1]) == pytest.approx(180.0)


def test_mosaic_collate_stays_640(tmp_path: Path):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    samples = [
        _sample(tmp_path / f"tile_{i}.png", color, (320, 240), [_box(80, 60, 40, 20)])
        for i, color in enumerate(colors)
    ]
    plain = _collate()
    letterbox = [plain([sample])[0][0] for sample in samples]
    collate = _collate(
        enable_mosaic=True,
        mosaic_prob=1.0,
        mosaic_ratio_range=(1.0, 1.0),
        mosaic_img_scale=(640, 640),
        mosaic_max_cached_images=8,
    )
    saw = False
    for i in range(8):
        images, _ = collate([samples[i % len(samples)]])
        assert images[0].shape == (3, 640, 640)
        if not torch.equal(images[0], letterbox[i % len(samples)]):
            saw = True
    assert saw


def test_mosaic_pastes_the_rectangle_without_black_bars(tmp_path: Path, monkeypatch):
    import oriented_det.data.mosaic as mosaic_mod

    monkeypatch.setattr(mosaic_mod.random, "uniform", lambda _a, _b: 1.0)
    monkeypatch.setattr(
        mosaic_mod,
        "sample_mosaic_affine",
        lambda out_w, out_h, _scale_gain, _translate_gain: (1.0, out_w / 2.0, out_h / 2.0),
    )
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    samples = [
        _sample(
            tmp_path / f"wide_{i}.png",
            color,
            (200, 100),
            [_box(187.5, 87.5, 12.5, 6.25)],
        )
        for i, color in enumerate(colors)
    ]
    collate = _collate(
        enable_mosaic=True,
        mosaic_prob=1.0,
        mosaic_img_scale=(640, 640),
        mosaic_max_cached_images=8,
        random_scale=0.5,
        random_translate=0.1,
    )
    mosaics = 0
    for i in range(8):
        images, targets = collate([samples[i % len(samples)]])
        assert images[0].shape == (3, 640, 640)
        flat = (images[0].permute(1, 2, 0).reshape(-1, 3) * 255).round().to(torch.int)
        black = int((flat.sum(dim=1) == 0).sum().item())
        colors_seen = {tuple(int(v) for v in pixel) for pixel in flat.unique(dim=0).tolist()}
        photo = [color for color in colors if color in colors_seen]
        if len(photo) >= 2:
            mosaics += 1
            assert black == 0
            assert targets[0]["rboxes"].shape[0] >= 1
            widths = targets[0]["rboxes"][:, 2]
            assert torch.any(torch.isclose(widths, torch.tensor(40.0), atol=0.6))
    assert mosaics >= 1


def test_scale_translate_keeps_inside_box_and_drops_one_that_leaves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "oriented_det.runtime.collate.sample_scale_translate",
        lambda width, height, scale_gain, translate_gain: (1.0, 40.0, 0.0),
    )
    sample = _sample(
        tmp_path / "canvas.png",
        (20, 40, 60),
        (640, 640),
        [
            _box(320, 320, 40, 40),
            _box(620, 320, 20, 20),
        ],
    )
    images, targets = _collate(random_scale=0.5, random_translate=0.1)([sample])
    assert images[0].shape == (3, 640, 640)
    kept = targets[0]["rboxes"]
    assert kept.shape[0] == 1
    assert float(kept[0, 0]) == pytest.approx(360.0)
    assert float(kept[0, 1]) == pytest.approx(320.0)
    assert float(kept[0, 2]) == pytest.approx(40.0)

    direct_kept, direct_labels = warp_boxes(
        [RBox(320, 320, 40, 40, 0.0), RBox(620, 320, 20, 20, 0.0)],
        640,
        640,
        1.0,
        40.0,
        0.0,
        0.5,
        [1, 1],
    )
    assert len(direct_kept) == 1
    assert direct_labels == [1]
    assert direct_kept[0].cx == pytest.approx(360.0)


def _paint(size: tuple[int, int], box: RBox) -> Image.Image:
    image = Image.new("RGB", size, (0, 0, 0))
    pixels = image.load()
    x1 = int(round(box.cx - box.width / 2.0))
    y1 = int(round(box.cy - box.height / 2.0))
    x2 = int(round(box.cx + box.width / 2.0))
    y2 = int(round(box.cy + box.height / 2.0))
    for y in range(max(0, y1), min(size[1], y2)):
        for x in range(max(0, x1), min(size[0], x2)):
            pixels[x, y] = (255, 0, 0)
    return image


def _bright_pixels_follow_box(tensor: torch.Tensor, box: torch.Tensor) -> None:
    red = tensor[0]
    ys, xs = torch.nonzero(red > 0.75, as_tuple=True)
    assert xs.numel() > 0
    cx, cy, width, height = [float(v) for v in box[:4]]
    # Pixel i covers [i, i+1). Its center is i + 0.5, the same frame as the box.
    err_x = float(xs.float().mean() + 0.5 - cx)
    err_y = float(ys.float().mean() + 0.5 - cy)
    assert err_x == pytest.approx(0.0, abs=0.5)
    assert err_y == pytest.approx(0.0, abs=0.5)
    x1, y1 = cx - width / 2.0, cy - height / 2.0
    x2, y2 = cx + width / 2.0, cy + height / 2.0
    centers_x = xs.float() + 0.5
    centers_y = ys.float() + 0.5
    inside = (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
    assert float(inside.float().mean()) == pytest.approx(1.0)


def test_box_stays_on_scaled_pixels(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "oriented_det.runtime.collate.sample_scale_translate",
        lambda width, height, scale_gain, translate_gain: (1.25, 18.5, -11.0),
    )
    box = RBox(180, 400, 90, 40, 0.0)
    image = _paint((640, 640), box)
    path = tmp_path / "painted.png"
    image.save(path)
    sample = DOTASample(
        image_path=path,
        width=640,
        height=640,
        annotations=(_box(box.cx, box.cy, box.width, box.height),),
    )
    images, targets = _collate(random_scale=0.5, random_translate=0.1)([sample])
    assert targets[0]["rboxes"].shape[0] == 1
    _bright_pixels_follow_box(images[0], targets[0]["rboxes"][0])

    wide = _paint((640, 480), RBox(200, 100, 100, 40, 0.0))
    wide_path = tmp_path / "wide_painted.png"
    wide.save(wide_path)
    wide_sample = DOTASample(
        image_path=wide_path,
        width=640,
        height=480,
        annotations=(_box(200, 100, 100, 40),),
    )
    monkeypatch.setattr(
        "oriented_det.runtime.collate.sample_scale_translate",
        lambda width, height, scale_gain, translate_gain: (1.4, -25.0, 30.0),
    )
    images, targets = _collate(random_scale=0.5, random_translate=0.1)([wide_sample])
    assert targets[0]["rboxes"].shape[0] == 1
    _bright_pixels_follow_box(images[0], targets[0]["rboxes"][0])


def test_validation_collate_does_not_scale(tmp_path: Path):
    sample = _sample(
        tmp_path / "val.png",
        (10, 10, 10),
        (640, 640),
        [_box(100, 100, 20, 20)],
    )
    _, targets = _collate()([sample])
    box = targets[0]["rboxes"][0]
    assert float(box[0]) == pytest.approx(100.0)
    assert float(box[1]) == pytest.approx(100.0)


def test_hsv_is_optional_and_keeps_image_size():
    pytest.importorskip("albumentations")
    from oriented_det.data.transforms import create_albumentations_augmentation

    plain = create_albumentations_augmentation()
    plain_names = [type(step).__name__ for step in plain.transform.transforms]
    assert "RandomHSVGain" not in plain_names

    aug = create_albumentations_augmentation(
        p_gamma=0,
        p_noise=0,
        p_blur=0,
        p_clahe=0,
        p_hsv=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    names = [type(step).__name__ for step in aug.transform.transforms]
    assert "RandomHSVGain" in names
    assert "GaussNoise" not in names
    out = aug(Image.new("RGB", (32, 24), (128, 64, 32)))
    assert out.size == (32, 24)


def test_hsv_jitter_is_multiplicative_gain():
    """Ultralytics ``augment_hsv`` semantics: gray stays gray, value scales and clips, hue wraps."""
    pytest.importorskip("albumentations")
    import numpy as np

    from oriented_det.data.transforms import RandomHSVGain

    t = RandomHSVGain(hgain=0.015, sgain=0.7, vgain=0.4, p=1.0)
    gray = np.full((4, 4, 3), 128, np.uint8)
    assert (t.apply(gray, gains=np.array([1.0, 1.7, 1.0])) == 128).all()

    v200 = np.full((4, 4, 3), 200, np.uint8)
    assert int(t.apply(v200, gains=np.array([1.0, 1.0, 1.4]))[0, 0, 0]) == 255
    assert int(t.apply(v200, gains=np.array([1.0, 1.0, 0.6]))[0, 0, 0]) == 120

    # A gain of 1.0 on every channel is the bare uint8 RGB→HSV→RGB round trip
    # (no additive shift). That round trip itself is lossy by a few levels.
    import cv2

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    round_trip = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), cv2.COLOR_HSV2RGB)
    assert (t.apply(img, gains=np.array([1.0, 1.0, 1.0])) == round_trip).all()

    # Drawn gains stay inside 1 ± fraction.
    for _ in range(20):
        g = t.get_params()["gains"]
        assert 0.985 <= g[0] <= 1.015
        assert 0.3 <= g[1] <= 1.7
        assert 0.6 <= g[2] <= 1.4
