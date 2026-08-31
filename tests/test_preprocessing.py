"""Tests for spatial preprocessing (pad / keep_ratio / crop / fixed)."""

from PIL import Image

from oriented_det.data.preprocessing import (
    apply_crop_mode,
    apply_keep_ratio_mode,
    apply_pad_mode,
    build_spatial_meta_from_dims,
    parse_canvas_size,
    remap_detections_to_original,
)
from oriented_det.geometry.rbox import RBox


def test_parse_canvas_size_square():
    assert parse_canvas_size("pad", [1024]) == (1024, 1024)
    assert parse_canvas_size("crop", [800, 600]) == (800, 600)


def test_pad_preserves_aspect_ratio():
    image = Image.new("RGB", (2000, 1000), color=(128, 64, 32))
    rb = RBox(1000, 500, 100, 50, 0.0)
    result = apply_pad_mode(image, [rb], [1024, 1024])
    out_w, out_h = result.image.size
    assert (out_w, out_h) == (1024, 1024)
    # Longer edge scaled to 1024 → content is 1024×512, centered vertically.
    assert result.meta.content_size == (512, 1024)
    assert abs(result.meta.scale - 1024 / 2000) < 1e-5


def test_crop_keeps_native_resolution():
    image = Image.new("RGB", (1500, 1200), color=(10, 20, 30))
    rb = RBox(800, 600, 40, 20, 0.1)
    result = apply_crop_mode(image, [rb], [1024, 1024], random_crop=False)
    assert result.image.size == (1024, 1024)
    assert result.meta.scale_x == 1.0 and result.meta.scale_y == 1.0
    # Center crop: offset (238, 88)
    assert result.meta.crop_left == (1500 - 1024) // 2
    assert result.meta.crop_top == (1200 - 1024) // 2


def test_crop_pads_when_smaller_than_target():
    image = Image.new("RGB", (800, 600), color=(1, 2, 3))
    result = apply_crop_mode(image, [], [1024, 1024], random_crop=False)
    assert result.image.size == (1024, 1024)


def test_pad_preserves_pixel_aspect_ratio():
    """Uniform scale: scaled_w / scaled_h == orig_w / orig_h."""
    image = Image.new("RGB", (1600, 900), color=(1, 1, 1))
    result = apply_pad_mode(image, [], [1024, 768])
    sh, sw = result.meta.content_size
    assert abs((sw / sh) - (1600 / 900)) < 0.02


def test_crop_never_resizes_pixels():
    """Crop only pads/crops; no bilinear resize → scale stays 1."""
    image = Image.new("RGB", (2000, 1500), color=(1, 1, 1))
    result = apply_crop_mode(image, [], [1024, 1024], random_crop=False)
    assert result.meta.scale_x == 1.0
    assert result.meta.scale_y == 1.0
    # Content aspect matches original where visible (center 1024×1024 window).
    assert result.meta.crop_left == (2000 - 1024) // 2
    assert result.meta.crop_top == (1500 - 1024) // 2


def test_remap_pad_detections_roundtrip():
    meta = build_spatial_meta_from_dims("pad", 2000, 1000, [1024, 1024])
    # Box at scaled-image center (512, 256) with pad_top=256 → maps to original center.
    dets = [{"rbox": RBox(512, 512, 10, 5, 0.0), "score": 0.9, "label": 1}]
    remapped = remap_detections_to_original(dets, meta)
    r = remapped[0]["rbox"]
    assert abs(r.cx - 1000) < 2
    assert abs(r.cy - 500) < 2


def test_keep_ratio_scales_without_square_pad():
    image = Image.new("RGB", (1166, 753), color=(128, 64, 32))
    rb = RBox(583, 376, 200, 40, 0.0)
    result = apply_keep_ratio_mode(image, [rb], [800, 800])
    out_w, out_h = result.image.size
    assert out_w == 800
    assert out_h == int(round(753 * (800 / 1166)))
    assert result.meta.mode == "keep_ratio"
    assert result.meta.pad_left == 0 and result.meta.pad_top == 0
    assert abs(result.meta.scale - 800 / 1166) < 1e-5


def test_remap_keep_ratio_detections_roundtrip():
    meta = build_spatial_meta_from_dims("keep_ratio", 1166, 753, [800, 800])
    scale = 800 / 1166
    dets = [{"rbox": RBox(583 * scale, 376 * scale, 10, 5, 0.0), "score": 0.9, "label": 1}]
    remapped = remap_detections_to_original(dets, meta)
    r = remapped[0]["rbox"]
    assert abs(r.cx - 583) < 2
    assert abs(r.cy - 376) < 2


def test_pad_image_tensors_to_batch_hw_keep_ratio_shapes():
    import torch
    from oriented_det.runtime.collate import pad_image_tensors_to_batch_hw

    short = torch.ones(3, 480, 800)
    tall = torch.ones(3, 576, 800) * 2
    out = pad_image_tensors_to_batch_hw([short, tall])
    assert out[0].shape == (3, 576, 800)
    assert out[1].shape == (3, 576, 800)
    assert torch.equal(out[0][:, :480, :], short)
    assert torch.count_nonzero(out[0][:, 480:, :]) == 0
    assert torch.equal(out[1], tall)
    assert pad_image_tensors_to_batch_hw([short])[0].shape == (3, 480, 800)
