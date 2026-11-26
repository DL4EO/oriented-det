"""Live DOTA multiscale inference helpers (Task 1 --resize-rates path)."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from oriented_det import RBox
from oriented_det.runtime.inference import (
    DOTA_MS_OVERLAP_PIXELS,
    DOTA_MS_RESIZE_RATES,
    map_detection_to_original_scale,
    parse_resize_rates,
    resize_image_for_ms_rate,
    run_inference_auto,
    run_inference_multiscale,
    run_inference_sliding_window,
)


def test_parse_resize_rates_none_and_csv():
    assert parse_resize_rates(None) is None
    assert parse_resize_rates("") is None
    assert parse_resize_rates("0.5,1.0,1.5") == [0.5, 1.0, 1.5]
    assert parse_resize_rates([0.5, 1.5]) == [0.5, 1.5]
    with pytest.raises(ValueError):
        parse_resize_rates("0.5,0")


def test_dota_ms_defaults():
    assert DOTA_MS_RESIZE_RATES == (0.5, 1.0, 1.5)
    assert DOTA_MS_OVERLAP_PIXELS == 500


def test_resize_image_for_ms_rate_bilinear_dims():
    image = Image.new("RGB", (1000, 800), 0)
    assert resize_image_for_ms_rate(image, 1.0).size == (1000, 800)
    assert resize_image_for_ms_rate(image, 0.5).size == (500, 400)
    assert resize_image_for_ms_rate(image, 1.5).size == (1500, 1200)


def test_map_detection_to_original_scale():
    det = {"rbox": RBox(100.0, 50.0, 20.0, 10.0, 0.1), "score": 0.9, "label": 1}
    mapped = map_detection_to_original_scale(det, 0.5)
    assert mapped["rbox"].cx == pytest.approx(200.0)
    assert mapped["rbox"].cy == pytest.approx(100.0)
    assert mapped["rbox"].width == pytest.approx(40.0)
    assert mapped["rbox"].height == pytest.approx(20.0)
    assert mapped["rbox"].angle == pytest.approx(det["rbox"].angle)
    assert map_detection_to_original_scale(det, 1.0)["rbox"].cx == pytest.approx(100.0)


def test_sliding_window_apply_nms_false_accepted():
    def fake(images):
        return [
            {
                "rboxes": [RBox(128.0, 128.0, 40.0, 20.0, 0.0)],
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]

    prep = {
        "resize_mode": "fixed",
        "target_size": (256, 256),
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
    }
    image = Image.new("RGB", (256, 256), 0)
    dets = run_inference_sliding_window(
        fake, image, "cpu", prep, score_threshold=0.5, nms_threshold=0.1,
        overlap_pixels=0, window_batch_size=1, apply_nms=False,
    )
    assert len(dets) == 1


def test_multiscale_maps_center_box_back_to_original():
    """A box at the center of a rate-0.5 canvas maps to the original center."""

    def fake(images):
        out = []
        for im in images:
            _, h, w = im.shape
            out.append(
                {
                    "rboxes": [RBox(w / 2.0, h / 2.0, 40.0, 20.0, 0.0)],
                    "scores": torch.tensor([0.99]),
                    "labels": torch.tensor([1]),
                }
            )
        return out

    prep = {
        "resize_mode": "fixed",
        "target_size": (256, 256),
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
    }
    # Fits one window at every rate so we exercise resize + map, not tiling math.
    image = Image.new("RGB", (200, 200), (8, 8, 8))
    dets = run_inference_multiscale(
        fake,
        image,
        "cpu",
        prep,
        resize_rates=[0.5],
        score_threshold=0.5,
        nms_threshold=0.1,
        overlap_pixels=0,
        window_batch_size=1,
    )
    assert len(dets) == 1
    assert dets[0]["rbox"].cx == pytest.approx(100.0, abs=1.0)
    assert dets[0]["rbox"].cy == pytest.approx(100.0, abs=1.0)


def test_run_inference_auto_resize_rates_dispatches():
    calls = {"n": 0}

    def fake(images):
        calls["n"] += len(images)
        return [
            {
                "rboxes": [RBox(32.0, 32.0, 16.0, 8.0, 0.0)],
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]

    prep = {
        "resize_mode": "fixed",
        "target_size": (64, 64),
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
    }
    image = Image.new("RGB", (64, 64), 0)
    dets = run_inference_auto(
        image=image,
        model=fake,
        device="cpu",
        preprocessing=prep,
        score_threshold=0.5,
        nms_threshold=0.1,
        resize_rates=[0.5, 1.0],
        overlap_pixels=0,
    )
    assert calls["n"] == 2  # one forward per rate (image fits canvas)
    assert len(dets) >= 1
