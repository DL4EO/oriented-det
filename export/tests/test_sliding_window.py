"""Sliding-window tiling for ONNX detect_image."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export.preprocess import (  # noqa: E402
    overlap_pixels_from_meta,
    slice_size_from_meta,
    sliding_window_grid,
    uses_sliding_window,
)
from export.runtime import detect_image  # noqa: E402


def _meta(**prep_extra: object) -> dict:
    prep = {
        "resize_mode": "fixed",
        "target_size": [8, 8],
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
        "overlap_pixels": 0,
    }
    prep.update(prep_extra)
    return {
        "preprocess": prep,
        "postprocess": {
            "score_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "nms_class_agnostic": False,
            "max_detections": 100,
            "nms_backend": "python",
        },
        "class_names": ["plane"],
        "production": {},
    }


def _box(cx: float, cy: float, score: float = 0.5) -> dict:
    return {
        "cx": cx,
        "cy": cy,
        "w": 2.0,
        "h": 2.0,
        "angle_rad": 0.0,
        "score": score,
        "label": 1,
        "class_name": "plane",
    }


def test_sliding_window_grid_matches_pytorch() -> None:
    from oriented_det.runtime.inference import _sliding_window_grid

    cases = (
        (2000, 2000, 1024, 1024, 200),
        (1848, 1848, 1024, 1024, 200),
        (800, 800, 1024, 1024, 200),
        (1024, 1024, 1024, 1024, 200),
        (20, 6, 8, 8, 0),
        (32, 32, 16, 16, 0),
    )
    for image_h, image_w, slice_h, slice_w, overlap in cases:
        got = sliding_window_grid(
            image_h, image_w, slice_h, slice_w, overlap_pixels=overlap
        )
        ref = _sliding_window_grid(
            image_h, image_w, slice_h, slice_w, overlap_pixels=overlap
        )
        assert got == ref


def test_uses_sliding_window_respects_slice_and_resize_mode() -> None:
    fixed = _meta(window_size=[16, 16])
    assert uses_sliding_window(16, 16, fixed) is False
    assert uses_sliding_window(17, 16, fixed) is True
    assert slice_size_from_meta(fixed) == (16, 16)

    legacy = _meta()
    assert uses_sliding_window(8, 8, legacy) is False
    assert uses_sliding_window(9, 8, legacy) is True
    assert slice_size_from_meta(legacy) == (8, 8)

    pad = _meta(resize_mode="pad", window_size=[8, 8], target_size=[8, 8])
    assert uses_sliding_window(40, 40, pad) is False
    keep = _meta(resize_mode="keep_ratio", window_size=[8, 8])
    assert uses_sliding_window(40, 40, keep) is False

    assert overlap_pixels_from_meta({"production": {"overlap_pixels": 64}}) == 64
    assert overlap_pixels_from_meta({}) == 200


def test_detect_image_tiles_when_larger_than_window_and_scales_boxes(monkeypatch) -> None:
    from export import runtime

    calls = []

    def fake_detect_array(blob, *_args, **_kwargs):
        calls.append(blob.shape)
        return [_box(4.0, 4.0)]

    monkeypatch.setattr(runtime, "detect_array", fake_detect_array)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    dets = detect_image(rgb, "model.onnx", _meta(window_size=[16, 16]))
    assert calls == [(1, 3, 8, 8)] * 4
    centers = sorted((round(d["cx"], 5), round(d["cy"], 5)) for d in dets)
    assert centers == [(8.0, 8.0), (8.0, 24.0), (24.0, 8.0), (24.0, 24.0)]
    assert all(d["w"] == pytest.approx(4.0) and d["h"] == pytest.approx(4.0) for d in dets)


def test_detect_image_merges_overlap_copies(monkeypatch) -> None:
    from export import runtime

    calls = {"n": 0}

    def fake_detect_array(*_args, **_kwargs):
        # Windows at x0=0 and x0=4. Both describe the object at image x=6.
        cx = 6.0 if calls["n"] == 0 else 2.0
        score = 0.9 if calls["n"] == 0 else 0.4
        calls["n"] += 1
        return [_box(cx, 4.0, score=score)]

    monkeypatch.setattr(runtime, "detect_array", fake_detect_array)
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    dets = detect_image(rgb, "model.onnx", _meta(window_size=[8, 8], overlap_pixels=4))
    assert calls["n"] == 2
    assert len(dets) == 1
    assert dets[0]["cx"] == pytest.approx(6.0)
    assert dets[0]["cy"] == pytest.approx(4.0)
    assert dets[0]["score"] == pytest.approx(0.9)


def test_detect_image_drops_centroids_in_the_pad_band(monkeypatch) -> None:
    from export import runtime

    def fake_detect_array(*_args, **_kwargs):
        return [_box(7.0, 4.0, score=0.2), _box(3.0, 4.0, score=0.6)]

    monkeypatch.setattr(runtime, "detect_array", fake_detect_array)
    rgb = np.zeros((20, 6, 3), dtype=np.uint8)
    dets = detect_image(rgb, "model.onnx", _meta(window_size=[8, 8], overlap_pixels=0))
    assert len(dets) == 3
    assert all(d["cx"] == pytest.approx(3.0) for d in dets)
    assert sorted(round(d["cy"], 5) for d in dets) == [4.0, 12.0, 16.0]


def test_detect_image_single_forward_when_image_fits(monkeypatch) -> None:
    from export import runtime

    seen = []

    def fake_detect_array(*args, **kwargs):
        seen.append(kwargs)
        return [_box(1.0, 1.0)]

    monkeypatch.setattr(runtime, "detect_array", fake_detect_array)
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    detect_image(rgb, "model.onnx", _meta(window_size=[16, 16]))
    assert len(seen) == 1
    assert seen[0]["orig_h"] == 10
    assert seen[0]["orig_w"] == 10

    seen.clear()
    wide = np.zeros((30, 30, 3), dtype=np.uint8)
    detect_image(wide, "model.onnx", _meta(resize_mode="pad", window_size=[8, 8]))
    assert len(seen) == 1
    assert seen[0]["orig_h"] == 30


def test_build_export_meta_records_window_and_overlap(tmp_path: Path) -> None:
    from export.scripts.export_onnx import build_export_meta

    class _Cfg:
        num_classes = 1
        preprocessing = {
            "resize_mode": "fixed",
            "target_size": [1024, 1024],
            "window_size": [512],
            "normalize_mean": [0.1, 0.2, 0.3],
            "normalize_std": [0.4, 0.5, 0.6],
        }
        production = {"overlap_pixels": 128, "score_threshold": 0.2}
        evaluation = {}

    class _Model:
        final_nms_iou_threshold = 0.1
        max_detections_per_image = 100
        score_threshold = 0.05

    class _Wrap:
        max_candidates = 10

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"x")
    meta = build_export_meta(
        mode="backbone",
        config=_Cfg(),
        model=_Model(),
        wrapper=_Wrap(),
        height=1024,
        width=1024,
        opset=17,
        dynamic_batch=False,
        config_path="cfg.json",
        checkpoint_path="m.pth",
        class_names=["plane"],
        output_names=["out_0"],
        onnx_path=onnx_path,
    )
    assert meta["preprocess"]["window_size"] == [512, 512]
    assert meta["preprocess"]["overlap_pixels"] == 128
    assert slice_size_from_meta(meta) == (512, 512)
    assert uses_sliding_window(600, 600, meta) is True
    assert uses_sliding_window(512, 512, meta) is False
