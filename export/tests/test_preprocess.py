"""Numpy preprocess matches torchvision ToTensor + Normalize."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export.preprocess import preprocess_from_meta, preprocess_rgb, scale_obb_to_original
from export.runtime import detections_to_records
from export.scripts.export_onnx import build_export_meta


def test_preprocess_shape_and_bgr_swap() -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    blob, h, w = preprocess_rgb(rgb, 8, 16, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    assert blob.shape == (1, 3, 8, 16)
    assert (h, w) == (20, 30)
    bgr = rgb[:, :, ::-1]
    blob_bgr, _, _ = preprocess_rgb(
        bgr, 8, 16, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], bgr=True
    )
    np.testing.assert_allclose(blob, blob_bgr, atol=1e-6)


def test_preprocess_matches_torchvision() -> None:
    pytest.importorskip("torch")
    import torchvision.transforms as T

    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 256, size=(40, 50, 3), dtype=np.uint8)
    mean = [0.464546, 0.437698, 0.376678]
    std = [0.280621, 0.250267, 0.235597]
    blob, _, _ = preprocess_rgb(rgb, 32, 32, mean=mean, std=std)

    pil = Image.fromarray(rgb, mode="RGB").resize((32, 32), Image.BILINEAR)
    ref = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])(pil).unsqueeze(0).numpy()
    np.testing.assert_allclose(blob, ref, atol=1e-5, rtol=1e-5)


def test_scale_obb_and_records() -> None:
    boxes = np.array([[512.0, 256.0, 64.0, 32.0, 0.1]], dtype=np.float32)
    scaled = scale_obb_to_original(boxes, orig_w=2048, orig_h=1024, canvas_w=1024, canvas_h=1024)
    np.testing.assert_allclose(scaled[0, :4], [1024.0, 256.0, 128.0, 32.0], atol=1e-5)
    padded = np.zeros((2, 7), dtype=np.float32)
    padded[0] = [1, 2, 3, 4, 0.2, 0.9, 3]
    recs = detections_to_records(padded, 1, ["a", "b", "Helicopter"])
    assert recs[0]["class_name"] == "Helicopter"
    assert recs[0]["score"] == pytest.approx(0.9)


def test_preprocess_from_meta_reads_nested_keys() -> None:
    meta = {
        "preprocess": {
            "target_size": [16, 24],
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
        }
    }
    rgb = np.ones((8, 8, 3), dtype=np.uint8) * 255
    blob, h, w = preprocess_from_meta(rgb, meta)
    assert blob.shape == (1, 3, 16, 24)
    assert (h, w) == (8, 8)
    np.testing.assert_allclose(blob, 1.0, atol=1e-5)


def test_build_export_meta_includes_preprocess_postprocess(tmp_path: Path) -> None:
    class _Cfg:
        num_classes = 6
        preprocessing = {
            "resize_mode": "fixed",
            "target_size": [1024, 1024],
            "normalize_mean": [0.1, 0.2, 0.3],
            "normalize_std": [0.4, 0.5, 0.6],
            "pad_size_divisor": 32,
        }
        production = {
            "score_threshold": 0.2,
            "final_nms_iou_threshold": 0.5,
            "nms_class_agnostic": True,
            "max_detections_per_image": 500,
            "final_nms_use_cpu": True,
            "inference_pre_nms_score_threshold": 0.05,
        }
        evaluation = {"score_threshold": 0.2}

    class _Model:
        final_nms_iou_threshold = 0.1
        max_detections_per_image = 2000
        score_threshold = 0.05
        fpn_strides = [8, 16]
        nms_pre = 2000
        min_bbox_size = 0.0

    class _Wrap:
        max_candidates = 4000

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"x")
    meta = build_export_meta(
        mode="rotated_fcos_pre_nms",
        config=_Cfg(),
        model=_Model(),
        wrapper=_Wrap(),
        height=1024,
        width=1024,
        opset=17,
        dynamic_batch=False,
        config_path="cfg.json",
        checkpoint_path="m.pth",
        class_names=["a"],
        output_names=["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
        onnx_path=onnx_path,
    )
    assert meta["preprocess"]["normalize_mean"] == [0.1, 0.2, 0.3]
    assert meta["preprocess"]["window_size"] is None
    assert meta["preprocess"]["overlap_pixels"] == 200
    assert meta["postprocess"]["score_threshold"] == 0.2
    assert meta["postprocess"]["nms_class_agnostic"] is True
    assert meta["postprocess"]["nms_backend"] == "python"
    assert meta["max_pre_nms_candidates"] == 4000
    assert meta["input"]["normalized"] is True
