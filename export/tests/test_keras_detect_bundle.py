"""Keras detect-bundle save/reload (requires TensorFlow + a tiny ONNX stub)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

tf = pytest.importorskip("tensorflow")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from export.postprocess import normalize_class_id_to_name  # noqa: E402
from export.tf_serving_model import (  # noqa: E402
    BUNDLED_ONNX_NAME,
    FasterRCNNDetectLayer,
    load_keras_detect_model,
    save_keras_detect_bundle,
)


def _tiny_identity_onnx(path: Path, height: int = 8, width: int = 8) -> None:
    """Write a minimal ONNX that yields empty pre-NMS outputs (count=0)."""
    import torch
    import torch.nn as nn

    class Stub(nn.Module):
        def forward(self, images: torch.Tensor):
            # Keep a data dependency on ``images`` so ONNX retains the input.
            zero = images.sum() * 0.0
            p = 4
            boxes = torch.zeros(p, 5, dtype=torch.float32) + zero
            scores = torch.zeros(p, dtype=torch.float32) + zero
            labels = torch.zeros(p, dtype=torch.int64)
            count = torch.zeros((), dtype=torch.int64)
            return boxes, scores, labels, count

    stub = Stub().eval()
    torch.onnx.export(
        stub,
        torch.zeros(1, 3, height, width, dtype=torch.float32),
        str(path),
        input_names=["images"],
        output_names=["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
        opset_version=17,
        do_constant_folding=False,
    )


def test_normalize_class_id_to_name_string_keys() -> None:
    assert normalize_class_id_to_name({"1": "a", 2: "b"}) == {1: "a", 2: "b"}


def test_keras_bundle_save_reload_and_relocate(tmp_path: Path) -> None:
    h, w = 8, 8
    src_onnx = tmp_path / "src.onnx"
    _tiny_identity_onnx(src_onnx, height=h, width=w)

    finalize_kwargs = {
        "nms_class_agnostic": True,
        "final_nms_iou_threshold": 0.5,
        "max_detections_per_image": 8,
        "final_nms_use_cpu": True,
        "score_threshold": 0.05,
        "per_class_score_threshold": {"a": 0.2},
        "class_id_to_name": {1: "a", 2: "b"},
        "max_output_slots": 8,
    }
    inputs = tf.keras.Input(shape=(3, h, w), batch_size=1, name="images")
    layer = FasterRCNNDetectLayer(
        onnx_path=BUNDLED_ONNX_NAME,
        ort_output_names=[
            "pre_nms_boxes",
            "pre_nms_scores",
            "pre_nms_labels",
            "pre_nms_count",
        ],
        finalize_kwargs=finalize_kwargs,
        max_output_slots=8,
    )
    out = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[out["detections"], out["num_detections"]])

    bundle = tmp_path / "bundle"
    meta = {
        "mode": "faster_rcnn_pre_nms",
        "input": {"shape": [1, 3, h, w]},
        "output_names": [
            "pre_nms_boxes",
            "pre_nms_scores",
            "pre_nms_labels",
            "pre_nms_count",
        ],
        "class_names": ["a", "b"],
        "production": {"score_threshold": 0.05},
    }
    save_keras_detect_bundle(model, bundle, meta, onnx_source=src_onnx)
    assert (bundle / "keras_model.keras").is_file()
    assert (bundle / BUNDLED_ONNX_NAME).is_file()
    assert (bundle / "export_meta.json").is_file()

    loaded = load_keras_detect_model(bundle)
    det_layer = next(l for l in loaded.layers if isinstance(l, FasterRCNNDetectLayer))
    assert det_layer.onnx_path == BUNDLED_ONNX_NAME
    assert Path(det_layer._resolved_onnx_path).is_file()
    assert det_layer.finalize_kwargs["class_id_to_name"] == {1: "a", 2: "b"}

    x = tf.zeros([1, 3, h, w], dtype=tf.float32)
    detections, num = loaded(x, training=False)
    assert int(num.numpy()) == 0
    assert tuple(detections.shape) == (8, 7)

    # Relocate the whole bundle directory; absolute paths must not be required.
    moved = tmp_path / "moved_bundle"
    shutil.move(str(bundle), str(moved))
    reloaded = load_keras_detect_model(moved / "keras_model.keras")
    detections2, num2 = reloaded(x, training=False)
    assert int(num2.numpy()) == 0
    assert np.allclose(detections.numpy(), detections2.numpy())
