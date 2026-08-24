"""SavedModel export/reload without oriented-det custom objects (requires TensorFlow)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

tf = pytest.importorskip("tensorflow")

from export.tf_savedmodel import (  # noqa: E402
    PreNmsCoreLayer,
    build_detect_model_from_pre_nms,
    call_saved_model,
    export_keras_saved_model,
)


def _pre_nms(images: tf.Tensor):
    z = tf.reduce_sum(images) * 0.0
    boxes = tf.constant(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [10.1, 10.0, 8.0, 4.0, 0.0],
            [40.0, 40.0, 6.0, 3.0, 0.0],
        ],
        dtype=tf.float32,
    ) + z
    scores = tf.constant([0.95, 0.90, 0.80], dtype=tf.float32) + z
    labels = tf.constant([1, 1, 2], dtype=tf.int32)
    count = tf.constant(3, dtype=tf.int32)
    return boxes, scores, labels, count


def test_savedmodel_reload_with_vanilla_tf(tmp_path: Path) -> None:
    finalize_kwargs = {
        "nms_class_agnostic": True,
        "final_nms_iou_threshold": 0.5,
        "max_detections_per_image": 8,
        "score_threshold": 0.05,
        "per_class_score_threshold": None,
        "class_id_to_name": {1: "a", 2: "b"},
        "max_output_slots": 8,
    }
    model = build_detect_model_from_pre_nms(
        _pre_nms, height=8, width=8, finalize_kwargs=finalize_kwargs
    )
    sm_dir = tmp_path / "saved_model"
    export_keras_saved_model(model, sm_dir)
    assert (sm_dir / "saved_model.pb").is_file()

    x = tf.zeros([1, 3, 8, 8], dtype=tf.float32)
    out_k = model(x, training=False)
    if isinstance(out_k, dict):
        det_k, num_k = out_k["detections"], out_k["num_detections"]
    else:
        det_k, num_k = out_k

    loaded = tf.saved_model.load(str(sm_dir))
    # Vanilla TF only — no oriented-det helper required.
    served = loaded.serve(x)
    det_v, num_v = served["detections"], served["num_detections"]
    det, num = call_saved_model(loaded, x)
    n_loaded = int(np.array(num).reshape(-1)[0])
    n_keras = int(np.array(num_k).reshape(-1)[0])
    n_vanilla = int(np.array(num_v).reshape(-1)[0])
    assert n_loaded == n_keras == n_vanilla == 2
    np.testing.assert_allclose(
        np.array(det)[:2],
        np.array(det_k)[:2],
        atol=1e-5,
    )
    np.testing.assert_allclose(np.array(det_v)[:2], np.array(det)[:2], atol=1e-5)
    # Duplicate box suppressed; two classes remain.
    labels = np.array(det)[: int(np.array(num).reshape(-1)[0]), 6]
    assert set(labels.astype(int).tolist()) == {1, 2}


def test_wrap_pre_nms_savedmodel_core(tmp_path: Path) -> None:
    """Nested SavedModel core + TF NMS, same path as onnx2tf wrap."""
    images = tf.keras.Input(shape=(3, 8, 8), batch_size=1, name="images")

    class _Core(tf.keras.layers.Layer):
        def call(self, x):
            z = tf.reduce_sum(x) * 0.0
            return {
                "pre_nms_boxes": tf.zeros([4, 5], dtype=tf.float32) + z,
                "pre_nms_scores": tf.zeros([4], dtype=tf.float32) + z,
                "pre_nms_labels": tf.zeros([4], dtype=tf.int32),
                "pre_nms_count": tf.constant(0, dtype=tf.int32),
            }

    core_model = tf.keras.Model(images, _Core(name="core")(images))
    core_dir = tmp_path / "core"
    export_keras_saved_model(core_model, core_dir)
    layer = PreNmsCoreLayer(core_dir, name="onnx2tf_core")
    model = build_detect_model_from_pre_nms(
        layer,
        height=8,
        width=8,
        finalize_kwargs={
            "nms_class_agnostic": True,
            "final_nms_iou_threshold": 0.5,
            "max_detections_per_image": 8,
            "score_threshold": 0.05,
            "per_class_score_threshold": None,
            "class_id_to_name": {1: "a"},
            "max_output_slots": 8,
        },
    )
    sm_dir = tmp_path / "saved_model"
    export_keras_saved_model(model, sm_dir)
    loaded = tf.saved_model.load(str(sm_dir))
    out = loaded.serve(tf.zeros([1, 3, 8, 8], dtype=tf.float32))
    assert int(np.array(out["num_detections"]).reshape(-1)[0]) == 0
    assert tuple(np.array(out["detections"]).shape) == (8, 7)
