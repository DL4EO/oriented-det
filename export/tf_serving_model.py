"""Keras-serializable Faster R-CNN detect pipeline (ORT core + Python rotated NMS)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from export.postprocess import finalize_detections_numpy, ort_pre_nms_to_detections


@tf.keras.utils.register_keras_serializable(package="oriented_det_export")
class FasterRCNNDetectLayer(tf.keras.layers.Layer):
    """Single end-to-end detect: ONNX Runtime pre-NMS + exact CPU rotated NMS."""

    def __init__(
        self,
        onnx_path: str,
        ort_output_names: List[str],
        finalize_kwargs: Dict[str, Any],
        max_output_slots: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.onnx_path = str(onnx_path)
        self.ort_output_names = list(ort_output_names)
        self.finalize_kwargs = dict(finalize_kwargs)
        self.max_output_slots = int(max_output_slots)

    def call(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        detections, num_detections = tf.numpy_function(
            self._infer_numpy,
            [images],
            (tf.float32, tf.int32),
        )
        detections.set_shape([self.max_output_slots, 7])
        num_detections.set_shape([])
        return {"detections": detections, "num_detections": num_detections}

    def _infer_numpy(self, images: np.ndarray) -> tuple[np.ndarray, np.int32]:
        detections, num = ort_pre_nms_to_detections(
            images,
            self.onnx_path,
            self.ort_output_names,
            self.finalize_kwargs,
        )
        return detections.astype(np.float32), np.int32(num)

    def get_config(self) -> Dict[str, Any]:
        return {
            "onnx_path": self.onnx_path,
            "ort_output_names": self.ort_output_names,
            "finalize_kwargs": self.finalize_kwargs,
            "max_output_slots": self.max_output_slots,
        }


def build_keras_detect_model(
    onnx_path: Path,
    meta: Dict[str, Any],
    finalize_kwargs: Dict[str, Any],
) -> tf.keras.Model:
    max_out = int(finalize_kwargs["max_output_slots"])
    inputs = tf.keras.Input(shape=(3, None, None), batch_size=1, name="images")
    # Fixed H,W enforced at call time; channel-first matches export convention.
    out = FasterRCNNDetectLayer(
        onnx_path=str(onnx_path.resolve()),
        ort_output_names=list(meta.get("output_names") or []),
        finalize_kwargs=finalize_kwargs,
        max_output_slots=max_out,
        name="faster_rcnn_detect",
    )(inputs)
    return tf.keras.Model(inputs=inputs, outputs=[out["detections"], out["num_detections"]])


def save_keras_detect_bundle(
    model: tf.keras.Model,
    output_path: Path,
    meta: Dict[str, Any],
) -> Path:
    """Save ``keras_model.keras`` + ``export_meta.json`` (load with :func:`load_keras_detect_model`)."""
    output_path.mkdir(parents=True, exist_ok=True)
    keras_path = output_path / "keras_model.keras"
    model.save(keras_path)
    (output_path / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return keras_path


def load_keras_detect_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path,
        custom_objects={"FasterRCNNDetectLayer": FasterRCNNDetectLayer},
    )
