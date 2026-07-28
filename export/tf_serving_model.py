"""Keras-serializable detect pipeline (ORT core + Python rotated NMS)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from export.postprocess import (
    normalize_finalize_kwargs,
    ort_pre_nms_to_detections,
)

BUNDLED_ONNX_NAME = "model.onnx"


def resolve_bundled_onnx(bundle_dir: Path, onnx_path: str) -> Path:
    """Resolve ONNX path relative to a detect-bundle directory."""
    bundle_dir = Path(bundle_dir)
    p = Path(onnx_path)
    if p.is_file():
        return p.resolve()
    for candidate in (bundle_dir / p.name, bundle_dir / BUNDLED_ONNX_NAME):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"ONNX not found for detect bundle (tried {onnx_path!r} under {bundle_dir}). "
        f"Expected {BUNDLED_ONNX_NAME} next to keras_model.keras."
    )


@tf.keras.utils.register_keras_serializable(package="oriented_det_export")
class FasterRCNNDetectLayer(tf.keras.layers.Layer):
    """End-to-end detect: ONNX Runtime pre-NMS + exact CPU rotated NMS.

    ``onnx_path`` should be the basename ``model.onnx`` (bundled next to the Keras
    file). Call :func:`load_keras_detect_model` so the path is resolved on load.
    """

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
        self.finalize_kwargs = normalize_finalize_kwargs(dict(finalize_kwargs))
        self.max_output_slots = int(max_output_slots)
        self._resolved_onnx_path: Optional[str] = None

    def call(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        detections, num_detections = tf.numpy_function(
            self._infer_numpy,
            [images],
            (tf.float32, tf.int32),
        )
        detections.set_shape([self.max_output_slots, 7])
        num_detections.set_shape([])
        return {"detections": detections, "num_detections": num_detections}

    def _onnx_for_infer(self) -> str:
        if self._resolved_onnx_path and Path(self._resolved_onnx_path).is_file():
            return self._resolved_onnx_path
        p = Path(self.onnx_path)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(
            f"ONNX path {self.onnx_path!r} is not a readable file. "
            "Load the bundle with export.tf_serving_model.load_keras_detect_model() "
            f"so {BUNDLED_ONNX_NAME} resolves next to keras_model.keras."
        )

    def _infer_numpy(self, images: np.ndarray) -> tuple[np.ndarray, np.int32]:
        detections, num = ort_pre_nms_to_detections(
            images,
            self._onnx_for_infer(),
            self.ort_output_names,
            self.finalize_kwargs,
        )
        return detections.astype(np.float32), np.int32(num)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        # Persist basename only so the bundle stays relocatable.
        cfg.update(
            {
                "onnx_path": Path(self.onnx_path).name,
                "ort_output_names": self.ort_output_names,
                "finalize_kwargs": self.finalize_kwargs,
                "max_output_slots": self.max_output_slots,
            }
        )
        return cfg


def build_keras_detect_model(
    onnx_path: Path,
    meta: Dict[str, Any],
    finalize_kwargs: Dict[str, Any],
) -> tf.keras.Model:
    max_out = int(finalize_kwargs["max_output_slots"])
    inputs = tf.keras.Input(shape=(3, None, None), batch_size=1, name="images")
    out = FasterRCNNDetectLayer(
        onnx_path=str(Path(onnx_path).name),
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
    *,
    onnx_source: Optional[Path] = None,
) -> Path:
    """Save ``keras_model.keras``, ``model.onnx``, and ``export_meta.json``."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    bundled_onnx = output_path / BUNDLED_ONNX_NAME
    if onnx_source is not None:
        src = Path(onnx_source)
        if not src.is_file():
            raise FileNotFoundError(f"ONNX source not found: {src}")
        if src.resolve() != bundled_onnx.resolve():
            shutil.copy2(src, bundled_onnx)
    elif not bundled_onnx.is_file():
        raise FileNotFoundError(
            f"Missing {bundled_onnx}; pass onnx_source= to copy the ONNX into the bundle."
        )

    for layer in model.layers:
        if isinstance(layer, FasterRCNNDetectLayer):
            layer.onnx_path = BUNDLED_ONNX_NAME
            layer._resolved_onnx_path = str(bundled_onnx.resolve())

    meta = dict(meta)
    meta["onnx_path"] = BUNDLED_ONNX_NAME
    meta["core_backend"] = meta.get("core_backend") or "onnxruntime_keras"

    keras_path = output_path / "keras_model.keras"
    model.save(keras_path)
    (output_path / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return keras_path


def load_keras_detect_model(path: Union[str, Path]) -> tf.keras.Model:
    """Load a detect bundle; resolve bundled ``model.onnx`` next to the Keras file."""
    path = Path(path)
    keras_path = path if path.suffix == ".keras" else path / "keras_model.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing Keras model: {keras_path}")
    bundle_dir = keras_path.parent
    model = tf.keras.models.load_model(
        keras_path,
        custom_objects={"FasterRCNNDetectLayer": FasterRCNNDetectLayer},
    )
    for layer in model.layers:
        if isinstance(layer, FasterRCNNDetectLayer):
            resolved = resolve_bundled_onnx(bundle_dir, layer.onnx_path)
            layer.onnx_path = BUNDLED_ONNX_NAME
            layer._resolved_onnx_path = str(resolved)
            layer.finalize_kwargs = normalize_finalize_kwargs(layer.finalize_kwargs)
    return model
