"""Build a TensorFlow SavedModel detect graph (ONNX core + TF rotated NMS).

Reload with vanilla TensorFlow::

    sm = tf.saved_model.load("./odet_export/saved_model")
    detections, num = call_saved_model(sm, images)

Export still needs oriented-det + onnx2tf. Inference does not.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import tensorflow as tf

from export.tf_nms import DetectPostprocessLayer

SAVED_MODEL_DIRNAME = "saved_model"


def _as_dict(raw: Any) -> Dict[str, tf.Tensor]:
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(i): v for i, v in enumerate(raw)}
    raise TypeError(f"Unexpected pre-NMS output type: {type(raw)!r}")


def unpack_pre_nms_outputs(raw: Any) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Pick ``(boxes, scores, labels, count)`` from a SavedModel/Keras output."""
    items = _as_dict(raw)
    boxes = scores = labels = count = None
    for key, val in items.items():
        name = key.lower()
        tensor = val
        rank = tensor.shape.rank
        if boxes is None and ("box" in name or (rank == 2 and tensor.shape[-1] == 5)):
            boxes = tensor
            continue
        if boxes is None and rank == 3 and tensor.shape[-1] == 5:
            boxes = tensor
            continue
        if scores is None and ("score" in name or (rank in (1, 2) and tensor.dtype.is_floating)):
            if rank == 2 and tensor.shape[-1] == 5:
                continue
            scores = tensor
            continue
        if labels is None and ("label" in name or (tensor.dtype.is_integer and rank in (1, 2))):
            if rank == 0:
                continue
            labels = tensor
            continue
        if count is None and ("count" in name or rank in (0, 1) and tensor.dtype.is_integer):
            if rank == 1 and tensor.shape[-1] not in (1, None) and tensor.shape.rank == 1:
                # vector of labels, not a scalar count
                if tensor.shape[0] not in (1, None):
                    continue
            count = tensor
            continue

    # Fallback by rank if names were Identity_0, ...
    leftover = [v for v in items.values()]
    if boxes is None:
        for tensor in leftover:
            if tensor.shape.rank in (2, 3) and tensor.shape[-1] == 5:
                boxes = tensor
                break
    if scores is None:
        for tensor in leftover:
            if tensor is boxes:
                continue
            if tensor.dtype.is_floating and tensor.shape.rank in (1, 2):
                scores = tensor
                break
    if labels is None:
        for tensor in leftover:
            if tensor is boxes or tensor is scores:
                continue
            if tensor.dtype.is_integer and tensor.shape.rank in (1, 2):
                labels = tensor
                break
    if count is None:
        for tensor in leftover:
            if tensor is boxes or tensor is scores or tensor is labels:
                continue
            if tensor.dtype.is_integer:
                count = tensor
                break

    missing = [
        name
        for name, val in (
            ("pre_nms_boxes", boxes),
            ("pre_nms_scores", scores),
            ("pre_nms_labels", labels),
            ("pre_nms_count", count),
        )
        if val is None
    ]
    if missing:
        keys = sorted(items)
        raise ValueError(
            f"Could not unpack pre-NMS outputs {missing} from keys {keys}. "
            "Expected boxes[P,5], scores[P], labels[P], count scalar."
        )
    return boxes, scores, labels, count


def _signature_fn(core: Any) -> Any:
    sigs = getattr(core, "signatures", None) or {}
    if "serving_default" in sigs:
        return sigs["serving_default"]
    if sigs:
        return list(sigs.values())[0]
    serve = getattr(core, "serve", None)
    if callable(serve):
        return serve
    raise ValueError("SavedModel has no signatures['serving_default'] or serve().")


def _core_input_layout(fn: Any) -> Tuple[str, bool]:
    """Return (input_name, expect_nhwc)."""
    structured = getattr(fn, "structured_input_signature", None)
    if not structured or not structured[1]:
        return "images", False
    name, spec = next(iter(structured[1].items()))
    shape = spec.shape
    expect_nhwc = shape.rank == 4 and shape[-1] == 3
    return str(name), bool(expect_nhwc)


class PreNmsCoreLayer(tf.keras.layers.Layer):
    """Call a converted ONNX SavedModel and return pre-NMS tensors."""

    def __init__(self, savedmodel_dir: Union[str, Path], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.savedmodel_dir = str(savedmodel_dir)
        self.core = tf.saved_model.load(self.savedmodel_dir)
        self._fn = _signature_fn(self.core)
        self._input_name, self._expect_nhwc = _core_input_layout(self._fn)

    def call(self, images: tf.Tensor):
        x = tf.transpose(images, [0, 2, 3, 1]) if self._expect_nhwc else images
        try:
            raw = self._fn(**{self._input_name: x})
        except TypeError:
            raw = self._fn(x)
        return unpack_pre_nms_outputs(raw)


def build_detect_model_from_pre_nms(
    pre_nms_call,
    *,
    height: int,
    width: int,
    finalize_kwargs: Dict[str, Any],
) -> tf.keras.Model:
    """Compose ``images → pre-NMS tensors → detections`` as a Keras model."""
    images = tf.keras.Input(shape=(3, height, width), batch_size=1, name="images")
    if isinstance(pre_nms_call, tf.keras.layers.Layer):
        boxes, scores, labels, count = pre_nms_call(images)
    else:

        class _PreNmsFn(tf.keras.layers.Layer):
            def call(self, x):
                return pre_nms_call(x)

        boxes, scores, labels, count = _PreNmsFn(name="pre_nms")(images)
    detections, num = DetectPostprocessLayer(finalize_kwargs, name="tf_postprocess")(
        [boxes, scores, labels, count]
    )
    return tf.keras.Model(
        inputs=images,
        outputs={"detections": detections, "num_detections": num},
        name="oriented_det_savedmodel",
    )


def export_keras_saved_model(model: tf.keras.Model, output_dir: Path) -> None:
    """Write a TF SavedModel directory (Keras 3 ``export`` or ``tf.saved_model.save``)."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "export"):
        model.export(str(output_dir))
        return
    tf.saved_model.save(model, str(output_dir))


def call_saved_model(loaded: Any, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Invoke a loaded SavedModel; returns ``(detections, num_detections)``."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    serve = getattr(loaded, "serve", None)
    raw = None
    if callable(serve):
        try:
            raw = serve(images)
        except TypeError:
            try:
                raw = serve(images=images)
            except TypeError:
                raw = None
    if raw is None:
        fn = _signature_fn(loaded)
        kwargs = {}
        structured = getattr(fn, "structured_input_signature", None)
        if structured and structured[1]:
            name = next(iter(structured[1].keys()))
            kwargs[name] = images
            raw = fn(**kwargs)
        else:
            raw = fn(images)
    if isinstance(raw, Mapping):
        det = raw.get("detections")
        num = raw.get("num_detections")
        if det is None:
            # first two values
            vals = list(raw.values())
            det, num = vals[0], vals[1]
        return det, num
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return raw[0], raw[1]
    raise TypeError(f"Unexpected SavedModel output: {type(raw)!r}")


def write_savedmodel_meta(output_dir: Path, meta: Dict[str, Any]) -> None:
    payload = dict(meta)
    payload["format"] = "tf_savedmodel"
    payload["core_backend"] = "onnx2tf_tf_nms"
    (Path(output_dir) / "export_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def build_savedmodel_from_onnx(
    onnx_path: Path,
    output_dir: Path,
    *,
    finalize_kwargs: Dict[str, Any],
    height: int,
    width: int,
    meta: Optional[Dict[str, Any]] = None,
    onnx2tf_workdir: Optional[Path] = None,
) -> Path:
    """onnx2tf convert ``onnx_path`` then wrap TF NMS into ``output_dir``."""
    from export.scripts.onnx_to_savedmodel import convert_onnx_to_savedmodel

    output_dir = Path(output_dir)
    tmp_ctx = None
    core_dir = Path(onnx2tf_workdir) if onnx2tf_workdir else None
    if core_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="odet_onnx2tf_")
        core_dir = Path(tmp_ctx.name) / "core"
    try:
        convert_onnx_to_savedmodel(Path(onnx_path), core_dir, keep_ncw=True)
        core_layer = PreNmsCoreLayer(core_dir, name="onnx2tf_core")
        model = build_detect_model_from_pre_nms(
            core_layer,
            height=height,
            width=width,
            finalize_kwargs=finalize_kwargs,
        )
        export_keras_saved_model(model, output_dir)
        if meta is not None:
            write_savedmodel_meta(output_dir, meta)
        return output_dir
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
