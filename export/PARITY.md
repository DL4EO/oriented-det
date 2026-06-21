# Export parity (Phase 1)

This document records what **matches** PyTorch full inference and what is **intentionally different** when using ONNX / TensorFlow SavedModel from `export/`.

## `backbone` export mode

| Aspect | Parity |
|--------|--------|
| FPN feature tensors vs `model.backbone(images)` | Same module, same weights; numeric closeness depends on ONNX/TF runtime (expect small float drift). |
| Full detections | **Not** produced; you must run RPN/ROI/RetinaNet head + decode + NMS elsewhere. |

## `retinanet_heads` export mode

| Aspect | Parity |
|--------|--------|
| Per-level cls/bbox tensors vs `RotatedRetinaNet.head(FPN)` | Same head, same weights. |
| Softmax, anchor generation, decode, NMS | **Out of ONNX** — identical to PyTorch only if you reimplement that pipeline on top of exported tensors. |

## `faster_rcnn_pre_nms` + detect bundle (Rotated Faster R-CNN)

| Aspect | Parity |
|--------|--------|
| Backbone, RPN, ROI align, decode (pre-NMS tensors) | Same weights and logic as `RotatedFasterRCNN` via `oriented_det.models.faster_rcnn_inference`; deterministic RPN top-k for export. |
| RPN proposals before ROI | Padded to `max_pre_nms_candidates` (= `rpn_post_nms_top_n` on the loaded model, including `production.*` overrides) so ONNX ROI align traces with fixed shapes. |
| Pre-NMS output padding | `pad_pre_nms_detections` uses `torch.cat` zero-padding (not slice assignment), which avoids invalid ONNX `Expand` when the dynamic candidate count is not exactly 0, 1, or `max_candidates`. |
| Final rotated NMS | **Exact** match with deploy when `production.final_nms_use_cpu: true` (polygon CPU path in `rotated_nms`). |
| `production.score_threshold` / per-class floors | Applied in Keras bundle after NMS (`export/postprocess.py`). |
| Sliding-window tiling, margin filter, GeoJSON | **Out of scope** — same as deploy API (see `oriented_det.runtime.inference`). |
| TF SavedModel reload of `tf.py_function` | **Not used**; load `keras_model.keras` with `FasterRCNNDetectLayer` custom object. |
| onnx2tf full graph | May fail on `ScatterND`; bundle uses **ONNX Runtime** for the core graph instead. |

## ONNX → TensorFlow

| Risk | Mitigation |
|------|------------|
| Operator gaps or bad lowers in `onnx2tf` | Pin `onnx`, `onnx2tf`, and `tensorflow` versions from `requirements-export.txt`; validate with `onnxruntime` on the `.onnx` file first. |
| Multi-output SavedModel | `predict_savedmodel.py` prints signatures; wire your serving client to the exported tensor names. |

## TFLite

| Aspect | Note |
|--------|------|
| Full graph with oriented NMS | Often **infeasible**; keep NMS on CPU or in application code. |
| Delegates | GPU/NPU delegates may reject ops; fall back to XNNPACK CPU. |

## Regression tests

`export/tests/test_export_wrappers.py` checks **wrapper forward shapes** against PyTorch without requiring ONNX or TensorFlow. Golden numeric `.npz` dumps are optional and not required for CI.
