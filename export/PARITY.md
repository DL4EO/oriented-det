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
| RPN proposals before ROI | Padded to `max_pre_nms_candidates` with **zero-area** xyxy pads; ROI candidates keep only positive-area boxes (dynamic mask). |
| Pre-NMS output padding | `pad_pre_nms_detections` uses `torch.cat` zero-padding (not slice assignment), which avoids invalid ONNX `Expand` when the dynamic candidate count is not exactly 0, 1, or `max_candidates`. |
| Detect bundle layout | `keras_model.keras` + **`model.onnx`** + `export_meta.json` (relative ONNX name; resolve via `load_keras_detect_model`). |
| Final rotated NMS | **Exact** match with deploy when `production.final_nms_use_cpu: true` (polygon CPU path in `rotated_nms`). Score floor is applied **before** NMS (equivalent for greedy NMS; much faster on dense pre-NMS pads). |
| `production.score_threshold` / per-class floors | Applied in Keras bundle after NMS (`export/postprocess.py`). |
| Sliding-window tiling, margin filter, GeoJSON | **Out of scope** — same as deploy API (see `oriented_det.runtime.inference`). |
| TF SavedModel reload of `tf.py_function` | **Not used**; load `keras_model.keras` with `FasterRCNNDetectLayer` custom object. |
| onnx2tf full graph | May fail on `ScatterND`; bundle uses **ONNX Runtime** for the core graph instead. |

## `oriented_rcnn_pre_nms` + detect bundle (Oriented R-CNN)

| Aspect | Parity |
|--------|--------|
| Backbone, midpoint RPN, OrientedROIAlign, decode | Same weights/logic as `OrientedRCNN` eval via `oriented_det.models.oriented_rcnn_inference`; deterministic midpoint RPN top-k for export. |
| RPN proposals before ROI | Padded oriented proposals with **zero-size** OBBs via always-on `cat([x[:k], zeros(k)])[:k]` (keeps pad in the ONNX graph even when the trace dummy already has `k` keeps); ROI candidates keep only `w>0` and `h>0` (pads are not min-clamped to 1). |
| OrientedROIAlign under ONNX | Masked all-N `grid_sample` per FPN level (no indexed ScatterND writes); ROI grids packed along H so features stay batch=1 (avoids ORT `Expand` OOM on `N×C×H×W`); masks use `reshape(-1,1,1,1)` (avoids Reshape size mismatch when proposal count varies). Eager path stays chunked. |
| Final NMS / score floors / Keras bundle | **Same** as Faster R-CNN detect bundle (`export/postprocess.py`). |
| onnx2tf full graph | Same ORT-inside-Keras mitigation; pure TF conversion remains experimental. |

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

`export/tests/test_export_wrappers.py` checks **wrapper forward shapes** against PyTorch without requiring ONNX or TensorFlow. Golden numeric `.npz` dumps are optional and not required for CI. Pre-NMS parity: `test_faster_rcnn_export_parity.py`, `test_oriented_rcnn_export_parity.py`.
