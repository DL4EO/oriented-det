# Export parity

The delivered graph is **ONNX pre-NMS** (`rotated_fcos_pre_nms` for FCOS). Preprocess stays in `export.preprocess`. Final rotated NMS is [`export/nms.py`](nms.py). The consumer stack (`preprocess`, `postprocess`, `nms`, `runtime`, `ort_runtime`) is copied into [`onnx_export/`](../onnx_export/README.md) at export so a consumer can run it **without oriented-det**.

## `rotated_fcos_pre_nms` (default)

| Aspect | Parity |
|--------|--------|
| Input | NCHW RGB, mean/std-normalized (same as `oriented_det.runtime.inference` training-style preprocess). |
| Backbone + FCOS head + DistanceAnglePointCoder decode | Same weights/logic as `RotatedFCOS` eval via `rotated_fcos_decode_pre_nms` (includes GroupNorm). |
| Per-level pre-NMS cap | Always `topk(min(nms_pre, H×W))` then score / size / finite filters. Invalid top-k rows are zeroed (`score=0`, `w=h=0`) and left in the padded tensor. |
| Pre-NMS output padding | `pad_pre_nms_detections`. Default `P = nms_pre ×` FPN levels. |
| Final NMS / score floors | `export.postprocess.finalize_detections_numpy` calling `export.nms` (numpy clip, or Shapely if `nms_backend=shapely`). Production `score_threshold` applied **before** NMS. Class-aware vs class-agnostic follows resolved `nms_class_agnostic` (`production` override, else `model`; export unset default is class-agnostic). Small float drift vs Shapely is possible on the `python` backend. |
| Sliding-window tiling | `detect_image` tiles `fixed` / `crop` images larger than `window_size` (else `target_size`). Same edge-flush grid and `target_size / window_size` box scale as `run_inference_sliding_window`. Overlap default 200 px. Per-window NMS, then a second NMS across windows. Interior overlap margin stays 0. `pad` / `keep_ratio` are a single resize. |
| GeoJSON | Out of scope — PyTorch deploy path. |

## `rotated_rtmdet_pre_nms`

| Aspect | Parity |
|--------|--------|
| Input | NCHW RGB, mean/std-normalized (same as training-style preprocess). Fixed H×W, batch 1. |
| Backbone + SepBN head + DistanceAnglePointCoder decode | Same weights/logic as `RotatedRTMDet` eval via `_rtmdet_level_topk_decode`. No centerness. |
| Per-level pre-NMS cap | Always `topk(min(nms_pre, H×W))` then score / size / finite filters. Invalid top-k rows are zeroed (`score=0`, `w=h=0`) and left in the padded tensor. |
| Pre-NMS output padding | `pad_pre_nms_detections`. Default `P = nms_pre ×` FPN levels (P3–P5). |
| Final NMS / score floors | `export.postprocess.finalize_detections_numpy`. Class-aware vs class-agnostic follows resolved `nms_class_agnostic` (same as FCOS / two-stage). |

## `faster_rcnn_pre_nms` / `oriented_rcnn_pre_nms`

| Aspect | Parity |
|--------|--------|
| Backbone, RPN, ROI, decode (pre-NMS tensors) | Same as the corresponding PyTorch eval path; deterministic RPN for export. |
| Final rotated NMS | `export.nms` greedy NMS (`python` or `shapely`), copied into `onnx_export/` with the rest of the consumer stack at export. |

## `backbone` / `retinanet_heads` / `rotated_fcos_heads`

Tensor-only subgraphs. Full detections need decode + NMS elsewhere. Prefer `rotated_fcos_pre_nms` for FCOS.

## Regression tests

Wrapper/ONNX parity: `test_export_wrappers.py`, `test_faster_rcnn_export_parity.py`, `test_oriented_rcnn_export_parity.py`, `test_rotated_fcos_export_parity.py`, `test_rotated_rtmdet_export_parity.py`. Preprocess vs torchvision: `test_preprocess.py`. Numpy NMS: `test_export_nms.py`, `test_postprocess.py`. Consumer stack without oriented-det: `test_standalone_runtime.py`.
