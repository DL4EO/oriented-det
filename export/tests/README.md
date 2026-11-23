# export/tests

- [test_preprocess.py](test_preprocess.py): RGB preprocess vs torchvision; export meta sidecar keys (incl. `nms_class_agnostic` resolution); box scale / records.
- [test_sliding_window.py](test_sliding_window.py): ONNX sliding-window grid, box mapping, and when `detect_image` tiles.
- [test_export_wrappers.py](test_export_wrappers.py): PyTorch-only shape tests for `export/wrappers.py`.
- [test_faster_rcnn_export_parity.py](test_faster_rcnn_export_parity.py): Pre-NMS export vs full `faster_rcnn_inference`.
- [test_oriented_rcnn_export_parity.py](test_oriented_rcnn_export_parity.py): Pre-NMS export vs `OrientedRCNN` eval.
- [test_rotated_fcos_export_parity.py](test_rotated_fcos_export_parity.py): Pre-NMS export vs `RotatedFCOS` eval; optional ONNX/ORT.
- [test_rotated_rtmdet_export_parity.py](test_rotated_rtmdet_export_parity.py): Pre-NMS export vs `RotatedRTMDet` eval; optional ONNX/ORT.
- [test_export_nms.py](test_export_nms.py): Rotated IoU / NMS from `export.nms` (python clip; Shapely if installed).
- [test_postprocess.py](test_postprocess.py): Score prefilter vs post-NMS filter parity; post-NMS `output_classes` allowlist; pairwise rotated IoU helpers.
- [test_per_class_score_threshold.py](test_per_class_score_threshold.py): Inference override of `production.per_class_score_threshold` (merge, `detect_array` / `detect_image`, infer CLI). No ONNX export.
- [test_standalone_runtime.py](test_standalone_runtime.py): Consumer stack (`demo` / `infer` / `runtime`) imports and runs without `oriented_det`. Copied `onnx_export/` bundle imports without the repo `export` package.
- [test_export_onnx_optional.py](test_export_onnx_optional.py): ONNX export + checker / ORT (skipped unless `onnx` / `onnxruntime` / `onnxscript` are installed).
- [test_export_cli.py](test_export_cli.py): ``odet export`` / ``python -m export`` help/import smoke. TensorFlow / hyphenated top-level ``odet export-*`` stay rejected. Zip bundle skips `__pycache__`.
- [test_ort_runtime.py](test_ort_runtime.py): ORT device/provider helpers.

Run from the oriented-det repo root:

```bash
pytest export/tests/
# or: make export-test
```

Wrapper/parity/CLI tests run without ONNX Runtime. Optional checker/ORT tests skip if `onnx` / `onnxruntime` / `onnxscript` are missing (`uv pip install -e ".[export]"`). `test_standalone_runtime.py` and `test_export_nms.py` do not import oriented-det.
