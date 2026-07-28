# export/tests

- [test_export_wrappers.py](test_export_wrappers.py): PyTorch-only shape tests for `export/wrappers.py` (runs in default CI with torch).
- [test_faster_rcnn_export_parity.py](test_faster_rcnn_export_parity.py): Pre-NMS export vs full `faster_rcnn_inference`; optional Hub pretrained checkpoint smoke.
- [test_oriented_rcnn_export_parity.py](test_oriented_rcnn_export_parity.py): Pre-NMS export vs `OrientedRCNN` eval; optional Hub pretrained smoke; optional ONNX/ORT.
- [test_keras_detect_bundle.py](test_keras_detect_bundle.py): Keras save/reload + relocatable `model.onnx` + `class_id_to_name` key coercion (requires TensorFlow).
- [test_ort_runtime.py](test_ort_runtime.py): ORT device/provider helpers (skipped without `onnxruntime`).
- [test_export_onnx_optional.py](test_export_onnx_optional.py): ONNX export + checker / ORT (skipped unless `onnx` and, for the ORT test, `onnxruntime` are installed).
- [test_export_cli.py](test_export_cli.py): `odet export-*` help/import smoke + `export_tf` orchestration (mocked; no Makefile).

Run from repo root:

```bash
pytest export/tests/
```

CI runs `pytest tests/ export/tests/`. Wrapper/parity/CLI tests run without TF; Keras bundle tests skip if TensorFlow is missing.
