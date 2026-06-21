# export/tests

- [test_export_wrappers.py](test_export_wrappers.py): PyTorch-only shape tests for `export/wrappers.py` (runs in default CI with torch).
- [test_faster_rcnn_export_parity.py](test_faster_rcnn_export_parity.py): Pre-NMS export vs full `faster_rcnn_inference`; optional deploy checkpoint smoke.
- [test_export_onnx_optional.py](test_export_onnx_optional.py): ONNX export + checker / ORT (skipped unless `onnx` and, for the ORT test, `onnxruntime` are installed).

Run from repo root:

```bash
pytest export/tests/
```

Expect **3 passed** (wrappers) and **2 skipped** (optional ONNX) when `onnx` / `onnxruntime` are not installed.
