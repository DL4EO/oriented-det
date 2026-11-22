# onnx_export

Consumer bundle written by [`export/`](../export/README.md): **ONNX graph** plus copies of the detection runtime from `export/`.

Source of truth stays under `export/`. At `odet export onnx`, these are copied next to `model.onnx`:

- `nms.py`, `preprocess.py`, `postprocess.py`, `runtime.py`, `ort_runtime.py`
- `demo.py`, `infer_onnx.py`
- `demo/planes_pleiades_neo.jpg`
- `requirements-runtime.txt`

Those copies are gitignored.

The copied folder is enough to run preprocess → ONNX Runtime → NMS. No oriented-det, no PyTorch, no rest of this repo.

```bash
make export-zip    # from oriented-det repo root → onnx_export.zip (no __pycache__)
cd onnx_export
pip install -r requirements-runtime.txt
python demo.py
python infer_onnx.py --onnx model.onnx --smoke
python infer_onnx.py --onnx model.onnx --images ./tiles --output ./predictions
```

```python
from runtime import detect_image

dets = detect_image("tile.jpg", "model.onnx")
```

`detect_image` tiles images larger than `window_size` (or `target_size` when `window_size` is absent) for `resize_mode` `fixed` and `crop`. Each window is resized to the ONNX canvas. See [`export/README.md`](../export/README.md).

In this repo you can still use the sources:

```bash
pip install -r export/requirements-runtime.txt
python -m export demo
python -m export infer --onnx ./onnx_export/model.onnx --smoke
```

Optional exact polygon IoU (CPU): `pip install shapely`, then `--nms-backend shapely`. Shapely is already a core oriented-det dependency; the copied bundle does not require it.

## Layout

| Path | Purpose |
|------|---------|
| `model.onnx` | Pre-NMS ONNX |
| `model.export_meta.json` | Mean/std, class names, score floor, NMS IoU |
| `nms.py` … `runtime.py` | Copies of `export/*.py` (detection pipeline) |
| `demo.py` / `infer_onnx.py` | Copies of `export/scripts/` |
| `requirements-runtime.txt` | Copy of `export/requirements-runtime.txt` |
| `demo/` | Bundled plane image + overlay/JSON from `python demo.py` |

ONNX input is NCHW RGB, already mean/std-normalized. Use `preprocess` / `detect_image`, not raw JPEGs as the ONNX input.
