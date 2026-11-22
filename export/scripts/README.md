# export/scripts

CLI implementations invoked by ``odet export <command>`` or ``python -m export <command>`` from the **oriented-det** repo root.

- **demo / infer:** [requirements-runtime.txt](../requirements-runtime.txt) only (numpy, Pillow, onnxruntime). The consumer stack (`runtime.py`, `preprocess.py`, `postprocess.py`, `nms.py`, `ort_runtime.py`, plus these scripts) is copied into `onnx_export/` at export. No oriented-det.
- **onnx / preds:** oriented-det plus [requirements-export.txt](../requirements-export.txt).

Default artifact directory is [`../../onnx_export/`](../../onnx_export/README.md). The deliverable is **`onnx_export/model.onnx`**.

| Script | `odet export` | Purpose |
|--------|-------------------|---------|
| [export_onnx.py](export_onnx.py) | `onnx` | PyTorch → ONNX + sidecar meta (preprocess / postprocess). Default `rotated_fcos_pre_nms`. |
| [infer_onnx.py](infer_onnx.py) | `infer` | ORT + Python NMS on a folder (or `--smoke`). Images larger than `window_size` are tiled (`fixed` / `crop`). Repeatable `--per-class-score-threshold CLASS=FLOAT` merges into the sidecar per-class map (case-insensitive; omitted classes stay). `--score-threshold` stays the global floor only. |
| [demo.py](demo.py) | `demo` | Bundled plane image + NMS assertions + overlay. |
| [save_predictions_onnx.py](save_predictions_onnx.py) | `preds` | Val split via ONNX + Python NMS (eval-val JSON). Uses the experiment `window_size` and overlap for sliding windows. |
| [zip_bundle.py](zip_bundle.py) | — | Zip `onnx_export/` without `__pycache__` (`make export-zip`). |

See [../README.md](../README.md) for full usage.
