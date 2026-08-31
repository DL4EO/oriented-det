# odet CLI

Install the package (`uv pip install -e .` or `pip install oriented-det`), then run subcommands via the **`odet`** entry point (see [`pyproject.toml`](../../pyproject.toml)).

| Command | Role |
|---------|------|
| `train` | Config-based training |
| `train-multi-gpu` | `torchrun` wrapper around `oriented_det.cli.train` |
| `preds` / `metrics` | Validation inference and offline metrics (`tools/save_predictions.py`) |
| `lr-finder`, `stats`, `tile-dota`, `image-demo`, `viewer` | Data and training utilities |
| `playground-csv`, `playground-to-dota` | Playground CSV / DOTA export |
| `hrsc-to-dota` | HRSC2016 XML → DOTA PNG + labels |
| `export-onnx` | ONNX export (`export.scripts.export_onnx`) |
| `export-tf` | ONNX + Keras detect bundle (`export.scripts.export_tf`) |
| `export-detect` | Keras bundle from existing ONNX (`export.scripts.build_faster_rcnn_savedmodel`) |
| `export-preds` | Val inference via Keras bundle (`export.scripts.save_predictions_tf`) |

TF export commands require `pip install "oriented-det[export]"` (or the export requirements). They call pure Python entrypoints — no Makefile. Default artifacts: `./odet_export/`. See [`export/README.md`](../../export/README.md).

Subcommands load implementations from [`tools/`](../../tools/) (train, preds, tiling, …). Reusable inference, checkpoint, and collate helpers live in [`oriented_det/runtime/`](../runtime/). See the main [README](../../README.md#repository-layout).
