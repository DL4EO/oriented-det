# odet CLI

Install the package (`uv pip install -e .`), then run subcommands via the **`odet`** entry point (see [`pyproject.toml`](../../pyproject.toml)).

| Command | Role |
|---------|------|
| `train` | Config-based training |
| `train-multi-gpu` | `torchrun` wrapper around `oriented_det.cli.train` |
| `preds` / `metrics` | Validation inference and offline metrics (`tools/save_predictions.py`) |
| `lr-finder`, `stats`, `tile-dota`, `image-demo`, `viewer` | Data and training utilities |
| `playground-csv`, `playground-to-dota` | Playground CSV / DOTA export |
| `export-onnx`, `export-tf` | ONNX / TensorFlow export (see [`export/`](../../export/)) |

Subcommands load implementations from [`tools/`](../../tools/) (train, preds, tiling, …). Reusable inference, checkpoint, and collate helpers live in [`oriented_det/runtime/`](../runtime/). See the main [README](../../README.md#repository-layout).
