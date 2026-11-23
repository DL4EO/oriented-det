# odet CLI

Install the package (`uv pip install -e .` or `pip install oriented-det`), then run subcommands via the **`odet`** entry point (see [`pyproject.toml`](../../pyproject.toml)). `python -m oriented_det.cli <command>` is equivalent.

| Command | Role |
|---------|------|
| `train` | Config-based training |
| `train-multi-gpu` | `torchrun` wrapper around `oriented_det.cli.train` |
| `preds` / `metrics` | Validation inference and offline metrics (`tools/save_predictions.py`) |
| `lr-finder`, `stats`, `tile-dota`, `image-demo`, `viewer` | Data and training utilities |
| `playground-csv`, `playground-to-dota` | Playground CSV / DOTA export |
| `hrsc-to-dota` | HRSC2016 XML → DOTA PNG + labels |
| `fair1m-to-dota` | FAIR1M XML → DOTA images + labels (optional holdout) |
| `coco-to-dota` | COCO polygons (or HRSID dump) → DOTA images + labels |
| `ssdd-to-dota` | SSDD XML/COCO/DOTA → DOTA images + labels |
| `dota-submit` | DOTA v1.0 Task 1 zip from Hub `hf://` weights, a training run, or `predictions.json` |
| `export` | ONNX producer (`onnx` / `infer` / `demo` / `preds`; [`export/README.md`](../../export/README.md)) |
| `labels-to-comma` | Convert DOTA label files to comma-separated format |

Subcommands load implementations from [`tools/`](../../tools/) (train, preds, tiling, …) except **`export`**, which loads [`export.cli`](../../export/cli.py). Reusable inference, checkpoint, and collate helpers live in [`oriented_det/runtime/`](../runtime/). See the main [README](../../README.md#repository-map).
