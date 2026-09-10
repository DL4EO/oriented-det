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
| `fair1m-to-dota` | FAIR1M XML → DOTA images + labels (optional holdout) |
| `dota-submit` | DOTA v1.0 Task 1 zip from Hub `hf://` weights, a training run, or `predictions.json` |
| `labels-to-comma` | Convert DOTA label files to comma-separated format |

Subcommands load implementations from [`tools/`](../../tools/) (train, preds, tiling, …). Reusable inference, checkpoint, and collate helpers live in [`oriented_det/runtime/`](../runtime/). See the main [README](../../README.md#repository-layout).
