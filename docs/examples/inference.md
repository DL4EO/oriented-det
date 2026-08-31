# Inference and predictions

Run inference on a validation split and save results for metrics or visualization.

## CLI

```bash
# Checkpoint/config auto-resolve from experiment-dir when omitted
odet preds --experiment-dir runs/oriented_rcnn/<timestamp> --data-split val
```

Or from the repo root after training:

```bash
make preds
make metrics
```

Inference-only (skip mAP / analysis):

```bash
odet preds --experiment-dir runs/oriented_rcnn/<timestamp> --data-split val --no-diagnostics
```

## Behavior

- Thresholds, NMS, and sliding-window overlap come from **`production.*`** in the experiment `config.json`
- Images larger than the model canvas use **sliding-window** tiling when `resize_mode` is `fixed` or `crop` (DOTA). **`pad`** (HRSC2016) always runs one training-style whole-image forward (scale long edge to the canvas, then pad).

## Output artifacts

Default directory: `predictions/<YYYYMMDD_HHMMSS>/` at the **repository root** (or `--output-dir`).

| File | When | Contents |
|------|------|----------|
| `predictions.json` | Always (inference) | Per-image detections (rboxes, scores, labels) |
| `analysis_iou0.50.json` | With diagnostics (default) | PR/F1 threshold sweep, per-class AP, confusion matrix, best-threshold block |
| `model_analysis_<timestamp>.md` | With diagnostics | Human-readable report: per-class gts/dets/recall/AP, optional per-class best thresholds |
| `tile_metrics.csv` | With `--save-tile-metrics-csv` | Per-tile precision/recall/F1 for hard-tile oversampling and `dataset.drop_easy_empty_tiles` |
| `visualizations/` | With `--save-visualizations` | Overlay images |

Per-class best-threshold tables are computed **by default**. Disable with `--no-per-class-threshold-analysis` (`--per-class-threshold-analysis` is kept for backward compatibility).

## Recompute metrics only

```bash
odet preds --metrics-from-json predictions/<timestamp> --config runs/.../config.json
```

Re-runs mAP and writes fresh `analysis_iou*.json` / `model_analysis_*.md` from an existing `predictions.json` without re-inference.

See the [tools reference on GitHub](https://github.com/DL4EO/oriented-det/blob/main/tools/README.md) and [Getting Started: Tools](../getting-started/tools.md).
