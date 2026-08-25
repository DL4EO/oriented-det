# Published eval-val reports (DOTA le90)

Frozen **`make eval-val`** metrics for each Hub pretrained slug: full val split (7,669 tiles, `filter_empty_gt=false`), score ≥ 0.05, production decode, mAP matching IoU 0.50.

Each subdirectory is named after the manifest **slug** and is **tracked in git** (reports and analysis only — no `predictions.json`; see below).

| File | In git | Purpose |
|------|--------|---------|
| `model_analysis.md` | yes | Human-readable report (mAP50, per-class AP, **GT alignment / mean best IoU**, confusion matrix) |
| `analysis_iou0.50.json` | yes | Structured metrics (`make metrics`); includes `gt_alignment_metrics` |
| `pr_curve.png`, `threshold_metrics.png` | yes | Evaluation plots (when present) |
| `predictions.json` | **no** | Raw detections — too large for GitHub; keep under gitignored [`predictions/`](../../predictions/) locally |

## Slug index

| Hub slug | eval-val mAP50 | Local `predictions.json` (viewer) |
|----------|----------------|-----------------------------------|
| `oriented_rcnn_dota_le90_3x` | 79.40% | `predictions/20260627_082942/` |
| `oriented_rcnn_dota_le90_1x` | 74.79% | `predictions/20260618_140030/` |
| `rotated_faster_rcnn_dota_le90_1x` | 77.57% | `predictions/20260710_044125/` |
| `rotated_faster_rcnn_dota_le90_3x` | 83.42% | `predictions/20260705_055252/` |
| `rotated_faster_rcnn_dota_le90_3x_ce` | 75.58% | regenerate (see below) |
| `rotated_retinanet_dota_le90_3x` | 71.52% | `predictions/20260615_005855/` |
| `rotated_retinanet_dota_le90_1x` | 64.14% | regenerate (see below) |
| `rotated_fcos_dota_le90_3x_riou` | 81.58% | `predictions/20260824_212907/` |
| `rotated_fcos_dota_le90_3x_kfiou_aux` | 77.18% | `predictions/20260820_121343/` |
| `rotated_fcos_dota_le90_3x` (local L1 baseline; not Hub) | 73.92% | `predictions/20260813_192333/` |

**Viewer** (needs `predictions.json` in the directory you pass):

```bash
make viewer VIEWER_PRED_DIR=predictions/20260627_082942 DOTA_DATA_ROOT=/path/to/DOTA-v1.0-tiled
```

Checkpoints and training logs live under **`runs/<model>/<timestamp>/`**, not prediction JSON.

## Publish reports after eval-val

Run inference into gitignored `predictions/`, then copy **lightweight** artifacts into `docs/eval-reports/<slug>/`:

```bash
EXPERIMENT=runs/oriented_rcnn/20260621-092802
SLUG=oriented_rcnn_dota_le90_3x
SCRATCH=predictions/$(date +%Y%m%d_%H%M%S)

odet preds --experiment-dir "$EXPERIMENT" --output-dir "$SCRATCH" --no-diagnostics
odet preds --metrics-from-json "$SCRATCH"

DEST=docs/eval-reports/$SLUG
mkdir -p "$DEST"
cp "$SCRATCH"/analysis_iou0.50.json "$SCRATCH"/pr_curve.png "$SCRATCH"/threshold_metrics.png "$DEST/" 2>/dev/null || true
cp "$SCRATCH"/model_analysis_*.md "$DEST/model_analysis.md"
# Leave predictions.json in $SCRATCH only (gitignored)
```

Update `oriented_det/pretrained/manifest.json` `eval_report` → `docs/eval-reports/<slug>/model_analysis.md`.
