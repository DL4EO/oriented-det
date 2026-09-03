# Published eval-val reports

Frozen **`make eval-val`** metrics for published Hub slugs (and a few historical local baselines). DOTA le90: full val split (7,669 tiles, `filter_empty_gt=false`). HRSC2016: ImageSets test (453 images, whole-image `keep_ratio`). Score ≥ 0.05, final NMS IoU **0.1** via `evaluation.final_nms_iou_threshold` (MMRotate test parity; deploy recipes ship production NMS **0.3**), mAP matching IoU 0.50.

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
| `rotated_faster_rcnn_dota_le90_3x` | 83.46% | `predictions/20260903_004825/` |
| `rotated_retinanet_dota_le90_3x` | 71.52% | `predictions/20260615_005855/` |
| `rotated_fcos_dota_le90_3x` | 82.32% | `predictions/20260901_053115/` |
| `oriented_rcnn_hrsc2016_le90_3x` | 90.41% | `predictions/20260831_011151/` |
| `rotated_faster_rcnn_hrsc2016_le90_3x` | 88.77% | `predictions/20260831_050947/` |
| `rotated_fcos_hrsc2016_le90_3x` | 88.34% | `predictions/20260831_033939/` |

Historical reports (not on Hub): [`oriented_rcnn_dota_le90_1x`](oriented_rcnn_dota_le90_1x/model_analysis.md) 74.79%, [`rotated_faster_rcnn_dota_le90_1x`](rotated_faster_rcnn_dota_le90_1x/model_analysis.md) 77.57%, [`rotated_faster_rcnn_dota_le90_3x_ce`](rotated_faster_rcnn_dota_le90_3x_ce/model_analysis.md) 75.58%, [`rotated_retinanet_dota_le90_1x`](rotated_retinanet_dota_le90_1x/model_analysis.md) 64.14%, [`rotated_fcos_dota_le90_3x_kfiou_aux`](rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md) 77.18%, [`rotated_fcos_dota_le90_3x_l1`](rotated_fcos_dota_le90_3x_l1/model_analysis.md) 73.92% (local L1 baseline).

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
