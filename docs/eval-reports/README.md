# Published eval-val reports

Frozen **`make eval-val`** metrics for published Hub slugs (and a few historical local baselines). Score ≥ 0.05, final NMS IoU **0.1** via `evaluation.final_nms_iou_threshold` (MMRotate test parity; recipes also ship production NMS **0.1**), mAP matching IoU 0.50.

**`make eval-val` is not always held-out.** DOTA zoo recipes train on train+val tiles, so DOTA eval-val is **leaky**; the real test is official **Task 1**. HRSC, FAIR1M, SSDD, and HRSID do **not** put test/val into training. See [Train / eval splits](../user-guide/data.md#train--eval-splits). Task 1 / zoo narrative: [OrientedDet v0.1.1](https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) and [v0.2.0 Rotated FCOS](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

| Dataset | What `make eval-val` scores | Held-out? |
|---------|-----------------------------|-----------|
| **DOTA v1.0** | val tiles (7,669, `filter_empty_gt=false`) that were also in train | **No** (leaky). Real test: **Task 1** |
| **HRSC2016** | ImageSets **test** (453 images, whole-image `keep_ratio`) | **Yes** — test is not in train |
| **FAIR1M** | official val tiles | **Yes** — val is not in train (no FAIR1M reports in this folder yet) |
| **SSDD** | official last-digit **test** | **Yes** — test is not in train ([90.34%](rotated_faster_rcnn_ssdd_le90_1x/model_analysis.md)) |
| **HRSID** | official **test** (no val) | **Yes** — test is not in train ([78.55%](rotated_faster_rcnn_hrsid_le90_1x/model_analysis.md)) |

Each subdirectory is named after the manifest **slug** and is **tracked in git** (reports and analysis only — no `predictions.json`; see below).

| File | In git | Purpose |
|------|--------|---------|
| `model_analysis.md` | yes | Human-readable report (`make eval-val` mAP50, per-class AP, **GT alignment / mean best IoU**, confusion matrix). **DOTA:** that mAP50 is leaky; slugs with official Task 1 also include an **Official Task 1 (hidden test)** table. **HRSC:** that mAP50 is held-out ImageSets test. |
| `analysis_iou0.50.json` | yes | Structured **eval-val** metrics (`make metrics`); includes `gt_alignment_metrics`. Not Task 1. |
| `pr_curve.png`, `threshold_metrics.png` | yes | Evaluation plots (when present) |
| `predictions.json` | **no** | Raw detections — too large for GitHub; keep under gitignored [`predictions/`](../../predictions/) locally |

## Slug index

| Hub slug | Official Task 1 (DOTA real test) | `make eval-val` mAP50 | Local `predictions.json` (viewer) |
|----------|----------------------------------|-----------------------|-----------------------------------|
| `oriented_rcnn_dota_le90_1x` | **76.73%** | 77.66% | `predictions/20260910_033748/` |
| `oriented_rcnn_dota_le90_3x` | **74.88%** | 82.92% | `predictions/20260917_093209/` |
| `rotated_faster_rcnn_dota_le90_1x` | **74.42%** | 77.55% | `predictions/20260908_015217/` |
| `rotated_faster_rcnn_dota_le90_3x` | **74.48%** | 83.46% | `predictions/20260903_004825/` |
| `rotated_retinanet_dota_le90_1x` | **71.72%** (OBB) | 71.12% | `predictions/20260920_015010/` |
| `rotated_retinanet_dota_le90_3x` | **73.89%** (OBB) | 78.56% | `predictions/20260921_052956/` |
| `rotated_retinanet_dota_le90_1x_hbb` | **67.87%** (HBB) | 68.20% | `predictions/20260912_232006/` |
| `rotated_retinanet_dota_le90_3x_hbb` | **70.70%** (HBB) | 76.51% | `predictions/20260913_234415/` |
| `rotated_fcos_dota_le90_1x` | **73.07%** | 75.13% | `predictions/20260908_132129/` |
| `rotated_fcos_dota_le90_3x` | **72.91%** | 82.32% | — |
| `rotated_rtmdet_dota_le90_3x` | **77.60%** | 82.75% | `predictions/20261003_021342/` |
| `oriented_rcnn_hrsc2016_le90_3x` | — | 90.41% | `predictions/20260831_011151/` |
| `rotated_faster_rcnn_hrsc2016_le90_3x` | — | 88.77% | `predictions/20260831_050947/` |
| `rotated_fcos_hrsc2016_le90_3x` | — | 88.34% | `predictions/20260831_033939/` |

DOTA `make eval-val` mAP50 is **leaky** (val tiles in train). HRSC rows are **held-out ImageSets test** (not leaky). Advertised Oriented R-CNN / Faster R-CNN / FCOS zoo is **1×** Task 1; 3× Hub slugs exist for AP75. RetinaNet Hub default is **OBB** (1× **71.72%** / 3× **73.89%**); circum-HBB is `*_hbb`.

Historical reports (not the advertised zoo): older Oriented R-CNN 1× leaky 74.79%, June Oriented R-CNN 3× leaky 79.40% NMS 0.50 (replaced by Hub Task 1 **74.88%** / eval-val 82.92%), [`rotated_faster_rcnn_dota_le90_3x_ce`](rotated_faster_rcnn_dota_le90_3x_ce/model_analysis.md) 75.58%, June RetinaNet 1× leaky 64.14% (replaced by Hub OBB Task 1 **71.72%**), June RetinaNet 3× leaky 71.52% (replaced by Hub OBB 3× Task 1 **73.89%**), [`rotated_fcos_dota_le90_3x_kfiou_aux`](rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md) 77.18%, [`rotated_fcos_dota_le90_3x_l1`](rotated_fcos_dota_le90_3x_l1/model_analysis.md) 73.92% (local L1 baseline). FCOS 3× stays on Hub; advertised FCOS zoo is 1× Task 1.

**Viewer** (needs `predictions.json` in the directory you pass):

```bash
make viewer VIEWER_PRED_DIR=predictions/20260627_082942 DOTA_DATA_ROOT=/path/to/data/DOTA-v1.0-tiled
```

Checkpoints and training logs live under **`runs/<model>/<timestamp>/`**, not prediction JSON.

## Publish reports after eval-val

Run inference into gitignored `predictions/`, then copy **lightweight** artifacts into `docs/eval-reports/<slug>/`:

```bash
EXPERIMENT=runs/oriented_rcnn/20260911-102320
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

Update `oriented_det/pretrained/manifest.json`: `eval_report` → `docs/eval-reports/<slug>/model_analysis.md`. **`eval_map50`** is this `make eval-val` mAP50 — **leaky val tiles for DOTA**, **held-out ImageSets test for HRSC**. **`eval_task1_map50`** is official DOTA v1.0 Task 1 (paste the server table into `model_analysis.md`; do not copy Task 1 into `analysis_iou0.50.json`).
