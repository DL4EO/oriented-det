# Model Analysis Report

Hub slug **`oriented_rcnn_dota_le90_1x`**. Official DOTA v1.0 Task 1 **76.73%** (AP75 50.24, COCO mAP 46.59). Deploy `production.score_threshold` **0.55** (this sweep’s best F1 0.60 − 0.05). Val tiles are in the train union; do not quote this report’s mAP50 as held-out accuracy.

- Generated at: `2026-09-10T04:58:49.104246`

## Model metadata
- Experiment dir: `runs/oriented_rcnn/20260908-144807`
- Checkpoint: `runs/oriented_rcnn/20260908-144807/checkpoints/best_mAP_0.80.pth`
- Checkpoint modified: `2026-09-10T03:11:22.131853`
- Config: `runs/oriented_rcnn/20260908-144807/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `109169`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6000`
- Precision at best threshold: `0.7823`
- Recall at best threshold: `0.8350`
- F1 at best threshold: `0.8078`
- F2 at best threshold: `0.8239`
- mAP50: `0.7766` (77.66%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7760`
- Global mean best IoU (same class): `0.7724` (median `0.8271`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7810 | 0.7791 | 0.8237 |
| `basketball-court` | 278 | 0.8773 | 0.8772 | 0.8999 |
| `bridge` | 666 | 0.6456 | 0.6437 | 0.7219 |
| `ground-track-field` | 216 | 0.8139 | 0.8044 | 0.8685 |
| `harbor` | 4298 | 0.7329 | 0.7313 | 0.7860 |
| `helicopter` | 157 | 0.7639 | 0.7382 | 0.7857 |
| `large-vehicle` | 9398 | 0.8107 | 0.8040 | 0.8405 |
| `plane` | 4731 | 0.8426 | 0.8424 | 0.8813 |
| `roundabout` | 256 | 0.7539 | 0.7539 | 0.8308 |
| `ship` | 18534 | 0.8149 | 0.8140 | 0.8330 |
| `small-vehicle` | 11357 | 0.7326 | 0.7240 | 0.7851 |
| `soccer-ball-field` | 260 | 0.8039 | 0.7938 | 0.8731 |
| `storage-tank` | 5031 | 0.6311 | 0.6307 | 0.8123 |
| `swimming-pool` | 693 | 0.6452 | 0.6452 | 0.7045 |
| `tennis-court` | 1529 | 0.8973 | 0.8914 | 0.9321 |
| **global** | 57768 | 0.7760 | 0.7724 | 0.8271 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1086 | 0.948 | 0.7501 |
| `basketball-court` | 278 | 519 | 0.993 | 0.8748 |
| `bridge` | 666 | 4244 | 0.806 | 0.5928 |
| `ground-track-field` | 216 | 834 | 0.935 | 0.8116 |
| `harbor` | 4298 | 7558 | 0.899 | 0.7587 |
| `helicopter` | 157 | 336 | 0.924 | 0.8764 |
| `large-vehicle` | 9398 | 17309 | 0.957 | 0.8829 |
| `plane` | 4731 | 5969 | 0.960 | 0.8826 |
| `roundabout` | 256 | 712 | 0.887 | 0.6948 |
| `ship` | 18534 | 32733 | 0.983 | 0.7082 |
| `small-vehicle` | 11357 | 25934 | 0.915 | 0.7949 |
| `soccer-ball-field` | 260 | 830 | 0.915 | 0.8169 |
| `storage-tank` | 5031 | 7212 | 0.741 | 0.6904 |
| `swimming-pool` | 693 | 1997 | 0.843 | 0.6592 |
| `tennis-court` | 1529 | 1896 | 0.969 | 0.8544 |
| **mAP** | | | | 0.7766 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.8500 | 0.7014 | 0.8516 | 0.7692 | 310 | 132 | 54 |
| `basketball-court` | 0.8000 | 0.8469 | 0.9748 | 0.9064 | 271 | 49 | 7 |
| `bridge` | 0.8000 | 0.6390 | 0.6351 | 0.6370 | 423 | 239 | 243 |
| `ground-track-field` | 0.9000 | 0.7741 | 0.8565 | 0.8132 | 185 | 54 | 31 |
| `harbor` | 0.5500 | 0.7976 | 0.8113 | 0.8044 | 3487 | 885 | 811 |
| `helicopter` | 0.4500 | 0.9178 | 0.8535 | 0.8845 | 134 | 12 | 23 |
| `large-vehicle` | 0.7000 | 0.9056 | 0.8553 | 0.8797 | 8038 | 838 | 1360 |
| `plane` | 0.8500 | 0.9384 | 0.9245 | 0.9314 | 4374 | 287 | 357 |
| `roundabout` | 0.8000 | 0.7353 | 0.7812 | 0.7576 | 200 | 72 | 56 |
| `ship` | 0.6500 | 0.7016 | 0.9207 | 0.7964 | 17065 | 7259 | 1469 |
| `small-vehicle` | 0.4500 | 0.7946 | 0.7400 | 0.7663 | 8404 | 2173 | 2953 |
| `soccer-ball-field` | 0.8500 | 0.8238 | 0.7731 | 0.7976 | 201 | 43 | 59 |
| `storage-tank` | 0.5500 | 0.8640 | 0.6645 | 0.7512 | 3343 | 526 | 1688 |
| `swimming-pool` | 0.8000 | 0.7733 | 0.6349 | 0.6973 | 440 | 129 | 253 |
| `tennis-court` | 0.8500 | 0.9269 | 0.9457 | 0.9362 | 1446 | 114 | 83 |

## Confusion matrix

Computed at score threshold `0.6000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 334 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| `basketball-court` | 0 | 274 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 |
| `bridge` | 0 | 0 | 481 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 185 |
| `ground-track-field` | 0 | 0 | 0 | 196 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| `harbor` | 0 | 0 | 0 | 0 | 3416 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 878 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 127 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 27 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8297 | 0 | 0 | 1 | 39 | 0 | 0 | 0 | 0 | 1061 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 287 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 215 | 0 | 0 | 0 | 0 | 0 | 0 | 41 |
| `ship` | 0 | 0 | 2 | 0 | 1 | 0 | 2 | 0 | 0 | 17256 | 0 | 0 | 0 | 2 | 0 | 1271 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 134 | 0 | 0 | 0 | 7701 | 0 | 0 | 0 | 0 | 3522 |
| `soccer-ball-field` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 219 | 0 | 0 | 0 | 40 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3295 | 0 | 0 | 1735 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 519 | 0 | 174 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1462 | 57 |
| `False Positive` | 219 | 66 | 483 | 118 | 778 | 6 | 1036 | 426 | 117 | 7575 | 1333 | 99 | 466 | 361 | 138 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
