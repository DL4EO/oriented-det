# Model Analysis Report

- Generated at: `2026-08-13T20:57:50.742979`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260812-105204`
- Checkpoint: `runs/rotated_fcos/20260812-105204/checkpoints/best_mAP_0.72.pth`
- Checkpoint modified: `2026-08-13T09:02:46.853466`
- Config: `runs/rotated_fcos/20260812-105204/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `146984`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.7374`
- Recall at best threshold: `0.8028`
- F1 at best threshold: `0.7687`
- F2 at best threshold: `0.7888`
- mAP50: `0.7392` (73.92%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7167`
- Global mean best IoU (same class): `0.7127` (median `0.7680`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7472 | 0.7472 | 0.7471 |
| `basketball-court` | 278 | 0.8119 | 0.8118 | 0.8559 |
| `bridge` | 666 | 0.6254 | 0.6228 | 0.6842 |
| `ground-track-field` | 216 | 0.6665 | 0.6088 | 0.6824 |
| `harbor` | 4298 | 0.6505 | 0.6475 | 0.6787 |
| `helicopter` | 157 | 0.7339 | 0.7055 | 0.7631 |
| `large-vehicle` | 9398 | 0.7320 | 0.7239 | 0.7768 |
| `plane` | 4731 | 0.7957 | 0.7954 | 0.8408 |
| `roundabout` | 256 | 0.8173 | 0.8171 | 0.8639 |
| `ship` | 18534 | 0.7314 | 0.7300 | 0.7764 |
| `small-vehicle` | 11357 | 0.6834 | 0.6760 | 0.7380 |
| `soccer-ball-field` | 260 | 0.7633 | 0.7547 | 0.8244 |
| `storage-tank` | 5031 | 0.6633 | 0.6631 | 0.7565 |
| `swimming-pool` | 693 | 0.6289 | 0.6289 | 0.6607 |
| `tennis-court` | 1529 | 0.8439 | 0.8389 | 0.8848 |
| **global** | 57768 | 0.7167 | 0.7127 | 0.7680 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 2027 | 0.978 | 0.7805 |
| `basketball-court` | 278 | 1071 | 0.960 | 0.8710 |
| `bridge` | 666 | 6666 | 0.763 | 0.5564 |
| `ground-track-field` | 216 | 1329 | 0.704 | 0.4721 |
| `harbor` | 4298 | 11574 | 0.816 | 0.7077 |
| `helicopter` | 157 | 611 | 0.892 | 0.8017 |
| `large-vehicle` | 9398 | 24957 | 0.899 | 0.7716 |
| `plane` | 4731 | 7679 | 0.951 | 0.8826 |
| `roundabout` | 256 | 1409 | 0.953 | 0.7809 |
| `ship` | 18534 | 41479 | 0.904 | 0.6742 |
| `small-vehicle` | 11357 | 30593 | 0.836 | 0.7210 |
| `soccer-ball-field` | 260 | 1419 | 0.904 | 0.8420 |
| `storage-tank` | 5031 | 9783 | 0.783 | 0.6922 |
| `swimming-pool` | 693 | 3059 | 0.846 | 0.6697 |
| `tennis-court` | 1529 | 3327 | 0.975 | 0.8640 |
| **mAP** | | | | 0.7392 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.4000 | 0.7646 | 0.8297 | 0.7958 | 302 | 93 | 62 |
| `basketball-court` | 0.3500 | 0.8533 | 0.9209 | 0.8858 | 256 | 44 | 22 |
| `bridge` | 0.3000 | 0.6661 | 0.5721 | 0.6155 | 381 | 191 | 285 |
| `ground-track-field` | 0.3000 | 0.6646 | 0.4861 | 0.5615 | 105 | 53 | 111 |
| `harbor` | 0.2500 | 0.7358 | 0.7569 | 0.7462 | 3253 | 1168 | 1045 |
| `helicopter` | 0.3000 | 0.9179 | 0.7834 | 0.8454 | 123 | 11 | 34 |
| `large-vehicle` | 0.2500 | 0.8134 | 0.8263 | 0.8198 | 7766 | 1782 | 1632 |
| `plane` | 0.3500 | 0.9223 | 0.9034 | 0.9128 | 4274 | 360 | 457 |
| `roundabout` | 0.3500 | 0.7308 | 0.8164 | 0.7712 | 209 | 77 | 47 |
| `ship` | 0.3000 | 0.6647 | 0.8284 | 0.7376 | 15354 | 7746 | 3180 |
| `small-vehicle` | 0.2500 | 0.7889 | 0.6879 | 0.7350 | 7813 | 2091 | 3544 |
| `soccer-ball-field` | 0.3000 | 0.8240 | 0.8462 | 0.8349 | 220 | 47 | 40 |
| `storage-tank` | 0.2000 | 0.8229 | 0.7122 | 0.7636 | 3583 | 771 | 1448 |
| `swimming-pool` | 0.2500 | 0.6281 | 0.7359 | 0.6777 | 510 | 302 | 183 |
| `tennis-court` | 0.3500 | 0.9272 | 0.9411 | 0.9341 | 1439 | 113 | 90 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 341 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 |
| `basketball-court` | 0 | 263 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| `bridge` | 0 | 0 | 426 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 240 |
| `ground-track-field` | 0 | 1 | 0 | 106 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 100 |
| `harbor` | 0 | 0 | 0 | 0 | 3253 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1043 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 129 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 7762 | 0 | 0 | 0 | 85 | 0 | 0 | 0 | 0 | 1551 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4385 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 346 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 230 | 0 | 0 | 0 | 1 | 0 | 0 | 25 |
| `ship` | 0 | 0 | 2 | 0 | 3 | 0 | 2 | 0 | 0 | 16071 | 1 | 0 | 0 | 0 | 0 | 2455 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 127 | 0 | 0 | 1 | 7808 | 0 | 0 | 0 | 0 | 3421 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 226 | 0 | 0 | 0 | 34 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3391 | 0 | 0 | 1639 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 510 | 0 | 183 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1461 | 58 |
| `False Positive` | 205 | 68 | 307 | 80 | 1164 | 22 | 1657 | 522 | 145 | 9102 | 2010 | 67 | 476 | 302 | 152 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
