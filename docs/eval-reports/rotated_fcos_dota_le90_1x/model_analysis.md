# Model Analysis Report

Hub slug **`rotated_fcos_dota_le90_1x`**. Official DOTA v1.0 Task 1 **73.07%** (AP75 40.40). Deploy `production.score_threshold` **0.2** (this sweep’s best F1 0.25 − 0.05). Val tiles are in the train union; do not quote this report’s mAP50 as held-out accuracy.

- Generated at: `2026-09-08T14:49:29.461999`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260908-023531`
- Checkpoint: `runs/rotated_fcos/20260908-023531/checkpoints/best_mAP_0.78.pth`
- Checkpoint modified: `2026-09-08T10:35:40.191581`
- Config: `runs/rotated_fcos/20260908-023531/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `132861`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.7700`
- Recall at best threshold: `0.8364`
- F1 at best threshold: `0.8018`
- F2 at best threshold: `0.8222`
- mAP50: `0.7513` (75.13%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7543`
- Global mean best IoU (same class): `0.7495` (median `0.7940`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7431 | 0.7430 | 0.7604 |
| `basketball-court` | 278 | 0.8470 | 0.8467 | 0.8818 |
| `bridge` | 666 | 0.6397 | 0.6380 | 0.6854 |
| `ground-track-field` | 216 | 0.7082 | 0.6242 | 0.7779 |
| `harbor` | 4298 | 0.6841 | 0.6812 | 0.7163 |
| `helicopter` | 157 | 0.7278 | 0.6957 | 0.7470 |
| `large-vehicle` | 9398 | 0.7804 | 0.7693 | 0.8064 |
| `plane` | 4731 | 0.8180 | 0.8180 | 0.8590 |
| `roundabout` | 256 | 0.7506 | 0.7506 | 0.8257 |
| `ship` | 18534 | 0.7779 | 0.7767 | 0.8033 |
| `small-vehicle` | 11357 | 0.7173 | 0.7084 | 0.7575 |
| `soccer-ball-field` | 260 | 0.7709 | 0.7496 | 0.8348 |
| `storage-tank` | 5031 | 0.6917 | 0.6911 | 0.7875 |
| `swimming-pool` | 693 | 0.6333 | 0.6333 | 0.6822 |
| `tennis-court` | 1529 | 0.8875 | 0.8824 | 0.9158 |
| **global** | 57768 | 0.7543 | 0.7495 | 0.7940 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 2011 | 0.942 | 0.7322 |
| `basketball-court` | 278 | 823 | 0.978 | 0.8767 |
| `bridge` | 666 | 5480 | 0.824 | 0.5802 |
| `ground-track-field` | 216 | 1108 | 0.741 | 0.5220 |
| `harbor` | 4298 | 8922 | 0.879 | 0.7332 |
| `helicopter` | 157 | 460 | 0.911 | 0.8611 |
| `large-vehicle` | 9398 | 23457 | 0.953 | 0.8686 |
| `plane` | 4731 | 6833 | 0.954 | 0.8895 |
| `roundabout` | 256 | 1264 | 0.883 | 0.6826 |
| `ship` | 18534 | 36997 | 0.968 | 0.6868 |
| `small-vehicle` | 11357 | 29887 | 0.905 | 0.7882 |
| `soccer-ball-field` | 260 | 1396 | 0.892 | 0.7688 |
| `storage-tank` | 5031 | 9575 | 0.828 | 0.7626 |
| `swimming-pool` | 693 | 2341 | 0.850 | 0.6525 |
| `tennis-court` | 1529 | 2307 | 0.976 | 0.8647 |
| **mAP** | | | | 0.7513 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.4000 | 0.7368 | 0.8077 | 0.7706 | 294 | 105 | 70 |
| `basketball-court` | 0.3500 | 0.8586 | 0.9173 | 0.8870 | 255 | 42 | 23 |
| `bridge` | 0.3000 | 0.6696 | 0.5721 | 0.6170 | 381 | 188 | 285 |
| `ground-track-field` | 0.2500 | 0.6630 | 0.5648 | 0.6100 | 122 | 62 | 94 |
| `harbor` | 0.2500 | 0.7720 | 0.7955 | 0.7835 | 3419 | 1010 | 879 |
| `helicopter` | 0.2500 | 0.8544 | 0.8599 | 0.8571 | 135 | 23 | 22 |
| `large-vehicle` | 0.2500 | 0.8486 | 0.8738 | 0.8610 | 8212 | 1465 | 1186 |
| `plane` | 0.3000 | 0.9234 | 0.9218 | 0.9226 | 4361 | 362 | 370 |
| `roundabout` | 0.3500 | 0.7107 | 0.7773 | 0.7425 | 199 | 81 | 57 |
| `ship` | 0.2500 | 0.6805 | 0.9148 | 0.7804 | 16954 | 7959 | 1580 |
| `small-vehicle` | 0.2000 | 0.7690 | 0.7808 | 0.7748 | 8867 | 2664 | 2490 |
| `soccer-ball-field` | 0.3000 | 0.7873 | 0.8115 | 0.7992 | 211 | 57 | 49 |
| `storage-tank` | 0.2000 | 0.8454 | 0.7380 | 0.7881 | 3713 | 679 | 1318 |
| `swimming-pool` | 0.3500 | 0.7568 | 0.6017 | 0.6704 | 417 | 134 | 276 |
| `tennis-court` | 0.2500 | 0.9179 | 0.9575 | 0.9373 | 1464 | 131 | 65 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 331 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 33 |
| `basketball-court` | 0 | 264 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 12 |
| `bridge` | 0 | 0 | 432 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 234 |
| `ground-track-field` | 0 | 0 | 0 | 114 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 87 |
| `harbor` | 0 | 0 | 0 | 0 | 3418 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 876 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 135 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8208 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 1101 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4404 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 327 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 215 | 0 | 0 | 0 | 0 | 0 | 0 | 41 |
| `ship` | 0 | 0 | 2 | 0 | 4 | 0 | 1 | 0 | 0 | 16954 | 0 | 0 | 0 | 2 | 0 | 1571 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 163 | 0 | 0 | 0 | 8179 | 0 | 0 | 0 | 0 | 3015 |
| `soccer-ball-field` | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 211 | 0 | 0 | 0 | 43 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3459 | 0 | 0 | 1572 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 496 | 0 | 197 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1464 | 55 |
| `False Positive` | 256 | 66 | 312 | 64 | 1006 | 23 | 1305 | 441 | 150 | 7955 | 1631 | 97 | 440 | 293 | 130 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
