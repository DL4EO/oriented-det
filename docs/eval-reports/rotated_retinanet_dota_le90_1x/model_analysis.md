# Model Analysis Report

- Generated at: `2026-06-28T11:26:48.752928`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260611-101135`
- Checkpoint: `runs/rotated_retinanet/20260611-101135/checkpoints/best_mAP_0.70.pth`
- Checkpoint modified: `2026-06-12T11:24:04.304696`
- Config: `runs/rotated_retinanet/20260611-101135/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `703747`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.4500`
- Precision at best threshold: `0.7033`
- Recall at best threshold: `0.6859`
- F1 at best threshold: `0.6945`
- F2 at best threshold: `0.6893`
- mAP50: `0.6414` (64.14%)

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 6711 | 0.931 | 0.7514 |
| `basketball-court` | 278 | 2852 | 0.928 | 0.7922 |
| `bridge` | 666 | 114347 | 0.736 | 0.4758 |
| `ground-track-field` | 216 | 3569 | 0.931 | 0.7418 |
| `harbor` | 4298 | 42934 | 0.764 | 0.5918 |
| `helicopter` | 157 | 1764 | 0.624 | 0.5269 |
| `large-vehicle` | 9398 | 107431 | 0.705 | 0.4473 |
| `plane` | 4731 | 27172 | 0.940 | 0.8780 |
| `roundabout` | 256 | 4592 | 0.879 | 0.6201 |
| `ship` | 18534 | 89876 | 0.801 | 0.5316 |
| `small-vehicle` | 11357 | 213557 | 0.786 | 0.6182 |
| `soccer-ball-field` | 260 | 2336 | 0.754 | 0.6045 |
| `storage-tank` | 5031 | 69223 | 0.700 | 0.5837 |
| `swimming-pool` | 693 | 9201 | 0.795 | 0.5969 |
| `tennis-court` | 1529 | 8182 | 0.965 | 0.8604 |
| **mAP** | | | | 0.6414 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.5000 | 0.7338 | 0.8104 | 0.7702 | 295 | 107 | 69 |
| `basketball-court` | 0.5000 | 0.7936 | 0.8022 | 0.7979 | 223 | 58 | 55 |
| `bridge` | 0.4500 | 0.5678 | 0.5345 | 0.5507 | 356 | 271 | 310 |
| `ground-track-field` | 0.4500 | 0.7642 | 0.7500 | 0.7570 | 162 | 50 | 54 |
| `harbor` | 0.5000 | 0.7344 | 0.6368 | 0.6821 | 2737 | 990 | 1561 |
| `helicopter` | 0.3500 | 0.7611 | 0.5478 | 0.6370 | 86 | 27 | 71 |
| `large-vehicle` | 0.4000 | 0.6198 | 0.6232 | 0.6215 | 5857 | 3593 | 3541 |
| `plane` | 0.5500 | 0.9130 | 0.8916 | 0.9021 | 4218 | 402 | 513 |
| `roundabout` | 0.4000 | 0.6348 | 0.6992 | 0.6654 | 179 | 103 | 77 |
| `ship` | 0.5000 | 0.6416 | 0.7620 | 0.6966 | 14123 | 7890 | 4411 |
| `small-vehicle` | 0.4500 | 0.8003 | 0.6088 | 0.6915 | 6914 | 1725 | 4443 |
| `soccer-ball-field` | 0.4000 | 0.7812 | 0.5769 | 0.6637 | 150 | 42 | 110 |
| `storage-tank` | 0.4000 | 0.8047 | 0.5283 | 0.6379 | 2658 | 645 | 2373 |
| `swimming-pool` | 0.5000 | 0.7276 | 0.5512 | 0.6273 | 382 | 143 | 311 |
| `tennis-court` | 0.6000 | 0.9217 | 0.9313 | 0.9265 | 1424 | 121 | 105 |

## Confusion matrix

Computed at score threshold `0.4500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 305 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 59 |
| `basketball-court` | 0 | 228 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 48 |
| `bridge` | 0 | 0 | 356 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 310 |
| `ground-track-field` | 0 | 0 | 0 | 152 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 57 |
| `harbor` | 0 | 0 | 0 | 0 | 2888 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 1405 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 61 | 0 | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 5477 | 0 | 0 | 1 | 70 | 0 | 0 | 0 | 0 | 3850 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4294 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 437 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 168 | 0 | 0 | 0 | 0 | 0 | 0 | 88 |
| `ship` | 0 | 0 | 2 | 0 | 9 | 0 | 2 | 0 | 0 | 14374 | 0 | 0 | 0 | 0 | 0 | 4147 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 162 | 0 | 0 | 0 | 6900 | 0 | 0 | 0 | 0 | 4295 |
| `soccer-ball-field` | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 103 | 0 | 0 | 2 | 145 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2414 | 0 | 0 | 2617 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 413 | 0 | 280 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1445 | 74 |
| `False Positive` | 143 | 72 | 269 | 50 | 1427 | 11 | 2820 | 499 | 87 | 8601 | 1669 | 32 | 358 | 219 | 181 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
