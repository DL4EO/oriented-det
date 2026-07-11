# Model Analysis Report

- Generated at: `2026-06-15T02:38:20.099444`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260612-121232`
- Checkpoint: `runs/rotated_retinanet/20260612-121232/checkpoints/best_mAP_0.76.pth`
- Checkpoint modified: `2026-06-14T04:55:42.998250`
- Config: `runs/rotated_retinanet/20260612-121232/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `80190`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.5000`
- Precision at best threshold: `0.7296`
- Recall at best threshold: `0.7110`
- F1 at best threshold: `0.7202`
- F2 at best threshold: `0.7147`
- mAP50: `0.7152` (71.52%)

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 626 | 0.951 | 0.7819 |
| `basketball-court` | 278 | 353 | 0.960 | 0.8529 |
| `bridge` | 666 | 1441 | 0.784 | 0.5775 |
| `ground-track-field` | 216 | 344 | 0.949 | 0.8268 |
| `harbor` | 4298 | 7566 | 0.835 | 0.6983 |
| `helicopter` | 157 | 157 | 0.854 | 0.8103 |
| `large-vehicle` | 9398 | 12242 | 0.735 | 0.5644 |
| `plane` | 4731 | 5710 | 0.942 | 0.8807 |
| `roundabout` | 256 | 384 | 0.855 | 0.7267 |
| `ship` | 18534 | 26882 | 0.779 | 0.5079 |
| `small-vehicle` | 11357 | 14360 | 0.710 | 0.6468 |
| `soccer-ball-field` | 260 | 307 | 0.773 | 0.6992 |
| `storage-tank` | 5031 | 6927 | 0.711 | 0.6427 |
| `swimming-pool` | 693 | 1080 | 0.789 | 0.6398 |
| `tennis-court` | 1529 | 1811 | 0.969 | 0.8719 |
| **mAP** | | | | 0.7152 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.6500 | 0.8047 | 0.8379 | 0.8210 | 305 | 74 | 59 |
| `basketball-court` | 0.5500 | 0.8609 | 0.9353 | 0.8966 | 260 | 42 | 18 |
| `bridge` | 0.5000 | 0.6616 | 0.6517 | 0.6566 | 434 | 222 | 232 |
| `ground-track-field` | 0.4500 | 0.7967 | 0.9074 | 0.8485 | 196 | 50 | 20 |
| `harbor` | 0.5000 | 0.7478 | 0.7576 | 0.7527 | 3256 | 1098 | 1042 |
| `helicopter` | 0.4500 | 0.9624 | 0.8153 | 0.8828 | 128 | 5 | 29 |
| `large-vehicle` | 0.4500 | 0.6953 | 0.6924 | 0.6939 | 6507 | 2851 | 2891 |
| `plane` | 0.5500 | 0.9293 | 0.9091 | 0.9191 | 4301 | 327 | 430 |
| `roundabout` | 0.6500 | 0.7991 | 0.7148 | 0.7546 | 183 | 46 | 73 |
| `ship` | 0.6000 | 0.6484 | 0.7272 | 0.6855 | 13478 | 7309 | 5056 |
| `small-vehicle` | 0.5000 | 0.8217 | 0.6198 | 0.7066 | 7039 | 1527 | 4318 |
| `soccer-ball-field` | 0.5000 | 0.8720 | 0.7077 | 0.7813 | 184 | 27 | 76 |
| `storage-tank` | 0.4000 | 0.7134 | 0.6293 | 0.6687 | 3166 | 1272 | 1865 |
| `swimming-pool` | 0.5500 | 0.7895 | 0.6061 | 0.6857 | 420 | 112 | 273 |
| `tennis-court` | 0.6000 | 0.9312 | 0.9562 | 0.9435 | 1462 | 108 | 67 |

## Confusion matrix

Computed at score threshold `0.5000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 336 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
| `basketball-court` | 0 | 263 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 14 |
| `bridge` | 0 | 0 | 434 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 232 |
| `ground-track-field` | 0 | 0 | 0 | 185 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 28 |
| `harbor` | 0 | 0 | 0 | 0 | 3256 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1039 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 123 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 6204 | 0 | 0 | 0 | 33 | 0 | 0 | 0 | 0 | 3161 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4349 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 382 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 56 |
| `ship` | 0 | 0 | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 13969 | 0 | 0 | 0 | 1 | 0 | 4559 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 121 | 0 | 0 | 0 | 7027 | 0 | 0 | 0 | 0 | 4209 |
| `soccer-ball-field` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 182 | 0 | 0 | 0 | 76 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2620 | 0 | 0 | 2411 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 442 | 0 | 251 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1469 | 50 |
| `False Positive` | 153 | 45 | 221 | 37 | 1094 | 4 | 2326 | 380 | 85 | 8455 | 1506 | 26 | 440 | 166 | 118 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
