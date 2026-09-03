# Model Analysis Report

- Generated at: `2026-09-03T02:04:34.825330`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260901-095802`
- Checkpoint: `runs/rotated_faster_rcnn/20260901-095802/checkpoints/best_mAP_0.88.pth`
- Checkpoint modified: `2026-09-02T20:15:25.355553`
- Config: `runs/rotated_faster_rcnn/20260901-095802/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `86439`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6500`
- Precision at best threshold: `0.8169`
- Recall at best threshold: `0.8885`
- F1 at best threshold: `0.8512`
- F2 at best threshold: `0.8732`
- mAP50: `0.8346` (83.46%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7784`
- Global mean best IoU (same class): `0.7772` (median `0.8253`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.8102 | 0.8102 | 0.8320 |
| `basketball-court` | 278 | 0.8922 | 0.8922 | 0.9107 |
| `bridge` | 666 | 0.7583 | 0.7583 | 0.8025 |
| `ground-track-field` | 216 | 0.8364 | 0.8287 | 0.8842 |
| `harbor` | 4298 | 0.7553 | 0.7546 | 0.7949 |
| `helicopter` | 157 | 0.7858 | 0.7858 | 0.8076 |
| `large-vehicle` | 9398 | 0.8056 | 0.8019 | 0.8324 |
| `plane` | 4731 | 0.8456 | 0.8455 | 0.8837 |
| `roundabout` | 256 | 0.8098 | 0.8074 | 0.8807 |
| `ship` | 18534 | 0.8006 | 0.8004 | 0.8267 |
| `small-vehicle` | 11357 | 0.7461 | 0.7444 | 0.7912 |
| `soccer-ball-field` | 260 | 0.8282 | 0.8227 | 0.8870 |
| `storage-tank` | 5031 | 0.6354 | 0.6353 | 0.8227 |
| `swimming-pool` | 693 | 0.6692 | 0.6692 | 0.7197 |
| `tennis-court` | 1529 | 0.9168 | 0.9150 | 0.9321 |
| **global** | 57768 | 0.7784 | 0.7772 | 0.8253 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 766 | 0.986 | 0.7941 |
| `basketball-court` | 278 | 390 | 1.000 | 0.9490 |
| `bridge` | 666 | 2216 | 0.907 | 0.7691 |
| `ground-track-field` | 216 | 417 | 0.954 | 0.8406 |
| `harbor` | 4298 | 6066 | 0.936 | 0.8505 |
| `helicopter` | 157 | 204 | 0.975 | 0.9091 |
| `large-vehicle` | 9398 | 12923 | 0.964 | 0.8919 |
| `plane` | 4731 | 5326 | 0.965 | 0.8914 |
| `roundabout` | 256 | 540 | 0.918 | 0.8048 |
| `ship` | 18534 | 29123 | 0.972 | 0.7507 |
| `small-vehicle` | 11357 | 18905 | 0.941 | 0.8695 |
| `soccer-ball-field` | 260 | 449 | 0.942 | 0.8863 |
| `storage-tank` | 5031 | 6025 | 0.746 | 0.7024 |
| `swimming-pool` | 693 | 1336 | 0.880 | 0.7326 |
| `tennis-court` | 1529 | 1753 | 0.996 | 0.8764 |
| **mAP** | | | | 0.8346 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.9500 | 0.7564 | 0.8874 | 0.8167 | 323 | 104 | 41 |
| `basketball-court` | 0.9000 | 0.8977 | 0.9784 | 0.9363 | 272 | 31 | 6 |
| `bridge` | 0.9000 | 0.7824 | 0.7613 | 0.7717 | 507 | 141 | 159 |
| `ground-track-field` | 0.9000 | 0.8082 | 0.9167 | 0.8590 | 198 | 47 | 18 |
| `harbor` | 0.5500 | 0.8364 | 0.8990 | 0.8666 | 3864 | 756 | 434 |
| `helicopter` | 0.6000 | 0.9869 | 0.9618 | 0.9742 | 151 | 2 | 6 |
| `large-vehicle` | 0.8500 | 0.9483 | 0.9031 | 0.9251 | 8487 | 463 | 911 |
| `plane` | 0.6500 | 0.9418 | 0.9508 | 0.9463 | 4498 | 278 | 233 |
| `roundabout` | 0.9000 | 0.8071 | 0.8008 | 0.8039 | 205 | 49 | 51 |
| `ship` | 0.7500 | 0.7142 | 0.9330 | 0.8091 | 17292 | 6919 | 1242 |
| `small-vehicle` | 0.6000 | 0.8789 | 0.8381 | 0.8580 | 9518 | 1311 | 1839 |
| `soccer-ball-field` | 0.8000 | 0.8848 | 0.9154 | 0.8998 | 238 | 31 | 22 |
| `storage-tank` | 0.5500 | 0.8935 | 0.6838 | 0.7747 | 3440 | 410 | 1591 |
| `swimming-pool` | 0.7500 | 0.7727 | 0.7605 | 0.7665 | 527 | 155 | 166 |
| `tennis-court` | 0.8000 | 0.9302 | 0.9856 | 0.9571 | 1507 | 113 | 22 |

## Confusion matrix

Computed at score threshold `0.6500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 351 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| `basketball-court` | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `bridge` | 0 | 0 | 572 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 |
| `ground-track-field` | 0 | 0 | 0 | 201 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| `harbor` | 0 | 0 | 0 | 0 | 3763 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 533 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 149 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8741 | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | 619 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4498 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 233 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 223 | 0 | 0 | 0 | 0 | 0 | 0 | 33 |
| `ship` | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 0 | 0 | 17492 | 0 | 0 | 0 | 0 | 0 | 1038 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 9368 | 0 | 0 | 0 | 0 | 1953 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 241 | 0 | 0 | 0 | 19 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3380 | 0 | 0 | 1651 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 552 | 0 | 141 |
| `tennis-court` | 0 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1509 | 14 |
| `False Positive` | 181 | 35 | 328 | 65 | 663 | 2 | 745 | 278 | 90 | 7251 | 1106 | 41 | 325 | 201 | 122 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
