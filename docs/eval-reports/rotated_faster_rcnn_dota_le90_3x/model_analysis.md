# Model Analysis Report

- Generated at: `2026-07-05T07:17:01.454945`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260703-075435`
- Checkpoint: `runs/rotated_faster_rcnn/20260703-075435/checkpoints/best_mAP_0.88.pth`
- Checkpoint modified: `2026-07-04T18:36:33.176335`
- Config: `runs/rotated_faster_rcnn/20260703-075435/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `90503`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.30`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.7000`
- Precision at best threshold: `0.8135`
- Recall at best threshold: `0.8917`
- F1 at best threshold: `0.8508`
- F2 at best threshold: `0.8749`
- mAP50: `0.8342` (83.42%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7734`
- Global mean best IoU (same class): `0.7720` (median `0.8141`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7806 | 0.7806 | 0.7809 |
| `basketball-court` | 278 | 0.8701 | 0.8701 | 0.8750 |
| `bridge` | 666 | 0.7415 | 0.7415 | 0.7881 |
| `ground-track-field` | 216 | 0.8305 | 0.8196 | 0.8581 |
| `harbor` | 4298 | 0.7464 | 0.7455 | 0.7814 |
| `helicopter` | 157 | 0.7847 | 0.7744 | 0.7893 |
| `large-vehicle` | 9398 | 0.7958 | 0.7914 | 0.8212 |
| `plane` | 4731 | 0.8450 | 0.8448 | 0.8685 |
| `roundabout` | 256 | 0.8055 | 0.8055 | 0.8768 |
| `ship` | 18534 | 0.7941 | 0.7938 | 0.8160 |
| `small-vehicle` | 11357 | 0.7488 | 0.7469 | 0.7847 |
| `soccer-ball-field` | 260 | 0.7936 | 0.7879 | 0.8379 |
| `storage-tank` | 5031 | 0.6378 | 0.6376 | 0.8187 |
| `swimming-pool` | 693 | 0.6599 | 0.6599 | 0.7139 |
| `tennis-court` | 1529 | 0.8978 | 0.8968 | 0.9175 |
| **global** | 57768 | 0.7734 | 0.7720 | 0.8141 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 762 | 0.986 | 0.8010 |
| `basketball-court` | 278 | 410 | 1.000 | 0.9616 |
| `bridge` | 666 | 2209 | 0.929 | 0.7790 |
| `ground-track-field` | 216 | 430 | 0.958 | 0.8467 |
| `harbor` | 4298 | 6965 | 0.943 | 0.8424 |
| `helicopter` | 157 | 206 | 0.975 | 0.9091 |
| `large-vehicle` | 9398 | 13855 | 0.962 | 0.8767 |
| `plane` | 4731 | 5455 | 0.991 | 0.8902 |
| `roundabout` | 256 | 504 | 0.922 | 0.8111 |
| `ship` | 18534 | 30351 | 0.977 | 0.7519 |
| `small-vehicle` | 11357 | 19449 | 0.953 | 0.8753 |
| `soccer-ball-field` | 260 | 479 | 0.938 | 0.8879 |
| `storage-tank` | 5031 | 6209 | 0.750 | 0.7007 |
| `swimming-pool` | 693 | 1441 | 0.869 | 0.7031 |
| `tennis-court` | 1529 | 1778 | 0.995 | 0.8764 |
| **mAP** | | | | 0.8342 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.9000 | 0.7435 | 0.9396 | 0.8301 | 342 | 118 | 22 |
| `basketball-court` | 0.9500 | 0.9097 | 0.9784 | 0.9428 | 272 | 27 | 6 |
| `bridge` | 0.9000 | 0.7731 | 0.7523 | 0.7626 | 501 | 147 | 165 |
| `ground-track-field` | 0.9000 | 0.8292 | 0.9213 | 0.8728 | 199 | 41 | 17 |
| `harbor` | 0.7500 | 0.8642 | 0.8704 | 0.8673 | 3741 | 588 | 557 |
| `helicopter` | 0.6000 | 0.9740 | 0.9554 | 0.9646 | 150 | 4 | 7 |
| `large-vehicle` | 0.8000 | 0.9158 | 0.9147 | 0.9152 | 8596 | 790 | 802 |
| `plane` | 0.7500 | 0.9457 | 0.9725 | 0.9589 | 4601 | 264 | 130 |
| `roundabout` | 0.7500 | 0.7213 | 0.8594 | 0.7843 | 220 | 85 | 36 |
| `ship` | 0.8000 | 0.7133 | 0.9348 | 0.8092 | 17326 | 6964 | 1208 |
| `small-vehicle` | 0.6000 | 0.8694 | 0.8596 | 0.8645 | 9763 | 1466 | 1594 |
| `soccer-ball-field` | 0.9500 | 0.9646 | 0.8385 | 0.8971 | 218 | 8 | 42 |
| `storage-tank` | 0.5500 | 0.8872 | 0.6814 | 0.7708 | 3428 | 436 | 1603 |
| `swimming-pool` | 0.8000 | 0.7344 | 0.7460 | 0.7402 | 517 | 187 | 176 |
| `tennis-court` | 0.8000 | 0.9240 | 0.9856 | 0.9538 | 1507 | 124 | 22 |

## Confusion matrix

Computed at score threshold `0.7000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 351 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| `basketball-court` | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `bridge` | 0 | 0 | 564 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 102 |
| `ground-track-field` | 0 | 0 | 0 | 201 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| `harbor` | 0 | 0 | 0 | 0 | 3789 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 506 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 147 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8700 | 0 | 0 | 0 | 39 | 0 | 0 | 0 | 0 | 659 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4608 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 123 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 222 | 0 | 0 | 0 | 0 | 0 | 0 | 34 |
| `ship` | 0 | 0 | 2 | 0 | 3 | 0 | 1 | 0 | 0 | 17565 | 0 | 0 | 0 | 2 | 0 | 961 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 9456 | 0 | 0 | 0 | 0 | 1871 |
| `soccer-ball-field` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 238 | 0 | 0 | 0 | 21 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3336 | 0 | 0 | 1695 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 543 | 0 | 150 |
| `tennis-court` | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1508 | 17 |
| `False Positive` | 171 | 40 | 302 | 58 | 647 | 3 | 954 | 270 | 96 | 7409 | 1065 | 40 | 307 | 238 | 132 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
