# Model Analysis Report

- Generated at: `2026-09-01T06:50:26.240598`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260831-052647`
- Checkpoint: `runs/rotated_fcos/20260831-052647/checkpoints/best_mAP_0.82.pth`
- Checkpoint modified: `2026-09-01T04:11:52.270189`
- Config: `runs/rotated_fcos/20260831-052647/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `98253`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.8034`
- Recall at best threshold: `0.8998`
- F1 at best threshold: `0.8489`
- F2 at best threshold: `0.8787`
- mAP50: `0.8232` (82.32%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7921`
- Global mean best IoU (same class): `0.7901` (median `0.8215`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.8142 | 0.8142 | 0.8336 |
| `basketball-court` | 278 | 0.8994 | 0.8994 | 0.9103 |
| `bridge` | 666 | 0.7268 | 0.7267 | 0.7697 |
| `ground-track-field` | 216 | 0.7453 | 0.6901 | 0.8348 |
| `harbor` | 4298 | 0.7604 | 0.7589 | 0.7977 |
| `helicopter` | 157 | 0.8041 | 0.7955 | 0.8310 |
| `large-vehicle` | 9398 | 0.8183 | 0.8129 | 0.8367 |
| `plane` | 4731 | 0.8393 | 0.8393 | 0.8783 |
| `roundabout` | 256 | 0.8290 | 0.8290 | 0.8702 |
| `ship` | 18534 | 0.8029 | 0.8024 | 0.8202 |
| `small-vehicle` | 11357 | 0.7553 | 0.7525 | 0.7839 |
| `soccer-ball-field` | 260 | 0.8532 | 0.8445 | 0.8967 |
| `storage-tank` | 5031 | 0.7420 | 0.7417 | 0.8197 |
| `swimming-pool` | 693 | 0.7006 | 0.7006 | 0.7308 |
| `tennis-court` | 1529 | 0.9161 | 0.9161 | 0.9290 |
| **global** | 57768 | 0.7921 | 0.7901 | 0.8215 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1057 | 0.975 | 0.7607 |
| `basketball-court` | 278 | 457 | 1.000 | 0.9779 |
| `bridge` | 666 | 2381 | 0.899 | 0.7133 |
| `ground-track-field` | 216 | 446 | 0.806 | 0.6311 |
| `harbor` | 4298 | 7205 | 0.941 | 0.8443 |
| `helicopter` | 157 | 313 | 0.968 | 0.9060 |
| `large-vehicle` | 9398 | 15210 | 0.979 | 0.8868 |
| `plane` | 4731 | 5608 | 0.962 | 0.8921 |
| `roundabout` | 256 | 605 | 0.957 | 0.8248 |
| `ship` | 18534 | 32588 | 0.985 | 0.7342 |
| `small-vehicle` | 11357 | 20515 | 0.953 | 0.8581 |
| `soccer-ball-field` | 260 | 687 | 0.965 | 0.8925 |
| `storage-tank` | 5031 | 7766 | 0.882 | 0.7789 |
| `swimming-pool` | 693 | 1540 | 0.915 | 0.7794 |
| `tennis-court` | 1529 | 1875 | 0.995 | 0.8685 |
| **mAP** | | | | 0.8232 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.4500 | 0.7297 | 0.9121 | 0.8107 | 332 | 123 | 32 |
| `basketball-court` | 0.3500 | 0.9026 | 1.0000 | 0.9488 | 278 | 30 | 0 |
| `bridge` | 0.3000 | 0.7204 | 0.7778 | 0.7480 | 518 | 201 | 148 |
| `ground-track-field` | 0.2000 | 0.7120 | 0.6296 | 0.6683 | 136 | 55 | 80 |
| `harbor` | 0.2500 | 0.8417 | 0.9016 | 0.8706 | 3875 | 729 | 423 |
| `helicopter` | 0.3000 | 0.9792 | 0.8981 | 0.9369 | 141 | 3 | 16 |
| `large-vehicle` | 0.2500 | 0.9021 | 0.9337 | 0.9176 | 8775 | 952 | 623 |
| `plane` | 0.2500 | 0.9338 | 0.9419 | 0.9378 | 4456 | 316 | 275 |
| `roundabout` | 0.3500 | 0.7649 | 0.9023 | 0.8280 | 231 | 71 | 25 |
| `ship` | 0.3000 | 0.7071 | 0.9299 | 0.8033 | 17235 | 7139 | 1299 |
| `small-vehicle` | 0.2000 | 0.8302 | 0.8757 | 0.8523 | 9945 | 2034 | 1412 |
| `soccer-ball-field` | 0.3500 | 0.9255 | 0.9077 | 0.9165 | 236 | 19 | 24 |
| `storage-tank` | 0.2000 | 0.8815 | 0.8002 | 0.8389 | 4026 | 541 | 1005 |
| `swimming-pool` | 0.3000 | 0.7709 | 0.7576 | 0.7642 | 525 | 156 | 168 |
| `tennis-court` | 0.3000 | 0.9300 | 0.9817 | 0.9551 | 1501 | 113 | 28 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 353 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| `basketball-court` | 0 | 278 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bridge` | 0 | 0 | 555 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 111 |
| `ground-track-field` | 0 | 0 | 0 | 126 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 83 |
| `harbor` | 0 | 0 | 0 | 0 | 3874 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 422 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8773 | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 565 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4456 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 235 | 0 | 0 | 0 | 0 | 0 | 0 | 21 |
| `ship` | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 17683 | 0 | 0 | 0 | 2 | 0 | 846 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | 0 | 1 | 9356 | 0 | 0 | 0 | 0 | 1950 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 241 | 0 | 0 | 0 | 19 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3832 | 0 | 0 | 1199 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 548 | 0 | 145 |
| `tennis-court` | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1509 | 17 |
| `False Positive` | 203 | 43 | 287 | 51 | 727 | 10 | 903 | 315 | 95 | 7869 | 1353 | 38 | 377 | 211 | 125 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
