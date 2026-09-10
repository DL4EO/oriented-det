# Model Analysis Report

Hub slug **`rotated_faster_rcnn_dota_le90_1x`**. Official DOTA v1.0 Task 1 **74.42%** (AP75 41.90). Deploy `production.score_threshold` **0.6** (this sweep’s best F1 0.65 − 0.05).

- Generated at: `2026-09-08T03:12:11.406984`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260907-124458`
- Checkpoint: `runs/rotated_faster_rcnn/20260907-124458/checkpoints/best_mAP_0.83.pth`
- Checkpoint modified: `2026-09-08T00:16:10.336852`
- Config: `runs/rotated_faster_rcnn/20260907-124458/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `105553`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6500`
- Precision at best threshold: `0.7922`
- Recall at best threshold: `0.8301`
- F1 at best threshold: `0.8107`
- F2 at best threshold: `0.8223`
- mAP50: `0.7755` (77.55%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7498`
- Global mean best IoU (same class): `0.7465` (median `0.8033`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7609 | 0.7607 | 0.7948 |
| `basketball-court` | 278 | 0.8547 | 0.8543 | 0.8777 |
| `bridge` | 666 | 0.6590 | 0.6577 | 0.7213 |
| `ground-track-field` | 216 | 0.7947 | 0.7843 | 0.8464 |
| `harbor` | 4298 | 0.6929 | 0.6913 | 0.7383 |
| `helicopter` | 157 | 0.7073 | 0.6854 | 0.7256 |
| `large-vehicle` | 9398 | 0.7724 | 0.7645 | 0.8089 |
| `plane` | 4731 | 0.8187 | 0.8187 | 0.8600 |
| `roundabout` | 256 | 0.7380 | 0.7265 | 0.8161 |
| `ship` | 18534 | 0.7814 | 0.7805 | 0.8121 |
| `small-vehicle` | 11357 | 0.7237 | 0.7171 | 0.7738 |
| `soccer-ball-field` | 260 | 0.7662 | 0.7550 | 0.8153 |
| `storage-tank` | 5031 | 0.6121 | 0.6121 | 0.7941 |
| `swimming-pool` | 693 | 0.6337 | 0.6337 | 0.6889 |
| `tennis-court` | 1529 | 0.8891 | 0.8838 | 0.9203 |
| **global** | 57768 | 0.7498 | 0.7465 | 0.8033 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 935 | 0.953 | 0.7646 |
| `basketball-court` | 278 | 482 | 0.993 | 0.8803 |
| `bridge` | 666 | 3103 | 0.817 | 0.5962 |
| `ground-track-field` | 216 | 594 | 0.935 | 0.7771 |
| `harbor` | 4298 | 7253 | 0.874 | 0.7507 |
| `helicopter` | 157 | 302 | 0.911 | 0.8611 |
| `large-vehicle` | 9398 | 16585 | 0.936 | 0.8702 |
| `plane` | 4731 | 5901 | 0.958 | 0.8930 |
| `roundabout` | 256 | 671 | 0.855 | 0.6868 |
| `ship` | 18534 | 31696 | 0.962 | 0.7098 |
| `small-vehicle` | 11357 | 26890 | 0.917 | 0.8068 |
| `soccer-ball-field` | 260 | 604 | 0.915 | 0.8180 |
| `storage-tank` | 5031 | 6820 | 0.728 | 0.6886 |
| `swimming-pool` | 693 | 1799 | 0.825 | 0.6634 |
| `tennis-court` | 1529 | 1918 | 0.974 | 0.8667 |
| **mAP** | | | | 0.7755 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.8000 | 0.7083 | 0.8407 | 0.7688 | 306 | 126 | 58 |
| `basketball-court` | 0.7500 | 0.8428 | 0.9640 | 0.8993 | 268 | 50 | 10 |
| `bridge` | 0.7500 | 0.6272 | 0.6441 | 0.6356 | 429 | 255 | 237 |
| `ground-track-field` | 0.8000 | 0.7702 | 0.8380 | 0.8027 | 181 | 54 | 35 |
| `harbor` | 0.6000 | 0.8053 | 0.8043 | 0.8048 | 3457 | 836 | 841 |
| `helicopter` | 0.4500 | 0.9195 | 0.8726 | 0.8954 | 137 | 12 | 20 |
| `large-vehicle` | 0.7000 | 0.8929 | 0.8515 | 0.8717 | 8002 | 960 | 1396 |
| `plane` | 0.8000 | 0.9396 | 0.9239 | 0.9317 | 4371 | 281 | 360 |
| `roundabout` | 0.7000 | 0.7007 | 0.7500 | 0.7245 | 192 | 82 | 64 |
| `ship` | 0.6500 | 0.6942 | 0.9152 | 0.7895 | 16963 | 7474 | 1571 |
| `small-vehicle` | 0.5000 | 0.8022 | 0.7648 | 0.7831 | 8686 | 2142 | 2671 |
| `soccer-ball-field` | 0.7000 | 0.8104 | 0.8385 | 0.8242 | 218 | 51 | 42 |
| `storage-tank` | 0.5500 | 0.8733 | 0.6565 | 0.7496 | 3303 | 479 | 1728 |
| `swimming-pool` | 0.7500 | 0.7288 | 0.6320 | 0.6770 | 438 | 163 | 255 |
| `tennis-court` | 0.6000 | 0.9173 | 0.9653 | 0.9407 | 1476 | 133 | 53 |

## Confusion matrix

Computed at score threshold `0.6500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 321 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 |
| `basketball-court` | 0 | 270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `bridge` | 0 | 0 | 456 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 210 |
| `ground-track-field` | 0 | 0 | 0 | 186 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 28 |
| `harbor` | 0 | 0 | 0 | 0 | 3388 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 907 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 123 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8109 | 0 | 0 | 1 | 45 | 0 | 0 | 0 | 0 | 1243 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4421 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 310 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 197 | 0 | 0 | 0 | 0 | 0 | 0 | 59 |
| `ship` | 0 | 0 | 2 | 0 | 3 | 0 | 2 | 0 | 0 | 16963 | 0 | 0 | 0 | 1 | 0 | 1563 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 0 | 8121 | 0 | 0 | 0 | 0 | 3146 |
| `soccer-ball-field` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 221 | 0 | 0 | 0 | 38 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3220 | 0 | 0 | 1811 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 482 | 0 | 211 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1472 | 47 |
| `False Positive` | 168 | 56 | 340 | 72 | 731 | 7 | 1016 | 349 | 92 | 7470 | 1296 | 62 | 367 | 256 | 129 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
