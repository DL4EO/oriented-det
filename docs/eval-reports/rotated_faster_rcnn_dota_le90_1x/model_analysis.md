# Model Analysis Report

- Generated at: `2026-07-10T06:10:33.659033`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260709-092028`
- Checkpoint: `runs/rotated_faster_rcnn/20260709-092028/checkpoints/best_mAP_0.83.pth`
- Checkpoint modified: `2026-07-09T21:03:28.154180`
- Config: `runs/rotated_faster_rcnn/20260709-092028/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `107912`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.30`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6500`
- Precision at best threshold: `0.7843`
- Recall at best threshold: `0.8272`
- F1 at best threshold: `0.8052`
- F2 at best threshold: `0.8182`
- mAP50: `0.7757` (77.57%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7415`
- Global mean best IoU (same class): `0.7384` (median `0.7876`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7360 | 0.7356 | 0.7565 |
| `basketball-court` | 278 | 0.8381 | 0.8381 | 0.8586 |
| `bridge` | 666 | 0.6436 | 0.6425 | 0.6988 |
| `ground-track-field` | 216 | 0.7946 | 0.7838 | 0.8333 |
| `harbor` | 4298 | 0.6896 | 0.6885 | 0.7244 |
| `helicopter` | 157 | 0.7354 | 0.7075 | 0.7170 |
| `large-vehicle` | 9398 | 0.7580 | 0.7514 | 0.7900 |
| `plane` | 4731 | 0.8196 | 0.8196 | 0.8460 |
| `roundabout` | 256 | 0.7391 | 0.7369 | 0.8194 |
| `ship` | 18534 | 0.7671 | 0.7662 | 0.7933 |
| `small-vehicle` | 11357 | 0.7201 | 0.7132 | 0.7648 |
| `soccer-ball-field` | 260 | 0.7438 | 0.7352 | 0.7940 |
| `storage-tank` | 5031 | 0.6167 | 0.6166 | 0.8001 |
| `swimming-pool` | 693 | 0.6253 | 0.6253 | 0.6773 |
| `tennis-court` | 1529 | 0.8763 | 0.8722 | 0.9036 |
| **global** | 57768 | 0.7415 | 0.7384 | 0.7876 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1139 | 0.959 | 0.7772 |
| `basketball-court` | 278 | 537 | 0.986 | 0.8773 |
| `bridge` | 666 | 3194 | 0.812 | 0.5885 |
| `ground-track-field` | 216 | 561 | 0.940 | 0.7913 |
| `harbor` | 4298 | 8824 | 0.884 | 0.7390 |
| `helicopter` | 157 | 337 | 0.949 | 0.9048 |
| `large-vehicle` | 9398 | 17451 | 0.936 | 0.8427 |
| `plane` | 4731 | 6005 | 0.984 | 0.8935 |
| `roundabout` | 256 | 644 | 0.867 | 0.6862 |
| `ship` | 18534 | 33207 | 0.961 | 0.6936 |
| `small-vehicle` | 11357 | 24667 | 0.919 | 0.8113 |
| `soccer-ball-field` | 260 | 640 | 0.888 | 0.7723 |
| `storage-tank` | 5031 | 6969 | 0.735 | 0.6928 |
| `swimming-pool` | 693 | 1728 | 0.840 | 0.6955 |
| `tennis-court` | 1529 | 2009 | 0.979 | 0.8691 |
| **mAP** | | | | 0.7757 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.8500 | 0.7532 | 0.8132 | 0.7820 | 296 | 97 | 68 |
| `basketball-court` | 0.8000 | 0.8684 | 0.9496 | 0.9072 | 264 | 40 | 14 |
| `bridge` | 0.8000 | 0.6514 | 0.6201 | 0.6354 | 413 | 221 | 253 |
| `ground-track-field` | 0.8500 | 0.7991 | 0.8102 | 0.8046 | 175 | 44 | 41 |
| `harbor` | 0.6500 | 0.8084 | 0.7764 | 0.7921 | 3337 | 791 | 961 |
| `helicopter` | 0.6000 | 0.9404 | 0.9045 | 0.9221 | 142 | 9 | 15 |
| `large-vehicle` | 0.7000 | 0.8632 | 0.8428 | 0.8529 | 7921 | 1255 | 1477 |
| `plane` | 0.7500 | 0.9336 | 0.9592 | 0.9462 | 4538 | 323 | 193 |
| `roundabout` | 0.6500 | 0.6940 | 0.7617 | 0.7263 | 195 | 86 | 61 |
| `ship` | 0.7500 | 0.6969 | 0.8967 | 0.7843 | 16619 | 7227 | 1915 |
| `small-vehicle` | 0.4000 | 0.7838 | 0.7842 | 0.7840 | 8906 | 2457 | 2451 |
| `soccer-ball-field` | 0.8000 | 0.8487 | 0.7769 | 0.8112 | 202 | 36 | 58 |
| `storage-tank` | 0.5000 | 0.8675 | 0.6704 | 0.7564 | 3373 | 515 | 1658 |
| `swimming-pool` | 0.6500 | 0.6901 | 0.7229 | 0.7061 | 501 | 225 | 192 |
| `tennis-court` | 0.6500 | 0.9155 | 0.9634 | 0.9388 | 1473 | 136 | 56 |

## Confusion matrix

Computed at score threshold `0.6500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 322 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 |
| `basketball-court` | 0 | 269 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| `bridge` | 0 | 0 | 461 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 205 |
| `ground-track-field` | 0 | 0 | 0 | 188 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
| `harbor` | 0 | 0 | 0 | 0 | 3337 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 955 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 137 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8040 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1318 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4572 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 195 | 0 | 0 | 0 | 0 | 0 | 0 | 61 |
| `ship` | 0 | 0 | 2 | 0 | 2 | 0 | 2 | 0 | 0 | 16954 | 0 | 0 | 0 | 2 | 0 | 1572 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 103 | 0 | 0 | 0 | 7877 | 0 | 0 | 0 | 0 | 3377 |
| `soccer-ball-field` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 210 | 0 | 0 | 0 | 48 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3246 | 0 | 0 | 1785 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 501 | 0 | 192 |
| `tennis-court` | 0 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1473 | 50 |
| `False Positive` | 172 | 56 | 343 | 72 | 788 | 5 | 1312 | 372 | 86 | 7796 | 1190 | 76 | 354 | 223 | 136 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
