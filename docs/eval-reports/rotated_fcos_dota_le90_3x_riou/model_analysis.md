# Model Analysis Report

- Generated at: `2026-08-24T22:48:49.420880`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260822-153943`
- Checkpoint: `runs/rotated_fcos/20260822-153943/checkpoints/best_mAP_0.88.pth`
- Checkpoint modified: `2026-08-23T18:01:23.063689`
- Config: `runs/rotated_fcos/20260822-153943/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `96432`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.8021`
- Recall at best threshold: `0.8924`
- F1 at best threshold: `0.8449`
- F2 at best threshold: `0.8728`
- mAP50: `0.8158` (81.58%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7836`
- Global mean best IoU (same class): `0.7817` (median `0.8159`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7764 | 0.7764 | 0.7702 |
| `basketball-court` | 278 | 0.8810 | 0.8806 | 0.8963 |
| `bridge` | 666 | 0.7118 | 0.7118 | 0.7556 |
| `ground-track-field` | 216 | 0.7418 | 0.7033 | 0.8234 |
| `harbor` | 4298 | 0.7466 | 0.7455 | 0.7822 |
| `helicopter` | 157 | 0.7536 | 0.7399 | 0.7889 |
| `large-vehicle` | 9398 | 0.8076 | 0.8023 | 0.8269 |
| `plane` | 4731 | 0.8354 | 0.8354 | 0.8713 |
| `roundabout` | 256 | 0.8379 | 0.8379 | 0.8695 |
| `ship` | 18534 | 0.7934 | 0.7928 | 0.8174 |
| `small-vehicle` | 11357 | 0.7482 | 0.7453 | 0.7802 |
| `soccer-ball-field` | 260 | 0.8431 | 0.8377 | 0.8889 |
| `storage-tank` | 5031 | 0.7421 | 0.7421 | 0.8216 |
| `swimming-pool` | 693 | 0.6888 | 0.6888 | 0.7221 |
| `tennis-court` | 1529 | 0.9076 | 0.9075 | 0.9229 |
| **global** | 57768 | 0.7836 | 0.7817 | 0.8159 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 907 | 0.978 | 0.7826 |
| `basketball-court` | 278 | 432 | 1.000 | 0.9611 |
| `bridge` | 666 | 2194 | 0.895 | 0.6845 |
| `ground-track-field` | 216 | 441 | 0.843 | 0.6193 |
| `harbor` | 4298 | 6842 | 0.941 | 0.8452 |
| `helicopter` | 157 | 363 | 0.917 | 0.8830 |
| `large-vehicle` | 9398 | 15234 | 0.974 | 0.8728 |
| `plane` | 4731 | 5608 | 0.964 | 0.8899 |
| `roundabout` | 256 | 585 | 0.961 | 0.8265 |
| `ship` | 18534 | 32217 | 0.974 | 0.7057 |
| `small-vehicle` | 11357 | 20018 | 0.947 | 0.8574 |
| `soccer-ball-field` | 260 | 541 | 0.962 | 0.8853 |
| `storage-tank` | 5031 | 7716 | 0.885 | 0.7792 |
| `swimming-pool` | 693 | 1482 | 0.913 | 0.7843 |
| `tennis-court` | 1529 | 1851 | 0.991 | 0.8597 |
| **mAP** | | | | 0.8158 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.4500 | 0.7550 | 0.9313 | 0.8339 | 339 | 110 | 25 |
| `basketball-court` | 0.3500 | 0.8846 | 0.9928 | 0.9356 | 276 | 36 | 2 |
| `bridge` | 0.3000 | 0.7481 | 0.7492 | 0.7487 | 499 | 168 | 167 |
| `ground-track-field` | 0.1500 | 0.6468 | 0.6528 | 0.6498 | 141 | 77 | 75 |
| `harbor` | 0.2500 | 0.8425 | 0.8876 | 0.8645 | 3815 | 713 | 483 |
| `helicopter` | 0.3000 | 0.9424 | 0.8344 | 0.8851 | 131 | 8 | 26 |
| `large-vehicle` | 0.2500 | 0.8852 | 0.9289 | 0.9065 | 8730 | 1132 | 668 |
| `plane` | 0.3000 | 0.9421 | 0.9387 | 0.9404 | 4441 | 273 | 290 |
| `roundabout` | 0.3500 | 0.7770 | 0.8984 | 0.8333 | 230 | 66 | 26 |
| `ship` | 0.2500 | 0.6903 | 0.9388 | 0.7956 | 17400 | 7806 | 1134 |
| `small-vehicle` | 0.2000 | 0.8292 | 0.8709 | 0.8496 | 9891 | 2037 | 1466 |
| `soccer-ball-field` | 0.3500 | 0.9144 | 0.9038 | 0.9091 | 235 | 22 | 25 |
| `storage-tank` | 0.1500 | 0.8439 | 0.8346 | 0.8392 | 4199 | 777 | 832 |
| `swimming-pool` | 0.2500 | 0.7120 | 0.8240 | 0.7639 | 571 | 231 | 122 |
| `tennis-court` | 0.3000 | 0.9337 | 0.9863 | 0.9593 | 1508 | 107 | 21 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 355 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| `basketball-court` | 0 | 278 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bridge` | 0 | 0 | 530 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 136 |
| `ground-track-field` | 0 | 0 | 0 | 124 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 90 |
| `harbor` | 0 | 0 | 0 | 0 | 3815 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 479 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 137 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8727 | 0 | 0 | 0 | 58 | 0 | 0 | 0 | 0 | 613 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 4468 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 261 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 236 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| `ship` | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 17400 | 0 | 0 | 0 | 2 | 0 | 1129 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 0 | 0 | 0 | 9357 | 0 | 0 | 0 | 0 | 1957 |
| `soccer-ball-field` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 245 | 0 | 0 | 0 | 13 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3795 | 0 | 0 | 1236 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 571 | 0 | 122 |
| `tennis-court` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1509 | 20 |
| `False Positive` | 200 | 48 | 220 | 46 | 711 | 15 | 1091 | 307 | 99 | 7802 | 1330 | 40 | 350 | 229 | 118 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
