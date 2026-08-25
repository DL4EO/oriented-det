# Model Analysis Report

- Generated at: `2026-08-20T13:42:12.635700`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260818-100049`
- Checkpoint: `runs/rotated_fcos/20260818-100049/checkpoints/best_mAP_0.84.pth`
- Checkpoint modified: `2026-08-19T12:58:46.219027`
- Config: `runs/rotated_fcos/20260818-100049/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `134639`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.7802`
- Recall at best threshold: `0.8412`
- F1 at best threshold: `0.8095`
- F2 at best threshold: `0.8282`
- mAP50: `0.7718` (77.18%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7609`
- Global mean best IoU (same class): `0.7567` (median `0.7995`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7743 | 0.7742 | 0.7885 |
| `basketball-court` | 278 | 0.8668 | 0.8657 | 0.8876 |
| `bridge` | 666 | 0.6765 | 0.6754 | 0.7332 |
| `ground-track-field` | 216 | 0.7292 | 0.6645 | 0.8174 |
| `harbor` | 4298 | 0.7208 | 0.7178 | 0.7593 |
| `helicopter` | 157 | 0.7616 | 0.7412 | 0.7970 |
| `large-vehicle` | 9398 | 0.7843 | 0.7760 | 0.8117 |
| `plane` | 4731 | 0.8222 | 0.8222 | 0.8612 |
| `roundabout` | 256 | 0.8056 | 0.8020 | 0.8671 |
| `ship` | 18534 | 0.7794 | 0.7785 | 0.8039 |
| `small-vehicle` | 11357 | 0.7237 | 0.7148 | 0.7645 |
| `soccer-ball-field` | 260 | 0.8222 | 0.8135 | 0.8761 |
| `storage-tank` | 5031 | 0.6789 | 0.6784 | 0.7668 |
| `swimming-pool` | 693 | 0.6693 | 0.6693 | 0.7108 |
| `tennis-court` | 1529 | 0.9024 | 0.8970 | 0.9205 |
| **global** | 57768 | 0.7609 | 0.7567 | 0.7995 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 2142 | 0.978 | 0.8092 |
| `basketball-court` | 278 | 899 | 0.989 | 0.8661 |
| `bridge` | 666 | 6404 | 0.845 | 0.6092 |
| `ground-track-field` | 216 | 1327 | 0.815 | 0.5054 |
| `harbor` | 4298 | 9749 | 0.902 | 0.7951 |
| `helicopter` | 157 | 664 | 0.911 | 0.8365 |
| `large-vehicle` | 9398 | 21668 | 0.954 | 0.8659 |
| `plane` | 4731 | 7229 | 0.960 | 0.8898 |
| `roundabout` | 256 | 1436 | 0.930 | 0.7418 |
| `ship` | 18534 | 38160 | 0.967 | 0.7087 |
| `small-vehicle` | 11357 | 27563 | 0.904 | 0.8014 |
| `soccer-ball-field` | 260 | 1379 | 0.935 | 0.8466 |
| `storage-tank` | 5031 | 10195 | 0.809 | 0.7376 |
| `swimming-pool` | 693 | 2897 | 0.879 | 0.6927 |
| `tennis-court` | 1529 | 2926 | 0.978 | 0.8704 |
| **mAP** | | | | 0.7718 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.4000 | 0.7734 | 0.8626 | 0.8156 | 314 | 92 | 50 |
| `basketball-court` | 0.3500 | 0.8717 | 0.9532 | 0.9107 | 265 | 39 | 13 |
| `bridge` | 0.2500 | 0.6071 | 0.6892 | 0.6456 | 459 | 297 | 207 |
| `ground-track-field` | 0.2500 | 0.6011 | 0.5231 | 0.5594 | 113 | 75 | 103 |
| `harbor` | 0.2500 | 0.8038 | 0.8285 | 0.8160 | 3561 | 869 | 737 |
| `helicopter` | 0.3000 | 0.9051 | 0.7898 | 0.8435 | 124 | 13 | 33 |
| `large-vehicle` | 0.2500 | 0.8597 | 0.8666 | 0.8631 | 8144 | 1329 | 1254 |
| `plane` | 0.3500 | 0.9341 | 0.9106 | 0.9222 | 4308 | 304 | 423 |
| `roundabout` | 0.3500 | 0.7003 | 0.8125 | 0.7523 | 208 | 89 | 48 |
| `ship` | 0.2500 | 0.6898 | 0.9170 | 0.7874 | 16996 | 7642 | 1538 |
| `small-vehicle` | 0.2000 | 0.7925 | 0.7901 | 0.7913 | 8973 | 2349 | 2384 |
| `soccer-ball-field` | 0.3500 | 0.8932 | 0.8038 | 0.8462 | 209 | 25 | 51 |
| `storage-tank` | 0.2000 | 0.8222 | 0.7205 | 0.7680 | 3625 | 784 | 1406 |
| `swimming-pool` | 0.3000 | 0.6955 | 0.6854 | 0.6904 | 475 | 208 | 218 |
| `tennis-court` | 0.3000 | 0.9223 | 0.9542 | 0.9380 | 1459 | 123 | 70 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 344 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| `basketball-court` | 0 | 272 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| `bridge` | 0 | 0 | 459 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 207 |
| `ground-track-field` | 0 | 0 | 0 | 113 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 94 |
| `harbor` | 0 | 0 | 0 | 0 | 3561 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 734 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8141 | 0 | 0 | 0 | 82 | 0 | 0 | 0 | 0 | 1175 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4404 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 327 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 219 | 0 | 0 | 0 | 0 | 0 | 0 | 37 |
| `ship` | 0 | 0 | 1 | 0 | 6 | 0 | 3 | 0 | 0 | 16995 | 0 | 0 | 0 | 0 | 0 | 1529 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 121 | 0 | 0 | 1 | 8311 | 0 | 0 | 0 | 0 | 2924 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 225 | 0 | 0 | 0 | 35 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3418 | 0 | 0 | 1613 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 520 | 0 | 173 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1468 | 51 |
| `False Positive` | 204 | 56 | 296 | 75 | 862 | 24 | 1208 | 459 | 159 | 7639 | 1460 | 62 | 504 | 304 | 148 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
