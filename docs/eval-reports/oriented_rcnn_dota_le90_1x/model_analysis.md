# Model Analysis Report

- Generated at: `2026-06-18T17:55:28.526364`

## Model metadata
- Experiment dir: `runs/oriented_rcnn/20260616-030231`
- Checkpoint: `runs/oriented_rcnn/20260616-030231/checkpoints/best_mAP_0.78.pth`
- Checkpoint modified: `2026-06-18T04:02:15.784359`
- Config: `runs/oriented_rcnn/20260616-030231/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `188125`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.50`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.7000`
- Precision at best threshold: `0.7464`
- Recall at best threshold: `0.7748`
- F1 at best threshold: `0.7603`
- F2 at best threshold: `0.7690`
- mAP50: `0.7479` (74.79%)

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 2100 | 0.951 | 0.7342 |
| `basketball-court` | 278 | 1036 | 0.989 | 0.8733 |
| `bridge` | 666 | 11962 | 0.856 | 0.5672 |
| `ground-track-field` | 216 | 1342 | 0.958 | 0.7863 |
| `harbor` | 4298 | 20759 | 0.900 | 0.7284 |
| `helicopter` | 157 | 582 | 0.930 | 0.8646 |
| `large-vehicle` | 9398 | 30828 | 0.950 | 0.7592 |
| `plane` | 4731 | 7364 | 0.982 | 0.8890 |
| `roundabout` | 256 | 900 | 0.883 | 0.6904 |
| `ship` | 18534 | 48325 | 0.977 | 0.7321 |
| `small-vehicle` | 11357 | 40415 | 0.886 | 0.7131 |
| `soccer-ball-field` | 260 | 1293 | 0.896 | 0.7622 |
| `storage-tank` | 5031 | 15289 | 0.739 | 0.6629 |
| `swimming-pool` | 693 | 3006 | 0.840 | 0.5958 |
| `tennis-court` | 1529 | 2924 | 0.972 | 0.8596 |
| **mAP** | | | | 0.7479 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.9500 | 0.7857 | 0.7253 | 0.7543 | 264 | 72 | 100 |
| `basketball-court` | 0.8000 | 0.8297 | 0.9460 | 0.8840 | 263 | 54 | 15 |
| `bridge` | 0.8500 | 0.6114 | 0.5811 | 0.5958 | 387 | 246 | 279 |
| `ground-track-field` | 0.9000 | 0.7595 | 0.8333 | 0.7947 | 180 | 57 | 36 |
| `harbor` | 0.7000 | 0.7742 | 0.7173 | 0.7447 | 3083 | 899 | 1215 |
| `helicopter` | 0.5500 | 0.8808 | 0.8471 | 0.8636 | 133 | 18 | 24 |
| `large-vehicle` | 0.7500 | 0.7387 | 0.7680 | 0.7531 | 7218 | 2553 | 2180 |
| `plane` | 0.7500 | 0.9221 | 0.9539 | 0.9378 | 4513 | 381 | 218 |
| `roundabout` | 0.7500 | 0.6966 | 0.7891 | 0.7399 | 202 | 88 | 54 |
| `ship` | 0.7500 | 0.6796 | 0.8614 | 0.7598 | 15966 | 7528 | 2568 |
| `small-vehicle` | 0.5500 | 0.8112 | 0.6576 | 0.7264 | 7468 | 1738 | 3889 |
| `soccer-ball-field` | 0.8500 | 0.8279 | 0.7769 | 0.8016 | 202 | 42 | 58 |
| `storage-tank` | 0.7000 | 0.8646 | 0.6217 | 0.7233 | 3128 | 490 | 1903 |
| `swimming-pool` | 0.6500 | 0.6136 | 0.6392 | 0.6261 | 443 | 279 | 250 |
| `tennis-court` | 0.8000 | 0.9208 | 0.9353 | 0.9280 | 1430 | 123 | 99 |

## Confusion matrix

Computed at score threshold `0.7000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 321 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 |
| `basketball-court` | 0 | 268 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| `bridge` | 0 | 0 | 452 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 214 |
| `ground-track-field` | 0 | 0 | 0 | 195 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 |
| `harbor` | 0 | 0 | 0 | 0 | 3083 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1212 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 35 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 7416 | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 1964 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4527 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 204 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 203 | 0 | 0 | 0 | 0 | 0 | 0 | 53 |
| `ship` | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 16266 | 0 | 0 | 0 | 0 | 0 | 2264 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 6705 | 0 | 0 | 0 | 0 | 4563 |
| `soccer-ball-field` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 218 | 0 | 0 | 0 | 40 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3128 | 0 | 0 | 1903 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 417 | 0 | 276 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1439 | 80 |
| `False Positive` | 193 | 68 | 509 | 101 | 896 | 10 | 2813 | 409 | 96 | 8101 | 936 | 90 | 490 | 226 | 148 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
