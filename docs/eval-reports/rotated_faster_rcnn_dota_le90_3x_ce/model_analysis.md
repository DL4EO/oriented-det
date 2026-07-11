# Model Analysis Report

- Generated at: `2026-06-28T14:44:49.044499`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260530-012517`
- Checkpoint: `runs/rotated_faster_rcnn/20260530-012517/checkpoints/best_mAP_0.81.pth`
- Checkpoint modified: `2026-05-31T10:49:38.152118`
- Config: `runs/rotated_faster_rcnn/20260530-012517/config.json`

## Source data
- Data root: `/home/jeffaudi/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `194122`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.15`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.7000`
- Precision at best threshold: `0.7623`
- Recall at best threshold: `0.7776`
- F1 at best threshold: `0.7699`
- F2 at best threshold: `0.7745`
- mAP50: `0.7558` (75.58%)

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 4302 | 0.953 | 0.7788 |
| `basketball-court` | 278 | 724 | 0.996 | 0.8721 |
| `bridge` | 666 | 30448 | 0.778 | 0.5360 |
| `ground-track-field` | 216 | 13782 | 0.968 | 0.8155 |
| `harbor` | 4298 | 26144 | 0.807 | 0.6375 |
| `helicopter` | 157 | 232 | 0.981 | 0.9085 |
| `large-vehicle` | 9398 | 16943 | 0.814 | 0.6778 |
| `plane` | 4731 | 6455 | 0.973 | 0.8919 |
| `roundabout` | 256 | 6687 | 0.922 | 0.7899 |
| `ship` | 18534 | 37196 | 0.908 | 0.7142 |
| `small-vehicle` | 11357 | 21669 | 0.848 | 0.7480 |
| `soccer-ball-field` | 260 | 18417 | 0.950 | 0.8512 |
| `storage-tank` | 5031 | 7802 | 0.722 | 0.6848 |
| `swimming-pool` | 693 | 1462 | 0.755 | 0.5666 |
| `tennis-court` | 1529 | 1859 | 0.995 | 0.8645 |
| **mAP** | | | | 0.7558 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.8000 | 0.7427 | 0.8407 | 0.7887 | 306 | 106 | 58 |
| `basketball-court` | 0.9500 | 0.8922 | 0.9820 | 0.9349 | 273 | 33 | 5 |
| `bridge` | 0.9000 | 0.6368 | 0.6607 | 0.6485 | 440 | 251 | 226 |
| `ground-track-field` | 0.9000 | 0.7899 | 0.8704 | 0.8282 | 188 | 50 | 28 |
| `harbor` | 0.7500 | 0.7332 | 0.7443 | 0.7387 | 3199 | 1164 | 1099 |
| `helicopter` | 0.7000 | 0.9675 | 0.9490 | 0.9582 | 149 | 5 | 8 |
| `large-vehicle` | 0.5500 | 0.7542 | 0.7328 | 0.7434 | 6887 | 2244 | 2511 |
| `plane` | 0.7500 | 0.9372 | 0.9491 | 0.9431 | 4490 | 301 | 241 |
| `roundabout` | 0.7500 | 0.7413 | 0.8281 | 0.7823 | 212 | 74 | 44 |
| `ship` | 0.7000 | 0.6726 | 0.8627 | 0.7559 | 15989 | 7783 | 2545 |
| `small-vehicle` | 0.5500 | 0.8129 | 0.7121 | 0.7592 | 8087 | 1861 | 3270 |
| `soccer-ball-field` | 0.9000 | 0.8715 | 0.8346 | 0.8527 | 217 | 32 | 43 |
| `storage-tank` | 0.5500 | 0.8882 | 0.6539 | 0.7533 | 3290 | 414 | 1741 |
| `swimming-pool` | 0.5000 | 0.6053 | 0.6263 | 0.6156 | 434 | 283 | 259 |
| `tennis-court` | 0.5000 | 0.9255 | 0.9836 | 0.9537 | 1504 | 121 | 25 |

## Confusion matrix

Computed at score threshold `0.7000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 315 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 |
| `basketball-court` | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `bridge` | 0 | 0 | 483 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 183 |
| `ground-track-field` | 0 | 0 | 0 | 199 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 16 |
| `harbor` | 0 | 0 | 0 | 0 | 3253 | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 1039 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 148 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 6626 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 0 | 2742 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 225 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 0 | 0 | 0 | 42 |
| `ship` | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 15989 | 0 | 0 | 0 | 2 | 0 | 2540 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 75 | 0 | 0 | 0 | 7634 | 0 | 0 | 0 | 0 | 3648 |
| `soccer-ball-field` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 232 | 0 | 0 | 0 | 27 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3178 | 0 | 0 | 1852 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 365 | 0 | 328 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1494 | 25 |
| `False Positive` | 121 | 45 | 472 | 103 | 1266 | 6 | 1764 | 320 | 85 | 7778 | 1220 | 103 | 306 | 187 | 114 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
