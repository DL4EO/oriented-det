# Model Analysis Report

- Generated at: `2026-06-27T12:15:14.620861`

## Model metadata
- Experiment dir: `runs/oriented_rcnn/20260621-092802`
- Checkpoint: `runs/oriented_rcnn/20260621-092802/checkpoints/best_mAP_0.82.pth`
- Checkpoint modified: `2026-06-26T20:24:58.292279`
- Config: `runs/oriented_rcnn/20260621-092802/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `163193`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.50`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.7500`
- Precision at best threshold: `0.7607`
- Recall at best threshold: `0.8090`
- F1 at best threshold: `0.7841`
- F2 at best threshold: `0.7989`
- mAP50: `0.7940` (79.40%)

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1117 | 0.978 | 0.7902 |
| `basketball-court` | 278 | 572 | 0.996 | 0.8495 |
| `bridge` | 666 | 17019 | 0.913 | 0.7081 |
| `ground-track-field` | 216 | 830 | 0.986 | 0.8301 |
| `harbor` | 4298 | 14422 | 0.933 | 0.7901 |
| `helicopter` | 157 | 323 | 0.968 | 0.9054 |
| `large-vehicle` | 9398 | 22771 | 0.959 | 0.7692 |
| `plane` | 4731 | 6576 | 0.985 | 0.8936 |
| `roundabout` | 256 | 889 | 0.918 | 0.8157 |
| `ship` | 18534 | 39938 | 0.979 | 0.7242 |
| `small-vehicle` | 11357 | 30988 | 0.898 | 0.7542 |
| `soccer-ball-field` | 260 | 732 | 0.954 | 0.8667 |
| `storage-tank` | 5031 | 22430 | 0.735 | 0.6517 |
| `swimming-pool` | 693 | 2267 | 0.846 | 0.6906 |
| `tennis-court` | 1529 | 2319 | 0.976 | 0.8711 |
| **mAP** | | | | 0.7940 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.9000 | 0.6917 | 0.9121 | 0.7867 | 332 | 148 | 32 |
| `basketball-court` | 0.9500 | 0.8839 | 0.9856 | 0.9320 | 274 | 36 | 4 |
| `bridge` | 0.9500 | 0.7646 | 0.6877 | 0.7241 | 458 | 141 | 208 |
| `ground-track-field` | 0.8000 | 0.7689 | 0.9398 | 0.8458 | 203 | 61 | 13 |
| `harbor` | 0.7500 | 0.8008 | 0.7680 | 0.7841 | 3301 | 821 | 997 |
| `helicopter` | 0.6000 | 0.9539 | 0.9236 | 0.9385 | 145 | 7 | 12 |
| `large-vehicle` | 0.8000 | 0.7487 | 0.8341 | 0.7891 | 7839 | 2631 | 1559 |
| `plane` | 0.8500 | 0.9485 | 0.9620 | 0.9552 | 4551 | 247 | 180 |
| `roundabout` | 0.8000 | 0.7770 | 0.8438 | 0.8090 | 216 | 62 | 40 |
| `ship` | 0.7500 | 0.6739 | 0.8965 | 0.7694 | 16615 | 8039 | 1919 |
| `small-vehicle` | 0.5000 | 0.8107 | 0.7278 | 0.7670 | 8266 | 1930 | 3091 |
| `soccer-ball-field` | 0.8000 | 0.8357 | 0.9000 | 0.8667 | 234 | 46 | 26 |
| `storage-tank` | 0.8000 | 0.8642 | 0.6136 | 0.7177 | 3087 | 485 | 1944 |
| `swimming-pool` | 0.7000 | 0.7058 | 0.7201 | 0.7129 | 499 | 208 | 194 |
| `tennis-court` | 0.8500 | 0.9261 | 0.9595 | 0.9425 | 1467 | 117 | 62 |

## Confusion matrix

Computed at score threshold `0.7500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 346 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 |
| `basketball-court` | 0 | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `bridge` | 0 | 0 | 561 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 105 |
| `ground-track-field` | 0 | 0 | 0 | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `harbor` | 0 | 0 | 0 | 0 | 3301 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 995 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 140 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8008 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 0 | 1361 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4577 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 154 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 219 | 0 | 0 | 0 | 0 | 0 | 0 | 37 |
| `ship` | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 0 | 0 | 16615 | 0 | 0 | 0 | 0 | 0 | 1913 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 0 | 0 | 0 | 7160 | 0 | 0 | 0 | 0 | 4113 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 | 0 | 0 | 0 | 21 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3139 | 0 | 0 | 1892 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 473 | 0 | 220 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1472 | 47 |
| `False Positive` | 189 | 47 | 511 | 65 | 816 | 5 | 2837 | 279 | 67 | 8037 | 765 | 54 | 605 | 172 | 127 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
