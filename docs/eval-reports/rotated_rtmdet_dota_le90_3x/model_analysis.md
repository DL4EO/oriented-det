# Model Analysis Report

Hub slug **`rotated_rtmdet_dota_le90_3x`**. Official DOTA v1.0 Task 1 **77.60%** (AP75 52.01, COCO mAP 49.02). Deploy `production.score_threshold` **0.45** (this sweep’s best F1 0.50 − 0.05). The mAP50 below is leaky eval-val (val tiles are in train), not the published number. eval-val − Task 1 is **5.15** (82.75 − 77.60). Metrics margin **0** (all 57,768 val objects).

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8923 |
| baseball-diamond | 0.8189 |
| bridge | 0.5348 |
| ground-track-field | 0.7288 |
| small-vehicle | 0.8104 |
| large-vehicle | 0.8290 |
| ship | 0.8885 |
| tennis-court | 0.9069 |
| basketball-court | 0.8623 |
| storage-tank | 0.8660 |
| soccer-ball-field | 0.6101 |
| roundabout | 0.6783 |
| harbor | 0.7725 |
| swimming-pool | 0.8256 |
| helicopter | 0.6159 |
| **mAP50** | **0.7760** |

- AP75: 0.5201
- COCO-style mAP: 0.4902
- Description submitted: oriented-det pretrained weights for RTMDet 3x (correct alias)
- Experiment: `runs/rotated_rtmdet/20261002-102116` (`best_mAP_0.88.pth`)
- Recipe: [`configs/rotated_rtmdet/dota_le90_3x_s.json`](../../../configs/rotated_rtmdet/dota_le90_3x_s.json)

- Generated at: `2026-10-03T04:35:35.234909`

## Model metadata
- Experiment dir: `runs/rotated_rtmdet/20261002-102116`
- Checkpoint: `runs/rotated_rtmdet/20261002-102116/checkpoints/best_mAP_0.88.pth`
- Checkpoint modified: `2026-10-02T18:26:26.554656`
- Config: `runs/rotated_rtmdet/20261002-102116/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `247861`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.5000`
- Precision at best threshold: `0.8114`
- Recall at best threshold: `0.9236`
- F1 at best threshold: `0.8639`
- F2 at best threshold: `0.8987`
- mAP50: `0.8275` (82.75%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.8022`
- Global mean best IoU (same class): `0.8007` (median `0.8277`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7990 | 0.7988 | 0.8111 |
| `basketball-court` | 278 | 0.8847 | 0.8843 | 0.8981 |
| `bridge` | 666 | 0.6818 | 0.6811 | 0.7229 |
| `ground-track-field` | 216 | 0.8322 | 0.8293 | 0.8634 |
| `harbor` | 4298 | 0.7619 | 0.7609 | 0.7996 |
| `helicopter` | 157 | 0.7540 | 0.7527 | 0.7872 |
| `large-vehicle` | 9398 | 0.8278 | 0.8225 | 0.8429 |
| `plane` | 4731 | 0.8392 | 0.8392 | 0.8731 |
| `roundabout` | 256 | 0.8057 | 0.8056 | 0.8414 |
| `ship` | 18534 | 0.8115 | 0.8109 | 0.8273 |
| `small-vehicle` | 11357 | 0.7758 | 0.7749 | 0.7981 |
| `soccer-ball-field` | 260 | 0.7868 | 0.7649 | 0.8757 |
| `storage-tank` | 5031 | 0.7747 | 0.7745 | 0.8338 |
| `swimming-pool` | 693 | 0.6817 | 0.6814 | 0.7179 |
| `tennis-court` | 1529 | 0.9122 | 0.9114 | 0.9237 |
| **global** | 57768 | 0.8022 | 0.8007 | 0.8277 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 4270 | 0.981 | 0.7755 |
| `basketball-court` | 278 | 2285 | 1.000 | 0.9658 |
| `bridge` | 666 | 19886 | 0.884 | 0.6717 |
| `ground-track-field` | 216 | 3437 | 0.977 | 0.7891 |
| `harbor` | 4298 | 16963 | 0.950 | 0.8508 |
| `helicopter` | 157 | 1096 | 0.949 | 0.8998 |
| `large-vehicle` | 9398 | 48136 | 0.983 | 0.8892 |
| `plane` | 4731 | 12983 | 0.969 | 0.8981 |
| `roundabout` | 256 | 3558 | 0.965 | 0.7930 |
| `ship` | 18534 | 49208 | 0.986 | 0.7361 |
| `small-vehicle` | 11357 | 51593 | 0.977 | 0.8853 |
| `soccer-ball-field` | 260 | 1936 | 0.881 | 0.7802 |
| `storage-tank` | 5031 | 24317 | 0.917 | 0.8594 |
| `swimming-pool` | 693 | 4331 | 0.893 | 0.7488 |
| `tennis-court` | 1529 | 3807 | 0.995 | 0.8701 |
| **mAP** | | | | 0.8275 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.6000 | 0.7260 | 0.8736 | 0.7930 | 318 | 120 | 46 |
| `basketball-court` | 0.7000 | 0.9306 | 0.9640 | 0.9470 | 268 | 20 | 10 |
| `bridge` | 0.4500 | 0.6241 | 0.7402 | 0.6772 | 493 | 297 | 173 |
| `ground-track-field` | 0.6000 | 0.7720 | 0.8935 | 0.8283 | 193 | 57 | 23 |
| `harbor` | 0.4500 | 0.8474 | 0.9118 | 0.8784 | 3919 | 706 | 379 |
| `helicopter` | 0.4500 | 0.9221 | 0.9045 | 0.9132 | 142 | 12 | 15 |
| `large-vehicle` | 0.6000 | 0.9154 | 0.9265 | 0.9209 | 8707 | 805 | 691 |
| `plane` | 0.6000 | 0.9406 | 0.9379 | 0.9392 | 4437 | 280 | 294 |
| `roundabout` | 0.5500 | 0.7348 | 0.8984 | 0.8084 | 230 | 83 | 26 |
| `ship` | 0.5500 | 0.7136 | 0.9446 | 0.8130 | 17507 | 7026 | 1027 |
| `small-vehicle` | 0.5000 | 0.8913 | 0.9088 | 0.8999 | 10321 | 1259 | 1036 |
| `soccer-ball-field` | 0.6500 | 0.9032 | 0.7538 | 0.8218 | 196 | 21 | 64 |
| `storage-tank` | 0.5000 | 0.9104 | 0.8299 | 0.8683 | 4175 | 411 | 856 |
| `swimming-pool` | 0.5000 | 0.7895 | 0.7576 | 0.7732 | 525 | 140 | 168 |
| `tennis-court` | 0.5500 | 0.9221 | 0.9836 | 0.9519 | 1504 | 127 | 25 |

## Confusion matrix

Computed at score threshold `0.5000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 339 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| `basketball-court` | 0 | 274 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| `bridge` | 0 | 0 | 429 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 237 |
| `ground-track-field` | 0 | 0 | 0 | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `harbor` | 0 | 0 | 0 | 0 | 3763 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 530 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 137 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8926 | 0 | 0 | 0 | 44 | 0 | 0 | 0 | 0 | 428 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 226 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 233 | 0 | 0 | 0 | 0 | 0 | 0 | 23 |
| `ship` | 0 | 0 | 1 | 0 | 2 | 0 | 2 | 0 | 0 | 17799 | 2 | 0 | 0 | 0 | 0 | 728 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 10320 | 0 | 0 | 0 | 0 | 1023 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 205 | 0 | 0 | 0 | 55 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4175 | 0 | 0 | 855 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 525 | 0 | 168 |
| `tennis-court` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1512 | 16 |
| `False Positive` | 184 | 60 | 171 | 87 | 537 | 10 | 1322 | 391 | 99 | 7523 | 1214 | 43 | 411 | 140 | 142 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
