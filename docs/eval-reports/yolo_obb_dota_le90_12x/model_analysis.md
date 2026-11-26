# Model Analysis Report

Hub slug **`yolo_obb_dota_le90_12x`**. Official DOTA v1.0 Task 1 **77.80%** (AP75 52.13, COCO mAP 49.32). Deploy `production.score_threshold` **0.45** (this sweep’s best F1 0.50 − 0.05). The mAP50 below is leaky eval-val (val tiles are in train), not the published number. eval-val − Task 1 is **2.33** (80.13 − 77.80). Metrics margin **0** (all 57,768 val objects). In-train leaky val at score ≥ 0.3 was **84.42%** (non-empty-tile training eval; not this report).

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8537 |
| baseball-diamond | 0.8403 |
| bridge | 0.5438 |
| ground-track-field | 0.6992 |
| small-vehicle | 0.7699 |
| large-vehicle | 0.8387 |
| ship | 0.8855 |
| tennis-court | 0.9054 |
| basketball-court | 0.8649 |
| storage-tank | 0.8637 |
| soccer-ball-field | 0.6686 |
| roundabout | 0.6686 |
| harbor | 0.7794 |
| swimming-pool | 0.8005 |
| helicopter | 0.6877 |
| **mAP50** | **0.7780** |

- AP75: 0.5213
- COCO-style mAP: 0.4932
- Description submitted: oriented-det pretrained weights YOLO-OBB 12x
- Experiment: `runs/yolo_obb/20261118-135903` (`best_mAP_0.84.pth`)
- Recipe: [`configs/yolo_obb/dota_le90_12x_s_coco.json`](../../../configs/yolo_obb/dota_le90_12x_s_coco.json)
- Loads COCO trunk `hf://yolo_obb_coco_300e`

- Generated at: `2026-11-24T20:57:25.135675`

## Model metadata
- Experiment dir: `runs/yolo_obb/20261118-135903`
- Checkpoint: `runs/yolo_obb/20261118-135903/checkpoints/best_mAP_0.84.pth`
- Checkpoint modified: `2026-11-20T03:11:12`
- Config: `runs/yolo_obb/20261118-135903/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `98831`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.5000`
- Precision at best threshold: `0.7834`
- Recall at best threshold: `0.9143`
- F1 at best threshold: `0.8438`
- F2 at best threshold: `0.8848`
- mAP50: `0.8013` (80.13%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7998`
- Global mean best IoU (same class): `0.7984` (median `0.8259`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7247 | 0.7247 | 0.7337 |
| `basketball-court` | 278 | 0.8835 | 0.8831 | 0.9048 |
| `bridge` | 666 | 0.6538 | 0.6538 | 0.7143 |
| `ground-track-field` | 216 | 0.8401 | 0.8401 | 0.8598 |
| `harbor` | 4298 | 0.7735 | 0.7731 | 0.8129 |
| `helicopter` | 157 | 0.7660 | 0.7617 | 0.7959 |
| `large-vehicle` | 9398 | 0.8332 | 0.8308 | 0.8483 |
| `plane` | 4731 | 0.8280 | 0.8279 | 0.8627 |
| `roundabout` | 256 | 0.6954 | 0.6954 | 0.7551 |
| `ship` | 18534 | 0.8202 | 0.8196 | 0.8328 |
| `small-vehicle` | 11357 | 0.7818 | 0.7792 | 0.7992 |
| `soccer-ball-field` | 260 | 0.7810 | 0.7408 | 0.8753 |
| `storage-tank` | 5031 | 0.7051 | 0.7047 | 0.7154 |
| `swimming-pool` | 693 | 0.6963 | 0.6963 | 0.7159 |
| `tennis-court` | 1529 | 0.9112 | 0.9087 | 0.9245 |
| **global** | 57768 | 0.7998 | 0.7984 | 0.8259 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1097 | 0.953 | 0.7286 |
| `basketball-court` | 278 | 567 | 0.996 | 0.8744 |
| `bridge` | 666 | 2102 | 0.842 | 0.6309 |
| `ground-track-field` | 216 | 827 | 0.995 | 0.7678 |
| `harbor` | 4298 | 6843 | 0.955 | 0.8421 |
| `helicopter` | 157 | 312 | 0.962 | 0.8723 |
| `large-vehicle` | 9398 | 15868 | 0.987 | 0.8973 |
| `plane` | 4731 | 6707 | 0.970 | 0.8723 |
| `roundabout` | 256 | 859 | 0.922 | 0.7168 |
| `ship` | 18534 | 31928 | 0.992 | 0.7619 |
| `small-vehicle` | 11357 | 19950 | 0.981 | 0.8732 |
| `soccer-ball-field` | 260 | 691 | 0.854 | 0.7203 |
| `storage-tank` | 5031 | 7461 | 0.938 | 0.8558 |
| `swimming-pool` | 693 | 1606 | 0.924 | 0.7518 |
| `tennis-court` | 1529 | 2013 | 0.991 | 0.8544 |
| **mAP** | | | | 0.8013 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.6000 | 0.7507 | 0.6951 | 0.7218 | 253 | 84 | 111 |
| `basketball-court` | 0.7000 | 0.8296 | 0.9281 | 0.8761 | 258 | 53 | 20 |
| `bridge` | 0.4000 | 0.5952 | 0.6757 | 0.6329 | 450 | 306 | 216 |
| `ground-track-field` | 0.6500 | 0.7203 | 0.8704 | 0.7883 | 188 | 73 | 28 |
| `harbor` | 0.4500 | 0.8489 | 0.8760 | 0.8622 | 3765 | 670 | 533 |
| `helicopter` | 0.4500 | 0.8951 | 0.9236 | 0.9091 | 145 | 17 | 12 |
| `large-vehicle` | 0.6500 | 0.9447 | 0.9165 | 0.9304 | 8613 | 504 | 785 |
| `plane` | 0.6000 | 0.8634 | 0.9393 | 0.8998 | 4444 | 703 | 287 |
| `roundabout` | 0.5000 | 0.6900 | 0.7305 | 0.7097 | 187 | 84 | 69 |
| `ship` | 0.6500 | 0.7118 | 0.9164 | 0.8013 | 16985 | 6876 | 1549 |
| `small-vehicle` | 0.5000 | 0.8591 | 0.8906 | 0.8746 | 10115 | 1659 | 1242 |
| `soccer-ball-field` | 0.5000 | 0.6809 | 0.7962 | 0.7340 | 207 | 97 | 53 |
| `storage-tank` | 0.4000 | 0.8663 | 0.8567 | 0.8615 | 4310 | 665 | 721 |
| `swimming-pool` | 0.4500 | 0.7237 | 0.8124 | 0.7655 | 563 | 215 | 130 |
| `tennis-court` | 0.7000 | 0.8829 | 0.9614 | 0.9205 | 1470 | 195 | 59 |

## Confusion matrix

Computed at score threshold `0.5000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 307 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 |
| `basketball-court` | 0 | 269 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| `bridge` | 0 | 0 | 372 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 294 |
| `ground-track-field` | 0 | 0 | 0 | 208 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `harbor` | 0 | 0 | 0 | 0 | 3581 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 713 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 140 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8973 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 408 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4523 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 208 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 187 | 0 | 0 | 0 | 0 | 0 | 0 | 69 |
| `ship` | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 17902 | 1 | 0 | 0 | 0 | 0 | 626 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 10114 | 0 | 0 | 0 | 0 | 1214 |
| `soccer-ball-field` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 207 | 0 | 0 | 0 | 52 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4009 | 0 | 0 | 1022 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 526 | 0 | 167 |
| `tennis-court` | 4 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1499 | 24 |
| `False Positive` | 219 | 89 | 171 | 130 | 515 | 16 | 1181 | 940 | 84 | 8573 | 1642 | 97 | 463 | 176 | 242 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
