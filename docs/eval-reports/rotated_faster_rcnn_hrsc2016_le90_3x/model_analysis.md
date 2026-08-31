# Model Analysis Report

- Generated at: `2026-08-31T05:10:37.559752`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260831-035851`
- Checkpoint: `runs/rotated_faster_rcnn/20260831-035851/checkpoints/best_mAP_0.89.pth`
- Checkpoint modified: `2026-08-31T04:57:34.306132`
- Config: `runs/rotated_faster_rcnn/20260831-035851/config.json`

## Source data
- Data root: `/path/to/data/HRSC2016`
- Data split: `val`
- Total images: `453`
- Total ground truth objects: `1228`
- Total predictions: `1545`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.9000`
- Precision at best threshold: `0.9311`
- Recall at best threshold: `0.9023`
- F1 at best threshold: `0.9165`
- F2 at best threshold: `0.9079`
- mAP50: `0.8877` (88.77%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7848`
- Global mean best IoU (same class): `0.7848` (median `0.8183`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1228 | 0.7848 | 0.7848 | 0.8183 |
| **global** | 1228 | 0.7848 | 0.7848 | 0.8183 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1188 | 1545 | 0.962 | 0.8877 |
| **mAP** | | | | 0.8877 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 0.9000 | 0.9311 | 0.9023 | 0.9165 | 1108 | 82 | 120 |

## Confusion matrix

Computed at score threshold `0.9000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | ship | Missed |
| --- | ---: | ---: |
| `ship` | 1108 | 120 |
| `False Positive` | 82 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
