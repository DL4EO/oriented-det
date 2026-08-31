# Model Analysis Report

- Generated at: `2026-08-31T03:40:14.371806`

## Model metadata
- Experiment dir: `runs/rotated_fcos/20260831-020019`
- Checkpoint: `runs/rotated_fcos/20260831-020019/checkpoints/best_mAP_0.89.pth`
- Checkpoint modified: `2026-08-31T02:30:45.564697`
- Config: `runs/rotated_fcos/20260831-020019/config.json`

## Source data
- Data root: `/path/to/data/HRSC2016`
- Data split: `val`
- Total images: `453`
- Total ground truth objects: `1228`
- Total predictions: `1558`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.2500`
- Precision at best threshold: `0.9060`
- Recall at best threshold: `0.9023`
- F1 at best threshold: `0.9041`
- F2 at best threshold: `0.9030`
- mAP50: `0.8834` (88.34%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7463`
- Global mean best IoU (same class): `0.7463` (median `0.7844`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1228 | 0.7463 | 0.7463 | 0.7844 |
| **global** | 1228 | 0.7463 | 0.7463 | 0.7844 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1188 | 1558 | 0.944 | 0.8834 |
| **mAP** | | | | 0.8834 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 0.2500 | 0.9060 | 0.9023 | 0.9041 | 1108 | 115 | 120 |

## Confusion matrix

Computed at score threshold `0.2500` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | ship | Missed |
| --- | ---: | ---: |
| `ship` | 1108 | 120 |
| `False Positive` | 115 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
