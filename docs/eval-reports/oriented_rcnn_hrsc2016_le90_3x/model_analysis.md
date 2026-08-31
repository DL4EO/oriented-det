# Model Analysis Report

- Generated at: `2026-08-31T01:12:42.938157`

## Model metadata
- Experiment dir: `runs/oriented_rcnn/20260830-163857`
- Checkpoint: `runs/oriented_rcnn/20260830-163857/checkpoints/best_mAP_0.90.pth`
- Checkpoint modified: `2026-08-30T18:55:47.844654`
- Config: `runs/oriented_rcnn/20260830-163857/config.json`

## Source data
- Data root: `/path/to/data/HRSC2016`
- Data split: `val`
- Total images: `453`
- Total ground truth objects: `1228`
- Total predictions: `1461`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.9000`
- Precision at best threshold: `0.9509`
- Recall at best threshold: `0.9300`
- F1 at best threshold: `0.9403`
- F2 at best threshold: `0.9341`
- mAP50: `0.9041` (90.41%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.8480`
- Global mean best IoU (same class): `0.8480` (median `0.8780`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1228 | 0.8480 | 0.8480 | 0.8780 |
| **global** | 1228 | 0.8480 | 0.8480 | 0.8780 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 1188 | 1461 | 0.979 | 0.9041 |
| **mAP** | | | | 0.9041 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 0.9000 | 0.9509 | 0.9300 | 0.9403 | 1142 | 59 | 86 |

## Confusion matrix

Computed at score threshold `0.9000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | ship | Missed |
| --- | ---: | ---: |
| `ship` | 1142 | 86 |
| `False Positive` | 59 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
