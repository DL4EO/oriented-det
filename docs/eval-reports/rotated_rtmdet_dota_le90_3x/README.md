# Rotated RTMDet S 3× eval

Hub slug **`rotated_rtmdet_dota_le90_3x`**. Advertised score is official DOTA v1.0 **Task 1 77.60%** (AP75 52.01, COCO mAP 49.02). Leaky `make eval-val` is **82.75%** (gap **5.15**; metrics margin 0, all 57,768 val objects). Deploy `production.score_threshold` **0.45** and `production.ignore_margin_pixels` **0**.

Report: [`model_analysis.md`](model_analysis.md). Recipe: [`dota_le90_3x_s.json`](../../../configs/rotated_rtmdet/dota_le90_3x_s.json). Run: `runs/rotated_rtmdet/20261002-102116`. `predictions.json` stays under gitignored `predictions/20261003_021342/`.
