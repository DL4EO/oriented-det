# YOLO-OBB S 12× eval

Hub slug **`yolo_obb_dota_le90_12x`**. Advertised score is official DOTA v1.0 **Task 1 77.80%** (AP75 52.13, COCO mAP 49.32). Leaky `make eval-val` is **80.13%** (gap **2.33**; metrics margin 0, all 57,768 val objects). In-train leaky val at score ≥ 0.3 was **84.42%**. Deploy `production.score_threshold` **0.45**.

Report: [`model_analysis.md`](model_analysis.md). Recipe: [`dota_le90_12x_s_coco.json`](../../../configs/yolo_obb/dota_le90_12x_s_coco.json). Run: `runs/yolo_obb/20261118-135903`. Local preds: `predictions/20261124_195315/`.
