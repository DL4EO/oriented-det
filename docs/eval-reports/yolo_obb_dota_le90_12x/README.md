# YOLO-OBB S 12× eval

Hub slug **`yolo_obb_dota_le90_12x`**. Advertised score is official DOTA v1.0 **Task 1 77.80%** (AP75 52.13, COCO mAP 49.32). In-train leaky val (score ≥ 0.3) is **84.42%**. Deploy `production.score_threshold` **0.2**.

Report: [`model_analysis.md`](model_analysis.md) (Task 1 table only; no `analysis_iou0.50.json` — leaky eval-val was not frozen as a preds dump). Recipe: [`dota_le90_12x_s_coco.json`](../../../configs/yolo_obb/dota_le90_12x_s_coco.json). Run: `runs/yolo_obb/20261118-135903`.
