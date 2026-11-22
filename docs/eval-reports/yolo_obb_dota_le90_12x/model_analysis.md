# Model Analysis Report

Hub slug **`yolo_obb_dota_le90_12x`**. Official DOTA v1.0 Task 1 **77.80%** (AP75 52.13, COCO mAP 49.32). Deploy `production.score_threshold` **0.2**. In-train leaky val mAP50 (score ≥ 0.3) is **84.42%** at epoch 144; that number is not a held-out `make eval-val` at score 0.05.

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
