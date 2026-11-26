# YOLO-OBB configs

Native **YOLO-OBB** (`model_type: yolo_obb`) — C2f CSP + PAFPN, distribution focal box head, task-aligned assigner. Clean-room. No pretrained backbone.

## Live recipes

| Recipe | Config | Notes |
|--------|--------|-------|
| COCO 2017 **300e S** | [`coco_300e_s.json`](coco_300e_s.json) | Horizontal pretrain. 80 classes, 640 letterbox, HSV only. SGD `0.01`, momentum `0.937` warmed from `0.8`, Nesterov, weight decay `5e-4` on conv weights only, linear LR to `1e-4`, EMA mix `0.0001`. Batch 64, warmup 3 epochs (5547 steps). Mosaic on, off for the last 10 epochs. Val keeps 300 boxes, score `0.001`, NMS `0.7`, mAP averaged from IoU 0.50 to 0.95. Hub slug **`yolo_obb_coco_300e`** (`eval_map50` **0.41**). Run `runs/yolo_obb/20261102-000543`. Train log was lost; sidecar config only. |
| DOTA le90 **12× S from COCO** | [`dota_le90_12x_s_coco.json`](dota_le90_12x_s_coco.json) | Hub slug **`yolo_obb_dota_le90_12x`**. Stands on the shared bases only. Loads `hf://yolo_obb_coco_300e`. 144 epochs. AdamW `1.25e-4` holds until epoch 72, then cosine to `1.25e-5`. Weight decay `0.05` on conv weights, off for norm and bias. EMA mix `0.0002`. BCE and the task-aligned assigner. ProbIoU at 7.5, DFL weight **0.75**, angle L1 0.2. Mosaic stays on, scale gain 0.9, translate 0.1, HSV `0.015 / 0.7 / 0.4`. Random rotate p=0.5 ±180°, with storage-tank and roundabout in 90° steps. mAP every 12 epochs. Mixup is not a training knob. Finished run `runs/yolo_obb/20261118-135903`: in-train leaky val mAP50 **84.42** at epoch 144 (`best_mAP_0.84.pth`); frozen `make eval-val` **80.13**. Official Task 1 **77.80** (AP75 52.13, COCO mAP 49.32). Ahead of RTMDet-S Hub 77.60; behind YOLO11-S single-scale 78.44 (those weights trained on multi-scale tiles). |

```bash
odet train --config configs/yolo_obb/coco_300e_s.json
odet train --config configs/yolo_obb/dota_le90_12x_s_coco.json
```

```bash
odet pretrained download yolo_obb_dota_le90_12x
make dota-submit EXPERIMENT=hf://yolo_obb_dota_le90_12x TEST_DIR=/path/to/data/DOTA-v1.0/test OUT=work_dirs/Task1_yolo_obb_12x_s
```

## Lessons learned

Ablations on the same COCO trunk and single-scale 1024 / overlap-200 tiles (Task 1 is the score that counts; leaky val over-ranks some knobs):

- **Loss and assigner.** ProbIoU beats rIoU on mAP50. Late cosine is a real gain and is what the 12× recipe keeps. QFL is a small win. Dynamic soft raises leaky (and swimming-pool) and spends AP75 — leave it off.
- **Schedule and aug.** Mosaic on + scale 0.9 + DFL 0.75 + 144 epochs is the jump to RTMDet parity (77.80). More epochs at full LR, a mosaic-off tail, and a native-tile fine-tune all raised leaky and lowered Task 1.
- **Optimizer.** SGD with gradient accumulation 8 cut steps 8×; the restarted classifier did not move (epoch-20 leaky ~44). AdamW with one step per batch is the working optimizer at batch 4 on this GPU.
- **Still open.** The 0.64 gap to YOLO11-S single-scale sits on plane / small-vehicle / swimming-pool / ground-track-field. Best remaining hypothesis is their multi-scale train tiles; mixup and true batch-32 SGD were not tested cleanly. Multi-scale stays optional.

## Training notes

The DOTA finetune sees **1024×1024**. The COCO pretrain sees **640×640**. While mosaic is on, each photo is scaled so its long side is 640 and pasted as that rectangle onto gray 114, with no black square. One affine then scales by a uniform factor in [0.5, 1.5], shifts by up to 10% of the 640 output, and fills new pixels with gray 114. That affine is the only scale; `mosaic_ratio_range` is not applied. Mosaic is off for the last 10 epochs on COCO; the DOTA 12× recipe leaves mosaic on. A box is cut to the canvas at the mosaic, affine, and scale/translate stages (`preprocessing.mosaic_clip_boxes`) and dropped when less than 10% of its area remains (`preprocessing.mosaic_min_visibility`), the Ultralytics `box_candidates` rule. Mosaic partners come from a 512-sample worker cache (`mosaic_max_cached_images`) rather than the default 40. Validation does not mosaic, scale, or translate, and validation letterboxes with gray 114. COCO flips horizontally only and does not random-rotate. The DOTA finetune flips horizontally or vertically. Each of the four tiles is rotated before paste when random rotate is on. Val / inference never mosaic or rotate. MixUp is not used. COCO color jitter is the Ultralytics / MMYOLO HSV gain (`hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, every image). The live COCO run `runs/yolo_obb/20261102-000543` predates the multiplicative HSV LUT fix and jitters additively. Logged COCO mAP is the mean of pycocotools 101-point AP from IoU 0.50 to 0.95 (`evaluation.map_ap_interpolation: "coco101"`). COCO uses its own train2017 mean and standard deviation at 640. The DOTA finetune keeps the DOTA tile stats.

Loss weights in the model base: box **7.5**, DFL **1.5**, classification **0.5**, le90 angle L1 **0.2**. The 12× DOTA recipe sets DFL to **0.75**. Soft targets, ProbIoU / rIoU / QFL / dynamic soft remain available as model knobs; the Hub DOTA recipe uses BCE, TAL, and ProbIoU. Nano is `configs/_base_/models/yolo_obb_n.json` (swap `backbone` to `yolo_obb_n`). There is no separate nano DOTA recipe.

COCO sets TAL top-k **10**, stride-dependent classifier prior, 16 px GT-side floor, gradient clip **10**, and bias warmup **0.1**. The DOTA 12× recipe leaves those at the model-base defaults (topk **13**, no stride prior).

Base fragments: [`yolo_obb_n.json`](../_base_/models/yolo_obb_n.json), [`yolo_obb_s.json`](../_base_/models/yolo_obb_s.json).

Parent: [configs/README.md](../README.md).
