# Rotated FCOS configs

Anchor-free single-stage oriented detector (v0.2). **DistanceAnglePointCoder** (`left, top, right, bottom, angle`), center-in-OBB assigner with per-level regress ranges, centerness, and box regression via **L1** (baseline), optional decoded **KFIoU** aux, or decoded **rIoU** primary (`1 -` differentiable polygon IoU). Monte-Carlo `pairwise_rotated_iou` is **not** a train loss.

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **1× DOTA decoded rIoU** (12 epochs, **lr 2.5e-3**, batch 2, train+val tiles, H+V+diagonal flips). P3–P7. mAP every **4** epochs. |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× decoded rIoU** — inherits 1×; 36 epochs, milestones `[24, 33]`, warmup **2000**. Same lr **2.5e-3**. **Hub recipe** `rotated_fcos_dota_le90_3x`. |
| [`dota_le90_1x_l1_kfiou_aux.json`](./dota_le90_1x_l1_kfiou_aux.json) | **1× L1 + KFIoU aux 0.1** — overrides 1× to L1 + aux; lr **2.5e-4**. |
| [`hrsc2016_le90_1x.json`](./hrsc2016_le90_1x.json) | **1× HRSC2016 rIoU** (native XML, single-class ship, `keep_ratio` + pad-32, H+V+diagonal flips + random rotate p=0.5 **±20°**, trainval/test). Same head/lr as DOTA 1× rIoU. |
| [`hrsc2016_le90_3x.json`](./hrsc2016_le90_3x.json) | **3× HRSC2016 rIoU** — inherits 1×; 36 epochs, milestones `[24, 33]`, `lr_scheduler_gamma` `[0.1, 0.5]`, same lr **2.5e-3**. Hub: `rotated_fcos_hrsc2016_le90_3x`. |

Former recipe filenames `dota_le90_1x_riou.json`, `dota_le90_*_kfiou_aux.json`, and `dota_le90_3x_l1.json` were renamed/folded into the table above (1× rIoU is now `dota_le90_1x.json`; KFIoU aux 1× is `dota_le90_1x_l1_kfiou_aux.json`). Hub slug `rotated_fcos_dota_le90_3x_kfiou_aux` remains for the published 3× KFIoU-aux weights.

Base model: [`../_base_/models/rotated_fcos_r50.json`](../_base_/models/rotated_fcos_r50.json).

## Results

DOTA1.0 (pretrain: **train+val / val**). Published mAP = **`make eval-val`** mAP50 (7,669 val tiles, `filter_empty_gt=false`, score ≥ **0.05**, **`evaluation.final_nms_iou_threshold: 0.1`**; deploy ships **`production.final_nms_iou_threshold: 0.3`**).

| Schedule | Config | Training run | Checkpoint | Train-time final mAP† | eval-val mAP50 | Report / Hub |
| :------: | :----: | :----------: | :--------: | :-------------------: | :------------: | :----------- |
| 1× L1 (historical) | — | `runs/rotated_fcos/20260811-064719` | `best_mAP_0.70.pth` | ~70% | **63.07%**‡ | — |
| 3× L1 (historical) | — | `runs/rotated_fcos/20260812-105204` | `best_mAP_0.72.pth` | **72.22%** | **73.92%** | [`docs/eval-reports/rotated_fcos_dota_le90_3x_l1/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x_l1/model_analysis.md) |
| 1× L1 + KFIoU aux | [`dota_le90_1x_l1_kfiou_aux.json`](./dota_le90_1x_l1_kfiou_aux.json) | `runs/rotated_fcos/20260814-074221` | `best_mAP_0.77.pth` | **76.53%** | **69.62%** | `predictions/20260814_181423/` |
| 1× rIoU | [`dota_le90_1x.json`](./dota_le90_1x.json) | `runs/rotated_fcos/20260821-071344` | `best_mAP_0.80.pth` | **80.43%** | **74.04%** | `predictions/20260822_152147/` |
| 3× L1 + KFIoU aux | — (Hub weights only) | `runs/rotated_fcos/20260818-100049` | `best_mAP_0.84.pth` | **84.07%** | **77.18%** | Hub `rotated_fcos_dota_le90_3x_kfiou_aux`; [`docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md) |
| **3× rIoU** | [`dota_le90_3x.json`](./dota_le90_3x.json) | `runs/rotated_fcos/20260831-052647` | `best_mAP_0.82.pth` | **82.49%** | **82.32%** | Hub `rotated_fcos_dota_le90_3x`; [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md) |

† Training periodic/final mAP uses `evaluation.score_threshold: 0.3` and non-empty tiles only.  
‡ 1× L1 eval-val was measured with production NMS **0.3**; re-run with NMS **0.1** before comparing to 3×.

**3× rIoU** is **+8.4** eval-val mAP50 vs this repo’s **3× L1** (82.32% vs 73.92%) and **+5.1** vs Hub **3× KFIoU aux** (77.18%). Viewer:

```bash
make viewer VIEWER_PRED_DIR=predictions/20260901_053115 DOTA_DATA_ROOT=/path/to/DOTA-v1.0-tiled
```

HRSC2016 (ImageSets **trainval / test**, 453 images). Whole-image `keep_ratio` + pad-32, decoded rIoU, rotate p=0.5 ±20°.

| Schedule | Config | Training run | Checkpoint | eval-val mAP50 | Hub |
| :------: | :----: | :----------: | :--------: | :------------: | :-- |
| **3× rIoU** | [`hrsc2016_le90_3x.json`](./hrsc2016_le90_3x.json) | `runs/rotated_fcos/20260831-020019` | `best_mAP_0.89.pth` | **88.34%** | `rotated_fcos_hrsc2016_le90_3x`; [`docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/`](../../docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/model_analysis.md) |

```bash
odet pretrained download rotated_fcos_dota_le90_3x
odet pretrained download rotated_fcos_hrsc2016_le90_3x
```

## Eval NMS (dense scenes)

`model.final_nms_iou_threshold` and **`evaluation.final_nms_iou_threshold`** are **0.1** (MMRotate FCOS test NMS); **`production.final_nms_iou_threshold`** is **0.3** for deploy / `image_demo`. `make eval-val` / `odet preds` prefer `evaluation.final_nms_iou_threshold`. Default NMS is class-aware; set `model.nms_class_agnostic: true` (and `production.nms_class_agnostic` if deploy should match) for lookalike vehicle classes.

## L1 vs rIoU recipe notes

- L1 recipes use `learning_rate` **2.5e-4** (0.0025 on L1 was unstable here).
- Decoded **rIoU** primary ([`dota_le90_1x.json`](./dota_le90_1x.json), [`dota_le90_3x.json`](./dota_le90_3x.json)) uses **0.0025**. Keep center radius 1.5, `max_detections_per_image` **2000**.
- FCOS `riou` is differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`), not sampling `pairwise_rotated_iou`. ROI `riou` is unchanged (still sampling).

## Decoded aux (KFIoU)

These FCOS knobs are **`aux_loss_type` / `aux_loss_weight`**, not two-stage **`roi_box_reg_aux_*`**. Keep **`box_reg_loss_type: l1`**. Set **`aux_loss_type: kfiou`** and **`aux_loss_weight`** (try **0.1**). Aux is decoded (stride restore when `norm_on_bbox`) and **centerness-weighted** like L1; logged as `loss_box_reg_aux`. Each positive’s aux is Gaussian overlap **plus** an aspect-gated heading term `ω sin²(2Δθ)` (`ω = exp(-log²(w*/h*)/λ²)` from GT size) so near-squares keep a θ gradient. **`aux_angle_weight`** (default **1.0**, **0** = Gaussian only) and **`aux_angle_lambda`** (default **1.0**). Sampling `riou` is rejected. Recipe: [`dota_le90_1x_l1_kfiou_aux.json`](./dota_le90_1x_l1_kfiou_aux.json).

```bash
odet train --config configs/rotated_fcos/dota_le90_1x.json
odet train --config configs/rotated_fcos/dota_le90_3x.json
odet train --config configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json
odet train --config configs/rotated_fcos/hrsc2016_le90_1x.json
odet train --config configs/rotated_fcos/hrsc2016_le90_3x.json
make wizard CONFIG=configs/rotated_fcos/dota_le90_3x.json
```

`make eval-val` / `odet preds` use **`evaluation.preds_score_threshold`** or **0.05** (published protocol), not `production.score_threshold` or train-val `evaluation.score_threshold: 0.3`. Deploy / `image_demo` still use `production.score_threshold`.
