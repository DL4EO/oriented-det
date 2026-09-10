# Rotated FCOS configs

Anchor-free single-stage oriented detector (v0.2). **DistanceAnglePointCoder** (`left, top, right, bottom, angle`), center-in-OBB assigner with per-level regress ranges, centerness, and box regression via **L1** (baseline), optional decoded **KFIoU** aux, or decoded **rIoU** primary (`1 -` differentiable polygon IoU). Monte-Carlo `pairwise_rotated_iou` is **not** a train loss.

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **Hub 1× DOTA decoded rIoU** (12 epochs, **lr 2.5e-3**, batch 2, train+val tiles, H+V+diagonal flips). Inherits [`dota_le90.json`](../_base_/datasets/dota_le90.json) (DOTA `odet stats` mean/std). Deploy `production.score_threshold` **0.2** (eval-val F1 0.25 − 0.05). P3–P7. mAP every **4** epochs. |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× DOTA decoded rIoU** — inherits 1×; 36 epochs, milestones `[24, 33]`. Hub: `rotated_fcos_dota_le90_3x` (eval-val **82.32%**). |
| [`dota_le90_1x_l1_kfiou_aux.json`](./dota_le90_1x_l1_kfiou_aux.json) | **1× L1 + KFIoU aux 0.1** — overrides 1× to L1 + aux; lr **2.5e-4**. |
| [`hrsc2016_le90_1x.json`](./hrsc2016_le90_1x.json) | **1× HRSC2016 rIoU** (native XML, single-class ship, `keep_ratio` + pad-32, H+V+diagonal flips + random rotate p=0.5 **±20°**, trainval/test). Same head/lr as DOTA 1× rIoU. Deploy `production.score_threshold` **0.2** (eval-val F1 0.25 − 0.05). |
| [`hrsc2016_le90_3x.json`](./hrsc2016_le90_3x.json) | **3× HRSC2016 rIoU** — inherits 1×; 36 epochs, milestones `[24, 33]`, `lr_scheduler_gamma` `[0.1, 0.5]`, same lr **2.5e-3**. Hub: `rotated_fcos_hrsc2016_le90_3x`. |

Former recipe filenames `dota_le90_1x_riou.json`, `dota_le90_*_kfiou_aux.json`, and `dota_le90_3x_l1.json` were renamed/folded into the table above (1× rIoU is now `dota_le90_1x.json`; KFIoU aux 1× is `dota_le90_1x_l1_kfiou_aux.json`).

Base model: [`../_base_/models/rotated_fcos_r50.json`](../_base_/models/rotated_fcos_r50.json). Recipes keep `loss.loss_type: focal` (unweighted). Set `focal_weighted` to apply `loss.class_weight_*` to sigmoid-focal class columns; `background_weight` is ignored.

## Results

DOTA1.0 (pretrain: **train+val**). Published number is **official DOTA v1.0 Task 1** mAP50 (hidden test; val tiles leak into train, so do not quote `make eval-val` as held-out accuracy). Deploy NMS IoU **0.1**.

Multi-GPU trainval can deadlock the NCCL reducer after backbone unfreeze if DDP is rebuilt with `find_unused_parameters=False` and a later batch skips a head (empty / lookalike-only). oriented-det no longer rebuilds DDP; the initial wrap stays. `make train-multi-gpu` sets `TORCH_DIST_TIMEOUT_SECONDS=1800` so a hang dies in 30 min instead of 24h.

| Backbone | Official Task 1 | AP75 | Angle | lr schd | Aug | BS | Config | Final config | Final log | Download |
| :----------------------: | :---: | :---: | :---: | :-----: | :-: | :--: | :----: | :----------: | :-------: | :----: |
| ResNet50 (1024,1024,200) | **73.07** | 40.40 | le90 | 1× | H+V+D | 2 | [`dota_le90_1x.json`](./dota_le90_1x.json) | [`rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.json`](../../pretrained/rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.json) | [`rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.log`](../../pretrained/rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.log) | `hf://rotated_fcos_dota_le90_1x` |
| ResNet50 (1024,1024,200) | 82.32† | — | le90 | 3× | H+V+D | 2 | [`dota_le90_3x.json`](./dota_le90_3x.json) | [`rotated_fcos_r50_fpn_dota_le90_3x-6e383331.json`](../../pretrained/rotated_fcos_r50_fpn_dota_le90_3x-6e383331.json) | [`rotated_fcos_r50_fpn_dota_le90_3x-6e383331.log`](../../pretrained/rotated_fcos_r50_fpn_dota_le90_3x-6e383331.log) | `hf://rotated_fcos_dota_le90_3x` |

† 3× Hub mAP is **eval-val** (no official Task 1). Advertised short schedule stays 1×. Hub 3× weights were trained with warmup 2000; the recipe is now 1× + 36 epochs only.

Eval reports: [`docs/eval-reports/rotated_fcos_dota_le90_1x/`](../../docs/eval-reports/rotated_fcos_dota_le90_1x/model_analysis.md) (1× run `runs/rotated_fcos/20260908-023531`), [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md) (3×).

```bash
odet pretrained download rotated_fcos_dota_le90_1x
make viewer VIEWER_PRED_DIR=predictions/20260908_132129 DOTA_DATA_ROOT=/path/to/DOTA-v1.0-tiled
```

HRSC2016 (ImageSets **trainval / test**, 453 images). Whole-image `keep_ratio` + pad-32, decoded rIoU, rotate p=0.5 ±20°.

| Schedule | Config | Training run | Checkpoint | eval-val mAP50 | Hub |
| :------: | :----: | :----------: | :--------: | :------------: | :-- |
| **3× rIoU** | [`hrsc2016_le90_3x.json`](./hrsc2016_le90_3x.json) | `runs/rotated_fcos/20260831-020019` | `best_mAP_0.89.pth` | **88.34%** | `rotated_fcos_hrsc2016_le90_3x`; [`docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/`](../../docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/model_analysis.md) |

```bash
odet pretrained download rotated_fcos_hrsc2016_le90_3x
```

## Eval NMS (dense scenes)

`model.final_nms_iou_threshold`, **`evaluation.final_nms_iou_threshold`**, and **`production.final_nms_iou_threshold`** are **0.1** (MMRotate FCOS test NMS). `make eval-val` / `odet preds` prefer `evaluation.final_nms_iou_threshold`. Default NMS is class-aware; set `model.nms_class_agnostic: true` (and `production.nms_class_agnostic` if deploy should match) for lookalike vehicle classes.

## L1 vs rIoU recipe notes

- L1 recipes use `learning_rate` **2.5e-4** (0.0025 on L1 was unstable here).
- Decoded **rIoU** primary ([`dota_le90_1x.json`](./dota_le90_1x.json)) uses **0.0025**. Keep center radius 1.5, `max_detections_per_image` **2000**.
- FCOS `riou` is differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`), not sampling `pairwise_rotated_iou`. ROI `riou` is unchanged (still sampling).

## Decoded aux (KFIoU)

These FCOS knobs are **`aux_loss_type` / `aux_loss_weight`**, not two-stage **`roi_box_reg_aux_*`**. Keep **`box_reg_loss_type: l1`**. Set **`aux_loss_type: kfiou`** and **`aux_loss_weight`** (try **0.1**). Aux is decoded (stride restore when `norm_on_bbox`) and **centerness-weighted** like L1; logged as `loss_box_reg_aux`. Each positive’s aux is Gaussian overlap **plus** an aspect-gated heading term `ω sin²(2Δθ)` (`ω = exp(-log²(w*/h*)/λ²)` from GT size) so near-squares keep a θ gradient. **`aux_angle_weight`** (default **1.0**, **0** = Gaussian only) and **`aux_angle_lambda`** (default **1.0**). Sampling `riou` is rejected. Recipe: [`dota_le90_1x_l1_kfiou_aux.json`](./dota_le90_1x_l1_kfiou_aux.json).

```bash
odet train --config configs/rotated_fcos/dota_le90_1x.json
odet train --config configs/rotated_fcos/dota_le90_3x.json
odet train --config configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json
odet train --config configs/rotated_fcos/hrsc2016_le90_1x.json
odet train --config configs/rotated_fcos/hrsc2016_le90_3x.json
make wizard CONFIG=configs/rotated_fcos/dota_le90_1x.json
```

`make eval-val` / `odet preds` use **`evaluation.preds_score_threshold`** or **0.05** (published protocol), not `production.score_threshold` or train-val `evaluation.score_threshold: 0.3`. Deploy / `image_demo` still use `production.score_threshold`.
