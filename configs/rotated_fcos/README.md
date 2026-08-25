# Rotated FCOS configs

Anchor-free single-stage oriented detector (v0.2). **DistanceAnglePointCoder** (`left, top, right, bottom, angle`), center-in-OBB assigner with per-level regress ranges, centerness, and box regression via **L1** (baseline), optional decoded **KFIoU** aux, or decoded **rIoU** primary (`1 -` differentiable polygon IoU). Monte-Carlo `pairwise_rotated_iou` is **not** a train loss.

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **1× DOTA L1** (12 epochs, **lr 2.5e-4**, batch 2, train+val tiles, H+V+diagonal flips). P3–P7. mAP every **4** epochs. |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× DOTA L1** (36 epochs, milestones `[24, 33]`); inherits 1×. Warmup **2000**. Exact CPU IoU for final mAP. |
| [`dota_le90_1x_kfiou_aux.json`](./dota_le90_1x_kfiou_aux.json) | **1× L1 + KFIoU aux 0.1** — keep L1 lr **2.5e-4**. Heading term on (`aux_angle_weight` 1.0, `λ` 1.0). |
| [`dota_le90_3x_kfiou_aux.json`](./dota_le90_3x_kfiou_aux.json) | **3× L1 + KFIoU aux** — inherits 1× aux; 36 epochs, milestones `[24, 33]`, warmup **2000**. Same lr **2.5e-4**. Hub slug. |
| [`dota_le90_1x_riou.json`](./dota_le90_1x_riou.json) | **1× decoded rIoU primary** (`box_reg_loss_type: riou`, no aux). **lr 2.5e-3**. |
| [`dota_le90_3x_riou.json`](./dota_le90_3x_riou.json) | **3× decoded rIoU** — inherits 1× rIoU; 36 epochs, milestones `[24, 33]`, warmup **2000**. Same lr **2.5e-3**. **Hub recipe.** |

Base model: [`../_base_/models/rotated_fcos_r50.json`](../_base_/models/rotated_fcos_r50.json).

## Results

DOTA1.0 (pretrain: **train+val / val**). Published mAP = **`make eval-val`** mAP50 (7,669 val tiles, `filter_empty_gt=false`, score ≥ **0.05**, **`production.final_nms_iou_threshold: 0.1`**).

| Schedule | Config | Training run | Checkpoint | Train-time final mAP† | eval-val mAP50 | Report / Hub |
| :------: | :----: | :----------: | :--------: | :-------------------: | :------------: | :----------- |
| 1× L1 | [`dota_le90_1x.json`](./dota_le90_1x.json) | `runs/rotated_fcos/20260811-064719` | `best_mAP_0.70.pth` | ~70% | **63.07%**‡ | — |
| 3× L1 | [`dota_le90_3x.json`](./dota_le90_3x.json) | `runs/rotated_fcos/20260812-105204` | `best_mAP_0.72.pth` | **72.22%** | **73.92%** | [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md) (local L1 baseline) |
| 1× L1 + KFIoU aux | [`dota_le90_1x_kfiou_aux.json`](./dota_le90_1x_kfiou_aux.json) | `runs/rotated_fcos/20260814-074221` | `best_mAP_0.77.pth` | **76.53%** | **69.62%** | `predictions/20260814_181423/` |
| 1× rIoU | [`dota_le90_1x_riou.json`](./dota_le90_1x_riou.json) | `runs/rotated_fcos/20260821-071344` | `best_mAP_0.80.pth` | **80.43%** | **74.04%** | `predictions/20260822_152147/` |
| 3× L1 + KFIoU aux | [`dota_le90_3x_kfiou_aux.json`](./dota_le90_3x_kfiou_aux.json) | `runs/rotated_fcos/20260818-100049` | `best_mAP_0.84.pth` | **84.07%** | **77.18%** | Hub `rotated_fcos_dota_le90_3x_kfiou_aux`; [`docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md) |
| **3× rIoU** | [`dota_le90_3x_riou.json`](./dota_le90_3x_riou.json) | `runs/rotated_fcos/20260822-153943` | `best_mAP_0.88.pth` | **88.03%** | **81.58%** | Hub `rotated_fcos_dota_le90_3x_riou`; [`docs/eval-reports/rotated_fcos_dota_le90_3x_riou/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x_riou/model_analysis.md) |

† Training periodic/final mAP uses `evaluation.score_threshold: 0.3` and non-empty tiles only.  
‡ 1× L1 eval-val was measured with production NMS **0.3**; re-run with NMS **0.1** before comparing to 3×.

**3× rIoU** is **+7.7** eval-val mAP50 vs this repo’s **3× L1** (81.58% vs 73.92%) and **+4.4** vs Hub **3× KFIoU aux** (77.18%). Viewer:

```bash
make viewer VIEWER_PRED_DIR=predictions/20260824_212907 DOTA_DATA_ROOT=/path/to/DOTA-v1.0-tiled
```

```bash
odet pretrained download rotated_fcos_dota_le90_3x_riou
```

## Eval NMS (dense scenes)

`model.final_nms_iou_threshold` and **`production.final_nms_iou_threshold`** are both **0.1** (MMRotate FCOS). `make eval-val` loads the **experiment** `config.json` (not the recipe file alone) — keep `production.*` aligned there after training, or override via CLI.

## L1 vs rIoU recipe notes

- L1 recipes use `learning_rate` **2.5e-4** (0.0025 on L1 was unstable here).
- Decoded **rIoU** primary ([`dota_le90_1x_riou.json`](./dota_le90_1x_riou.json), [`dota_le90_3x_riou.json`](./dota_le90_3x_riou.json)) uses **0.0025**. Keep center radius 1.5, `max_detections_per_image` **2000**.
- FCOS `riou` is differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`), not sampling `pairwise_rotated_iou`. ROI `riou` is unchanged (still sampling).

## Decoded aux (KFIoU)

Keep **`box_reg_loss_type: l1`**. Set **`aux_loss_type: kfiou`** and **`aux_loss_weight`** (try **0.1**). Aux is decoded (stride restore when `norm_on_bbox`) and **centerness-weighted** like L1; logged as `loss_box_reg_aux`. Each positive’s aux is Gaussian overlap **plus** an aspect-gated heading term `ω sin²(2Δθ)` (`ω = exp(-log²(w*/h*)/λ²)` from GT size) so near-squares keep a θ gradient. **`aux_angle_weight`** (default **1.0**, **0** = Gaussian only) and **`aux_angle_lambda`** (default **1.0**). Sampling `riou` is rejected.

```bash
odet train --config configs/rotated_fcos/dota_le90_1x_kfiou_aux.json
odet train --config configs/rotated_fcos/dota_le90_3x_kfiou_aux.json
odet train --config configs/rotated_fcos/dota_le90_1x_riou.json
odet train --config configs/rotated_fcos/dota_le90_3x_riou.json
make wizard CONFIG=configs/rotated_fcos/dota_le90_3x_riou.json
```

`production.score_threshold` stays **0.05** (from 1×) so `make eval-val` matches the published protocol. Periodic train mAP uses `evaluation.score_threshold: 0.3` for speed.

TF/ONNX detect export: `odet export-tf --mode rotated_fcos_pre_nms --config … --checkpoint …` (see [`export/README.md`](../../export/README.md)).
