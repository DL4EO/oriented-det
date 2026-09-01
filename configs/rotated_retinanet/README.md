# Rotated RetinaNet

See the [main README](../../README.md) for installation and [configs/README.md](../README.md) for config layout.

> [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017). Rotated variant per [MMRotate](https://arxiv.org/abs/2203.06617) (Zhou et al., ACM MM 2022).

<!-- [ALGORITHM] -->

## Config files

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **1× DOTA pretrain** (12 epochs, lr 0.0025, batch 2, train+val tiles, H+V+diagonal flips). Rotated IoU anchor matching; mAP every **4** epochs. Hub: `rotated_retinanet_dota_le90_1x`. |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× DOTA pretrain** (36 epochs, milestones [24, 33]); inherits 1×. `compute_map_final: true`, exact CPU IoU for final mAP. Hub: `rotated_retinanet_dota_le90_3x`. |

Hub **`eval_map50`** in the manifest is from **`odet preds`** on val tiles (see [pretrained/README.md](../../pretrained/README.md)), not the training **final mAP** printed at the end of `train.log`.

### First run (1× baseline)

```bash
python tools/train.py --config configs/rotated_retinanet/dota_le90_1x.json
```

### ProbIoU as primary (tried; not a Hub recipe)

We ran a **1× ProbIoU primary** ablation (decoded ProbIoU + encoded L1 aux **0.1**, same loss stack pattern as Rotated Faster R-CNN 1×). In-training mAP was essentially tied with L1 (~63% vs L1 eval-val **64.14%**) with large vehicle-class regressions — **not published**. The code path remains (`roi_box_reg_main_loss_type: probiou`); there is no checked-in recipe config. Prefer the L1 1×/3× Hub recipes above.

**FPN P6/P7:** With `fpn_extra_level: true`, torchvision emits FPN keys `p6`/`p7`. Earlier releases dropped those keys in `extract_backbone_features`, training on 3 levels only and logging a stride mismatch warning — restart after updating to pick up all 5 levels (strides 8–128).

### 3× from ImageNet

```bash
python tools/train.py --config configs/rotated_retinanet/dota_le90_3x.json
```

Uses `evaluation.score_threshold: 0.3` and `model.max_detections_per_image: 300` during training validation (faster mAP). Deploy defaults stay in `production.*` (0.05 score, 2000 dets). Final mAP uses exact CPU IoU on the best checkpoint.

**TensorBoard:** The focal classification loss is logged as `train/loss_classifier` (same tag as Rotated Faster R-CNN), alongside `train/loss_box_reg`.

## Abstract

Rotated RetinaNet is a single-stage oriented object detection framework that extends RetinaNet to handle rotated bounding boxes. It uses oriented anchors with rotation angles and predicts oriented bounding boxes with 5 parameters (cx, cy, w, h, angle). The model uses focal loss for classification to address class imbalance and smooth L1 loss for regression. Unlike two-stage detectors, Rotated RetinaNet performs detection in a single forward pass, making it faster while maintaining competitive accuracy.

## Architecture Overview

Rotated RetinaNet is a single-stage detector that extends the standard RetinaNet framework (horizontal detection) to handle oriented objects:

### Backbone and Feature Pyramid Network (FPN)

- **Backbone**: ResNet (typically ResNet50 or ResNet101) extracts features from input images
- **FPN**: Feature Pyramid Network generates multi-scale feature maps {P3, P4, P5, P6, P7} with strides [8, 16, 32, 64, 128]
- **Feature Channels**: Each FPN level outputs 256 channels

### Detection Head

The detection head consists of two parallel branches attached to each FPN level:

- **Classification Branch**: 
  - Shared stack of 3×3 conv layers (`model.retinanet_stacked_convs`, MMRotate default **4**)
  - Outputs classification logits: `num_anchors × num_classes` per spatial location
  - Uses focal loss to handle class imbalance
  
- **Regression Branch**:
  - Same shared conv stack as classification
  - Outputs box regression: `num_anchors × 5` parameters per spatial location
  - Predicts oriented bounding boxes with format (cx, cy, w, h, angle)
  - Uses L1 loss for regression when `model.box_reg_loss_type: "l1"` (MMRotate default)

### Anchor Design

MMRotate’s `RotatedAnchorGenerator` builds **horizontal** priors in axis-aligned form, then represents them as `(cx, cy, w, h, theta)` with **`theta = 0`** at anchor init (rotation is predicted by the regression branch; see [MMRotate `RotatedAnchorGenerator`](https://mmrotate.readthedocs.io/en/stable/_modules/mmrotate/core/anchor/anchor_generator.html)). This repo matches that: **one reference angle per location** (`0` rad). **`model.anchor_angles` is not a supported config key** for any model type (strict `model` section rejects it).

- **Scales**: MMRotate RetinaNet uses `anchor_octave_base_scale=4` and `anchor_scales_per_octave=3` (3 scales per FPN level)
- **Aspect Ratios**: `[0.5, 1.0, 2.0]` (1:2, 1:1, 2:1)
- **Angles**: fixed `[0.0]` only (not user-configurable for this model family)
- **Total Anchors**: **9** per spatial location with MMRotate parity settings (3 scales × 3 ratios × 1 angle)

## Focal Loss

Focal Loss is the key innovation of RetinaNet that addresses the extreme class imbalance in dense object detection:

### Formula

**FL(p_t) = -α_t(1 - p_t)^γ log(p_t)**

Where:
- **p_t**: Predicted probability for the true class
- **α_t**: Weighting factor (typically α = 0.25 for Rotated RetinaNet)
- **γ**: Focusing parameter (typically γ = 2.0)

### Parameters

**Alpha (α)**:
- **Role**: Balances the importance of positive and negative examples
- **Default**: 0.25 (Rotated RetinaNet standard)
- **Effect**: Addresses class imbalance between foreground and background

**Gamma (γ)**:
- **Role**: "Focusing parameter" that down-weights easy examples
- **Default**: 2.0 (Rotated RetinaNet standard)
- **Effect**: 
  - When p_t → 1 (easy example), (1 - p_t)^γ → 0, down-weighting the loss
  - When p_t → 0 (hard example), (1 - p_t)^γ → 1, maintaining full loss
  - Higher γ values increase focus on hard examples

### Why Focal Loss?

In dense detection, there are typically thousands of background anchors per image but only a few positive anchors. Standard cross-entropy loss is dominated by the easy negative examples. Focal loss:
- Down-weights easy examples (both positive and negative)
- Focuses training on hard examples
- Prevents the large number of easy negatives from overwhelming the loss

## Box Encoding: DeltaXYWHAHBBoxCoder

Rotated RetinaNet uses a 5-parameter encoding scheme for oriented bounding boxes:

**Encoding** (from oriented anchor to oriented GT):
- dx = (gt_cx - anchor_cx) / anchor_w
- dy = (gt_cy - anchor_cy) / anchor_h
- dw = log(gt_w / anchor_w)
- dh = log(gt_h / anchor_h)
- da = (gt_angle - anchor_angle) / (norm_factor × π), with edge_swap optimization

**Decoding** (from anchor + deltas to oriented box):
- pred_cx = anchor_cx + dx × anchor_w
- pred_cy = anchor_cy + dy × anchor_h
- pred_w = anchor_w × exp(dw)
- pred_h = anchor_h × exp(dh)
- pred_angle = anchor_angle + da × (norm_factor × π), with edge_swap

**Key Parameters**:
- `norm_factor=2.0`: Scales angle delta to [-0.5, 0.5] range for le90 convention
- `edge_swap=True`: Optimizes angle representation by swapping width/height when beneficial

## Key Differences from Two-Stage Detectors

| Aspect | Rotated RetinaNet | Two-Stage Detectors |
|--------|-------------------|---------------------|
| **Architecture** | Single-stage (direct prediction) | Two-stage (proposals → refinement) |
| **Speed** | Faster inference | Slower inference |
| **Complexity** | Simpler pipeline | More complex (RPN + ROI head) |
| **Anchor Matching** | All anchors evaluated | Only proposals evaluated |
| **Loss Function** | Focal loss (handles imbalance) | Cross-entropy (with sampling) |
| **Memory** | Higher (all anchors) | Lower (only proposals) |

## Implementation Details

### Alignment with MMRotate

[`dota_le90_1x.json`](./dota_le90_1x.json) matches [MMRotate Rotated RetinaNet 1× le90](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py):

- **FPN**: `fpn_returned_layers: [2, 3, 4]` (C3–C5) + `fpn_extra_level: true` for stride-128 P7
- **Anchors**: `anchor_octave_base_scale: 4`, `anchor_scales_per_octave: 3`, ratios `[0.5, 1.0, 2.0]`, angle `0`
- **Head**: separate `cls_convs` / `reg_convs` (4×3×3 each) + 3×3 `conv_cls` / `conv_bbox` (MMRotate `RetinaHead`)
- **FPN extra levels**: `LastLevelP6P7` convs on C5 (`fpn_extra_level: true`), not max-pool P6
- **Assigner**: rotated IoU (`use_hbb_for_matching: false`), pos 0.5 / neg 0.4, `min_pos_iou=0` with low-quality matching
- **Evaluation**: mAP every 4 epochs (`compute_map_every_n_epochs: 4`); non-mAP val epochs skip CPU GT–IoU matching (forward + detection counts only)
- **Inference (val/train)**: GPU sampling NMS (`model.final_nms_use_cpu: false`); class-aware by default (`model.nms_class_agnostic: false`); set `model.nms_class_agnostic: true` (and `production.nms_class_agnostic` if deploy should match) for lookalike vehicle classes; pre-NMS score filter at `inference_pre_nms_score_threshold` (0.05)
- **Box coder**: `roi_norm_factor: null`, `roi_edge_swap: true`, L1 regression loss
- **Schedule**: 12 epochs, lr 0.0025, milestones [8, 11], batch 2, trainval tiles, diagonal flips
- **Inference**: score 0.05, NMS IoU **0.1**, max 2000 dets/image

### MS+RR (future)

Stronger DOTA recipes often use **MS** (multi-scale tiling, e.g. overlap 500 px) and **RR** (`PolyRandomRotate`). Neither is in the current 1×/3× Hub recipes; use `dota_le90_1x.json` / `dota_le90_3x.json` for the flip-only baseline first.

### Anchor Assignment Strategy

Following MMRotate's design:
- **Positive anchors**: IoU > 0.5 with ground-truth oriented box (configurable via `positive_iou_threshold`)
- **Negative anchors**: IoU < 0.4 with all ground-truth boxes (configurable via `negative_iou_threshold`)
- **Ignored anchors**: Between thresholds (0.4 ≤ IoU ≤ 0.5)

Assignment uses **rotated IoU** (`use_hbb_for_matching: false`, MMRotate `RBboxOverlaps2D`).

### Training Configuration (MMRotate 1×)

From [`dota_le90_1x.json`](./dota_le90_1x.json):

- **Learning rate**: 0.0025, divided by 10 at epochs **8** and **11**
- **Optimizer**: SGD, momentum 0.9, weight decay 1e-4, max grad norm 35
- **Batch size**: 2
- **Epochs**: 12
- **Image size**: 1024×1024 tiles, overlap 200
- **Augmentation**: horizontal + vertical + **diagonal** flips (RRandomFlip parity)
- **Focal loss**: `alpha=0.25`, `gamma=2.0`

### Inference Configuration

- **Score threshold**: 0.05
- **NMS**: Oriented NMS IoU **0.1** (`model.final_nms_iou_threshold`)
- **Max detections**: 2000 per image

### Memory Efficiency

Our implementation processes anchors per-level to avoid memory issues:
- Each FPN level is processed independently
- Anchors are not concatenated across levels (avoids creating huge tensors)
- Losses are accumulated across levels
- This approach scales well to large images with millions of anchors

## Performance

See **Results and models** below for this repo’s eval-val numbers. Rotated RetinaNet is the lighter single-stage baseline in the Hub zoo (vs Oriented R-CNN / Faster R-CNN).

## Results and models

DOTA1.0 (pretrain: **train+val / val**). mAP = **`make eval-val`** mAP50 (7,669 val tiles).

| Backbone | mAP (eval-val) | Angle | lr schd | Aug | BS | Config | Final config | Final log | Download |
| :----------------------: | :---: | :---: | :-----: | :-: | :--: | :----: | :----------: | :-------: | :----: |
| ResNet50 (1024,1024,200) | 64.14 | le90 | 1× | H+V+D | 2 | [`dota_le90_1x.json`](./dota_le90_1x.json) | [`rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json`](../../pretrained/rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json) | [`rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.log`](../../pretrained/rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.log) | `hf://rotated_retinanet_dota_le90_1x` |
| ResNet50 (1024,1024,200) | 71.52 | le90 | 3× | H+V+D | 2 | [`dota_le90_3x.json`](./dota_le90_3x.json) | [`rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.json`](../../pretrained/rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.json) | [`rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.log`](../../pretrained/rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.log) | `hf://rotated_retinanet_dota_le90_3x` |

Eval reports: [`docs/eval-reports/rotated_retinanet_dota_le90_1x/`](../../docs/eval-reports/rotated_retinanet_dota_le90_1x/model_analysis.md), [`docs/eval-reports/rotated_retinanet_dota_le90_3x/`](../../docs/eval-reports/rotated_retinanet_dota_le90_3x/model_analysis.md).

## Usage

### Training

```bash
python tools/train.py --config configs/rotated_retinanet/dota_le90_1x.json
```

### Training with FP16 (mixed precision)

Either pass AMP on the CLI (merged after config load):

```bash
python tools/train.py --config configs/rotated_retinanet/dota_le90_3x.json --batch-size 2 --use-amp
```

Or add `"../_base_/fp16.json"` to the `_base_` list in a derived config.

### Override Parameters

You can override any parameter from the command line:

```bash
python tools/train.py \
    --config configs/rotated_retinanet/dota_le90_3x.json \
    --batch-size 4 \
    --use-amp
```

## Citation

```
@article{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  journal={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
