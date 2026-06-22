# Rotated Faster R-CNN

See the [main README](../../README.md) for installation and [configs/README.md](../README.md) for config layout.

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## Config Inheritance Notes

Config bases are merged in order, with later bases overriding earlier keys. Place `../_base_/fp16.json` after schedule bases when AMP should stay enabled, because `../_base_/schedules/1x.json` sets `training.use_amp` to `false`.

### Partial freeze: `training.freeze_backbone_epochs` / `training.freeze_rpn_epochs`

Independent **0-based** epoch thresholds (same index as checkpoints): while `epoch < freeze_backbone_epochs`, `backbone.*` is frozen; while `epoch < freeze_rpn_epochs`, `rpn_head.*` is frozen. The ROI head always trains. Use different values when changing anchor ratios so the RPN can adapt earlier than the backbone (or the reverse). **RetinaNet** has no `rpn_head`; `freeze_rpn_epochs` is a no-op. With `use_lr_param_groups: true`, frozen tensors stay in optimizer param groups until unfrozen. Example:

```json
"training": {
  "use_lr_param_groups": true,
  "freeze_backbone_epochs": 2,
  "freeze_rpn_epochs": 0
}
```

## Quick Training Health Checklist

Use this one-page checklist for any Rotated Faster R-CNN run.

### A) Run liveness

- Training runs inside `tmux` (or similar) so it survives editor/session disconnects.
- Log file keeps updating each epoch (`tail -f runs/<model>/<timestamp>/train.log`).
- No repeated runtime crashes (OOM, NCCL/collective failures, NaN/Inf losses).

### B) Learning signals

- Training loss trends downward over time (small short-term noise is normal).
- ROI matching hints improve in early epochs (more positives / better match rate).
- Validation forward pass time is stable (large jumps can indicate pipeline issues).

### C) Recall health

- `GT cover rate pre-eval-threshold` should rise or stabilize after warmup.
- `GT cover rate post-eval-threshold` should not trend downward for many evals.
- `GT lost by eval-threshold filtering` should stay controlled (not continuously growing).

### D) mAP checkpoints

- Compare only epochs where mAP is actually computed (`compute_map_every_n_epochs`).
- Expect noisy single points; judge trend across multiple checkpoints.
- If loss improves but mAP stalls, inspect score thresholds, NMS, and class balance.

### E) Precision / duplicates

- Watch `Avg Detections per Image` and duplicate-rate diagnostics (if enabled).
- High detections/image + low matched accuracy usually means too many weak boxes.
- High wrong-class overlap with decent IoU suggests classification confusion.

### F) Action guide

- **Continue**: loss down + mAP/coverage trend up or stable.
- **Tune thresholds/NMS**: recall is good, but precision and duplicates are poor.
- **Tune assignment/anchors/loss**: recall remains low (many missed GTs).
- **Stop and debug**: repeated runtime errors or sustained metric collapse.

<!-- [ALGORITHM] -->

## Abstract

MMRotate’s Rotated Faster R-CNN uses a **horizontal RPN** (axis-aligned proposals) and then regresses
rotated boxes in the RoI head. This repository’s `RotatedFasterRCNN` implementation follows that
structure: **horizontal RPN → horizontal RoIAlign → 5D rotated box regression → rotated NMS**.

## Architecture Overview

Rotated Faster R-CNN extends Faster R-CNN by using horizontal proposals and a rotated ROI regression head:

### Stage 1: Horizontal RPN (MMRotate-style)

The RPN generates **horizontal** proposals using horizontal anchors (angle \(= 0\)).

- **Anchor Design**: Horizontal anchors (angle 0) across FPN levels.
- **RPN Head**: Lightweight fully-convolutional network with:
  - Shared 3×3 convolutional layer
  - Classification branch: outputs objectness scores (1 channel per anchor)
  - Regression branch: outputs 4 parameters per anchor (dx, dy, dw, dh)
- **Box Encoding**: Uses `DeltaXYWHBBoxCoder` - predicts only position and size deltas for horizontal boxes.

### Stage 2: RoI head (horizontal RoIAlign + rotated regression)

The second stage extracts horizontal RoI features, then classifies and regresses rotated boxes:

- **Horizontal RoIAlign**: Uses `torchvision.ops.roi_align` on horizontal proposals (MMDet/MMRotate-style extractor).
- **Classification**: Predicts probability over K+1 classes (K object classes + background)
- **Regression**: Refines oriented bounding boxes using `DeltaXYWHTHBBoxCoder` (5 params w.r.t. horizontal RoI).
- **Class-Agnostic Regression**: Uses shared regression parameters across all classes (MMRotate format)

## Key Differences from Oriented R-CNN

Rotated Faster R-CNN differs from Oriented R-CNN in the proposal generation strategy:

| Aspect | Rotated Faster R-CNN (this repo) | Oriented R-CNN (this repo) |
|--------|-----------------------------------|-----------------------------|
| **RPN Anchors** | Horizontal anchors (angle 0) | Horizontal anchors (angle 0) |
| **RPN Regression** | 4 params (dx, dy, dw, dh) | 6 params midpoint offsets (dx, dy, dw, dh, da, db) |
| **RoIAlign** | Horizontal | Rotated |
| **RoI coder** | `DeltaXYWHTHBBoxCoder` | `DeltaXYWHTRBBoxCoder`-style (`proj_xy` option) |

## Box Encoding Schemes

### RPN Stage: DeltaXYWHBBoxCoder (4 parameters)

The RPN uses a 4-parameter horizontal-box encoding:

**Encoding** (from horizontal anchor to horizontal GT/proposal target):
- dx = (gt_cx - anchor_cx) / anchor_w
- dy = (gt_cy - anchor_cy) / anchor_h
- dw = log(gt_w / anchor_w)
- dh = log(gt_h / anchor_h)

**Decoding** (from anchor + deltas to horizontal xyxy proposal):
- pred_cx = anchor_cx + dx × anchor_w
- pred_cy = anchor_cy + dy × anchor_h
- pred_w = anchor_w × exp(dw)
- pred_h = anchor_h × exp(dh)

This approach is efficient because:
- Reduces regression parameters from 5 to 4
- The ROI head, not the RPN, predicts the final angle
- Simpler than predicting angle directly

### ROI Head Stage: DeltaXYWHTHBBoxCoder (5 parameters)

The ROI head uses a 5-parameter horizontal-to-rotated encoding:

**Encoding** (from horizontal RoI to oriented GT):
- dx = (gt_cx - proposal_cx) / proposal_w
- dy = (gt_cy - proposal_cy) / proposal_h
- dw = log(gt_w / proposal_w)
- dh = log(gt_h / proposal_h)
- da = (gt_angle - proposal_angle) / (norm_factor × π), with edge_swap optimization

**Decoding** (from proposal + deltas to refined oriented box):
- pred_cx = proposal_cx + dx × proposal_w
- pred_cy = proposal_cy + dy × proposal_h
- pred_w = proposal_w × exp(dw)
- pred_h = proposal_h × exp(dh)
- pred_angle = proposal_angle + da × (norm_factor × π), with edge_swap

**Key Parameters**:
- `norm_factor=2.0`: Scales angle delta to [-0.5, 0.5] range for le90 convention
- `edge_swap=True`: Optimizes angle representation by swapping width/height when beneficial

## Implementation Details

### Alignment with MMRotate

Our implementation targets MMRotate semantics:

- **RPN Stage**:
  - Horizontal anchors.
  - Regression outputs 4 parameters (dx, dy, dw, dh).
  
- **ROI Head**:
  - Horizontal RoIAlign and single 5D SmoothL1 regression loss over encoded targets (MMRotate-style).
  - **add_gt_as_proposals** (config `model.add_gt_as_proposals`, default `true`).

### Training defaults (DOTA-style recipes)

Typical published baselines use **SGD** \(momentum 0.9, weight decay 1e-4\), **batch size 2**, **12 epochs**, **MultiStepLR** with milestones at epochs **8** and **11** (\(\gamma=0.1\)), **lr=0.005**, **1024×1024** tiles, and **horizontal-anchor** matching with **`use_hbb_for_matching: true`**. [`dota_le90_1x.json`](./dota_le90_1x.json) is the full 1× recipe; [`dota_le90_3x.json`](./dota_le90_3x.json) inherits it and extends training to **36 epochs** with milestones **24/33**. Both use **FP32**, cross-entropy classification, and **ProbIoU** auxiliary ROI regression (`roi_box_reg_iou_loss_type: probiou`, `roi_box_reg_probiou_mode: l1`, weight **0.1**).

## Config files in this folder

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **1× DOTA pretrain** — 12 epochs, lr 0.005, MultiStep @ 8/11, train+val tiles, H+V+diagonal flips, ProbIoU ROI aux. |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× DOTA pretrain** — inherits 1×; 36 epochs, milestones [24, 33]. Hub: `rotated_faster_rcnn_dota_le90_3x`. |

### First run (1× baseline)

```bash
python tools/train.py --config configs/rotated_faster_rcnn/dota_le90_1x.json
```

### 3× from ImageNet

```bash
python tools/train.py --config configs/rotated_faster_rcnn/dota_le90_3x.json
```

## Results and models

DOTA1.0 (pretrain: **train+val / val**). mAP = **`make eval-val`** mAP50 (7,669 val tiles). Hub manifest **`eval_map50`** uses the same `odet preds` protocol — not training **`compute_map_final`** mAP (see [pretrained/README.md](../../pretrained/README.md)).

| Backbone | mAP (eval-val) | Angle | lr schd | Aug | BS | Config | Download |
| :----------------------: | :---: | :---: | :-----: | :-: | :--: | :----: | :----: |
| ResNet50 (1024,1024,200) | 76.41 | le90 | 3× | H+V+D | 2 | [`dota_le90_3x.json`](./dota_le90_3x.json) | `hf://rotated_faster_rcnn_dota_le90_3x` |

Eval report: [`predictions/20260615_082332/`](../../predictions/20260615_082332/).
