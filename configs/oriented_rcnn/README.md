# Oriented R-CNN

See the [main README](../../README.md) for installation and [configs/README.md](../README.md) for config layout.

> [Oriented R-CNN for Object Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

Current state-of-the-art two-stage detectors generate oriented proposals through time-consuming schemes. This diminishes the detectors' speed, thereby becoming the computational bottleneck in advanced oriented object detection systems. This work proposes an effective and simple oriented object detection framework, termed Oriented R-CNN, which is a general two-stage oriented detector with promising accuracy and efficiency. To be specific, in the first stage, we propose an oriented Region Proposal Network (oriented RPN) that directly generates high-quality oriented proposals in a nearly cost-free manner. The second stage is oriented R-CNN head for refining oriented Regions of Interest (oriented RoIs) and recognizing them.

## Architecture Overview

Oriented R-CNN is a two-stage detector that addresses the computational bottleneck in oriented proposal generation. The architecture consists of:

### Stage 1: Oriented RPN

The oriented RPN in this codebase follows the Oriented R-CNN paper: it starts from **horizontal anchors** (axis‑aligned, angle = 0) and uses a **midpoint offset representation** to obtain oriented proposals:

- **Anchor Design**: Uses horizontal anchors with 3 aspect ratios (1:2, 1:1, 2:1) at each spatial location across FPN levels {P2, P3, P4, P5, P6}
- **Anchor Scales**: Anchor areas are 32², 64², 128², 256², 512² pixels on respective FPN levels
- **RPN Head**: Lightweight fully-convolutional network with:
  - Shared 3×3 convolutional layer
  - Classification branch: outputs objectness scores (1 channel per anchor)
  - Regression branch: outputs 6 parameters per anchor (δx, δy, δw, δh, δα, δβ)
- **Parameter Efficiency**: Uses approximately 1/3000 the parameters of RoI Transformer+ and 1/15 of rotated RPN

### Stage 2: Oriented R-CNN Head

The second stage refines oriented proposals and performs classification:

- **Rotated RoIAlign**: Extracts rotation-invariant features from oriented proposals
  - Converts oriented proposals (parallelograms) to oriented rectangles by extending the shorter diagonal
  - Projects oriented rectangles to feature maps using rotation transformation
  - Divides each RoI into m×m grids (default m=7) and samples features using bilinear interpolation
  - Processes boxes in chunks for memory efficiency during training
- **Classification**: Predicts probability over K+1 classes (K object classes + background)
- **Regression**: Refines oriented bounding boxes for each object class (class-agnostic regression)

## Midpoint Offset Representation

The key innovation of Oriented R-CNN is the **midpoint offset representation** for oriented objects. An oriented bounding box is represented by 6 parameters:

**O = (x, y, w, h, Δα, Δβ)**

Where:
- **(x, y)**: Center coordinates of the external rectangle (axis-aligned bounding box)
- **(w, h)**: Width and height of the external rectangle
- **Δα**: Offset of the top vertex (v₁) relative to the midpoint of the top side (x, y - h/2)
- **Δβ**: Offset of the right vertex (v₂) relative to the midpoint of the right side (x + w/2, y)

The four vertices of the oriented box are computed as:
- v₁ = (x + Δα, y - h/2)
- v₂ = (x + w/2, y + Δβ)
- v₃ = (x - Δα, y + h/2)
- v₄ = (x - w/2, y - Δβ)

**Encoding Process** (from horizontal proposal to oriented box):
Given a horizontal proposal (px, py, pw, ph) and ground-truth oriented box O = (xg, yg, wg, hg, Δαg, Δβg):
- dx = (xg - px) / pw
- dy = (yg - py) / ph
- dw = log(wg / pw)
- dh = log(hg / ph)
- da = (ga - gx) / gw, where ga is the x-coordinate of the top vertex
- db = (gb - gy) / gh, where gb is the y-coordinate of the right vertex

**Decoding Process** (from horizontal proposal + deltas to oriented box):
Given horizontal proposal (px, py, pw, ph) and deltas (dx, dy, dw, dh, da, db):
- gx = px + pw × dx
- gy = py + ph × dy
- gw = pw × exp(dw)
- gh = ph × exp(dh)
- ga = gx + da × gw (top vertex x-coordinate)
- gb = gy + db × gh (right vertex y-coordinate)
- Then reconstruct the oriented box from these parameters

This representation:
- Inherits the horizontal regression mechanism from Faster R-CNN
- Provides bounded constraints for predicting oriented proposals (Δα, Δβ are bounded to [-0.5, 0.5])
- Enables efficient proposal generation without dense rotated anchors
- Uses approximately 1/3000 the parameters of RoI Transformer+ and 1/15 of rotated RPN

## Implementation Details

### Model Variants

This repository exposes two MMRotate‑style two‑stage detectors:

1. **`OrientedRCNN`** (paper-faithful)
   - Stage 1: **6D midpoint RPN** (axis‑aligned anchors, predicts midpoint deltas directly and is supervised in the RPN loss)
   - Stage 2: Rotated RoIAlign + oriented ROI head

2. **`RotatedFasterRCNN`** (MMRotate-style)
   - Stage 1: Horizontal RPN (xyxy proposals)
   - Stage 2: Horizontal RoIAlign + rotated box regression

### Alignment with Original Paper

Our `OrientedRCNN` implementation aligns with the original Oriented R-CNN paper (ICCV 2021):

1. **Horizontal Anchors**: Uses axis-aligned anchors (not rotated anchors) with standard aspect ratios
   - Default configs in `configs/_base_/models/oriented_rcnn_r50.json`:  
     `anchor_scales=[8]`, `anchor_ratios=[0.5, 1.0, 2.0]` (horizontal RPN priors; fixed in code)
2. **6-Parameter Regression**: RPN regression branch outputs 6 parameters (dx, dy, dw, dh, da, db) via `MidpointOffsetCoder`
3. **Two-Stage Design**: 6D midpoint RPN → oriented proposals → Oriented ROI head
4. **Rotated RoIAlign**: Uses oriented ROI alignment to extract rotation-invariant features

### Alignment with MMRotate

Our implementation is compatible with MMRotate's Oriented R-CNN implementation:

- **RPN Stage**:
  - Horizontal RPN priors (not configurable via JSON).
  - Regression outputs **6 parameters** (dx, dy, dw, dh, da, db) and is trained with `MidpointOffsetCoder`.

- **ROI Head**:
  - Uses `DeltaXYWHTRBBoxCoder`-style targets with optional `proj_xy`.
  - Parity config sets `roi_proj_xy=true`, `roi_norm_factor=null`, and `edge_swap=true`.

## Config files in this folder

| File | Purpose |
|------|---------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | **Full DOTA 1× recipe** — plain ROI cross-entropy (MMRotate-style baseline) |
| [`dota_le90_1x_class_weighted.json`](./dota_le90_1x_class_weighted.json) | **1× class-balance experiment** — focal + effective-num weights, overrides for weak classes |
| [`dota_le90_3x.json`](./dota_le90_3x.json) | **3× DOTA pretrain** — inherits 1×; 36 epochs, milestones [24, 33] |

### Class-weighted 1× recipe (`dota_le90_1x_class_weighted.json`)

Inherits [`dota_le90_1x.json`](./dota_le90_1x.json) and only changes the **`loss`** block:

| Setting | Value | Why |
|---------|-------|-----|
| `loss_type` | `focal_weighted` | Focal loss down-weights easy negatives; class weights address imbalance |
| `class_weight_method` | `effective_num` (β=0.9999) | Stronger boost for rare DOTA classes than `sqrt` |
| `focal_alpha` / `focal_gamma` | 0.25 / 2.0 | Standard focal settings for dense detection |
| `class_weight_schedule_type` | `linear_ramp` (epochs 0→4) | Ramp weights from uniform → computed to avoid early instability |
| `class_weight_overrides` | see config | Extra up-weight on **weak frequent** classes from baseline 1× eval: `small-vehicle`, `ship`, plus `bridge`, `harbor`, `swimming-pool`, `storage-tank` |

Rare classes (e.g. `ground-track-field`, `helicopter`) already reach the **3.0 clip** under `effective_num` without overrides.

**Train:**

```bash
odet train --config configs/oriented_rcnn/dota_le90_1x_class_weighted.json
```

**Optional fine-tune** from a plain-CE 1× checkpoint (faster A/B): set in config or override:

```json
"checkpoint": {
  "load_from_experiment": "runs/oriented_rcnn/20260616-030231",
  "resume_from_checkpoint_epoch": false,
  "load_optimizer_state": false,
  "load_scheduler_state": false
}
```

That loads `best_*.pth` and retrains with the new loss for 12 epochs.

### Loss Functions

**RPN Loss**:
- Classification: Cross-entropy loss for objectness prediction
- Regression: Smooth L1 loss for 6D midpoint-offset regression (dx, dy, dw, dh, da, db)

**ROI Loss**:
- Classification: Cross-entropy loss (or focal loss with `roi_loss_type="focal"`)
- Regression: Smooth L1 loss for oriented box refinement (5 parameters: dx, dy, dw, dh, da)

### Anchor Assignment Strategy

Following the paper's design:
- **Positive anchors**: IoU > 0.7 with ground-truth external rectangle, OR highest IoU > 0.3
- **Negative anchors**: IoU < 0.3 with all ground-truth boxes
- **Invalid anchors**: Ignored during training (0.3 ≤ IoU ≤ 0.7, not highest)

Note: IoU computation uses the **external rectangles** (axis-aligned bounding boxes) of oriented ground-truth boxes, not the oriented boxes themselves.

**Train-time matching (`use_hbb_for_matching: true`):** RPN and ROI assigners use **axis-aligned (HBB) IoU** on GPU (chunked anchor matcher + `oriented_box_hbb_iou_gpu` for proposals). Predictions remain oriented; only assignment uses HBB. This matches the Rotated Faster R-CNN / RetinaNet DOTA recipes and avoids slow exact rotated IoU during matching. Set explicitly in [`dota_le90_1x.json`](./dota_le90_1x.json) and [`oriented_rcnn_r50.json`](../_base_/models/oriented_rcnn_r50.json).

### Training Configuration

Default hyperparameters (DOTA dataset):
- **Learning rate**: 0.005 (initial), divided by 10 at epochs 8 and 11
- **Optimizer**: SGD with momentum 0.9, weight decay 0.0001
- **Batch size**: 2 (MMRotate default; see **Throughput** below)
- **Epochs**: 12
- **Image size**: 1024×1024 patches (stride 824, overlap 200)
- **Data augmentation**: Horizontal and vertical flipping

### Throughput and batch size

Oriented R-CNN is slower per step than Rotated Faster R-CNN (~2–3×) because **`OrientedROIAlign`** (rotated `grid_sample` over ~2000 proposals/image) dominates — not anchor/proposal IoU matching.

| Setting | Guidance |
|---------|----------|
| **`batch_size: 2`** | Default in DOTA recipes; matches MMRotate; safe on 24 GB GPUs (e.g. L4). |
| **Larger batch (4+)** | Can improve GPU utilization **if memory allows**. Memory scales roughly with `batch_size × rpn_post_nms_top_n` RoIs. Try `batch_size: 4` with **`use_amp: true`** first; batch 6 is unlikely to fit without lowering `rpn_post_nms_top_n` or `roi_batch_size_per_image`. |
| **Learning rate** | If you increase batch size, scale LR linearly (e.g. bs 4 → `learning_rate: 0.01` at the reference bs 2). |
| **vs RetinaNet batch 6** | One-stage RetinaNet has no per-image RoI align over thousands of proposals; Oriented R-CNN usually **cannot** use the same batch size on the same GPU. |

**Do not** set `use_hbb_for_matching: false` for speed — that switches assignment to exact rotated IoU and is typically **slower**, not faster.

### Inference Configuration

- **RPN**: Top 2000 proposals per FPN level before NMS, horizontal NMS with IoU threshold 0.8
- **Final proposals**: Top 1000 proposals after merging all levels
- **ROI NMS**: Poly NMS with IoU threshold 0.1 per class
- **Score threshold**: 0.05 (configurable via `eval_score_threshold`)

### Rotated RoIAlign Implementation

The Rotated RoIAlign operation extracts rotation-invariant features from oriented proposals:

1. **Parallelogram to Rectangle Conversion**: Oriented proposals from RPN are typically parallelograms. Each parallelogram is converted to an oriented rectangle by extending the shorter diagonal to match the longer diagonal length.

2. **Feature Map Projection**: The oriented rectangle (x, y, w, h, θ) is projected to the feature map F with stride s:
   - x_r = x / s, y_r = y / s, w_r = w / s, h_r = h / s, θ_r = θ

3. **Grid Sampling**: Each rotated RoI is divided into m×m grids (default m=7). For each grid cell (i, j), features are sampled using bilinear interpolation with rotation transformation R(·) applied to map from box-local coordinates to feature map coordinates.

4. **Memory Optimization**: Our implementation processes boxes in chunks (default `chunk_size=32`) to avoid memory explosion during backward pass. Gradient checkpointing can be enabled for further memory reduction (~2x less memory, ~30% slower).

This implementation aligns with the paper's description and provides efficient feature extraction while maintaining rotation invariance.

## Performance

Reported results on DOTA dataset (from paper):
- **ResNet50-FPN**: 75.87% mAP at 15.1 FPS (1024×1024, RTX 2080Ti)
- **ResNet101-FPN**: 76.28% mAP
- **Multi-scale**: 80.87% mAP (R-50-FPN)

The method achieves state-of-the-art accuracy while maintaining competitive efficiency compared to one-stage detectors.

## Results and models

### OrientedDet

`dota_le90_1x.json` trained on DOTA train+val tiles and evaluated on the full val tile split (`filter_empty_gt=false`) reaches **74.79% mAP50**. Hub slug: `oriented_rcnn_dota_le90_1x`; eval report: `predictions/20260618_140030/model_analysis_20260618_175528.md`.

| Config | Final config | Final log | Schedule | Training run | Checkpoint | eval-val mAP50 | Hub slug |
|--------|--------------|-----------|----------|--------------|------------|----------------|----------|
| [`dota_le90_1x.json`](./dota_le90_1x.json) | [`oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.json`](../../pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.json) | [`oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.log`](../../pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.log) | 1× (12 ep) | `runs/oriented_rcnn/20260616-030231` | `best_mAP_0.78.pth` | 74.79% | `oriented_rcnn_dota_le90_1x` |

### MMRotate reference

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | MMRotate config name |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 75.69 | le90  |   1x    |   8.46   |      16.2      |  -  |     2      | `oriented_rcnn_r50_fpn_1x_dota_le90` |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90_20220127_100150.log.json)                   |
| ResNet50 (1024,1024,200) | 75.63 | le90  |   1x    |   7.37   |      21.2      |  -  |     2      | `oriented_rcnn_r50_fpn_fp16_1x_dota_le90` |         [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90_20220303_195049.log.json)         |

## Usage

### Training

```bash
# Edit dataset paths in the config, then:
odet train --config configs/oriented_rcnn/dota_le90_1x.json

# 1× with focal + class weights (weak-class experiment):
odet train --config configs/oriented_rcnn/dota_le90_1x_class_weighted.json

# Primary benchmark (36 epochs):
odet train --config configs/oriented_rcnn/dota_le90_3x.json
```

### Override Parameters

You can override parameters from the command line:

```bash
odet train \
    --config configs/oriented_rcnn/dota_le90_3x.json \
    --batch-size 4 \
    --use-amp
```

## Citation

```
@InProceedings{Xie_2021_ICCV,
  author = {Xie, Xingxing and Cheng, Gong and Wang, Jiabao and Yao, Xiwen and Han, Junwei},
  title = {Oriented R-CNN for Object Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2021},
  pages = {3520-3529} }
```
