# Models

Detectors: **Oriented R-CNN**, **Rotated Faster R-CNN**, **Rotated RetinaNet**, **Rotated FCOS**. Config: [Configuration](../../docs/user-guide/configuration.md) (`model_type`). User guide: [Models](../../docs/user-guide/models.md).

## RPN anchor angles (Python only)

Training JSON / `ModelConfig` does **not** expose `anchor_angles` (horizontal RPN priors are the default and match MMRotate-style setups). For experiments or legacy checkpoint matching, you may pass **`anchor_angles=[...]`** into **`RotatedFasterRCNN`**, **`OrientedRCNN`**, or **`RotatedRetinaNet`** constructors in code. Multi-angle RPN banks are usually **not** recommended for accuracy or speed versus the default.

## Training vs export paths

| Mode | Code path | Notes |
|------|-----------|--------|
| **Training** (`model.train()`) | `RotatedFasterRCNN.forward` → `self.roi_align` → eager `horizontal_roi_align` (per-FPN loop) | Never calls `faster_rcnn_inference.py`. `torch.onnx.is_in_onnx_export()` is false. |
| **Eval / deploy (Faster)** | `faster_rcnn_inference.faster_rcnn_inference()` | Shared with PyTorch inference and export wrappers. |
| **ONNX export (Faster)** | Same as eval + `horizontal_roi_align` **masked** branch when `is_in_onnx_export()` | Fixed-shape RoIAlign for traceability; numerically equivalent to eager path (see `tests/test_roi.py::test_horizontal_roi_align_eager_matches_onnx_export_path`). |
| **Eval (Oriented R-CNN)** | Inline in `OrientedRCNN.forward` (midpoint RPN + `OrientedROIAlign`) | Set `model._deterministic_rpn = True` for export-parity comparisons. |
| **ONNX export (Oriented)** | `oriented_rcnn_inference_pre_nms_padded` + `oriented_roi_align` **masked** branch | Packed `grid_sample` (no feature `Expand`); always-on proposal `cat` pad; same Keras detect bundle as Faster. |
| **Eval (FCOS)** | `rotated_fcos_decode_pre_nms` + class-aware rotated NMS | Anchor-free decode (`DistanceAnglePointCoder`). |
| **ONNX / TF export (FCOS)** | `rotated_fcos_inference_pre_nms_padded` | Same pre-NMS tensors as two-stage; Keras detect bundle does NMS. Mode: `rotated_fcos_pre_nms`. |

Regression guards: `tests/test_roi.py` (eager vs export RoIAlign), `tests/test_models.py::TestRotatedFasterRCNN::test_full_training_forward_and_backward`, `export/tests/test_faster_rcnn_export_parity.py`, `export/tests/test_oriented_rcnn_export_parity.py`, `export/tests/test_rotated_fcos_export_parity.py`.

## Shared inference (`faster_rcnn_inference.py` / `oriented_rcnn_inference.py`)

`RotatedFasterRCNN` eval forwards through `faster_rcnn_inference()` (decode + rotated NMS). The same module powers ONNX export (`export/wrappers.RotatedFasterRCNNPreNmsExportWrapper`) with deterministic RPN top-k and padded proposals for traceable ROI align.

`OrientedRCNN` export uses `oriented_rcnn_inference_pre_nms_padded` (`export/wrappers.OrientedRCNNPreNmsExportWrapper`) with deterministic midpoint RPN and padded oriented proposals.

`RotatedFCOS` export uses `rotated_fcos_inference_pre_nms_padded` (`export/wrappers.RotatedFCOSPreNmsExportWrapper`): backbone + head + decode, padded to `nms_pre ×` FPN levels. Same Keras detect bundle as the two-stage models.

## Rotated Faster R-CNN Proposal Filtering

`RotatedFasterRCNN` defaults (RPN/ROI IoU assign thresholds, ROI `target_stds`, post-RPN caps, final rotated NMS IoU) follow MMRotate’s DOTA le90 Faster R-CNN config unless overridden.

ROI training loss uses `compute_horizontal_roi_loss` (Rotated Faster R-CNN). Defaults match MMRotate via `compute_horizontal_roi_loss_mmrotate` (`reg_norm='sampled_all'`). Configurable via:

- `roi_box_reg_main_loss_type`: `smooth_l1` (encoded primary, default) or decoded `probiou` / `riou` / `kfiou`
- `roi_box_reg_norm`: `sampled_all` (MMDet avg_factor over pos+neg sample count) or `positives_only` (per-dim mean over positives)
- `roi_box_reg_iou_weight`: decoded aux when main is Smooth L1 (optional `roi_box_reg_iou_schedule_*`)
- `roi_box_reg_smooth_l1_aux_weight`: encoded Smooth L1 aux when main is decoded
- `roi_box_reg_angle_weight` (5th encoded dim; optional `roi_box_reg_angle_schedule_*`), `roi_match_low_quality`, `roi_min_pos_iou`
- `roi_proj_xy`: encode/decode ROI dx/dy in the proposal local frame (`true` in DOTA base configs; no-op for axis-aligned xyxy RoIs, required for non-horizontal proposal angles)

Encoded Smooth L1 (main or aux) applies **directly to all five encoded channels** (MMRotate / MMDet `L1Loss` on bbox targets), including the angle channel after `norm_factor` and `target_stds` normalization. Optional `roi_box_reg_angle_weight` scales only the 5th channel.

**Oriented R-CNN** uses `compute_oriented_roi_loss` with the same encoded Smooth L1 and MMDet `avg_factor` normalization (`roi_box_reg_norm: sampled_all` by default).

**Rotated RetinaNet** uses `compute_oriented_retinanet_loss`. Set `roi_box_reg_main_loss_type: probiou` (or `riou` / `kfiou`) for a decoded primary loss; add encoded L1/Smooth L1 aux via `roi_box_reg_smooth_l1_aux_weight` (type from `box_reg_loss_type`, default `l1` in DOTA recipes). Decoded regression randomly subsamples at most `roi_batch_size_per_image` (default **512**) **positive anchors per image across all FPN levels** (not per level). `loss_box_reg` normalizes by the number of sampled positives used in regression. Classification still uses every matched anchor. When main is `smooth_l1` (default), optional decoded aux uses `roi_box_reg_iou_weight`.

### `roi_inference_top_class_only` (two-stage models)

`RotatedFasterRCNN` and `OrientedRCNN` use **`model.roi_inference_top_class_only`** only in **eval / inference** (after the ROI head), not during training loss:

- **`false` (default):** keep every foreground class whose softmax probability is above `inference_pre_nms_score_threshold` for each proposal. This recovers more candidates when the classifier is still weak.
- **`true`:** take the **argmax** foreground class per proposal, then apply the same threshold to that score (MMRotate-style one detection hypothesis per RoI).

**Recommended:** leave **`false` during early training** (validation / snapshots where the head is immature). Switch to **`true` for late fine-tuning and final inference** when you want MMRotate-like behavior, fewer duplicate class hypotheses per RoI, and scores comparable to single-label decoding.

`RotatedFasterRCNN` applies **no objectness score threshold** when generating RPN proposals (training or inference): the RPN keeps top-k proposals by score only, like MMRotate. `model.inference_pre_nms_score_threshold` applies exclusively to **ROI-head** class scores before final rotated NMS (MMRotate `test_cfg.rcnn.score_thr`).

### MMRotate parity notes (two-stage detectors)

- **FPN levels:** the RPN runs on all 5 levels (P2–P6, strides 4–64; `include_pool_level=True` keeps torchvision's stride-64 max-pool level). **Both** two-stage detectors restrict ROI extraction to the **first 4 FPN levels** (strides 4–32): `horizontal_roi_align` (Rotated Faster R-CNN) and `oriented_roi_align` (Oriented R-CNN), matching MMRotate `SingleRoIExtractor` / `RotatedSingleRoIExtractor`.
- **RoIAlign:** `horizontal_roi_align` uses `aligned=True` (half-pixel aligned), matching mmcv's `RoIAlign` default.
- **Backbone BN:** frozen statistics (`FrozenBatchNorm2d`) by default, matching MMRotate `norm_eval=True`. See `backbones/README.md`.
- **Loss normalization:** RPN and ROI SmoothL1 box-regression losses are summed over positives and divided by the **total** number of sampled anchors/RoIs (MMDet `avg_factor`), including Oriented R-CNN midpoint RPN and oriented ROI stages.
- **Assignment IoU:** RPN stages use HBB IoU when `use_hbb_for_matching: true` (MMRotate horizontal RPN). Oriented R-CNN ROI matching uses rotated IoU by default (`roi_use_hbb_for_matching: false`). RetinaNet uses rotated IoU (`use_hbb_for_matching: false`).

RPN proposal pruning uses horizontal xyxy proposals and `torchvision.ops.nms` on GPU.
The RPN proposal geometry is horizontal; rotated geometry is introduced by the ROI
regression head.

This is intentional for MMRotate-style Rotated Faster R-CNN behavior. Final ROI
classification/regression predicts oriented boxes and final detections use the
rotated backend abstraction.

Rotated final NMS and rotated assignment should route through
`oriented_det.ops.rotated_ops`. The default backend is this repo's parallel GPU
sampling implementation (`ORIENTED_DET_ROTATED_BACKEND=gpu_sample`); CPU is for
debug/reference checks. We do not rely on MMCV. If profiling shows a large win,
add in-repo CUDA kernels behind the same abstraction after the first release.

## Rotated RetinaNet classification (sigmoid focal loss)

`RotatedRetinaNet` follows MMRotate's `FocalLoss(use_sigmoid=True)` exactly:

- The head outputs **`num_anchors * num_classes`** classification channels (K independent binary classifiers per anchor, **no background channel**). Bias init is `-log((1-π)/π)` with π=0.01 so every class starts at sigmoid ≈ 0.01.
- Training uses **sigmoid focal loss** (`sigmoid_focal_loss_sum`): one-hot binary targets per anchor, `alpha` (default 0.25) weighting positive entries and `1-alpha` weighting negatives, summed over all anchors/levels and normalized by the **total number of positive anchors** in the batch (MMDet `avg_factor`).
- Inference scores are **`sigmoid(logits)`** per class; the best class per anchor is kept (labels stay 1-indexed downstream).

This replaced an earlier softmax background+K formulation whose background-bias init plus uniform-alpha focal loss starved the classification head of gradients (cls grad norm ~1000x smaller than bbox), producing 0 detections after 12 epochs.

## Rotated FCOS (anchor-free)

`RotatedFCOS` (`model_type: rotated_fcos`) follows MMRotate’s Rotated FCOS architecture:

- **Coder:** `DistanceAnglePointCoder` — point → `(left, top, right, bottom, angle)` (channel order matches MMRotate code).
- **Assigner:** center-in-OBB + per-level `regress_ranges` + optional center sampling (radius 1.5×stride).
- **Head:** GN+ReLU cls/reg towers, `conv_cls` / `conv_bbox`(4) / `conv_angle` / `conv_centerness`; `norm_on_bbox`, `centerness_on_reg`, `scale_angle`.
- **Losses:** sigmoid focal (cls), centerness BCE, and box reg via **`box_reg_loss_type`**:
  - **`l1`** (default / L1 recipes): centerness-weighted L1 on encoded ltrb + le90-wrapped angle L1.
  - **`kfiou`**: decode to absolute OBBs (×stride when `norm_on_bbox`), then centerness-weighted [`kfiou_loss_per_box`](../ops/kfiou.py).
  - **`riou`**: same decode path, then centerness-weighted [`riou_loss_per_box`](../ops/diff_iou_rotated.py) (`1 -` differentiable polygon IoU). Not sampling `pairwise_rotated_iou`.
- **Aux:** **`aux_loss_type`** (`kfiou` / `probiou`) + **`aux_loss_weight`** (0 disables). Decoded, centerness-weighted; logged as `loss_box_reg_aux`. Gaussian overlap plus an aspect-gated heading term (`aux_angle_weight`, default 1.0; `aux_angle_lambda` default 1.0). Prefer L1 primary + aux 0.1. Aux `riou` is rejected.
- **Inference:** `sigmoid(cls) * sigmoid(centerness)` → decode → class-aware rotated NMS. Keep **`production.final_nms_iou_threshold: 0.1`** aligned with `model.*`; 0.3 leaves dense duplicates at eval-val.

Configs: `configs/rotated_fcos/dota_le90_{1x,3x}.json` (L1), `dota_le90_{1x,3x}_kfiou_aux.json` (L1 + KFIoU aux), `dota_le90_{1x,3x}_riou.json` (decoded rIoU primary, lr 2.5e-3). Results: [`configs/rotated_fcos/README.md`](../../configs/rotated_fcos/README.md).

**1× L1 recipe:** `learning_rate=2.5e-4`, `max_detections_per_image=2000`, NMS IoU **0.1**. L1 **3×** eval-val **73.92%** (local baseline). **Hub:** 3× decoded rIoU **81.58%** (`rotated_fcos_dota_le90_3x_riou`, `runs/rotated_fcos/20260822-153943`); report [`docs/eval-reports/rotated_fcos_dota_le90_3x_riou/`](../../docs/eval-reports/rotated_fcos_dota_le90_3x_riou/model_analysis.md). 3× L1 + KFIoU aux **77.18%** remains as `rotated_fcos_dota_le90_3x_kfiou_aux`.

**Checkpoint break (v0.1.1):** RetinaNet now uses MMRotate-style **separate cls/reg 4-conv towers** with **3×3 prediction heads** and **P6/P7 convs on C5** (`LastLevelP6P7`). Pre-change checkpoints (`head.convs`, 1×1 `conv_cls`/`conv_bbox`, `extra_fpn_conv`) are incompatible.

### Rotated RetinaNet MMRotate alignment

- **Head:** independent `cls_convs` / `reg_convs` (default 4×3×3 each) + 3×3 `conv_cls` / `conv_bbox` (MMRotate `RetinaHead`).
- **FPN P6/P7:** `fpn_extra_level: true` attaches torchvision `LastLevelP6P7` on C5 (`add_extra_convs='on_input'`), not max-pool P6 + manual P7 conv.
- **5 FPN levels (P3–P7)** with strides `[8, 16, 32, 64, 128]` when `fpn_returned_layers: [2,3,4]`.
- **`min_pos_iou=0`** in anchor assignment (MMRotate `MaxIoUAssigner`).
- **Regression loss:** encoded L1/SmoothL1 summed over positives, normalized by batch positive count (MMDet `avg_factor`).
- **Rotated IoU assignment** (`use_hbb_for_matching: false`).
- **le90 angle wrap in `edge_swap` encoding** (`norm_angle_le90` in `encode_oriented_boxes`).

### `final_nms_use_cpu` (exact final NMS)

Set **`model.final_nms_use_cpu`** to **`true`** in JSON / `ModelConfig` so **post-head final oriented NMS only** uses the **polygon IoU Python path on CPU** (`rotated_nms(..., force_cpu=True)` for two-stage models; RetinaNet skips its GPU NMS branch). RPN NMS, anchor/ROI matching, and `ORIENTED_DET_ROTATED_BACKEND` elsewhere are unchanged—so training stays fast; validation/inference final dedup is slower but matches exact greedy NMS on true rotated IoU.

RPN anchor assignment also uses HBB overlap when `use_hbb_for_matching` is true. That path computes HBB IoU in large chunks and keeps only the best GT per anchor and best anchor per GT. This avoids thousands of tiny GPU launches and avoids materializing a full `anchors x GT` matrix for P2, where a single image can have millions of anchors.

First-batch timing probes are gated by code-only debug flags and are disabled by
default: `TRACE_FIRST_TRAIN_FORWARD_TIMING` in `oriented_rcnn.py` and
`TRACE_RPN_LOSS_TIMING` in `oriented_rpn.py`.
