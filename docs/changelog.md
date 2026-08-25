# Changelog

All notable changes to OrientedDet will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-08-25

### Added

- **Rotated FCOS** (`model_type: rotated_fcos`) — anchor-free single-stage detector: DistanceAnglePointCoder, center-in-OBB assigner, centerness, and L1 / KFIoU / decoded rIoU box regression. Recipes under [`configs/rotated_fcos/`](../configs/rotated_fcos/).
- Differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`) for FCOS **`box_reg_loss_type: riou`** (`1 - IoU`). Distinct from sampling `pairwise_rotated_iou`. Recipes [`dota_le90_1x_riou.json`](../configs/rotated_fcos/dota_le90_1x_riou.json) / [`dota_le90_3x_riou.json`](../configs/rotated_fcos/dota_le90_3x_riou.json) (lr 2.5e-3).
- Hub slug **`rotated_fcos_dota_le90_3x_riou`** (81.58% eval-val mAP50) — Rotated FCOS 3× decoded rIoU; report [`docs/eval-reports/rotated_fcos_dota_le90_3x_riou/`](eval-reports/rotated_fcos_dota_le90_3x_riou/model_analysis.md).
- Hub slug **`rotated_fcos_dota_le90_3x_kfiou_aux`** (77.18% eval-val mAP50) — Rotated FCOS 3× L1 + KFIoU aux; report [`docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/`](eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md).
- TF/ONNX export mode **`rotated_fcos_pre_nms`** — Rotated FCOS decode + pad uses the same Keras detect bundle as two-stage models (`odet export-tf --mode rotated_fcos_pre_nms`).

### Removed

- FCOS 1× ProbIoU-aux recipe (`dota_le90_1x_probiou_aux.json`). Train-time mAP50 66.8% vs 76.5% for 1× KFIoU aux on the same protocol.

## [0.1.1] - 2026-07-11

### Added

- **ProbIoU ROI regression** for Rotated Faster R-CNN (`roi_box_reg_main_loss_type: probiou` + Smooth L1 aux).
- Hub slugs **`rotated_faster_rcnn_dota_le90_3x`** (83.42% eval-val mAP50) and **`rotated_faster_rcnn_dota_le90_1x`** (77.57% eval-val mAP50).
- `dataset.train_includes_val` config flag (Airbus Playground: train on all folds; val fold for monitoring only).
- Source provenance metadata in training runs (`git_commit`, package version, config hash).
- Eval reports under `docs/eval-reports/`; `make eval-val` full-tile protocol documented.

### Changed (MMRotate parity)

- ROI regression loss: encoded-space Smooth L1 on all 5 channels (MMRotate), replacing radian periodic angle loss that under-weighted angle gradients vs MMRotate.
- Oriented R-CNN: MMDet `avg_factor` for midpoint RPN and oriented ROI losses; training RPN proposals no longer score-filtered; ROI matching defaults to rotated IoU (`roi_use_hbb_for_matching: false`); oriented RoIAlign uses first 4 FPN levels only.
- Rotated RetinaNet: separate cls/reg 4-conv towers with 3×3 prediction heads; P6/P7 via `LastLevelP6P7` on C5; rotated IoU assignment; encoded L1 reg loss with `avg_factor` normalization.

### Breaking

- **RetinaNet checkpoints** from before this release are incompatible (`head.convs` / 1×1 heads / `extra_fpn_conv` removed). Re-train or use Hub weights published after this change.

## [0.1.0] - 2026-05-27

### Added

- Core geometry (Polygon, QBox, RBox) and transforms
- Rotated IoU, NMS, and optional GPU kernels
- DOTA loader, tiling, augmentations, oriented mAP
- Airbus Playground CSV dataset support
- Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet
- JSON config training via `odet train`
- Pretrained weights on Hugging Face Hub (`dl4eo/oriented-det-pretrained`), including Oriented R-CNN 1× DOTA le90 (`74.79%` eval-val mAP50) and Oriented R-CNN 3× DOTA le90 (`79.40%` eval-val mAP50)
- MkDocs user guide and API reference

### Notes

- Public home: https://github.com/DL4EO/oriented-det
