# Changelog

All notable changes to OrientedDet will be documented in this file.

## [Unreleased]

### Added

- Optional TensorFlow **SavedModel** export (`odet export-tf --saved-model` / `odet export-savedmodel`): onnx2tf pre-NMS core + TF rotated NMS. Reload with `tf.saved_model.load` (no oriented-det at inference). Conversion can fail on ScatterND; the Keras detect bundle remains the supported full-detect path.
- **Random rotate** (train only) — `preprocessing.enable_random_rotate` / `random_rotate_prob` / `random_rotate_angle_range` (degrees). MMRotate `PolyRandomRotate` after flips (`auto_bound=False`). FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN HRSC 3× use p=0.5 **±20°**. Oriented R-CNN / Faster R-CNN 1× and DOTA stay off. ±180° on FCOS 6× diverged after epoch 12.
- **HRSC2016** dataset loader (`dataset.format: hrsc2016`) — official XML + ImageSets, single-class `ship`, le90 via the DOTA polygon path. Recipes: [`configs/oriented_rcnn/hrsc2016_le90_1x.json`](../configs/oriented_rcnn/hrsc2016_le90_1x.json), [`configs/oriented_rcnn/hrsc2016_le90_3x.json`](../configs/oriented_rcnn/hrsc2016_le90_3x.json), [`configs/rotated_faster_rcnn/hrsc2016_le90_1x.json`](../configs/rotated_faster_rcnn/hrsc2016_le90_1x.json), [`configs/rotated_faster_rcnn/hrsc2016_le90_3x.json`](../configs/rotated_faster_rcnn/hrsc2016_le90_3x.json), [`configs/rotated_fcos/hrsc2016_le90_1x.json`](../configs/rotated_fcos/hrsc2016_le90_1x.json), [`configs/rotated_fcos/hrsc2016_le90_3x.json`](../configs/rotated_fcos/hrsc2016_le90_3x.json). Optional DOTA export: `odet hrsc-to-dota`.
- **`resize_mode: keep_ratio`** — MMRotate-style long-edge scale without square pad; training/inference then apply `pad_size_divisor` (bottom-right). HRSC Oriented R-CNN, Faster R-CNN, and FCOS 1×/3× use this canvas. Oriented R-CNN uses Smooth L1 main + ProbIoU aux 0.1. Faster R-CNN keeps DOTA ProbIoU main + Smooth L1 aux 0.1. HRSC FCOS recipes use decoded **rIoU** (lr **2.5e-3**), not L1. FCOS HRSC 3× uses `lr_scheduler_gamma: [0.1, 0.5]`.
- **`training.lr_scheduler_gamma`** accepts a number (same factor every drop) or a list (one factor per `lr_scheduler_milestones` entry).
- Hub slug **`oriented_rcnn_hrsc2016_le90_3x`** (90.41% eval-val mAP50) — Oriented R-CNN 3× on HRSC2016 test (rotate ±20°, gamma 0.1); report [`docs/eval-reports/oriented_rcnn_hrsc2016_le90_3x/`](eval-reports/oriented_rcnn_hrsc2016_le90_3x/model_analysis.md).
- Hub slug **`rotated_faster_rcnn_hrsc2016_le90_3x`** (88.77% eval-val mAP50) — Faster R-CNN 3× on HRSC2016 test; report [`docs/eval-reports/rotated_faster_rcnn_hrsc2016_le90_3x/`](eval-reports/rotated_faster_rcnn_hrsc2016_le90_3x/model_analysis.md).
- Hub slug **`rotated_fcos_hrsc2016_le90_3x`** (88.34% eval-val mAP50) — FCOS 3× decoded rIoU on HRSC2016 test; report [`docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/`](eval-reports/rotated_fcos_hrsc2016_le90_3x/model_analysis.md).
- **`dataset.drop_easy_empty_tiles`** — with `tile_metrics_csv`, drop train tiles that are vacuous true negatives (`tp=fp=fn=0`) before `max_train_samples` and hard-tile oversampling. Empty tiles with false positives stay and can be oversampled. Keep `filter_empty_gt: false` so those hard empties remain in the loader.

### Fixed

- **Diagonal flip angle** — MMRotate `RRandomFlip(direction='diagonal')` mirrors box centers and **keeps θ** (early return). We incorrectly applied `π − θ`, so diagonal samples (and diagonal + random rotate) had boxes at the wrong orientation. Horizontal / vertical flips were already correct.
- **`keep_ratio` + `pad_size_divisor` 32** no longer crashes on P6: torchvision `max_pool` on an odd feature map (e.g. 576×800 → 9×13) is anisotropic; stride derivation now falls back to configured `[4, 8, 16, 32, 64]` (MMRotate).
- **`keep_ratio` collate** bottom-right pads a batch to a shared H×W (e.g. 480×800 + 576×800) so `torch.stack` works. Per-image `content_size` is unchanged; `odet preds` stays one image + divisor pad.

### Removed

- HRSC2016 **6×** recipes (`oriented_rcnn/hrsc2016_le90_6x.json`, `rotated_fcos/hrsc2016_le90_6x.json`). 3× ±20° is the long schedule; 6× was only +0.2 mAP on Oriented R-CNN and FCOS 6× never beat that 3×.
- Notebook geometric transform classes (`Rotate`, `HorizontalFlip`, `VerticalFlip`, `DiagonalFlip`, `Compose`, `OrientedTransform`). Image+box augs are only `apply_random_train_flips` / `apply_random_train_rotate` and `apply_flip_to_*` / `apply_rotate_to_*`.

### Changed

- **NMS split (DOTA + HRSC)** — `model.final_nms_iou_threshold: 0.1` (train val), `production.final_nms_iou_threshold: 0.3` (deploy / `image_demo`), and new **`evaluation.final_nms_iou_threshold: 0.1`** for `odet preds` / `make eval-val` (MMRotate test parity). Resolver: `resolve_preds_final_nms_iou_threshold`.
- Rotated FCOS DOTA **`dota_le90_3x.json`** is now the decoded rIoU 3× recipe (was `dota_le90_3x_riou.json`). The previous L1 3× is [`dota_le90_3x_l1.json`](../configs/rotated_fcos/dota_le90_3x_l1.json). Hub slug **`rotated_fcos_dota_le90_3x`** (was `rotated_fcos_dota_le90_3x_riou`).

- **ROI box-reg aux keys** — `roi_box_reg_iou_weight` / `roi_box_reg_iou_loss_type` / `roi_box_reg_smooth_l1_aux_weight` are now `roi_box_reg_aux_weight` + `roi_box_reg_aux_loss_type` (`smooth_l1` \| `probiou` \| `riou` \| `kfiou`). Schedule fields are `roi_box_reg_aux_schedule_*`. Weights are unchanged (no retrain). Old keys still load with a deprecation warning. Hub slugs are unchanged.
- **`training.lr_scheduler_cosine_t_max`** remaps to **`lr_scheduler_cosine_epochs`** (same integer; deprecation warning). Mixing both with different values is an error.

- **HRSC2016 eval** (`resize_mode: pad` / `keep_ratio`) uses the same whole-image scale forward as training. DOTA `fixed`/`crop` eval-val still native-tiles oversized rasters.
- **HRSC two-stage** `max_detections_per_image` **2000** (was 100). Final NMS for published eval-val stays **0.1** via `evaluation.final_nms_iou_threshold`; recipes now ship **production NMS 0.3** like DOTA.

## [0.2.0] - 2026-08-25

### Added

- **Rotated FCOS** (`model_type: rotated_fcos`) — anchor-free single-stage detector: DistanceAnglePointCoder, center-in-OBB assigner, centerness, and L1 / KFIoU / decoded rIoU box regression. Recipes under [`configs/rotated_fcos/`](../configs/rotated_fcos/).
- Differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`) for FCOS **`box_reg_loss_type: riou`** (`1 - IoU`). Distinct from sampling `pairwise_rotated_iou`. Recipes [`dota_le90_1x_riou.json`](../configs/rotated_fcos/dota_le90_1x_riou.json) / [`dota_le90_3x.json`](../configs/rotated_fcos/dota_le90_3x.json) (lr 2.5e-3).
- Hub slug **`rotated_fcos_dota_le90_3x`** (81.58% eval-val mAP50) — Rotated FCOS 3× decoded rIoU; report [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md).
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
