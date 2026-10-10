# Changelog

All notable changes to OrientedDet will be documented in this file.

## [Unreleased]

Toward **v0.4.1** (native YOLO-OBB). The detector is already in the tree (`model_type: yolo_obb`). Next is COCO pretrain and DOTA. Not part of the 0.4.0 notes.

## [0.4.0] - 2026-10-10

Rotated RTMDet. `rotated_rtmdet_dota_le90_3x` (CSPNeXt-S + PAFPN) is on the Hub at official Task 1 **77.60%**, the top of the DOTA zoo. Native YOLO-OBB is **0.4.1**.

### Added

- **Rotated RTMDet** (`model_type: rotated_rtmdet`) — clean-room CSPNeXt backbone + CSPNeXt-PAFPN neck, SepBN head (shared convs, per-level BN), dynamic soft-label assigner, Quality Focal Loss + decoded rotated IoU box loss. Base fragments: [`rotated_rtmdet_cspnext_s.json`](../configs/_base_/models/rotated_rtmdet_cspnext_s.json) and [`rotated_rtmdet_cspnext_tiny.json`](../configs/_base_/models/rotated_rtmdet_cspnext_tiny.json) (tiny has no Hub weights).
- **RTMDet-S 3× Hub** — `rotated_rtmdet_dota_le90_3x`. Official Task 1 **77.60%** (AP75 52.01, COCO mAP 49.02; deploy **0.45**; leaky eval-val 82.75%). Recipe: [`dota_le90_3x_s.json`](../configs/rotated_rtmdet/dota_le90_3x_s.json) (36 epochs, AdamW, EMA, random rotate p=0.5 ±180°, mosaic off). Report: [`docs/eval-reports/rotated_rtmdet_dota_le90_3x/`](eval-reports/rotated_rtmdet_dota_le90_3x/model_analysis.md).
- **EMA** — `training.use_ema` / `ema_momentum` / `ema_gamma`. Validation and `best_mAP_*.pth` use the EMA weights; epoch checkpoints keep raw weights plus `ema_state_dict` for resume.
- **AdamW and weight-decay groups** — `training.optimizer: adamw` (`adamw_betas`), `weight_decay_norm_mult` / `weight_decay_bias_mult`. `max_grad_norm: null` disables clipping. Defaults (SGD, multipliers 1, clip 1.0) are unchanged.
- **`lr_scheduler_type: cosine_annealing_late`** — flat LR after warmup until `lr_scheduler_cosine_start_epoch`, then cosine to `lr_scheduler_cosine_eta_min`.
- **`preprocessing.random_rotate_rect_labels`** — tiles containing those classes rotate by 90° steps only (RTMDet uses `storage-tank` and `roundabout`).
- **Cached mosaic** (`preprocessing.enable_mosaic` and `mosaic_*` keys), off by default and off in every published recipe.
- `tools/bench_latency.py` — single-tile forward latency (backbone + head + decode + NMS) for several detectors on the same canvas and device.
- **OSSDD FCOS and Faster R-CNN 3×** — [`ossdd_le90_vh_3x.json`](../configs/rotated_fcos/ossdd_le90_vh_3x.json) and [`ossdd_le90_vh_3x.json`](../configs/rotated_faster_rcnn/ossdd_le90_vh_3x.json) inherit VH 1× finetunes of the DOTA 1× hubs (36 epochs, MultiStep milestones 24/33, 512 crop, batch 4).
- **OSSDD mAP** — `evaluation.map_iou_thresholds` averages the area under the precision-recall curve over rotated IoU 0.10 to 0.95 step 0.05. Set on the OSSDD recipes only. `best_metric` mAP uses that average. `iou_threshold` stays 0.5 for GT-cover and the precision-recall operating point.
- **RTMDet ONNX** — `odet export onnx --mode rotated_rtmdet_pre_nms`. CSPNeXt + SepBN head + box decode in the graph; rotated NMS stays in Python. Fixed H×W, batch 1, same pre-NMS tensors as FCOS.

### Changed

- **Evaluation split into `val_*` and `test_*`.** Periodic val and `make eval-val` use `val_score_threshold` (often 0.3). `test_map` and `make eval-test` use `test_score_threshold` (null → **0.05**). The 1× schedule sets `test_map: true` and `val_map_at_end: false`. DOTA and FAIR1M point `test_tiles_dir` at the val tiles (leaky). `dota-submit` still requires the unlabeled official test and does not read `test_tiles_dir`. Older keys (`train_val_score_threshold`, `preds_score_threshold`, `compute_map_final`, `compute_map_test`, …) remap on load. Published Hub sidecars in `pretrained/` use the new names.
- **`odet preds` / `make metrics` metrics margin** defaults to **0** (every GT and detection is scored). It no longer reads `production.ignore_margin_pixels`, tile overlap, or a margin stored in `predictions.json`. Pass `--metrics-margin-pixels` for a positive band. Deploy / `image_demo` still use `production.ignore_margin_pixels`.
- Tutorial notebooks write to **`work_dirs/`** in a checkout (was `tutorial_work/`). `ORIENTED_DET_TUTORIAL_WORK` and the Kaggle `/kaggle/working` default are unchanged.

ONNX export covers Rotated FCOS, Rotated RTMDet, Oriented R-CNN, and Rotated Faster R-CNN.

## [0.3.1] - 2026-09-21

Patch: Rotated RetinaNet Hub default is **OBB** (no suffix). Circum-HBB lives at `*_hbb`. Official Task 1 **71.72%** (1×) / **73.89%** (3×).

### Added

- **Rotated RetinaNet 1× OBB Hub** — `rotated_retinanet_dota_le90_1x` is now exact convex IoU assign (`use_hbb_for_matching: false`). Official Task 1 **71.72%** (AP75 43.46, COCO mAP 42.18; deploy **0.25**; leaky eval-val 71.12%; ahead of MMRotate OBB 68.42%). Recipe: [`dota_le90_1x.json`](../configs/rotated_retinanet/dota_le90_1x.json).
- **Rotated RetinaNet 3× OBB Hub** — `rotated_retinanet_dota_le90_3x` inherits 1× OBB (36 epochs). Official Task 1 **73.89%** (AP75 47.11, COCO mAP 44.63; deploy **0.25**; leaky eval-val 78.56%). Recipe: [`dota_le90_3x.json`](../configs/rotated_retinanet/dota_le90_3x.json).
- **HBB Hub slugs** — previous circum-HBB weights: `rotated_retinanet_dota_le90_1x_hbb` (**67.87%** Task 1) and `rotated_retinanet_dota_le90_3x_hbb` (**70.70%** Task 1).
- **`odet export`** — ONNX-only (`onnx`, `infer`, `demo`, `preds`). Same CLI as `python -m export`. TensorFlow / SavedModel stay removed.

### Changed

- Un-suffixed RetinaNet 1×/3× recipes are OBB; HBB is `dota_le90_*_hbb.json`. v0.3.0 `hf://rotated_retinanet_dota_le90_1x` (HBB) and `…_3x` (HBB) map to the `_hbb` slugs in 0.3.1. Old Hub filenames stay for 0.3.0 clients.
- RetinaNet OBB deploy floor is **0.25** (eval-val F1 0.30 − 0.05). HBB stays **0.35**.

### Fixed

- Deploy `generate_description.py` treats a DOTA class list as **v1 full** by set equality (Hub sidecars are alphabetical, not `DOTA_V1_CLASSES` order).
- RetinaNet convex-IoU hull padding no longer fills unused shoelace slots with an exterior dummy vertex (that clamped some IoUs to 1.0). Unused slots repeat the first hull vertex.

## [0.3.0] - 2026-09-19

Dataset release: native **HRSC2016**, **FAIR1M**, **SSDD**, and **HRSID** loaders; HRSC Hub 3× zoo; DOTA zoo refreshed to official Task 1 numbers (advertise / finetune from **1×**). Restores **ONNX-only** export (`python -m export`) after dropping the 0.2 TensorFlow stack.

### Added

- **HRSC2016** (`dataset.format: hrsc2016`) — official XML + ImageSets, single-class `ship`, le90 via the DOTA polygon path. Whole-image `keep_ratio` + pad-32. Optional `odet hrsc-to-dota`. Recipes: Oriented R-CNN / Faster R-CNN / FCOS 1× and 3× under `configs/*/hrsc2016_le90_*.json`. Hub 3× held-out ImageSets test: **`oriented_rcnn_hrsc2016_le90_3x`** 90.41%, **`rotated_faster_rcnn_hrsc2016_le90_3x`** 88.77%, **`rotated_fcos_hrsc2016_le90_3x`** 88.34%. FCOS HRSC 1×/3× and two-stage 3× use random rotate p=0.5 **±20°**.
- **FAIR1M** (`dataset.format: fair1m`) — 37-class XML loader, official + Kaggle layouts, `odet fair1m-to-dota` then tile 1024/200. 1× recipes finetune the matching **DOTA 1× Hub** checkpoint (37-way classifier re-init; **no FAIR1M zoo**). Local Faster R-CNN 1× tiled-val mAP50 **36.70%**. Notebook: [`notebooks/kaggle_fair1m_tutorial.ipynb`](../notebooks/kaggle_fair1m_tutorial.ipynb). See [data.md](user-guide/data.md#fair1m).
- **SSDD** (`dataset.format: ssdd`) — SAR ship chips, official last-digit train/test, VOC XML / COCO / DOTA, keep-ratio **608**. 1× Faster R-CNN finetunes `hf://rotated_faster_rcnn_dota_le90_1x` (**no SSDD zoo**). Local held-out test mAP50 **90.34%**. Optional `odet ssdd-to-dota`. Notebook: [`notebooks/ssdd_finetune_tutorial.ipynb`](../notebooks/ssdd_finetune_tutorial.ipynb).
- **HRSID** (`dataset.format: hrsid`) — COCO polygons → le90 rbox (n-gons via min-area rect), official train/test (no val), keep-ratio **800**. 1× recipes finetune matching DOTA 1× Hub (**no HRSID zoo**). Local Faster R-CNN 1× held-out test mAP50 **78.55%** (Wei HBB AP50 **>84.7%**, not a drop-in match). Generic `odet coco-to-dota`.
- **DOTA Hub (official v1.0 Task 1)** — advertised / finetune init is **1×**. 3× stays on Hub for AP75. Deploy `production.score_threshold` is eval-val global F1 − **0.05**. Reports under [`docs/eval-reports/`](eval-reports/).
  - **`oriented_rcnn_dota_le90_1x`** 76.73% (AP75 50.24, COCO mAP 46.59; deploy **0.55**; leaky eval-val 77.66%)
  - **`oriented_rcnn_dota_le90_3x`** 74.88% (AP75 51.23, COCO mAP 46.91; deploy **0.55**; leaky eval-val 82.92%) — September retrain with the diagonal-flip θ fix
  - **`rotated_faster_rcnn_dota_le90_1x`** 74.42% (AP75 41.90, COCO mAP 42.74; deploy **0.6**; leaky eval-val 77.55%)
  - **`rotated_faster_rcnn_dota_le90_3x`** 74.48% (AP75 45.39, COCO mAP 43.94; deploy **0.6**; leaky eval-val 83.46%)
  - **`rotated_fcos_dota_le90_1x`** 73.07% (AP75 40.40, COCO mAP 41.56; deploy **0.2**; leaky eval-val 75.13%)
  - **`rotated_fcos_dota_le90_3x`** 72.91% (AP75 45.39, COCO mAP 44.18; deploy **0.2**; leaky eval-val 82.32%)
  - **`rotated_retinanet_dota_le90_1x`** 67.87% circum-HBB (AP75 40.08, COCO mAP 38.91; deploy **0.35**; leaky eval-val 68.20%; ahead of MMRotate HBB 64.55%)
  - **`rotated_retinanet_dota_le90_3x`** 70.70% circum-HBB (AP75 43.34, COCO mAP 41.59; deploy **0.35**; leaky eval-val 76.51%)
- **DOTA Task 1 submit** — `odet dota-submit` / `make dota-submit` writes `Task1_*.txt` + zip from unlabeled official test (`--checkpoint hf://<slug>`, `--experiment-dir`, or `--from-json`). Inference NMS is **0.1**. Test labels stay private; upload to the [DOTA v1.0 Task 1 server](https://captain-whu.github.io/DOTA/evaluation.html).
- **`resize_mode: keep_ratio`** — MMRotate-style long-edge scale without square pad; then `pad_size_divisor` (bottom-right). HRSC Oriented R-CNN / Faster R-CNN / FCOS 1×/3× use this canvas.
- **Random rotate** (train only) — `preprocessing.enable_random_rotate` / `random_rotate_prob` / `random_rotate_angle_range`. MMRotate `PolyRandomRotate` after flips. RetinaNet 1× RR ablation: [`dota_le90_1x_rr.json`](../configs/rotated_retinanet/dota_le90_1x_rr.json) (p=0.5 **±180°**; not Hub).
- **`training.lr_scheduler_gamma`** accepts a number or a list (one factor per milestone).
- **`evaluation.train_val_score_threshold: 0.3`** on all leaf recipes (in-train mAP / `best_mAP`; dataclass/schema default). `preds_score_threshold` / `production.score_threshold` unchanged.
- **`dataset.drop_easy_empty_tiles`** and **`dataset.class_tile_oversample_classes`** for train-tile sampling.
- **Class-agnostic final NMS** on FCOS / RetinaNet via `model.nms_class_agnostic` / `production.nms_class_agnostic` (default `false`). DOTA / HRSC recipes stay class-aware.
- **`model.anchor_angles`** (degrees) for Rotated RetinaNet; Hub 1×/3× stay `[0]`.
- **Rotated RetinaNet 1× OBB** — [`dota_le90_1x_obb.json`](../configs/rotated_retinanet/dota_le90_1x_obb.json) sets `use_hbb_for_matching: false` (not Hub).
- **ONNX export restored** (`export/`, `python -m export` / `make export-onnx`) — pre-NMS ONNX plus Python preprocess / rotated NMS for **Rotated FCOS**, **Oriented R-CNN**, and **Rotated Faster R-CNN**. Supported canvas is **fixed H×W** (DOTA **1024** tiles). Consumers need only numpy, Pillow, and ONNX Runtime (no PyTorch, no oriented-det). Producer extra: `uv pip install -e ".[export]"`. Not an `odet` subcommand. **Out of this graph:** `keep_ratio` (HRSC / SSDD / HRSID), sliding-window tiling, RetinaNet detect (heads-only subgraph only). TensorFlow / SavedModel / TFLite are not brought back.

### Changed

- **`evaluation.train_val_score_threshold`** renames the former `evaluation.score_threshold`. Legacy `score_threshold` still loads with a deprecation warning. In-train mAP uses **`evaluation` only** (`production.score_threshold` stays deploy / `image_demo`). `make eval-val` / `odet preds` use CLI → `evaluation.preds_score_threshold` → **0.05**.
- DOTA / HRSC recipes and Hub sidecars set **`production.final_nms_iou_threshold: 0.1`** (was 0.3 / 0.5 on some sidecars) so deploy / `image_demo` match train val and eval-val NMS. Weights are unchanged. HRSC two-stage `max_detections_per_image` is **2000**.
- Deploy floors are eval-val global F1 − **0.05**: DOTA Oriented R-CNN **0.55**, Faster R-CNN **0.6**, RetinaNet **0.35**, FCOS **0.2**; HRSC Oriented R-CNN / Faster R-CNN **0.85**, FCOS **0.2**. 3× inherits the 1× floor. `make eval-val` still uses **0.05**.
- **DOTA 3× recipes** are 1× + 36 epochs (milestones `[24, 33]`) only. **Finetune from 1×**, not 3× — 3× leaky eval-val is train+val tile memorization; official AP50 is a wash or a drop; AP75 is the 3× gain.
- **Rotated RetinaNet DOTA Hub** pins `use_hbb_for_matching: true` (circum-HBB). `_base_/models/rotated_retinanet_r50.json` remains `false` (MMRotate OBB default). MaxIoU concatenates P3–P7 then splits by level; assignment is RetinaNet-only (`retinanet_assign.py`).
- Manifest **`eval_map50`** is leaky eval-val for DOTA Task 1 slugs; official scores live in **`eval_task1_map50`**. **`make eval-val` is leaky on DOTA** and **held-out on HRSC / FAIR1M / SSDD / HRSID**. Canonical table: [Train / eval splits](user-guide/data.md#train--eval-splits).
- DOTA FCOS / RetinaNet 1× inherit [`dota_le90.json`](../configs/_base_/datasets/dota_le90.json) mean/std (were ImageNet). FCOS DOTA **`dota_le90_3x.json`** is the decoded rIoU 3× recipe (was `dota_le90_3x_riou.json`).
- **Sliding-window last tiles flush to the image edge** (`tile_dota.py` parity). Per-window centroid margin default is **0**. Micro-batch default is **8** (GPU) / **4** (CPU); unset `ORIENTED_DET_WINDOW_BATCH_SIZE` no longer VRAM-probes. Set `auto` to restore probing.
- **`odet tile-dota`** reads JPEG/TIFF/BMP as well as PNG (`--output-format auto`). **`odet fair1m-to-dota`** default `--image-format original` copies JPEG/PNG.
- Minimum PyTorch 2.4 / torchvision 0.19; NumPy 2 is supported. Docs, convert-tool examples, Gradio viewer default, and published Hub sidecar `source_code_root` use `/path/to/data` and `/path/to/oriented-det`.
- **`loss.focal_weighted`** scales FCOS / RetinaNet sigmoid focal per class. ROI box-reg aux keys rename to `roi_box_reg_aux_weight` + `roi_box_reg_aux_loss_type`. **`training.lr_scheduler_cosine_t_max`** remaps to **`lr_scheduler_cosine_epochs`**. Old keys still load with a deprecation warning.
- **`make wizard` / `--wizard`** FCOS FPN nudges are class-agnostic (pooled GT width vs finest stride). HRSC `keep_ratio` / `pad` eval uses the same whole-image scale as training. `make eval-val` takes `EXPERIMENT=`, not `CONFIG=`.

### Fixed

- **Diagonal flip angle** — MMRotate `RRandomFlip(direction='diagonal')` mirrors box centers and **keeps θ**. We incorrectly applied `π − θ`.
- **`keep_ratio` + `pad_size_divisor` 32** no longer crashes on P6 (odd feature-map `max_pool`); stride derivation falls back to `[4, 8, 16, 32, 64]`. Collate bottom-right pads a batch to a shared H×W.
- **Rotated RetinaNet `proj_xy`** — encode/decode dx/dy in the anchor local frame (no-op for Hub `θ = 0` priors).
- **`odet preds --checkpoint hf://<slug>`** uses the pretrained sidecar (no longer the latest `runs/` JSON). **`dota-submit` after Hub inference** includes `output_dir` / `predictions_json` in metadata.
- **SSDD dump discovery** no longer treats `HRSID_JPG` as SSDD. `python -m oriented_det.cli` works via `__main__.py`. FAIR1M large rasters set `Image.MAX_IMAGE_PIXELS = None`.

### Removed

- TensorFlow / SavedModel / TFLite export, and `odet export-*`. ONNX is `python -m export` with `oriented-det[export]`.
- Hub slugs **`rotated_faster_rcnn_dota_le90_3x_ce`** and **`rotated_fcos_dota_le90_3x_kfiou_aux`** (eval reports stay under [`docs/eval-reports/`](eval-reports/)).
- Rotated RetinaNet 1× oriented-priors ablation (`dota_le90_1x_oriented_anchors.json`). Hub stays `θ = 0`.
- HRSC2016 **6×** recipes. 3× ±20° is the long schedule.
- Notebook geometric transform classes (`Rotate`, `HorizontalFlip`, …). Image+box augs are `apply_random_train_flips` / `apply_random_train_rotate` and `apply_flip_to_*` / `apply_rotate_to_*`.

## [0.2.0] - 2026-08-25

### Added

- **Rotated FCOS** (`model_type: rotated_fcos`) — anchor-free single-stage detector: DistanceAnglePointCoder, center-in-OBB assigner, centerness, and L1 / KFIoU / decoded rIoU box regression. Recipes under [`configs/rotated_fcos/`](../configs/rotated_fcos/).
- Differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`) for FCOS **`box_reg_loss_type: riou`** (`1 - IoU`). Distinct from sampling `pairwise_rotated_iou`. Recipes [`dota_le90_1x_riou.json`](../configs/rotated_fcos/dota_le90_1x_riou.json) / [`dota_le90_3x.json`](../configs/rotated_fcos/dota_le90_3x.json) (lr 2.5e-3).
- Hub slug **`rotated_fcos_dota_le90_3x`** (81.58% eval-val mAP50) — Rotated FCOS 3× decoded rIoU; report [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md).
- Hub slug **`rotated_fcos_dota_le90_3x_kfiou_aux`** (77.18% eval-val mAP50) — Rotated FCOS 3× L1 + KFIoU aux; report [`docs/eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/`](eval-reports/rotated_fcos_dota_le90_3x_kfiou_aux/model_analysis.md).
- TF/ONNX export mode **`rotated_fcos_pre_nms`** — Rotated FCOS decode + pad uses the same Keras detect bundle as two-stage models.

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
