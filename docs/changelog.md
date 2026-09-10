# Changelog

All notable changes to OrientedDet will be documented in this file.

## [Unreleased]

### Added

- Hub slug **`oriented_rcnn_dota_le90_1x`** from `runs/oriented_rcnn/20260908-144807` (`oriented_rcnn_r50_fpn_dota_le90_1x-725c244f`) — official DOTA v1.0 Task 1 **76.73%** (AP75 50.24, COCO mAP 46.59). Deploy `production.score_threshold` **0.55** (eval-val F1 0.60 − 0.05). Report [`docs/eval-reports/oriented_rcnn_dota_le90_1x/`](eval-reports/oriented_rcnn_dota_le90_1x/model_analysis.md).
- Hub slug **`rotated_faster_rcnn_dota_le90_3x`** restored (`rotated_faster_rcnn_r50_fpn_dota_le90_3x-9951acc6`) — official DOTA v1.0 Task 1 **74.48%** (AP75 45.39). Advertised short schedule stays 1× (`rotated_faster_rcnn_dota_le90_1x`, Task 1 74.42% / AP75 41.90). Report [`docs/eval-reports/rotated_faster_rcnn_dota_le90_3x/`](eval-reports/rotated_faster_rcnn_dota_le90_3x/model_analysis.md).
- Hub slug **`rotated_fcos_dota_le90_1x`** from `runs/rotated_fcos/20260908-023531` — official DOTA v1.0 Task 1 **73.07%** (AP75 40.40). Deploy `production.score_threshold` **0.2** (eval-val F1 0.25 − 0.05). DOTA zoo recipe is 1×; `rotated_fcos_dota_le90_3x` remains on Hub. Report [`docs/eval-reports/rotated_fcos_dota_le90_1x/`](eval-reports/rotated_fcos_dota_le90_1x/model_analysis.md).
- Hub slug **`rotated_faster_rcnn_dota_le90_1x`** from `runs/rotated_faster_rcnn/20260907-124458` — official DOTA v1.0 Task 1 **74.42%** (AP75 41.90). Deploy `production.score_threshold` **0.6** (eval-val F1 0.65 − 0.05). Report [`docs/eval-reports/rotated_faster_rcnn_dota_le90_1x/`](eval-reports/rotated_faster_rcnn_dota_le90_1x/model_analysis.md).
- **FAIR1M** dataset support (`dataset.format: fair1m`) — 37-class XML loader (`fair1m.py` / `fair1m_classes.py`), official + Kaggle layouts including ollypowell `Notebook_Working/{train,val}_labels` (`t_N`/`v_N.jpg` → `N.xml`), `odet fair1m-to-dota` with deterministic `--val-fraction` holdout, then tile 1024/200. 1× recipes finetune the matching **DOTA 1× Hub** checkpoint (37-way classifier re-init; no FAIR1M zoo): [`configs/oriented_rcnn/fair1m_le90_1x.json`](../configs/oriented_rcnn/fair1m_le90_1x.json), [`configs/rotated_faster_rcnn/fair1m_le90_1x.json`](../configs/rotated_faster_rcnn/fair1m_le90_1x.json), [`configs/rotated_fcos/fair1m_le90_1x.json`](../configs/rotated_fcos/fair1m_le90_1x.json). See [data.md](user-guide/data.md#fair1m). Kaggle notebook: [`notebooks/kaggle_fair1m_tutorial.ipynb`](../notebooks/kaggle_fair1m_tutorial.ipynb).
- **DOTA Task 1 submit** — `odet dota-submit` / `make dota-submit` writes `Task1_*.txt` + zip from unlabeled official test inference (`--checkpoint hf://<slug>`, `--experiment-dir`, or `--from-json`). Hub slugs use the pretrained sidecar JSON (same as `odet image-demo`). Inference NMS is **0.1** (eval-val), not Hub sidecar `production.final_nms_iou_threshold` (often 0.5). Test labels stay private; upload to the [DOTA v1.0 Task 1 server](https://captain-whu.github.io/DOTA/evaluation.html). Sliding-window last tiles flush to the image edge (`tile_dota.py` parity); per-window margin **0** keeps overlap copies, then NMS (live ResultMerge-style, not MMRotate on-disk pre-tile merge).
- **Random rotate** (train only) — `preprocessing.enable_random_rotate` / `random_rotate_prob` / `random_rotate_angle_range` (degrees). MMRotate `PolyRandomRotate` after flips (`auto_bound=False`). FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN HRSC 3× use p=0.5 **±20°**. Oriented R-CNN / Faster R-CNN 1× and DOTA Hub stay off. RetinaNet 1× RR ablation: [`dota_le90_1x_rr.json`](../configs/rotated_retinanet/dota_le90_1x_rr.json) (p=0.5 **±180°**). ±180° on FCOS 6× diverged after epoch 12.
- **HRSC2016** dataset loader (`dataset.format: hrsc2016`) — official XML + ImageSets, single-class `ship`, le90 via the DOTA polygon path. Recipes: [`configs/oriented_rcnn/hrsc2016_le90_1x.json`](../configs/oriented_rcnn/hrsc2016_le90_1x.json), [`configs/oriented_rcnn/hrsc2016_le90_3x.json`](../configs/oriented_rcnn/hrsc2016_le90_3x.json), [`configs/rotated_faster_rcnn/hrsc2016_le90_1x.json`](../configs/rotated_faster_rcnn/hrsc2016_le90_1x.json), [`configs/rotated_faster_rcnn/hrsc2016_le90_3x.json`](../configs/rotated_faster_rcnn/hrsc2016_le90_3x.json), [`configs/rotated_fcos/hrsc2016_le90_1x.json`](../configs/rotated_fcos/hrsc2016_le90_1x.json), [`configs/rotated_fcos/hrsc2016_le90_3x.json`](../configs/rotated_fcos/hrsc2016_le90_3x.json). Optional DOTA export: `odet hrsc-to-dota`.
- **`resize_mode: keep_ratio`** — MMRotate-style long-edge scale without square pad; training/inference then apply `pad_size_divisor` (bottom-right). HRSC Oriented R-CNN, Faster R-CNN, and FCOS 1×/3× use this canvas. Oriented R-CNN uses Smooth L1 main + ProbIoU aux 0.1. Faster R-CNN keeps DOTA ProbIoU main + Smooth L1 aux 0.1. HRSC FCOS recipes use decoded **rIoU** (lr **2.5e-3**), not L1. FCOS HRSC 3× uses `lr_scheduler_gamma: [0.1, 0.5]`.
- **`training.lr_scheduler_gamma`** accepts a number (same factor every drop) or a list (one factor per `lr_scheduler_milestones` entry).
- Hub slug **`oriented_rcnn_hrsc2016_le90_3x`** (90.41% eval-val mAP50) — Oriented R-CNN 3× on HRSC2016 test (rotate ±20°, gamma 0.1); report [`docs/eval-reports/oriented_rcnn_hrsc2016_le90_3x/`](eval-reports/oriented_rcnn_hrsc2016_le90_3x/model_analysis.md).
- Hub slug **`rotated_faster_rcnn_hrsc2016_le90_3x`** (88.77% eval-val mAP50) — Faster R-CNN 3× on HRSC2016 test; report [`docs/eval-reports/rotated_faster_rcnn_hrsc2016_le90_3x/`](eval-reports/rotated_faster_rcnn_hrsc2016_le90_3x/model_analysis.md).
- Hub slug **`rotated_fcos_hrsc2016_le90_3x`** (88.34% eval-val mAP50) — FCOS 3× decoded rIoU on HRSC2016 test; report [`docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/`](eval-reports/rotated_fcos_hrsc2016_le90_3x/model_analysis.md).
- **`dataset.drop_easy_empty_tiles`** — with `tile_metrics_csv`, drop train tiles that are vacuous true negatives (`tp=fp=fn=0`) before `max_train_samples` and hard-tile oversampling. Empty tiles with false positives stay and can be oversampled. Keep `filter_empty_gt: false` so those hard empties remain in the loader.
- **`dataset.class_tile_oversample_classes`** — optional train-tile oversampling by GT class presence (no metrics CSV). `class_tile_oversample_factor` (default `1.0`) and `class_tile_oversample_min_count` (default `1`). Reuses the hard-tile sampler; weights multiply when both paths are on. Lookalike routing labels are never a match; unknown names are warned once and ignored.
- **Class-agnostic final NMS** on Rotated FCOS and Rotated RetinaNet via `model.nms_class_agnostic` / `production.nms_class_agnostic` (default `false`, same contract as two-stage). DOTA / HRSC recipes stay class-aware.
- **`model.anchor_angles`** (degrees) for Rotated RetinaNet; Hub 1×/3× stay `[0]`.
- **Rotated RetinaNet 1× RR** — [`dota_le90_1x_rr.json`](../configs/rotated_retinanet/dota_le90_1x_rr.json) inherits Hub 1× and enables `PolyRandomRotate` p=0.5 **±180°** (MMRotate RR; not Hub).

### Changed

- DOTA Faster R-CNN **finetune init is 1×** (`hf://rotated_faster_rcnn_dota_le90_1x`). 3× overfits train+val tiles (leaky eval-val − Task 1 is 9.0 vs 3.1 on 1×); official AP50 is a wash (74.48 vs 74.42). AP75 is the 3× gain (45.39 vs 41.90). See [configs/rotated_faster_rcnn/README.md](../configs/rotated_faster_rcnn/README.md#1x-vs-3x-for-finetune).
- **`odet tile-dota`** reads JPEG/TIFF/BMP as well as PNG. `--output-format auto` (default) writes JPEG tiles from JPEG sources and PNG otherwise; `--output-format png|jpg` forces a format (`--jpeg-quality` default 95). **`odet fair1m-to-dota`** default `--image-format original` copies JPEG/PNG instead of rewriting PNG.
- Oriented R-CNN DOTA 1× deploy floor `production.score_threshold` **0.55** (eval-val F1 0.60 − 0.05). 3× recipe pins **0.7** so Hub `oriented_rcnn_dota_le90_3x` does not inherit.
- **DOTA / HRSC 1× recipes** carry Hub deploy `production.score_threshold` (eval-val F1 − **0.05**). Oriented R-CNN DOTA 1× is **0.7** (was 0.05). DOTA 3× is 1× + 36 epochs (milestones `[24, 33]`) only — FCOS 3× no longer sets warmup 2000; RetinaNet 3× no longer sets train-time `max_detections_per_image` 300. Restored [`configs/rotated_fcos/dota_le90_3x.json`](../configs/rotated_fcos/dota_le90_3x.json).
- **DOTA FCOS / RetinaNet 1×** inherit [`configs/_base_/datasets/dota_le90.json`](../configs/_base_/datasets/dota_le90.json) (DOTA `odet stats` mean/std) like Oriented R-CNN and Faster R-CNN. Previous one-stage recipes used ImageNet defaults.

### Fixed

- **FAIR1M large rasters** — Pillow `DecompressionBombWarning` on ~95 MP satellite images during `odet fair1m-to-dota` / dataset load. Same as `tile_dota.py`: `Image.MAX_IMAGE_PIXELS = None`.
- **`dota-submit` after Hub zoo inference** — `run_inference_and_save` metadata had no `output_dir`, so the zip step crashed with `KeyError` after a successful 937-image run. Metadata now includes `output_dir` / `predictions_json`; convert existing JSON with `--from-json`.
- **`odet preds --checkpoint hf://<slug>`** no longer fills a missing `--config` from the latest `runs/` experiment (that mixed zoo weights with another run’s JSON). The pretrained sidecar is used, same as `odet image-demo`.
- **`make eval-val` / `odet preds` score floor** — no longer uses `production.score_threshold` or train-val `evaluation.score_threshold` (often 0.3). Resolver is CLI → `evaluation.preds_score_threshold` → **0.05**. Deploy / train val are unchanged.
- **Diagonal flip angle** — MMRotate `RRandomFlip(direction='diagonal')` mirrors box centers and **keeps θ** (early return). We incorrectly applied `π − θ`, so diagonal samples (and diagonal + random rotate) had boxes at the wrong orientation. Horizontal / vertical flips were already correct.
- **Rotated RetinaNet `proj_xy`** — encode/decode dx/dy in the anchor local frame (MMRotate `DeltaXYWHAOBBoxCoder`). No-op for Hub horizontal priors; required for `model.anchor_angles` other than `[0]`.
- **`keep_ratio` + `pad_size_divisor` 32** no longer crashes on P6: torchvision `max_pool` on an odd feature map (e.g. 576×800 → 9×13) is anisotropic; stride derivation now falls back to configured `[4, 8, 16, 32, 64]` (MMRotate).
- **`keep_ratio` collate** bottom-right pads a batch to a shared H×W (e.g. 480×800 + 576×800) so `torch.stack` works. Per-image `content_size` is unchanged; `odet preds` stays one image + divisor pad.

### Removed

- **Rotated RetinaNet 1× oriented-priors ablation** (`configs/rotated_retinanet/dota_le90_1x_oriented_anchors.json`, run `20260905-043814`, eval `predictions/20260906_234246`). eval-val **64.63%** vs Hub 1× **64.14%**; ship / small-vehicle / bridge down. Hub stays `θ = 0`.
- Hub slugs **`oriented_rcnn_dota_le90_1x`**, **`rotated_faster_rcnn_dota_le90_3x_ce`**, **`rotated_retinanet_dota_le90_1x`**, and **`rotated_fcos_dota_le90_3x_kfiou_aux`**. Eval reports stay under [`docs/eval-reports/`](eval-reports/). DOTA Faster R-CNN zoo is 1× (`rotated_faster_rcnn_dota_le90_1x`).
- HRSC2016 **6×** recipes (`oriented_rcnn/hrsc2016_le90_6x.json`, `rotated_fcos/hrsc2016_le90_6x.json`). 3× ±20° is the long schedule; 6× was only +0.2 mAP on Oriented R-CNN and FCOS 6× never beat that 3×.
- Notebook geometric transform classes (`Rotate`, `HorizontalFlip`, `VerticalFlip`, `DiagonalFlip`, `Compose`, `OrientedTransform`). Image+box augs are only `apply_random_train_flips` / `apply_random_train_rotate` and `apply_flip_to_*` / `apply_rotate_to_*`.
- **`export/`** (ONNX / TensorFlow tooling), the `odet export-*` subcommands, and the `oriented-det[export]` extra. This repo no longer ships a TF/ONNX export pipeline.

### Changed

- **Sliding-window last tiles flush to the image edge** (same as `tile_dota.py`): the last origin is `size - canvas` so the window is a full slice of real pixels, not a leftover pad. Per-window centroid margin default is **0** (keep overlap copies, then NMS) for `odet preds`, `image-demo`, deploy, and `dota-submit`. Pass `--window-margin-pixels` / `--ignore-margin-pixels` to drop the overlap band.
- **Sliding-window micro-batch default is 8** (GPU) / **4** (CPU). Unset `ORIENTED_DET_WINDOW_BATCH_SIZE` no longer binary-searches max VRAM (empty-canvas probe overestimates two-stage usage and can OOM mid-run). Set `auto` to restore probing. `make` now exports `ORIENTED_DET_WINDOW_BATCH_SIZE=8` (was 64, and previously not passed through `TRAIN_ENV`).
- Recipe and Hub sidecar **`production.final_nms_iou_threshold`** is **0.1** (was 0.3 / 0.5 on some DOTA sidecars) so deploy / `image_demo` match train val and eval-val NMS. Weights are unchanged.
- **Rotated RetinaNet assignment** — rotated MaxIoU matching no longer goes through the shared `oriented_box_iou_gpu` path (slow on dense P3 / 27-anchor grids). `retinanet_assign.py` is RetinaNet-only; two-stage RPN/ROI matching is unchanged.
- DOTA 3× Hub deploy floors — `production.score_threshold` is eval-val global F1 − **0.05** on all four slugs: Oriented R-CNN **0.7** (F1 0.75), Faster R-CNN **0.6** (F1 0.65), RetinaNet **0.45** (F1 0.50), FCOS **0.2** (F1 0.25). Recipes, Hub sidecars, and docs updated. `make eval-val` still uses **0.05**.
- HRSC 3× Hub deploy floors — same F1 − **0.05** rule: Oriented R-CNN / Faster R-CNN **0.85** (F1 0.90), FCOS **0.2** (F1 0.25). Recipes, Hub sidecars, and docs updated.
- Hub slug **`rotated_faster_rcnn_dota_le90_3x`** refreshed from `runs/rotated_faster_rcnn/20260901-095802` (**83.46%** eval-val mAP50, was 83.42%). Weight stem is now `rotated_faster_rcnn_r50_fpn_dota_le90_3x-9951acc6`. Deploy `production.score_threshold` **0.6** (eval-val F1 0.65 − 0.05). Report [`docs/eval-reports/rotated_faster_rcnn_dota_le90_3x/`](eval-reports/rotated_faster_rcnn_dota_le90_3x/model_analysis.md).
- **`loss.focal_weighted`** now scales Rotated FCOS and Rotated RetinaNet sigmoid focal loss per class (same `loss.class_weight_*` as ROI heads). `focal` stays unweighted. `background_weight` is ignored on one-stage one-hot focal.
- **`make wizard` / `--wizard`** FCOS FPN and box-reg nudges are class-agnostic: pooled GT width vs finest stride (keep P3 + decoded `kfiou` when boxes occupy few cells), not DOTA class names such as small-vehicle.
- Hub slug **`rotated_fcos_dota_le90_3x`** refreshed from `runs/rotated_fcos/20260831-052647` (**82.32%** eval-val mAP50, was 81.58%). Weight stem is now `rotated_fcos_r50_fpn_dota_le90_3x-6e383331` (dropped leftover `_riou` in the filename). Report [`docs/eval-reports/rotated_fcos_dota_le90_3x/`](eval-reports/rotated_fcos_dota_le90_3x/model_analysis.md).
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
