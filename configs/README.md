# Configs

Training and evaluation are driven by **JSON config files** with MMRotate-style **`_base_`** inheritance. Base paths may be **file-relative** (same directory as the recipe), **absolute**, or prefixed with **`@odet:`** (path relative to the oriented-det repository root).

## DOTA pretrained models (model zoo)

OrientedDet publishes **DOTA le90 pretrain** checkpoints on Hugging Face Hub. See [pretrained/README.md](../pretrained/README.md) and per-model READMEs below. Write-ups: [v0.1.1 zoo / MMRotate parity](https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) · [Rotated Faster R-CNN / ProbIoU](https://deeplearning.earth/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) · [v0.2.0 Rotated FCOS](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

**Split protocol (`make eval-val` is not always held-out):**

| Dataset | Train | `make eval-val` | Held-out test |
|---------|-------|-----------------|---------------|
| **DOTA v1.0** | train **+ val** tiles (`train_tiles_dirs`; MMRotate pretrain) | **Leaky** — val tiles are also in train | **Official Task 1** (`make dota-submit`; labels not public) |
| **HRSC2016** | ImageSets **trainval** | ImageSets **test** (453 images) | Same as eval-val — test is **not** in train |
| **FAIR1M** | official **train** tiles only | official **val** tiles | Gaofen test labels are not public; val is **not** in train |
| **SSDD** | official last-digit **train** | official last-digit **test** | Same as eval-val — test is **not** in train |
| **HRSID** | official **train** | official **test** | Same as eval-val — no official val |

Do **not** call HRSC, FAIR1M, SSDD, or HRSID `make eval-val` leaky. That word is DOTA-only. Full table: [Data guide — Train / eval splits](../docs/user-guide/data.md#train--eval-splits).

**mAP in README tables** for Faster R-CNN, FCOS, Oriented R-CNN 1× / **3×**, and **RetinaNet 1× / 3×** DOTA is **official Task 1** (the real hidden test). FCOS 3× is also Task 1 (**72.91%**; leaky eval-val 82.32%). Training-time periodic mAP may be higher (non-empty tiles only). Advertised Oriented R-CNN / Faster R-CNN / FCOS zoo is **1×**; 3× is the long-schedule Hub slug (tight boxes / AP75). See [oriented_rcnn/README.md — 1x vs 3x](oriented_rcnn/README.md#1x-vs-3x).

| Model | Config | Hub slug | mAP50 |
|-------|--------|----------|----------------|
| Oriented R-CNN 1× | [oriented_rcnn/dota_le90_1x.json](oriented_rcnn/dota_le90_1x.json) | `oriented_rcnn_dota_le90_1x` | **76.73%** official Task 1 |
| Oriented R-CNN 3× | [oriented_rcnn/dota_le90_3x.json](oriented_rcnn/dota_le90_3x.json) | `oriented_rcnn_dota_le90_3x` | **74.88%** official Task 1 |
| Rotated RetinaNet 1× (OBB) | [rotated_retinanet/dota_le90_1x.json](rotated_retinanet/dota_le90_1x.json) | `rotated_retinanet_dota_le90_1x` | **71.72%** official Task 1 |
| Rotated RetinaNet 3× (OBB) | [rotated_retinanet/dota_le90_3x.json](rotated_retinanet/dota_le90_3x.json) | `rotated_retinanet_dota_le90_3x` | **73.89%** official Task 1 |
| Rotated RetinaNet 1× (HBB) | [rotated_retinanet/dota_le90_1x_hbb.json](rotated_retinanet/dota_le90_1x_hbb.json) | `rotated_retinanet_dota_le90_1x_hbb` | **67.87%** official Task 1 |
| Rotated RetinaNet 3× (HBB) | [rotated_retinanet/dota_le90_3x_hbb.json](rotated_retinanet/dota_le90_3x_hbb.json) | `rotated_retinanet_dota_le90_3x_hbb` | **70.70%** official Task 1 |
| Rotated Faster R-CNN 1× | [rotated_faster_rcnn/dota_le90_1x.json](rotated_faster_rcnn/dota_le90_1x.json) | `rotated_faster_rcnn_dota_le90_1x` | **74.42%** official Task 1 |
| Rotated Faster R-CNN 3× | [rotated_faster_rcnn/dota_le90_3x.json](rotated_faster_rcnn/dota_le90_3x.json) | `rotated_faster_rcnn_dota_le90_3x` | **74.48%** official Task 1 |
| Rotated FCOS 1× rIoU | [rotated_fcos/dota_le90_1x.json](rotated_fcos/dota_le90_1x.json) | `rotated_fcos_dota_le90_1x` | **73.07%** official Task 1 |
| Rotated FCOS 3× rIoU | [rotated_fcos/dota_le90_3x.json](rotated_fcos/dota_le90_3x.json) | `rotated_fcos_dota_le90_3x` | **72.91%** official Task 1 |
| Rotated RTMDet-S 3× | [rotated_rtmdet/dota_le90_3x_s.json](rotated_rtmdet/dota_le90_3x_s.json) | `rotated_rtmdet_dota_le90_3x` | **77.60%** official Task 1 |

## HRSC2016 pretrained models

**Training: ImageSets trainval.** **Evaluation: ImageSets test** (453 images) — **held-out**, not in training. Whole-image `keep_ratio` + pad-32. FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN 3× use random rotate p=0.5 ±20°; Oriented R-CNN / Faster R-CNN 1× leave rotate off.

| Model | Config | Hub slug | held-out test mAP50 |
|-------|--------|----------|----------------|
| Oriented R-CNN 3× | [oriented_rcnn/hrsc2016_le90_3x.json](oriented_rcnn/hrsc2016_le90_3x.json) | `oriented_rcnn_hrsc2016_le90_3x` | 90.41% |
| Rotated Faster R-CNN 3× | [rotated_faster_rcnn/hrsc2016_le90_3x.json](rotated_faster_rcnn/hrsc2016_le90_3x.json) | `rotated_faster_rcnn_hrsc2016_le90_3x` | 88.77% |
| Rotated FCOS 3× rIoU | [rotated_fcos/hrsc2016_le90_3x.json](rotated_fcos/hrsc2016_le90_3x.json) | `rotated_fcos_hrsc2016_le90_3x` | 88.34% |

### FAIR1M le90 (train locally — finetune DOTA 1× Hub; no FAIR1M zoo)

**Training: official train tiles only.** **Evaluation: official val tiles** — **held-out**, not in training (unlike DOTA). Tiled official val, not Gaofen hidden test (labels not public). Mid-30s mAP50 is expected — FAIR1M is 37 fine-grained classes, not DOTA’s 15.

| Recipe | Config | tiled-val mAP50 |
|--------|--------|-----------------|
| Oriented R-CNN 1× | [oriented_rcnn/fair1m_le90_1x.json](oriented_rcnn/fair1m_le90_1x.json) | (not run) |
| Rotated Faster R-CNN 1× | [rotated_faster_rcnn/fair1m_le90_1x.json](rotated_faster_rcnn/fair1m_le90_1x.json) | **36.70%** (`runs/rotated_faster_rcnn/20260910-072116`) |
| Rotated FCOS 1× rIoU | [rotated_fcos/fair1m_le90_1x.json](rotated_fcos/fair1m_le90_1x.json) | (not run) |

See [Data guide — FAIR1M](../docs/user-guide/data.md#fair1m) (local 1× story, literature band, class-confusion diagnosis).

### SSDD / HRSID le90 (train locally — finetune DOTA 1× Hub; no SAR zoo)

**SSDD:** official last-digit train / held-out test, keep-ratio 608. **HRSID:** official train / held-out test (no val), keep-ratio 800. Both 1× recipes re-init the 1-way `ship` head from the matching DOTA Hub checkpoint.

| Recipe | Config | held-out test mAP50 |
|--------|--------|---------------------|
| Faster R-CNN 1× SSDD | [rotated_faster_rcnn/ssdd_le90_1x.json](rotated_faster_rcnn/ssdd_le90_1x.json) | **90.34%** (`runs/rotated_faster_rcnn/20260918-130546`; [report](../docs/eval-reports/rotated_faster_rcnn_ssdd_le90_1x/model_analysis.md)) |
| Oriented R-CNN 1× HRSID | [oriented_rcnn/hrsid_le90_1x.json](oriented_rcnn/hrsid_le90_1x.json) | (not run) |
| Faster R-CNN 1× HRSID | [rotated_faster_rcnn/hrsid_le90_1x.json](rotated_faster_rcnn/hrsid_le90_1x.json) | **78.55%** (`runs/rotated_faster_rcnn/20260918-134544`; [report](../docs/eval-reports/rotated_faster_rcnn_hrsid_le90_1x/model_analysis.md); Wei HBB AP50 **>84.7%**) |
| FCOS 1× HRSID | [rotated_fcos/hrsid_le90_1x.json](rotated_fcos/hrsid_le90_1x.json) | (not run) |

See [Data guide — SSDD](../docs/user-guide/data.md#ssdd) and [HRSID](../docs/user-guide/data.md#hrsid).

Download: `odet pretrained download <slug>` or `"load_from_checkpoint": "hf://<slug>"`. Published eval-val reports: [`docs/eval-reports/`](../docs/eval-reports/) (markdown + analysis JSON; `predictions.json` stays in gitignored [`predictions/`](../predictions/) for the viewer).

| Prefix | Resolves under | Override env |
|--------|----------------|--------------|
| `@odet:` | `ORIENTED_DET_ROOT` (repo root) | `ORIENTED_DET_ROOT`, or installed package location |

Example in an external project that keeps local dataset fragments:

```json
{
  "_base_": [
    "my_dataset.json",
    "@odet:configs/_base_/models/oriented_rcnn_r50.json",
    "@odet:configs/_base_/schedules/1x.json"
  ]
}
```

## Documentation

- **[Configuration reference](../docs/user-guide/configuration.md)** — human-readable guide: sections, recipes, CLI, `production` vs `evaluation`
- **[config.schema.json](config.schema.json)** — machine-readable types and defaults (kept in sync with `oriented_det/train/config.py`)

## Layout

- **[_base_/](_base_)** — Dataset, model, schedule, FP16, preprocessing, augmentation fragments (do not run directly; included via `_base_`).
- **[oriented_rcnn/](oriented_rcnn/)** — Oriented R-CNN (horizontal RPN + midpoint-offset + oriented ROI). See [oriented_rcnn/README.md](oriented_rcnn/README.md).
- **[rotated_faster_rcnn/](rotated_faster_rcnn/)** — Rotated Faster R-CNN. Standard recipe: ProbIoU main ROI loss, Smooth L1 aux 0.1, angle weight 1.0 (`dota_le90_1x.json`; 3× via `dota_le90_3x.json`). See [rotated_faster_rcnn/README.md](rotated_faster_rcnn/README.md).
- **[rotated_retinanet/](rotated_retinanet/)** — Rotated RetinaNet (one-stage). See [rotated_retinanet/README.md](rotated_retinanet/README.md).
- **[rotated_fcos/](rotated_fcos/)** — Rotated FCOS (anchor-free one-stage; L1, optional KFIoU aux, or decoded rIoU primary). See [rotated_fcos/README.md](rotated_fcos/README.md).
- **[rotated_rtmdet/](rotated_rtmdet/)** — Rotated RTMDet (CSPNeXt-S + PAFPN speed tier; Hub 3×). See [rotated_rtmdet/README.md](rotated_rtmdet/README.md).

## Muted keys

Prefix a key with **`_muted_`** to keep an alternate value in the file without affecting training (stripped before validation):

```json
{
  "training": {
    "learning_rate": 0.002,
    "_muted_learning_rate": 0.005
  }
}
```

## Full option reference

All options, types, and defaults: **[config.schema.json](config.schema.json)** (synced with **oriented_det/train/config.py** and **tools/train.py**). Human-readable detail: **[Configuration reference](../docs/user-guide/configuration.md)**.

**Key order** (schema, saved `config.json`, docs): switches first (`lr_scheduler_type`, `loss_type`, `enable_albumentation`), then grouped prefixes (`lr_*`, `roi_*`, `rpn_*`). Top-level sections:

| Section | Description |
|--------|-------------|
| `_base_` | Base config path(s): relative to current file, `@odet:…`, or absolute |
| `model_type` | `oriented_rcnn`, `rotated_faster_rcnn`, `rotated_retinanet`, or `rotated_fcos` |
| `dataset` | data_root, format (dota / airbus_playground / hrsc2016 / fair1m / ssdd / hrsid), train_tiles_dir, val_tiles_dir, **train_tiles_dirs**, **val_tiles_dirs** (optional lists; union without on-disk merge), **same_folder** (DOTA: images and .txt in same dir), **overlap** (tile overlap px, even; deploy uses margin = overlap/2), annotations_file, split_file, **val_split_id**, **train_includes_val** (Airbus: train on all folds; val fold for monitoring only), **train_split** / **val_split** (HRSC2016 ImageSets / FAIR1M splits; HRSC defaults trainval/test, FAIR1M train/val), difficult_strategy, **difficult_tags** (Airbus: exact tags → difficult=1 + strip, e.g. `["Partially Hidden"]`; use with strategy `ignore` for don't-care — not lookalike), **filter_empty_gt** (drop tiles/images with no GT after filters; MMRotate parity), **drop_easy_empty_tiles** (with `tile_metrics_csv`: drop train `tp=fp=fn=0` tiles, keep empty+FP for hard-tile oversampling), **class_tile_oversample_classes** / **class_tile_oversample_factor** / **class_tile_oversample_min_count** (optional GT class-presence oversampling; composes with hard-tile weights), max_train_samples, max_val_samples, **max_samples_shuffle_seed** (deterministic spread when capping), allowed_classes, ignore_labels, map_labels, **lookalike_labels** (optional aliases; reserved name `lookalike` is always a hard-negative token, never a class — map confusers with `map_labels`, e.g. `{"Confuser":"lookalike"}`; see [data.md — Lookalike confusers](../docs/user-guide/data.md#lookalike-confusers)) |
| `data_loader` | batch_size, num_workers, shuffle, pin_memory |
| `model` | backbone, fpn_*, anchor_*, target_means/stds, roi_* (loss, batch, iou, schedule, **roi_proj_xy**), rpn_*, use_hbb_for_matching, add_gt_as_proposals, **rpn_nms_threshold** (proposal NMS), **final_nms_iou_threshold** / **final_nms_iou_schedule_*** (post–ROI-head NMS), **final_nms_use_cpu** (exact polygon final NMS on CPU), nms_class_agnostic, max_detections_per_image, inference_pre_nms_score_threshold |
| `training` | **lr_scheduler_type** first, then lr_scheduler_*, lr_warmup_steps, lr_scaling_*, use_lr_param_groups, lr_mult_*, then num_epochs, learning_rate, momentum, weight_decay, use_amp, gradient_accumulation_steps, max_grad_norm, loss_weights. See [Training — Learning rate scheduling](../docs/user-guide/training.md#learning-rate-scheduling) and [_base_/schedules/README.md](_base_/schedules/README.md). |
| `evaluation` | **`val_score_threshold`** (periodic val / `make eval-val` / `best_mAP`), **`test_score_threshold`** for **`test_map` / `make eval-test`** (null → **0.05**), iou_threshold, compute_map_*; **`final_nms_iou_threshold`** for eval-val NMS only (DOTA+HRSC: **0.1**, MMRotate test parity) |
| `production` | Optional **deploy / `image_demo`** overrides: `score_threshold` via `resolve_inference_score_threshold` (does **not** override train-val mAP; `odet preds` ignores it). DOTA 1× recipes set this to eval-val global F1 − **0.05** (Oriented R-CNN **0.55**, Faster R-CNN **0.6**, RetinaNet OBB **0.25** / HBB **0.35**, FCOS **0.2**); 3× inherits those floors. HRSC 1× recipes do the same (Oriented R-CNN / Faster R-CNN **0.85**, FCOS **0.2**). `per_class_score_threshold` merges on top of `evaluation` for preds/deploy only (`merge_per_class_score_thresholds`; mAP IoU is `evaluation.iou_threshold` only). **Decode (deploy / `image_demo`)** — RPN/NMS/threshold fields patch the loaded model via `apply_inference_config_to_model` (DOTA+HRSC+FAIR1M: final NMS **0.1**). **`odet preds`** prefers `evaluation.final_nms_iou_threshold` when set. Not applied during `tools/train.py` (training uses `model.*`, final NMS **0.1**). **Deploy / tiling** — `overlap_pixels`, `ignore_margin_pixels`, canvas flags (see **config.schema.json**). |
| `checkpoint` | load_from_checkpoint, load_from_experiment, **discover_previous_run**, resume_from_checkpoint_epoch, load_optimizer_state, load_scheduler_state, load_include_prefixes, load_exclude_prefixes, start_epoch, best_metric, higher_is_better |
| `loss` | loss_type (`focal_weighted` now also scales FCOS/RetinaNet sigmoid focal per class; `background_weight` is ROI-only), class_weight_method, background_weight, focal_alpha, focal_gamma, label_smoothing, **roi_grouped_ce_*** (coarse-to-fine ROI classifier curriculum in one run) |
| `preprocessing` | resize_mode, target_size, normalize_mean, normalize_std, pad_size_divisor, enable_flip_horizontal, enable_flip_vertical, enable_flip_diagonal, enable_random_rotate, random_rotate_prob, random_rotate_angle_range |
| Top-level | **enable_albumentation**, enable_profiling |
| `augmentation` | Albumentations params (when enable_albumentation is true) |
| `tensorboard` | log_debug_anchors_proposals, vis_score_threshold |

## PyPI package (`oriented_det.configs`)

A subset of this tree is copied into **`oriented_det/configs/`** for wheels (see **`oriented_det/configs/vendored_manifest.txt`**), including DOTA 1×/3×, HRSC, FAIR1M, SSDD, and HRSID recipes. After editing files here:

```bash
make sync-configs    # update vendored copy
make check-configs   # verify before commit (also runs in CI)
```

## Training

From the repo root:

```bash
python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json
```

Override batch size or AMP: `--batch-size`, `--use-amp`, `--no-amp`. See [tools/README.md](../tools/README.md) and the [main README](../README.md).
