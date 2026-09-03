# Configs

Training and evaluation are driven by **JSON config files** with MMRotate-style **`_base_`** inheritance. Base paths may be **file-relative** (same directory as the recipe), **absolute**, or prefixed with **`@odet:`** (path relative to the oriented-det repository root).

## DOTA pretrained models (model zoo)

OrientedDet publishes **DOTA le90 pretrain** checkpoints on Hugging Face Hub. See [pretrained/README.md](../pretrained/README.md) and per-model READMEs below.

**Split protocol (important):**

- **Training: train+val** — `train_tiles_dirs` unions `train` and `val` tile roots (DOTA benchmark pretrain, same idea as MMRotate).
- **Evaluation: val** — mAP is on the **val** tile set only.
- This is **not** a fine-tune train/val holdout: val tiles are included in training.

**mAP in README tables** uses **`make eval-val`** mAP50 (all 7,669 val tiles, `filter_empty_gt=false`, rotated IoU ≥ 0.50). Training-time periodic mAP may be higher (non-empty tiles only).

| Model | Config | Hub slug | eval-val mAP50 |
|-------|--------|----------|----------------|
| Oriented R-CNN 3× | [oriented_rcnn/dota_le90_3x.json](oriented_rcnn/dota_le90_3x.json) | `oriented_rcnn_dota_le90_3x` | 79.40% |
| Rotated RetinaNet 3× | [rotated_retinanet/dota_le90_3x.json](rotated_retinanet/dota_le90_3x.json) | `rotated_retinanet_dota_le90_3x` | 71.52% |
| Rotated Faster R-CNN 3× | [rotated_faster_rcnn/dota_le90_3x.json](rotated_faster_rcnn/dota_le90_3x.json) | `rotated_faster_rcnn_dota_le90_3x` | 83.46% |
| Rotated FCOS 3× rIoU | [rotated_fcos/dota_le90_3x.json](rotated_fcos/dota_le90_3x.json) | `rotated_fcos_dota_le90_3x` | 82.32% |

## HRSC2016 pretrained models

**Training: ImageSets trainval.** **Evaluation: ImageSets test** (453 images). Whole-image `keep_ratio` + pad-32. FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN 3× use random rotate p=0.5 ±20°; Oriented R-CNN / Faster R-CNN 1× leave rotate off.

| Model | Config | Hub slug | eval-val mAP50 |
|-------|--------|----------|----------------|
| Oriented R-CNN 3× | [oriented_rcnn/hrsc2016_le90_3x.json](oriented_rcnn/hrsc2016_le90_3x.json) | `oriented_rcnn_hrsc2016_le90_3x` | 90.41% |
| Rotated Faster R-CNN 3× | [rotated_faster_rcnn/hrsc2016_le90_3x.json](rotated_faster_rcnn/hrsc2016_le90_3x.json) | `rotated_faster_rcnn_hrsc2016_le90_3x` | 88.77% |
| Rotated FCOS 3× rIoU | [rotated_fcos/hrsc2016_le90_3x.json](rotated_fcos/hrsc2016_le90_3x.json) | `rotated_fcos_hrsc2016_le90_3x` | 88.34% |

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
| `dataset` | data_root, format (dota / airbus_playground / hrsc2016), train_tiles_dir, val_tiles_dir, **train_tiles_dirs**, **val_tiles_dirs** (optional lists; union without on-disk merge), **same_folder** (DOTA: images and .txt in same dir), **overlap** (tile overlap px, even; deploy uses margin = overlap/2), annotations_file, split_file, **val_split_id**, **train_includes_val** (Airbus: train on all folds; val fold for monitoring only), **train_split** / **val_split** (HRSC2016 ImageSets; defaults trainval / test), difficult_strategy, **difficult_tags** (Airbus: exact tags → difficult=1 + strip, e.g. `["Partially Hidden"]`; use with strategy `ignore` for don't-care — not lookalike), **filter_empty_gt** (drop tiles/images with no GT after filters; MMRotate parity), **drop_easy_empty_tiles** (with `tile_metrics_csv`: drop train `tp=fp=fn=0` tiles, keep empty+FP for hard-tile oversampling), **class_tile_oversample_classes** / **class_tile_oversample_factor** / **class_tile_oversample_min_count** (optional GT class-presence oversampling; composes with hard-tile weights), max_train_samples, max_val_samples, **max_samples_shuffle_seed** (deterministic spread when capping), allowed_classes, ignore_labels, map_labels, **lookalike_labels** (optional aliases; reserved name `lookalike` is always a hard-negative token, never a class — map confusers with `map_labels`, e.g. `{"Confuser":"lookalike"}`; see [data.md — Lookalike confusers](../docs/user-guide/data.md#lookalike-confusers)) |
| `data_loader` | batch_size, num_workers, shuffle, pin_memory |
| `model` | backbone, fpn_*, anchor_*, target_means/stds, roi_* (loss, batch, iou, schedule, **roi_proj_xy**), rpn_*, use_hbb_for_matching, add_gt_as_proposals, **rpn_nms_threshold** (proposal NMS), **final_nms_iou_threshold** / **final_nms_iou_schedule_*** (post–ROI-head NMS), **final_nms_use_cpu** (exact polygon final NMS on CPU), nms_class_agnostic, max_detections_per_image, inference_pre_nms_score_threshold |
| `training` | **lr_scheduler_type** first, then lr_scheduler_*, lr_warmup_steps, lr_scaling_*, use_lr_param_groups, lr_mult_*, then num_epochs, learning_rate, momentum, weight_decay, use_amp, gradient_accumulation_steps, max_grad_norm, loss_weights. See [Training — Learning rate scheduling](../docs/user-guide/training.md#learning-rate-scheduling) and [_base_/schedules/README.md](_base_/schedules/README.md). |
| `evaluation` | score_threshold (train val), **`preds_score_threshold`** for **`odet preds` / `make eval-val`** (null → **0.05**), iou_threshold, compute_map_*; **`final_nms_iou_threshold`** for eval-val NMS only (DOTA+HRSC: **0.1**, MMRotate test parity) |
| `production` | Optional overrides: **train val / deploy score** — `score_threshold` overrides `evaluation.score_threshold` when set (`odet preds` ignores it). DOTA 3× Hub recipes set this to eval-val global F1 − **0.05** (Oriented R-CNN **0.7**, Faster R-CNN **0.6**, RetinaNet **0.45**, FCOS **0.2**); HRSC 3× Hub recipes do the same (Oriented R-CNN / Faster R-CNN **0.85**, FCOS **0.2**). `per_class_score_threshold` merges on top of `evaluation` (mAP IoU is `evaluation.iou_threshold` only). **Decode (deploy / `image_demo`)** — RPN/NMS/threshold fields patch the loaded model via `apply_inference_config_to_model` (DOTA+HRSC: final NMS **0.3**). **`odet preds`** prefers `evaluation.final_nms_iou_threshold` when set. Not applied during `tools/train.py` (training uses `model.*`, final NMS **0.1**). **Deploy / tiling** — `overlap_pixels`, `ignore_margin_pixels`, canvas flags (see **config.schema.json**). |
| `checkpoint` | load_from_checkpoint, load_from_experiment, **discover_previous_run**, resume_from_checkpoint_epoch, load_optimizer_state, load_scheduler_state, load_include_prefixes, load_exclude_prefixes, start_epoch, best_metric, higher_is_better |
| `loss` | loss_type (`focal_weighted` now also scales FCOS/RetinaNet sigmoid focal per class; `background_weight` is ROI-only), class_weight_method, background_weight, focal_alpha, focal_gamma, label_smoothing, **roi_grouped_ce_*** (coarse-to-fine ROI classifier curriculum in one run) |
| `preprocessing` | resize_mode, target_size, normalize_mean, normalize_std, pad_size_divisor, enable_flip_horizontal, enable_flip_vertical, enable_flip_diagonal, enable_random_rotate, random_rotate_prob, random_rotate_angle_range |
| Top-level | **enable_albumentation**, enable_profiling |
| `augmentation` | Albumentations params (when enable_albumentation is true) |
| `tensorboard` | log_debug_anchors_proposals, vis_score_threshold |

## PyPI package (`oriented_det.configs`)

A subset of this tree is copied into **`oriented_det/configs/`** for wheels (see **`oriented_det/configs/vendored_manifest.txt`**), including Oriented R-CNN, Faster R-CNN, RetinaNet, and Rotated FCOS DOTA recipes. After editing files here:

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
