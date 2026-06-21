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
| Rotated RetinaNet 1× | [rotated_retinanet/dota_le90_1x.json](rotated_retinanet/dota_le90_1x.json) | `rotated_retinanet_dota_le90_1x` | 64.14% |
| Rotated RetinaNet 3× | [rotated_retinanet/dota_le90_3x.json](rotated_retinanet/dota_le90_3x.json) | `rotated_retinanet_dota_le90_3x` | 71.52% |
| Rotated Faster R-CNN 1× | [rotated_faster_rcnn/dota_le90_1x.json](rotated_faster_rcnn/dota_le90_1x.json) | — | TBD |
| Rotated Faster R-CNN 3× | [rotated_faster_rcnn/dota_le90_3x.json](rotated_faster_rcnn/dota_le90_3x.json) | `rotated_faster_rcnn_dota_le90_3x` | 76.41% |
| Oriented R-CNN 1× | [oriented_rcnn/dota_le90_1x.json](oriented_rcnn/dota_le90_1x.json) | `oriented_rcnn_dota_le90_1x` | 74.79% |
| Oriented R-CNN 3× | [oriented_rcnn/dota_le90_3x.json](oriented_rcnn/dota_le90_3x.json) | — | TBD |

Download: `odet pretrained download <slug>` or `"load_from_checkpoint": "hf://<slug>"`.

| Prefix | Resolves under | Override env |
|--------|----------------|--------------|
| `@odet:` | `ORIENTED_DET_ROOT` (repo root) | `ORIENTED_DET_ROOT`, or installed package location |

Example in an external project that keeps local dataset fragments:

```json
{
  "_base_": [
    "my_dataset.json",
    "@odet:configs/_base_/models/rotated_faster_rcnn_r50.json",
    "@odet:configs/_base_/schedules/3x.json"
  ]
}
```

## Documentation

- **[Configuration reference](../docs/user-guide/configuration.md)** — human-readable guide: sections, recipes, CLI, `production` vs `evaluation`
- **[config.schema.json](config.schema.json)** — machine-readable types and defaults (kept in sync with `oriented_det/train/config.py`)

## Layout

- **[_base_/](_base_)** — Dataset, model, schedule, FP16, preprocessing, augmentation fragments (do not run directly; included via `_base_`).
- **[oriented_rcnn/](oriented_rcnn/)** — Oriented R-CNN (horizontal RPN + midpoint-offset + oriented ROI). See [oriented_rcnn/README.md](oriented_rcnn/README.md).
- **[rotated_faster_rcnn/](rotated_faster_rcnn/)** — Rotated Faster R-CNN. 1× baseline: `dota_le90_1x.json`; default `make train`: `dota_le90_3x.json`. See [rotated_faster_rcnn/README.md](rotated_faster_rcnn/README.md).
- **[rotated_retinanet/](rotated_retinanet/)** — Rotated RetinaNet (one-stage). See [rotated_retinanet/README.md](rotated_retinanet/README.md).

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
| `model_type` | `oriented_rcnn`, `rotated_faster_rcnn`, or `rotated_retinanet` |
| `dataset` | data_root, format (dota / airbus_playground), train_tiles_dir, val_tiles_dir, **train_tiles_dirs**, **val_tiles_dirs** (optional lists; union without on-disk merge), **same_folder** (DOTA: images and .txt in same dir), **overlap** (tile overlap px, even; deploy uses margin = overlap/2), annotations_file, split_file, difficult_strategy, **filter_empty_gt** (DOTA: drop tiles with no GT after filters; MMRotate parity), max_train_samples, max_val_samples, **max_samples_shuffle_seed** (deterministic spread when capping), allowed_classes, ignore_labels, map_labels |
| `data_loader` | batch_size, num_workers, shuffle, pin_memory |
| `model` | backbone, fpn_*, anchor_*, target_means/stds, roi_* (loss, batch, iou, schedule), rpn_*, use_hbb_for_matching, add_gt_as_proposals, **rpn_nms_threshold** (proposal NMS), **final_nms_iou_threshold** / **final_nms_iou_schedule_*** (post–ROI-head NMS), **final_nms_use_cpu** (exact polygon final NMS on CPU), nms_class_agnostic, max_detections_per_image, inference_pre_nms_score_threshold |
| `training` | **lr_scheduler_type** first, then lr_scheduler_*, lr_warmup_steps, lr_scaling_*, use_lr_param_groups, lr_mult_*, then num_epochs, learning_rate, momentum, weight_decay, use_amp, gradient_accumulation_steps, max_grad_norm, loss_weights. See [Training — Learning rate scheduling](../docs/user-guide/training.md#learning-rate-scheduling) and [_base_/schedules/README.md](_base_/schedules/README.md). |
| `evaluation` | score_threshold, iou_threshold, compute_map_final, compute_map_every_n_epochs |
| `production` | Optional overrides: **val/mAP** — `score_threshold` overrides `evaluation` when set; `per_class_score_threshold` merges on top of `evaluation` (mAP IoU is `evaluation.iou_threshold` only). **Decode (checkpoint inference only)** — RPN/NMS/threshold fields patch the loaded model via `apply_inference_config_to_model` in `load_model_from_checkpoint` (deploy, `save_predictions`, `image_demo`); not applied during `tools/train.py` (training uses `model.*` only). **Deploy / tiling** — `overlap_pixels`, `ignore_margin_pixels`, canvas flags (see **config.schema.json**). |
| `checkpoint` | load_from_checkpoint, load_from_experiment, **discover_previous_run**, resume_from_checkpoint_epoch, load_optimizer_state, load_scheduler_state, load_include_prefixes, load_exclude_prefixes, start_epoch, best_metric, higher_is_better |
| `loss` | loss_type, class_weight_method, background_weight, focal_alpha, focal_gamma, label_smoothing, **roi_grouped_ce_*** (coarse-to-fine ROI classifier curriculum in one run) |
| `preprocessing` | resize_mode, target_size, normalize_mean, normalize_std, pad_size_divisor, enable_flip_horizontal, enable_flip_vertical, enable_flip_diagonal |
| Top-level | **enable_albumentation**, enable_profiling |
| `augmentation` | Albumentations params (when enable_albumentation is true) |
| `tensorboard` | log_debug_anchors_proposals, vis_score_threshold |

## PyPI package (`oriented_det.configs`)

A subset of this tree is copied into **`oriented_det/configs/`** for wheels (see **`oriented_det/configs/vendored_manifest.txt`**), including Rotated RetinaNet and Rotated Faster R-CNN DOTA recipes. After editing files here:

```bash
make sync-configs    # update vendored copy
make check-configs   # verify before commit (also runs in CI)
```

## Training

From the repo root:

```bash
python tools/train.py --config configs/rotated_faster_rcnn/dota_le90_3x.json
```

Override batch size or AMP: `--batch-size`, `--use-amp`, `--no-amp`. See [tools/README.md](../tools/README.md) and the [main README](../README.md).
