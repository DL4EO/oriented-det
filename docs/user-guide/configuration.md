# Configuration reference

Training and checkpoint inference are driven by **JSON experiment configs** loaded by `TrainingExperimentConfig` (`oriented_det/train/config.py`) and `tools/train.py`. Machine-readable field definitions: `configs/config.schema.json`.

## Overview

| Mechanism | Module | Role |
|-----------|--------|------|
| JSON files + `_base_` | `oriented_det/utils/config.py` — `load_config()` | MMRotate-style inheritance, merge, path resolution |
| Strict dataclasses | `oriented_det/train/config.py` | Unknown keys raise `ValueError` |
| Saved runs | `runs/<model_type>/<timestamp>/config.json` | Written at train start; used by eval/inference tools |

### Inheritance (`_base_`)

Child configs list one or more base paths (relative to the child file):

```json
{
  "_base_": [
    "../_base_/models/oriented_rcnn_r50.json",
    "../_base_/schedules/1x.json"
  ],
  "training": { "learning_rate": 0.002 }
}
```

Bases are loaded recursively and merged in order; the child overrides earlier values.

### Muted keys (`_muted_*`)

Keys prefixed with `_muted_` are stripped before validation so you can keep alternate values in the same file without affecting behavior:

```json
{
  "training": {
    "learning_rate": 0.002,
    "_muted_learning_rate": 0.005
  }
}
```

### CLI overrides (`tools/train.py`)

| Flag | Effect |
|------|--------|
| `--config` | Required path to JSON |
| `--batch-size` | Overrides `data_loader.batch_size` |
| `--use-amp` / `--no-amp` | Overrides `training.use_amp` |
| `--debug` | Extra logging, TensorBoard diagnostics, per-class mAP breakdown |
| `--wizard` | Data/config diagnostics before training |
| `--local-rank` | Set by `torchrun` for DDP |

Distributed training: use `tools/train_multi_gpu.py` with `torchrun`.

## `model_type`

Must be one of the values accepted by `tools/train.py` (also sets `runs/<model_type>/`):

| Value | Detector | Notes |
|-------|----------|--------|
| `oriented_rcnn` | `OrientedRCNN` | Horizontal RPN + midpoint-offset proposals + oriented ROI |
| `rotated_faster_rcnn` | `RotatedFasterRCNN` | Oriented RPN + oriented ROI (MMRotate-style) |
| `rotated_retinanet` | `RotatedRetinaNet` | One-stage; focal head; no RPN/ROI |
| `rotated_fcos` | `RotatedFCOS` | Anchor-free one-stage; centerness; no anchors/RPN/ROI |

Base model fragments: `configs/_base_/models/oriented_rcnn_r50.json`, `rotated_faster_rcnn_r50.json`, `rotated_retinanet_r50.json`, `rotated_fcos_r50.json`.

## Top-level sections

| Section | Purpose |
|---------|---------|
| `model_type` | Which detector `train.py` builds |
| `dataset` | Paths, format, tiling overlap, difficult GT, caps, hard-tile oversampling |
| `preprocessing` | Resize, normalize (MMDet-style RGB mean/std on [0,1]), pad, train flips, optional random rotate |
| `data_loader` | `batch_size`, `num_workers`, `shuffle`, `pin_memory` |
| `model` | Backbone, FPN, anchors, RPN/ROI/NMS, inference thresholds |
| `training` | Epochs, LR, schedulers, AMP, grad accum, freeze phases, early stopping |
| `loss` | ROI loss recipe + class weights (two-stage); RetinaNet / FCOS use focal from `loss` / `model` |
| `evaluation` | Score/IoU thresholds, mAP during training |
| `production` | Deploy/inference overrides (see below) |
| `checkpoint` | Resume, discover prior run, best-metric selection |
| `tensorboard` | Debug anchor/proposal images, viz score floor |
| `enable_albumentation` | Toggle Albumentations pipeline |
| `enable_profiling` | Training step profiler |
| `augmentation` | Albumentations limits/probabilities when enabled |

Saved-only metadata (optional in hand-written configs): `class_map`, `class_names`, `num_classes`, `experiment_timestamp`.

## Section reference

Full key lists, types, and defaults: **`configs/config.schema.json`**. Below: behavior that is easy to misconfigure.

### `dataset`

| Key | Default | Notes |
|-----|---------|--------|
| `format` | `dota` | `dota` → `train_tiles_dir` / `val_tiles_dir`; `airbus_playground` → `annotations_file` + `split_file`; `hrsc2016` → official `FullDataSet` + `ImageSets` under `data_root` |
| `train_split` / `val_split` | null | HRSC2016 ImageSets names for the train/val roles. null → `trainval` / `test` |
| `same_folder` | `false` | If true, images and `.txt` labels live directly under tile dirs |
| `overlap` | `16` | Tile overlap (px, even); `0` = none. Deploy margin defaults to `overlap/2` when `production.ignore_margin_pixels` is null |
| `difficult_strategy` | `drop` | `drop` \| `ignore` \| `keep` for DOTA difficult flag |
| `difficult_tags` | null | Airbus: exact tags (e.g. `["Partially Hidden"]`) → `difficult=1` and strip from class name at load/generation |
| `filter_empty_gt` | `false` | Drop samples with no GT after difficult/class filters (DOTA tiles and HRSC2016 images; MMRotate parity). Don't-care-only / lookalike-only tiles are kept when those boxes remain |
| `max_train_samples` / `max_val_samples` | null | Cap dataset size; use `max_samples_shuffle_seed` for spread sampling |
| `tile_metrics_csv` | null | From `save_predictions --save-tile-metrics-csv`; enables hard-tile oversampling |
| `hard_tile_metric_column` / `hard_tile_threshold` / `hard_tile_oversample_factor` | `f1` / `0.8` / `2.0` | Tiles with metric strictly below the threshold are oversampled |
| `drop_easy_empty_tiles` | `false` | Requires `tile_metrics_csv`. Drops train tiles with `tp=fp=fn=0` (no GT, no preds) before `max_train_samples` and oversampling. Empty tiles with FPs stay and can be oversampled. Keep `filter_empty_gt: false` so those hard empties remain in the loader |
| `annotations_file` / `split_file` | null | Required for `format: airbus_playground` |
| `val_split_id` / `train_includes_val` | `0` / `false` | Airbus fold CSVs |
| `ignore_labels` / `map_labels` / `lookalike_labels` | null | Drop / rename / hard-negative aliases (see [Data Loading](data.md)) |

### `model` (by detector)

**Shared:** `backbone`, `pretrained_backbone`, `frozen_stages` / `trainable_layers`, `fpn_*`, `anchor_scales`, `anchor_ratios`, `target_means` / `target_stds`, `inference_pre_nms_score_threshold`, `final_nms_*`, `max_detections_per_image`, `use_hbb_for_matching`.

| Key | Oriented R-CNN | Rotated Faster R-CNN | RetinaNet |
|-----|----------------|----------------------|-----------|
| `roi_proj_xy` | Yes (MMRotate parity) | Yes (no-op for horizontal RoIs) | — |
| `rpn_min_size` | — | Yes | Reuses `rpn_*` for **anchor assign** thresholds (pos/neg IoU, batch size); not an RPN head |
| `add_gt_as_proposals` | Yes | Yes | N/A |
| RPN IoU thresholds | Midpoint-offset RPN defaults | Standard oriented RPN | See `rpn_positive_iou_threshold`, `rpn_negative_iou_threshold`, … |
| `roi_focal_*`, `roi_norm_factor`, `roi_edge_swap` | ROI head | ROI head | Reused for **RetinaNet focal cls** (`loss.loss_type: focal` wires these in `train.py`) |

**RetinaNet note:** There is no RPN or ROI module. `tools/train.py` maps `model.rpn_*` to oriented-anchor matching and `model.roi_*` / `loss.focal_*` to the classification head. Final NMS is class-aware by default; set `model.nms_class_agnostic: true` (and `production.nms_class_agnostic` if deploy should match) to suppress overlapping lookalike classes (car/truck). See [configs/rotated_retinanet/README.md](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_retinanet/README.md).

**FCOS note:** Anchor-free; ignore `anchor_*` / RPN / ROI keys. Box regression is `model.box_reg_loss_type` (`l1`, `kfiou`, `riou`) plus optional `aux_loss_type` / `aux_loss_weight`. Head knobs are `fcos_*`. Final NMS is class-aware by default; set `model.nms_class_agnostic: true` (and `production.nms_class_agnostic` if deploy should match) to suppress overlapping lookalike classes (car/truck). See [configs/rotated_fcos/README.md](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_fcos/README.md).

`roi_box_reg_angle_weight` scales the angle (5th encoded dim) SmoothL1 term in ROI box regression (two-stage models only). Optional `roi_box_reg_angle_schedule_epochs` / `roi_box_reg_angle_schedule_values` piecewise-schedule that weight by 0-based epoch (`values` length = `len(epochs) + 1`; when either field is null, `roi_box_reg_angle_weight` stays constant). The engine calls `set_roi_box_reg_angle_weight_for_epoch(epoch)` each epoch.

`roi_box_reg_aux_weight` > 0 enables an auxiliary box-reg term that is **not** the primary loss. Set **`roi_box_reg_aux_loss_type`** (`probiou` / `riou` / `kfiou` when main is `smooth_l1`; `smooth_l1` when main is decoded). Optional `roi_box_reg_aux_schedule_epochs` / `roi_box_reg_aux_schedule_values` piecewise-schedule that weight by 0-based epoch. Use `roi_box_reg_kfiou_fun` / `roi_box_reg_probiou_mode` for the decoded metric, whichever side it is on.

For **ProbIoU (or rIoU/KFIoU) as primary** ROI loss on Rotated Faster R-CNN, set `roi_box_reg_main_loss_type` and add encoded Smooth L1 aux with `roi_box_reg_aux_weight` / `roi_box_reg_aux_loss_type: smooth_l1`. Control Smooth L1 scale with `roi_box_reg_norm` (`sampled_all` = MMRotate, `positives_only` = mean over positives). Recipe: [`configs/rotated_faster_rcnn/dota_le90_1x.json`](../../configs/rotated_faster_rcnn/dota_le90_1x.json) (3× via [`dota_le90_3x.json`](../../configs/rotated_faster_rcnn/dota_le90_3x.json)). Legacy keys `roi_box_reg_iou_weight` / `roi_box_reg_smooth_l1_aux_weight` still load with a deprecation warning.

### `training`

- **`lr_scheduler_type`**: `multistep`/`step`, `reduce_on_plateau`, `one_cycle`, `cosine_annealing`, `cosine_annealing_with_tail` — see `configs/_base_/schedules/README.md` and [Training — Learning rate scheduling](training.md#learning-rate-scheduling). **`lr_scheduler_gamma`** is a number (same factor every drop) or a list (one factor per milestone). Cosine phase length is **`lr_scheduler_cosine_epochs`** (legacy `lr_scheduler_cosine_t_max` remaps).
- **`freeze_backbone_epochs` / `freeze_rpn_epochs`**: Freeze modules for early epochs (ROI still trains).
- **`early_stop_*`**: Optional stop when metric plateaus.

### `loss` (two-stage ROI)

| `loss_type` | Behavior |
|-------------|----------|
| `cross_entropy` | Plain ROI CE |
| `class_weighted` | CE + dataset-derived weights (default) |
| `focal` / `focal_weighted` | Focal ROI; weighted variant uses class weights |
| `none` | Legacy: falls back to `model.roi_loss_type` |

### `evaluation` vs `production`

| Context | Score threshold | IoU for mAP | Final detection NMS |
|---------|-----------------|-------------|---------------------|
| Training loop | `evaluation.score_threshold` (often **0.3**); `production.score_threshold` overrides when set | `evaluation.iou_threshold` | **`model.final_nms_iou_threshold`** (recipes: **0.1**) — `production` is **not** applied in `train.py` |
| `odet preds` / `make eval-val` | **`evaluation.preds_score_threshold`** when set, else **0.05** (ignores production/train floors); per-class maps still merge | Always `evaluation.iou_threshold` | **`evaluation.final_nms_iou_threshold`** when set (recipes: **0.1**), else production/model |
| Deploy / `image_demo` | `production.score_threshold` else `evaluation.score_threshold` | n/a | **`production.final_nms_iou_threshold`** (recipes: **0.3**) via `apply_inference_config_to_model` |

`production.overlap_pixels` (default 200 when null) and `ignore_margin_pixels` (default `dataset.overlap / 2`) control native sliding-window inference for `fixed`/`crop` in `oriented_det.runtime.inference` (`odet preds`, `save_predictions`, deploy). `resize_mode: pad` / `keep_ratio` do not native-tile; they use the training whole-image scale path (`keep_ratio` then `pad_size_divisor`).

### `checkpoint`

| Key | Notes |
|-----|--------|
| `discover_previous_run` | Newest run under `runs/<model_type>/` when no explicit load path |
| `resume_from_checkpoint_epoch` | `true` → latest `checkpoint_epoch_*.pth`; `false` → prefer `best_*` |
| `best_metric` | e.g. `mAP` or `total_loss`; pairs with `higher_is_better` |

## Recipe catalog

Top-level configs (inherit bases under `configs/_base_/`):

| Config | Model | Use |
|--------|-------|-----|
| `configs/oriented_rcnn/dota_le90_1x.json` | Oriented R-CNN | **Default** `make train`; 1× DOTA pretrain (full recipe) |
| `configs/oriented_rcnn/dota_le90_3x.json` | Oriented R-CNN | 3× pretrain (inherits 1×) |
| `configs/rotated_faster_rcnn/dota_le90_1x.json` | Rotated Faster R-CNN | **1× DOTA pretrain** (full recipe) |
| `configs/rotated_faster_rcnn/dota_le90_3x.json` | Rotated Faster R-CNN | 3× pretrain (inherits 1×) |
| `configs/rotated_faster_rcnn/hrsc2016_le90_1x.json` | Rotated Faster R-CNN | 1× HRSC2016 (keep-ratio, rotate off) |
| `configs/rotated_faster_rcnn/hrsc2016_le90_3x.json` | Rotated Faster R-CNN | 3× HRSC2016 (inherits 1×, ±20° rotate) |
| `configs/rotated_retinanet/dota_le90_1x.json` | RetinaNet | **1× DOTA pretrain** (full recipe) |
| `configs/rotated_retinanet/dota_le90_3x.json` | RetinaNet | 3× DOTA pretrain (inherits 1×) |
| `configs/rotated_fcos/dota_le90_1x.json` | Rotated FCOS | 1× DOTA decoded rIoU |
| `configs/rotated_fcos/dota_le90_3x.json` | Rotated FCOS | **Hub** 3× decoded rIoU |
| `configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json` | Rotated FCOS | 1× L1 + KFIoU aux |
| `configs/rotated_fcos/hrsc2016_le90_1x.json` | Rotated FCOS | 1× HRSC2016 rIoU |
| `configs/rotated_fcos/hrsc2016_le90_3x.json` | Rotated FCOS | 3× HRSC2016 rIoU |

**Bases (not run directly):** `configs/_base_/datasets/`, `configs/_base_/schedules/{1x,3x,6x}.json`, `fp16`, `preprocessing`, `augmentation`.

Layout and muted-key examples: see repo `configs/README.md`.

## Programmatic use

```python
from pathlib import Path
from oriented_det.train.config import TrainingExperimentConfig

config = TrainingExperimentConfig.load(Path("configs/oriented_rcnn/dota_le90_1x.json"))
config.print_summary()
config.save(Path("my_experiment.json"))
```

For the training engine (AMP, checkpoints, schedulers), see [Training](training.md).

## See also

- [Training](training.md) — `train()`, `CheckpointManager`, LR schedules
- [Data Loading](data.md) — DOTA and Airbus loaders
- [Models](models.md) — detector APIs and loss components
- [Utilities](utils.md) — `load_config`, `FrozenConfig`, dotted overrides
