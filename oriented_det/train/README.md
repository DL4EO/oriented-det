# Training engine

Training loop, evaluation, checkpointing, and **`TrainingExperimentConfig`**. Full JSON reference: [Configuration](../../docs/user-guide/configuration.md). User guide: [Training](../../docs/user-guide/training.md).

## Source provenance (`git_commit`, …)

When training starts, rank 0 stamps the saved `config.json` and prints to `train.log`:

- **`git_commit`** — full hash of the framework source tree (the installed `oriented_det` package root, not `ORIENTED_DET_PROJECT_ROOT`)
- **`git_describe`** — `git describe --dirty --always --tags`
- **`git_dirty`** — uncommitted changes at launch
- **`git_branch`**, **`git_commit_date`**, **`package_version`**, **`source_code_root`**

Use these fields to compare runs when the recipe JSON is unchanged but code moved forward (e.g. architecture parity commits). Recipe files do not set these keys; they are runtime-only like `experiment_timestamp` and `source_recipe`.

To reconstruct provenance for an old run without stamps, align `train.log` start time with `git rev-list -1 --before='<started_at>' HEAD` on the framework repo.

## RPN anchor priors

Anchor-based detectors (`RotatedFasterRCNN`, `OrientedRCNN`, `RotatedRetinaNet`) default to **horizontal RPN priors** (`theta = 0`), aligned with MMRotate-style generators. **`model.anchor_angles` is not a config field**—if present in JSON, strict loading rejects it as an unknown `model` key. Advanced use: pass **`anchor_angles=...`** only when constructing a model in Python (see `oriented_det/models/README.md`).

`RotatedFCOS` is anchor-free (`model_type: rotated_fcos`) and does not use RPN anchors.

## FCOS box regression (`model.box_reg_loss_type`, `model.aux_loss_type`)

- **`box_reg_loss_type`**: `l1` (default), `kfiou`, or `riou` (decoded differentiable polygon IoU). DOTA Hub FCOS is 3× rIoU ([`dota_le90_3x.json`](../../configs/rotated_fcos/dota_le90_3x.json), lr **2.5e-3**); L1 + KFIoU aux remains published as `rotated_fcos_dota_le90_3x_kfiou_aux`.
- **`aux_loss_type`** / **`aux_loss_weight`**: decoded `kfiou` or `probiou` on positives (centerness-weighted). Weight **0** disables. Typical **0.1**. Aux `riou` is rejected. Logged as `loss_box_reg_aux`. Aux is Gaussian overlap plus an aspect-gated heading term (`ω sin²(2Δθ)`); **`aux_angle_weight`** (default **1.0**, **0** disables the heading term) and **`aux_angle_lambda`** (default **1.0**) control it.

Recipes: [`configs/rotated_fcos/dota_le90_1x.json`](../../configs/rotated_fcos/dota_le90_1x.json), [`dota_le90_3x.json`](../../configs/rotated_fcos/dota_le90_3x.json), [`dota_le90_1x_l1_kfiou_aux.json`](../../configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json).

## Debug mode (`--debug`)

When training with `--debug` (or `make train DEBUG=1`), extra logs are printed and written to TensorBoard to help diagnose training/eval issues (data, losses, matching, mAP):

- **Config summary**: Dataset sizes, eval score/iou thresholds, classes, mAP frequency.
- **First batch**: GT counts per image (min/max/mean) and per-class counts to spot data/annotation issues.
- **Each epoch**: Full loss breakdown (all `loss_*` components) and current learning rate.
- **TensorBoard**: RPN/ROI metrics (e.g. `roi_num_pos`, `roi_match_rate`; for Rotated Faster R-CNN these reflect **RPN-only** proposals, not the training pool after GT append) and anchor/proposal visualizations when `log_debug_anchors_proposals` is enabled. Validation prediction images (`val/predictions`) overlay the source **filename** (basename with extension) at the top-left when the dataloader provides `image_filename` (DOTA train script collate does).
- **After mAP**: Per-class AP, detections per class, ground truths per class, GT cover rates (pre/post score threshold), detection score stats, and optionally **GT–IoU diagnostics** when `evaluation.extended_gt_metrics` is true (mean/median of each GT’s best IoU against all raw detections vs same-class dets only, **per-class mean best IoU table**, counts of GTs with **0% best IoU** (no overlap with any raw det), high IoU but wrong class vs no box above the eval IoU threshold, histograms of best IoU into `[0,0.25)`, `[0.25,0.5)`, `[0.5,0.75)`, `[0.75,1]`, and **class-agnostic duplicate rate** on score-thresholded boxes: sort by score, greedy “keep”; a box is duplicate if its IoU with any kept higher-score box is ≥ the eval IoU threshold—reports micro and macro duplicate rates). Leave `extended_gt_metrics` false (default) for faster validation.

**Score thresholds:** `evaluation.score_threshold` is the global minimum confidence for metrics. Optional `evaluation.per_class_score_threshold` maps class names to per-class minimum scores (post-NMS); classes not listed use the global value.

**Rotated IoU for mAP / GT cover:** `evaluation.use_exact_rotated_iou` (default `true`) selects exact CPU polygon IoU vs GPU sampling IoU for periodic mAP and GT-cover / accuracy matching. It does **not** change final detection NMS or `production.*` decode settings. For a fast in-training mAP with an exact publishable number at the end, set `use_exact_rotated_iou: false` and `use_exact_rotated_iou_for_final_map: true` with `compute_map_final: true`.

## Production section (`production`)

**Legacy experiment configs:** obsolete `model.gpu_oriented_iou_samples` (removed June 2026; sample count is now env-driven via `ORIENTED_DET_GPU_*` in `ops/README.md`) is stripped on load so older runs under `runs/` still work with `odet preds` / checkpoint reload.

Optional root section **`production`** holds overrides for:

- **Training validation / mAP** — **`effective_eval_metric_thresholds`** merges **`evaluation`** with **`production`**: **`score_threshold`** uses **`production.score_threshold`** when set, else **`evaluation.score_threshold`**; **`iou_threshold`** always comes from **`evaluation`**; **`per_class_score_threshold`** merges like deploy (evaluation entries, then production overrides). **`tools/train.py`** passes the merged triple into `train()` / `evaluate()`. Training forward NMS uses **`model.final_nms_iou_threshold`** (not `production`). **`evaluation.use_exact_rotated_iou`** controls mAP matching IoU only.

- **Final detection NMS split** — Recipes ship **`model.final_nms_iou_threshold: 0.1`** (train val / MMRotate-style), **`production.final_nms_iou_threshold: 0.3`** (deploy / `image_demo`), and **`evaluation.final_nms_iou_threshold: 0.1`** for **`odet preds` / `make eval-val`** published protocol. Resolver: **`resolve_preds_final_nms_iou_threshold`** (CLI → `evaluation` → production-patched model).

- **Decode / NMS on the loaded checkpoint model** — Non-null **`production.inference_pre_nms_score_threshold`**, **`final_nms_iou_threshold`**, **`final_nms_use_cpu`**, **`max_detections_per_image`**, **`nms_class_agnostic`**, **`roi_inference_top_class_only`**, **`rpn_pre_nms_top_n`**, **`rpn_post_nms_top_n`**, **`rpn_nms_threshold`** are applied by **`apply_inference_config_to_model`** in **`tools/save_predictions.load_model_from_checkpoint`** (deploy, **`image_demo`**, **`test_single_image`**, and as a base before eval-val may override NMS). **`tools/train.py` does not call this**.

- **Deploy / `image_demo`** — Global score uses **`resolve_inference_score_threshold`** (**`production.score_threshold`** else **`evaluation.score_threshold`**). Per-class floors use the same merge as validation (**`evaluation`** then **`production`**). Sliding-window overlap defaults to **200 px** per axis unless **`production.overlap_pixels`** is set (**`resolve_inference_sliding_window_overlap_pixels`**). Edge margin from **`production.ignore_margin_pixels`** when set, else **`dataset.overlap`/2**. Canvas: **`production.stick_to_model_canvas`** (default **true** when null), **`production.use_first_image_canvas`** (default **false** when null).

Any field **`null`** or omitted keeps the usual **`evaluation`** / **`model`** / env source.

Use these logs to compare schedule (epochs, LR), loss balance, and proposal/ROI behavior with MMRotate or other baselines.

## Epoch timing, ETA, and mAP duration

After each epoch (rank 0), the engine prints **wall-clock** time for that epoch, a **rolling average** over completed epochs, and an **ETA** until the scheduled end of training. Durations of **24 hours or more** are shown as ``Xd Yh Zm`` (days, hours, minutes) instead of only hours. The ETA uses the rolling average and includes an extra one-epoch allowance when **`evaluation.compute_map_final`** is enabled (approximate cost of the post-training mAP pass on the best checkpoint). The same line summarizes **config-derived context**: training/validation batch counts, **`gradient_accumulation_steps`**, current epoch index, and **`mAP every N epochs`** when `compute_map_every_n_epochs > 0`.

When mAP is computed inside `evaluate()`, the log line **`mAP computation completed in …`** reports only the **oriented mAP / AP** phase (`compute_oriented_map`), not the full validation forward pass.

## Start/end timestamps and final timing summary

At the beginning of `train()`, rank 0 prints **`Training started at: …`** (wall-clock ISO 8601 in the local timezone). After the full run—including optional **final mAP** on the best checkpoint—rank 0 prints a **Training timing summary** block: start and finish times, **total wall time**, and **per-epoch** mean / min / max when epoch timings exist. The returned `history` dict includes a **`timing`** entry with the same structured fields (`started_at`, `finished_at`, `total_wall_seconds`, and when applicable `epoch_wall_times_seconds`, `epochs_completed`, `mean_epoch_seconds`, `min_epoch_seconds`, `max_epoch_seconds`) for scripts or downstream logging.

## TensorBoard `train/learning_rate`

When optimizer **per-module param groups** are enabled (`training.use_lr_param_groups`), the first group is usually the backbone with a reduced LR. The scalar `train/learning_rate` is the **config-scale reference LR** (same scale as `training.learning_rate` after batch/DDP scaling), not the backbone group’s LR alone.

**Per-milestone drops:** `training.lr_scheduler_gamma` is a number (same factor every drop) or a list of the same length as `lr_scheduler_milestones`. Oriented R-CNN HRSC 3× uses `[0.1, 0.5]`.

**Cosine phase:** `training.lr_scheduler_cosine_epochs` is PyTorch `T_max` (else `num_epochs`). Legacy `lr_scheduler_cosine_t_max` still loads with a deprecation warning.

**LR multipliers:** `lr_mult_backbone` applies to `backbone.*`. **`lr_mult_head`** (optional): when set, it applies to RetinaNet `head.*` and, for two-stage models, to both `rpn_head.*` and `roi_head.*` (same value for RPN and ROI, overriding `lr_mult_rpn` / `lr_mult_roi` for those tensors). When `lr_mult_head` is unset (`null`), `head.*` uses `lr_mult_other`, and RPN/ROI use `lr_mult_rpn` / `lr_mult_roi`. Any other trainable parameter uses `lr_mult_other`.

## Product repo layout (`ORIENTED_DET_PROJECT_ROOT`)

Optional: set **`ORIENTED_DET_PROJECT_ROOT`** to an external project directory when invoking `odet train` / `make train`. Training then writes **`runs/<model_type>/<timestamp>/`** there instead of under the oriented-det install tree.

**Pretrained Hub weights** (`pretrained/*.pth` in configs) are resolved under the **oriented-det install** (`<install>/pretrained/`), not under the product repo. Use **`ORIENTED_DET_PRETRAINED_DIR`** to override the cache location. See [oriented_det/pretrained/README.md](../pretrained/README.md).

## ROI grouped CE curriculum (`loss.roi_grouped_ce_*`)

Single-run alternative to multi-stage training (e.g. DOTA → `plane` → 6 classes): keep the full classifier head, but for early epochs optimize **coarse groups** instead of fine subtype labels.

- **`loss.roi_grouped_ce_enabled`**: turn on grouped cross-entropy for ROI classification (requires `loss_type` `cross_entropy` or `class_weighted`).
- **`loss.roi_grouped_ce_groups`**: map group name → list of class names, e.g. `{"plane": ["Bomber Aircraft", "Fighter Aircraft", ...]}`. Each class may appear in at most one group; unlisted classes always use fine CE.
- **`loss.roi_grouped_ce_schedule_type`**: `step` (default) — fully grouped until `schedule_end_epoch`, then fine CE; `linear_ramp` — blend grouped→fine between `schedule_start_epoch` and `schedule_end_epoch` (`schedule_power` controls easing).
- Background (label 0) always uses standard cross-entropy.

The training loop calls `set_grouped_ce_alpha_for_epoch(epoch)` each epoch (same hook point as class-weight schedules). Focal loss ignores grouped CE (use CE / class_weighted).


## ROI angle weight schedule (`model.roi_box_reg_angle_schedule_*`)

`model.roi_box_reg_angle_weight` scales the angle (5th encoded dim) SmoothL1 term in ROI box regression. An optional piecewise schedule ramps that weight by 0-based epoch (same convention as `roi_box_reg_aux_schedule_*`).

- **`model.roi_box_reg_angle_schedule_epochs`**: epoch boundaries.
- **`model.roi_box_reg_angle_schedule_values`**: weight per segment; length = `len(epochs) + 1`. When either field is null, `roi_box_reg_angle_weight` stays constant.

Each training epoch, the engine calls `set_roi_box_reg_angle_weight_for_epoch(epoch)` on two-stage models.

## ROI auxiliary box-reg weight schedule (`model.roi_box_reg_aux_schedule_*`)

When `model.roi_box_reg_aux_weight` > 0, an optional piecewise schedule can change the auxiliary term during training (decoded rIoU/KFIoU/ProbIoU when main is Smooth L1; encoded Smooth L1 when main is decoded).

- **`model.roi_box_reg_aux_schedule_epochs`**: 0-based epoch boundaries (same convention as `freeze_backbone_epochs` and `final_nms_iou_schedule_epochs`).
- **`model.roi_box_reg_aux_schedule_values`**: weight per segment; length = `len(epochs) + 1`. When either schedule field is null, `roi_box_reg_aux_weight` stays constant.

Each training epoch, the engine calls `set_roi_box_reg_aux_weight_for_epoch(epoch)` on two-stage models (or `set_box_reg_aux_weight_for_epoch` on RetinaNet).

## ROI main regression loss (`model.roi_box_reg_main_loss_type`, Rotated Faster R-CNN)

- **`smooth_l1`** (default): encoded Smooth L1 primary; optional decoded aux via `roi_box_reg_aux_weight` / `roi_box_reg_aux_loss_type`.
- **`probiou` / `riou` / `kfiou`**: decoded primary on positive RoIs; optional encoded Smooth L1 aux via `roi_box_reg_aux_weight` with `roi_box_reg_aux_loss_type: smooth_l1`.
- **`roi_box_reg_norm`**: `sampled_all` (MMRotate avg_factor) or `positives_only` (per-dim mean over positives). Default `sampled_all` preserves existing DOTA recipes.

Example (ProbIoU main + Smooth L1 aux, see [`dota_le90_1x.json`](../../configs/rotated_faster_rcnn/dota_le90_1x.json)):

```json
"model": {
  "roi_box_reg_main_loss_type": "probiou",
  "roi_box_reg_probiou_mode": "l1",
  "roi_box_reg_aux_weight": 0.1,
  "roi_box_reg_aux_loss_type": "smooth_l1",
  "roi_box_reg_norm": "positives_only"
}
```

Example (piecewise ROI IoU aux weight):

```json
"model": {
  "roi_box_reg_aux_weight": 0.1,
  "roi_box_reg_aux_schedule_epochs": [24, 28],
  "roi_box_reg_aux_schedule_values": [0.1, 0.05, 0.0]
}
```

Epochs 0–23 → **0.1**, 24–27 → **0.05**, 28+ → **0.0** (display epochs 1–24 / 25–28 / 29+).

## Partial freeze: backbone vs RPN (`training.freeze_backbone_epochs`, `training.freeze_rpn_epochs`)

Two independent thresholds (0-based loop epoch, same as checkpoint `epoch`): `freeze_backbone_epochs` freezes `backbone.*`; `freeze_rpn_epochs` freezes `rpn_head.*` on two-stage models. The ROI head always trains. Helpers: `set_backbone_requires_grad`, `set_rpn_requires_grad`, `model_has_rpn_head` in `oriented_det/train/utils.py`. With `use_lr_param_groups: true` and either threshold \(>0\), `tools/train.py` builds optimizer param groups with `include_frozen_parameters=True` so frozen weights stay in the optimizer until unfrozen.

**DDP:** Multi-GPU training wraps the model with `find_unused_parameters=True` whenever either freeze threshold is \(>0\) (frozen parameters skip the loss graph; `False` would break the reducer). At the first loop epoch where `epoch >= max(freeze_backbone_epochs, freeze_rpn_epochs)`, the engine **re-wraps** the same inner module with `find_unused_parameters=False` so later epochs avoid the extra autograd traversal and PyTorch warnings.

## Capping train/val size

`max_train_samples` / `max_val_samples` default to the **first N** samples in dataset order (often many tiles from the same source image). Set **`dataset.max_samples_shuffle_seed`** to an integer (e.g. `42`) to instead take **N indices from a deterministic shuffle** of the full split (same cap, spread across the listing; train and val each use the same seed with independent permutations).

When **`dataset.tile_metrics_csv`** is set, **`dataset.drop_easy_empty_tiles: true`** removes vacuous train tiles (`tp=fp=fn=0`) **before** this cap, so `max_train_samples` counts remaining tiles (including empty tiles with false positives). Hard-tile oversampling runs after the cap.

## Checkpoint and Resume Behavior

- Standard checkpoints store model, optimizer, and LR scheduler state.
- **Checkpoint loading is config-only** (`checkpoint.*` in JSON). There are no `--resume` / `--from-scratch` / `--from-last-experiment` flags on `tools/train.py`.
- **`checkpoint.load_from_checkpoint`**: optional path to a `.pth` file (highest precedence when the file exists).
- **`checkpoint.load_from_experiment`**: optional directory under `runs/<model_type>/…` whose `checkpoints/` folder is used.
- **`checkpoint.discover_previous_run`** (default **false**): when **true**, if neither an existing checkpoint file nor `load_from_experiment` is set, training sets `load_from_experiment` to the **newest other** run directory (same ordering as before). When **false**, `null` paths mean **no weights loaded** from runs (backbone still follows `model.pretrained_backbone`).
- **`checkpoint.resume_from_checkpoint_epoch`**: when loading from an experiment directory, **true** prefers **latest** `checkpoint_epoch_*.pth` and continues epochs with optimizer/scheduler per **`load_optimizer_state`** / **`load_scheduler_state`**; **false** prefers **best** `best_*` checkpoint and typically fine-tunes from epoch 0 unless your config says otherwise.
- **`checkpoint.load_scheduler_state`** (default **true** when resuming): set **`false`** with **`lr_warmup_steps: 0`** to **rebuild** the LR schedule after load (see `tools/train.py`). Use **`lr_scheduler_type: cosine_annealing`** for PyTorch `CosineAnnealingLR` (restarts if `num_epochs` > `T_max`), or **`cosine_annealing_with_tail`** for cosine + fixed `tail_lr` (`cosine_epochs` + `tail_epochs` = `num_epochs`).
- **Class-count mismatch on load:** Tensors with shape mismatch (e.g. `fc_cls` when changing class count) are skipped; use `checkpoint.load_include_prefixes` / `load_exclude_prefixes` to control partial loads.
- Gradient accumulation flushes any final partial window at epoch end, so tail microbatches are no longer dropped when the number of batches is not divisible by `gradient_accumulation_steps`.

## Final mAP on the best checkpoint (`evaluation.compute_map_final`)

When `compute_map_final` is true (default), after the training loop finishes the engine loads **`checkpoint_manager`’s best checkpoint** (same metric as training, e.g. mAP) and runs one full validation with mAP on rank 0. Under DDP, the resolved best path is broadcast so all ranks still run `evaluate()` and join validation collectives.

## Multi-GPU validation (`evaluate` + DDP)

With DDP, every rank runs the validation forward pass (same `DistributedSampler` shard as training). After the loop, detections and ground truths are merged on rank 0 with `torch.distributed.gather_object`.

That collective uses a **Gloo** process group (CPU) instead of the default **NCCL** group. NCCL can raise `RuntimeError: NCCL Error 1: unhandled cuda error` when gathering large pickled payloads or when a latent async CUDA error surfaces at the next collective. `torch.cuda.synchronize()` and `dist.barrier()` run immediately before the gather so GPU work is finished and ranks are aligned.

If `gather_object` still fails, training logs a warning and falls back to **per-rank shard** metrics on rank 0 (mAP is then incomplete vs. the full validation set).

After merge and mAP on rank 0, **all ranks** synchronize on the same Gloo group before leaving `evaluate()`. That avoids the previous failure mode where non-zero ranks blocked on `train()`’s barrier while rank 0 was still inside `evaluate()` for longer than Gloo’s default **30 minute** collective timeout (`Timed out waiting 1800000ms`).

Timeouts: the Gloo gather group is created with a **24 hour** default; `tools/train.py` uses **`TORCH_DIST_TIMEOUT_SECONDS`** (default `86400`) for `init_process_group` so NCCL/Gloo defaults are not stuck at 30 minutes on long periodic mAP.

## Training run health checklist (tmux-friendly)

Use this quick checklist while a long multi-GPU run is active in `tmux`.

### 1) Process is alive and log is moving

- `tmux ls` shows your training session.
- `tail -f runs/<model>/<timestamp>/train.log` continues printing batch/epoch lines.
- If the log is frozen for much longer than one normal epoch, check GPU/process state before assuming convergence.

### 2) Core learning signals look sane

- **Train loss** trends down over epochs (small oscillations are normal).
- **ROI hints** show rising positive matches / match-rate over early epochs.
- No repeated `RuntimeError`, `CUDA out of memory`, or distributed collectives failures.

### 3) Validation recall does not collapse

- Track `GT cover rate pre-eval-threshold` and `GT cover rate post-eval-threshold`.
- These matching metrics (and `Accuracy` / `Correct Predictions`) follow the **mAP schedule** (`compute_map_every_n_epochs`): on other epochs the log prints `Accuracy / GT cover: (skipped — computed on mAP epochs)` instead of misleading zeros.
- Healthy early behavior is usually upward or stable after warmup; a sustained drop across several epochs is a warning.
- `GT lost by eval-threshold filtering` should not continuously grow; that often means thresholds are too strict or confidence is collapsing.

### 4) mAP checkpoints (periodic)

- Compare only epochs where mAP is computed (`compute_map_every_n_epochs`). Logged mAP deltas use the **last computed** mAP (skipped epochs do not reset the baseline to `-1`).
- Expect some noise, but the medium-term trend should improve or stabilize.
- If mAP plateaus early while loss still falls, inspect class imbalance, score thresholds, and duplicate suppression.

### 5) Precision/duplication sanity

- Watch `Avg Detections per Image` and duplicate metrics when `extended_gt_metrics` is enabled.
- Very high detections/image with low matched accuracy often indicates too many low-quality boxes (proposal/NMS/score-threshold issue).
- Rising "wrong-class overlap" with good IoU suggests classification confusion rather than localization failure.

### 6) Decision checkpoints (practical triage)

- **Continue** when loss decreases and mAP/GT coverage improve at each eval checkpoint.
- **Tune thresholds / NMS** when recall is good but precision is poor (many duplicates, many low-score false positives).
- **Tune assignment/anchors/loss** when GT coverage stays low (many missed GTs at IoU threshold).
- **Stop and debug** on repeated runtime/distributed errors, NaN/Inf losses, or hard metric collapse for multiple eval cycles.
