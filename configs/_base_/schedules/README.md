# Schedules (LR and epochs)

This folder holds base schedule configs that set `training.num_epochs` and LR scheduler settings. They are meant to be included via `_base_` (e.g. `"_base_": ["../_base_/schedules/3x.json"]`). Augmentation is defined in **[_base_/augmentation.json](../augmentation.json)**; do not duplicate it in schedule files.

## Schedule files

- **1x.json** — 12 epochs, StepLR/MultiStepLR milestones `[8, 11]`
- **3x.json** — 36 epochs, milestones `[24, 33]`
- **6x.json** — extends 3x with more epochs (see file)

Keep this folder limited to generic schedule bases. Dataset-specific settings belong in `configs/_base_/datasets/`, and AMP is composed separately with `configs/_base_/fp16.json`.

When no `lr_scheduler_type` is set, the default is **MultiStepLR** (if `lr_scheduler_milestones` is set) or **StepLR** (using `lr_scheduler_step_epochs` and `lr_scheduler_gamma`). Optional **warmup** is controlled by `lr_warmup_steps`.

## LR scheduler types

You can override the scheduler by setting `training.lr_scheduler_type` in your config.

| `lr_scheduler_type`   | Description |
|------------------------|-------------|
| (none) / `multistep` / `step` | **MultiStepLR** or **StepLR** (default). Use `lr_scheduler_milestones` for MultiStepLR, else StepLR. Supports `lr_warmup_steps`. |
| `reduce_on_plateau`    | **ReduceLROnPlateau**. Reduces LR when the monitored metric stops improving. Use `lr_scheduler_plateau_metric` (`"total_loss"` → minimize, `"mAP"` → maximize), `lr_scheduler_plateau_factor`, `lr_scheduler_plateau_patience`. No warmup. |
| `one_cycle` / `onecycle` | **OneCycleLR**. LR is updated every optimizer step (warmup then decay). Uses `lr_scheduler_one_cycle_pct_start`, `lr_scheduler_one_cycle_div_factor`, `lr_scheduler_one_cycle_final_div_factor`. No warmup (built-in). |
| `cosine_annealing` / `cosine` | **PyTorch CosineAnnealingLR**. `T_max` from `lr_scheduler_cosine_epochs` or `lr_scheduler_cosine_t_max` (else `num_epochs`). If `num_epochs` > `T_max`, LR **restarts** after each minimum (SGDR-style). Tail keys are ignored. |
| `cosine_annealing_with_tail` / `cosine_with_tail` | **CosineAnnealingWithFixedTailLR**: cosine for `cosine_epochs` (or `t_max`), then constant `tail_lr` for `tail_epochs` (`cosine + tail` must equal `num_epochs`). |

### Example: cosine 20 epochs + fixed tail 4 epochs

```json
"training": {
  "num_epochs": 24,
  "learning_rate": 0.0025,
  "lr_scheduler_type": "cosine_annealing_with_tail",
  "lr_scheduler_cosine_epochs": 20,
  "lr_scheduler_cosine_tail_epochs": 4,
  "lr_scheduler_cosine_eta_min": 1e-5,
  "lr_scheduler_cosine_tail_lr": 5e-5,
  "lr_warmup_steps": 500
}
```

### Example: ReduceLROnPlateau (monitor mAP)

```json
"training": {
  "num_epochs": 72,
  "learning_rate": 0.05,
  "lr_scheduler_type": "reduce_on_plateau",
  "lr_scheduler_plateau_metric": "mAP",
  "lr_scheduler_plateau_factor": 0.5,
  "lr_scheduler_plateau_patience": 4
}
```

### Example: OneCycleLR

```json
"training": {
  "num_epochs": 36,
  "learning_rate": 0.05,
  "lr_scheduler_type": "one_cycle",
  "lr_scheduler_one_cycle_pct_start": 0.3,
  "lr_scheduler_one_cycle_div_factor": 25.0,
  "lr_scheduler_one_cycle_final_div_factor": 1e4
}
```

### Example: Cosine annealing

```json
"training": {
  "num_epochs": 72,
  "learning_rate": 0.05,
  "lr_scheduler_type": "cosine_annealing",
  "lr_scheduler_cosine_eta_min": 1e-6,
  "lr_warmup_steps": 50
}
```
