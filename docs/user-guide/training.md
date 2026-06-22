# Training Module

The training module provides efficient training loops with modern PyTorch features.

## Overview

The `oriented_det.train` module provides:

- **`train_one_epoch`** - Single epoch training
- **`train`** - Full training loop
- **`evaluate`** - Validation/evaluation
- **`CheckpointManager`** - Automatic checkpointing
- **`MetricTracker`** - Metric aggregation

## Features

- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Automatic checkpointing
- ✅ Metric tracking
- ✅ Robust error handling

## Basic Training

### Single Epoch

```python
from oriented_det.train import train_one_epoch
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)

metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    epoch=0,
    use_amp=True,
    gradient_accumulation_steps=4,
)
```

### Full Training Loop

```python
from oriented_det.train import train, CheckpointManager

checkpoint_manager = CheckpointManager(
    "checkpoints/",
    best_metric="mAP",
    higher_is_better=True
)

history = train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=12,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
    use_amp=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
)
```

## Checkpointing

### CheckpointManager

```python
from oriented_det.train import CheckpointManager

manager = CheckpointManager(
    save_dir="checkpoints/",
    best_metric="mAP",
    higher_is_better=True,
    save_freq=1  # Save every epoch
)

# Save checkpoint
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=5,
    metrics={"mAP": 0.75}
)

# Load checkpoint
state = manager.load_checkpoint("checkpoints/best_model.pth")
model.load_state_dict(state["model"])
```

## Metric Tracking

### MetricTracker

```python
from oriented_det.train import MetricTracker

tracker = MetricTracker()

# Update metrics
tracker.update({"loss": 0.5, "mAP": 0.75})

# Get averages
avg_metrics = tracker.average()
print(f"Average loss: {avg_metrics['loss']}")

# Reset
tracker.reset()
```

## Advanced Options

### Mixed Precision

```python
metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # Enable automatic mixed precision
)
```

**Benefits:**
- ~2x faster training
- Lower memory usage
- No accuracy loss observed in practice

### Gradient Accumulation

Train with large effective batch sizes:

```python
metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * batch_size
)
```

**Use Case:** When GPU memory is limited but you want a larger effective batch size for stable training.

### Gradient Clipping

Prevent exploding gradients:

```python
metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    max_grad_norm=1.0,  # Clip gradients to norm 1.0
)
```

**When to Use:** If losses are oscillating wildly or training is unstable.

### Loss Weighting

Weight different loss components:

```python
metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    loss_weights={
        "loss_objectness": 1.0,
        "loss_rpn_box_reg": 1.0,
        "loss_classifier": 1.0,
        "loss_box_reg": 1.0,
    },
)
```

**Note:** All losses have equal weight (1.0) by default. Curriculum learning (gradual classifier weight increase) was removed because it caused models to collapse to single-class predictions in early training.

### TensorBoard Logging

TensorBoard logging is integrated on top of the existing `MetricTracker`. To enable it, pass a `SummaryWriter` to the training functions:

```python
from torch.utils.tensorboard import SummaryWriter
from oriented_det.train import train

# Create TensorBoard writer
writer = SummaryWriter(log_dir="runs/experiment_1")

# Train with TensorBoard logging
history = train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=12,
    val_loader=val_loader,
    writer=writer,  # Enable TensorBoard logging
)

# View logs with: tensorboard --logdir runs/experiment_1
```

**Logged Metrics:**
- **Per-step metrics** (during training): `train/{loss_name}`, `train/learning_rate`
- **Per-epoch metrics** (after each epoch): `train_epoch/{metric_name}`, `val/{metric_name}`
- **Prediction images** (during validation): `val/predictions` - Shows up to 4 validation images with predictions (red boxes) and ground truth (green boxes) overlaid; the source image **filename** (basename with extension) is drawn at the top-left when the dataloader provides it

### Performance Profiling

Use PyTorch Profiler to identify training bottlenecks and optimize performance:

```python
from oriented_det.train.profiler import TrainingProfiler
from torch.profiler import ProfilerActivity, schedule

# Create profiler
profiler = TrainingProfiler(
    log_dir="runs/profiling",
    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
)

# Profile a few training steps
model.train()
with profiler:
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5:  # Profile first 5 batches
            break
        # ... training code ...
        profiler.step()

# Print summary
profiler.print_summary(sort_by="cuda_time_total", row_limit=30)

# View detailed trace in TensorBoard
# tensorboard --logdir runs/profiling
```

**What to Look For:**
- **Data loading bottlenecks**: If `DataLoader` operations take significant time, increase `num_workers`
- **GPU utilization**: If GPU kernels are idle, increase `batch_size` or enable mixed precision
- **Memory issues**: Check memory profiling to identify memory-intensive operations
- **Slow operations**: Look for operations with high `cuda_time_total` or `cpu_time_total`

**Common Optimizations:**
- Increase `num_workers` if data loading is slow (try 4-8 workers)
- Increase `batch_size` if GPU utilization is low (<80%)
- Use mixed precision (AMP) if not already enabled
- Profile memory to identify memory bottlenecks
- Use `pin_memory=True` in DataLoader for faster CPU→GPU transfers

## Learning Rate Scheduling

### WarmupScheduler

The `WarmupScheduler` provides learning rate warmup that gradually increases the learning rate from 0 to the base learning rate over a specified number of optimizer steps. This helps stabilize training in the early stages, especially when using large learning rates or training from scratch.

**When to use warmup:**
- Training from scratch (no pretrained weights)
- Using large learning rates (>0.01)
- Training is unstable in early epochs
- Fine-tuning with large learning rate changes

**Basic usage:**

```python
from oriented_det.train import WarmupScheduler
from torch.optim import SGD, lr_scheduler

# Create optimizer with base learning rate
optimizer = SGD(model.parameters(), lr=0.01)

# Create base scheduler (e.g., StepLR)
base_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

# Wrap with warmup scheduler
warmup_scheduler = WarmupScheduler(
    optimizer=optimizer,
    base_scheduler=base_scheduler,
    warmup_steps=500,  # Warmup for 500 optimizer steps
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        # ... training step ...
        optimizer.step()
        warmup_scheduler.step()  # Step per optimizer step during warmup
    
    warmup_scheduler.step_epoch()  # Step base scheduler per epoch
```

**How it works:**
- During warmup: Learning rate increases linearly from 0 to base LR over `warmup_steps`
- After warmup: Uses the base scheduler (e.g., StepLR, CosineAnnealingLR)
- `step()`: Called after each `optimizer.step()` during warmup
- `step_epoch()`: Called after each epoch to advance the base scheduler

**Example with CosineAnnealingLR:**

```python
from oriented_det.train import WarmupScheduler
from torch.optim import AdamW, lr_scheduler

optimizer = AdamW(model.parameters(), lr=1e-4)

# Cosine annealing scheduler
base_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

# Warmup for 5% of total steps
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.05 * total_steps)  # 5% warmup

warmup_scheduler = WarmupScheduler(
    optimizer=optimizer,
    base_scheduler=base_scheduler,
    warmup_steps=warmup_steps,
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        # ... training step ...
        optimizer.step()
        warmup_scheduler.step()  # Per-step during warmup
    
    warmup_scheduler.step_epoch()  # Per-epoch for base scheduler
```

**Calculating warmup steps:**

A common approach is to use 5-10% of total training steps:

```python
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.05 * total_steps)  # 5% warmup
# Or
warmup_steps = int(0.10 * total_steps)  # 10% warmup
```

For example:
- 1000 batches per epoch × 12 epochs = 12,000 total steps
- 5% warmup = 600 steps (~0.6 epochs)
- 10% warmup = 1,200 steps (~1.2 epochs)

**Best practices:**
- Start with 5% warmup (conservative)
- Increase to 10% if training is still unstable
- Use warmup when learning rate > 0.01
- Monitor learning rate curve in TensorBoard to verify warmup behavior
- After warmup, base scheduler takes over (e.g., StepLR, CosineAnnealingLR)

**Visualization:**

The learning rate schedule looks like:
```
LR
↑
|     ╱───────╲        ← Cosine annealing (after warmup)
|    ╱         ╲
|   ╱           ╲
|  ╱             ╲
| ╱               ╲
|╱                 ╲
|───────────────────→ Steps
  ↑
  Warmup (linear increase)
```

### Scheduler types (config-driven)

When using `tools/train.py` with a JSON config, you can choose the LR scheduler via `training.lr_scheduler_type`. The following types are supported in addition to the default **MultiStepLR** / **StepLR** (with optional warmup):

| `lr_scheduler_type`    | Scheduler             | Stepping        | Typical use |
|-------------------------|----------------------|-----------------|-------------|
| (none) / `multistep` / `step` | MultiStepLR or StepLR | Once per epoch  | Default; use `lr_scheduler_milestones` for MultiStepLR. |
| `reduce_on_plateau`     | ReduceLROnPlateau    | Once per epoch (after validation) | Reduce LR when a metric (e.g. loss or mAP) plateaus. |
| `one_cycle` / `onecycle`| OneCycleLR           | Every optimizer step | Single cycle: warmup then decay over full training. |
| `cosine_annealing` / `cosine` | PyTorch CosineAnnealingLR | Once per epoch  | Cosine decay; SGDR restarts if `num_epochs` > `T_max`. |
| `cosine_annealing_with_tail` / `cosine_with_tail` | CosineAnnealingWithFixedTailLR | Once per epoch | Cosine phase then constant `tail_lr`; `cosine_epochs` + `tail_epochs` = `num_epochs`. |

### ReduceLROnPlateau

Reduces the learning rate when the monitored metric stops improving. Useful when you don’t know a good step schedule in advance.

**Config (JSON):**

```json
"training": {
  "lr_scheduler_type": "reduce_on_plateau",
  "lr_scheduler_plateau_metric": "mAP",
  "lr_scheduler_plateau_factor": 0.5,
  "lr_scheduler_plateau_patience": 4
}
```

- **`lr_scheduler_plateau_metric`**: `"total_loss"` (minimize) or `"mAP"` (maximize). The engine uses this to call `scheduler.step(metric_value)` after each validation.
- **`lr_scheduler_plateau_factor`**: New LR = current LR × factor when the metric plateaus.
- **`lr_scheduler_plateau_patience`**: Number of epochs without improvement before reducing LR.

Warmup is not applied with ReduceLROnPlateau; the scheduler drives LR entirely.

### OneCycleLR

OneCycleLR updates the learning rate every optimizer step: a linear warmup phase, then a cosine or linear decay over the rest of training. Total steps = `num_epochs × (batches_per_epoch // gradient_accumulation_steps)`.

**Config (JSON):**

```json
"training": {
  "lr_scheduler_type": "one_cycle",
  "lr_scheduler_one_cycle_pct_start": 0.3,
  "lr_scheduler_one_cycle_div_factor": 25.0,
  "lr_scheduler_one_cycle_final_div_factor": 1e4
}
```

- **`pct_start`**: Fraction of total steps used for warmup (e.g. 0.3 = 30%).
- **`div_factor`**: Initial LR = `max_lr / div_factor`.
- **`final_div_factor`**: Minimum LR = initial_lr / final_div_factor.

No separate warmup config is used; OneCycleLR has its own warmup. The training engine steps this scheduler every optimizer step via an internal wrapper.

### CosineAnnealingLR (`cosine_annealing`)

Standard PyTorch cosine decay. `T_max` from `lr_scheduler_cosine_epochs`, `lr_scheduler_cosine_t_max`, or `num_epochs`. If **`num_epochs` > `T_max`**, the LR **restarts** after each minimum (SGDR-style). Tail keys are ignored.

```json
"training": {
  "lr_scheduler_type": "cosine_annealing",
  "lr_scheduler_cosine_t_max": 72,
  "lr_scheduler_cosine_eta_min": 1e-6,
  "lr_warmup_steps": 50
}
```

### Cosine + fixed tail (`cosine_annealing_with_tail`)

Cosine for `cosine_epochs` (or `t_max`), then constant `lr_scheduler_cosine_tail_lr` for `tail_epochs`; the two phases must sum to `num_epochs`.

```json
"training": {
  "lr_scheduler_type": "cosine_annealing_with_tail",
  "lr_scheduler_cosine_epochs": 20,
  "lr_scheduler_cosine_tail_epochs": 4,
  "lr_scheduler_cosine_eta_min": 1e-5,
  "lr_scheduler_cosine_tail_lr": 5e-5
}
```

When using the training CLI, set `training.lr_scheduler_type` and the corresponding keys in your config (or in a base schedule under `configs/_base_/schedules/`). See `configs/_base_/schedules/README.md` for schedule examples.

## Configuration

<a id="nested-config-inheritance"></a>

JSON experiment configs drive `tools/train.py`. The canonical reference is **[Configuration](configuration.md)** (all sections, `model_type` values, recipe catalog, `evaluation` vs `production`, CLI flags).

Quick load:

```python
from pathlib import Path
from oriented_det.train.config import TrainingExperimentConfig

config = TrainingExperimentConfig.load(
    Path("configs/oriented_rcnn/dota_le90_1x.json")
)
config.print_summary()
```

`tools/train.py` builds the model from `config.model_type` (`oriented_rcnn`, `rotated_faster_rcnn`, or `rotated_retinanet`) and writes `runs/<model_type>/<timestamp>/config.json`.

## Understanding Loss Components

For two-stage detectors like `OrientedRCNN`, there are four main loss components:

### Stage 1: Region Proposal Network (RPN) Losses

**`loss_objectness`** - Classification loss for the RPN
- Predicts whether an anchor contains an object (foreground) or not (background)
- Binary classification: object vs. background
- Helps the model learn where objects are located in the image

**`loss_rpn_box_reg`** - Box regression loss for the RPN
- Refines anchor boxes to better match ground truth objects
- Operates on **anchors** (predefined boxes at different scales/ratios/angles)
- Learns to transform anchors into better object proposals
- Uses Smooth L1 loss on `[cx, cy, w, h, angle]` parameters

### Stage 2: ROI Head Losses

**`loss_classifier`** - Classification loss for the ROI head
- Predicts the specific class of each proposal (e.g., "plane", "ship", "small-vehicle")
- Multi-class classification across all object classes
- Only computed on positive proposals (IoU ≥ threshold with ground truth)

**`loss_box_reg`** - Box regression loss for the ROI head
- Refines proposals into final, precise bounding boxes
- Operates on **proposals** (output from RPN stage)
- Provides fine-grained localization refinement
- Uses Smooth L1 loss on `[cx, cy, w, h, angle]` parameters

### Key Differences: `loss_rpn_box_reg` vs `loss_box_reg`

| Aspect | `loss_rpn_box_reg` | `loss_box_reg` |
|--------|-------------------|----------------|
| **Stage** | Stage 1 (RPN) | Stage 2 (ROI) |
| **Input** | Anchors (predefined) | Proposals (from RPN) |
| **Purpose** | Generate object proposals | Refine proposals to final boxes |
| **IoU Threshold** | 0.5 (positive anchors) | 0.4 (positive proposals) |
| **Sampling** | 256 anchors per image | 512 proposals per image |
| **Precision** | Coarse (finding objects) | Fine (accurate localization) |

### Training Flow

```
Image → Backbone → FPN Features
                        ↓
                   RPN Head
                        ↓
        [loss_objectness + loss_rpn_box_reg]
                        ↓
              Oriented Proposals
                        ↓
                   ROI Align
                        ↓
                   ROI Head
                        ↓
        [loss_classifier + loss_box_reg]
                        ↓
              Final Detections
```

Both box regression losses use the same loss function (Smooth L1), but they operate at different stages:
- **RPN stage**: Learns to find and roughly localize objects
- **ROI stage**: Learns to classify and precisely localize objects

## Evaluation

```python
from oriented_det.train import evaluate

metrics = evaluate(
    model=model,
    data_loader=val_loader,
    device=device,
    eval_compute_map_final=True,  # After training, load best checkpoint and compute mAP once
)
```

## Best Practices

### Data Loading

- Use `num_workers=4-8` for faster data loading
- Enable `pin_memory=True` for faster CPU→GPU transfers
- Monitor data loading time in profiler

### Mixed Precision

- Always use `use_amp=True` for training (2x faster, lower memory)
- No accuracy loss observed in practice

### Gradient Accumulation

- Use for large effective batch sizes without increasing memory
- Example: `batch_size=2` with `gradient_accumulation_steps=4` = effective batch size of 8

### Monitoring

- Watch for `sampled_fg` hitting 128 cap (foreground bottleneck in RPN)
- Check loss component trends (all should decrease steadily)
- Monitor GPU utilization (should be >80% during training)

### Memory Optimization

For `OrientedRCNN` with limited GPU memory:

- Use `roi_chunk_size=16` with `roi_use_checkpoint=True` for 8-16GB GPUs
- Use `roi_chunk_size=32` with `roi_use_checkpoint=False` for 24GB+ GPUs
- Monitor memory usage during training to find optimal settings

## Complete Example

```python
import torch
from oriented_det import OrientedRCNN
from oriented_det.data import build_dota_loader
from oriented_det.train import train, CheckpointManager
from torch.optim import AdamW

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrientedRCNN(num_classes=15).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Data
train_loader = build_dota_loader("/path/to/dota", split="train", batch_size=4)
val_loader = build_dota_loader("/path/to/dota", split="val", batch_size=4)

# Checkpointing
checkpoint_manager = CheckpointManager("checkpoints/", best_metric="mAP")

# Train
history = train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=12,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
    use_amp=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
)
```

## See Also

- [API Reference](../api/train.md) - Complete API documentation
- [Models Guide](models.md) - Model architectures

