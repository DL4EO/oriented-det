# Training API Reference

::: oriented_det.train
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true

## Important Notes

### Training Features

The training engine provides efficient features:

- **Mixed Precision Training**: Automatic FP16 support via `use_amp=True`
- **Gradient Accumulation**: Train with large effective batch sizes
- **Checkpointing**: Automatic saving with best model tracking
- **Metric Tracking**: Built-in metric aggregation and reporting
- **TensorBoard Logging**: Optional TensorBoard integration for visualization
- **Robust Error Handling**: Graceful recovery from batch errors

### Loss Components

For two-stage detectors like `OrientedRCNN`, there are four main loss components:

1. **`loss_objectness`** (RPN): Binary classification (object vs. background)
   - Expected: 0.5-0.7 early training → 0.1-0.3 converged

2. **`loss_rpn_box_reg`** (RPN): Box regression for refining anchors into proposals
   - Expected: 0.8-1.5 early training → 0.3-0.6 converged

3. **`loss_classifier`** (ROI): Multi-class classification for object classes
   - Expected: 0.5-1.0 early training → 0.2-0.4 converged

4. **`loss_box_reg`** (ROI): Box regression for refining proposals into final boxes
   - Expected: 0.8-1.5 early training → 0.2-0.5 converged

### Memory Optimization

For memory-efficient training with `OrientedRCNN`:

- Use `roi_chunk_size` parameter to process ROIs in chunks (default: 32)
- Enable `roi_use_checkpoint` for gradient checkpointing (~2x less memory, ~30% slower)
- Recommended: `roi_chunk_size=16` with checkpointing for 8-16GB GPUs

## Examples

### Basic Training

```python
from oriented_det.train import train_one_epoch, train
from oriented_det.train import MetricTracker, CheckpointManager

# Single epoch
metrics = train_one_epoch(
    model=model,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # Automatic mixed precision
    gradient_accumulation_steps=4,
)

# Full training loop
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
    max_grad_norm=1.0,  # Gradient clipping
)
```

### TensorBoard Logging

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

### Learning rate schedulers

The training loop accepts any PyTorch-style LR scheduler. When using **ReduceLROnPlateau**, pass `lr_scheduler_plateau_metric` to `train()` so the engine calls `scheduler.step(metric_value)` with the correct validation metric (e.g. `"total_loss"` or `"mAP"`).

When using **`tools/train.py`** with a JSON config, set `training.lr_scheduler_type` to one of:

- **`multistep` / `step`** (default) — MultiStepLR or StepLR; optional warmup via `lr_warmup_steps`.
- **`reduce_on_plateau`** — ReduceLROnPlateau; configure `lr_scheduler_plateau_metric`, `lr_scheduler_plateau_factor`, `lr_scheduler_plateau_patience`.
- **`one_cycle`** — OneCycleLR (stepped every optimizer step); configure `lr_scheduler_one_cycle_*` options.
- **`cosine_annealing` / `cosine`** — PyTorch CosineAnnealingLR (`lr_scheduler_cosine_epochs`; legacy `lr_scheduler_cosine_t_max` remaps).
- **`cosine_annealing_with_tail` / `cosine_with_tail`** — Cosine then fixed `lr_scheduler_cosine_tail_lr`.

See the [Training user guide — Learning rate scheduling](../user-guide/training.md#learning-rate-scheduling) for config examples and usage.

### Performance Profiling

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

