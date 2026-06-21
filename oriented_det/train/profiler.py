"""Training profiler utilities for performance analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import torch
    from torch.profiler import (
        ProfilerActivity,
        profile,
        record_function,
        schedule as profiler_schedule,
        tensorboard_trace_handler,
    )
except ImportError:
    torch = None  # type: ignore
    ProfilerActivity = None  # type: ignore
    profile = None  # type: ignore
    record_function = None  # type: ignore
    profiler_schedule = None  # type: ignore
    tensorboard_trace_handler = None  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for profiling utilities.")


class TrainingProfiler:
    """Context manager for profiling PyTorch training.
    
    This profiler helps identify bottlenecks in training by tracking:
    - GPU kernel execution time
    - CPU operations
    - Memory usage
    - Data loading time
    - Model forward/backward pass time
    
    Example:
        ```python
        from oriented_det.train.profiler import TrainingProfiler
        
        profiler = TrainingProfiler(
            log_dir="runs/profiling",
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            profile_memory=True,
        )
        
        with profiler:
            # Your training code here
            for batch in train_loader:
                loss = model(batch)
                loss.backward()
                optimizer.step()
        
        # View results in TensorBoard:
        # tensorboard --logdir runs/profiling
        ```
    """
    
    def __init__(
        self,
        log_dir: str | Path,
        activities: Optional[list] = None,
        schedule: Optional[Any] = None,
        on_trace_ready: Optional[Any] = None,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        experimental_config: Optional[Any] = None,
    ):
        """Initialize training profiler.
        
        Args:
            log_dir: Directory to save profiling traces (for TensorBoard)
            activities: List of activities to profile (default: [CUDA, CPU])
            schedule: Profiling schedule (default: warmup 1, active 2, repeat 1)
            on_trace_ready: Callback when trace is ready (default: TensorBoard handler)
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record stack traces
            with_flops: Whether to estimate FLOPs (requires torch >= 1.12)
            experimental_config: Experimental profiler configuration
        """
        _require_torch()
        
        if activities is None:
            activities = []
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            activities.append(ProfilerActivity.CPU)
        
        if schedule is None and profiler_schedule is not None:
            # Default schedule: warmup 1 step, active 2 steps, repeat 1 time
            schedule = profiler_schedule(
                wait=1,  # Skip first step (warmup)
                warmup=1,  # Warmup for 1 step
                active=2,  # Profile for 2 steps
                repeat=1,  # Repeat 1 time
            )
        
        if on_trace_ready is None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            on_trace_ready = tensorboard_trace_handler(str(log_dir))
        
        self.log_dir = Path(log_dir)
        self.profiler = profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            experimental_config=experimental_config,
        )
    
    def __enter__(self):
        """Enter profiling context."""
        self.profiler.__enter__()
        return self
    
    def __exit__(self, *args):
        """Exit profiling context."""
        return self.profiler.__exit__(*args)
    
    def step(self):
        """Advance profiler to next step."""
        self.profiler.step()
    
    def export_chrome_trace(self, path: str | Path):
        """Export trace to Chrome trace format.
        
        Args:
            path: Path to save trace file (.json)
        """
        self.profiler.export_chrome_trace(str(path))
    
    def export_stacks(self, path: str | Path, metric: str = "self_cuda_time_total"):
        """Export stack traces.
        
        Args:
            path: Path to save stacks file (.txt)
            metric: Metric to sort by (e.g., "self_cuda_time_total", "self_cpu_time_total")
        """
        self.profiler.export_stacks(str(path), metric=metric)
    
    def key_averages(self, group_by_input_shapes: bool = False, group_by_stack_n: int = 0):
        """Get key averages from profiler.
        
        Args:
            group_by_input_shapes: Group by input tensor shapes
            group_by_stack_n: Group by stack trace (n levels)
        
        Returns:
            EventList with averaged events
        """
        return self.profiler.key_averages(
            group_by_input_shapes=group_by_input_shapes,
            group_by_stack_n=group_by_stack_n,
        )
    
    def print_summary(self, sort_by: str = "cuda_time_total", row_limit: int = 20):
        """Print profiling summary.
        
        Args:
            sort_by: Metric to sort by (e.g., "cuda_time_total", "cpu_time_total", "self_cuda_time_total")
            row_limit: Maximum number of rows to print
        """
        events = self.key_averages()
        print(f"\n{'='*80}")
        print(f"Profiling Summary (sorted by {sort_by})")
        print(f"{'='*80}")
        print(events.table(sort_by=sort_by, row_limit=row_limit))
        print(f"{'='*80}\n")


def profile_training_step(
    model: Any,
    images: Any,
    targets: Any,
    optimizer: Any,
    device: torch.device,
    use_amp: bool = False,
    profiler: Optional[TrainingProfiler] = None,
) -> dict[str, float]:
    """Profile a single training step.
    
    Args:
        model: Model to train
        images: Input images
        targets: Target annotations
        optimizer: Optimizer
        device: Device to train on
        use_amp: Use automatic mixed precision
        profiler: Optional profiler context manager
    
    Returns:
        Dictionary with timing information
    """
    _require_torch()
    
    import time
    
    timings = {}
    
    # Data loading time (if applicable)
    data_start = time.time()
    if isinstance(images, list):
        images = [img.to(device) if torch.is_tensor(img) else img for img in images]
    elif torch.is_tensor(images):
        images = images.to(device)
    timings["data_to_device"] = time.time() - data_start
    
    # Forward pass
    forward_start = time.time()
    if profiler is not None:
        with record_function("forward_pass"):
            if use_amp:
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        loss_dict = model(images, targets)
                except (ImportError, TypeError):
                    from torch.cuda.amp import autocast
                    with autocast():
                        loss_dict = model(images, targets)
            else:
                loss_dict = model(images, targets)
    else:
        if use_amp:
            try:
                from torch.amp import autocast
                with autocast('cuda'):
                    loss_dict = model(images, targets)
            except (ImportError, TypeError):
                from torch.cuda.amp import autocast
                with autocast():
                    loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)
    timings["forward"] = time.time() - forward_start
    
    # Loss computation
    loss_start = time.time()
    total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
    timings["loss_computation"] = time.time() - loss_start
    
    # Backward pass
    backward_start = time.time()
    if profiler is not None:
        with record_function("backward_pass"):
            total_loss.backward()
    else:
        total_loss.backward()
    timings["backward"] = time.time() - backward_start
    
    # Optimizer step
    optimizer_start = time.time()
    if profiler is not None:
        with record_function("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()
    timings["optimizer"] = time.time() - optimizer_start
    
    timings["total"] = sum(timings.values())
    
    return timings


__all__ = [
    "TrainingProfiler",
    "profile_training_step",
]

