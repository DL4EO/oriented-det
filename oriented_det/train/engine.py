"""Efficient and robust training engine for oriented detection."""

from __future__ import annotations

import math
import random
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    # Try new API first, fall back to deprecated API
    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast
    # Import distributed for DDP synchronization
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None  # type: ignore
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    GradScaler = None  # type: ignore
    autocast = None  # type: ignore
    DataLoader = None  # type: ignore
    dist = None  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None  # type: ignore
    np = None  # type: ignore

from .utils import (
    CheckpointManager,
    MetricTracker,
    filter_detections_by_score_threshold,
    model_has_rpn_head,
    scores_labels_pass_threshold,
    set_backbone_requires_grad,
    set_rpn_requires_grad,
)

try:
    from .profiler import TrainingProfiler
except ImportError:
    TrainingProfiler = None  # type: ignore

try:
    from ..data.evaluation import (
        Detection,
        GroundTruth,
        _aggregate_gt_best_iou_samples,
        compute_oriented_map,
        detection_matches_ground_truth_class,
        format_gt_best_iou_alignment_table_from_dict,
        gt_best_iou_alignment_metrics_to_dict,
    )
    from ..geometry import RBox
    from ..utils import viz
except ImportError:
    Detection = None  # type: ignore
    GroundTruth = None  # type: ignore
    compute_oriented_map = None  # type: ignore
    RBox = None  # type: ignore
    viz = None  # type: ignore


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for training engine.")


def _format_duration_hms(seconds: float) -> str:
    """Format a duration in seconds for logs (e.g. ``45.2s``, ``3m 12s``, ``2h 15m``, ``2d 16h 33m``)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 90:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    if total >= 86400:  # 24 hours
        d, rem = divmod(total, 86400)
        h, rem = divmod(rem, 3600)
        m, _ = divmod(rem, 60)
        return f"{d}d {h}h {m}m"
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


# gather_object pickles detections/GTs and uses a collective; the default NCCL group can raise
# "NCCL Error 1: unhandled cuda error" on large payloads or when surfacing async GPU errors.
# A Gloo group carries CPU-side buffers and avoids that NCCL path.
_gloo_gather_object_pg = None

# Gloo default collective timeout is 30 minutes; rank 0 can exceed that during full-val
# merge + mAP while other ranks wait at a barrier — use a long timeout for this CPU group.
_GLOO_GATHER_TIMEOUT = timedelta(hours=24)


def _get_gloo_gather_object_group():
    global _gloo_gather_object_pg
    if dist is None or not dist.is_initialized():
        return None
    if _gloo_gather_object_pg is None:
        try:
            _gloo_gather_object_pg = dist.new_group(
                backend="gloo", timeout=_GLOO_GATHER_TIMEOUT
            )
        except TypeError:
            # PyTorch without timeout= on new_group
            _gloo_gather_object_pg = dist.new_group(backend="gloo")
    return _gloo_gather_object_pg


# ImageNet normalization constants (default for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB


def _optimizer_reference_learning_rate(optimizer: Any) -> float:
    """Learning rate at config scale (`training.learning_rate`), not necessarily param group 0.

    When `tools/train.py` uses per-module param groups, group 0 is typically backbone with a
    smaller LR; schedulers still scale all groups together, so dividing group 0's LR by its
    stored ``lr_multiplier`` recovers the reference base LR for logging.
    """
    pg0 = optimizer.param_groups[0]
    mult = pg0.get("lr_multiplier")
    if mult is not None and mult > 0:
        return float(pg0["lr"]) / float(mult)
    return float(pg0["lr"])


def _format_effective_lrs(optimizer: Any, lr_scheduler: Any = None) -> str:
    """Human-readable LR string for epoch logs (reference + per-group effective LRs)."""
    try:
        sched_lrs = list(lr_scheduler.get_last_lr()) if lr_scheduler is not None else None
    except Exception:
        sched_lrs = None

    parts: list[str] = []
    for i, pg in enumerate(getattr(optimizer, "param_groups", []) or []):
        name = str(pg.get("group_name", i))
        if sched_lrs is not None and i < len(sched_lrs):
            lr_val = float(sched_lrs[i])
        else:
            lr_val = float(pg.get("lr", 0.0))
        parts.append(f"{name}={lr_val:.3e}")

    ref_lr = _optimizer_reference_learning_rate(optimizer)
    if parts:
        return f"ref={ref_lr:.3e} ({', '.join(parts)})"
    return f"ref={ref_lr:.3e}"


def _tensor_to_pil_image(
    tensor: torch.Tensor, 
    denormalize: bool = True,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
) -> Image.Image:
    """Convert a tensor image to PIL Image.
    
    Args:
        tensor: Image tensor in (C, H, W) format. Can be:
            - In [0, 1] range (raw ToTensor output)
            - Normalized (mean subtracted, std divided)
        denormalize: If True, reverse normalization before converting.
            Set to True when images were normalized for model input.
        mean: Normalization mean values [R, G, B]. Defaults to ImageNet mean.
        std: Normalization std values [R, G, B]. Defaults to ImageNet std.
    
    Returns:
        PIL Image in RGB format
    """
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for image visualization.")
    
    # Use provided constants or default to ImageNet
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    # Convert tensor to numpy array
    if torch.is_tensor(tensor):
        tensor = tensor.clone().detach().cpu()
        
        # Reverse normalization if applied
        # (tensor has negative values or values significantly > 1)
        if denormalize:
            # Use provided normalization constants (defaults to ImageNet)
            mean_tensor = torch.tensor(mean).view(3, 1, 1)
            std_tensor = torch.tensor(std).view(3, 1, 1)
            # Reverse normalization: x_orig = x_norm * std + mean
            tensor = tensor * std_tensor + mean_tensor
        
        # Clamp values to [0, 1] and convert to [0, 255]
        tensor = tensor.clamp(0, 1)
        # Convert (C, H, W) to (H, W, C)
        if tensor.dim() == 3:
            array = tensor.numpy().transpose(1, 2, 0)
        else:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D")
        array = (array * 255).astype(np.uint8)
    else:
        array = np.array(tensor)
    
    return Image.fromarray(array, "RGB")


def _draw_filename_caption(image: Image.Image, caption: str) -> Image.Image:
    """Draw source filename at top-left so TensorBoard views map back to disk."""
    if Image is None or not caption or not str(caption).strip():
        return image
    from PIL import ImageDraw, ImageFont

    text = str(caption).strip()
    max_len = 200
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    draw = ImageDraw.Draw(image)
    font = None
    for path in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, 16)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 6
    x0, y0 = 4, 4
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw.rectangle([x0, y0, x1, y1], fill=(16, 16, 16))
    draw.text((x0 + pad, y0 + pad), text, fill=(255, 255, 255), font=font)
    return image


def _visualize_predictions(
    image: Image.Image,
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_names: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
) -> Image.Image:
    """Visualize predictions and ground truth on an image.
    
    This function correctly visualizes oriented bounding boxes by converting
    RBoxes to polygons, which preserves the rotation angle.
    
    Args:
        image: PIL Image to draw on
        predictions: List of Detection objects (predictions)
        ground_truths: List of GroundTruth objects
        class_names: Optional list of class names for labels
        caption: Optional source filename (or label) drawn at top-left after boxes
    
    Returns:
        PIL Image with visualizations
    """
    if viz is None or RBox is None:
        raise RuntimeError("Visualization utilities are required.")
    
    # Draw ground truth boxes in green
    if ground_truths:
        # Convert RBoxes to polygons - this preserves the rotation angle
        # Each RBox is converted to a 4-point polygon representing the rotated rectangle
        # The polygon points are the actual rotated corners, not axis-aligned bounds
        gt_polygons = [gt.rbox.to_polygon().points for gt in ground_truths]
        gt_specs = [
            viz.DrawingSpec(outline=(0, 255, 0), width=2)  # Green for ground truth
            for _ in ground_truths
        ]
        image = viz.draw_polygons(image, gt_polygons, specs=gt_specs)
    
    # Draw prediction boxes in red with class and score labels
    if predictions:
        # Convert RBoxes to polygons - this preserves the rotation angle
        # The polygon points represent the actual rotated rectangle corners
        pred_polygons = [det.rbox.to_polygon().points for det in predictions]
        pred_specs = [
            viz.DrawingSpec(outline=(255, 0, 0), width=2)  # Red for predictions
            for _ in predictions
        ]
        pred_labels = [
            f"{det.class_name}: {det.score:.2f}" for det in predictions
        ]
        image = viz.draw_polygons(
            image,
            pred_polygons,
            specs=pred_specs,
            labels=pred_labels,
            label_color=(255, 255, 255),
        )
    
    if caption:
        image = _draw_filename_caption(image, caption)
    return image


def _visualize_boxes(
    image: Image.Image,
    boxes_tensor: torch.Tensor,
    color: tuple[int, int, int] = (255, 255, 0),
    max_boxes: int = 500,
    sample_spread: bool = True,
) -> Image.Image:
    """Draw oriented boxes (e.g. anchors or proposals) on an image for debugging.
    
    Args:
        image: PIL Image to draw on
        boxes_tensor: Tensor [N, 5] with format [cx, cy, w, h, angle]
        color: RGB outline color
        max_boxes: Maximum number of boxes to draw (subsampled if N > max_boxes)
        sample_spread: If True and N > max_boxes, randomly sample indices so boxes are spread
            across the set (avoids bias toward score-sorted start, so all orientations appear).
    
    Returns:
        PIL Image with boxes drawn
    """
    if viz is None or RBox is None or boxes_tensor is None or boxes_tensor.numel() == 0:
        return image
    try:
        from ..models.utils import tensor_to_rboxes
    except ImportError:
        return image
    N = int(boxes_tensor.shape[0])
    if N == 0:
        return image
    n = min(N, max_boxes)
    boxes_cpu = boxes_tensor.detach().cpu()
    if sample_spread and N > n:
        # Random sample so we see a mix of orientations; proposals are often score-sorted so
        # taking the first N would show mostly one angle (e.g. horizontal). Fixed seed for reproducibility.
        gen = torch.Generator(device=boxes_cpu.device)
        gen.manual_seed(0)
        indices = torch.randperm(N, generator=gen, device=boxes_cpu.device)[:n]
        subset = boxes_cpu[indices]
    else:
        subset = boxes_cpu[:n]
    rboxes = tensor_to_rboxes(subset)
    if not rboxes:
        return image
    spec = viz.DrawingSpec(outline=color, width=1)
    specs = [spec] * len(rboxes)
    return viz.draw_boxes(image, rboxes, specs=specs)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int = 0,
    epoch_idx: Optional[int] = None,
    epoch_count: Optional[int] = None,
    print_freq: int = 10,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    metric_tracker: Optional[MetricTracker] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    writer: Optional[Any] = None,
    global_step: Optional[int] = None,
    profiler: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    rank: Optional[int] = None,
    progress_stream: Optional[Any] = None,
    debug: bool = False,
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: Frequency of printing metrics
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Use automatic mixed precision
        metric_tracker: Optional metric tracker
        loss_weights: Optional weights for different loss components
        writer: Optional TensorBoard SummaryWriter for logging
        global_step: Optional global step counter (will be incremented internally)
        profiler: Optional TrainingProfiler instance for performance profiling
        rank: Optional rank for DDP (only rank 0 logs/prints, None for single-GPU)
        progress_stream: If set (e.g. original stderr when stdout is teed to a log file),
            tqdm writes here so the progress bar is not duplicated in the log file.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    _require_torch()
    
    model.train()
    metric_tracker = metric_tracker or MetricTracker()
    # Get device type from device parameter (e.g., 'cuda' or 'cpu')
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    # Initialize GradScaler - check which API is available
    if use_amp and GradScaler is not None:
        # Check if we're using the new torch.amp API (PyTorch 2.3+) or old torch.cuda.amp API
        # In PyTorch 2.2.2, torch.amp.GradScaler doesn't exist, so we use torch.cuda.amp.GradScaler
        # which doesn't take a device parameter (CUDA is implicit)
        try:
            # Try to import the new API to see if it exists
            from torch.amp import GradScaler as AmpGradScaler
            # New API (PyTorch 2.3+): GradScaler(device) - device is first positional arg
            scaler = AmpGradScaler(device_type)
        except ImportError:
            # Old API (torch.cuda.amp.GradScaler, PyTorch 2.2.2): no device parameter, CUDA is implicit
            # Only use old API if CUDA is available, otherwise skip scaler
            if device_type == 'cuda':
                scaler = GradScaler()
            else:
                scaler = None
    else:
        scaler = None
    
    optimizer.zero_grad()
    
    num_batches = len(data_loader)
    total_loss = 0.0
    current_step = global_step if global_step is not None else 0
    skip_optimizer_step = False  # Set when we skip backward due to nan/inf loss

    # Create progress bar (only on rank 0 or when rank is None).
    # When progress_stream is set (e.g. original stderr), tqdm writes there so the log file doesn't get every update.
    import sys
    pbar = None
    if tqdm is not None and (rank is None or rank == 0):
        tqdm_file = progress_stream if progress_stream is not None else sys.stderr
        if epoch_idx is not None and epoch_count is not None:
            epoch_desc = f"Epoch {int(epoch_idx)}/{int(epoch_count)}"
        else:
            epoch_desc = f"Epoch {epoch + 1}"
        pbar = tqdm(
            total=num_batches,
            desc=epoch_desc,
            unit="batch",
            ncols=120,
            leave=True,
            file=tqdm_file,
            disable=False,
        )
        if progress_stream is None:
            sys.stdout.flush()
    
    
    for batch_idx, batch in enumerate(data_loader):
        step_start = time.time()
        
        try:
            # Unpack batch (format depends on collate function)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images, targets = batch.get("images", []), batch.get("targets", [])
            
            # Move to device
            if isinstance(images, list):
                images = [img.to(device) if torch.is_tensor(img) else img for img in images]
            elif torch.is_tensor(images):
                images = images.to(device)
            
            # Move targets to device (tensors in target dicts)
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict):
                        for key, value in target.items():
                            if torch.is_tensor(value):
                                target[key] = value.to(device)
            
            # Debug: first-batch GT stats (per-image and per-class counts) to spot data/annotation issues.
            if debug and batch_idx == 0 and (rank is None or rank == 0) and isinstance(targets, list):
                from collections import Counter
                gt_per_image = []
                class_counts = Counter()
                for t in targets:
                    if isinstance(t, dict) and "labels" in t:
                        labels = t["labels"]
                        n = len(labels) if torch.is_tensor(labels) else len(labels)
                        gt_per_image.append(n)
                        if torch.is_tensor(labels):
                            for c in labels.cpu().tolist():
                                class_counts[c] += 1
                        else:
                            for c in labels:
                                class_counts[c] += 1
                if gt_per_image:
                    print(
                        f"  [debug] First batch GT: images={len(gt_per_image)}, "
                        f"GT/image min={min(gt_per_image)}, max={max(gt_per_image)}, mean={sum(gt_per_image)/len(gt_per_image):.1f}; "
                        f"per-class counts: {dict(class_counts.most_common(10))}",
                        flush=True,
                    )
            
            # Forward pass
            if use_amp and autocast is not None:
                try:
                    # PyTorch 2.x: device_type required - use device type from device parameter
                    with autocast(device_type=device_type):
                        loss_dict = model(images, targets)
                except TypeError:
                    # Older PyTorch (torch.cuda.amp.autocast): no device_type parameter, CUDA is implicit
                    # Old API only accepts keyword arguments: enabled, dtype, cache_enabled
                    # Do not pass device_type as positional argument - it would be interpreted as enabled='cuda' (wrong!)
                    with autocast():
                        loss_dict = model(images, targets)
            else:
                loss_dict = model(images, targets)
            
            # Extract and weight losses
            if not isinstance(loss_dict, dict):
                raise ValueError("Model forward must return a dict of losses.")
            
            total_loss_value = 0.0
            for key, value in loss_dict.items():
                if not torch.is_tensor(value):
                    continue
                weight = 1.0 if loss_weights is None else loss_weights.get(key, 1.0)
                total_loss_value += weight * value
            
            # Normalize by accumulation steps
            total_loss_value = total_loss_value / gradient_accumulation_steps

            # Robustness: skip backward and optimizer step when loss is nan/inf to avoid corrupting weights
            loss_is_invalid = not (torch.is_tensor(total_loss_value) and torch.isfinite(total_loss_value).all())
            if loss_is_invalid:
                skip_optimizer_step = True
                if rank is None or rank == 0:
                    import warnings
                    warnings.warn(
                        f"Train batch {batch_idx}: loss is nan or inf (total_loss_value={total_loss_value}), "
                        "skipping backward and optimizer step for this accumulation window."
                    )
                # Use zero for logging so epoch average is not skewed
                total_loss_value_safe = total_loss_value
                if torch.is_tensor(total_loss_value):
                    total_loss_value_safe = torch.tensor(0.0, device=total_loss_value.device, dtype=total_loss_value.dtype)
            else:
                # Backward pass
                if use_amp and scaler is not None:
                    scaler.scale(total_loss_value).backward()
                else:
                    total_loss_value.backward()
                total_loss_value_safe = total_loss_value

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if skip_optimizer_step:
                    optimizer.zero_grad()
                    skip_optimizer_step = False
                    grad_norm = None
                else:
                    # Gradient clipping
                    grad_norm = None
                    if max_grad_norm is not None:
                        if use_amp and scaler is not None:
                            scaler.unscale_(optimizer)
                        # Compute gradient norm before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    else:
                        # Compute gradient norm even if not clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

                    # Optimizer step
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                
                # Advance profiler step if provided
                if profiler is not None:
                    profiler.step()
                # Increment step counter and log to TensorBoard (per optimizer step)
                current_step += 1
                
                # Step learning rate scheduler only for WarmupScheduler (per optimizer step).
                # Do not step StepLR/etc. here — they are stepped once per epoch below.
                if lr_scheduler is not None and hasattr(lr_scheduler, 'step_epoch'):
                    lr_scheduler.step()
                
                if writer is not None:
                    # Update metrics for logging (use safe value when loss was invalid)
                    metrics = {k: float(v.item()) if torch.is_tensor(v) and torch.isfinite(v).all() else float(v) if not torch.is_tensor(v) else 0.0 for k, v in loss_dict.items()}
                    metrics["total_loss"] = float(total_loss_value_safe.item() * gradient_accumulation_steps)
                    for key, value in metrics.items():
                        # ROI diagnostics: log to TensorBoard only when debug is enabled (otherwise stdout/train.log only).
                        if key.startswith("roi_") and not debug:
                            continue
                        writer.add_scalar(f"train/{key}", value, current_step)
                    # Log learning rate (config-scale base, not backbone-only when param groups are used)
                    current_lr = _optimizer_reference_learning_rate(optimizer)
                    writer.add_scalar("train/learning_rate", current_lr, current_step)
                    # Log gradient norm
                    if grad_norm is not None:
                        writer.add_scalar("train/grad_norm", float(grad_norm), current_step)
                    
                    # Log per-component gradient norms (TensorBoard)
                    if grad_norm is not None:
                        # Get the actual model (unwrap DDP if needed)
                        actual_model = model.module if hasattr(model, 'module') else model
                        
                        # Compute gradient norms per component
                        component_grad_norms = {}
                        for name, param in actual_model.named_parameters():
                            if param.grad is not None:
                                param_grad_norm = param.grad.data.norm(2).item()
                                # Group by component (backbone, rpn, roi)
                                if 'backbone' in name:
                                    component = 'backbone'
                                elif 'rpn' in name.lower():
                                    component = 'rpn'
                                elif 'roi' in name.lower() or 'head' in name.lower():
                                    component = 'roi'
                                else:
                                    component = 'other'
                                
                                if component not in component_grad_norms:
                                    component_grad_norms[component] = []
                                component_grad_norms[component].append(param_grad_norm)
                        
                        # Log aggregated gradient norms per component
                        for component, norms in component_grad_norms.items():
                            if norms:
                                avg_norm = sum(norms) / len(norms)
                                max_norm = max(norms)
                                writer.add_scalar(f"train/grad_norm_{component}_avg", avg_norm, current_step)
                                writer.add_scalar(f"train/grad_norm_{component}_max", max_norm, current_step)
                        
                        # Also log per-component total gradient norm (sum of squares)
                        for component, norms in component_grad_norms.items():
                            if norms:
                                total_norm = math.sqrt(sum(n * n for n in norms))
                                writer.add_scalar(f"train/grad_norm_{component}_total", total_norm, current_step)
            
            # Update metrics (for MetricTracker; use safe values when loss was invalid)
            metrics = {}
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    metrics[k] = float(v.item()) if torch.isfinite(v).all() else 0.0
                else:
                    metrics[k] = float(v)
            metrics["total_loss"] = float(total_loss_value_safe.item() * gradient_accumulation_steps)
            metric_tracker.update(metrics)
            
            total_loss += metrics["total_loss"]
            
            # Update progress bar
            elapsed = time.time() - step_start
            metric_tracker.update_time(elapsed)
            
            # Update progress bar every batch (to avoid duplicate refreshes)
            if pbar is not None and (rank is None or rank == 0):
                # Compute summary for postfix (use rolling window, min of print_freq or current batch)
                window = min(print_freq, batch_idx + 1)
                summary = metric_tracker.get_summary(window=window)
                loss_val = summary.get('total_loss', 0.0)
                avg_time = summary.get('time_per_step', elapsed)
                
                # Calculate speed (batches per second) and ETA
                batches_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
                remaining_batches = num_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * avg_time if avg_time > 0 else 0.0
                
                # Format ETA as HH:MM:SS or MM:SS
                if eta_seconds >= 3600:
                    eta_str = f"{int(eta_seconds // 3600):d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                else:
                    eta_str = f"{int(eta_seconds // 60):d}:{int(eta_seconds % 60):02d}"
                
                # Update postfix and position together (single refresh)
                loss_str = f"Loss: {loss_val:.4f}"
                speed_str = f"{batches_per_sec:.2f}it/s"
                eta_str_tqdm = f"ETA: {eta_str}"
                pbar.set_postfix_str(f"{loss_str} | {speed_str} | {eta_str_tqdm}")
                pbar.update(1)  # This will refresh with the updated postfix
            
            # Print progress (every print_freq batches or at the end) - only for non-tqdm case
            if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == num_batches:
                if rank is None or rank == 0:
                    if pbar is None:  # Only print if tqdm is not available
                        summary = metric_tracker.get_summary(window=print_freq)
                        loss_val = summary.get('total_loss', 0.0)
                        avg_time = summary.get('time_per_step', elapsed)
                        
                        # Calculate speed (batches per second) and ETA
                        batches_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
                        remaining_batches = num_batches - (batch_idx + 1)
                        eta_seconds = remaining_batches * avg_time if avg_time > 0 else 0.0
                        
                        # Format ETA as HH:MM:SS or MM:SS
                        if eta_seconds >= 3600:
                            eta_str = f"{int(eta_seconds // 3600):d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                        else:
                            eta_str = f"{int(eta_seconds // 60):d}:{int(eta_seconds % 60):02d}"
                        
                        disp_epoch = int(epoch_idx) if epoch_idx is not None else int(epoch + 1)
                        disp_total = int(epoch_count) if epoch_count is not None else None
                        if disp_total is not None:
                            prefix = f"Epoch {disp_epoch}/{disp_total}"
                        else:
                            prefix = f"Epoch {disp_epoch}"
                        print(
                            f"{prefix} [{batch_idx + 1}/{num_batches}] "
                            f"Loss: {loss_val:.4f} | "
                            f"Speed: {batches_per_sec:.2f}it/s | "
                            f"ETA: {eta_str}"
                        )
                    # Force flush to ensure output is visible
                    import sys
                    sys.stdout.flush()
        
        except Exception as e:
            if pbar is not None:
                pbar.close()
            if rank is None or rank == 0:
                print(f"Error in batch {batch_idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    # Flush any partial accumulation window at epoch end so tail microbatches are not dropped.
    remainder_batches = num_batches % gradient_accumulation_steps
    if num_batches > 0 and remainder_batches != 0:
        if skip_optimizer_step:
            optimizer.zero_grad()
            skip_optimizer_step = False
            grad_norm = None
        else:
            grad_norm = None
            if max_grad_norm is not None:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        if profiler is not None:
            profiler.step()
        current_step += 1

        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_epoch'):
            lr_scheduler.step()

        if writer is not None:
            metrics = {
                k: float(v.item()) if torch.is_tensor(v) and torch.isfinite(v).all() else float(v) if not torch.is_tensor(v) else 0.0
                for k, v in loss_dict.items()
            }
            metrics["total_loss"] = float(total_loss_value_safe.item() * gradient_accumulation_steps)
            for key, value in metrics.items():
                if key.startswith("roi_") and not debug:
                    continue
                writer.add_scalar(f"train/{key}", value, current_step)
            current_lr = _optimizer_reference_learning_rate(optimizer)
            writer.add_scalar("train/learning_rate", current_lr, current_step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", float(grad_norm), current_step)

        if rank is None or rank == 0:
            print(
                f"  Flushed final accumulation window with {remainder_batches} batch(es) at epoch end.",
                flush=True,
            )

    # Close progress bar
    if pbar is not None:
        pbar.close()
    
    # Final metrics
    avg_metrics = metric_tracker.get_summary()
    avg_metrics["epoch"] = epoch
    
    # Single line to stdout (so log file gets one line per epoch when stdout is teed)
    if rank is None or rank == 0:
        loss_val = avg_metrics.get("total_loss", 0.0)
        disp_epoch = int(epoch_idx) if epoch_idx is not None else int(epoch + 1)
        disp_total = int(epoch_count) if epoch_count is not None else None
        if disp_total is not None:
            print(
                f"Epoch {disp_epoch}/{disp_total} complete: {num_batches}/{num_batches} batches, "
                f"loss: {loss_val:.4f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {disp_epoch} complete: {num_batches}/{num_batches} batches, loss: {loss_val:.4f}",
                flush=True,
            )
        print(f"  Effective LR: {_format_effective_lrs(optimizer, lr_scheduler)}", flush=True)
        # ROI assignment diagnostics (Rotated Faster R-CNN: RPN proposals only, before add_gt_as_proposals).
        if "roi_num_pos" in avg_metrics:
            print(
                "  ROI hints: "
                f"pos={avg_metrics.get('roi_num_pos', 0.0):.1f}, "
                f"bg={avg_metrics.get('roi_num_bg', 0.0):.1f}, "
                f"ignore={avg_metrics.get('roi_num_ignore', 0.0):.1f}, "
                f"sampled_fg={avg_metrics.get('roi_sampled_fg', 0.0):.1f}, "
                f"sampled_bg={avg_metrics.get('roi_sampled_bg', 0.0):.1f}, "
                f"matched_gt={avg_metrics.get('roi_matched_gt', 0.0):.1f}, "
                f"match_rate={avg_metrics.get('roi_match_rate', 0.0):.2%}",
                flush=True,
            )
        # Debug: full loss breakdown to compare with MMRotate and spot unbalanced losses.
        if debug:
            loss_keys = [k for k in avg_metrics if k not in ("epoch", "_global_step") and (k == "total_loss" or k.startswith("loss_"))]
            if loss_keys:
                parts = [f"{k}={avg_metrics[k]:.4f}" for k in sorted(loss_keys)]
                print("  [debug] Loss breakdown: " + ", ".join(parts), flush=True)
            if writer is not None:
                current_lr = _optimizer_reference_learning_rate(optimizer)
                print(f"  [debug] Learning rate: {current_lr:.6e}", flush=True)
    
    # Log epoch-level metrics to TensorBoard
    if writer is not None:
        for key, value in avg_metrics.items():
            if key != "epoch":
                # ROI diagnostics: log to TensorBoard when debug is enabled.
                if key.startswith("roi_") and not debug:
                    continue
                writer.add_scalar(f"train_epoch/{key}", value, epoch)
        # Store updated step in metrics for retrieval by train() function
        avg_metrics["_global_step"] = current_step
    
    return avg_metrics


def _iou_hist_bucket(x: float) -> int:
    """Bucket index for [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0]."""
    x = min(1.0, max(0.0, x))
    if x < 0.25:
        return 0
    if x < 0.5:
        return 1
    if x < 0.75:
        return 2
    return 3


def _eval_pairwise_rbox_iou(
    boxes_a: Sequence[Any],
    boxes_b: Sequence[Any],
    *,
    use_exact_rotated_iou: bool,
    device: Optional[Any] = None,
) -> list[list[float]]:
    """Pairwise IoU matrix for validation matching (mAP schedule metrics only).

    Args:
        boxes_a: First box collection (RBox or tensor rows).
        boxes_b: Second box collection.
        use_exact_rotated_iou: Exact CPU polygon IoU when True; GPU sampling when False.
        device: Torch device for the sampling path (defaults to CUDA when available).
    """
    if not boxes_a:
        return []
    if not boxes_b:
        return [[] for _ in boxes_a]

    from ..ops import iou as iou_ops

    if use_exact_rotated_iou:
        return iou_ops.batch_rbox_iou(list(boxes_a), list(boxes_b))

    _require_torch()
    from ..ops.gpu_ops import oriented_box_iou_gpu
    from ..ops.utils import rboxes_to_tensor

    dev = device
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ta = rboxes_to_tensor(boxes_a, device=dev)
    tb = rboxes_to_tensor(boxes_b, device=dev)
    return oriented_box_iou_gpu(ta, tb).detach().cpu().tolist()


def _per_image_class_agnostic_duplicate_stats(
    detections_post: list,
    iou_threshold: float,
    *,
    use_exact_rotated_iou: bool = True,
    device: Optional[Any] = None,
) -> Tuple[int, int]:
    """Count class-agnostic duplicate detections on one image (post score threshold).

    Sorts by score descending. A box is a duplicate if its IoU with any higher-score
    (already kept) box is >= ``iou_threshold``. Ignores class labels.

    Returns:
        (num_duplicate, num_total) where num_total == len(detections_post).
    """
    n = len(detections_post)
    if n == 0:
        return 0, 0
    sorted_dets = sorted(detections_post, key=lambda d: -d.score)
    sorted_rboxes = [d.rbox for d in sorted_dets]
    dup = 0
    kept_indices: list[int] = []
    for det_idx, det in enumerate(sorted_dets):
        overlap = False
        if kept_indices:
            iou_row = _eval_pairwise_rbox_iou(
                [det.rbox],
                [sorted_rboxes[i] for i in kept_indices],
                use_exact_rotated_iou=use_exact_rotated_iou,
                device=device,
            )[0]
            overlap = any(v >= iou_threshold for v in iou_row)
        if overlap:
            dup += 1
        else:
            kept_indices.append(det_idx)
    return dup, n


def _compute_val_stats_from_dicts(
    all_detections: Dict[str, list],
    all_ground_truths: Dict[str, list],
    score_threshold: float,
    iou_threshold: float,
    id_to_class: Dict[int, str],
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    extended_gt_metrics: bool = False,
    compute_matching_metrics: bool = True,
    use_exact_rotated_iou: bool = True,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute validation stats (GT cover, accuracy, score stats) from detection and GT dicts.

    Used after gathering validation results from all DDP ranks so rank 0 can
    recompute metrics on the merged full validation set.

    Returns dict with: total_ground_truths, gt_covered_pre_eval_threshold,
    gt_covered_post_eval_threshold, gt_lost_by_eval_threshold, total_correct,
    total_predictions, gt_classes_set, pred_classes_set, max_detection_score,
    sum_detection_scores, num_detection_scores, log_only GT–IoU diagnostics, and
    (when extended_gt_metrics) zero-best-IoU GT count, class-agnostic duplicate rate aggregates.

    ``use_exact_rotated_iou`` follows ``evaluation.use_exact_rotated_iou`` (exact CPU polygon
    IoU vs GPU sampling). Does not affect model inference / final NMS.
    """
    total_ground_truths = 0
    gt_covered_pre_eval_threshold = 0
    gt_covered_post_eval_threshold = 0
    gt_lost_by_eval_threshold = 0
    total_correct = 0
    total_predictions = 0
    gt_classes_set = set()
    pred_classes_set = set()
    max_detection_score = 0.0
    sum_detection_scores = 0.0
    num_detection_scores = 0

    best_iou_any_list: list[float] = []
    best_iou_same_list: list[float] = []
    best_iou_any_by_class: Dict[str, list[float]] = defaultdict(list)
    best_iou_same_by_class: Dict[str, list[float]] = defaultdict(list)
    buckets_any = [0, 0, 0, 0]
    buckets_same = [0, 0, 0, 0]
    gt_count_wrong_class_overlap = 0
    gt_count_no_box_above_iou = 0
    gt_count_zero_best_iou_any = 0

    # Class-agnostic duplicate rate (extended_gt_metrics): greedy by score vs eval IoU threshold
    dup_macro_sum = 0.0
    dup_macro_n_images = 0
    dup_total_redundant = 0
    dup_total_post = 0

    for image_id, ground_truths in all_ground_truths.items():
        detections = all_detections.get(image_id, [])
        total_ground_truths += len(ground_truths)
        for det in detections:
            pred_classes_set.add(det.class_name)
            v = det.score
            max_detection_score = max(max_detection_score, v)
            sum_detection_scores += v
            num_detection_scores += 1
        for gt in ground_truths:
            gt_classes_set.add(gt.class_name)

        raw_detections = detections
        detections_post = filter_detections_by_score_threshold(
            detections, score_threshold, per_class_score_threshold, id_to_class
        )
        total_predictions += len(detections_post)

        if extended_gt_metrics:
            d_dup, d_tot = _per_image_class_agnostic_duplicate_stats(
                detections_post,
                iou_threshold,
                use_exact_rotated_iou=use_exact_rotated_iou,
                device=device,
            )
            if d_tot > 0:
                dup_macro_sum += d_dup / d_tot
                dup_macro_n_images += 1
            dup_total_redundant += d_dup
            dup_total_post += d_tot

        raw_iou: Optional[list[list[float]]] = None
        post_iou: Optional[list[list[float]]] = None
        if compute_matching_metrics and len(ground_truths) > 0:
            gt_rboxes = [g.rbox for g in ground_truths]
            if raw_detections:
                raw_iou = _eval_pairwise_rbox_iou(
                    [d.rbox for d in raw_detections],
                    gt_rboxes,
                    use_exact_rotated_iou=use_exact_rotated_iou,
                    device=device,
                )
            if detections_post:
                post_iou = _eval_pairwise_rbox_iou(
                    [d.rbox for d in detections_post],
                    gt_rboxes,
                    use_exact_rotated_iou=use_exact_rotated_iou,
                    device=device,
                )

        if compute_matching_metrics and len(ground_truths) > 0 and raw_iou is not None:
            matched_gt_pre = set()
            matched_gt_post = set()
            for gt_idx, gt in enumerate(ground_truths):
                if extended_gt_metrics:
                    best_any = 0.0
                    best_same = 0.0
                    for det_idx, det in enumerate(raw_detections):
                        iou = raw_iou[det_idx][gt_idx]
                        if iou > best_any:
                            best_any = iou
                        if detection_matches_ground_truth_class(det, gt) and iou > best_same:
                            best_same = iou
                    best_iou_any_list.append(best_any)
                    best_iou_same_list.append(best_same)
                    cname = gt.class_name or "unknown"
                    best_iou_any_by_class[cname].append(best_any)
                    best_iou_same_by_class[cname].append(best_same)
                    buckets_any[_iou_hist_bucket(best_any)] += 1
                    buckets_same[_iou_hist_bucket(best_same)] += 1
                    if best_any <= 0.0:
                        gt_count_zero_best_iou_any += 1
                    if best_same >= iou_threshold:
                        pass
                    elif best_any >= iou_threshold:
                        gt_count_wrong_class_overlap += 1
                    else:
                        gt_count_no_box_above_iou += 1

                for det_idx, det in enumerate(raw_detections):
                    if det.class_id != gt.class_id:
                        continue
                    if raw_iou[det_idx][gt_idx] >= iou_threshold:
                        matched_gt_pre.add(gt_idx)
                        break
                if post_iou is not None:
                    for det_idx, det in enumerate(detections_post):
                        if det.class_id != gt.class_id:
                            continue
                        if post_iou[det_idx][gt_idx] >= iou_threshold:
                            matched_gt_post.add(gt_idx)
                            break
            gt_covered_pre_eval_threshold += len(matched_gt_pre)
            gt_covered_post_eval_threshold += len(matched_gt_post)
            gt_lost_by_eval_threshold += max(0, len(matched_gt_pre) - len(matched_gt_post))

        if compute_matching_metrics and post_iou is not None and len(ground_truths) > 0:
            for det_idx, det in enumerate(detections_post):
                best_iou = 0.0
                best_gt_class = None
                for gt_idx, gt in enumerate(ground_truths):
                    det_iou = post_iou[det_idx][gt_idx]
                    if det_iou > best_iou and det_iou >= iou_threshold:
                        best_iou = det_iou
                        best_gt_class = gt.class_name
                if best_gt_class is not None and det.class_name == best_gt_class:
                    total_correct += 1

    out: Dict[str, Any] = {
        "total_ground_truths": total_ground_truths,
        "gt_covered_pre_eval_threshold": gt_covered_pre_eval_threshold,
        "gt_covered_post_eval_threshold": gt_covered_post_eval_threshold,
        "gt_lost_by_eval_threshold": gt_lost_by_eval_threshold,
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "gt_classes_set": gt_classes_set,
        "pred_classes_set": pred_classes_set,
        "max_detection_score": max_detection_score,
        "sum_detection_scores": sum_detection_scores,
        "num_detection_scores": num_detection_scores,
    }
    if extended_gt_metrics:
        mean_any = statistics.fmean(best_iou_any_list) if best_iou_any_list else 0.0
        mean_same = statistics.fmean(best_iou_same_list) if best_iou_same_list else 0.0
        median_any = float(statistics.median(best_iou_any_list)) if best_iou_any_list else 0.0
        median_same = float(statistics.median(best_iou_same_list)) if best_iou_same_list else 0.0
        out["log_only_gt_mean_best_iou_any"] = mean_any
        out["log_only_gt_mean_best_iou_same_class"] = mean_same
        out["log_only_gt_median_best_iou_any"] = median_any
        out["log_only_gt_median_best_iou_same_class"] = median_same
        out["log_only_gt_count_wrong_class_overlap_at_iou"] = gt_count_wrong_class_overlap
        out["log_only_gt_count_no_det_iou_above_threshold"] = gt_count_no_box_above_iou
        out["log_only_gt_count_zero_best_iou_any"] = gt_count_zero_best_iou_any
        n_gt_iou = len(best_iou_any_list)
        out["log_only_gt_rate_zero_best_iou_any"] = (
            gt_count_zero_best_iou_any / n_gt_iou if n_gt_iou > 0 else 0.0
        )
        out["log_only_gt_best_iou_any_buckets"] = list(buckets_any)
        out["log_only_gt_best_iou_same_class_buckets"] = list(buckets_same)
        out["log_only_gt_class_agnostic_dup_rate_macro"] = (
            dup_macro_sum / dup_macro_n_images if dup_macro_n_images > 0 else 0.0
        )
        out["log_only_gt_class_agnostic_dup_rate_micro"] = (
            dup_total_redundant / dup_total_post if dup_total_post > 0 else 0.0
        )
        out["log_only_gt_class_agnostic_dup_redundant_boxes"] = int(dup_total_redundant)
        out["log_only_gt_class_agnostic_dup_post_threshold_boxes"] = int(dup_total_post)
        out["log_only_gt_class_agnostic_dup_macro_image_count"] = int(dup_macro_n_images)
        align = _aggregate_gt_best_iou_samples(
            best_iou_any_by_class,
            best_iou_same_by_class,
        )
        out["log_only_gt_alignment_metrics"] = gt_best_iou_alignment_metrics_to_dict(align)

    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    metric_tracker: Optional[MetricTracker] = None,
    writer: Optional[Any] = None,
    epoch: Optional[int] = None,
    class_map: Optional[Dict[str, int]] = None,
    class_names: Optional[Sequence[str]] = None,
    score_threshold: float = 0.05,
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    vis_score_threshold: Optional[float] = None,
    iou_threshold: float = 0.5,
    extended_gt_metrics: bool = False,
    compute_map: bool = True,
    log_images: bool = True,
    max_images_to_log: int = 4,
    fixed_image_index: int = 0,
    log_random_image: bool = True,
    image_pool_max_size: int = 32,
    log_debug_anchors_proposals: bool = False,
    normalize_mean: Optional[list[float]] = None,
    normalize_std: Optional[list[float]] = None,
    vis_image_size: Optional[Tuple[int, int]] = None,
    rank: Optional[int] = None,
    progress_stream: Optional[Any] = None,
    debug: bool = False,
    eval_use_exact_rotated_iou: bool = True,
    compute_matching_metrics: bool = True,
) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Note: When using DDP, this function bypasses DDP synchronization during forward pass
    by using model.module. This prevents deadlocks when only rank 0 runs validation
    while other ranks wait at barriers.
    
    Args:
        model: Model to evaluate (may be DDP-wrapped)
        data_loader: DataLoader for validation data
        device: Device to evaluate on
        metric_tracker: Optional metric tracker
        writer: Optional TensorBoard SummaryWriter for logging
        epoch: Optional epoch number for logging
        class_map: Optional mapping from class names to class IDs (for converting predictions)
        class_names: Optional list of class names (for evaluation)
        score_threshold: Minimum confidence score for detections (metrics and matching)
        vis_score_threshold: Min score for boxes drawn in TensorBoard images; None = use score_threshold
        iou_threshold: IoU threshold for mAP calculation
        compute_map: Whether to compute mAP (can be slow with many detections)
        log_images: Whether to log prediction images to TensorBoard
        max_images_to_log: Maximum number of images to log per epoch (to avoid slowing down training)
        fixed_image_index: Index in the validation order of the image to always log (0 = first image of first batch).
        log_random_image: If True, also log one randomly selected image from the pool (in addition to the fixed one).
        image_pool_max_size: Maximum number of validation images to keep in the pool for random selection.
        log_debug_anchors_proposals: If True and model supports it, log one randomly selected validation image with anchors and one with RPN proposals to TensorBoard (for debugging). Random selection avoids always using the first image, which may have no ground truth.
        normalize_mean: Normalization mean values [R, G, B] for denormalization. Defaults to ImageNet mean.
        normalize_std: Normalization std values [R, G, B] for denormalization. Defaults to ImageNet std.
        vis_image_size: Optional (height, width) of the content region for TensorBoard images. When set, the image tensor is cropped to this size before drawing boxes so that annotations (in preprocessing target space) align with the displayed image. Default None uses the full tensor.
        rank: Optional rank for DDP (only rank 0 logs/prints, None for single-GPU)
        progress_stream: If set (e.g. original stderr when stdout is teed to a log file),
            tqdm writes here so the progress bar is not duplicated in the log file.
    
    Returns:
        Dictionary of evaluation metrics including mAP, per-class AP, GT classes, pred classes, and accuracy
    """
    _require_torch()
    import sys
    # Extract underlying model if DDP-wrapped (avoids DDP sync during validation)
    # CRITICAL: Set eval mode on underlying model, not DDP-wrapped model
    # Calling .eval() on DDP model can trigger synchronization operations
    # Also disable DDP's require_backward_grad_sync if DDP-wrapped to prevent any sync attempts
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()

    # If DDP-wrapped, disable gradient synchronization during validation
    # This ensures no NCCL operations occur even if something tries to sync
    original_sync_state = None
    if hasattr(model, 'require_backward_grad_sync'):
        # Temporarily disable gradient sync (though we're in no_grad anyway)
        original_sync_state = model.require_backward_grad_sync
        model.require_backward_grad_sync = False
    
    metric_tracker = metric_tracker or MetricTracker()
    
    # Create reverse mapping from class_id to class_name
    id_to_class = {}
    if class_map is not None:
        id_to_class = {v: k for k, v in class_map.items()}
    
    # Collect all detections and ground truths
    all_detections: Dict[str, List[Detection]] = {}
    all_ground_truths: Dict[str, List[GroundTruth]] = {}
    
    # Track classes and counts
    gt_classes_set = set()
    pred_classes_set = set()
    total_correct = 0
    total_predictions = 0
    total_ground_truths = 0
    # Log-only diagnostics: how many GTs are covered before/after eval score filtering.
    gt_covered_pre_eval_threshold = 0
    gt_covered_post_eval_threshold = 0
    gt_lost_by_eval_threshold = 0
    max_detection_score = 0.0
    sum_detection_scores = 0.0
    num_detection_scores = 0
    
    # Pool of (vis_tensor, image_id) for TensorBoard: we pick fixed + random from this
    image_pool: Optional[list[tuple[Any, str]]] = [] if (writer is not None and log_images) else None
    # Debug images: anchors and proposals from first batch (only when log_debug_anchors_proposals and model supports it)
    debug_anchors_vis: Optional[torch.Tensor] = None
    debug_proposals_vis: Optional[torch.Tensor] = None
    
    # Create progress bar for validation (only on rank 0 or when rank is None).
    # When progress_stream is set, tqdm writes there so the log file doesn't get every update.
    num_batches = len(data_loader)
    # When logging debug anchors/proposals, pick a random batch so we don't always use the first image (which may have no GT).
    debug_batch_idx = random.randint(0, num_batches - 1) if (log_debug_anchors_proposals and num_batches > 0) else -1
    pbar = None
    if tqdm is not None and (rank is None or rank == 0):
        tqdm_file = progress_stream if progress_stream is not None else sys.stderr
        pbar = tqdm(
            total=num_batches,
            desc=f"Validation",
            unit="batch",
            ncols=120,
            leave=True,
            file=tqdm_file,
            disable=False,
        )
    
    for batch_idx, batch in enumerate(data_loader):
        step_start = time.time()
        try:
            # Unpack batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images, targets = batch.get("images", []), batch.get("targets", [])
            
            # Move to device
            if isinstance(images, list):
                images = [img.to(device) if torch.is_tensor(img) else img for img in images]
            elif torch.is_tensor(images):
                images = images.to(device)
            
            # Optional: ask model to return anchors and proposals for TensorBoard (random batch when log_debug_anchors_proposals)
            if batch_idx == debug_batch_idx and writer is not None and log_images and log_debug_anchors_proposals:
                setattr(eval_model, '_return_anchors_proposals', True)
            
            # Forward pass
            # CRITICAL: When all ranks participate in validation (DistributedSampler),
            # we can use the DDP-wrapped model directly. DDP will handle synchronization
            # correctly since all ranks are participating.
            # When only rank 0 validates, use eval_model (model.module) to bypass DDP.
            # Check if this is distributed validation (all ranks have data) or single-rank
            if rank is not None and dist is not None and dist.is_initialized():
                # Distributed validation: all ranks participate, use DDP model
                outputs = model(images)
            else:
                # Single-rank validation: use underlying model to avoid DDP sync
                outputs = eval_model(images)
            
            # Process each image in the batch
            if not isinstance(outputs, list):
                outputs = [outputs]
            if not isinstance(targets, list):
                targets = [targets]
            
            # Ensure images is a list for iteration
            if not isinstance(images, list):
                images = [images]
            # Pick random image index for debug anchors/proposals when this is the chosen batch
            debug_img_idx = -1
            if log_debug_anchors_proposals and batch_idx == debug_batch_idx and debug_anchors_vis is None:
                n_imgs = len(images)
                debug_img_idx = random.randint(0, n_imgs - 1) if n_imgs > 0 else 0
            
            # Track detections in current batch
            batch_detections = 0
            batch_detections_050 = 0  # count at score >= 0.5 (more informative when model is overconfident)

            for i, (output, target, image_tensor) in enumerate(zip(outputs, targets, images)):
                # Get image_id
                image_id = target.get("image_id", f"batch_{batch_idx}_img_{i}")
                if isinstance(image_id, torch.Tensor):
                    image_id = str(image_id.item())
                elif not isinstance(image_id, str):
                    image_id = str(image_id)
                display_name = target.get("image_filename")
                if display_name is None:
                    caption = image_id
                elif isinstance(display_name, torch.Tensor):
                    caption = str(display_name.item())
                elif not isinstance(display_name, str):
                    caption = str(display_name)
                else:
                    caption = display_name
                
                # Extract predictions (only rboxes format supported)
                rboxes = output.get("rboxes", [])
                scores = output.get("scores", torch.zeros(0))
                labels = output.get("labels", torch.zeros(0, dtype=torch.int64))
                
                # Ensure rboxes is a list of RBox objects
                if isinstance(rboxes, torch.Tensor) and len(rboxes) > 0:
                    # Convert tensor [N, 5] to list of RBox objects
                    if RBox is not None:
                        from ..models.utils import tensor_to_rboxes
                        rboxes = tensor_to_rboxes(rboxes)
                    else:
                        rboxes = []
                elif not isinstance(rboxes, list):
                    rboxes = []
                
                # Count detections at score >= 0.5 (for reporting; helps when eval threshold is low and model outputs 100)
                if len(scores) > 0:
                    if torch.is_tensor(scores):
                        batch_detections_050 += (scores >= 0.5).sum().item()
                        max_detection_score = max(max_detection_score, float(scores.max().item()))
                        sum_detection_scores += float(scores.sum().item())
                        num_detection_scores += scores.numel()
                    else:
                        batch_detections_050 += sum(1 for s in scores if float(s) >= 0.5)
                        for s in scores:
                            v = float(s)
                            max_detection_score = max(max_detection_score, v)
                            sum_detection_scores += v
                            num_detection_scores += 1
                # Keep a copy of raw predictions before eval score filtering for diagnostics.
                raw_scores = scores
                raw_labels = labels
                raw_rboxes = list(rboxes) if isinstance(rboxes, list) else []

                # Filter by score threshold (global and optional per-class)
                if len(scores) > 0:
                    if per_class_score_threshold and id_to_class:
                        mask = scores_labels_pass_threshold(
                            scores,
                            labels,
                            score_threshold,
                            per_class_score_threshold,
                            id_to_class,
                        )
                    else:
                        mask = scores >= score_threshold
                    scores = scores[mask]
                    labels = labels[mask]
                    if isinstance(rboxes, list) and len(rboxes) > 0:
                        rboxes = [rbox for rbox, keep in zip(rboxes, mask) if keep]

                # When padding was applied, detections are in padded image coords; GTs are in content coords. Clamp detection centers to content rect for correct IoU.
                def _clamp_rbox_to_content(rb, c_h: float, c_w: float):
                    cx_c = max(0.0, min(c_w, rb.cx))
                    cy_c = max(0.0, min(c_h, rb.cy))
                    return RBox(cx_c, cy_c, rb.width, rb.height, rb.angle)

                clamp_rbox = lambda r: r
                content_size = target.get("content_size")
                if content_size is not None and RBox is not None:
                    content_h, content_w = float(content_size[0]), float(content_size[1])
                    im_h = float(image_tensor.shape[-2])
                    im_w = float(image_tensor.shape[-1])
                    if (content_h, content_w) != (im_h, im_w):
                        clamp_rbox = lambda r, ch=content_h, cw=content_w: _clamp_rbox_to_content(r, ch, cw)

                # Convert predictions to Detection objects
                detections = []
                raw_detections = []
                if RBox is not None and Detection is not None:
                    # Build pre-filter detections (for log diagnostics only)
                    num_raw_dets = len(raw_scores)
                    if isinstance(raw_rboxes, list) and len(raw_rboxes) == num_raw_dets:
                        for rbox, score, label_id in zip(raw_rboxes, raw_scores, raw_labels):
                            label_id_int = int(label_id.item()) if torch.is_tensor(label_id) else int(label_id)
                            class_name = id_to_class.get(label_id_int, f"class_{label_id_int}")
                            raw_detections.append(Detection(
                                rbox=clamp_rbox(rbox),
                                score=float(score.item()) if torch.is_tensor(score) else float(score),
                                class_id=label_id_int,
                                class_name=class_name,
                                image_id=image_id,
                            ))

                    # Ensure all lists have the same length
                    num_dets = len(scores)
                    if isinstance(rboxes, list) and len(rboxes) == num_dets:
                        for j, (rbox, score, label_id) in enumerate(zip(rboxes, scores, labels)):
                            label_id_int = int(label_id.item()) if torch.is_tensor(label_id) else int(label_id)
                            class_name = id_to_class.get(label_id_int, f"class_{label_id_int}")
                            pred_classes_set.add(class_name)
                            detections.append(Detection(
                                rbox=clamp_rbox(rbox),
                                score=float(score.item()) if torch.is_tensor(score) else float(score),
                                class_id=label_id_int,
                                class_name=class_name,
                                image_id=image_id,
                            ))
                            total_predictions += 1
                
                batch_detections += len(detections)
                # Store raw (pre-score-threshold) detections so DDP gather can recompute pre/post stats on merged data
                all_detections[image_id] = raw_detections
                
                # Extract ground truths (only rboxes format supported)
                gt_labels = target.get("labels", torch.zeros(0, dtype=torch.int64))
                gt_rboxes = target.get("rboxes", None)
                gt_labels_ignore = target.get("labels_ignore", torch.zeros(0, dtype=torch.int64))
                gt_rboxes_ignore = target.get("rboxes_ignore", None)
                
                
                # Convert ground truths to GroundTruth objects
                ground_truths = []
                if gt_rboxes is not None:
                    # Handle rboxes (can be tensor [N, 5] or list of RBox objects)
                    if isinstance(gt_rboxes, torch.Tensor):
                        # Convert tensor [N, 5] to list of RBox objects
                        if RBox is not None:
                            from ..models.utils import tensor_to_rboxes
                            gt_rboxes_list = tensor_to_rboxes(gt_rboxes)
                        else:
                            gt_rboxes_list = []
                    elif isinstance(gt_rboxes, list):
                        gt_rboxes_list = gt_rboxes
                    else:
                        gt_rboxes_list = []
                    
                    # Use rboxes if available
                    if len(gt_rboxes_list) > 0:
                        for rbox, label_id in zip(gt_rboxes_list, gt_labels):
                            if RBox is None or GroundTruth is None:
                                break
                            label_id_int = int(label_id.item()) if torch.is_tensor(label_id) else int(label_id)
                            class_name = id_to_class.get(label_id_int, f"class_{label_id_int}")
                            gt_classes_set.add(class_name)
                            ground_truths.append(GroundTruth(
                                rbox=rbox,
                                class_id=label_id_int,
                                class_name=class_name,
                                difficult=0,
                                image_id=image_id,
                            ))

                # Optional ignore GTs (e.g. DOTA difficult objects when difficult_strategy="ignore").
                if gt_rboxes_ignore is not None:
                    if isinstance(gt_rboxes_ignore, torch.Tensor):
                        if RBox is not None:
                            from ..models.utils import tensor_to_rboxes
                            gt_rboxes_ign_list = tensor_to_rboxes(gt_rboxes_ignore)
                        else:
                            gt_rboxes_ign_list = []
                    elif isinstance(gt_rboxes_ignore, list):
                        gt_rboxes_ign_list = gt_rboxes_ignore
                    else:
                        gt_rboxes_ign_list = []

                    if len(gt_rboxes_ign_list) > 0:
                        for rbox, label_id in zip(gt_rboxes_ign_list, gt_labels_ignore):
                            if RBox is None or GroundTruth is None:
                                break
                            label_id_int = int(label_id.item()) if torch.is_tensor(label_id) else int(label_id)
                            class_name = id_to_class.get(label_id_int, f"class_{label_id_int}")
                            gt_classes_set.add(class_name)
                            ground_truths.append(GroundTruth(
                                rbox=rbox,
                                class_id=label_id_int,
                                class_name=class_name,
                                difficult=1,
                                image_id=image_id,
                            ))
                # Note: No fallback for "boxes" format - only "rboxes" is supported
                
                all_ground_truths[image_id] = ground_truths
                total_ground_truths += len(ground_truths)

                # Add visualization to pool for TensorBoard (capped to avoid OOM).
                # When log_debug_anchors_proposals is True, only add the same random image we use for debug (anchors/proposals) so predictions show that one image.
                add_to_pool = image_pool is not None and len(image_pool) < image_pool_max_size
                if add_to_pool and log_debug_anchors_proposals:
                    add_to_pool = batch_idx == debug_batch_idx and i == debug_img_idx
                if add_to_pool:
                    try:
                        # Use content region so image and boxes share the same coordinate system.
                        # (Boxes are in preprocessing target size; tensor may be padded.)
                        vis_tensor = image_tensor
                        if vis_image_size is not None:
                            vh, vw = vis_image_size[0], vis_image_size[1]
                            _, th, tw = image_tensor.shape
                            if th >= vh and tw >= vw:
                                vis_tensor = image_tensor[:, :vh, :vw].clone()
                        # Convert tensor to PIL Image
                        pil_image = _tensor_to_pil_image(
                            vis_tensor,
                            mean=normalize_mean,
                            std=normalize_std,
                        )
                        # Boxes to draw: use vis_score_threshold if set, else score_threshold (detections)
                        vis_thresh = vis_score_threshold if vis_score_threshold is not None else score_threshold
                        detections_for_vis = [d for d in raw_detections if d.score >= vis_thresh]
                        # Create visualization
                        vis_image = _visualize_predictions(
                            pil_image.copy(),
                            detections_for_vis,
                            ground_truths,
                            class_names=class_names,
                            caption=caption,
                        )
                        # Convert back to tensor for TensorBoard (C, H, W) format
                        if np is not None:
                            vis_array = np.array(vis_image).transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                            vis_tensor = torch.from_numpy(vis_array).float() / 255.0
                            image_pool.append((vis_tensor, image_id))
                        # Debug: anchors and proposals from a random image (model may set output["anchors"], output["proposals"])
                        if batch_idx == debug_batch_idx and i == debug_img_idx and log_debug_anchors_proposals and debug_anchors_vis is None:
                            anchors_t = output.get("anchors")
                            proposals_t = output.get("proposals")
                            if anchors_t is not None and isinstance(anchors_t, torch.Tensor) and anchors_t.numel() > 0:
                                try:
                                    pil_anchors = _tensor_to_pil_image(image_tensor, mean=normalize_mean, std=normalize_std)
                                    vis_anchors = _visualize_boxes(pil_anchors.copy(), anchors_t, color=(255, 200, 0), max_boxes=500)
                                    vis_anchors = _draw_filename_caption(vis_anchors, caption)
                                    if np is not None:
                                        arr = np.array(vis_anchors).transpose(2, 0, 1)
                                        debug_anchors_vis = torch.from_numpy(arr).float() / 255.0
                                except Exception:
                                    pass
                            if proposals_t is not None and isinstance(proposals_t, torch.Tensor) and proposals_t.numel() > 0:
                                try:
                                    pil_proposals = _tensor_to_pil_image(image_tensor, mean=normalize_mean, std=normalize_std)
                                    vis_proposals = _visualize_boxes(pil_proposals.copy(), proposals_t, color=(0, 200, 255), max_boxes=500)
                                    vis_proposals = _draw_filename_caption(vis_proposals, caption)
                                    if np is not None:
                                        arr = np.array(vis_proposals).transpose(2, 0, 1)
                                        debug_proposals_vis = torch.from_numpy(arr).float() / 255.0
                                except Exception:
                                    pass
                    except Exception as e:
                        # Silently skip if visualization fails (e.g., missing dependencies)
                        pass
            
            # Track basic metrics (detections per image, not total per batch)
            num_images_in_batch = len(outputs) if isinstance(outputs, list) else 1
            dets_per_image = batch_detections / num_images_in_batch if num_images_in_batch > 0 else 0.0
            dets_per_image_050 = batch_detections_050 / num_images_in_batch if num_images_in_batch > 0 else 0.0
            metrics = {"num_detections": dets_per_image, "num_detections_050": dets_per_image_050}
            metric_tracker.update(metrics)
            
            # Clear debug flag after the batch we used so subsequent batches don't return anchors/proposals
            if batch_idx == debug_batch_idx and getattr(eval_model, '_return_anchors_proposals', False):
                setattr(eval_model, '_return_anchors_proposals', False)
            
            # Update progress bar
            elapsed = time.time() - step_start
            metric_tracker.update_time(elapsed)
            
            if pbar is not None:
                # Update progress bar every batch
                pbar.update(1)
                
                # Update metrics display periodically (every 10 batches or at the end)
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    summary = metric_tracker.get_summary(window=10)
                    # Average detections per batch over the window
                    avg_dets = summary.get('num_detections', 0)
                    avg_time = summary.get('time_per_step', elapsed)
                    
                    # Calculate speed and ETA
                    batches_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
                    remaining_batches = num_batches - (batch_idx + 1)
                    eta_seconds = remaining_batches * avg_time if avg_time > 0 else 0.0
                    
                    # Format ETA
                    if eta_seconds >= 3600:
                        eta_str = f"{int(eta_seconds // 3600):d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                    else:
                        eta_str = f"{int(eta_seconds // 60):d}:{int(eta_seconds % 60):02d}"
                    
                    num_dets_str = f"Dets/img: {avg_dets:.1f}"
                    speed_str = f"{batches_per_sec:.2f}it/s"
                    eta_str_tqdm = f"ETA: {eta_str}"
                    pbar.set_postfix_str(f"{num_dets_str} | {speed_str} | {eta_str_tqdm}")
            elif rank is None or rank == 0:
                # Fallback: print progress with speed and ETA if tqdm is not available
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    summary = metric_tracker.get_summary(window=10)
                    avg_dets = summary.get('num_detections', 0)
                    avg_time = summary.get('time_per_step', elapsed)
                    
                    # Calculate speed and ETA
                    batches_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
                    remaining_batches = num_batches - (batch_idx + 1)
                    eta_seconds = remaining_batches * avg_time if avg_time > 0 else 0.0
                    
                    # Format ETA
                    if eta_seconds >= 3600:
                        eta_str = f"{int(eta_seconds // 3600):d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                    else:
                        eta_str = f"{int(eta_seconds // 60):d}:{int(eta_seconds % 60):02d}"
                    
                    print(
                        f"Validation [{batch_idx + 1}/{num_batches}] "
                        f"Dets/img: {avg_dets:.1f} | "
                        f"Speed: {batches_per_sec:.2f}it/s | "
                        f"ETA: {eta_str}"
                    )
                    import sys
                    sys.stdout.flush()
        
        except Exception as e:
            if pbar is not None:
                pbar.close()
            if rank is None or rank == 0:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Close progress bar
    if pbar is not None:
        pbar.close()
    
    # Single line to stdout (so log file gets one line when stdout is teed)
    if (rank is None or rank == 0) and num_batches > 0:
        print(f"Validation complete: {num_batches} batches", flush=True)

    # DDP: gather all_detections and all_ground_truths from all ranks to rank 0, merge, and recompute stats on full val set
    did_gather = False
    val_stats_extra: Dict[str, Any] = {}
    if dist is not None and dist.is_initialized() and rank is not None:
        world_size = dist.get_world_size()
        if world_size > 1:
            payload = (list(all_detections.items()), list(all_ground_truths.items()))
            object_list = [None] * world_size if rank == 0 else None
            # Finish GPU work before collectives; helps surface async CUDA errors here, not in gather.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Ensure every rank reaches gather together after validation.
            dist.barrier()
            gather_group = _get_gloo_gather_object_group()
            gather_success = False
            try:
                if gather_group is not None:
                    try:
                        dist.gather_object(payload, object_list, dst=0, group=gather_group)
                    except TypeError:
                        # PyTorch without `group=` on gather_object — use default process group
                        dist.gather_object(payload, object_list, dst=0)
                else:
                    dist.gather_object(payload, object_list, dst=0)
                gather_success = True
            except AttributeError:
                # Older PyTorch may not have gather_object; keep shard-only metrics
                object_list = None
            except RuntimeError as e:
                # Last resort if Gloo gather still fails: log and fall back to per-rank shard metrics.
                if rank == 0:
                    print(
                        f"Warning: dist.gather_object failed ({e!r}); "
                        "using per-rank validation shard only (mAP/stats may be incomplete).",
                        flush=True,
                    )
                object_list = None
            if gather_success:
                if rank == 0:
                    did_gather = True
                    merged_detections = {}
                    merged_ground_truths = {}
                    for (det_items, gt_items) in object_list:
                        for k, v in dict(det_items).items():
                            merged_detections[k] = merged_detections.get(k, []) + list(v)
                        for k, v in dict(gt_items).items():
                            merged_ground_truths[k] = merged_ground_truths.get(k, []) + list(v)
                    all_detections = merged_detections
                    all_ground_truths = merged_ground_truths
                    stats = _compute_val_stats_from_dicts(
                        all_detections,
                        all_ground_truths,
                        score_threshold,
                        iou_threshold,
                        id_to_class,
                        per_class_score_threshold,
                        extended_gt_metrics,
                        compute_matching_metrics=compute_matching_metrics,
                        use_exact_rotated_iou=eval_use_exact_rotated_iou,
                        device=device,
                    )
                    total_ground_truths = stats["total_ground_truths"]
                    gt_covered_pre_eval_threshold = stats["gt_covered_pre_eval_threshold"]
                    gt_covered_post_eval_threshold = stats["gt_covered_post_eval_threshold"]
                    gt_lost_by_eval_threshold = stats["gt_lost_by_eval_threshold"]
                    total_correct = stats["total_correct"]
                    total_predictions = stats["total_predictions"]
                    gt_classes_set = stats["gt_classes_set"]
                    pred_classes_set = stats["pred_classes_set"]
                    max_detection_score = stats["max_detection_score"]
                    sum_detection_scores = stats["sum_detection_scores"]
                    num_detection_scores = stats["num_detection_scores"]
                    val_stats_extra = stats
                else:
                    did_gather = True
                    total_ground_truths = 0
                    gt_covered_pre_eval_threshold = 0
                    gt_covered_post_eval_threshold = 0
                    gt_lost_by_eval_threshold = 0
                    total_correct = 0
                    total_predictions = 0
                    gt_classes_set = set()
                    pred_classes_set = set()
                    max_detection_score = 0.0
                    sum_detection_scores = 0.0
                    num_detection_scores = 0

    # Single-process / single-GPU: recompute stats from full dicts (includes GT–IoU diagnostics).
    if not did_gather and (rank is None or rank == 0):
        val_stats_extra = _compute_val_stats_from_dicts(
            all_detections,
            all_ground_truths,
            score_threshold,
            iou_threshold,
            id_to_class,
            per_class_score_threshold,
            extended_gt_metrics,
            compute_matching_metrics=compute_matching_metrics,
            use_exact_rotated_iou=eval_use_exact_rotated_iou,
            device=device,
        )
        total_ground_truths = val_stats_extra["total_ground_truths"]
        gt_covered_pre_eval_threshold = val_stats_extra["gt_covered_pre_eval_threshold"]
        gt_covered_post_eval_threshold = val_stats_extra["gt_covered_post_eval_threshold"]
        gt_lost_by_eval_threshold = val_stats_extra["gt_lost_by_eval_threshold"]
        total_correct = val_stats_extra["total_correct"]
        total_predictions = val_stats_extra["total_predictions"]
        gt_classes_set = val_stats_extra["gt_classes_set"]
        pred_classes_set = val_stats_extra["pred_classes_set"]
        max_detection_score = val_stats_extra["max_detection_score"]
        sum_detection_scores = val_stats_extra["sum_detection_scores"]
        num_detection_scores = val_stats_extra["num_detection_scores"]

    # Compute DOTA metrics
    summary = metric_tracker.get_summary()

    # Compute mAP if evaluation utilities are available and requested (only on rank 0 when we gathered)
    if compute_map and compute_oriented_map is not None and Detection is not None and GroundTruth is not None:
        if did_gather and rank is not None and rank != 0:
            summary["mAP"] = -1.0
        else:
            try:
                # mAP uses score-filtered detections (all_detections stores raw for DDP stats)
                detections_for_map = {
                    k: filter_detections_by_score_threshold(
                        v, score_threshold, per_class_score_threshold, id_to_class
                    )
                    for k, v in all_detections.items()
                }
                total_dets = sum(len(dets) for dets in detections_for_map.values())
                total_gts = sum(len(gts) for gts in all_ground_truths.values())
                num_images = len(all_detections)
                if num_images == 0:
                    num_images = 1
                if rank is None or rank == 0:
                    print(f"\nComputing mAP (this may take a while)...")
                    print(
                        f"  Eval filters: score≥{score_threshold:.4f}, IoU≥{iou_threshold:.2f}"
                        + (
                            f", per-class floors on {len(per_class_score_threshold)} class(es)"
                            if per_class_score_threshold
                            else ""
                        )
                    )
                    print(f"  Images: {num_images:,}")
                    print(
                        f"  Total detections (post score filter): {total_dets:,} "
                        f"({total_dets/num_images:.1f} per image)"
                    )
                    print(f"  Total ground truths: {total_gts:,} ({total_gts/num_images:.1f} per image)")
                    if total_dets > 50000:
                        print(f"  Warning: Large number of detections may slow down mAP computation")
                        print(
                            f"  Consider increasing evaluation.score_threshold "
                            f"(current: {score_threshold}) for faster mAP matching"
                        )
                _map_t0 = time.perf_counter()
                mean_ap, class_aps, _class_metrics = compute_oriented_map(
                    detections=detections_for_map,
                    ground_truths=all_ground_truths,
                    iou_threshold=iou_threshold,
                    class_names=class_names,
                    device=device,
                    progress_stream=progress_stream,
                    use_exact_rotated_iou=eval_use_exact_rotated_iou,
                )
                _map_elapsed = time.perf_counter() - _map_t0
                summary["mAP"] = mean_ap
                for class_name, ap in class_aps.items():
                    summary[f"AP_{class_name}"] = ap
                if rank is None or rank == 0:
                    print(
                        f"  mAP computation completed in {_format_duration_hms(_map_elapsed)}. "
                        f"mAP: {mean_ap:.4f}"
                    )
                    if debug and class_aps:
                        print("  [debug] Per-class AP:", flush=True)
                        for cls_name, ap_val in sorted(class_aps.items(), key=lambda x: -x[1]):
                            print(f"    {cls_name}: {ap_val:.4f}", flush=True)
                        from collections import Counter
                        det_by_cls = Counter()
                        gt_by_cls = Counter()
                        for dets in detections_for_map.values():
                            for d in dets:
                                det_by_cls[d.class_name] += 1
                        for gts in all_ground_truths.values():
                            for g in gts:
                                gt_by_cls[g.class_name] += 1
                        print("  [debug] Detections per class:", dict(det_by_cls.most_common()), flush=True)
                        print("  [debug] Ground truths per class:", dict(gt_by_cls.most_common()), flush=True)
            except Exception as e:
                if rank is None or rank == 0:
                    print(f"Warning: Could not compute mAP: {e}")
                    import traceback
                    traceback.print_exc()
                summary["mAP"] = 0.0
    elif not compute_map:
        # mAP computation skipped
        summary["mAP"] = -1.0  # Use -1 to indicate it was skipped
    
    # Add class information
    summary["gt_classes"] = sorted(list(gt_classes_set))
    summary["pred_classes"] = sorted(list(pred_classes_set))
    
    # Compute accuracy (only when matching metrics were computed this epoch; otherwise the
    # counters are untouched zeros and printing them would look like a broken model).
    if compute_matching_metrics:
        if total_predictions > 0:
            summary["accuracy"] = total_correct / total_predictions
        else:
            summary["accuracy"] = 0.0
        summary["total_correct"] = total_correct
    
    summary["total_predictions"] = total_predictions
    summary["total_ground_truths"] = total_ground_truths
    summary["max_detection_score"] = max_detection_score
    if num_detection_scores > 0:
        summary["mean_detection_score"] = sum_detection_scores / num_detection_scores
    else:
        summary["mean_detection_score"] = 0.0

    # Log-only diagnostics (do not send to TensorBoard).
    summary["eval_score_threshold"] = score_threshold
    summary["eval_iou_threshold"] = iou_threshold
    if per_class_score_threshold:
        summary["eval_per_class_score_threshold"] = dict(per_class_score_threshold)
    if not compute_matching_metrics:
        pass  # GT-cover diagnostics skipped this epoch (same schedule as mAP)
    elif val_stats_extra:
        summary["log_only_gt_covered_pre_eval_threshold"] = val_stats_extra["gt_covered_pre_eval_threshold"]
        summary["log_only_gt_covered_post_eval_threshold"] = val_stats_extra["gt_covered_post_eval_threshold"]
        summary["log_only_gt_lost_by_eval_threshold"] = val_stats_extra["gt_lost_by_eval_threshold"]
        for _k, _v in val_stats_extra.items():
            if _k.startswith("log_only_gt_"):
                summary[_k] = _v
        if total_ground_truths > 0:
            summary["log_only_gt_cover_rate_pre_eval_threshold"] = (
                val_stats_extra["gt_covered_pre_eval_threshold"] / total_ground_truths
            )
            summary["log_only_gt_cover_rate_post_eval_threshold"] = (
                val_stats_extra["gt_covered_post_eval_threshold"] / total_ground_truths
            )
        else:
            summary["log_only_gt_cover_rate_pre_eval_threshold"] = 0.0
            summary["log_only_gt_cover_rate_post_eval_threshold"] = 0.0
    else:
        summary["log_only_gt_covered_pre_eval_threshold"] = gt_covered_pre_eval_threshold
        summary["log_only_gt_covered_post_eval_threshold"] = gt_covered_post_eval_threshold
        summary["log_only_gt_lost_by_eval_threshold"] = gt_lost_by_eval_threshold
        if total_ground_truths > 0:
            summary["log_only_gt_cover_rate_pre_eval_threshold"] = gt_covered_pre_eval_threshold / total_ground_truths
            summary["log_only_gt_cover_rate_post_eval_threshold"] = gt_covered_post_eval_threshold / total_ground_truths
        else:
            summary["log_only_gt_cover_rate_pre_eval_threshold"] = 0.0
            summary["log_only_gt_cover_rate_post_eval_threshold"] = 0.0

    # Debug: print GT cover rates and score stats to diagnose low mAP vs MMRotate.
    if debug and (rank is None or rank == 0):
        if total_ground_truths > 0:
            print(
                f"  [debug] GT cover (pre score thresh): {gt_covered_pre_eval_threshold}/{total_ground_truths} "
                f"({summary['log_only_gt_cover_rate_pre_eval_threshold']:.2%})",
                flush=True,
            )
            print(
                f"  [debug] GT cover (post score thresh): {gt_covered_post_eval_threshold}/{total_ground_truths} "
                f"({summary['log_only_gt_cover_rate_post_eval_threshold']:.2%}), "
                f"GT lost by threshold: {gt_lost_by_eval_threshold}",
                flush=True,
            )
        print(
            f"  [debug] Detection scores: mean={summary.get('mean_detection_score', 0):.4f}, "
            f"max={summary.get('max_detection_score', 0):.4f}, "
            f"total_predictions={total_predictions}, total_correct={total_correct}",
            flush=True,
        )
    
    # Build final list for TensorBoard: fixed image (selected) + optional random image(s)
    images_to_log: Optional[list[tuple[Any, str]]] = None
    if image_pool is not None and len(image_pool) > 0:
        images_to_log = []
        used_indices: set[int] = set()
        # Always include the fixed (selected) image: first image of first batch by default
        idx = min(fixed_image_index, len(image_pool) - 1)
        images_to_log.append(image_pool[idx])
        used_indices.add(idx)
        # Add one randomly selected image (different from fixed if pool has > 1)
        if log_random_image and len(image_pool) > 1:
            other_indices = [i for i in range(len(image_pool)) if i not in used_indices]
            if other_indices:
                random_idx = random.choice(other_indices)
                images_to_log.append(image_pool[random_idx])
                used_indices.add(random_idx)
        # Fill up to max_images_to_log with more random picks (without replacement)
        remaining = max_images_to_log - len(images_to_log)
        if remaining > 0:
            candidates = [i for i in range(len(image_pool)) if i not in used_indices]
            for _ in range(min(remaining, len(candidates))):
                pick = random.choice(candidates)
                candidates.remove(pick)
                images_to_log.append(image_pool[pick])

    # Log validation metrics to TensorBoard
    if writer is not None and epoch is not None:
        for key, value in summary.items():
            # Keep log-only diagnostics out of TensorBoard by design.
            if key.startswith("log_only_"):
                continue
            # Do not log mAP when it was not computed (sentinel -1)
            if key == "mAP" and value == -1.0:
                continue
            # Skip non-numeric values for scalar logging
            if isinstance(value, (int, float)):
                writer.add_scalar(f"val/{key}", value, epoch)
            elif isinstance(value, list):
                # Log class lists as text or count
                writer.add_scalar(f"val/{key}_count", len(value), epoch)
        
        # Log prediction images to TensorBoard
        if images_to_log is not None and len(images_to_log) > 0:
            try:
                # Create a grid of images for TensorBoard
                from torchvision.utils import make_grid
                vis_tensors = [img_tensor for img_tensor, _ in images_to_log]
                grid = make_grid(vis_tensors, nrow=min(2, len(vis_tensors)), padding=2)
                writer.add_image("val/predictions", grid, epoch)
            except Exception as e:
                # Silently skip if image logging fails (e.g., missing torchvision)
                pass
        # Log debug images: anchors and proposals (same random image as val/predictions when log_debug_anchors_proposals)
        if log_debug_anchors_proposals:
            try:
                if debug_anchors_vis is not None:
                    writer.add_image("val/debug_anchors", debug_anchors_vis, epoch)
                if debug_proposals_vis is not None:
                    writer.add_image("val/debug_proposals", debug_proposals_vis, epoch)
            except Exception:
                pass
    
    # Restore DDP sync state if we modified it
    if hasattr(model, 'require_backward_grad_sync') and original_sync_state is not None:
        model.require_backward_grad_sync = original_sync_state

    # DDP: rank 0 may run merge + stats + mAP for a long time after the val forward loop;
    # non-zero ranks used to return immediately and block on train()'s post-val barrier,
    # tripping Gloo's default 30-minute barrier timeout. Sync here so every rank leaves
    # evaluate() together (after rank 0's mAP / logging).
    if dist is not None and dist.is_initialized() and rank is not None:
        if dist.get_world_size() > 1:
            try:
                _gg = _get_gloo_gather_object_group()
                if _gg is not None:
                    dist.barrier(group=_gg)
                else:
                    dist.barrier()
            except TypeError:
                dist.barrier()

    return summary


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    num_epochs: int = 10,
    val_loader: Optional[DataLoader] = None,
    lr_scheduler: Optional[Any] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    print_freq: int = 10,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    loss_weights: Optional[Union[Dict[str, float], Callable[[int], Optional[Dict[str, float]]]]] = None,
    roi_class_weights: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    start_epoch: int = 0,
    writer: Optional[Any] = None,
    class_map: Optional[Dict[str, int]] = None,
    class_names: Optional[Sequence[str]] = None,
    eval_score_threshold: float = 0.05,
    eval_per_class_score_threshold: Optional[Dict[str, float]] = None,
    eval_vis_score_threshold: Optional[float] = None,
    eval_iou_threshold: float = 0.5,
    eval_extended_gt_metrics: bool = False,
    eval_compute_map_final: bool = True,
    eval_compute_map_every_n_epochs: int = 0,
    log_images: bool = True,
    max_images_to_log: int = 4,
    fixed_image_index: int = 0,
    log_random_image: bool = True,
    image_pool_max_size: int = 128,
    log_debug_anchors_proposals: bool = False,
    profiler: Optional[Any] = None,
    normalize_mean: Optional[list[float]] = None,
    normalize_std: Optional[list[float]] = None,
    vis_image_size: Optional[Tuple[int, int]] = None,
    train_sampler: Optional[Any] = None,
    val_sampler: Optional[Any] = None,
    rank: Optional[int] = None,
    progress_stream: Optional[Any] = None,
    debug: bool = False,
    lr_scheduler_plateau_metric: str = "total_loss",
    early_stop_patience: Optional[int] = None,
    early_stop_metric: str = "mAP",
    early_stop_min_delta: float = 0.0,
    early_stop_higher_is_better: Optional[bool] = None,
    freeze_backbone_epochs: int = 0,
    freeze_rpn_epochs: int = 0,
    eval_use_exact_rotated_iou: bool = True,
    eval_use_exact_rotated_iou_for_final_map: Optional[bool] = None,
) -> Dict[str, Any]:
    """Complete training loop.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        val_loader: Optional validation DataLoader
        lr_scheduler: Optional learning rate scheduler
        checkpoint_manager: Optional checkpoint manager
        print_freq: Frequency of printing metrics
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm
        use_amp: Use automatic mixed precision
        loss_weights: Loss component weights
        start_epoch: Starting epoch (for resuming)
        writer: Optional TensorBoard SummaryWriter for logging
        class_map: Optional mapping from class names to class IDs (for evaluation)
        class_names: Optional list of class names (for evaluation)
        eval_score_threshold: Minimum confidence score for evaluation detections
        eval_iou_threshold: IoU threshold for mAP calculation
        eval_compute_map_final: If True, after training load the best checkpoint and compute mAP once
        eval_compute_map_every_n_epochs: If >0, compute mAP every N epochs during training (current model)
            (in addition to the final epoch). Use 0 to compute only at final epoch.
        log_images: Whether to log prediction images to TensorBoard
        max_images_to_log: Maximum number of images to log per epoch
        fixed_image_index: Index in validation order of the image to always log (0 = first image of first batch)
        log_random_image: If True, also log one randomly selected image from the pool
        image_pool_max_size: Maximum size of the pool from which random images are drawn
        log_debug_anchors_proposals: If True, log anchors and RPN proposals to TensorBoard (random val image each epoch; for debugging).
        profiler: Optional TrainingProfiler instance for performance profiling (should be used as context manager)
        normalize_mean: Normalization mean values [R, G, B] for denormalization. Defaults to ImageNet mean.
        normalize_std: Normalization std values [R, G, B] for denormalization. Defaults to ImageNet std.
        vis_image_size: Optional (height, width) for TensorBoard prediction images; when set, image is cropped to this size so boxes align (default (1024, 1024) when passed from train script).
        train_sampler: Optional DistributedSampler for training (call set_epoch each epoch)
        rank: Optional rank for DDP (only rank 0 saves checkpoints and runs validation)
        progress_stream: If set (e.g. original stderr when stdout is teed to a log file),
            tqdm writes here so the progress bar is not duplicated in the log file.
        early_stop_patience: If set, stop after this many epochs without improvement on early_stop_metric
            (skipped epochs where mAP was not computed, e.g. mAP=-1).
        early_stop_metric: Validation metric key to monitor (default mAP).
        early_stop_min_delta: Minimum relative change to count as improvement.
        early_stop_higher_is_better: If None, inferred from metric name (mAP / AP_* -> True).
        freeze_backbone_epochs: If >0, freeze ``backbone.*`` while ``epoch <`` this value (0-based).
        freeze_rpn_epochs: If >0, freeze ``rpn_head.*`` while ``epoch <`` this value; no-op without RPN.
    
    Returns:
        Dictionary with training history
    """
    _require_torch()
    
    train_tracker = MetricTracker()
    val_tracker = MetricTracker() if val_loader is not None else None
    
    history = {
        "train": [],
        "val": [],
        "early_stopped": False,
    }
    
    # Calculate global step offset based on start_epoch
    global_step = start_epoch * len(train_loader) if start_epoch > 0 else 0
    
    # Track previous epoch's validation metrics for delta computation
    previous_val_metrics: Optional[Dict[str, Any]] = None
    
    _es_patience = int(early_stop_patience) if early_stop_patience is not None else 0
    _es_metric_key = (early_stop_metric or "mAP").strip()
    _es_best: Optional[float] = None
    _es_epochs_no_improve = 0
    
    def _infer_early_stop_higher_is_better() -> bool:
        if early_stop_higher_is_better is not None:
            return bool(early_stop_higher_is_better)
        lk = _es_metric_key.lower()
        if lk == "map" or lk.startswith("ap_") or lk.startswith("ap"):
            return True
        if "loss" in lk:
            return False
        return True
    
    _es_higher = _infer_early_stop_higher_is_better()
    
    epoch_wall_times: List[float] = []
    train_loop_t0 = time.perf_counter()
    train_started_at_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    if rank is None or rank == 0:
        print(f"Training started at: {train_started_at_iso}")

    # After partial freeze (freeze_backbone_epochs / freeze_rpn_epochs), all parameters train; rebuild DDP
    # with find_unused_parameters=False to avoid extra autograd traversals and PyTorch warnings.
    ddp_rebuilt_after_partial_freeze = False
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_t0 = time.perf_counter()
        val_metrics: Dict[str, Any] = {}
        if rank is None or rank == 0:
            epoch_idx = (epoch - start_epoch + 1)
            print(f"\nEpoch {epoch_idx}/{num_epochs}")
            print("-" * 50)
        
        # Set epoch for DistributedSampler (required for proper shuffling across epochs)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Also set epoch for validation sampler if using DistributedSampler
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        train_model = model.module if hasattr(model, 'module') else model
        _fb = int(freeze_backbone_epochs or 0)
        _fr = int(freeze_rpn_epochs or 0)
        if _fb > 0 or _fr > 0:
            freeze_bb = epoch < _fb
            freeze_rpn = epoch < _fr
            set_backbone_requires_grad(train_model, freeze=freeze_bb)
            set_rpn_requires_grad(train_model, freeze=freeze_rpn)
            if rank is None or rank == 0:
                if (
                    epoch == start_epoch
                    or (_fb > 0 and epoch == _fb)
                    or (_fr > 0 and epoch == _fr)
                ):
                    parts = []
                    if _fb > 0:
                        parts.append(
                            f"backbone {'FROZEN' if freeze_bb else 'trainable'} (freeze_backbone_epochs={_fb})"
                        )
                    if _fr > 0 and model_has_rpn_head(train_model):
                        parts.append(
                            f"RPN {'FROZEN' if freeze_rpn else 'trainable'} (freeze_rpn_epochs={_fr})"
                        )
                    if parts:
                        print(f"  {' | '.join(parts)} — loop epoch {epoch}")

        _freeze_span = max(_fb, _fr)
        if (
            not ddp_rebuilt_after_partial_freeze
            and _freeze_span > 0
            and epoch >= _freeze_span
            and dist is not None
            and dist.is_initialized()
            and isinstance(model, nn.parallel.DistributedDataParallel)
        ):
            inner = model.module
            model = nn.parallel.DistributedDataParallel(
                inner,
                device_ids=None,
                output_device=None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            ddp_rebuilt_after_partial_freeze = True
            if rank is None or rank == 0:
                print(
                    "  DDP: rebuilt with find_unused_parameters=False (partial freeze ended; faster reductions)."
                )

        # Update NMS IoU threshold from schedule (e.g. Rotated RetinaNet: lower = more suppression)
        if hasattr(train_model, "set_final_nms_iou_for_epoch"):
            train_model.set_final_nms_iou_for_epoch(epoch)
        if hasattr(train_model, "set_roi_box_reg_iou_weight_for_epoch"):
            train_model.set_roi_box_reg_iou_weight_for_epoch(epoch)
        elif hasattr(train_model, "set_box_reg_iou_weight_for_epoch"):
            train_model.set_box_reg_iou_weight_for_epoch(epoch)
        if hasattr(train_model, "set_roi_box_reg_angle_weight_for_epoch"):
            train_model.set_roi_box_reg_angle_weight_for_epoch(epoch)
        # Update ROI class weights from schedule (optional; supports ramp-up to avoid early classifier collapse).
        if roi_class_weights is not None:
            try:
                w = roi_class_weights(epoch)
            except TypeError:
                # Backward compat: tolerate a constant tensor passed accidentally.
                w = roi_class_weights  # type: ignore[assignment]
            if w is not None:
                w = w.to(device=device, dtype=torch.float32)
                if hasattr(train_model, "set_class_weights_tensor"):
                    train_model.set_class_weights_tensor(w)
                elif hasattr(train_model, "roi_class_weights_tensor"):
                    train_model.roi_class_weights_tensor = w  # type: ignore[attr-defined]

        if hasattr(train_model, "set_grouped_ce_alpha_for_epoch"):
            train_model.set_grouped_ce_alpha_for_epoch(epoch)
        
        # Get loss weights for this epoch (support callable for dynamic weights)
        epoch_loss_weights = None
        if loss_weights is not None:
            if callable(loss_weights):
                epoch_loss_weights = loss_weights(epoch)
            else:
                epoch_loss_weights = loss_weights
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epoch_idx=(epoch - start_epoch + 1),
            epoch_count=num_epochs,
            print_freq=print_freq,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            metric_tracker=train_tracker,
            loss_weights=epoch_loss_weights,
            writer=writer,
            global_step=global_step,
            profiler=profiler,
            lr_scheduler=lr_scheduler,
            rank=rank,
            progress_stream=progress_stream,
            debug=debug,
        )
        # Update global step from metrics if available
        if writer is not None and "_global_step" in train_metrics:
            global_step = train_metrics.pop("_global_step")
        history["train"].append(train_metrics)
        
        # Synchronize all ranks before validation (ensure training is complete on all ranks)
        # CRITICAL: All ranks must reach this barrier before validation starts
        # This prevents DDP operations in the next training epoch from interfering with validation
        if rank is not None and dist is not None and dist.is_initialized():
            dist.barrier()
        
        # Validate: All ranks participate in forward pass (prevents DDP deadlock)
        # Only rank 0 collects/logs metrics to avoid duplication
        if val_loader is not None and val_tracker is not None:
            # Compute mAP every N epochs during training (optional). Final mAP uses best model after loop.
            epoch_idx = (epoch - start_epoch + 1)
            map_every_n = max(0, int(eval_compute_map_every_n_epochs or 0))
            is_periodic_map_epoch = map_every_n > 0 and (epoch_idx % map_every_n == 0)
            compute_map_this_epoch = is_periodic_map_epoch

            if compute_map_this_epoch and (rank is None or rank == 0):
                print(f"\nComputing mAP (periodic evaluation: every {map_every_n} epoch(s))...")
            # Keep expensive GT–IoU diagnostics on the same schedule as mAP by default.
            extended_gt_this_epoch = bool(eval_extended_gt_metrics) and bool(compute_map_this_epoch)
            matching_metrics_this_epoch = compute_map_this_epoch or extended_gt_this_epoch
            
            # All ranks run validation forward pass (prevents DDP deadlock)
            # Only rank 0 collects metrics and logs
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                metric_tracker=val_tracker if (rank is None or rank == 0) else None,
                writer=writer if (rank is None or rank == 0) else None,
                epoch=epoch,
                class_map=class_map,
                class_names=class_names,
                score_threshold=eval_score_threshold,
                per_class_score_threshold=eval_per_class_score_threshold,
                vis_score_threshold=eval_vis_score_threshold,
                iou_threshold=eval_iou_threshold,
                extended_gt_metrics=extended_gt_this_epoch,
                compute_map=compute_map_this_epoch and (rank is None or rank == 0),  # Only rank 0 computes mAP
                log_images=log_images and (rank is None or rank == 0),  # Only rank 0 logs images
                max_images_to_log=max_images_to_log,
                fixed_image_index=fixed_image_index,
                log_random_image=log_random_image,
                image_pool_max_size=image_pool_max_size,
                log_debug_anchors_proposals=log_debug_anchors_proposals,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                vis_image_size=vis_image_size,
                rank=rank,
                progress_stream=progress_stream,
                debug=debug,
                eval_use_exact_rotated_iou=eval_use_exact_rotated_iou,
                compute_matching_metrics=matching_metrics_this_epoch,
            )
            # Only rank 0 has meaningful metrics; other ranks return empty dict
            if rank is None or rank == 0:
                history["val"].append(val_metrics)
                print(_format_validation_metrics(val_metrics, previous_val_metrics))
                previous_val_metrics = _snapshot_val_metrics_for_comparison(
                    val_metrics, previous_val_metrics
                )
            else:
                # Non-zero ranks return empty metrics (they still ran forward pass for DDP sync)
                history["val"].append({})
        
        # Synchronize all DDP ranks after validation.
        #
        # IMPORTANT: rank 0 may spend a long time computing mAP while other ranks wait.
        # Using an NCCL barrier can trip the NCCL watchdog timeout and abort the job.
        # Prefer a Gloo barrier when available (we already create a Gloo group for gather_object).
        if rank is not None and dist is not None and dist.is_initialized():
            try:
                gather_group = _get_gloo_gather_object_group()
                if gather_group is not None:
                    dist.barrier(group=gather_group)
                else:
                    dist.barrier()
            except TypeError:
                # Older PyTorch: barrier may not accept group=
                dist.barrier()
        
        # Learning rate scheduling (per epoch for standard schedulers)
        # Note: Warmup schedulers are stepped per optimizer step in train_one_epoch
        if lr_scheduler is not None:
            # Handle warmup scheduler's base scheduler stepping per epoch
            if hasattr(lr_scheduler, 'step_epoch'):
                # Custom warmup scheduler - step base scheduler per epoch
                lr_scheduler.step_epoch()
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loader is not None and val_metrics:
                    metric_val = val_metrics.get(
                        lr_scheduler_plateau_metric,
                        train_metrics.get(lr_scheduler_plateau_metric, 0.0),
                    )
                    # Do not step on sentinel mAP=-1 when periodic mAP was skipped this epoch
                    if (
                        lr_scheduler_plateau_metric.strip().lower() == "map"
                        and isinstance(metric_val, (int, float))
                        and float(metric_val) == -1.0
                    ):
                        pass
                    else:
                        lr_scheduler.step(metric_val)
                else:
                    metric_val = train_metrics.get(lr_scheduler_plateau_metric, 0.0)
                    lr_scheduler.step(metric_val)
            else:
                # Standard PyTorch scheduler (StepLR, etc.) — step once per epoch.
                # PyTorch 2.x uses LRScheduler; older versions use _LRScheduler.
                _base = getattr(torch.optim.lr_scheduler, 'LRScheduler', None) or getattr(torch.optim.lr_scheduler, '_LRScheduler', None)
                if _base is not None and isinstance(lr_scheduler, _base):
                    lr_scheduler.step()
            # Custom warmup schedulers are already stepped per optimizer step in train_one_epoch
        
        # Checkpointing (only on rank 0 for DDP, or all ranks if rank is None)
        if checkpoint_manager is not None and (rank is None or rank == 0):
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch,
                metrics=train_metrics,
            )
            if val_loader is not None and (rank is None or rank == 0):
                # Use val_metrics if available, otherwise train_metrics
                metrics_for_best = (
                    val_metrics if (len(history["val"]) > 0 and history["val"][-1]) else train_metrics
                )
            else:
                # No validation pass: best checkpoint follows training metrics
                metrics_for_best = train_metrics
            checkpoint_manager.save_best(
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch,
                metrics=metrics_for_best,
            )
        
        if rank is None or rank == 0:
            epoch_wall_s = time.perf_counter() - epoch_t0
            epoch_wall_times.append(epoch_wall_s)
            avg_s = sum(epoch_wall_times) / len(epoch_wall_times)
            remaining_epochs = start_epoch + num_epochs - epoch - 1
            map_n = max(0, int(eval_compute_map_every_n_epochs or 0))
            want_final_map = bool(eval_compute_map_final and val_loader is not None)
            parts = [
                f"  Timing: {_format_duration_hms(epoch_wall_s)} this epoch (avg {_format_duration_hms(avg_s)})",
            ]
            if remaining_epochs > 0:
                eta_total = avg_s * remaining_epochs + (avg_s if want_final_map else 0.0)
                second = (
                    f"ETA ~{_format_duration_hms(eta_total)} for {remaining_epochs} epoch(s) left"
                )
                if map_n > 0:
                    second += f", mAP every {map_n} epoch(s)"
                if want_final_map:
                    second += " and final mAP."
                else:
                    second += "."
                parts.append(second)
            elif want_final_map:
                parts.append(f"ETA ~{_format_duration_hms(avg_s)} for final mAP on best checkpoint.")
            else:
                parts.append("Training loop finished (no final mAP).")
            print(" | ".join(parts))
        
        # Early stopping on validation metric (rank 0 decides; broadcast to all DDP ranks)
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if _es_patience > 0 and val_loader is not None and (rank is None or rank == 0):
            raw = val_metrics.get(_es_metric_key) if val_metrics else None
            valid = False
            v = 0.0
            if raw is not None:
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    valid = False
                else:
                    valid = not math.isnan(v) and not (_es_metric_key.lower() == "map" and v == -1.0)
            if valid:
                if _es_best is None:
                    _es_best = v
                    _es_epochs_no_improve = 0
                else:
                    improved = (
                        (v > _es_best + early_stop_min_delta)
                        if _es_higher
                        else (v < _es_best - early_stop_min_delta)
                    )
                    if improved:
                        _es_best = v
                        _es_epochs_no_improve = 0
                    else:
                        _es_epochs_no_improve += 1
                if _es_epochs_no_improve >= _es_patience:
                    stop_flag[0] = 1
                    history["early_stopped"] = True
                    history["early_stop_epoch"] = int(epoch)
                    history["early_stop_best_metric"] = float(_es_best) if _es_best is not None else None
                    print(
                        f"\nEarly stopping: {_es_metric_key} did not improve for {_es_patience} "
                        f"epoch(s) (best={_es_best:.6f})."
                    )
        if rank is not None and dist is not None and dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() != 0:
            break
        
        # Free unused GPU memory between epochs to reduce OOM risk from fragmentation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Final mAP with best checkpoint (after training loop)
    # DDP: all ranks must call evaluate() because evaluate() uses dist.gather_object on the val set.
    # If only rank 0 runs evaluate(), gather_object waits forever for other ranks → hung job, no mAP in log.
    _ran_final_map_evaluate = False
    if eval_compute_map_final and val_loader is not None:
        best_path = checkpoint_manager.best_checkpoint_path if checkpoint_manager is not None else None
        # In DDP, only rank 0 updates checkpoint_manager.best_checkpoint_path during save_best().
        # Broadcast the resolved best checkpoint path so every rank enters final evaluate()
        # and participates in the validation collectives.
        if rank is not None and dist is not None and dist.is_initialized():
            best_path_str = str(best_path) if best_path is not None else None
            best_path_list = [best_path_str] if rank == 0 else [None]
            dist.broadcast_object_list(best_path_list, src=0)
            best_path = Path(best_path_list[0]) if best_path_list[0] else None
        if best_path is not None and best_path.exists():
            _ran_final_map_evaluate = True
            _final_map_exact_iou = (
                eval_use_exact_rotated_iou
                if eval_use_exact_rotated_iou_for_final_map is None
                else bool(eval_use_exact_rotated_iou_for_final_map)
            )
            if rank is None or rank == 0:
                print("\nComputing final mAP using best checkpoint...")
                print(
                    f"  Final mAP IoU backend: "
                    f"{'exact CPU polygon' if _final_map_exact_iou else 'GPU sampling (approx)'}"
                )
            ckpt = torch.load(best_path, map_location=device)
            model_to_load = model.module if hasattr(model, "module") else model
            state_dict = ckpt["model_state_dict"]
            # Strip DDP "module." prefix so keys match the unwrapped model (same machine, saved under DDP)
            ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
            model_has_module = any(k.startswith("module.") for k in model_to_load.state_dict().keys())
            if ckpt_has_module and not model_has_module:
                state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
            elif model_has_module and not ckpt_has_module:
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            model_to_load.load_state_dict(state_dict, strict=True)
            final_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                metric_tracker=val_tracker if (rank is None or rank == 0) else None,
                writer=writer if (rank is None or rank == 0) else None,
                epoch=start_epoch + num_epochs - 1,
                class_map=class_map,
                class_names=class_names,
                score_threshold=eval_score_threshold,
                per_class_score_threshold=eval_per_class_score_threshold,
                vis_score_threshold=eval_vis_score_threshold,
                iou_threshold=eval_iou_threshold,
                extended_gt_metrics=eval_extended_gt_metrics,
                compute_map=(rank is None or rank == 0),
                log_images=False,
                max_images_to_log=0,
                fixed_image_index=fixed_image_index,
                log_random_image=False,
                image_pool_max_size=image_pool_max_size,
                log_debug_anchors_proposals=False,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                vis_image_size=vis_image_size,
                rank=rank,
                progress_stream=progress_stream,
                debug=debug,
                eval_use_exact_rotated_iou=_final_map_exact_iou,
                compute_matching_metrics=True,
            )
            if rank is None or rank == 0:
                print(_format_validation_metrics(final_metrics))
        elif rank is None or rank == 0:
            print("\nSkipping final mAP: no best checkpoint saved.")

    # DDP: rank 0 runs compute_oriented_map inside evaluate() (minutes); other ranks skip mAP and return
    # from evaluate() first. Without this barrier they exit train() and destroy_process_group() while
    # rank 0 is still in mAP → NCCL ALLREDUCE watchdog timeout (default ~30 min) on shutdown.
    if (
        _ran_final_map_evaluate
        and rank is not None
        and dist is not None
        and dist.is_initialized()
    ):
        dist.barrier()
    
    # Close TensorBoard writer if provided
    if writer is not None:
        writer.close()
    
    finished_at_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    total_wall_s = time.perf_counter() - train_loop_t0
    timing: Dict[str, Any] = {
        "started_at": train_started_at_iso,
        "finished_at": finished_at_iso,
        "total_wall_seconds": float(total_wall_s),
    }
    if epoch_wall_times:
        timing["epoch_wall_times_seconds"] = list(epoch_wall_times)
        timing["epochs_completed"] = len(epoch_wall_times)
        timing["mean_epoch_seconds"] = float(sum(epoch_wall_times) / len(epoch_wall_times))
        timing["min_epoch_seconds"] = float(min(epoch_wall_times))
        timing["max_epoch_seconds"] = float(max(epoch_wall_times))
    history["timing"] = timing
    if rank is None or rank == 0:
        print("\n" + "=" * 80)
        print("Training timing summary")
        print("=" * 80)
        print(f"  Started:   {train_started_at_iso}")
        print(f"  Finished:  {finished_at_iso}")
        print(f"  Total wall: {_format_duration_hms(total_wall_s)} ({total_wall_s:.1f} s)")
        if epoch_wall_times:
            print(
                f"  Epochs:    {len(epoch_wall_times)} completed "
                f"(mean {_format_duration_hms(timing['mean_epoch_seconds'])}, "
                f"min {_format_duration_hms(timing['min_epoch_seconds'])}, "
                f"max {_format_duration_hms(timing['max_epoch_seconds'])})"
            )
        else:
            print("  Epochs:    no epoch timing recorded (0 loop iterations).")
        if history.get("early_stopped"):
            print("  Note:      training stopped early (early stopping).")
        print("=" * 80)
    
    return history


def _is_skipped_map_value(value: Any) -> bool:
    """True when mAP was not computed this epoch (sentinel -1)."""
    return isinstance(value, (int, float)) and float(value) == -1.0


def _snapshot_val_metrics_for_comparison(
    metrics: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Snapshot for epoch-over-epoch deltas; keep last computed mAP/AP when skipped."""
    snap = dict(metrics)
    if _is_skipped_map_value(snap.get("mAP")) and prior is not None:
        prev_map = prior.get("mAP")
        if isinstance(prev_map, (int, float)) and not _is_skipped_map_value(prev_map):
            snap["mAP"] = prev_map
            for key, value in prior.items():
                if key.startswith("AP_"):
                    snap[key] = value
    return snap


def _format_validation_metrics(metrics: Dict[str, Any], previous_metrics: Optional[Dict[str, Any]] = None) -> str:
    """Format validation metrics for human-readable display with optional delta from previous epoch.
    
    Args:
        metrics: Dictionary of validation metrics
        previous_metrics: Optional dictionary of previous epoch's metrics for delta calculation
        
    Returns:
        Formatted string with one metric per line, including deltas where applicable
    """
    def _format_with_delta(key: str, label: str, value: Any, fmt: str = ":.6f", reverse_sign: bool = False) -> str:
        """Format a metric value with optional delta from previous epoch."""
        if not isinstance(value, (int, float)):
            return f"  {label}: {value}"
        
        # Strip colon from format string if present (format() expects .2f, not :.2f)
        fmt_clean = fmt.lstrip(":") if fmt.startswith(":") else fmt
        
        # Format the current value
        if isinstance(value, float):
            formatted_value = format(value, fmt_clean)
        else:
            formatted_value = str(value)
        
        line = f"  {label}: {formatted_value}"
        
        # Add delta if previous metrics are available
        if previous_metrics is not None and key in previous_metrics:
            prev_value = previous_metrics[key]
            if key == "mAP" and (
                _is_skipped_map_value(value) or _is_skipped_map_value(prev_value)
            ):
                return line
            if isinstance(prev_value, (int, float)) and not _is_skipped_map_value(value):
                delta = value - prev_value
                if reverse_sign:
                    delta = -delta  # For losses, positive delta means improvement
                
                delta_str = format(abs(delta), fmt_clean)
                if delta > 0:
                    line += f"  ↑{delta_str}"
                elif delta < 0:
                    line += f"  ↓{delta_str}"
                else:
                    line += f"  →{delta_str}"
        
        return line
    
    lines = []
    lines.append("Validation Metrics:")
    lines.append("-" * 50)

    eval_sc = metrics.get("eval_score_threshold")
    eval_iou = metrics.get("eval_iou_threshold", metrics.get("iou_threshold"))
    if isinstance(eval_sc, (int, float)) or isinstance(eval_iou, (int, float)):
        sc_part = f"score≥{float(eval_sc):.4f}" if isinstance(eval_sc, (int, float)) else "score=?"
        iou_part = f"IoU≥{float(eval_iou):.2f}" if isinstance(eval_iou, (int, float)) else "IoU=?"
        lines.append(f"  Eval thresholds (mAP / matching): {sc_part}, {iou_part}")
    eval_pc = metrics.get("eval_per_class_score_threshold")
    if isinstance(eval_pc, dict) and eval_pc:
        lines.append(f"  Per-class score floors: {eval_pc}")
    
    # Loss metrics (lower is better, so reverse sign for delta)
    if "total_loss" in metrics:
        lines.append(_format_with_delta("total_loss", "Total Loss", metrics["total_loss"], reverse_sign=True))
    if "loss_classifier" in metrics:
        lines.append(_format_with_delta("loss_classifier", "Classifier Loss", metrics["loss_classifier"], reverse_sign=True))
    if "loss_box_reg" in metrics:
        lines.append(_format_with_delta("loss_box_reg", "Box Regression Loss", metrics["loss_box_reg"], reverse_sign=True))
    if "loss_objectness" in metrics:
        lines.append(_format_with_delta("loss_objectness", "Objectness Loss", metrics["loss_objectness"], reverse_sign=True))
    if "loss_rpn_box_reg" in metrics:
        lines.append(_format_with_delta("loss_rpn_box_reg", "RPN Box Regression Loss", metrics["loss_rpn_box_reg"], reverse_sign=True))
    
    # Detection metrics (num_detections uses eval score threshold; num_detections_050 is more informative when model is overconfident)
    if "num_detections" in metrics:
        det_thr_label = (
            f"score≥{float(eval_sc):.4f}"
            if isinstance(eval_sc, (int, float))
            else "at eval threshold"
        )
        lines.append(
            _format_with_delta(
                "num_detections",
                f"Avg Detections per Image ({det_thr_label})",
                metrics["num_detections"],
                ":.2f",
            )
        )
    if "num_detections_050" in metrics:
        lines.append(_format_with_delta("num_detections_050", "Avg Detections per Image (score≥0.5)", metrics["num_detections_050"], ":.2f"))
    if "max_detection_score" in metrics:
        lines.append(f"  Max detection score: {metrics['max_detection_score']:.3f}")
    if "mean_detection_score" in metrics:
        lines.append(f"  Mean detection score: {metrics['mean_detection_score']:.3f}")
    if "time_per_step" in metrics:
        lines.append(_format_with_delta("time_per_step", "Time per Step", metrics["time_per_step"], ":.4f", reverse_sign=True))
    
    # Accuracy metrics (higher is better)
    if "accuracy" in metrics:
        acc = metrics["accuracy"]
        lines.append(_format_with_delta("accuracy", "Accuracy", acc, ":.4f") + f" ({acc*100:.2f}%)")
    else:
        lines.append("  Accuracy / GT cover: (skipped — computed on mAP epochs)")
    if "total_correct" in metrics and "total_predictions" in metrics:
        # Denominator is number of model predictions; when 0/0 the model made no detections above threshold
        lines.append(f"  Correct Predictions: {metrics['total_correct']}/{metrics['total_predictions']} (matched detections)")
    if "total_ground_truths" in metrics:
        lines.append(f"  Ground Truth objects: {metrics['total_ground_truths']}")
    if "log_only_gt_covered_pre_eval_threshold" in metrics:
        lines.append(
            f"  GT covered pre-eval-threshold: "
            f"{int(metrics['log_only_gt_covered_pre_eval_threshold'])}/"
            f"{int(metrics.get('total_ground_truths', 0))}"
        )
    if "log_only_gt_covered_post_eval_threshold" in metrics:
        lines.append(
            f"  GT covered post-eval-threshold: "
            f"{int(metrics['log_only_gt_covered_post_eval_threshold'])}/"
            f"{int(metrics.get('total_ground_truths', 0))}"
        )
    if "log_only_gt_lost_by_eval_threshold" in metrics:
        lines.append(
            f"  GT lost by eval-threshold filtering: "
            f"{int(metrics['log_only_gt_lost_by_eval_threshold'])}"
        )
    if "log_only_gt_cover_rate_pre_eval_threshold" in metrics:
        lines.append(
            f"  GT cover rate pre-eval-threshold: "
            f"{float(metrics['log_only_gt_cover_rate_pre_eval_threshold']):.2%}"
        )
    if "log_only_gt_cover_rate_post_eval_threshold" in metrics:
        lines.append(
            f"  GT cover rate post-eval-threshold: "
            f"{float(metrics['log_only_gt_cover_rate_post_eval_threshold']):.2%}"
        )

    if "log_only_gt_mean_best_iou_any" in metrics:
        thr = metrics.get("eval_iou_threshold", metrics.get("iou_threshold", 0.5))
        if not isinstance(thr, (int, float)):
            thr = 0.5
        lines.append(
            f"  GT IoU vs raw dets (eval IoU threshold={float(thr):.2f}; "
            "per-GT max IoU over all dets / over same-class dets):"
        )
        lines.append(
            f"    mean best IoU (any class): {float(metrics['log_only_gt_mean_best_iou_any']):.4f}, "
            f"median: {float(metrics.get('log_only_gt_median_best_iou_any', 0.0)):.4f}"
        )
        lines.append(
            f"    mean best IoU (correct class): {float(metrics['log_only_gt_mean_best_iou_same_class']):.4f}, "
            f"median: {float(metrics.get('log_only_gt_median_best_iou_same_class', 0.0)):.4f}"
        )
        align_raw = metrics.get("log_only_gt_alignment_metrics")
        if isinstance(align_raw, dict) and align_raw.get("per_class"):
            table = format_gt_best_iou_alignment_table_from_dict(align_raw)
            if table:
                lines.append("    per-class mean best IoU (raw detections):")
                for table_line in table.splitlines():
                    lines.append(f"      {table_line}")
    if "log_only_gt_count_wrong_class_overlap_at_iou" in metrics:
        wco = int(metrics["log_only_gt_count_wrong_class_overlap_at_iou"])
        nhi = int(metrics.get("log_only_gt_count_no_det_iou_above_threshold", 0))
        lines.append(
            f"    GTs with IoU≥thresh but no same-class hit (wrong-class / misaligned): {wco}"
        )
        lines.append(
            f"    GTs with no detection above IoU thresh (missed / poor loc): {nhi}"
        )
    if "log_only_gt_count_zero_best_iou_any" in metrics:
        zcnt = int(metrics["log_only_gt_count_zero_best_iou_any"])
        zrate = float(metrics.get("log_only_gt_rate_zero_best_iou_any", 0.0))
        lines.append(
            f"    GTs with 0% best IoU vs any detection (no spatial overlap): "
            f"{zcnt} ({zrate:.2%} of GTs)"
        )
    if "log_only_gt_best_iou_any_buckets" in metrics and isinstance(
        metrics["log_only_gt_best_iou_any_buckets"], list
    ):
        ba = metrics["log_only_gt_best_iou_any_buckets"]
        bs = metrics.get("log_only_gt_best_iou_same_class_buckets")
        if len(ba) == 4:
            lines.append(
                f"    histogram best IoU (any class) "
                f"[0,0.25)/[0.25,0.5)/[0.5,0.75)/[0.75,1]: "
                f"{ba[0]}/{ba[1]}/{ba[2]}/{ba[3]}"
            )
        if isinstance(bs, list) and len(bs) == 4:
            lines.append(
                f"    histogram best IoU (correct class) "
                f"[0,0.25)/[0.25,0.5)/[0.5,0.75)/[0.75,1]: "
                f"{bs[0]}/{bs[1]}/{bs[2]}/{bs[3]}"
            )

    if "log_only_gt_class_agnostic_dup_rate_micro" in metrics:
        thr_d = metrics.get("eval_iou_threshold", metrics.get("iou_threshold", 0.5))
        if not isinstance(thr_d, (int, float)):
            thr_d = 0.5
        micro = float(metrics["log_only_gt_class_agnostic_dup_rate_micro"])
        macro = float(metrics.get("log_only_gt_class_agnostic_dup_rate_macro", 0.0))
        red = int(metrics.get("log_only_gt_class_agnostic_dup_redundant_boxes", 0))
        tot = int(metrics.get("log_only_gt_class_agnostic_dup_post_threshold_boxes", 0))
        nimg = int(metrics.get("log_only_gt_class_agnostic_dup_macro_image_count", 0))
        lines.append(
            f"  Class-agnostic duplicate rate (post score thresh; IoU≥{float(thr_d):.2f} vs a higher-score box): "
            f"micro={micro:.2%} ({red}/{tot} boxes), "
            f"macro mean over images with detections={macro:.2%} ({nimg} images)"
        )

    # mAP metrics (higher is better)
    if "mAP" in metrics:
        map_val = metrics["mAP"]
        if map_val == -1.0:
            lines.append("  mAP: (skipped)")
        else:
            delta_line = _format_with_delta("mAP", "mAP", map_val, ":.4f")
            lines.append(delta_line + f" ({map_val*100:.2f}%)")
    
    # Per-class APs (sorted)
    ap_keys = sorted([k for k in metrics.keys() if k.startswith("AP_")])
    if ap_keys:
        lines.append("  Per-Class AP:")
        for ap_key in ap_keys:
            class_name = ap_key[3:]  # Remove "AP_" prefix
            ap_val = metrics[ap_key]
            if previous_metrics is not None and ap_key in previous_metrics:
                prev_ap = previous_metrics[ap_key]
                if isinstance(prev_ap, (int, float)) and not _is_skipped_map_value(
                    metrics.get("mAP")
                ):
                    delta = ap_val - prev_ap
                    delta_str = format(abs(delta), ".4f")
                    if delta > 0:
                        delta_indicator = f"  ↑{delta_str}"
                    elif delta < 0:
                        delta_indicator = f"  ↓{delta_str}"
                    else:
                        delta_indicator = f"  →{delta_str}"
                else:
                    delta_indicator = ""
            else:
                delta_indicator = ""
            lines.append(f"    {class_name}: {ap_val:.4f}{delta_indicator} ({ap_val*100:.2f}%)")
    
    # Class information
    if "gt_classes" in metrics and isinstance(metrics["gt_classes"], list):
        gt_classes = metrics["gt_classes"]
        if gt_classes:
            lines.append(f"  Ground Truth Classes: {len(gt_classes)} classes")
            if len(gt_classes) <= 10:
                lines.append(f"    {', '.join(map(str, gt_classes))}")
    
    if "pred_classes" in metrics and isinstance(metrics["pred_classes"], list):
        pred_classes = metrics["pred_classes"]
        if pred_classes:
            lines.append(f"  Predicted Classes: {len(pred_classes)} classes")
            if len(pred_classes) <= 10:
                lines.append(f"    {', '.join(map(str, pred_classes))}")
    
    return "\n".join(lines)


__all__ = [
    "train_one_epoch",
    "evaluate",
    "train",
]
