#!/usr/bin/env python3
"""Learning rate finder for oriented object detection training.

Runs a short training sweep with exponentially increasing learning rate and records
loss vs LR. Stops early when smoothed loss exceeds ``stop_mult`` times the best
smoothed loss so far (fastai ``LRFinder`` / ``stop_div``). Before running fastai-style
heuristics, the trace is truncated after the first divergence on **raw** loss
(``loss > stop_mult ×`` running minimum). Applies valley / steep / minimum / slide on
that prefix plus the usual fastai trim. Single-GPU only.

Usage:
    python tools/lr_finder.py --config configs/rotated_faster_rcnn/dota_le90_3x.json
    python tools/lr_finder.py --config configs/.../config.json --num-steps 150 --output lr_finder.png
    python tools/lr_finder.py --config configs/.../config.json --no-amp --restore

After running, use the suggested LR in your config's training.learning_rate (and
optionally scale by batch size / world size as in train.py). The model weights are
modified during the sweep; for clean training, start from scratch or load a
checkpoint (do not reuse the same run directory). If you hit GPU OOM, try
--batch-size 1 or 2. With a large batch size (e.g. from config), each step can
be slow; override with --batch-size 4 or 8 for faster sweeps.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import TypedDict

# Set CUDA memory allocation before importing PyTorch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Ensure tools dir is on path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from oriented_det.data import DOTADataset, AirbusPlaygroundCSVDataset, build_dota_split_dataset
from oriented_det.train.config import LossConfig, TrainingExperimentConfig
from oriented_det.train.utils import capped_subset_indices
from oriented_det.utils import get_device

from oriented_det.runtime.collate import create_collate_fn, create_train_augmentation, check_directories
from train import (
    analyze_class_distribution,
    build_optimizer_param_groups,
    compute_class_weights,
    create_model_from_config,
)


def _get_preprocessing_from_config(config: TrainingExperimentConfig):
    """Extract preprocessing (resize, norm, pad, flips) from config. Top-level enable_flip_* override preprocessing."""
    prep = getattr(config, "preprocessing", None)
    if prep is not None:
        from oriented_det.data.preprocessing import parse_canvas_size

        resize_mode = getattr(prep, "resize_mode", "fixed")
        ts = getattr(prep, "target_size", [1024, 1024])
        resize_to = parse_canvas_size(resize_mode, ts)
        norm_mean = getattr(prep, "normalize_mean", None)
        norm_std = getattr(prep, "normalize_std", None)
        pad_div = getattr(prep, "pad_size_divisor", 32)
        flip_h = getattr(prep, "enable_flip_horizontal", True)
        flip_v = getattr(prep, "enable_flip_vertical", True)
        flip_d = getattr(prep, "enable_flip_diagonal", False)
    else:
        resize_mode = "fixed"
        resize_to = (1024, 1024)
        norm_mean = norm_std = None
        pad_div = 32
        flip_h = True
        flip_v = True
        flip_d = False
    return resize_mode, resize_to, norm_mean, norm_std, pad_div, flip_h, flip_v, flip_d


class FastaiSuggestions(TypedDict, total=False):
    """Learning rates (for the sweep batch size) from fastai `lr_find` heuristics."""

    valley: float
    steep: float
    minimum: float
    slide: float


def trim_trace_for_suggestions(
    lrs: list[float],
    losses: list[float],
    num_it: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match fastai `Learner.lr_find`: skip first `num_it//10` points and last 5; truncate at first non-finite loss."""
    start = num_it // 10
    if len(lrs) <= start:
        return np.array([], dtype=float), np.array([], dtype=float)
    end = -5 if len(lrs) - start > 5 else None
    lrs_arr = np.array(lrs[start:end], dtype=float)
    losses_arr = np.array(losses[start:end], dtype=float)
    bad = ~np.isfinite(losses_arr)
    if np.any(bad):
        first_bad = int(np.nonzero(bad)[0][0])
        lrs_arr = lrs_arr[:first_bad]
        losses_arr = losses_arr[:first_bad]
    return lrs_arr, losses_arr


def suggest_valley(lrs: np.ndarray, losses: np.ndarray) -> float | None:
    """Longest 'valley' on the loss curve; LR at ~first third of that segment (fastai `valley`)."""
    n = int(losses.shape[0])
    if n < 1:
        return None
    max_start, max_end = 0, 0
    lds = [1] * n
    loss_list = losses.tolist()
    for i in range(1, n):
        for j in range(0, i):
            if loss_list[i] < loss_list[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
        if lds[max_end] < lds[i]:
            max_end = i
    max_start = max_end - lds[max_end]
    sections = (max_end - max_start) / 3.0
    idx = max_start + int(sections) + int(sections / 2)
    idx = int(np.clip(idx, 0, n - 1))
    return float(lrs[idx])


def suggest_steep(lrs: np.ndarray, losses: np.ndarray) -> float | None:
    """LR where d(loss)/d(log lr) is most negative — steepest descent (fastai `steep`)."""
    if lrs.size < 2:
        return None
    log_lr0 = np.log(np.maximum(lrs[:-1], 1e-20))
    log_lr1 = np.log(np.maximum(lrs[1:], 1e-20))
    dlog = log_lr1 - log_lr0
    with np.errstate(divide="ignore", invalid="ignore"):
        grads = (losses[1:] - losses[:-1]) / dlog
    if not np.any(np.isfinite(grads)):
        return None
    i = int(np.nanargmin(grads))
    return float(lrs[i])


def suggest_minimum(lrs: np.ndarray, losses: np.ndarray) -> float | None:
    """LR at minimum loss divided by 10 (fastai `minimum`)."""
    if lrs.size < 1 or not np.any(np.isfinite(losses)):
        return None
    finite = np.isfinite(losses)
    imin = int(np.argmin(np.where(finite, losses, np.inf)))
    lr_min = float(lrs[imin])
    return lr_min / 10.0


def suggest_slide(
    lrs: np.ndarray,
    losses: np.ndarray,
    *,
    lr_diff: int = 15,
    thresh: float = 0.005,
    adjust_value: float = 1.0,
) -> float | None:
    """Interval slide on loss gradient (fastai `slide`)."""
    if lrs.size < lr_diff + 1:
        return None
    loss_grad = np.gradient(losses.astype(float))
    r_idx = -1
    l_idx = r_idx - lr_diff
    local_min_lr = float(lrs[l_idx])
    while l_idx >= -len(losses) and abs(loss_grad[r_idx] - loss_grad[l_idx]) > thresh:
        local_min_lr = float(lrs[l_idx])
        r_idx -= 1
        l_idx -= 1
    return float(local_min_lr) * adjust_value


def compute_fastai_suggestions(
    lrs: list[float],
    losses: list[float],
    num_it: int,
) -> FastaiSuggestions:
    """Run fastai-style suggestion functions on the trimmed LR / loss trace."""
    lt, lo = trim_trace_for_suggestions(lrs, losses, num_it)
    out: FastaiSuggestions = {}
    if lt.size == 0:
        return out
    v = suggest_valley(lt, lo)
    if v is not None and math.isfinite(v):
        out["valley"] = v
    s = suggest_steep(lt, lo)
    if s is not None and math.isfinite(s):
        out["steep"] = s
    m = suggest_minimum(lt, lo)
    if m is not None and math.isfinite(m):
        out["minimum"] = m
    sl = suggest_slide(lt, lo)
    if sl is not None and math.isfinite(sl):
        out["slide"] = sl
    return out


def pick_primary_suggestion(suggestions: FastaiSuggestions, start_lr: float, end_lr: float) -> float:
    """Prefer valley (fastai default), then steep, minimum, slide; else geometric mid of sweep."""
    for key in ("valley", "steep", "minimum", "slide"):
        if key in suggestions:
            lr = suggestions[key]
            if math.isfinite(lr):
                return float(np.clip(lr, start_lr, end_lr))
    return float(math.sqrt(start_lr * end_lr))


def truncate_after_divergence(
    lrs: list[float],
    losses: list[float],
    *,
    stop_mult: float = 4.0,
    min_loss_floor: float = 1e-8,
) -> tuple[list[float], list[float], int | None]:
    """Keep only the prefix before catastrophic loss.

    Cuts from the first index ``i`` where ``losses[i]`` is non-finite, or where
    ``losses[i] > stop_mult * best`` with ``best = min(losses[0], ..., losses[i-1])``
    (running minimum strictly before ``i``). Matches the spirit of fastai's
    ``stop_div`` guard on raw points so bogus post-spike values (e.g. zeros) are
    excluded from suggestions.

    Returns ``(lrs[:i], losses[:i], i)`` when truncated, else copies of the inputs
    and ``None`` for ``i``.
    """
    if not lrs or not losses or len(lrs) != len(losses):
        return list(lrs), list(losses), None
    best = math.inf
    for i in range(len(losses)):
        L = losses[i]
        if not math.isfinite(L):
            if i == 0:
                return [], [], 0
            return lrs[:i], losses[:i], i
        if i == 0:
            best = L
            continue
        if L < best:
            best = L
            continue
        if best >= min_loss_floor and L > stop_mult * best:
            return lrs[:i], losses[:i], i
    return list(lrs), list(losses), None


def run_lr_finder(
    config_path: Path,
    *,
    batch_size: int | None = None,
    num_steps: int = 100,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    use_amp: bool = True,
    restore_state: bool = False,
    output_plot: str | None = None,
    early_stop: bool = True,
    stop_mult: float = 4.0,
    smooth_beta: float = 0.98,
) -> tuple[list[float], list[float], float, float, int, int, FastaiSuggestions]:
    """
    Run LR finder sweep and return
    (lrs, losses, suggested_lr, proposed_lr_for_config, config_batch_size, sweep_batch_size, fastai_suggestions).

    ``lrs`` / ``losses`` are the full recorded sweep (possibly ended by early stop).
    Suggestions are computed on ``truncate_after_divergence`` then fastai trim.

    suggested_lr: primary pick (fastai order: valley → steep → minimum → slide), clamped to the sweep range.
    fastai_suggestions: all heuristics that succeeded on the trimmed trace (same batch size as the sweep).
    """
    config = TrainingExperimentConfig.load(config_path)
    config_batch_size = config.data_loader.batch_size
    if batch_size is not None:
        config.data_loader.batch_size = batch_size
    sweep_batch_size = config.data_loader.batch_size
    config.training.use_amp = use_amp

    device = get_device()
    dataset_format = getattr(config.dataset, "format", "dota").lower()

    if dataset_format == "dota":
        check_directories(
            config.dataset.get_train_tile_roots(),
            config.dataset.get_val_tile_roots(),
        )
        if not config.dataset.has_dota_tiles_config():
            raise ValueError(
                "DOTA format requires dataset.train_tiles_dir(s) and dataset.val_tiles_dir(s)."
            )
        same_folder = getattr(config.dataset, "same_folder", False)
        train_dataset = build_dota_split_dataset(
            config.dataset.get_train_tile_roots(),
            split="train",
            same_folder=same_folder,
            difficult_strategy=config.dataset.difficult_strategy,
            allowed_classes=config.dataset.allowed_classes,
            ignore_labels=config.dataset.ignore_labels,
            filter_empty_gt=getattr(config.dataset, "filter_empty_gt", False),
        )
    else:
        if not config.dataset.annotations_file or not config.dataset.split_file:
            raise ValueError(
                "Airbus Playground format requires dataset.annotations_file and dataset.split_file."
            )
        train_dataset = AirbusPlaygroundCSVDataset(
            data_root=config.dataset.data_root,
            split="train",
            annotations_file=config.dataset.annotations_file,
            split_file=config.dataset.split_file,
            val_split_id=config.dataset.val_split_id,
            difficult_strategy=config.dataset.difficult_strategy,
            allowed_classes=config.dataset.allowed_classes,
            ignore_labels=config.dataset.ignore_labels,
            map_labels=config.dataset.map_labels,
        )

    class_names = train_dataset.get_class_names()
    class_map = {name: i + 1 for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    if getattr(config.dataset, "max_train_samples", None) is not None:
        shuffle_seed = getattr(config.dataset, "max_samples_shuffle_seed", None)
        idx = capped_subset_indices(
            len(train_dataset),
            int(config.dataset.max_train_samples),
            shuffle_seed=shuffle_seed,
        )
        train_dataset = Subset(train_dataset, idx)

    class_counts, _, _, _ = analyze_class_distribution(train_dataset)
    loss_config = config.loss
    if loss_config.loss_type in ("class_weighted", "focal_weighted") and class_counts:
        roi_class_weights, _computed_class_weights = compute_class_weights(
            class_counts,
            None,
            loss_config.class_weight_method,
            class_weight_overrides=getattr(loss_config, "class_weight_overrides", None),
        )
        if getattr(loss_config, "background_weight", None) is not None:
            roi_class_weights["background"] = float(loss_config.background_weight)
    else:
        roi_class_weights = None

    config.class_map = class_map
    config.class_names = class_names
    config.num_classes = num_classes

    resize_mode, resize_to, norm_mean, norm_std, pad_div, flip_h, flip_v, flip_d = _get_preprocessing_from_config(
        config
    )

    if config.enable_albumentation:
        train_aug = create_train_augmentation(
            brightness_limit=config.augmentation.brightness_limit,
            contrast_limit=config.augmentation.contrast_limit,
            gamma_limit=config.augmentation.gamma_limit,
            gauss_noise_var_limit=config.augmentation.gauss_noise_var_limit,
            blur_limit=config.augmentation.blur_limit,
            clahe_clip_limit=config.augmentation.clahe_clip_limit,
            p_brightness_contrast=config.augmentation.p_brightness_contrast,
            p_gamma=config.augmentation.p_gamma,
            p_noise=config.augmentation.p_noise,
            p_blur=config.augmentation.p_blur,
            p_clahe=config.augmentation.p_clahe,
        )
        collate_fn = create_collate_fn(
            config.class_map,
            augmentation=train_aug,
            normalize=True,
            resize_mode=resize_mode,
            resize_to=resize_to,
            pad_size_divisor=pad_div,
            enable_flip_horizontal=flip_h,
            enable_flip_vertical=flip_v,
            enable_flip_diagonal=flip_d,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )
    else:
        collate_fn = create_collate_fn(
            config.class_map,
            augmentation=None,
            normalize=True,
            resize_mode=resize_mode,
            resize_to=resize_to,
            pad_size_divisor=pad_div,
            enable_flip_horizontal=flip_h,
            enable_flip_vertical=flip_v,
            enable_flip_diagonal=flip_d,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data_loader.batch_size,
        shuffle=True,
        num_workers=config.data_loader.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data_loader.pin_memory if torch.cuda.is_available() else False,
    )

    model, _ = create_model_from_config(
        config, num_classes, device, roi_class_weights=roi_class_weights
    )
    if roi_class_weights is not None and hasattr(model, "set_class_weights"):
        model.set_class_weights(config.class_map, device=device)

    use_param_groups = bool(getattr(config.training, "use_lr_param_groups", False))
    if use_param_groups:
        param_groups, group_summary = build_optimizer_param_groups(
            model, base_lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay, config=config,
        )
        optimizer = optim.SGD(
            param_groups,
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
        multipliers = {
            g["group_name"]: group_summary[g["group_name"]]["multiplier"]
            for g in optimizer.param_groups
            if "group_name" in g
        }
    else:
        multipliers = None
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )

    if restore_state:
        state_before = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    use_amp = config.training.use_amp
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    scaler = None
    if use_amp and device_type == "cuda":
        try:
            from torch.amp import GradScaler as AmpGradScaler
            scaler = AmpGradScaler(device_type)
        except ImportError:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    autocast_ctx = None
    if use_amp:
        try:
            from torch.amp import autocast
            autocast_ctx = autocast(device_type=device_type)
        except (ImportError, TypeError):
            from torch.cuda.amp import autocast
            autocast_ctx = autocast()

    lrs: list[float] = []
    losses: list[float] = []
    data_iter = iter(train_loader)
    smooth_loss: float | None = None
    best_smooth = float("inf")

    bs = config.data_loader.batch_size
    print(f"  Running {num_steps} steps (batch_size={bs})...", flush=True)
    for step in range(num_steps):
        lr = start_lr * (end_lr / start_lr) ** (step / max(1, num_steps - 1))
        for i, group in enumerate(optimizer.param_groups):
            if multipliers and "group_name" in group:
                group["lr"] = lr * multipliers[group["group_name"]]
            else:
                group["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        else:
            images = batch.get("images", [])
            targets = batch.get("targets", [])

        if isinstance(images, list):
            images = [x.to(device) if torch.is_tensor(x) else x for x in images]
        elif torch.is_tensor(images):
            images = images.to(device)
        for t in targets:
            if isinstance(t, dict):
                for k, v in t.items():
                    if torch.is_tensor(v):
                        t[k] = v.to(device)

        optimizer.zero_grad()
        if autocast_ctx is not None and scaler is not None:
            with autocast_ctx:
                loss_dict = model(images, targets)
            total_loss_tensor = sum(
                v for v in loss_dict.values() if torch.is_tensor(v)
            )
            total = total_loss_tensor.item() if torch.is_tensor(total_loss_tensor) else float(total_loss_tensor)
            if not math.isfinite(total):
                total = float("inf")
            else:
                scaler.scale(total_loss_tensor).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss_dict = model(images, targets)
            total_loss_tensor = sum(
                v for v in loss_dict.values() if torch.is_tensor(v)
            )
            total = total_loss_tensor.item() if torch.is_tensor(total_loss_tensor) else float(total_loss_tensor)
            if not math.isfinite(total):
                total = float("inf")
            else:
                total_loss_tensor.backward()
                optimizer.step()

        lrs.append(lr)
        losses.append(total)

        # Print every 5 steps (and first/last) so progress is visible; flush so it appears immediately
        if step % 5 == 0 or step == num_steps - 1:
            print(f"  step {step + 1}/{num_steps}  lr={lr:.2e}  loss={total:.4f}", flush=True)

        # fastai LRFinder: stop when smoothed loss explodes vs best smoothed (stop_div)
        if (
            early_stop
            and math.isfinite(total)
            and smooth_beta >= 0.0
            and smooth_beta < 1.0
        ):
            if smooth_loss is None:
                smooth_loss = total
            else:
                smooth_loss = smooth_beta * smooth_loss + (1.0 - smooth_beta) * total
            if smooth_loss < best_smooth:
                best_smooth = smooth_loss
            if (
                best_smooth < float("inf")
                and math.isfinite(smooth_loss)
                and smooth_loss > stop_mult * best_smooth
            ):
                print(
                    f"  Early stop: smoothed loss {smooth_loss:.4g} > {stop_mult:g} × "
                    f"best smoothed {best_smooth:.4g} (fastai-style).",
                    flush=True,
                )
                break
        if not math.isfinite(total):
            print("  Non-finite loss; stopping sweep.", flush=True)
            break

    if restore_state and state_before:
        model.load_state_dict(state_before, strict=True)
        print("  Model state restored.")

    lrs_sugg, losses_sugg, div_cut = truncate_after_divergence(
        lrs, losses, stop_mult=stop_mult
    )
    if div_cut is not None:
        print(
            f"  Truncated {len(lrs) - len(lrs_sugg)} point(s) after divergence for "
            f"suggestions (raw loss > {stop_mult:g} × best-so-far before that step).",
            flush=True,
        )
    suggestions = compute_fastai_suggestions(lrs_sugg, losses_sugg, num_steps)
    if not suggestions:
        losses_arr = np.array(losses, dtype=float)
        if not np.any(np.isfinite(losses_arr)):
            print("  Warning: all losses were non-finite; using geometric mid of sweep for primary LR.")
        else:
            print("  Warning: no fastai-style suggestion on trimmed trace; using geometric mid of sweep.")
    suggested_lr = pick_primary_suggestion(suggestions, start_lr, end_lr)

    # Scale suggested LR to config batch size (linear scaling: LR ∝ batch_size)
    if sweep_batch_size != config_batch_size and sweep_batch_size > 0:
        proposed_lr_for_config = suggested_lr * (config_batch_size / sweep_batch_size)
    else:
        proposed_lr_for_config = suggested_lr

    if output_plot and lrs and losses:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.semilogx(lrs, losses, "b-", alpha=0.8, label="loss")
            if div_cut is not None and 0 <= div_cut < len(lrs):
                ax.axvline(
                    lrs[div_cut],
                    color="gray",
                    linestyle="-",
                    alpha=0.6,
                    label=f"divergence cut @ lr={lrs[div_cut]:.2e}",
                )
            colors = {"valley": "green", "steep": "red", "minimum": "orange", "slide": "purple"}
            for name, lr in suggestions.items():
                ax.axvline(
                    lr,
                    color=colors.get(name, "gray"),
                    linestyle="--",
                    alpha=0.85,
                    label=f"{name} = {lr:.2e}",
                )
            def _matches_any_suggestion(lr: float) -> bool:
                return any(math.isclose(lr, v, rel_tol=1e-5, abs_tol=0.0) for v in suggestions.values())

            if not suggestions or not _matches_any_suggestion(suggested_lr):
                ax.axvline(
                    suggested_lr,
                    color="black",
                    linestyle=":",
                    label=f"primary = {suggested_lr:.2e}",
                )
            ax.set_xlabel("Learning rate")
            ax.set_ylabel("Loss")
            ax.set_title("LR finder (fastai-style suggestions)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_plot, dpi=150)
            plt.close(fig)
            print(f"  Plot saved to {output_plot}")
        except Exception as e:
            print(f"  Could not save plot: {e}")

    return (
        lrs,
        losses,
        suggested_lr,
        proposed_lr_for_config,
        config_batch_size,
        sweep_batch_size,
        suggestions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Learning rate finder for oriented detection training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps in the LR sweep",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        default=1e-7,
        help="Start learning rate",
    )
    parser.add_argument(
        "--end-lr",
        type=float,
        default=10.0,
        help="End learning rate",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision for the sweep",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore model state after the sweep (so you can train without reloading)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save loss vs LR plot to this path (e.g. lr_finder.png)",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable fastai-style early stop (smoothed loss > stop_mult × best smoothed)",
    )
    parser.add_argument(
        "--stop-mult",
        type=float,
        default=4.0,
        metavar="K",
        help="Early-stop and divergence truncation when loss/smoothed loss exceeds K × best-so-far",
    )
    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=0.98,
        help="EMA beta for smoothed loss during early stop (0 ≤ beta < 1; ignored if --no-early-stop)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    if not args.no_early_stop and not (0.0 <= args.smooth_beta < 1.0):
        raise SystemExit("--smooth-beta must satisfy 0 <= beta < 1 (or pass --no-early-stop).")
    if args.stop_mult <= 0:
        raise SystemExit("--stop-mult must be positive.")

    print("LR finder: loading config and building model/dataloader...")
    lrs, losses, suggested, proposed_for_config, config_bs, sweep_bs, suggestions = run_lr_finder(
        config_path,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        use_amp=not args.no_amp,
        restore_state=args.restore,
        output_plot=args.output,
        early_stop=not args.no_early_stop,
        stop_mult=args.stop_mult,
        smooth_beta=args.smooth_beta,
    )
    print()
    scale = (config_bs / sweep_bs) if sweep_bs > 0 else 1.0
    if suggestions:
        print("Fastai-style suggestions (trimmed trace; batch_size=%d):" % sweep_bs)
        for name in ("valley", "steep", "minimum", "slide"):
            if name not in suggestions:
                continue
            lr = suggestions[name]
            print(f"  {name:8s}  {lr:.2e}")
        print()
        print(
            "Primary suggestion (valley → steep → minimum → slide) "
            f"for batch_size={sweep_bs}:  {suggested:.2e}"
        )
    else:
        print(f"No per-heuristic suggestions; primary LR (fallback):  {suggested:.2e}")
    print()
    if sweep_bs != config_bs:
        if suggestions:
            print("Proposed for config (linear scaling to config batch_size):")
            for name in ("valley", "steep", "minimum", "slide"):
                if name not in suggestions:
                    continue
                prop = suggestions[name] * scale
                print(f"  {name:8s}  {prop:.2e}")
            print()
        print(f"  batch_size={config_bs} (primary):  {proposed_for_config:.2e}  → training.learning_rate")
    else:
        print("Use primary value in config training.learning_rate:")
        print(f"  {proposed_for_config:.2e}")
    print()
    print("Scale by world size / gradient accumulation if needed (see train.py).")


if __name__ == "__main__":
    main()
