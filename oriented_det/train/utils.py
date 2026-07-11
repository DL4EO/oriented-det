"""Training utilities: collation, metrics, checkpointing, schedulers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import os
import shutil
import time

try:
    import torch
    import torch.optim as optim
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    optim = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore

from ..geometry import RBox


def get_project_root() -> Path:
    """Root for ``runs/``, ``predictions/``, etc.

    ``ORIENTED_DET_PROJECT_ROOT`` points at an external project directory
    when the framework is ``pip install -e`` from another tree.
    """
    env = os.environ.get("ORIENTED_DET_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # For normal (non-editable) installs, writing under site-packages is undesirable.
    # Default to the current working directory so `odet train` behaves like a CLI tool.
    return Path.cwd().resolve()


def get_framework_source_root() -> Path:
    """Directory containing the installed ``oriented_det`` package (framework source tree)."""
    from oriented_det import __file__ as _pkg_init

    return Path(_pkg_init).resolve().parent.parent


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git subcommand; return stripped stdout or ``None`` on failure."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def capture_source_provenance(
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Capture git revision and package version for experiment reproducibility.

    Uses the framework install tree (not ``ORIENTED_DET_PROJECT_ROOT``) so runs
    record the code that was actually imported, including editable installs.
    """
    root = (repo_root or get_framework_source_root()).resolve()
    commit = _run_git(["rev-parse", "HEAD"], root)
    describe = _run_git(["describe", "--dirty", "--always", "--tags"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    commit_date = _run_git(["log", "-1", "--format=%cI"], root)

    dirty: Optional[bool] = None
    if describe is not None:
        dirty = describe.endswith("-dirty")
    elif commit is not None:
        status = _run_git(["status", "--porcelain"], root)
        dirty = bool(status)

    package_version: Optional[str] = None
    try:
        from importlib.metadata import version

        package_version = version("oriented-det")
    except Exception:
        pass

    return {
        "source_code_root": str(root),
        "git_commit": commit,
        "git_describe": describe,
        "git_dirty": dirty,
        "git_branch": branch,
        "git_commit_date": commit_date,
        "package_version": package_version,
    }


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for training utilities.")


def strip_ddp_parameter_prefix(name: str) -> str:
    """Normalize DistributedDataParallel ``module.*`` parameter names."""
    return name[7:] if name.startswith("module.") else name


def capped_subset_indices(
    n_total: int, n_cap: int, *, shuffle_seed: Optional[int] = None
) -> List[int]:
    """Pick ``min(n_cap, n_total)`` dataset indices for a :class:`torch.utils.data.Subset` cap.

    ``shuffle_seed is None`` (default): legacy **first-N** indices (dataset enumeration order).
    When set: shuffle ``range(n_total)`` with :class:`random.Random` and take the first ``n`` —
    deterministic for a given seed and spreads the cap across the listing (e.g. mixed tiles).
    """
    import random

    n_total_i = int(n_total)
    n_cap_i = int(n_cap)
    if n_total_i <= 0:
        return []
    n = min(n_cap_i, n_total_i)
    if n <= 0:
        return []
    if shuffle_seed is None:
        return list(range(n))
    indices = list(range(n_total_i))
    random.Random(int(shuffle_seed)).shuffle(indices)
    return indices[:n]


def model_has_rpn_head(model: "nn.Module") -> bool:
    """True if the module is a two-stage detector with ``rpn_head.*`` parameters."""
    _require_torch()
    for name, _ in model.named_parameters():
        if strip_ddp_parameter_prefix(name).startswith("rpn_head."):
            return True
    return False


def set_backbone_requires_grad(model: "nn.Module", *, freeze: bool) -> None:
    """Freeze or unfreeze all ``backbone.*`` parameters (two-stage and RetinaNet)."""
    _require_torch()
    trainable = not freeze
    for name, param in model.named_parameters():
        if strip_ddp_parameter_prefix(name).startswith("backbone."):
            param.requires_grad_(trainable)


def set_rpn_requires_grad(model: "nn.Module", *, freeze: bool) -> None:
    """Freeze or unfreeze ``rpn_head.*`` parameters. No-op if the model has no RPN (e.g. RetinaNet)."""
    _require_torch()
    if not model_has_rpn_head(model):
        return
    trainable = not freeze
    for name, param in model.named_parameters():
        if strip_ddp_parameter_prefix(name).startswith("rpn_head."):
            param.requires_grad_(trainable)


def get_best_checkpoint_path(checkpoint_dir: Path) -> Optional[Path]:
    """Return the path to the best checkpoint in the directory, if any.
    
    Looks for checkpoint_best.pth (legacy) first, then best_*.pth (e.g. best_mAP_0.42.pth).
    """
    checkpoint_dir = Path(checkpoint_dir)
    legacy = checkpoint_dir / "checkpoint_best.pth"
    if legacy.exists():
        return legacy
    best_files = sorted(checkpoint_dir.glob("best_*.pth"))
    return best_files[0] if best_files else None


@dataclass
class MetricTracker:
    """Efficient metric tracking during training."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_times: List[float] = []
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values."""
        for key, value in metrics.items():
            self.metrics[key].append(float(value))
    
    def update_time(self, elapsed: float) -> None:
        """Record step time."""
        self.step_times.append(elapsed)
    
    def get_average(self, key: str, window: Optional[int] = None) -> float:
        """Get average of a metric, optionally over a window."""
        values = self.metrics[key]
        if not values:
            return 0.0
        if window is None:
            return sum(values) / len(values)
        recent = values[-window:]
        return sum(recent) / len(recent)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get most recent value of a metric."""
        values = self.metrics[key]
        return values[-1] if values else None
    
    def get_summary(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        for key in self.metrics:
            summary[key] = self.get_average(key, window=window)
        if self.step_times:
            recent_times = self.step_times[-window:] if window else self.step_times
            summary["time_per_step"] = sum(recent_times) / len(recent_times)
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step_times.clear()


def collate_dota_samples(batch: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Collate function for DOTA dataset samples.
    
    Converts DOTASample objects to format expected by models.
    Loads images from paths and converts them to tensors.
    """
    _require_torch()
    
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("PIL/Pillow is required to load images.")
    
    try:
        from torchvision import transforms as T
    except ImportError:
        raise RuntimeError("torchvision is required for image transforms.")
    
    images = []
    targets = []
    
    # Convert PIL Image to tensor transform
    to_tensor = T.ToTensor()
    
    for sample in batch:
        # Load image from path
        image_path = sample.image_path
        try:
            pil_image = Image.open(image_path).convert("RGB")
            # Convert to tensor (C, H, W) format with values in [0, 1]
            image_tensor = to_tensor(pil_image)
            images.append(image_tensor)
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {image_path}: {e}") from e
        
        # Convert annotations to target format
        # Apply le90 normalization (MMRotate standard for DOTA)
        # This ensures width >= height and angles in [-π/2, π/2) range
        from oriented_det.geometry.rbox import normalize_le90
        
        rboxes = []
        labels = []
        class_to_id = {}
        for ann in sample.annotations:
            if ann.class_name not in class_to_id:
                class_to_id[ann.class_name] = len(class_to_id)
            # Normalize to le90 convention (width >= height, angle in [-π/2, π/2))
            normalized_rbox = normalize_le90(ann.rbox)
            rboxes.append(normalized_rbox)
            labels.append(class_to_id[ann.class_name])
        
        target = {
            "rboxes": rboxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": sample.image_path.stem if hasattr(sample, "image_path") else None,
            "image_filename": Path(sample.image_path).name
            if hasattr(sample, "image_path") and sample.image_path
            else None,
        }
        
        targets.append(target)
    
    return images, targets


def collate_fn_generic(batch: List[Dict[str, Any]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Generic collate function for detection datasets.
    
    Expects batch items to be dicts with 'image' and 'target' keys.
    """
    _require_torch()
    
    images = []
    targets = []
    
    for item in batch:
        images.append(item.get("image"))
        targets.append(item.get("target", {}))
    
    return images, targets


class CheckpointManager:
    """Manages model checkpointing with robust error handling."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        keep_last_n: int = 5,
        best_metric: Optional[str] = None,
        higher_is_better: bool = True,
    ):
        _require_torch()
        self.checkpoint_dir = Path(checkpoint_dir)
        # Directory is created on first save() so multi-GPU training creates it only on rank 0
        self.keep_last_n = keep_last_n
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better
        self.best_value: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.checkpoint_history: List[Path] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        *,
        suffix: str = "",
        custom_filename: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint.
        
        If custom_filename is set (e.g. "best_mAP_0.42.pth"), that name is used
        and the checkpoint is not added to history (treated as a best checkpoint).
        
        Returns path to saved checkpoint.
        """
        _require_torch()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if custom_filename:
            filename = custom_filename if custom_filename.endswith(".pth") else f"{custom_filename}.pth"
        elif suffix:
            filename = f"checkpoint_{suffix}.pth"
        elif epoch is not None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        else:
            filename = f"checkpoint_{int(time.time())}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Atomic write: write to temp file first, then rename. Retry on transient I/O errors
        # (e.g. NFS "unexpected pos" / "file write failed"). Fallback: write to local /tmp then move.
        temp_path = checkpoint_path.with_suffix(".tmp")
        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                torch.save(checkpoint, temp_path)
                temp_path.replace(checkpoint_path)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        if last_error is not None:
            # Fallback: write to local temp dir (avoids NFS/network fs write failures)
            local_tmp = Path(os.environ.get("TMPDIR", "/tmp"))
            if local_tmp.exists() and os.access(local_tmp, os.W_OK):
                fallback_temp = local_tmp / f"oriented_det_checkpoint_{os.getpid()}_{filename}.tmp"
                try:
                    torch.save(checkpoint, fallback_temp)
                    shutil.move(str(fallback_temp), str(checkpoint_path))
                except Exception as e2:
                    if fallback_temp.exists():
                        try:
                            fallback_temp.unlink()
                        except OSError:
                            pass
                    raise RuntimeError(
                        f"Failed to save checkpoint after {max_attempts} attempts and local fallback: {last_error!s}; fallback: {e2!s}"
                    ) from e2
            else:
                raise RuntimeError(f"Failed to save checkpoint: {last_error}") from last_error
        
        # Only add to history if it's not a best checkpoint (best is managed separately)
        if not custom_filename and suffix != "best":
            self.checkpoint_history.append(checkpoint_path)
            
            # Clean up old checkpoints (excluding best checkpoint)
            while len(self.checkpoint_history) > self.keep_last_n:
                old_checkpoint = self.checkpoint_history.pop(0)
                # Don't delete if it's the best checkpoint
                if old_checkpoint != self.best_checkpoint_path and old_checkpoint.exists():
                    old_checkpoint.unlink()
        
        return checkpoint_path
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        """Save checkpoint if it's the best according to best_metric.
        
        The best checkpoint is stored independently and is not subject to
        the keep_last_n cleanup, ensuring it's always preserved.
        """
        if self.best_metric is None or metrics is None:
            return None
        
        current_value = metrics.get(self.best_metric)
        if current_value is None:
            return None
        # Do not save best checkpoint when mAP was not computed (sentinel -1)
        if str(self.best_metric).strip().lower() == "map" and current_value == -1.0:
            return None

        is_better = (
            (self.best_value is None) or
            (self.higher_is_better and current_value > self.best_value) or
            (not self.higher_is_better and current_value < self.best_value)
        )
        
        if is_better:
            # Delete old best checkpoint if it exists
            if self.best_checkpoint_path is not None and self.best_checkpoint_path.exists():
                try:
                    self.best_checkpoint_path.unlink()
                except Exception as e:
                    # Log but don't fail if deletion fails
                    print(f"Warning: Failed to delete old best checkpoint {self.best_checkpoint_path}: {e}")
            
            self.best_value = current_value
            # Build filename with metric name and value, e.g. best_mAP_0.42.pth or best_total_loss_1.23.pth
            safe_metric = "".join(c if c.isalnum() or c == "_" else "_" for c in str(self.best_metric))
            if isinstance(current_value, float):
                value_str = f"{current_value:.2f}" if safe_metric.lower() == "map" else f"{current_value:.4f}"
            else:
                value_str = str(current_value)
            custom_filename = f"best_{safe_metric}_{value_str}.pth"
            self.best_checkpoint_path = self.save(
                model, optimizer, scheduler, epoch, metrics, custom_filename=custom_filename
            )
            return self.best_checkpoint_path
        
        return None
    
    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        *,
        strict: bool = False,
        include_prefixes: Optional[List[str]] = None,
        exclude_prefixes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            strict: If True, raise error on mismatch. If False, warn and load matching weights.
        
        Returns:
            Loaded checkpoint dict.
        """
        _require_torch()
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
        
        # Load model state dict
        try:
            # Support multiple checkpoint formats:
            # - this repo: {"model_state_dict": ...}
            # - MMDetection/MMRotate: {"state_dict": ...}
            # - raw state_dict: {param_name: tensor, ...}
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
                checkpoint_state_dict = checkpoint
            else:
                raise KeyError("Unsupported checkpoint format: expected model_state_dict or state_dict.")

            # MMRotate → torchvision BackboneWithFPN naming bridge:
            # MMRotate backbone keys are typically "backbone.*" (ResNet), while this repo's
            # torchvision backbone lives under "backbone.body.*".
            model_state_keys = list(model.state_dict().keys())
            model_has_backbone_body = any(k.startswith("backbone.body.") for k in model_state_keys)
            ckpt_has_backbone_body = any(k.startswith("backbone.body.") for k in checkpoint_state_dict.keys())
            ckpt_has_backbone = any(k.startswith("backbone.") for k in checkpoint_state_dict.keys())
            if model_has_backbone_body and ckpt_has_backbone and not ckpt_has_backbone_body:
                remapped = {}
                for k, v in checkpoint_state_dict.items():
                    if k.startswith("backbone."):
                        suffix = k[len("backbone."):]
                        remapped[f"backbone.body.{suffix}"] = v
                    else:
                        remapped[k] = v
                checkpoint_state_dict = remapped
                print("  Detected MMRotate-style backbone keys; remapped backbone.* -> backbone.body.* for load.")

            ckpt_has_module_prefix = any(k.startswith("module.") for k in checkpoint_state_dict.keys())
            model_has_module_prefix = any(k.startswith("module.") for k in model_state_keys)

            # Handle common DDP/DataParallel prefix mismatch automatically.
            if ckpt_has_module_prefix and not model_has_module_prefix:
                checkpoint_state_dict = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in checkpoint_state_dict.items()
                }
                print("  Detected DDP checkpoint keys (module.*); stripping prefix for load.")
            elif model_has_module_prefix and not ckpt_has_module_prefix:
                checkpoint_state_dict = {f"module.{k}": v for k, v in checkpoint_state_dict.items()}
                print("  Model expects module.* keys; adding prefix to checkpoint state dict.")

            # Optional selective loading by prefix (config-driven transfer learning).
            include_prefixes = [p for p in (include_prefixes or []) if p]
            exclude_prefixes = [p for p in (exclude_prefixes or []) if p]
            if include_prefixes or exclude_prefixes:
                original_count = len(checkpoint_state_dict)
                filtered_state_dict: Dict[str, Any] = {}
                for key, value in checkpoint_state_dict.items():
                    key_no_module = key[7:] if key.startswith("module.") else key
                    if include_prefixes and not any(
                        key_no_module.startswith(prefix) for prefix in include_prefixes
                    ):
                        continue
                    if exclude_prefixes and any(
                        key_no_module.startswith(prefix) for prefix in exclude_prefixes
                    ):
                        continue
                    filtered_state_dict[key] = value
                checkpoint_state_dict = filtered_state_dict
                print(
                    "  Selective checkpoint loading enabled: "
                    f"kept {len(checkpoint_state_dict)}/{original_count} tensors "
                    f"(include={include_prefixes or None}, exclude={exclude_prefixes or None})."
                )

            # Drop keys whose tensor shapes differ from the model (e.g. ROI classifier when
            # num_classes changes, or RPN convs when anchor counts differ). PyTorch raises on
            # shape mismatch even when strict=False, which would otherwise abort the whole load.
            model_sd = model.state_dict()
            shape_skipped_detail: List[tuple] = []
            aligned_state_dict: Dict[str, Any] = {}
            for key, value in checkpoint_state_dict.items():
                if key not in model_sd:
                    continue
                if value.shape != model_sd[key].shape:
                    shape_skipped_detail.append(
                        (key, tuple(value.shape), tuple(model_sd[key].shape))
                    )
                    continue
                aligned_state_dict[key] = value
            if shape_skipped_detail:
                print(
                    f"  Skipped {len(shape_skipped_detail)} checkpoint tensors with shape mismatch "
                    f"(e.g. different num_classes or RPN anchor layout); model keeps init for those."
                )
                for key, ckpt_sh, model_sh in shape_skipped_detail[:10]:
                    print(f"     - {key}: checkpoint {ckpt_sh} vs model {model_sh}")
                if len(shape_skipped_detail) > 10:
                    print(f"     ... and {len(shape_skipped_detail) - 10} more")
            checkpoint_state_dict = aligned_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint_state_dict, strict=False
            )
            
            # Print warnings if there are mismatches
            if missing_keys:
                print(f"\n⚠️  Warning: Model architecture mismatch - {len(missing_keys)} keys not found in checkpoint:")
                print(f"   Missing keys (model expects but checkpoint doesn't have):")
                for key in missing_keys[:10]:  # Show first 10
                    print(f"     - {key}")
                if len(missing_keys) > 10:
                    print(f"     ... and {len(missing_keys) - 10} more")
                print(f"   These layers will be initialized randomly.")
            
            if unexpected_keys:
                print(f"\n⚠️  Warning: Model architecture mismatch - {len(unexpected_keys)} keys in checkpoint not used:")
                print(f"   Unexpected keys (checkpoint has but model doesn't need):")
                for key in unexpected_keys[:10]:  # Show first 10
                    print(f"     - {key}")
                if len(unexpected_keys) > 10:
                    print(f"     ... and {len(unexpected_keys) - 10} more")
                print(f"   These weights will be ignored.")
            
            if missing_keys or unexpected_keys:
                print(f"\n   Note: Training will continue with matching weights loaded and others initialized randomly.")
                print(f"   This is normal when changing model architecture (e.g., different num_classes, backbone, etc.)\n")
            
            # If strict=True was requested, raise error if there are mismatches
            if strict and (missing_keys or unexpected_keys):
                raise RuntimeError(
                    f"Strict loading failed: {len(missing_keys)} missing keys, "
                    f"{len(unexpected_keys)} unexpected keys. Set strict=False to allow partial loading."
                )
                
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load model state: {e}") from e
            else:
                print(f"\n⚠️  Warning: Failed to load model state: {e}")
                print(
                    "   Training will continue with default initialization for layers that "
                    "failed to load from the checkpoint.\n"
                )
        
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to load optimizer state: {e}")
                print(f"   Optimizer will be initialized with default state.\n")
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to load scheduler state: {e}")
                print(f"   Scheduler will be initialized with default state.\n")
        
        return checkpoint
    
    def get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoint_history:
            return None
        return self.checkpoint_history[-1]
    
    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self.best_checkpoint_path


def _lr_scheduler_base_class() -> Any:
    """PyTorch 2.x ``LRScheduler`` or legacy ``_LRScheduler`` base for custom schedulers."""
    _require_torch()
    return getattr(torch.optim.lr_scheduler, "LRScheduler", None) or torch.optim.lr_scheduler._LRScheduler


class CosineAnnealingWithFixedTailLR(_lr_scheduler_base_class()):  # type: ignore[misc, valid-type]
    """Cosine decay for ``cosine_epochs`` scheduler steps, then a constant LR tail.

    Matches :class:`torch.optim.lr_scheduler.CosineAnnealingLR` for
    ``last_epoch in [1, cosine_epochs]`` (same closed form as PyTorch's ``_get_closed_form_lr``),
    then holds ``tail_lr`` for ``last_epoch > cosine_epochs``. The ``last_epoch == 0`` branch keeps
    the current optimizer LR (same as PyTorch cosine) so it composes with epoch-based stepping after warmup.

    ``cosine_epochs`` corresponds to ``T_max`` in ``CosineAnnealingLR`` (minimum LR is reached when
    ``last_epoch == cosine_epochs``).
    """

    def __init__(
        self,
        optimizer: "optim.Optimizer",
        cosine_epochs: int,
        eta_min: float = 0.0,
        tail_lr: Optional[float] = None,
        last_epoch: int = -1,
    ) -> None:
        _require_torch()
        self.cosine_epochs = int(cosine_epochs)
        if self.cosine_epochs < 1:
            raise ValueError("cosine_epochs must be >= 1")
        self.eta_min = float(eta_min)
        self.tail_lr = float(self.eta_min if tail_lr is None else tail_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch <= self.cosine_epochs:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1.0 + math.cos(math.pi * float(self.last_epoch) / float(self.cosine_epochs)))
                / 2.0
                for base_lr in self.base_lrs
            ]
        return [self.tail_lr for _ in self.base_lrs]


def _resolve_cosine_phase_epochs(training: Any) -> int:
    """Cosine period length: ``lr_scheduler_cosine_epochs``, then ``lr_scheduler_cosine_t_max``, else ``num_epochs``."""
    num_epochs = int(training.num_epochs)
    cosine_epochs_cfg = getattr(training, "lr_scheduler_cosine_epochs", None)
    t_max_cfg = getattr(training, "lr_scheduler_cosine_t_max", None)
    if cosine_epochs_cfg is not None:
        phase = int(cosine_epochs_cfg)
    elif t_max_cfg is not None:
        phase = int(t_max_cfg)
    else:
        phase = num_epochs
    if phase < 1:
        raise ValueError(f"cosine phase length must be >= 1, got {phase}")
    return phase


def resolve_pytorch_cosine_t_max(training: Any) -> int:
    """``T_max`` for :class:`torch.optim.lr_scheduler.CosineAnnealingLR` (PyTorch default, including restarts)."""
    return _resolve_cosine_phase_epochs(training)


def resolve_cosine_with_tail_lengths(training: Any) -> Tuple[int, int]:
    """Cosine phase + fixed-LR tail lengths for ``cosine_annealing_with_tail`` (must sum to ``num_epochs``)."""
    num_epochs = int(training.num_epochs)
    cosine_epochs = _resolve_cosine_phase_epochs(training)
    tail_epochs_cfg = int(getattr(training, "lr_scheduler_cosine_tail_epochs", 0) or 0)
    if tail_epochs_cfg > 0:
        tail_epochs = tail_epochs_cfg
    else:
        tail_epochs = num_epochs - cosine_epochs
    if tail_epochs < 1:
        raise ValueError(
            f"cosine_annealing_with_tail needs tail_epochs >= 1; got cosine_epochs={cosine_epochs}, "
            f"num_epochs={num_epochs}. Set lr_scheduler_cosine_tail_epochs or shorten the cosine phase."
        )
    if cosine_epochs + tail_epochs != num_epochs:
        raise ValueError(
            f"cosine_epochs ({cosine_epochs}) + tail_epochs ({tail_epochs}) must equal "
            f"num_epochs ({num_epochs}) for cosine_annealing_with_tail"
        )
    return cosine_epochs, tail_epochs


def create_pytorch_cosine_lr_scheduler(
    optimizer: "optim.Optimizer",
    training: Any,
    *,
    last_epoch: int = -1,
) -> Tuple[Any, int, float]:
    """PyTorch ``CosineAnnealingLR``; if ``num_epochs`` > ``T_max``, LR restarts after each minimum."""
    _require_torch()
    t_max = resolve_pytorch_cosine_t_max(training)
    eta_min = float(getattr(training, "lr_scheduler_cosine_eta_min", 1e-6))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=eta_min,
        last_epoch=last_epoch,
    )
    return scheduler, t_max, eta_min


def create_cosine_with_tail_lr_scheduler(
    optimizer: "optim.Optimizer",
    training: Any,
    *,
    last_epoch: int = -1,
) -> Tuple[Any, int, int, float]:
    """Cosine decay then constant ``lr_scheduler_cosine_tail_lr`` (no PyTorch restart)."""
    _require_torch()
    eta_min = float(getattr(training, "lr_scheduler_cosine_eta_min", 1e-6))
    tail_lr_cfg = getattr(training, "lr_scheduler_cosine_tail_lr", None)
    tail_lr = float(eta_min if tail_lr_cfg is None else tail_lr_cfg)
    cosine_epochs, tail_epochs = resolve_cosine_with_tail_lengths(training)
    scheduler = CosineAnnealingWithFixedTailLR(
        optimizer,
        cosine_epochs=cosine_epochs,
        eta_min=eta_min,
        tail_lr=tail_lr,
        last_epoch=last_epoch,
    )
    return scheduler, cosine_epochs, tail_epochs, tail_lr


def format_pytorch_cosine_scheduler_description(
    t_max: int,
    eta_min: float,
    warmup_steps: int,
    num_epochs: int,
) -> str:
    """Human-readable line for ``cosine_annealing`` (PyTorch scheduler)."""
    restart_note = ""
    if num_epochs > t_max:
        restart_note = f"; SGDR-style restarts when num_epochs ({num_epochs}) > T_max"
    core = f"CosineAnnealingLR (T_max={t_max}, eta_min={eta_min:g}{restart_note})"
    if warmup_steps > 0:
        return f"{core} + warmup {warmup_steps} steps"
    return core


def format_cosine_with_tail_scheduler_description(
    cosine_epochs: int,
    tail_epochs: int,
    eta_min: float,
    tail_lr: float,
    warmup_steps: int,
) -> str:
    """Human-readable line for ``cosine_annealing_with_tail``."""
    core = (
        f"CosineAnnealingWithFixedTailLR (cosine_epochs={cosine_epochs}, "
        f"tail_epochs={tail_epochs}, eta_min={eta_min:g}, tail_lr={tail_lr:g})"
    )
    if warmup_steps > 0:
        return f"{core} + warmup {warmup_steps} steps"
    return core


def save_training_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save training configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


class WarmupScheduler:
    """Learning rate scheduler with warmup that wraps a base scheduler.
    
    During warmup, the learning rate increases linearly from 0 to the base learning rate.
    After warmup, it uses the base scheduler (stepped per epoch, not per optimizer step).
    
    This is useful for stabilizing training in the early stages, especially when using
    large learning rates or training from scratch.
    
    Example:
        >>> optimizer = optim.SGD(model.parameters(), lr=0.001)
        >>> base_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        >>> warmup_scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_steps=500)
        >>> 
        >>> # During training loop:
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         # ... training step ...
        >>>         optimizer.step()
        >>>         warmup_scheduler.step()  # Step per optimizer step during warmup
        >>>     warmup_scheduler.step_epoch()  # Step base scheduler per epoch
    """
    
    def __init__(self, optimizer: optim.Optimizer, base_scheduler: optim.lr_scheduler._LRScheduler, warmup_steps: int):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: The optimizer whose learning rate will be scheduled
            base_scheduler: The base scheduler (e.g., StepLR) to use after warmup
            warmup_steps: Number of optimizer steps for warmup (not epochs)
        """
        _require_torch()
        if optim is None:
            raise RuntimeError("torch.optim is required for WarmupScheduler.")
        
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.base_lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
        self.current_step = 0
        self.warmup_complete = False
        
        # Set initial learning rate to 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0
    
    def step(self) -> None:
        """Step the scheduler (called per optimizer step during warmup).
        
        This should be called after each optimizer.step() during the warmup period.
        After warmup completes, this method does nothing and step_epoch() should
        be used instead to step the base scheduler.
        """
        if not self.warmup_complete:
            self.current_step += 1
            
            if self.current_step <= self.warmup_steps:
                # Linear warmup per param group.
                progress = self.current_step / self.warmup_steps
                for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                    param_group['lr'] = base_lr * progress
            else:
                # Warmup complete - restore each group's base LR and mark as complete
                # The base scheduler will handle further updates per epoch
                for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                    param_group['lr'] = base_lr
                self.warmup_complete = True
    
    def step_epoch(self) -> None:
        """Step the base scheduler (called per epoch after warmup).
        
        This should be called once per epoch after warmup completes.
        During warmup, this method does nothing.
        """
        if self.warmup_complete:
            self.base_scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get the last learning rate for each parameter group."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_complete': self.warmup_complete,
            'base_scheduler': self.base_scheduler.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_complete = state_dict.get('warmup_complete', False)
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def effective_score_threshold_for_class_name(
    class_name: str,
    global_threshold: float,
    per_class: Optional[Dict[str, float]],
) -> float:
    """Return min score for a class: per-class map entry or global default."""
    if not per_class:
        return float(global_threshold)
    if class_name in per_class:
        return float(per_class[class_name])
    cn_lower = str(class_name).lower()
    for k, v in per_class.items():
        if str(k).lower() == cn_lower:
            return float(v)
    return float(global_threshold)


def filter_detections_by_score_threshold(
    detections: List[Any],
    global_threshold: float,
    per_class: Optional[Dict[str, float]],
    id_to_class: Optional[Dict[int, str]] = None,
) -> List[Any]:
    """Filter detection list by global and optional per-class score thresholds."""
    if not per_class:
        return [d for d in detections if float(d.score) >= float(global_threshold)]
    out: List[Any] = []
    for d in detections:
        name = str(getattr(d, "class_name", "") or "")
        if not name and id_to_class is not None:
            name = id_to_class.get(int(getattr(d, "class_id", 0)), "")
        thr = effective_score_threshold_for_class_name(name, global_threshold, per_class)
        if float(d.score) >= thr:
            out.append(d)
    return out


def scores_labels_pass_threshold(
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    global_threshold: float,
    per_class: Optional[Dict[str, float]],
    id_to_class: Dict[int, str],
) -> "torch.Tensor":
    """Boolean mask: keep detection i if score[i] >= threshold for labels[i]'s class."""
    _require_torch()
    if scores.numel() == 0:
        return scores >= global_threshold
    if not per_class:
        return scores >= global_threshold
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for i in range(int(scores.shape[0])):
        lid = int(labels[i].item())
        cname = id_to_class.get(lid, f"class_{lid}")
        thr = effective_score_threshold_for_class_name(cname, global_threshold, per_class)
        mask[i] = scores[i] >= thr
    return mask


class OneCycleWrapper:
    """Wrapper for OneCycleLR so it is stepped every optimizer step (like WarmupScheduler).

    OneCycleLR must be stepped each batch; the main train loop only steps schedulers
    once per epoch. This wrapper exposes step() (forwarded to OneCycleLR) and
    step_epoch() (no-op), so train_one_epoch can call step() every batch and the
    main loop's step_epoch() does nothing.
    """

    def __init__(self, one_cycle_scheduler: Any) -> None:
        self._scheduler = one_cycle_scheduler

    def step(self) -> None:
        self._scheduler.step()

    def step_epoch(self) -> None:
        # No-op: OneCycleLR is stepped per batch only
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self._scheduler.load_state_dict(state_dict)


__all__ = [
    "MetricTracker",
    "CheckpointManager",
    "CosineAnnealingWithFixedTailLR",
    "capped_subset_indices",
    "collate_dota_samples",
    "collate_fn_generic",
    "create_cosine_with_tail_lr_scheduler",
    "create_pytorch_cosine_lr_scheduler",
    "format_cosine_with_tail_scheduler_description",
    "format_pytorch_cosine_scheduler_description",
    "resolve_cosine_with_tail_lengths",
    "resolve_pytorch_cosine_t_max",
    "save_training_config",
    "WarmupScheduler",
    "OneCycleWrapper",
    "effective_score_threshold_for_class_name",
    "filter_detections_by_score_threshold",
    "scores_labels_pass_threshold",
]
