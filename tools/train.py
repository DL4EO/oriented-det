#!/usr/bin/env python3
"""Standalone training script for oriented object detection.

This script loads training configuration from JSON files and supports multiple model types.
Supports both single-GPU and multi-GPU (distributed) training. For multi-GPU, use
tools/train_multi_gpu.py or torchrun; the script detects RANK/WORLD_SIZE/LOCAL_RANK.

Usage:
    # Single-GPU
    python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json [--batch-size 4] [--use-amp]

    # Multi-GPU (via launcher)
    python tools/train_multi_gpu.py --config configs/.../config.json [--nproc-per-node 4]

Command-line options:
    --config          Path to config JSON file (required)
    --batch-size      Override batch size from config
    --use-amp         Enable automatic mixed precision training (overrides config)
    --no-amp          Disable automatic mixed precision training (overrides config)
    --debug           Enable debug tracing (RPN/ROI match statistics)
    --wizard          Run data/config diagnostics and print recommendations
    --local-rank      Set by torchrun for distributed training

Checkpoint loading is controlled only by the JSON ``checkpoint`` section (paths,
``discover_previous_run``, ``resume_from_checkpoint_epoch``, etc.).

The script will:
- Load configuration from JSON file
- Load the DOTA dataset
- Discover classes automatically
- Configure class imbalance handling
- Train the model (single- or multi-GPU)
- Save checkpoints and TensorBoard logs (rank 0 only in multi-GPU)
"""

import os
import sys

# Set CUDA memory allocation configuration before importing PyTorch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# On macOS (MPS): enable CPU fallback for ops not implemented on Metal (e.g. grid_sampler_2d_backward).
# Must be set before importing torch so the MPS backend picks it up.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# cuDNN can raise: RuntimeError: GET was unable to find an engine to execute this computation
# Common causes: conflicting LD_LIBRARY_PATH (system cuDNN vs wheels), or cudnn.benchmark
# picking a bad algorithm. Workarounds:
#   ORIENTED_DET_CUDNN_BENCHMARK=0  — disable cudnn benchmark (often fixes; slightly slower convs)
#   python tools/train.py ... --no-amp  — rule out AMP/autocast interaction
#   unset LD_LIBRARY_PATH  — if an old CUDA/cuDNN path shadows PyTorch's bundled libs
_cudnn_bm = os.environ.get("ORIENTED_DET_CUDNN_BENCHMARK", "").lower()
if _cudnn_bm in ("0", "false", "no", "off"):
    torch.backends.cudnn.benchmark = False

import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
import csv as csv_module
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
# TensorBoard is optional at import time (tests import tools.train utilities).
try:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore
from datetime import datetime
from collections import Counter
from typing import Optional, Dict, Any, List, Union, Set, Tuple
import math
import numpy as np
import sys
import traceback
import argparse

from oriented_det import OrientedRCNN, RotatedFasterRCNN, RotatedRetinaNet, RotatedFCOS
from oriented_det.data import (
    build_split_dataset,
    dataset_format_name,
    format_airbus_empty_gt_filter_log,
    format_dota_empty_gt_filter_log,
    format_hrsc_empty_gt_filter_log,
    split_class_names,
)
from oriented_det.train import train, CheckpointManager, WarmupScheduler, OneCycleWrapper, get_best_checkpoint_path
from oriented_det.train.utils import (
    capped_subset_indices,
    capture_source_provenance,
    create_cosine_with_tail_lr_scheduler,
    create_multistep_lr_scheduler,
    create_pytorch_cosine_lr_scheduler,
    format_cosine_with_tail_scheduler_description,
    format_pytorch_cosine_scheduler_description,
    model_has_rpn_head,
    set_backbone_requires_grad,
    set_rpn_requires_grad,
)
from oriented_det.train.config import (
    TrainingExperimentConfig,
    LossConfig,
    effective_eval_metric_thresholds,
    config_use_exact_rotated_iou_for_map,
    config_use_exact_rotated_iou_for_final_map,
)
from oriented_det.utils import enable_tracing


def _dataset_index_to_image_stem(dataset, idx: int) -> str:
    """Resolve image stem for train dataset index (DOTA, Airbus, Subset, or wrappers)."""
    from torch.utils.data import ConcatDataset

    if isinstance(dataset, Subset):
        return _dataset_index_to_image_stem(dataset.dataset, int(dataset.indices[idx]))
    if isinstance(dataset, ConcatDataset):
        if idx < 0:
            raise IndexError(idx)
        offset = 0
        for sub in dataset.datasets:
            sub_len = len(sub)
            if idx < offset + sub_len:
                return _dataset_index_to_image_stem(sub, idx - offset)
            offset += sub_len
        raise IndexError(idx)
    if hasattr(dataset, "_annotation_files"):
        return Path(dataset._annotation_files[idx]).stem
    if hasattr(dataset, "_samples_meta"):
        row = dataset._samples_meta[idx]
        return Path(str(row.get("tile_relpath", ""))).stem
    raise ValueError("dataset.tile_metrics_csv requires DOTA (annotation list) or Airbus Playground (_samples_meta)")


def _load_tile_metrics_by_stem(csv_path: Path, metric_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            key = row.get("image_id") or row.get("image_name") or ""
            if not key:
                continue
            stem = Path(key).stem
            raw = row.get(metric_col)
            if raw is None or raw == "":
                continue
            try:
                out[stem] = float(raw)
            except ValueError:
                continue
    return out


def _stems_vacuous_true_negatives_from_tile_csv(csv_path: Path) -> Set[str]:
    """Stems with tp=fp=fn=0 in tile metrics (no GT, no preds at eval threshold).

    Legacy CSVs may still have f1=0 for these rows; they must not be treated as hard tiles.
    """
    stems: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        fields = reader.fieldnames or []
        if not all(c in fields for c in ("tp", "fp", "fn")):
            return stems
        for row in reader:
            try:
                tp = int(row["tp"])
                fp = int(row["fp"])
                fn = int(row["fn"])
            except (ValueError, KeyError):
                continue
            if tp == 0 and fp == 0 and fn == 0:
                key = row.get("image_id") or row.get("image_name") or ""
                if key:
                    stems.add(Path(key).stem)
    return stems


def _tile_csv_has_count_columns(csv_path: Path) -> bool:
    with csv_path.open(newline="", encoding="utf-8") as f:
        fields = csv_module.DictReader(f).fieldnames or []
    return all(c in fields for c in ("tp", "fp", "fn"))


def _drop_easy_empty_tiles(dataset: Dataset, csv_path: Path) -> Tuple[Dataset, int]:
    """Remove vacuous true-negative tiles (tp=fp=fn=0) from the train set.

    Tiles with no CSV row are kept. Empty tiles with false positives stay so
    hard-tile oversampling can up-weight them.
    """
    if not _tile_csv_has_count_columns(csv_path):
        raise ValueError(
            "dataset.drop_easy_empty_tiles requires tp, fp, and fn columns in "
            f"tile_metrics_csv ({csv_path})"
        )
    vacuous_stems = _stems_vacuous_true_negatives_from_tile_csv(csv_path)
    keep = [
        i
        for i in range(len(dataset))
        if _dataset_index_to_image_stem(dataset, i) not in vacuous_stems
    ]
    n_dropped = len(dataset) - len(keep)
    if n_dropped == 0:
        return dataset, 0
    if not keep:
        raise ValueError(
            "dataset.drop_easy_empty_tiles removed every train tile "
            f"(all were tp=fp=fn=0 in {csv_path})"
        )
    return Subset(dataset, keep), n_dropped


class _HardTileExpandedDataset(Dataset):
    """Repeat hard tile indices so DistributedSampler can oversample (DDP)."""

    def __init__(self, base: Dataset, order: List[int]):
        self.base = base
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int):
        return self.base[self.order[idx]]


from oriented_det.runtime.collate import create_collate_fn, create_train_augmentation, check_directories

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================


def analyze_class_distribution(dataset):
    """Analyze class distribution in dataset and compute class weights."""
    print("Analyzing class distribution in training set...")
    
    class_counts = Counter()
    total_objects = 0
    
    for sample in dataset:
        for ann in sample.annotations:
            class_counts[ann.class_name] += 1
            total_objects += 1
    
    # Handle empty set (e.g. overfit on one image with no matching annotations)
    if not class_counts:
        return class_counts, {}, {}, []
    
    # Sort by count (most frequent first)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    # Compute class weights
    class_weights_inv_freq = {}
    class_weights_sqrt = {}
    
    for class_name, count in sorted_classes:
        # Inverse frequency weighting (normalized)
        weight_inv_freq = total_objects / (len(class_counts) * count)
        class_weights_inv_freq[class_name] = weight_inv_freq
        # Square root weighting (softer)
        weight_sqrt = np.sqrt(max_count / count)
        class_weights_sqrt[class_name] = weight_sqrt
    
    return class_counts, class_weights_inv_freq, class_weights_sqrt, sorted_classes


def compute_class_weights(class_counts, class_weights_dict, method="sqrt", class_weight_overrides=None):
    """Compute and normalize class weights.

    Returns:
        (final_weights, computed_weights): ``computed_weights`` are mean-normalized and clipped
        before ``class_weight_overrides``; ``final_weights`` include overrides.

    class_weight_overrides: optional dict mapping class name -> weight (e.g. {"truck": 0.25})
    to explicitly down-weight dominant classes that the model over-predicts.
    """
    if method == "sqrt":
        weights_dict = {k: np.sqrt(max(class_counts.values()) / v) 
                        for k, v in class_counts.items()}
    elif method == "inv_freq":
        total = sum(class_counts.values())
        weights_dict = {k: total / (len(class_counts) * v) 
                       for k, v in class_counts.items()}
    elif method == "effective_num":
        # Class-Balanced Loss (Cui et al., 2019): w_c = (1-β)/(1-β^n_c)
        # β close to 1.0 yields smoother weights than inv_freq.
        beta = 0.9999
        weights_dict = {k: (1.0 - beta) / max(1e-12, (1.0 - (beta ** float(v))))
                        for k, v in class_counts.items()}
    elif method == "log_inv_freq":
        total = float(sum(class_counts.values()))
        weights_dict = {k: math.log((total / max(1.0, float(v))) + 1.0)
                        for k, v in class_counts.items()}
    else:
        raise ValueError(f"Unknown CLASS_WEIGHT_METHOD: {method}")
    
    # Normalize weights to have mean=1.0
    weight_values = list(weights_dict.values())
    mean_weight = np.mean(weight_values)
    weights_dict_normalized = {k: v / mean_weight for k, v in weights_dict.items()}
    
    # Clip extreme weights to prevent training instability
    MAX_WEIGHT = 3.0
    MIN_WEIGHT = 0.3
    weights_dict_normalized = {
        k: np.clip(v, MIN_WEIGHT, MAX_WEIGHT) 
        for k, v in weights_dict_normalized.items()
    }
    
    computed_weights = {k: float(v) for k, v in weights_dict_normalized.items()}

    # Apply per-class overrides (e.g. to further down-weight a dominant class like truck)
    if class_weight_overrides:
        for class_name, weight in class_weight_overrides.items():
            if class_name in weights_dict_normalized:
                weights_dict_normalized[class_name] = float(weight)
            # background is set separately via loss.background_weight in train.py

    return weights_dict_normalized, computed_weights


def _print_class_weight_table(
    roi_class_weights: Dict[str, float],
    computed_weights: Dict[str, float],
    class_counts: Dict[str, int],
    *,
    method: str,
    class_weight_overrides: Optional[Dict[str, float]] = None,
) -> None:
    """Log per-class ROI weights: computed (pre-override), optional override, and final."""
    print(f"\nUsing {method} class weighting")
    overrides = class_weight_overrides or {}
    if overrides:
        print(f"  Config overrides (replace computed after clip): {overrides}")
    print(f"\nClass weights (normalized, mean=1.0):")
    print(f"  {'Class Name':<24} {'computed':>10} {'override':>10} {'final':>10}  {'count':>8}")
    print("  " + "-" * 76)
    foreground = [
        (name, roi_class_weights[name])
        for name in roi_class_weights
        if name not in ("background", "__background__")
    ]
    for class_name, final_w in sorted(foreground, key=lambda x: x[1], reverse=True):
        comp = computed_weights.get(class_name, final_w)
        ov = overrides.get(class_name)
        ov_str = f"{float(ov):>10.3f}" if ov is not None else f"{'—':>10}"
        count = class_counts.get(class_name, 0)
        print(f"  {class_name:<24} {comp:>10.3f} {ov_str} {final_w:>10.3f}  {count:>8,}")
    for bg_key in ("background", "__background__"):
        if bg_key in roi_class_weights:
            final_w = roi_class_weights[bg_key]
            print(f"  {bg_key:<24} {'—':>10} {'—':>10} {final_w:>10.3f}  {'n/a':>8}")
            break


def _weights_dict_to_tensor(
    weights_dict: Dict[str, float],
    class_map: Dict[str, int],
    num_classes: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert weights dict (incl optional 'background') to tensor [num_classes+1]."""
    w = torch.ones(num_classes + 1, dtype=torch.float32, device=device)
    if "background" in weights_dict:
        w[0] = float(weights_dict["background"])
    for name, cid in class_map.items():
        if 1 <= int(cid) <= num_classes and name in weights_dict:
            w[int(cid)] = float(weights_dict[name])
    return w


def _ramp(t: float, power: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    p = max(1e-6, float(power))
    return t ** p


def _round_ratio(value: float) -> float:
    """Round anchor ratio to readable, stable values."""
    if value <= 0:
        return 1.0
    if value >= 5:
        return round(value, 1)
    if value >= 2:
        return round(value, 2)
    return round(value, 3)


def gather_wizard_stats(dataset) -> Dict[str, Any]:
    """Collect geometry and density statistics from training samples."""
    per_image_objects: List[int] = []
    widths: List[float] = []
    heights: List[float] = []
    aspects: List[float] = []
    areas: List[float] = []
    angles: List[float] = []

    for sample in dataset:
        num_objs = len(sample.annotations)
        per_image_objects.append(num_objs)
        for ann in sample.annotations:
            w = max(float(ann.rbox.width), 1e-6)
            h = max(float(ann.rbox.height), 1e-6)
            widths.append(w)
            heights.append(h)
            aspects.append(max(w / h, h / w))
            areas.append(w * h)
            angle = float(ann.rbox.angle)
            wrapped = ((angle + (math.pi / 2.0)) % math.pi) - (math.pi / 2.0)
            angles.append(wrapped)

    return {
        "num_images": len(per_image_objects),
        "num_objects": len(widths),
        "per_image_objects": np.array(per_image_objects, dtype=np.float32),
        "widths": np.array(widths, dtype=np.float32),
        "heights": np.array(heights, dtype=np.float32),
        "aspects": np.array(aspects, dtype=np.float32),
        "areas": np.array(areas, dtype=np.float32),
        "angles": np.array(angles, dtype=np.float32),
    }


def print_wizard_recommendations(
    stats: Dict[str, Any],
    config: TrainingExperimentConfig,
    *,
    world_size: int,
) -> None:
    """Print data-driven config recommendations from training distribution."""
    if stats["num_images"] == 0:
        print("\n[Wizard] Skipped: no training images found.")
        return

    per_img = stats["per_image_objects"]
    widths = stats["widths"]
    heights = stats["heights"]
    aspects = stats["aspects"]
    areas = stats["areas"]
    angles = stats["angles"]

    empty_ratio = float((per_img == 0).mean()) if per_img.size > 0 else 0.0
    mean_obj = float(per_img.mean()) if per_img.size > 0 else 0.0
    p95_obj = float(np.percentile(per_img, 95)) if per_img.size > 0 else 0.0
    p99_obj = float(np.percentile(per_img, 99)) if per_img.size > 0 else 0.0

    print(f"\n{'='*80}")
    print("Wizard: Training Data Diagnostics")
    print(f"{'='*80}")
    print(f"Images: {stats['num_images']:,}")
    print(f"Objects: {stats['num_objects']:,}")
    print(f"Objects/image: mean={mean_obj:.2f}, p95={p95_obj:.1f}, p99={p99_obj:.1f}, max={int(per_img.max()) if per_img.size > 0 else 0}")
    print(f"Empty images: {empty_ratio:.2%}")

    if widths.size > 0:
        print(
            "Box size (pixels): "
            f"w p10/p50/p90={np.percentile(widths, 10):.1f}/{np.percentile(widths, 50):.1f}/{np.percentile(widths, 90):.1f}, "
            f"h p10/p50/p90={np.percentile(heights, 10):.1f}/{np.percentile(heights, 50):.1f}/{np.percentile(heights, 90):.1f}"
        )
        print(
            "Aspect ratio (max(w/h,h/w)): "
            f"p50={np.percentile(aspects, 50):.2f}, p90={np.percentile(aspects, 90):.2f}, p99={np.percentile(aspects, 99):.2f}"
        )
        print(
            "Area (pixels^2): "
            f"p10/p50/p90={np.percentile(areas, 10):.1f}/{np.percentile(areas, 50):.1f}/{np.percentile(areas, 90):.1f}"
        )
        print(f"Angle spread (std, wrapped le90): {float(np.std(angles)):.3f} rad")

    print(f"\n[Wizard] Recommendations:")
    model_type = str(getattr(config, "model_type", "")).lower()
    is_fcos = model_type == "rotated_fcos"

    current_max_det = int(getattr(config.model, "max_detections_per_image", 100))
    # Dense aerial scenes need headroom above p99 objects/image; DOTA recipes often keep 2000.
    suggested_max_det = int(np.clip(math.ceil(max(20.0, p99_obj) * 3.0), 100, 3000))
    if is_fcos:
        # Match MMRotate Rotated FCOS test_cfg.max_per_img=2000 for DOTA-scale density.
        suggested_max_det = max(suggested_max_det, 2000)
    if current_max_det < int(0.6 * suggested_max_det):
        print(
            f"- max_detections_per_image: {current_max_det} -> ~{suggested_max_det} "
            f"(current is likely tight vs p99 objects/image={p99_obj:.1f})."
        )
    elif current_max_det > int(2.0 * suggested_max_det) and not is_fcos:
        print(
            f"- max_detections_per_image: {current_max_det} is high for this data; try ~{suggested_max_det} "
            "(can reduce false positives and evaluation overhead)."
        )
    else:
        print(f"- max_detections_per_image: {current_max_det} looks reasonable for this dataset.")

    if is_fcos:
        strides = [int(s) for s in getattr(config.model, "fpn_strides", [8, 16, 32, 64, 128])]
        min_stride = min(strides) if strides else 8
        if widths.size > 0:
            w10 = float(np.percentile(widths, 10))
            w50 = float(np.percentile(widths, 50))
            cells_p10 = w10 / float(min_stride)
            print(
                f"- fpn_strides: finest={min_stride}px; GT width p10/p50={w10:.1f}/{w50:.1f}px "
                f"(~{cells_p10:.1f}/{w50 / float(min_stride):.1f} cells on P3). "
                "Small vehicles near 1–2 cells are hard for encoded L1; keep P3 and "
                "prefer decoded `kfiou` for overlap."
            )
        radius = float(getattr(config.model, "fcos_center_sample_radius", 1.5))
        if radius < 1.5:
            print(
                f"- fcos_center_sample_radius: {radius} -> 1.5 "
                "(MMRotate default; too-small radius starves positives on dense tiles)."
            )
        else:
            print(f"- fcos_center_sample_radius: {radius} matches MMRotate (1.5×stride).")
        box_loss = str(getattr(config.model, "box_reg_loss_type", "l1")).lower()
        aux_type = getattr(config.model, "aux_loss_type", None)
        aux_w = float(getattr(config.model, "aux_loss_weight", 0.0) or 0.0)
        if box_loss in ("l1", "smooth_l1"):
            print(
                f"- box_reg_loss_type: {box_loss} (L1 baseline). Dense small classes "
                "(ship/small-vehicle/storage-tank) often lag; try L1 + decoded aux "
                "(configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json)."
            )
        elif box_loss == "kfiou":
            print("- box_reg_loss_type: kfiou (decoded KFIoU primary). Prefer L1 + aux recipes.")
        elif box_loss == "riou":
            print(
                "- box_reg_loss_type: riou (decoded differentiable polygon IoU, 1-IoU). "
                "Recipe configs/rotated_fcos/dota_le90_1x.json uses lr 2.5e-3. "
                "Not sampling pairwise_rotated_iou."
            )
        else:
            print(f"- box_reg_loss_type: {box_loss}")
        if aux_w > 0.0:
            ang_w = float(getattr(config.model, "aux_angle_weight", 1.0) or 0.0)
            ang_lam = float(getattr(config.model, "aux_angle_lambda", 1.0) or 1.0)
            print(
                f"- aux_loss_type: {aux_type} weight={aux_w:g} (centerness-weighted decoded "
                f"+ gated angle weight={ang_w:g} λ={ang_lam:g}). "
                "Keep L1 primary and lr 2.5e-4."
            )
        else:
            print("- aux_loss_weight: 0 (decoded aux off).")
        print(
            "- anchors / target_stds: skipped (anchor-free DistanceAnglePointCoder; "
            "not DeltaXYWHA)."
        )
        print("- tip: run `make lr-finder` to tune learning_rate.")
    else:
        current_ratios = [
            float(r) for r in getattr(config.model, "anchor_ratios", [0.5, 1.0, 2.0]) if float(r) > 0
        ]
        current_aspects = [max(r, 1.0 / r) for r in current_ratios]
        if aspects.size > 0:
            q70 = float(np.percentile(aspects, 70))
            q90 = float(np.percentile(aspects, 90))
            q99 = float(np.percentile(aspects, 99))
            suggested_ratios = sorted(
                {
                    _round_ratio(1.0 / max(q90, 1.0)),
                    _round_ratio(1.0 / max(q70, 1.0)),
                    1.0,
                    _round_ratio(max(q70, 1.0)),
                    _round_ratio(max(q90, 1.0)),
                }
            )
            current_min_aspect = min(current_aspects) if current_aspects else 1.0
            current_max_aspect = max(current_aspects) if current_aspects else 1.0
            coverage = float(
                ((aspects >= current_min_aspect) & (aspects <= current_max_aspect)).mean()
            ) if aspects.size > 0 else 1.0
            if coverage < 0.9 or current_max_aspect < 0.8 * q90:
                print(
                    "- anchor_ratios: current="
                    f"{current_ratios}, suggested~{suggested_ratios} "
                    f"(current covers {coverage:.1%} of GT aspect spread; p90={q90:.2f}, p99={q99:.2f})."
                )
            else:
                print(
                    f"- anchor_ratios: current {current_ratios} already covers GT aspect spread well "
                    f"(coverage {coverage:.1%})."
                )

        current_target_stds = list(getattr(config.model, "target_stds", (0.1, 0.1, 0.2, 0.2, 0.1)))
        if widths.size > 0:
            spread_w = float(np.std(np.log(np.clip(widths, 1e-6, None))))
            spread_h = float(np.std(np.log(np.clip(heights, 1e-6, None))))
            spread_wh = 0.5 * (spread_w + spread_h)
            spread_angle = float(np.std(angles))
            wh_std = float(np.clip(0.12 + 0.18 * spread_wh, 0.15, 0.4))
            angle_std = float(
                np.clip(0.05 + 0.25 * (spread_angle / (math.pi / 2.0)), 0.05, 0.25)
            )
            suggested_target_stds = [0.1, 0.1, round(wh_std, 3), round(wh_std, 3), round(angle_std, 3)]
            needs_update = len(current_target_stds) == 5 and (
                abs(float(current_target_stds[2]) - suggested_target_stds[2]) > 0.08
                or abs(float(current_target_stds[4]) - suggested_target_stds[4]) > 0.05
            )
            if needs_update:
                print(
                    f"- target_stds: current={current_target_stds}, suggested~{suggested_target_stds} "
                    "(derived from GT scale/angle spread; useful when box-reg is unstable)."
                )
            else:
                print(f"- target_stds: current {current_target_stds} looks close to data spread.")

    batch_size = int(config.data_loader.batch_size)
    grad_acc = max(1, int(config.training.gradient_accumulation_steps))
    global_batch = max(1, batch_size * max(1, world_size))
    batches_per_epoch = int(math.ceil(stats["num_images"] / global_batch))
    opt_steps_per_epoch = int(math.ceil(batches_per_epoch / grad_acc))

    lr_scaling = 1.0
    if grad_acc > 1 and config.training.lr_scaling_with_accumulation != "none":
        if config.training.lr_scaling_with_accumulation == "linear":
            lr_scaling *= grad_acc
        elif config.training.lr_scaling_with_accumulation == "sqrt":
            lr_scaling *= math.sqrt(grad_acc)
    if world_size > 1 and bool(getattr(config.training, "lr_scale_with_world_size", False)):
        lr_scaling *= world_size
    scaled_lr = float(config.training.learning_rate) * lr_scaling

    warmup_base_epochs = 1.0 if scaled_lr >= 0.01 else 0.5
    suggested_warmup = int(np.clip(round(warmup_base_epochs * opt_steps_per_epoch), 50, 2000))
    current_warmup = int(getattr(config.training, "lr_warmup_steps", 0))
    if current_warmup < int(0.5 * suggested_warmup):
        print(
            f"- lr_warmup_steps: {current_warmup} -> ~{suggested_warmup} "
            f"(scaled LR≈{scaled_lr:.5f}, optimizer steps/epoch≈{opt_steps_per_epoch})."
        )
    elif current_warmup > int(2.0 * suggested_warmup):
        print(
            f"- lr_warmup_steps: {current_warmup} is long for this setup; try ~{suggested_warmup} "
            f"(scaled LR≈{scaled_lr:.5f})."
        )
    else:
        print(f"- lr_warmup_steps: {current_warmup} looks reasonable.")

    if empty_ratio > 0.4:
        print(
            "- data quality: many empty tiles detected (>40%). Consider denser tiling/overlap or "
            "increase positive sampling pressure."
        )

    print(f"{'='*80}\n")


def create_model_from_config(
    config: TrainingExperimentConfig,
    num_classes: int,
    device: torch.device,
    roi_class_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
):
    """Create model based on config model_type."""
    model_type = config.model_type.lower()
    
    # Determine loss type and class weights from config
    loss_config = config.loss
    if loss_config.loss_type in ["cross_entropy", "class_weighted"]:
        roi_loss_type = "cross_entropy"
    elif loss_config.loss_type in ["focal", "focal_weighted"]:
        roi_loss_type = "focal"
    else:
        roi_loss_type = config.model.roi_loss_type
    
    # Use loss-section focal params when using focal/focal_weighted so config.loss.focal_alpha applies
    if loss_config.loss_type in ["focal", "focal_weighted"]:
        roi_focal_alpha = getattr(loss_config, "focal_alpha", config.model.roi_focal_alpha)
        roi_focal_gamma = getattr(loss_config, "focal_gamma", config.model.roi_focal_gamma)
    else:
        roi_focal_alpha = config.model.roi_focal_alpha
        roi_focal_gamma = config.model.roi_focal_gamma
    roi_label_smoothing = getattr(config.loss, "label_smoothing", 0.0)

    # Convert target_means/stds from list to tuple
    target_means = tuple(config.model.target_means) if isinstance(config.model.target_means, list) else config.model.target_means
    target_stds = tuple(config.model.target_stds) if isinstance(config.model.target_stds, list) else config.model.target_stds
    
    use_hbb = getattr(config.model, "use_hbb_for_matching", False)
    inference_pre_nms_score_threshold = getattr(config.model, "inference_pre_nms_score_threshold", 0.05)
    fpn_returned_layers = getattr(config.model, "fpn_returned_layers", None)
    fpn_strides = getattr(config.model, "fpn_strides", None)
    # Resolve trainable_layers from frozen_stages if set (MMRotate: frozen_stages=1 -> train stages 2,3,4 -> trainable_layers=3)
    frozen_stages = getattr(config.model, "frozen_stages", None)
    if frozen_stages is not None:
        trainable_layers = 5 if frozen_stages == 0 else max(1, 4 - frozen_stages)
    else:
        trainable_layers = config.model.trainable_layers
    if model_type == "rotated_faster_rcnn":
        model = RotatedFasterRCNN(
            num_classes=num_classes,
            backbone_name=config.model.backbone,
            pretrained_backbone=config.model.pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=fpn_returned_layers,
            fpn_strides=fpn_strides,
            anchor_scales=config.model.anchor_scales,
            anchor_ratios=config.model.anchor_ratios,
            roi_loss_type=roi_loss_type,
            roi_class_weights=roi_class_weights,
            roi_focal_alpha=roi_focal_alpha,
            roi_focal_gamma=roi_focal_gamma,
            roi_label_smoothing=roi_label_smoothing,
            target_means=target_means,
            target_stds=target_stds,
            roi_norm_factor=config.model.roi_norm_factor,
            roi_edge_swap=config.model.roi_edge_swap,
            roi_proj_xy=getattr(config.model, "roi_proj_xy", False),
            roi_box_reg_angle_weight=getattr(config.model, "roi_box_reg_angle_weight", 1.0),
            roi_box_reg_angle_schedule_epochs=getattr(
                config.model, "roi_box_reg_angle_schedule_epochs", None
            ),
            roi_box_reg_angle_schedule_values=getattr(
                config.model, "roi_box_reg_angle_schedule_values", None
            ),
            roi_box_reg_aux_weight=getattr(config.model, "roi_box_reg_aux_weight", 0.0),
            roi_box_reg_aux_loss_type=getattr(config.model, "roi_box_reg_aux_loss_type", None),
            roi_box_reg_kfiou_fun=getattr(config.model, "roi_box_reg_kfiou_fun", None),
            roi_box_reg_probiou_mode=getattr(config.model, "roi_box_reg_probiou_mode", None),
            roi_box_reg_main_loss_type=getattr(
                config.model, "roi_box_reg_main_loss_type", "smooth_l1"
            ),
            roi_box_reg_norm=getattr(config.model, "roi_box_reg_norm", "sampled_all"),
            use_hbb_for_matching=use_hbb,
            inference_pre_nms_score_threshold=inference_pre_nms_score_threshold,
            rpn_min_size=getattr(config.model, "rpn_min_size", 0.0),
            rpn_pre_nms_top_n=getattr(config.model, "rpn_pre_nms_top_n", 2000),
            rpn_post_nms_top_n=getattr(config.model, "rpn_post_nms_top_n", 2000),
            max_detections_per_image=getattr(config.model, "max_detections_per_image", 2000),
            rpn_nms_threshold=getattr(config.model, "rpn_nms_threshold", 0.7),
            final_nms_iou_threshold=config.model.final_nms_iou_threshold,
            nms_class_agnostic=getattr(config.model, "nms_class_agnostic", False),
            final_nms_use_cpu=getattr(config.model, "final_nms_use_cpu", False),
            roi_batch_size_per_image=getattr(config.model, "roi_batch_size_per_image", 512),
            rpn_batch_size_per_image=getattr(config.model, "rpn_batch_size_per_image", 256),
            rpn_min_pos_iou=getattr(config.model, "rpn_min_pos_iou", 0.3),
            rpn_match_low_quality=getattr(config.model, "rpn_match_low_quality", True),
            roi_match_low_quality=getattr(config.model, "roi_match_low_quality", False),
            roi_min_pos_iou=getattr(config.model, "roi_min_pos_iou", 0.5),
            add_gt_as_proposals=getattr(config.model, "add_gt_as_proposals", True),
            rpn_positive_iou_threshold=getattr(config.model, "rpn_positive_iou_threshold", 0.7),
            rpn_negative_iou_threshold=getattr(config.model, "rpn_negative_iou_threshold", 0.3),
            roi_positive_iou_threshold=getattr(config.model, "roi_positive_iou_threshold", 0.5),
            roi_negative_iou_threshold=getattr(config.model, "roi_negative_iou_threshold", 0.5),
            final_nms_iou_schedule_epochs=config.model.final_nms_iou_schedule_epochs,
            final_nms_iou_schedule_values=config.model.final_nms_iou_schedule_values,
            roi_box_reg_aux_schedule_epochs=config.model.roi_box_reg_aux_schedule_epochs,
            roi_box_reg_aux_schedule_values=config.model.roi_box_reg_aux_schedule_values,
            roi_inference_top_class_only=getattr(
                config.model, "roi_inference_top_class_only", False
            ),
        )
    elif model_type == "oriented_rcnn":
        model = OrientedRCNN(
            num_classes=num_classes,
            backbone_name=config.model.backbone,
            pretrained_backbone=config.model.pretrained_backbone,
            trainable_layers=trainable_layers,
            anchor_scales=config.model.anchor_scales,
            returned_layers=fpn_returned_layers,
            fpn_strides=fpn_strides,
            anchor_ratios=config.model.anchor_ratios,
            roi_loss_type=roi_loss_type,
            roi_class_weights=roi_class_weights,
            roi_focal_alpha=roi_focal_alpha,
            roi_focal_gamma=roi_focal_gamma,
            roi_label_smoothing=roi_label_smoothing,
            target_means=target_means,
            target_stds=target_stds,
            roi_norm_factor=config.model.roi_norm_factor,
            roi_edge_swap=config.model.roi_edge_swap,
            roi_proj_xy=getattr(config.model, "roi_proj_xy", False),
            roi_box_reg_angle_weight=getattr(config.model, "roi_box_reg_angle_weight", 1.0),
            roi_box_reg_angle_schedule_epochs=getattr(
                config.model, "roi_box_reg_angle_schedule_epochs", None
            ),
            roi_box_reg_angle_schedule_values=getattr(
                config.model, "roi_box_reg_angle_schedule_values", None
            ),
            roi_box_reg_aux_weight=getattr(config.model, "roi_box_reg_aux_weight", 0.0),
            roi_box_reg_aux_loss_type=getattr(config.model, "roi_box_reg_aux_loss_type", None),
            roi_box_reg_kfiou_fun=getattr(config.model, "roi_box_reg_kfiou_fun", None),
            roi_box_reg_probiou_mode=getattr(config.model, "roi_box_reg_probiou_mode", None),
            use_hbb_for_matching=use_hbb,
            roi_use_hbb_for_matching=getattr(config.model, "roi_use_hbb_for_matching", False),
            roi_box_reg_norm=getattr(config.model, "roi_box_reg_norm", "sampled_all"),
            inference_pre_nms_score_threshold=inference_pre_nms_score_threshold,
            rpn_pre_nms_top_n=getattr(config.model, "rpn_pre_nms_top_n", 2000),
            rpn_post_nms_top_n=getattr(config.model, "rpn_post_nms_top_n", 1000),
            max_detections_per_image=getattr(config.model, "max_detections_per_image", 100),
            rpn_nms_threshold=getattr(config.model, "rpn_nms_threshold", 0.7),
            final_nms_iou_threshold=config.model.final_nms_iou_threshold,
            nms_class_agnostic=getattr(config.model, "nms_class_agnostic", False),
            final_nms_use_cpu=getattr(config.model, "final_nms_use_cpu", False),
            roi_batch_size_per_image=getattr(config.model, "roi_batch_size_per_image", 512),
            rpn_batch_size_per_image=getattr(config.model, "rpn_batch_size_per_image", 256),
            rpn_min_pos_iou=getattr(config.model, "rpn_min_pos_iou", 0.3),
            rpn_match_low_quality=getattr(config.model, "rpn_match_low_quality", True),
            roi_match_low_quality=getattr(config.model, "roi_match_low_quality", False),
            rpn_positive_iou_threshold=getattr(config.model, "rpn_positive_iou_threshold", 0.5),
            rpn_negative_iou_threshold=getattr(config.model, "rpn_negative_iou_threshold", 0.2),
            roi_positive_iou_threshold=getattr(config.model, "roi_positive_iou_threshold", 0.4),
            roi_negative_iou_threshold=getattr(config.model, "roi_negative_iou_threshold", 0.3),
            final_nms_iou_schedule_epochs=config.model.final_nms_iou_schedule_epochs,
            final_nms_iou_schedule_values=config.model.final_nms_iou_schedule_values,
            roi_box_reg_aux_schedule_epochs=config.model.roi_box_reg_aux_schedule_epochs,
            roi_box_reg_aux_schedule_values=config.model.roi_box_reg_aux_schedule_values,
            add_gt_as_proposals=getattr(config.model, "add_gt_as_proposals", True),
            roi_inference_top_class_only=getattr(
                config.model, "roi_inference_top_class_only", False
            ),
        )
    elif model_type == "rotated_retinanet":
        model = RotatedRetinaNet(
            num_classes=num_classes,
            backbone_name=config.model.backbone,
            pretrained_backbone=config.model.pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=fpn_returned_layers,
            fpn_strides=fpn_strides,
            fpn_extra_level=getattr(config.model, "fpn_extra_level", False),
            anchor_scales=config.model.anchor_scales,
            anchor_ratios=config.model.anchor_ratios,
            octave_base_scale=getattr(config.model, "anchor_octave_base_scale", None),
            scales_per_octave=getattr(config.model, "anchor_scales_per_octave", None),
            stacked_convs=getattr(config.model, "retinanet_stacked_convs", 4),
            positive_iou_threshold=getattr(config.model, "rpn_positive_iou_threshold", 0.5),
            negative_iou_threshold=getattr(config.model, "rpn_negative_iou_threshold", 0.4),
            focal_alpha=roi_focal_alpha,
            focal_gamma=roi_focal_gamma,
            target_means=target_means,
            target_stds=target_stds,
            norm_factor=config.model.roi_norm_factor,
            edge_swap=config.model.roi_edge_swap,
            box_reg_weight=getattr(config.model, "box_reg_weight", 1.0),
            box_reg_loss_type=getattr(config.model, "box_reg_loss_type", "smooth_l1"),
            box_reg_aux_weight=getattr(config.model, "roi_box_reg_aux_weight", 0.0),
            box_reg_aux_loss_type=getattr(config.model, "roi_box_reg_aux_loss_type", None),
            box_reg_kfiou_fun=getattr(config.model, "roi_box_reg_kfiou_fun", None),
            box_reg_probiou_mode=getattr(config.model, "roi_box_reg_probiou_mode", None),
            box_reg_main_loss_type=getattr(
                config.model, "roi_box_reg_main_loss_type", "smooth_l1"
            ),
            reg_sample_size_per_image=getattr(
                config.model, "roi_batch_size_per_image", 512
            ),
            use_hbb_for_matching=config.model.use_hbb_for_matching,
            score_threshold=inference_pre_nms_score_threshold,
            final_nms_iou_threshold=config.model.final_nms_iou_threshold,
            max_detections_per_image=getattr(config.model, "max_detections_per_image", 100),
            final_nms_iou_schedule_epochs=config.model.final_nms_iou_schedule_epochs,
            final_nms_iou_schedule_values=config.model.final_nms_iou_schedule_values,
            roi_box_reg_aux_schedule_epochs=config.model.roi_box_reg_aux_schedule_epochs,
            roi_box_reg_aux_schedule_values=config.model.roi_box_reg_aux_schedule_values,
            final_nms_use_cpu=getattr(config.model, "final_nms_use_cpu", False),
        )
    elif model_type == "rotated_fcos":
        rr = getattr(config.model, "fcos_regress_ranges", None)
        regress_ranges = None
        if rr is not None:
            regress_ranges = [tuple(pair) for pair in rr]
        model = RotatedFCOS(
            num_classes=num_classes,
            backbone_name=config.model.backbone,
            pretrained_backbone=config.model.pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=fpn_returned_layers,
            fpn_strides=fpn_strides,
            fpn_extra_level=getattr(config.model, "fpn_extra_level", True),
            stacked_convs=getattr(config.model, "fcos_stacked_convs", 4),
            center_sampling=getattr(config.model, "fcos_center_sampling", True),
            center_sample_radius=getattr(config.model, "fcos_center_sample_radius", 1.5),
            norm_on_bbox=getattr(config.model, "fcos_norm_on_bbox", True),
            centerness_on_reg=getattr(config.model, "fcos_centerness_on_reg", True),
            scale_angle=getattr(config.model, "fcos_scale_angle", True),
            regress_ranges=regress_ranges,
            focal_alpha=roi_focal_alpha,
            focal_gamma=roi_focal_gamma,
            box_reg_weight=getattr(config.model, "box_reg_weight", 1.0),
            box_reg_loss_type=getattr(config.model, "box_reg_loss_type", "l1"),
            aux_loss_type=getattr(config.model, "aux_loss_type", None),
            aux_loss_weight=getattr(config.model, "aux_loss_weight", 0.0),
            aux_angle_weight=getattr(config.model, "aux_angle_weight", 1.0),
            aux_angle_lambda=getattr(config.model, "aux_angle_lambda", 1.0),
            angle_weight=getattr(config.model, "fcos_angle_weight", 1.0),
            score_threshold=inference_pre_nms_score_threshold,
            final_nms_iou_threshold=config.model.final_nms_iou_threshold,
            max_detections_per_image=getattr(config.model, "max_detections_per_image", 2000),
            nms_pre=getattr(config.model, "fcos_nms_pre", 2000),
            final_nms_iou_schedule_epochs=config.model.final_nms_iou_schedule_epochs,
            final_nms_iou_schedule_values=config.model.final_nms_iou_schedule_values,
            final_nms_use_cpu=getattr(config.model, "final_nms_use_cpu", False),
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: rotated_faster_rcnn, "
            "oriented_rcnn, rotated_retinanet, rotated_fcos"
        )
    
    model.to(device)
    return model, roi_loss_type


def _strip_module_prefix(param_name: str) -> str:
    """Normalize DDP/DataParallel parameter names for prefix matching."""
    return param_name[7:] if param_name.startswith("module.") else param_name


def build_optimizer_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    config: TrainingExperimentConfig,
    *,
    include_frozen_parameters: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Build optimizer param groups with per-module LR multipliers.

    - ``backbone.*`` → ``lr_mult_backbone``
    - ``head.*`` (e.g. RetinaNet) → ``lr_mult_head`` if set, else ``lr_mult_other``
    - ``rpn_head.*`` / ``roi_head.*`` → if ``lr_mult_head`` is set, both use that multiplier;
      otherwise ``lr_mult_rpn`` / ``lr_mult_roi``
    - any other trainable tensor → ``lr_mult_other``

    When ``include_frozen_parameters`` is True, parameters with ``requires_grad=False`` are still
    added to their groups so the optimizer state stays aligned when unfreezing later; SGD skips
    parameters with no gradient.
    """
    multipliers = {
        "backbone": float(getattr(config.training, "lr_mult_backbone", 0.5)),
        "rpn": float(getattr(config.training, "lr_mult_rpn", 0.5)),
        "roi": float(getattr(config.training, "lr_mult_roi", 2.0)),
        "other": float(getattr(config.training, "lr_mult_other", 1.0)),
    }
    lr_mult_head = getattr(config.training, "lr_mult_head", None)

    grouped_params: Dict[str, List[torch.nn.Parameter]] = {
        "backbone": [],
        "rpn": [],
        "roi": [],
        "head": [],
        "other": [],
    }
    grouped_counts: Dict[str, int] = {k: 0 for k in grouped_params}

    for name, param in model.named_parameters():
        if not include_frozen_parameters and not param.requires_grad:
            continue
        clean_name = _strip_module_prefix(name)
        if clean_name.startswith("backbone."):
            group_name = "backbone"
        elif clean_name.startswith("head."):
            group_name = "head"
        elif clean_name.startswith("rpn_head."):
            group_name = "rpn"
        elif clean_name.startswith("roi_head."):
            group_name = "roi"
        else:
            group_name = "other"
        grouped_params[group_name].append(param)
        grouped_counts[group_name] += param.numel()

    def _effective_multiplier(group_name: str) -> float:
        if group_name == "backbone":
            return multipliers["backbone"]
        if lr_mult_head is not None:
            if group_name in ("head", "rpn", "roi"):
                return float(lr_mult_head)
            return multipliers["other"]
        if group_name == "head":
            return multipliers["other"]
        return multipliers[group_name]

    param_groups: List[Dict[str, Any]] = []
    group_summary: Dict[str, Dict[str, float]] = {}
    for group_name in ["backbone", "rpn", "roi", "head", "other"]:
        params = grouped_params[group_name]
        if not params:
            continue
        eff_mult = _effective_multiplier(group_name)
        group_lr = base_lr * eff_mult
        param_groups.append(
            {
                "params": params,
                "lr": group_lr,
                "weight_decay": weight_decay,
                "group_name": group_name,
                # Used by the training engine to recover config-scale LR for TensorBoard (group 0 is often backbone).
                "lr_multiplier": float(eff_mult),
            }
        )
        group_summary[group_name] = {
            "lr": float(group_lr),
            "multiplier": float(eff_mult),
            "num_params": float(grouped_counts[group_name]),
        }

    return param_groups, group_summary


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train oriented object detection models on DOTA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=None,
        help="Enable automatic mixed precision training (overrides config)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision training (overrides config)"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Local rank for distributed training (set by torchrun)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs: config summary, loss breakdown, RPN/ROI stats to TensorBoard, "
             "per-class mAP/det/GT counts, GT cover rates, first-batch GT stats. Use to diagnose low mAP vs MMRotate."
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Run data/config diagnostics and print recommendations before training"
    )
    args = parser.parse_args()
    
    # Detect distributed training (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = os.environ.get("DIST_BACKEND", "auto")
        if backend == "auto":
            backend = "nccl" if dist.is_backend_available("nccl") else "gloo"
        if backend == "gloo":
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            init_method = f"tcp://{master_addr}:{master_port}"
        else:
            init_method = "env://"
        if not dist.is_initialized():
            from datetime import timedelta
            # Long runs: periodic full-val mAP on rank 0 can exceed 30 minutes while other
            # ranks wait on Gloo barriers; use a generous default (override via TORCH_DIST_TIMEOUT).
            _dist_timeout_s = int(
                os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", str(24 * 3600))
            )
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=max(60, _dist_timeout_s)),
            )
    else:
        from oriented_det.utils import get_device
        device = get_device()

    # Load config from file (supports vendored configs for PyPI installs)
    from oriented_det.utils.config import _framework_config_roots  # local import to keep CLI fast

    raw = str(args.config)
    config_path = Path(raw)
    if not config_path.exists():
        # Common UX: allow `--config configs/...` after `pip install oriented-det`
        rel = raw.replace("\\", "/").lstrip("/")
        if rel.startswith("configs/"):
            rel = rel[len("configs/") :]
        for fw_root in _framework_config_roots():
            candidate = (fw_root / rel).resolve()
            if candidate.exists():
                config_path = candidate
                break
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {raw}. "
            "Tip: pass a path to a JSON config, or use a built-in path like "
            "'configs/oriented_rcnn/dota_le90_1x.json'."
        )
    
    config = TrainingExperimentConfig.load(config_path)

    # Apply command-line overrides
    if args.batch_size is not None:
        config.data_loader.batch_size = args.batch_size
        if rank == 0:
            print(f"Overriding batch size to: {args.batch_size}")
    
    if args.no_amp:
        config.training.use_amp = False
        if rank == 0:
            print("Overriding AMP: disabled")
    elif args.use_amp:
        config.training.use_amp = True
        if rank == 0:
            print("Overriding AMP: enabled")
    
    # Set experiment timestamp if not set
    if config.experiment_timestamp is None:
        config.experiment_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create experiment directory and tee stdout/stderr to train.log (rank 0 only; multi-GPU safe)
    # In wizard mode we intentionally skip run-folder/log creation.
    from oriented_det.train.utils import get_project_root

    project_root = get_project_root()
    try:
        config.source_recipe = str(config_path.resolve().relative_to(project_root))
    except ValueError:
        config.source_recipe = str(config_path.resolve())

    if not args.wizard:
        for key, value in capture_source_provenance().items():
            setattr(config, key, value)

    EXPERIMENT_DIR = project_root / "runs" / config.model_type / config.experiment_timestamp
    _log_file = None
    _orig_stdout = _orig_stderr = None
    if not args.wizard and rank == 0:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _log_path = EXPERIMENT_DIR / "train.log"
        _log_file = open(_log_path, "w", encoding="utf-8")

        class _Tee:
            def __init__(self, stream, file):
                self._stream = stream
                self._file = file
            def write(self, s):
                self._stream.write(s)
                self._file.write(s)
                self._stream.flush()
                self._file.flush()
            def flush(self):
                self._stream.flush()
                self._file.flush()
            def writable(self):
                return True

        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        sys.stdout = _Tee(_orig_stdout, _log_file)
        sys.stderr = _Tee(_orig_stderr, _log_file)
        print(f"Loading configuration from: {config_path}")
        print(f"Training log file: {_log_path}")
    elif rank == 0:
        print(f"Loading configuration from: {config_path}")
    
    use_amp = config.training.use_amp
    # MPS backend can fail on float16 in backward (ScalarType::Float/Int/Bool expected).
    # Disable AMP when using MPS unless the user explicitly requested it and we support it later.
    if use_amp and (not is_distributed) and device.type == "mps":
        use_amp = False
        config.training.use_amp = False
        if rank == 0:
            print("Note: AMP disabled on MPS (Apple Silicon) to avoid backward pass dtype errors.")

    if rank == 0:
        print("=" * 80)
        print(f"{config.model_type.upper()} Training")
        print("=" * 80)
        if config.git_commit:
            dirty_suffix = " (dirty working tree)" if config.git_dirty else ""
            git_label = config.git_describe or config.git_commit[:12]
            print(f"Source git: {git_label}{dirty_suffix}")
            if config.git_branch:
                print(f"Source branch: {config.git_branch}")
            if config.git_commit_date:
                print(f"Source commit date: {config.git_commit_date}")
        if config.package_version:
            print(f"Package: oriented-det {config.package_version}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        print(f"MPS (Apple Silicon) Available: {mps_available}")
        if torch.cuda.is_available():
            if is_distributed:
                print(f"Distributed: {world_size} GPUs")
            else:
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        elif mps_available:
            print("Using MPS (Metal) for GPU acceleration")
        print(f"Mixed Precision (AMP): {use_amp}")
        print()
    
    dataset_format = dataset_format_name(config.dataset)

    if rank == 0 and dataset_format == "dota":
        print("Checking dataset directories...")
        check_directories(
            config.dataset.get_train_tile_roots(),
            config.dataset.get_val_tile_roots(),
        )
        print()
    
    # Create datasets
    print("Loading datasets...")
    train_filter_empty = getattr(config.dataset, "filter_empty_gt", False)
    val_filter_empty = False if dataset_format == "airbus_playground" else train_filter_empty
    train_dataset = build_split_dataset(
        config.dataset, "train", filter_empty_gt=train_filter_empty
    )
    val_dataset = build_split_dataset(
        config.dataset, "val", filter_empty_gt=val_filter_empty
    )

    if rank == 0 and dataset_format == "airbus_playground":
        map_labels = config.dataset.map_labels or {}
        ignore_labels = set(config.dataset.ignore_labels or [])
        raw_in_csv = train_dataset.get_raw_class_names_from_csv()
        ann_path = getattr(train_dataset, "annotations_path", None)
        print(
            f"\nUsing Airbus Playground CSV dataset: {config.dataset.annotations_file}, "
            f"{config.dataset.split_file} (val_split_id={config.dataset.val_split_id})"
        )
        if ann_path is not None:
            print(f"  annotations path: {ann_path}")
        print(
            "  (val_split_id is the split.csv fold id used as validation when the split column is numeric.)"
        )
        if getattr(config.dataset, "train_includes_val", False):
            print(
                "  train_includes_val: true — training uses all folds; "
                f"fold {config.dataset.val_split_id} is still used for validation/monitoring only."
            )
        print(f"  map_labels: {len(map_labels)} key(s). Raw labels in CSV (train): {len(raw_in_csv)} distinct — {raw_in_csv[:15]}{'...' if len(raw_in_csv) > 15 else ''}")
        if ignore_labels:
            print(f"  ignore_labels: {len(ignore_labels)} value(s) — entries matching these labels are dropped.")
        lookalike_extra = getattr(config.dataset, "lookalike_labels", None) or []
        print(
            "  lookalike: reserved token 'lookalike' is a hard-negative routing label "
            "(not a semantic class); map confusers via map_labels."
        )
        if lookalike_extra:
            print(f"  lookalike_labels aliases: {list(lookalike_extra)}")
        if len(raw_in_csv) <= 1:
            print(f"  ⚠ Only one raw label in CSV: check that training reads the regenerated file (config uses annotations_file: annotations.csv; not annotations.cvs).")
        if train_filter_empty:
            print("Airbus filter_empty_gt (train split only; val keeps all tiles):")
            print(format_airbus_empty_gt_filter_log(train_dataset, split="train"))
    elif rank == 0 and dataset_format == "hrsc2016":
        print(
            f"\nUsing HRSC2016 dataset: {config.dataset.data_root} "
            f"(train ImageSets={getattr(train_dataset, 'split', 'trainval')}, "
            f"val ImageSets={getattr(val_dataset, 'split', 'test')})"
        )
        if train_filter_empty:
            print("HRSC2016 filter_empty_gt:")
            print(format_hrsc_empty_gt_filter_log(train_dataset, split="train"))
            print(format_hrsc_empty_gt_filter_log(val_dataset, split="val"))
    elif rank == 0 and train_filter_empty:
        print("DOTA filter_empty_gt (MMRotate-style):")
        print(format_dota_empty_gt_filter_log(train_dataset, split="train"))
        print(format_dota_empty_gt_filter_log(val_dataset, split="val"))

    # Class list from full train split before max_* caps: a limited subset must not shrink num_classes.
    class_names = split_class_names(train_dataset, config.dataset)

    tile_csv = getattr(config.dataset, "tile_metrics_csv", None)
    csv_path = Path(tile_csv) if tile_csv else None
    drop_easy_empty = bool(getattr(config.dataset, "drop_easy_empty_tiles", False))
    if drop_easy_empty:
        if csv_path is None:
            raise ValueError("dataset.drop_easy_empty_tiles requires dataset.tile_metrics_csv")
        if not csv_path.is_file():
            raise FileNotFoundError(f"dataset.tile_metrics_csv not found: {csv_path}")
        n_before_drop = len(train_dataset)
        train_dataset, n_dropped_easy_empty = _drop_easy_empty_tiles(train_dataset, csv_path)
        if rank == 0:
            print(
                f"Drop easy-empty tiles: {csv_path.name}, dropped {n_dropped_easy_empty} / "
                f"{n_before_drop} vacuous tiles (tp=fp=fn=0); {len(train_dataset)} train tiles remain"
            )

    shuffle_seed = getattr(config.dataset, "max_samples_shuffle_seed", None)
    # Optionally limit to first N samples (e.g. for overfit sanity check)
    if getattr(config.dataset, "max_train_samples", None) is not None:
        idx_train = capped_subset_indices(
            len(train_dataset),
            int(config.dataset.max_train_samples),
            shuffle_seed=shuffle_seed,
        )
        train_dataset = Subset(train_dataset, idx_train)
        if rank == 0:
            _how = (
                f"deterministic shuffle (max_samples_shuffle_seed={shuffle_seed})"
                if shuffle_seed is not None
                else "first-N dataset order"
            )
            print(f"Limited training set to {len(idx_train)} sample(s) — {_how}")
    if getattr(config.dataset, "max_val_samples", None) is not None:
        idx_val = capped_subset_indices(
            len(val_dataset),
            int(config.dataset.max_val_samples),
            shuffle_seed=shuffle_seed,
        )
        val_dataset = Subset(val_dataset, idx_val)
        if rank == 0:
            _how = (
                f"deterministic shuffle (max_samples_shuffle_seed={shuffle_seed})"
                if shuffle_seed is not None
                else "first-N dataset order"
            )
            print(f"Limited validation set to {len(idx_val)} sample(s) — {_how}\n")

    # Report discovered classes (from full train before caps)
    if rank == 0:
        print(f"\nFound {len(class_names)} classes:")
        for i, cls_name in enumerate(class_names):
            print(f"  {i}: {cls_name}")
    
    # Create class mapping (class_name -> class_id)
    class_map = {cls_name: i + 1 for i, cls_name in enumerate(class_names)}
    num_foreground_classes = len(class_names)  # model and config: foreground only (cls_head has +1 for background)
    
    if rank == 0:
        print(f"\nNumber of classes (foreground): {num_foreground_classes}")
        print(f"Class mapping: {class_map}\n")
    
    weighted_train_sampler: Optional[WeightedRandomSampler] = None
    if csv_path is not None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"dataset.tile_metrics_csv not found: {csv_path}")
        metric_col = getattr(config.dataset, "hard_tile_metric_column", "f1")
        h_thr = float(getattr(config.dataset, "hard_tile_threshold", 0.8))
        factor = float(getattr(config.dataset, "hard_tile_oversample_factor", 2.0))
        stem_metrics = _load_tile_metrics_by_stem(csv_path, metric_col)
        vacuous_stems = _stems_vacuous_true_negatives_from_tile_csv(csv_path)
        hard_indices: List[int] = []
        for i in range(len(train_dataset)):
            stem = _dataset_index_to_image_stem(train_dataset, i)
            m = stem_metrics.get(stem)
            if m is not None and m < h_thr and stem not in vacuous_stems:
                hard_indices.append(i)
        extra = max(0, int(round(factor)) - 1)
        if is_distributed:
            order = list(range(len(train_dataset)))
            for hi in hard_indices:
                for _ in range(extra):
                    order.append(hi)
            train_dataset = _HardTileExpandedDataset(train_dataset, order)
            if rank == 0:
                print(
                    f"Hard-tile oversampling (DDP): {csv_path.name}, {len(hard_indices)} tiles with "
                    f"{metric_col} < {h_thr}; ~{factor}x via {extra} extra index(es) per hard tile"
                )
        else:
            w = torch.ones(len(train_dataset), dtype=torch.double)
            for hi in hard_indices:
                w[hi] = factor
            weighted_train_sampler = WeightedRandomSampler(
                w, num_samples=len(train_dataset), replacement=True
            )
            if rank == 0:
                print(
                    f"Hard-tile oversampling: {csv_path.name}, {len(hard_indices)} tiles with "
                    f"{metric_col} < {h_thr}; WeightedRandomSampler weight={factor} for those indices"
                )
    
    # Analyze class distribution
    class_counts, class_weights_inv_freq, class_weights_sqrt, sorted_classes = analyze_class_distribution(train_dataset)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Class Distribution Analysis - Training Set")
        print(f"{'='*80}")
        print(f"Total objects: {sum(class_counts.values()):,}")
        print(f"Total classes: {len(class_counts)}")
        if class_counts and sorted_classes:
            print(f"\n{'Class Name':<20} {'Count':<12} {'Percentage':<12}")
            print("-" * 80)
            total_objects = sum(class_counts.values())
            for class_name, count in sorted_classes:
                percentage = (count / total_objects) * 100
                print(f"{class_name:<20} {count:>10,}  {percentage:>10.2f}%")
            print("-" * 80)
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}x")
            print(f"Most frequent class: {sorted_classes[0][0]} ({sorted_classes[0][1]:,} instances)")
            print(f"Least frequent class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]:,} instances)")
        else:
            print("(No objects in subset; e.g. overfit on one image with no matching annotations.)")
    
    # Configure class imbalance handling
    loss_config = config.loss
    if loss_config.loss_type in ["class_weighted", "focal_weighted"] and class_counts:
        # Compute target (end) weights from counts.
        method = str(getattr(loss_config, "class_weight_method", "sqrt"))
        overrides = getattr(loss_config, "class_weight_overrides", None)
        if method == "effective_num":
            beta = float(getattr(loss_config, "class_weight_beta", 0.9999))
            # Inline effective_num so we can use config beta.
            weights_dict = {k: (1.0 - beta) / max(1e-12, (1.0 - (beta ** float(v))))
                            for k, v in class_counts.items()}
            # Normalize mean=1, clip, overrides (mirror compute_class_weights)
            mean_weight = float(np.mean(list(weights_dict.values())))
            roi_class_weights = {k: float(v / mean_weight) for k, v in weights_dict.items()}
            MAX_WEIGHT = 3.0
            MIN_WEIGHT = 0.3
            roi_class_weights = {k: float(np.clip(v, MIN_WEIGHT, MAX_WEIGHT)) for k, v in roi_class_weights.items()}
            computed_class_weights = {k: float(v) for k, v in roi_class_weights.items()}
            if overrides:
                for class_name, weight in overrides.items():
                    if class_name in roi_class_weights:
                        roi_class_weights[class_name] = float(weight)
        else:
            roi_class_weights, computed_class_weights = compute_class_weights(
                class_counts,
                None,
                method,
                class_weight_overrides=overrides,
            )
        if getattr(loss_config, "background_weight", None) is not None:
            roi_class_weights["background"] = float(loss_config.background_weight)
        if rank == 0:
            _print_class_weight_table(
                roi_class_weights,
                computed_class_weights,
                class_counts,
                method=loss_config.class_weight_method,
                class_weight_overrides=overrides,
            )
    else:
        roi_class_weights = None
        if rank == 0:
            print(f"\nClass weighting disabled (loss_type={loss_config.loss_type})")
    
    if rank == 0:
        print(f"\nLoss configuration: {loss_config.loss_type}")
        if loss_config.loss_type in ["focal", "focal_weighted"]:
            print(f"  Focal Loss Alpha: {loss_config.focal_alpha}")
            print(f"  Focal Loss Gamma: {loss_config.focal_gamma}")
        print()

    if args.wizard:
        if rank == 0:
            print("Wizard mode enabled: gathering dataset statistics...")
            wizard_stats = gather_wizard_stats(train_dataset)
            print_wizard_recommendations(
                wizard_stats,
                config,
                world_size=world_size if is_distributed else 1,
            )
            print("Wizard mode complete: exiting without creating a run directory or starting training.")
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return
    
    # Set runtime data (num_classes = foreground only, so checkpoint and inference stay consistent)
    config.class_map = class_map
    config.class_names = class_names
    config.num_classes = num_foreground_classes
    
    # Checkpoint source: fully config-driven (paths, discover_previous_run, resume_from_checkpoint_epoch).
    from oriented_det.train.utils import get_project_root

    project_root = get_project_root()
    runs_dir = project_root / "runs" / config.model_type
    current_experiment_dir = runs_dir / config.experiment_timestamp
    experiment_dirs: List[Path] = []
    if runs_dir.exists():
        experiment_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d != current_experiment_dir],
            reverse=True,
        )

    ckpt_raw = getattr(config.checkpoint, "load_from_checkpoint", None)
    has_explicit_ckpt_path = bool(ckpt_raw is not None and str(ckpt_raw).strip())
    ckpt_exists = False
    if has_explicit_ckpt_path:
        from oriented_det.pretrained import ensure_checkpoint

        resolved_ckpt = ensure_checkpoint(ckpt_raw, quiet=(rank != 0))
        config.checkpoint.load_from_checkpoint = resolved_ckpt
        ckpt_exists = resolved_ckpt.exists()
        if is_distributed and dist.is_initialized():
            dist.barrier()

    if config.checkpoint.load_from_experiment is None:
        discover = bool(getattr(config.checkpoint, "discover_previous_run", False))
        if discover and not has_explicit_ckpt_path:
            if experiment_dirs:
                config.checkpoint.load_from_experiment = str(experiment_dirs[0])
                if rank == 0:
                    print(
                        f"discover_previous_run: using newest run {experiment_dirs[0].name}"
                    )
                    if config.checkpoint.resume_from_checkpoint_epoch:
                        print(
                            "  Loading latest checkpoint_epoch_*.pth (resume_from_checkpoint_epoch=true)"
                        )
                    else:
                        print(
                            "  Loading best checkpoint; training starts at epoch 0 unless resumed "
                            "(resume_from_checkpoint_epoch=false)"
                        )
            elif rank == 0:
                print(
                    f"Warning: discover_previous_run is true but no other run directory found under {runs_dir}."
                )
        elif discover and has_explicit_ckpt_path and not ckpt_exists and rank == 0:
            print(
                "Warning: discover_previous_run is true and load_from_checkpoint points to a missing file; "
                "not auto-setting load_from_experiment. Fix the checkpoint path or omit it to discover a run."
            )
    
    if rank == 0:
        print(f"\nTraining configuration created:")
        print(f"  Model type: {config.model_type}")
        print(f"  Experiment timestamp: {config.experiment_timestamp}")
        print(f"  Number of classes: {config.num_classes}")
        print(f"  Batch size: {config.data_loader.batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Epochs: {config.training.num_epochs}")
    
    # Create collate functions
    if rank == 0:
        print(f"\nCreating collate functions...")
    if config.enable_albumentation:
        train_augmentation = create_train_augmentation(
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
    # Preprocessing from config (resize + normalization); inference will use the same.
    prep = getattr(config, "preprocessing", None)
    if prep is not None:
        from oriented_det.data.preprocessing import parse_canvas_size

        resize_mode = getattr(prep, "resize_mode", "fixed")
        ts = getattr(prep, "target_size", [1024, 1024])
        resize_to = parse_canvas_size(resize_mode, ts)
        norm_mean = getattr(prep, "normalize_mean", None)
        norm_std = getattr(prep, "normalize_std", None)
        pad_div = getattr(prep, "pad_size_divisor", 32)
    else:
        resize_mode = "fixed"
        resize_to = (1024, 1024)
        norm_mean = norm_std = None
        pad_div = 32

    flip_h = getattr(prep, "enable_flip_horizontal", True) if prep is not None else True
    flip_v = getattr(prep, "enable_flip_vertical", True) if prep is not None else True
    flip_d = getattr(prep, "enable_flip_diagonal", False) if prep is not None else False
    rotate_on = getattr(prep, "enable_random_rotate", False) if prep is not None else False
    rotate_prob = getattr(prep, "random_rotate_prob", 0.5) if prep is not None else 0.5
    rotate_range = getattr(prep, "random_rotate_angle_range", 180.0) if prep is not None else 180.0

    if config.enable_albumentation:
        train_collate_fn = create_collate_fn(
            config.class_map,
            augmentation=train_augmentation,
            normalize=True,
            resize_mode=resize_mode,
            resize_to=resize_to,
            pad_size_divisor=pad_div,
            enable_flip_horizontal=flip_h,
            enable_flip_vertical=flip_v,
            enable_flip_diagonal=flip_d,
            enable_random_rotate=rotate_on,
            random_rotate_prob=rotate_prob,
            random_rotate_angle_range=rotate_range,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
            difficult_strategy=getattr(config.dataset, "difficult_strategy", "drop"),
            lookalike_labels=getattr(config.dataset, "lookalike_labels", None),
        )
        if rank == 0:
            print("  - Training: with Albumentations augmentation")
    else:
        train_collate_fn = create_collate_fn(
            config.class_map,
            augmentation=None,
            normalize=True,
            resize_mode=resize_mode,
            resize_to=resize_to,
            pad_size_divisor=pad_div,
            enable_flip_horizontal=flip_h,
            enable_flip_vertical=flip_v,
            enable_flip_diagonal=flip_d,
            enable_random_rotate=rotate_on,
            random_rotate_prob=rotate_prob,
            random_rotate_angle_range=rotate_range,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
            difficult_strategy=getattr(config.dataset, "difficult_strategy", "drop"),
            lookalike_labels=getattr(config.dataset, "lookalike_labels", None),
        )
        if rank == 0:
            flip_parts = []
            if flip_h:
                flip_parts.append("horizontal")
            if flip_v:
                flip_parts.append("vertical")
            if flip_d:
                flip_parts.append("diagonal")
            flip_msg = ", ".join(flip_parts) if flip_parts else "none"
            rotate_msg = (
                f"rotate p={rotate_prob:g} ±{rotate_range:g}°" if rotate_on else "rotate off"
            )
            print(
                f"  - Training: no Albumentations augmentation (flips: {flip_msg}; {rotate_msg})"
            )

    val_collate_fn = create_collate_fn(
        config.class_map,
        augmentation=None,
        normalize=True,
        resize_mode=resize_mode,
        resize_to=resize_to,
        pad_size_divisor=pad_div,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=False,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
        difficult_strategy=getattr(config.dataset, "difficult_strategy", "drop"),
        lookalike_labels=getattr(config.dataset, "lookalike_labels", None),
        random_crop=False,
    )
    if rank == 0:
        print("  - Validation: no augmentation")
    
    # Create data loaders (DistributedSampler when multi-GPU; num_workers=0 for DDP)
    if rank == 0:
        print(f"\nCreating data loaders...")
    num_workers = 0 if is_distributed else config.data_loader.num_workers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=config.data_loader.shuffle
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_shuffle = False
    elif weighted_train_sampler is not None:
        train_sampler = weighted_train_sampler
        val_sampler = None
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = config.data_loader.shuffle
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data_loader.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=config.data_loader.pin_memory if (torch.cuda.is_available()) else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data_loader.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=config.data_loader.pin_memory if (torch.cuda.is_available()) else False,
    )
    
    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batches per epoch: {len(train_loader)}")
        if is_distributed:
            print(f"Effective batch size: {config.data_loader.batch_size * world_size * config.training.gradient_accumulation_steps}")
    
    # Create model
    if rank == 0:
        print(f"\nCreating model...")
        print(f"Using device: {device}")
    
    model, roi_loss_type = create_model_from_config(
        config,
        num_foreground_classes,
        device,
        roi_class_weights=roi_class_weights,
    )

    # Optional schedule: ramp from uniform weights -> computed weights over epochs.
    class_weight_schedule = None
    sched_type = (getattr(loss_config, "class_weight_schedule_type", None) or "").strip().lower()
    if roi_class_weights is not None and sched_type in ("linear_ramp", "ramp"):
        start_epoch = int(getattr(loss_config, "class_weight_schedule_start_epoch", 0) or 0)
        end_epoch = int(getattr(loss_config, "class_weight_schedule_end_epoch", 0) or 0)
        power = float(getattr(loss_config, "class_weight_schedule_power", 1.0) or 1.0)
        if end_epoch > start_epoch:
            w_end = _weights_dict_to_tensor(
                roi_class_weights, class_map, num_foreground_classes, device=device
            )
            # Uniform start (but keep explicit background weight if provided)
            start_dict = {"background": float(roi_class_weights.get("background", 1.0))}
            w_start = _weights_dict_to_tensor(
                start_dict, class_map, num_foreground_classes, device=device
            )

            def class_weight_schedule(epoch: int) -> torch.Tensor:
                t = (float(epoch) - float(start_epoch)) / float(end_epoch - start_epoch)
                a = _ramp(t, power)
                return (1.0 - a) * w_start + a * w_end
            if rank == 0:
                print(f"\nClass weight schedule enabled: {sched_type} (epochs {start_epoch}→{end_epoch}, power={power})")
        elif rank == 0:
            print(f"\nNote: class_weight_schedule_type set but end_epoch <= start_epoch; schedule disabled.")
    
    # Configure class weights on the model (only for models that support it, e.g. OrientedRCNN, RotatedFasterRCNN; RetinaNet uses focal loss and does not)
    if roi_class_weights is not None and isinstance(roi_class_weights, dict) and hasattr(model, "set_class_weights"):
        model.set_class_weights(config.class_map, device=device)
        if rank == 0:
            if getattr(model, "roi_class_weights", None) is not None:
                print(f"✓ Class weights configured for {len(roi_class_weights)} classes")
            else:
                print("⚠ Class weights requested but not active on this model instance")

    from oriented_det.train.grouped_ce import configure_roi_grouped_ce

    _model_for_grouped_ce = model
    grouped_ce_active = configure_roi_grouped_ce(
        _model_for_grouped_ce,
        loss_config,
        config.class_map,
        num_foreground_classes=num_foreground_classes,
        device=device,
    )
    if rank == 0 and grouped_ce_active:
        sched = getattr(loss_config, "roi_grouped_ce_schedule_type", None) or "step"
        g_end = int(getattr(loss_config, "roi_grouped_ce_schedule_end_epoch", 0) or 0)
        g_start = int(getattr(loss_config, "roi_grouped_ce_schedule_start_epoch", 0) or 0)
        groups = getattr(loss_config, "roi_grouped_ce_groups", None) or {}
        print(
            f"\nROI grouped CE curriculum enabled: schedule={sched!r}, "
            f"epochs {g_start}→{g_end}, {len(groups)} group(s)"
        )
    
    # Wrap in DDP for multi-GPU
    # Partial freeze (freeze_backbone_epochs / freeze_rpn_epochs) toggles requires_grad on subsets;
    # those parameters skip the loss graph → DDP must traverse unused params or reduction breaks.
    _fb_pre_ddp = int(getattr(config.training, "freeze_backbone_epochs", 0) or 0)
    _fr_pre_ddp = int(getattr(config.training, "freeze_rpn_epochs", 0) or 0)
    _ddp_find_unused = (_fb_pre_ddp > 0 or _fr_pre_ddp > 0)
    if is_distributed:
        if _ddp_find_unused and rank == 0:
            print(
                "  DDP: find_unused_parameters=True (required while backbone/RPN are frozen by epoch)."
            )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            output_device=None,
            find_unused_parameters=_ddp_find_unused,
            broadcast_buffers=False,
        )
        model_for_weights = model.module
    else:
        model_for_weights = model
    
    if rank == 0:
        print(f"\nComplete {config.model_type.upper()} model created:")
        print(f"  Backbone: {config.model.backbone}")
        print(f"  Number of classes: {config.num_classes} (foreground)")
        print(f"  Pretrained backbone: {config.model.pretrained_backbone}")
        _fs = getattr(config.model, "frozen_stages", None)
        _tl = (5 if _fs == 0 else max(1, 4 - _fs)) if _fs is not None else config.model.trainable_layers
        print(f"  Trainable backbone layers: {_tl} ({'all layers' if _tl == 5 else f'last {_tl} stages'})" + (f" (frozen_stages={_fs})" if _fs is not None else ""))
        _aa = getattr(model_for_weights, "anchor_angles", None)
        if _aa:
            print(
                "  RPN anchor reference angles (fixed horizontal priors, not configurable): "
                f"{[f'{a*180/math.pi:.1f}°' for a in _aa]}"
            )
        print(f"  Anchor scales: {config.model.anchor_scales}")
        print(f"  Anchor ratios: {config.model.anchor_ratios}")
        _fpn_returned = getattr(config.model, "fpn_returned_layers", None)
        _fpn_strides = getattr(config.model, "fpn_strides", None)
        if _fpn_strides is None:
            _fpn_strides = [8, 16, 32, 64] if _fpn_returned == [2, 3, 4] else [4, 8, 16, 32, 64]
        _num_levels = len(_fpn_strides)
        _level_names = [f"P{i + 2}" for i in range(_num_levels)] if _fpn_returned != [2, 3, 4] else [f"P{i + 3}" for i in range(_num_levels)]
        print(f"  FPN strides: {_fpn_strides}")
        print(f"  FPN levels: {_num_levels} ({', '.join(_level_names)})")
        print(f"  ROI Loss Type: {roi_loss_type}")
        if is_distributed:
            print(f"  Distributed: {world_size} GPUs")
        
        total_params = sum(p.numel() for p in model_for_weights.parameters())
        trainable_params = sum(p.numel() for p in model_for_weights.parameters() if p.requires_grad)
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
    if rank == 0:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_DIR = EXPERIMENT_DIR
    if rank == 0 and SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is not installed. Install with `pip install tensorboard` "
            "or `pip install oriented-det` (PyPI package includes it as a dependency)."
        )
    writer = SummaryWriter(log_dir=str(LOG_DIR)) if rank == 0 else None
    
    if rank == 0:
        config.save(EXPERIMENT_DIR / "config.json")
        print(f"\nConfiguration saved to: {EXPERIMENT_DIR / 'config.json'}")
        # Log config to TensorBoard (Text tab) for easy viewing
        if writer is not None:
            config_str = (EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8")
            writer.add_text("config", f"```json\n{config_str}\n```", 0)
    
    # Apply learning rate scaling for gradient accumulation and DDP
    original_lr = config.training.learning_rate
    scaling_factors = []
    if config.training.gradient_accumulation_steps > 1 and config.training.lr_scaling_with_accumulation != "none":
        if config.training.lr_scaling_with_accumulation == "linear":
            scaling_factors.append(config.training.gradient_accumulation_steps)
        elif config.training.lr_scaling_with_accumulation == "sqrt":
            scaling_factors.append(math.sqrt(config.training.gradient_accumulation_steps))
    if is_distributed and bool(getattr(config.training, "lr_scale_with_world_size", False)):
        scaling_factors.append(world_size)
    if scaling_factors:
        total_scaling = math.prod(scaling_factors)
        config.training.learning_rate = original_lr * total_scaling
        if rank == 0:
            print(f"\n  📈 Learning Rate Scaling:")
            print(f"     Base LR: {original_lr:.6f}")
            if is_distributed and bool(getattr(config.training, "lr_scale_with_world_size", False)):
                print(f"     DDP scaling: × {world_size} (world_size)")
            if config.training.gradient_accumulation_steps > 1 and config.training.lr_scaling_with_accumulation != "none":
                print(f"     Gradient accumulation: × {scaling_factors[0]:.3f}")
            print(f"     Total scaling: × {total_scaling:.3f}")
            print(f"     Scaled LR: {config.training.learning_rate:.6f}")
            eff_bs = config.data_loader.batch_size * (world_size if is_distributed else 1) * config.training.gradient_accumulation_steps
            print(f"     Effective batch size: {eff_bs}")
    
    # Create optimizer and scheduler
    use_lr_param_groups = bool(getattr(config.training, "use_lr_param_groups", False))
    freeze_backbone_epochs = int(getattr(config.training, "freeze_backbone_epochs", 0) or 0)
    freeze_rpn_epochs = int(getattr(config.training, "freeze_rpn_epochs", 0) or 0)
    use_phase_freeze = freeze_backbone_epochs > 0 or freeze_rpn_epochs > 0
    if use_phase_freeze and rank == 0:
        print(
            f"\nPartial freeze (0-based epoch indices; ROI always trains): "
            f"freeze_backbone_epochs={freeze_backbone_epochs}, freeze_rpn_epochs={freeze_rpn_epochs}"
        )
    if use_lr_param_groups:
        optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(
            model,
            base_lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            config=config,
            include_frozen_parameters=use_phase_freeze,
        )
        optimizer = optim.SGD(
            optimizer_param_groups,
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
        if rank == 0:
            print("\nOptimizer param groups enabled:")
            for group_name in ["backbone", "rpn", "roi", "head", "other"]:
                if group_name not in optimizer_group_summary:
                    continue
                details = optimizer_group_summary[group_name]
                print(
                    f"  - {group_name:8s} "
                    f"lr={details['lr']:.6f} "
                    f"(x{details['multiplier']:.2f}), "
                    f"params={int(details['num_params']):,}"
                )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
    
    # Create LR scheduler (type from config: multistep/step, reduce_on_plateau, one_cycle, cosine_annealing)
    sched_type = (getattr(config.training, "lr_scheduler_type", None) or "").strip().lower()
    lr_milestones = getattr(config.training, "lr_scheduler_milestones", None)
    plateau_metric = getattr(config.training, "lr_scheduler_plateau_metric", "total_loss")

    if sched_type == "reduce_on_plateau":
        mode = "max" if plateau_metric.strip().lower() == "map" else "min"
        factor = getattr(config.training, "lr_scheduler_plateau_factor", 0.1)
        patience = getattr(config.training, "lr_scheduler_plateau_patience", 5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
        )
        if rank == 0:
            print(f"\nLR scheduler: ReduceLROnPlateau (metric={plateau_metric}, mode={mode}, factor={factor}, patience={patience})")
    elif sched_type in ("one_cycle", "onecycle"):
        steps_per_epoch = max(1, len(train_loader) // config.training.gradient_accumulation_steps)
        total_steps = config.training.num_epochs * steps_per_epoch
        pct_start = getattr(config.training, "lr_scheduler_one_cycle_pct_start", 0.3)
        div_factor = getattr(config.training, "lr_scheduler_one_cycle_div_factor", 25.0)
        final_div_factor = getattr(config.training, "lr_scheduler_one_cycle_final_div_factor", 1e4)
        one_cycle = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        lr_scheduler = OneCycleWrapper(one_cycle)
        if rank == 0:
            print(
                f"\nLR scheduler: OneCycleLR (total_steps={total_steps}, pct_start={pct_start}, "
                f"div_factor={div_factor}, final_div_factor={final_div_factor})"
            )
    elif sched_type in ("cosine_annealing", "cosine"):
        base_lr_scheduler, t_max, eta_min = create_pytorch_cosine_lr_scheduler(
            optimizer, config.training,
        )
        sched_desc = format_pytorch_cosine_scheduler_description(
            t_max,
            eta_min,
            config.training.lr_warmup_steps,
            config.training.num_epochs,
        )
        if config.training.lr_warmup_steps > 0:
            lr_scheduler = WarmupScheduler(
                optimizer, base_lr_scheduler, config.training.lr_warmup_steps,
            )
            if rank == 0:
                print(f"\nLR scheduler: {sched_desc}")
        else:
            lr_scheduler = base_lr_scheduler
            if rank == 0:
                print(f"\nLR scheduler: {sched_desc}")
    elif sched_type in ("cosine_annealing_with_tail", "cosine_with_tail"):
        eta_min = float(getattr(config.training, "lr_scheduler_cosine_eta_min", 1e-6))
        base_lr_scheduler, cosine_ep, tail_ep, tail_lr = create_cosine_with_tail_lr_scheduler(
            optimizer, config.training,
        )
        sched_desc = format_cosine_with_tail_scheduler_description(
            cosine_ep, tail_ep, eta_min, tail_lr, config.training.lr_warmup_steps,
        )
        if config.training.lr_warmup_steps > 0:
            lr_scheduler = WarmupScheduler(
                optimizer, base_lr_scheduler, config.training.lr_warmup_steps,
            )
            if rank == 0:
                print(f"\nLR scheduler: {sched_desc}")
        else:
            lr_scheduler = base_lr_scheduler
            if rank == 0:
                print(f"\nLR scheduler: {sched_desc}")
    else:
        # Default: MultiStepLR or StepLR
        if isinstance(lr_milestones, list) and len(lr_milestones) > 0:
            base_lr_scheduler, milestone_gammas = create_multistep_lr_scheduler(
                optimizer, config.training
            )
        else:
            step_gamma = config.training.lr_scheduler_gamma
            if isinstance(step_gamma, (list, tuple)):
                raise ValueError(
                    "training.lr_scheduler_gamma as a list requires "
                    "training.lr_scheduler_milestones (one factor per milestone)."
                )
            base_lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.lr_scheduler_step_epochs,
                gamma=float(step_gamma),
            )
        if config.training.lr_warmup_steps > 0:
            lr_scheduler = WarmupScheduler(
                optimizer, base_lr_scheduler, config.training.lr_warmup_steps,
            )
            if rank == 0:
                print(f"\nLearning rate warmup enabled: {config.training.lr_warmup_steps} optimizer steps")
                if use_lr_param_groups:
                    warmup_targets = ", ".join(f"{lr:.6f}" for lr in lr_scheduler.base_lrs)
                    print(f"  Warmup target LRs: [{warmup_targets}] over {config.training.lr_warmup_steps} steps")
                else:
                    print(f"  Warmup: 0 → {config.training.learning_rate} over {config.training.lr_warmup_steps} steps")
                if isinstance(lr_milestones, list) and len(lr_milestones) > 0:
                    if milestone_gammas is not None:
                        print(
                            "  After warmup: MultiStepLR at epochs "
                            f"{[int(m) for m in lr_milestones]}, gammas={milestone_gammas}"
                        )
                    else:
                        print(
                            "  After warmup: MultiStepLR at epochs "
                            f"{[int(m) for m in lr_milestones]}, gamma={config.training.lr_scheduler_gamma}"
                        )
                else:
                    print(
                        f"  After warmup: StepLR every {config.training.lr_scheduler_step_epochs} epochs, "
                        f"gamma={config.training.lr_scheduler_gamma}"
                    )
        else:
            lr_scheduler = base_lr_scheduler
            if rank == 0:
                print("\nLearning rate warmup disabled")
    
    # Create checkpoint manager (best checkpoint: config checkpoint.best_metric, e.g. "mAP" or "total_loss")
    best_metric = getattr(config.checkpoint, "best_metric", "total_loss") or "total_loss"
    higher_is_better_cfg = getattr(config.checkpoint, "higher_is_better", None)
    if higher_is_better_cfg is not None:
        higher_is_better = bool(higher_is_better_cfg)
    else:
        higher_is_better = str(best_metric).strip().lower() == "map"
    checkpoint_manager = CheckpointManager(
        CHECKPOINT_DIR,
        best_metric=best_metric,
        higher_is_better=higher_is_better,
        keep_last_n=3,
    )
    eval_thr_sc, eval_thr_pc, eval_thr_iou = effective_eval_metric_thresholds(config)
    eval_use_exact_rotated_iou = config_use_exact_rotated_iou_for_map(config)
    eval_use_exact_rotated_iou_for_final_map = config_use_exact_rotated_iou_for_final_map(config)
    
    if rank == 0:
        print(f"\nExperiment directory: {EXPERIMENT_DIR}")
        print(f"TensorBoard logging enabled. Logs saved to: {LOG_DIR}")
        print(f"View logs with: tensorboard --logdir {LOG_DIR.parent}")
        print(f"View this experiment with: tensorboard --logdir {LOG_DIR}")
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {config.training.num_epochs}")
        print(f"  Learning rate: {config.training.learning_rate:.6f}")
        if use_lr_param_groups:
            lh = getattr(config.training, "lr_mult_head", None)
            extra = f", lr_mult_head={lh}" if lh is not None else ""
            print(
                "  LR multipliers (backbone/rpn/roi/other"
                f"{extra}): "
                f"{config.training.lr_mult_backbone}/"
                f"{config.training.lr_mult_rpn}/"
                f"{config.training.lr_mult_roi}/"
                f"{config.training.lr_mult_other}"
            )
        print(f"  Batch size per GPU: {config.data_loader.batch_size}")
        print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
        print(f"  Best checkpoint metric: {best_metric} (higher_is_better={higher_is_better})")
        eff_bs = config.data_loader.batch_size * (world_size if is_distributed else 1) * config.training.gradient_accumulation_steps
        print(f"  Effective batch size: {eff_bs}")
        print(f"  Mixed precision: {config.training.use_amp}")
        _eval_thr_extra = (
            f", per_class_score_threshold={eval_thr_pc}" if eval_thr_pc else ""
        )
        print(
            f"  Eval (mAP / val matching): score_threshold={eval_thr_sc}, "
            f"iou_threshold={eval_thr_iou}{_eval_thr_extra}"
            " (production.score_threshold overrides evaluation when set)"
        )
        print(
            f"  Eval IoU backend (mAP / GT cover): "
            f"{'exact CPU polygon' if eval_use_exact_rotated_iou else 'GPU sampling (approx)'}"
        )
        print(
            f"  Eval IoU backend (final mAP): "
            f"{'exact CPU polygon' if eval_use_exact_rotated_iou_for_final_map else 'GPU sampling (approx)'}"
        )
        
        # Debug: config summary
        if getattr(args, "debug", False):
            print(f"\n  [debug] Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
            print(
                f"\n  [debug] Eval: score_threshold={eval_thr_sc}, iou_threshold={eval_thr_iou} "
                f"(effective; production overrides evaluation when set)"
            )
            print(f"  [debug] Classes: {config.class_names}")
            map_every = getattr(config.evaluation, "compute_map_every_n_epochs", 0)
            print(f"  [debug] mAP computed every N epochs: {map_every if map_every else 'final only'}")
    
    # Load checkpoint if specified
    checkpoint_loaded = False
    checkpoint_epoch = None
    start_epoch = config.checkpoint.start_epoch
    checkpoint_path = None

    # Direct checkpoint path takes precedence over load_from_experiment
    if getattr(config.checkpoint, "load_from_checkpoint", None) is not None:
        from oriented_det.pretrained import ensure_checkpoint

        direct_path = ensure_checkpoint(
            config.checkpoint.load_from_checkpoint, quiet=(rank != 0)
        )
        if direct_path.exists():
            checkpoint_path = direct_path
            if rank == 0:
                print(f"\nLoading checkpoint from config path: {checkpoint_path}")
        elif rank == 0:
            print(f"\nWarning: load_from_checkpoint set but file not found: {direct_path}")

    if checkpoint_path is None and config.checkpoint.load_from_experiment is not None:
        experiment_dir = Path(config.checkpoint.load_from_experiment)
        checkpoint_dir = experiment_dir / "checkpoints"
        
        # Latest checkpoint when resume_from_checkpoint_epoch; else prefer best-metric checkpoint
        latest_checkpoint = None
        if checkpoint_dir.exists():
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"), reverse=True)
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[0]
        
        # Fine-tune from experiment dir (resume_from_checkpoint_epoch=false): prefer best-metric checkpoint if it exists
        # resume_from_checkpoint_epoch: use latest checkpoint so training continues from last epoch
        if config.checkpoint.resume_from_checkpoint_epoch:
            checkpoint_path = latest_checkpoint if (latest_checkpoint and latest_checkpoint.exists()) else None
            if checkpoint_path and rank == 0:
                print(f"\nLoading latest checkpoint from: {checkpoint_path}")
        else:
            checkpoint_path = get_best_checkpoint_path(checkpoint_dir)
            if checkpoint_path is not None:
                if rank == 0:
                    print(f"\nLoading best checkpoint from: {checkpoint_path}")
            elif latest_checkpoint and latest_checkpoint.exists():
                checkpoint_path = latest_checkpoint
                if rank == 0:
                    print(f"\nLoading latest checkpoint from: {checkpoint_path} (no best_*.pth found)")
            else:
                checkpoint_path = None
        if checkpoint_path is None and rank == 0:
            print(f"\nWarning: No checkpoint found in {checkpoint_dir}, starting from scratch")
    
    if checkpoint_path and checkpoint_path.exists():
        include_prefixes = getattr(config.checkpoint, "load_include_prefixes", None)
        exclude_prefixes = getattr(config.checkpoint, "load_exclude_prefixes", None)
        selective_restore = bool(include_prefixes or exclude_prefixes)
        load_optimizer_state = bool(config.checkpoint.load_optimizer_state)
        load_scheduler_state = False
        forced_optimizer_load = False  # True when we force load due to resume (config had false)
        # When resuming from checkpoint epoch, always load optimizer state for proper continuation
        if config.checkpoint.resume_from_checkpoint_epoch:
            forced_optimizer_load = not load_optimizer_state
            load_optimizer_state = True
            load_scheduler_state = bool(getattr(config.checkpoint, "load_scheduler_state", True))
        if selective_restore and load_optimizer_state:
            load_optimizer_state = False
            load_scheduler_state = False
            if rank == 0:
                print(
                    "  Selective model restore requested; disabling optimizer state load "
                    "to avoid param-group mismatch."
                )
        opt_to_load = optimizer if load_optimizer_state else None
        sched_to_load = lr_scheduler if load_scheduler_state else None
        model_to_load = model_for_weights if is_distributed else model
        checkpoint = checkpoint_manager.load(
            checkpoint_path,
            model_to_load,
            opt_to_load,
            sched_to_load,
            strict=False,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
        )
        checkpoint_epoch = checkpoint.get("epoch", None)
        checkpoint_loaded = True

        if rank == 0:
            if not load_optimizer_state:
                print("  Note: Optimizer state not loaded (only model weights)")
            elif forced_optimizer_load:
                print("  Resume: loading optimizer state (overrides config)")
            if load_scheduler_state:
                print("  Resume: loading scheduler state")
            else:
                print("  Resume: scheduler state not loaded (optional cosine rebuild for continuation)")
            if checkpoint_epoch is not None:
                print(f"Checkpoint was saved at epoch {checkpoint_epoch}")
            if "metrics" in checkpoint:
                print(f"Checkpoint metrics: {checkpoint['metrics']}")
            resume_from_ckpt_epoch = bool(config.checkpoint.resume_from_checkpoint_epoch)
            if selective_restore and resume_from_ckpt_epoch:
                resume_from_ckpt_epoch = False
                print(
                    "  Selective restore is active; keeping start_epoch from config "
                    f"({start_epoch}) instead of checkpoint epoch."
                )
            if resume_from_ckpt_epoch:
                start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
                print(f"Resuming from epoch {start_epoch} (checkpoint epoch + 1)")
            else:
                print(f"Checkpoint loaded, but starting from epoch {start_epoch} (manual override)")
        else:
            resume_from_ckpt_epoch = bool(config.checkpoint.resume_from_checkpoint_epoch)
            if selective_restore and resume_from_ckpt_epoch:
                resume_from_ckpt_epoch = False
            if resume_from_ckpt_epoch:
                start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

        # Continuation training: original cosine may have T_max=num_epochs from the first run; reloading
        # that state leaves almost no LR schedule left. Rebuild cosine with a longer T_max and
        # last_epoch from the checkpoint (requires lr_warmup_steps=0 so the LR schedule is epoch-based).
        if (
            bool(config.checkpoint.resume_from_checkpoint_epoch)
            and not selective_restore
            and not load_scheduler_state
            and load_optimizer_state
            and checkpoint_epoch is not None
            and sched_type in (
                "cosine_annealing",
                "cosine",
                "cosine_annealing_with_tail",
                "cosine_with_tail",
            )
            and int(getattr(config.training, "lr_warmup_steps", 0) or 0) == 0
        ):
            eta_min_cont = float(getattr(config.training, "lr_scheduler_cosine_eta_min", 1e-6))
            if sched_type in ("cosine_annealing_with_tail", "cosine_with_tail"):
                lr_scheduler, cosine_ep, tail_ep, tail_lr = create_cosine_with_tail_lr_scheduler(
                    optimizer,
                    config.training,
                    last_epoch=int(checkpoint_epoch),
                )
                sched_desc = format_cosine_with_tail_scheduler_description(
                    cosine_ep, tail_ep, eta_min_cont, tail_lr, 0,
                )
            else:
                lr_scheduler, t_max, eta_min_cont = create_pytorch_cosine_lr_scheduler(
                    optimizer,
                    config.training,
                    last_epoch=int(checkpoint_epoch),
                )
                sched_desc = format_pytorch_cosine_scheduler_description(
                    t_max, eta_min_cont, 0, config.training.num_epochs,
                )
            if rank == 0:
                print(
                    f"\nRebuilt cosine schedule for continuation (scheduler state not loaded): "
                    f"{sched_desc}, last_epoch={int(checkpoint_epoch)}"
                )

    if not checkpoint_loaded:
        start_epoch = 0
        if rank == 0:
            print("\nStarting training from scratch (no checkpoint loaded)")
    
    if rank == 0:
        print(f"\nStarting epoch: {start_epoch}")
        if checkpoint_loaded:
            print(f"Checkpoint loaded: Model weights from epoch {checkpoint_epoch}")

    if use_phase_freeze:
        set_backbone_requires_grad(
            model_for_weights,
            freeze=start_epoch < freeze_backbone_epochs,
        )
        set_rpn_requires_grad(
            model_for_weights,
            freeze=start_epoch < freeze_rpn_epochs,
        )
        if rank == 0:
            print(
                f"  Backbone: {'FROZEN' if start_epoch < freeze_backbone_epochs else 'trainable'} "
                f"(freeze_backbone_epochs={freeze_backbone_epochs}) | "
                f"RPN: {'FROZEN' if start_epoch < freeze_rpn_epochs else 'trainable'} "
                f"(freeze_rpn_epochs={freeze_rpn_epochs}) at start epoch {start_epoch}"
            )

    # production.* decode/NMS fields are for saved config + checkpoint inference only
    # (apply_inference_config_to_model in tools/save_predictions.load_model_from_checkpoint;
    # deploy/image_demo). Do not patch the live training model here — model.* stays canonical.

    # Initialize profiler if enabled (single-GPU only; DDP uses same code path on all ranks)
    profiler = None
    if config.enable_profiling and not is_distributed:
        from oriented_det.train.profiler import TrainingProfiler
        from torch.profiler import ProfilerActivity, schedule as profiler_schedule
        
        PROFILING_DIR = EXPERIMENT_DIR / "profiling"
        PROFILING_DIR.mkdir(parents=True, exist_ok=True)
        
        profiler_schedule_config = profiler_schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1,
        )
        
        profiler = TrainingProfiler(
            log_dir=str(PROFILING_DIR),
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=profiler_schedule_config,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
        )
        
        print(f"\nProfiling enabled!")
        print(f"  Profiling directory: {PROFILING_DIR}")
    else:
        print("\nProfiling disabled")
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)
        print(f"TensorBoard: tensorboard --logdir {LOG_DIR.parent}")
        print("=" * 80)
        print()
    
    debug_mode = getattr(args, "debug", False)
    if debug_mode:
        enable_tracing()
        if rank == 0:
            print("Debug tracing enabled (RPN/ROI match statistics).")
    
    log_debug_anchors = getattr(getattr(config, "tensorboard", None), "log_debug_anchors_proposals", False) or debug_mode
    
    try:
        if profiler is not None:
            if rank == 0:
                print("Profiling enabled - wrapping training in profiler context")
            with profiler:
                history = train(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    num_epochs=config.training.num_epochs,
                    val_loader=val_loader,
                    lr_scheduler=lr_scheduler,
                    checkpoint_manager=checkpoint_manager,
                    use_amp=config.training.use_amp,
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                    max_grad_norm=config.training.max_grad_norm,
                    loss_weights=getattr(config.training, 'loss_weights', None),
                    roi_class_weights=class_weight_schedule,
                    start_epoch=start_epoch,
                    writer=writer,
                    class_map=config.class_map,
                    class_names=config.class_names,
                    eval_score_threshold=eval_thr_sc,
                    eval_per_class_score_threshold=eval_thr_pc,
                    eval_vis_score_threshold=getattr(getattr(config, "tensorboard", None), "vis_score_threshold", None),
                    eval_iou_threshold=eval_thr_iou,
                    eval_extended_gt_metrics=getattr(config.evaluation, "extended_gt_metrics", False),
                    eval_compute_map_final=config.evaluation.compute_map_final,
                    eval_compute_map_every_n_epochs=getattr(config.evaluation, "compute_map_every_n_epochs", 0),
                    log_debug_anchors_proposals=log_debug_anchors,
                    profiler=profiler,
                    normalize_mean=norm_mean,
                    normalize_std=norm_std,
                    vis_image_size=resize_to,
                    train_sampler=train_sampler if is_distributed else None,
                    val_sampler=val_sampler if is_distributed else None,
                    rank=rank if is_distributed else None,
                    progress_stream=_orig_stderr if _orig_stderr is not None else None,
                    debug=debug_mode,
                    lr_scheduler_plateau_metric=plateau_metric,
                    early_stop_patience=getattr(config.training, "early_stop_patience", None),
                    early_stop_metric=getattr(config.training, "early_stop_metric", "mAP"),
                    early_stop_min_delta=getattr(config.training, "early_stop_min_delta", 0.0),
                    early_stop_higher_is_better=getattr(config.training, "early_stop_higher_is_better", None),
                    freeze_backbone_epochs=freeze_backbone_epochs,
                    freeze_rpn_epochs=freeze_rpn_epochs,
                    eval_use_exact_rotated_iou=eval_use_exact_rotated_iou,
                    eval_use_exact_rotated_iou_for_final_map=eval_use_exact_rotated_iou_for_final_map,
                )
            if rank == 0:
                print("\n" + "=" * 80)
                print("Profiling Summary:")
                profiler.print_summary(sort_by="cuda_time_total", row_limit=20)
        else:
            history = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                num_epochs=config.training.num_epochs,
                val_loader=val_loader,
                lr_scheduler=lr_scheduler,
                checkpoint_manager=checkpoint_manager,
                use_amp=config.training.use_amp,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                max_grad_norm=config.training.max_grad_norm,
                loss_weights=getattr(config.training, 'loss_weights', None),
                roi_class_weights=class_weight_schedule,
                start_epoch=start_epoch,
                writer=writer,
                class_map=config.class_map,
                class_names=config.class_names,
                eval_score_threshold=eval_thr_sc,
                eval_per_class_score_threshold=eval_thr_pc,
                eval_vis_score_threshold=getattr(getattr(config, "tensorboard", None), "vis_score_threshold", None),
                eval_iou_threshold=eval_thr_iou,
                eval_extended_gt_metrics=getattr(config.evaluation, "extended_gt_metrics", False),
                eval_compute_map_final=config.evaluation.compute_map_final,
                eval_compute_map_every_n_epochs=getattr(config.evaluation, "compute_map_every_n_epochs", 0),
                log_debug_anchors_proposals=log_debug_anchors,
                profiler=None,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
                vis_image_size=resize_to,
                train_sampler=train_sampler if is_distributed else None,
                val_sampler=val_sampler if is_distributed else None,
                rank=rank if is_distributed else None,
                progress_stream=_orig_stderr if _orig_stderr is not None else None,
                debug=debug_mode,
                lr_scheduler_plateau_metric=plateau_metric,
                early_stop_patience=getattr(config.training, "early_stop_patience", None),
                early_stop_metric=getattr(config.training, "early_stop_metric", "mAP"),
                early_stop_min_delta=getattr(config.training, "early_stop_min_delta", 0.0),
                early_stop_higher_is_better=getattr(config.training, "early_stop_higher_is_better", None),
                freeze_backbone_epochs=freeze_backbone_epochs,
                freeze_rpn_epochs=freeze_rpn_epochs,
                eval_use_exact_rotated_iou=eval_use_exact_rotated_iou,
                eval_use_exact_rotated_iou_for_final_map=eval_use_exact_rotated_iou_for_final_map,
            )
        
        if rank == 0:
            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print("=" * 80)
            print(f"Experiment directory: {EXPERIMENT_DIR}")
            print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
            print(f"  TensorBoard logs saved to: {LOG_DIR}")
            print(f"  Config saved to: {EXPERIMENT_DIR / 'config.json'}")
            if profiler is not None:
                print(f"  Profiling traces saved to: {PROFILING_DIR}")
            print(f"\nView TensorBoard with:")
            print(f"  tensorboard --logdir {LOG_DIR.parent}")
            print(f"  tensorboard --logdir {LOG_DIR}")
        
    except KeyboardInterrupt:
        if rank == 0:
            print("\n\nTraining interrupted by user")
            print(f"Experiment directory: {EXPERIMENT_DIR}")
            print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
            print(f"  Config saved to: {EXPERIMENT_DIR / 'config.json'}")
        if writer is not None:
            writer.close()
    except Exception as e:
        if rank == 0:
            print(f"\n\nTraining error: {e}")
            import traceback as _tb
            _tb.print_exc()
            print(f"\nExperiment directory: {EXPERIMENT_DIR}")
            print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
            print(f"  Config saved to: {EXPERIMENT_DIR / 'config.json'}")
        if writer is not None:
            writer.close()
        raise
    finally:
        if writer is not None:
            writer.close()
        # Restore stdout/stderr and close training log file (rank 0)
        if _log_file is not None:
            if _orig_stdout is not None:
                sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
            _log_file.close()
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
