#!/usr/bin/env python3
"""Dataset statistics, sanity checks, and normalization (mean/std) for training configs.

Uses the same dataset and target size as your training config. Can run:
- Sanity checks: missing images, failed loads, empty annotations, duplicate paths
- Dataset stats: class distribution, annotations per image, image dimensions
- Normalization: per-channel mean and std in [0, 1] for config.preprocessing

Usage:
    # Full run (sanity checks + stats + normalization)
    python tools/dataset_stats.py --config configs/.../config.json

    # Stats and sanity only (no normalization; faster)
    python tools/dataset_stats.py --config path/to/config.json --stats-only

    # Normalization only (legacy behavior)
    python tools/dataset_stats.py --config path/to/config.json --normalization-only

    # Limit samples (e.g. for quick normalization)
    python tools/dataset_stats.py --config path/to/config.json --max-samples 500

    # Use validation split
    python tools/dataset_stats.py --config path/to/config.json --split val
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset
from torchvision import transforms as T
from PIL import Image

# Add tools directory so helpers can be imported when run from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from oriented_det.data import DOTADataset, AirbusPlaygroundCSVDataset, build_dota_split_dataset
from oriented_det.train.config import TrainingExperimentConfig


def build_dataset_from_config(config: TrainingExperimentConfig, split: str):
    """Build train or val dataset from config (same logic as tools/train.py)."""
    dataset_format = getattr(config.dataset, "format", "dota").lower()
    if dataset_format == "airbus_playground":
        if config.dataset.annotations_file is None or config.dataset.split_file is None:
            raise ValueError(
                "Airbus Playground requires dataset.annotations_file and dataset.split_file."
            )
        return AirbusPlaygroundCSVDataset(
            data_root=config.dataset.data_root,
            split=split,
            annotations_file=config.dataset.annotations_file,
            split_file=config.dataset.split_file,
            val_split_id=config.dataset.val_split_id,
            difficult_strategy=config.dataset.difficult_strategy,
            allowed_classes=config.dataset.allowed_classes,
            ignore_labels=config.dataset.ignore_labels,
            map_labels=config.dataset.map_labels,
        )
    if not config.dataset.has_dota_tiles_config():
        raise ValueError(
            "DOTA format requires dataset.train_tiles_dir(s) and dataset.val_tiles_dir(s)."
        )
    tile_roots = (
        config.dataset.get_train_tile_roots()
        if split == "train"
        else config.dataset.get_val_tile_roots()
    )
    same_folder = getattr(config.dataset, "same_folder", False)
    return build_dota_split_dataset(
        tile_roots,
        split=split,
        same_folder=same_folder,
        difficult_strategy=config.dataset.difficult_strategy,
        allowed_classes=config.dataset.allowed_classes,
        ignore_labels=config.dataset.ignore_labels,
        filter_empty_gt=getattr(config.dataset, "filter_empty_gt", False),
    )


def get_target_size(config: TrainingExperimentConfig) -> tuple[int, int]:
    """Return (height, width) from config preprocessing (same as training)."""
    from oriented_det.data.preprocessing import parse_canvas_size

    prep = getattr(config, "preprocessing", None)
    if prep is not None:
        mode = getattr(prep, "resize_mode", "fixed")
        ts = getattr(prep, "target_size", [1024, 1024])
        return parse_canvas_size(mode, ts)
    return (1024, 1024)


def run_sanity_checks(dataset, max_samples: int | None = None) -> dict[str, Any]:
    """Run sanity checks on the dataset. Returns a dict of counts and lists of problematic indices/paths."""
    n = len(dataset)
    if max_samples is not None and n > max_samples:
        n = max_samples

    missing = []
    failed_load = []
    empty_annot = []
    seen_paths: set[str] = set()
    duplicates = []

    for idx in range(n):
        sample = dataset[idx]
        path = Path(sample.image_path)
        path_str = str(path.resolve())

        if not path.exists():
            missing.append((idx, path_str))
            continue

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            failed_load.append((idx, path_str))
            continue

        if not getattr(sample, "annotations", None) or len(sample.annotations) == 0:
            empty_annot.append((idx, path_str))

        if path_str in seen_paths:
            duplicates.append((idx, path_str))
        seen_paths.add(path_str)

    return {
        "total_checked": n,
        "missing_images": missing,
        "failed_load": failed_load,
        "empty_annotations": empty_annot,
        "duplicate_paths": duplicates,
        "n_missing": len(missing),
        "n_failed_load": len(failed_load),
        "n_empty_annot": len(empty_annot),
        "n_duplicates": len(duplicates),
    }


def _min_max_mean_int(values: list[int]):
    """Return (min, max, mean) for a list of ints; mean rounded to 2 decimals."""
    if not values:
        return None, None, None
    return min(values), max(values), round(sum(values) / len(values), 2)


def _min_max_mean_float(values: list[float]):
    """Return (min, max, mean) for a list of floats; mean rounded to 2 decimals."""
    if not values:
        return None, None, None
    return min(values), max(values), round(sum(values) / len(values), 2)


def run_dataset_stats(dataset, max_samples: int | None = None) -> dict[str, Any]:
    """Collect dataset statistics: class counts, annotations per image, image and annotation dimensions."""
    n = len(dataset)
    if max_samples is not None and n > max_samples:
        indices = range(max_samples)
        n = max_samples
    else:
        indices = range(n)

    class_counts: Counter[str] = Counter()
    objs_per_image: list[int] = []
    image_widths: list[int] = []
    image_heights: list[int] = []
    ann_widths: list[float] = []
    ann_heights: list[float] = []

    for idx in indices:
        sample = dataset[idx]
        n_objs = len(getattr(sample, "annotations", ()))
        objs_per_image.append(n_objs)
        for ann in getattr(sample, "annotations", ()):
            class_name = getattr(ann, "class_name", None) or str(ann)
            class_counts[class_name] += 1
            rbox = getattr(ann, "rbox", None)
            if rbox is not None:
                ann_widths.append(float(getattr(rbox, "width", 0)))
                ann_heights.append(float(getattr(rbox, "height", 0)))
        w = getattr(sample, "width", None)
        h = getattr(sample, "height", None)
        if w is not None:
            image_widths.append(int(w))
        if h is not None:
            image_heights.append(int(h))

    o_min, o_max, o_mean = _min_max_mean_int(objs_per_image)
    iw_min, iw_max, iw_mean = _min_max_mean_int(image_widths)
    ih_min, ih_max, ih_mean = _min_max_mean_int(image_heights)
    aw_min, aw_max, aw_mean = _min_max_mean_float(ann_widths)
    ah_min, ah_max, ah_mean = _min_max_mean_float(ann_heights)

    return {
        "n_samples": n,
        "class_counts": dict(class_counts.most_common()),
        "annotations_per_image": {"min": o_min, "max": o_max, "mean": o_mean},
        "image_width": {"min": iw_min, "max": iw_max, "mean": iw_mean},
        "image_height": {"min": ih_min, "max": ih_max, "mean": ih_mean},
        "annotation_width": {"min": aw_min, "max": aw_max, "mean": aw_mean},
        "annotation_height": {"min": ah_min, "max": ah_max, "mean": ah_mean},
        "total_annotations": sum(class_counts.values()),
    }


def compute_mean_std(
    dataset,
    target_size: tuple[int, int],
    max_samples: int | None = None,
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean and std over dataset images in [0, 1] scale.

    Images are loaded, resized to target_size (same as training pipeline), and
    converted with ToTensor() before accumulating pixel statistics.
    Returns (mean, std) as lists of length 3 (RGB).
    """
    to_tensor = T.ToTensor()
    target_h, target_w = target_size

    if max_samples is not None and len(dataset) > max_samples:
        dataset = Subset(dataset, range(max_samples))

    n_pixels = 0
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)

    total = len(dataset)
    for idx in range(total):
        sample = dataset[idx]
        image_path = sample.image_path
        if not Path(image_path).exists():
            print(f"Warning: image not found {image_path}", file=sys.stderr)
            continue
        try:
            pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to load {image_path}: {e}", file=sys.stderr)
            continue
        pil = pil.resize((target_w, target_h), Image.BILINEAR)
        tensor = to_tensor(pil)
        n = tensor.shape[1] * tensor.shape[2]
        sum_c += tensor.sum(dim=(1, 2)).double()
        sum_sq_c += (tensor ** 2).sum(dim=(1, 2)).double()
        n_pixels += n
        if (idx + 1) % 200 == 0 or idx == total - 1:
            print(f"  Processed {idx + 1}/{total} images ...", flush=True)

    if n_pixels == 0:
        raise RuntimeError("No valid pixels from any image. Check dataset paths and --max-samples.")

    mean = (sum_c / n_pixels).tolist()
    var = (sum_sq_c / n_pixels) - (sum_c / n_pixels) ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var).tolist()
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset sanity checks, statistics, and normalization mean/std for config.preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON (e.g. configs/.../config.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of images to use for stats/normalization (default: all).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val"),
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only run sanity checks and dataset stats; skip normalization.",
    )
    parser.add_argument(
        "--normalization-only",
        action="store_true",
        help="Only compute normalization mean/std; skip sanity checks and stats.",
    )
    args = parser.parse_args()

    if args.stats_only and args.normalization_only:
        print("Error: cannot use both --stats-only and --normalization-only.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        repo_root = _SCRIPT_DIR.parent
        config_path = repo_root / args.config
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config: {config_path}")
    config = TrainingExperimentConfig.load(config_path)
    print(f"Building {args.split} dataset...")
    dataset = build_dataset_from_config(config, args.split)
    print(f"Dataset size: {len(dataset)} images")
    if args.max_samples:
        print(f"Using at most {args.max_samples} samples for stats/normalization")

    run_both = not args.stats_only and not args.normalization_only

    if run_both or args.stats_only:
        # Sanity checks
        print("\n--- Sanity checks ---")
        sanity = run_sanity_checks(dataset, args.max_samples)
        print(f"  Checked: {sanity['total_checked']} samples")
        print(f"  Missing images: {sanity['n_missing']}")
        print(f"  Failed to load: {sanity['n_failed_load']}")
        print(f"  Empty annotations: {sanity['n_empty_annot']}")
        print(f"  Duplicate paths: {sanity['n_duplicates']}")
        if sanity["missing_images"]:
            for idx, p in sanity["missing_images"][:5]:
                print(f"    [{idx}] {p}")
            if len(sanity["missing_images"]) > 5:
                print(f"    ... and {len(sanity['missing_images']) - 5} more")
        if sanity["failed_load"]:
            for idx, p in sanity["failed_load"][:5]:
                print(f"    [{idx}] {p}")
            if len(sanity["failed_load"]) > 5:
                print(f"    ... and {len(sanity['failed_load']) - 5} more")

        # Dataset stats
        print("\n--- Dataset stats ---")
        stats = run_dataset_stats(dataset, args.max_samples)
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Annotations per image: min={stats['annotations_per_image']['min']}, max={stats['annotations_per_image']['max']}, mean={stats['annotations_per_image']['mean']}")
        if stats.get("image_width", {}).get("min") is not None:
            print(f"  Image width:  min={stats['image_width']['min']}, max={stats['image_width']['max']}, mean={stats['image_width']['mean']}")
        if stats.get("image_height", {}).get("min") is not None:
            print(f"  Image height: min={stats['image_height']['min']}, max={stats['image_height']['max']}, mean={stats['image_height']['mean']}")
        if (
            stats.get("image_width", {}).get("min") is not None
            and stats.get("image_height", {}).get("min") is not None
            and stats["image_width"]["min"] == stats["image_width"]["max"]
            and stats["image_height"]["min"] == stats["image_height"]["max"]
        ):
            print(f"  (all images have the same size: {stats['image_width']['min']}×{stats['image_height']['min']})")
        if stats.get("annotation_width", {}).get("min") is not None:
            print(f"  Annotation (OBB) width:  min={stats['annotation_width']['min']:.2f}, max={stats['annotation_width']['max']:.2f}, mean={stats['annotation_width']['mean']:.2f}")
        if stats.get("annotation_height", {}).get("min") is not None:
            print(f"  Annotation (OBB) length: min={stats['annotation_height']['min']:.2f}, max={stats['annotation_height']['max']:.2f}, mean={stats['annotation_height']['mean']:.2f}")
        print("  Class counts:")
        for cls, cnt in list(stats["class_counts"].items())[:20]:
            print(f"    {cls}: {cnt}")
        if len(stats["class_counts"]) > 20:
            print(f"    ... and {len(stats['class_counts']) - 20} more classes")

    if run_both or args.normalization_only:
        target_size = get_target_size(config)
        print(f"\n--- Normalization (target size {target_size}) ---")
        print("Computing mean and std (images in [0,1] after resize)...")
        mean, std = compute_mean_std(dataset, target_size, args.max_samples)
        print()
        print("Results (RGB, [0, 1] scale — use in config.preprocessing):")
        print(f"  normalize_mean: {[round(x, 6) for x in mean]}")
        print(f"  normalize_std:  {[round(x, 6) for x in std]}")
        print()
        print("JSON snippet for your config:")
        print('  "preprocessing": {')
        print(f'    "normalize_mean": {mean},')
        print(f'    "normalize_std": {std}')
        print("  }")


if __name__ == "__main__":
    main()
