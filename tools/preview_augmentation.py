#!/usr/bin/env python3
"""Preview training augmentations from an experiment config.

Loads the train split, applies the same collate path as training (Albumentations,
random flips, resize), and writes comparison grids so you can sanity-check
augmentation.json / recipe overrides before a long run.

Usage (from odet-planes/, with oriented-det venv active):

    python tools/preview_augmentation.py --config configs/oriented_rcnn/dota_le90_1x.json
    python tools/preview_augmentation.py --config runs/oriented_rcnn/20260616-030231/config.json --num-images 4 --variants 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from dataclasses import fields

from oriented_det.data import AirbusPlaygroundCSVDataset, build_dota_split_dataset, dota_dataset_class_names
from oriented_det.geometry.rbox import RBox
from oriented_det.runtime.collate import create_collate_fn, create_train_augmentation
from oriented_det.train.config import (
    AugmentationConfig,
    CheckpointConfig,
    DataLoaderConfig,
    DatasetConfig,
    EvaluationConfig,
    LossConfig,
    ModelConfig,
    PreprocessingConfig,
    ProductionConfig,
    TensorboardConfig,
    TrainingConfig,
    TrainingExperimentConfig,
    _normalize_legacy_loss_type,
)
from oriented_det.utils import viz
from oriented_det.utils.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


_SECTION_CLASSES = {
    "dataset": DatasetConfig,
    "augmentation": AugmentationConfig,
    "data_loader": DataLoaderConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "production": ProductionConfig,
    "checkpoint": CheckpointConfig,
    "loss": LossConfig,
    "tensorboard": TensorboardConfig,
    "preprocessing": PreprocessingConfig,
}


def _filter_section(section: dict, dc_cls: type) -> dict:
    valid = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in section.items() if k in valid}


def load_training_config(path: Path) -> TrainingExperimentConfig:
    """Load config; ignore unknown keys in sections (preview-only lenient mode)."""
    try:
        return TrainingExperimentConfig.load(path)
    except ValueError as exc:
        if "Unknown key" not in str(exc):
            raise

    frozen_cfg = load_config(path)
    config_dict = frozen_cfg.to_dict()
    _normalize_legacy_loss_type(config_dict)

    for key in list(config_dict):
        if config_dict[key] is None and key in _SECTION_CLASSES:
            del config_dict[key]

    root_valid = {f.name for f in fields(TrainingExperimentConfig)}
    root = {k: v for k, v in config_dict.items() if k in root_valid}

    for section, dc_cls in _SECTION_CLASSES.items():
        if section not in config_dict or config_dict[section] is None:
            continue
        root[section] = dc_cls(**_filter_section(config_dict[section], dc_cls))

    return TrainingExperimentConfig(**root)


def resolve_class_map(
    config: TrainingExperimentConfig, dataset
) -> tuple[dict[str, int], list[str]]:
    if config.class_map:
        names = list(config.class_names or [])
        return dict(config.class_map), names
    if hasattr(dataset, "get_class_names"):
        names = dataset.get_class_names()
    else:
        names = dota_dataset_class_names(dataset)
    class_map = {name: i + 1 for i, name in enumerate(names)}
    return class_map, names


def resolve_config_path(config_arg: str) -> Path:
    raw = Path(config_arg)
    if raw.is_absolute() and raw.exists():
        return raw
    for root in (_repo_root(), Path.cwd()):
        candidate = root / raw
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config not found: {config_arg}")


def build_dataset_from_config(config: TrainingExperimentConfig, split: str):
    """Build train or val dataset (same logic as tools/train.py / dataset_stats.py)."""
    dataset_format = getattr(config.dataset, "format", "dota").lower()
    filter_empty = getattr(config.dataset, "filter_empty_gt", False) if split == "train" else False

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
            filter_empty_gt=filter_empty,
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
        filter_empty_gt=filter_empty,
    )


def get_resize_and_flips(config: TrainingExperimentConfig) -> tuple[str, tuple[int, int], bool, bool, bool, int]:
    from oriented_det.data.preprocessing import parse_canvas_size

    prep = getattr(config, "preprocessing", None)
    if prep is not None:
        resize_mode = getattr(prep, "resize_mode", "fixed")
        ts = getattr(prep, "target_size", [1024, 1024])
        resize_to = parse_canvas_size(resize_mode, ts)
    else:
        resize_mode = "fixed"
        resize_to = (1024, 1024)
    flip_h = getattr(prep, "enable_flip_horizontal", True) if prep is not None else True
    flip_v = getattr(prep, "enable_flip_vertical", True) if prep is not None else True
    flip_d = getattr(prep, "enable_flip_diagonal", False) if prep is not None else False
    pad_div = getattr(prep, "pad_size_divisor", 32) if prep is not None else 32
    return resize_mode, resize_to, flip_h, flip_v, flip_d, pad_div


def build_train_augmentation_from_config(config: TrainingExperimentConfig):
    if not config.enable_albumentation:
        return None
    aug = config.augmentation
    return create_train_augmentation(
        brightness_limit=aug.brightness_limit,
        contrast_limit=aug.contrast_limit,
        gamma_limit=aug.gamma_limit,
        gauss_noise_var_limit=aug.gauss_noise_var_limit,
        blur_limit=aug.blur_limit,
        clahe_clip_limit=aug.clahe_clip_limit,
        p_brightness_contrast=aug.p_brightness_contrast,
        p_gamma=aug.p_gamma,
        p_noise=aug.p_noise,
        p_blur=aug.p_blur,
        p_clahe=aug.p_clahe,
    )


def make_collate_fn(
    config: TrainingExperimentConfig,
    *,
    augmentation: Any,
    enable_flip_horizontal: bool,
    enable_flip_vertical: bool,
    enable_flip_diagonal: bool,
):
    resize_mode, resize_to, _, _, _, pad_div = get_resize_and_flips(config)
    prep = getattr(config, "preprocessing", None)
    norm_mean = getattr(prep, "normalize_mean", None) if prep is not None else None
    norm_std = getattr(prep, "normalize_std", None) if prep is not None else None
    return create_collate_fn(
        config.class_map,
        augmentation=augmentation,
        normalize=False,
        resize_mode=resize_mode,
        resize_to=resize_to,
        pad_size_divisor=pad_div,
        enable_flip_horizontal=enable_flip_horizontal,
        enable_flip_vertical=enable_flip_vertical,
        enable_flip_diagonal=enable_flip_diagonal,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
        difficult_strategy=getattr(config.dataset, "difficult_strategy", "drop"),
    )


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    t = image_tensor.detach().cpu().float().clamp(0, 1)
    if t.dim() != 3 or t.shape[0] != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def draw_targets(
    image: Image.Image,
    target: dict,
    class_names: list[str],
    title: str,
) -> Image.Image:
    image = image.copy()
    rboxes = target.get("rboxes")
    labels = target.get("labels")
    if rboxes is None or rboxes.numel() == 0:
        specs = []
        box_objs = []
    else:
        box_objs = [
            RBox(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            )
            for row in rboxes
        ]
        specs = [viz.DrawingSpec(outline=(0, 255, 0), width=2) for _ in box_objs]
        image = viz.draw_boxes(image, box_objs, specs=specs)
        if labels is not None and len(labels) == len(box_objs):
            label_texts = []
            for lid in labels.tolist():
                idx = int(lid) - 1
                if 0 <= idx < len(class_names):
                    label_texts.append(class_names[idx])
                else:
                    label_texts.append(str(lid))
            polys = [b.to_polygon().points for b in box_objs]
            for poly, text in zip(polys, label_texts):
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x, y = min(xs), min(ys)
                draw = ImageDraw.Draw(image)
                draw.text((x, y - 12), text, fill=(255, 255, 0))

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0, 0, image.width, 22], fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    return image


def stitch_row(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    h = max(im.height for im in panels)
    w = sum(im.width for im in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), (32, 32, 32))
    x = 0
    for im in panels:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def stitch_grid(rows: list[Image.Image], gap: int = 12) -> Image.Image:
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), (24, 24, 24))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas


def print_augmentation_summary(config: TrainingExperimentConfig, resize_to: tuple[int, int]) -> None:
    prep = getattr(config, "preprocessing", None)
    resize_mode, _, flip_h, flip_v, flip_d, _ = get_resize_and_flips(config)
    print(f"  enable_albumentation: {config.enable_albumentation}")
    print(f"  resize_mode: {resize_mode}")
    print(f"  resize_to (H×W): {resize_to[0]}×{resize_to[1]}")
    print(f"  flips: horizontal={flip_h}, vertical={flip_v}, diagonal={flip_d}")
    if config.enable_albumentation:
        aug = config.augmentation
        print("  augmentation:")
        for key in (
            "brightness_limit",
            "contrast_limit",
            "gamma_limit",
            "gauss_noise_var_limit",
            "blur_limit",
            "clahe_clip_limit",
            "p_brightness_contrast",
            "p_gamma",
            "p_noise",
            "p_blur",
            "p_clahe",
        ):
            print(f"    {key}: {getattr(aug, key)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Training recipe or resolved run config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/previews/augmentation/<config_stem>)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of distinct tiles to preview (default: 3)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=4,
        help="Random augmented variants per tile (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for tile and augmentation sampling",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices (overrides random sampling)",
    )
    parser.add_argument(
        "--include-albumentations-only",
        action="store_true",
        help="Add a column with Albumentations only (no random flips)",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    print(f"Loading config: {config_path}")
    config = load_training_config(config_path)

    dataset = build_dataset_from_config(config, args.split)
    class_map, class_names = resolve_class_map(config, dataset)
    config.class_map = class_map
    config.class_names = class_names
    if len(dataset) == 0:
        print("Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    resize_mode, resize_to, flip_h, flip_v, flip_d, _ = get_resize_and_flips(config)
    print(f"Building {args.split} dataset ({len(dataset)} samples)")
    print_augmentation_summary(config, resize_to)

    train_aug = build_train_augmentation_from_config(config)
    collate_baseline = make_collate_fn(
        config,
        augmentation=None,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
    )
    collate_train = make_collate_fn(
        config,
        augmentation=train_aug,
        enable_flip_horizontal=flip_h,
        enable_flip_vertical=flip_v,
        enable_flip_diagonal=flip_d,
    )
    collate_albu_only = None
    if args.include_albumentations_only and train_aug is not None:
        collate_albu_only = make_collate_fn(
            config,
            augmentation=train_aug,
            enable_flip_horizontal=False,
            enable_flip_vertical=False,
            enable_flip_diagonal=False,
        )

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        rng = random.Random(args.seed)
        k = min(args.num_images, len(dataset))
        indices = rng.sample(range(len(dataset)), k=k)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _repo_root() / "previews" / "augmentation" / config_path.stem
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": str(config_path),
        "split": args.split,
        "indices": indices,
        "seed": args.seed,
        "variants": args.variants,
        "enable_albumentation": config.enable_albumentation,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    grid_rows: list[Image.Image] = []
    for idx in indices:
        sample = dataset[idx]
        stem = Path(sample.image_path).stem
        panels: list[Image.Image] = []

        images, targets = collate_baseline([sample])
        panels.append(
            draw_targets(
                tensor_to_pil(images[0]),
                targets[0],
                class_names,
                "baseline (resize, no aug)",
            )
        )

        if collate_albu_only is not None:
            images, targets = collate_albu_only([sample])
            panels.append(
                draw_targets(
                    tensor_to_pil(images[0]),
                    targets[0],
                    class_names,
                    "albumentations only",
                )
            )

        label_prefix = "train aug" if train_aug is not None else "train flip"
        for v in range(args.variants):
            random.seed(args.seed + idx * 1000 + v)
            images, targets = collate_train([sample])
            panels.append(
                draw_targets(
                    tensor_to_pil(images[0]),
                    targets[0],
                    class_names,
                    f"{label_prefix} #{v + 1}",
                )
            )

        row = stitch_row(panels)
        row_path = out_dir / f"{idx:05d}_{stem}.png"
        row.save(row_path)
        grid_rows.append(row)
        print(f"  wrote {row_path.name}")

    combined = stitch_grid(grid_rows)
    combined_path = out_dir / "grid_all.png"
    combined.save(combined_path)
    print(f"\nCombined grid: {combined_path}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
