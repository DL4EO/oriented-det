"""Build train/val datasets from experiment ``dataset`` config."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .airbus_playground import AirbusPlaygroundCSVDataset
from .dota import (
    build_dota_split_dataset,
    collect_dota_split_image_paths,
    dota_dataset_class_names,
)
from .hrsc2016 import (
    HRSC2016Dataset,
    resolve_hrsc2016_imageset_split,
)

SUPPORTED_DATASET_FORMATS = ("dota", "airbus_playground", "hrsc2016")


def dataset_format_name(dataset_cfg) -> str:
    return (getattr(dataset_cfg, "format", None) or "dota").strip().lower()


def build_split_dataset(
    dataset_cfg,
    split: str,
    *,
    filter_empty_gt: Optional[bool] = None,
):
    """Build one split from ``TrainingExperimentConfig.dataset``.

    ``split`` is the training-loop role (``train`` / ``val``). For HRSC2016 this
    is mapped through ``dataset.train_split`` / ``dataset.val_split`` (defaults
    ``trainval`` / ``test``). ImageSets names (``trainval``, ``test``, …) are
    also accepted directly.
    """
    fmt = dataset_format_name(dataset_cfg)
    if filter_empty_gt is None:
        filter_empty_gt = bool(getattr(dataset_cfg, "filter_empty_gt", False))

    if fmt == "airbus_playground":
        if dataset_cfg.annotations_file is None or dataset_cfg.split_file is None:
            raise ValueError(
                "Airbus Playground dataset format requires dataset.annotations_file "
                "and dataset.split_file."
            )
        role = split.strip().lower()
        if role not in {"train", "val"}:
            raise ValueError(f"Airbus dataset supports only 'train' or 'val' split, got {split!r}.")
        return AirbusPlaygroundCSVDataset(
            data_root=dataset_cfg.data_root,
            split=role,
            annotations_file=dataset_cfg.annotations_file,
            split_file=dataset_cfg.split_file,
            val_split_id=getattr(dataset_cfg, "val_split_id", 0),
            train_includes_val=(
                getattr(dataset_cfg, "train_includes_val", False) if role == "train" else False
            ),
            difficult_strategy=dataset_cfg.difficult_strategy,
            allowed_classes=dataset_cfg.allowed_classes,
            ignore_labels=dataset_cfg.ignore_labels,
            lookalike_labels=getattr(dataset_cfg, "lookalike_labels", None),
            map_labels=getattr(dataset_cfg, "map_labels", None),
            difficult_tags=getattr(dataset_cfg, "difficult_tags", None),
            filter_empty_gt=filter_empty_gt,
        )

    if fmt == "hrsc2016":
        imageset = resolve_hrsc2016_imageset_split(dataset_cfg, split)
        return HRSC2016Dataset(
            data_root=dataset_cfg.data_root,
            split=imageset,
            difficult_strategy=dataset_cfg.difficult_strategy,
            allowed_classes=dataset_cfg.allowed_classes,
            ignore_labels=dataset_cfg.ignore_labels,
            lookalike_labels=getattr(dataset_cfg, "lookalike_labels", None),
            filter_empty_gt=filter_empty_gt,
        )

    if fmt != "dota":
        raise ValueError(
            f"Unsupported dataset.format {fmt!r}. "
            f"Expected one of: {', '.join(SUPPORTED_DATASET_FORMATS)}."
        )

    if not dataset_cfg.has_dota_tiles_config():
        raise ValueError(
            "DOTA dataset format requires dataset.train_tiles_dir or "
            "dataset.train_tiles_dirs, and dataset.val_tiles_dir or dataset.val_tiles_dirs."
        )
    role = split.strip().lower()
    if role == "train":
        tile_roots = dataset_cfg.get_train_tile_roots()
    elif role == "val":
        tile_roots = dataset_cfg.get_val_tile_roots()
    else:
        raise ValueError(f"DOTA split must be 'train' or 'val', got {split!r}.")
    same_folder = getattr(dataset_cfg, "same_folder", False)
    return build_dota_split_dataset(
        tile_roots,
        split=role,
        same_folder=same_folder,
        difficult_strategy=dataset_cfg.difficult_strategy,
        allowed_classes=dataset_cfg.allowed_classes,
        ignore_labels=dataset_cfg.ignore_labels,
        lookalike_labels=getattr(dataset_cfg, "lookalike_labels", None),
        filter_empty_gt=filter_empty_gt,
    )


def split_class_names(dataset, dataset_cfg) -> List[str]:
    """Class names for a dataset built by :func:`build_split_dataset`."""
    if dataset_format_name(dataset_cfg) == "dota":
        return dota_dataset_class_names(dataset)
    return dataset.get_class_names()


def collect_split_images(
    config,
    data_root: Path,
    data_split: str = "val",
    val_dir: Optional[Path] = None,
    *,
    filter_empty_gt: Optional[bool] = None,
) -> Tuple[List[Path], Optional[Path], str]:
    """Return image paths, label directory (DOTA), and dataset format string.

    ``filter_empty_gt``: when ``None``, use ``config.dataset.filter_empty_gt``
    (training). ``tools/save_predictions.py`` always passes ``False`` so
    inference includes all tiles.
    """
    from dataclasses import replace

    dataset_format = dataset_format_name(getattr(config, "dataset", None))
    data_root = Path(data_root)

    if dataset_format in ("airbus_playground", "hrsc2016"):
        ds_config = config.dataset
        if dataset_format == "airbus_playground":
            if not getattr(ds_config, "annotations_file", None) or not getattr(ds_config, "split_file", None):
                raise ValueError(
                    "Airbus Playground format requires dataset.annotations_file and dataset.split_file."
                )
            if data_split not in ("train", "val"):
                raise ValueError(f"Airbus dataset supports only train or val split, got {data_split!r}.")
        ds_cfg = replace(ds_config, data_root=data_root)
        if filter_empty_gt is None:
            if dataset_format == "airbus_playground":
                filter_empty_gt = False
            else:
                filter_empty_gt = bool(getattr(ds_config, "filter_empty_gt", False))
        dataset = build_split_dataset(ds_cfg, data_split, filter_empty_gt=filter_empty_gt)
        split_images = [Path(dataset[idx].image_path) for idx in range(len(dataset))]
        return split_images, None, dataset_format

    if getattr(config, "dataset", None):
        if data_split == "val" and val_dir is not None:
            tile_roots = [Path(val_dir)]
        elif data_split == "val":
            tile_roots = config.dataset.get_val_tile_roots()
        elif data_split == "train":
            tile_roots = config.dataset.get_train_tile_roots()
        else:
            tile_roots = [data_root / data_split]
        same_folder = getattr(config.dataset, "same_folder", False)
        difficult_strategy = getattr(config.dataset, "difficult_strategy", "drop")
        allowed_classes = getattr(config.dataset, "allowed_classes", None)
        ignore_labels = getattr(config.dataset, "ignore_labels", None)
        if filter_empty_gt is None:
            filter_empty_gt = bool(getattr(config.dataset, "filter_empty_gt", False))
        split_images = collect_dota_split_image_paths(
            tile_roots,
            split=data_split,
            same_folder=same_folder,
            difficult_strategy=difficult_strategy,
            allowed_classes=allowed_classes,
            ignore_labels=ignore_labels,
            filter_empty_gt=filter_empty_gt,
            log_filter_empty_gt=filter_empty_gt,
        )
        if not split_images:
            raise ValueError(f"No images found under DOTA tile roots: {tile_roots}")
        label_dir = None
        if len(tile_roots) == 1:
            split_root = tile_roots[0]
            split_tiles_dir = split_root / "tiles_1024"
            if same_folder:
                label_dir = split_root
            elif (split_root / "labels").exists():
                label_dir = split_root / "labels"
            elif split_tiles_dir.exists() and (split_tiles_dir / "labels").exists():
                label_dir = split_tiles_dir / "labels"
            elif (split_root / "labelTxt").exists():
                label_dir = split_root / "labelTxt"
        return split_images, label_dir, dataset_format

    split_root = data_root / data_split
    split_tiles_dir = split_root / "tiles_1024"
    same_folder = getattr(getattr(config, "dataset", None), "same_folder", False)

    if same_folder:
        image_dir = split_root
        label_dir = split_root
    elif split_tiles_dir.exists() and (split_tiles_dir / "images").exists():
        image_dir = split_tiles_dir / "images"
        label_dir = split_tiles_dir / "labels"
    elif (split_root / "images").exists():
        image_dir = split_root / "images"
        label_dir = (
            split_root / "labels"
            if (split_root / "labels").exists()
            else split_root / "labelTxt"
        )
    else:
        raise ValueError(f"Could not find images under {split_tiles_dir} or {split_root}")

    split_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not split_images:
        raise ValueError(f"No images found in {image_dir}")
    return split_images, label_dir, dataset_format
