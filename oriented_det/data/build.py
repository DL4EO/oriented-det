"""Build train/val datasets from experiment ``dataset`` config."""

from __future__ import annotations

from typing import List, Optional

from .airbus_playground import AirbusPlaygroundCSVDataset
from .dota import build_dota_split_dataset, dota_dataset_class_names
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
