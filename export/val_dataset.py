"""Validation image enumeration (shared with tools/save_predictions.py semantics)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from oriented_det.data.airbus_playground import AirbusPlaygroundCSVDataset
from oriented_det.data.dota import collect_dota_split_image_paths
from oriented_det.train.config import TrainingExperimentConfig


def collect_split_images(
    config: TrainingExperimentConfig,
    data_root: Path,
    data_split: str = "val",
    val_dir: Optional[Path] = None,
    *,
    filter_empty_gt: Optional[bool] = None,
) -> Tuple[List[Path], Optional[Path], str]:
    """Return image paths, label directory (DOTA), and dataset format string.

    ``filter_empty_gt``: when ``None``, use ``config.dataset.filter_empty_gt`` (training).
    ``tools/save_predictions.py`` (``make preds`` / ``make metrics``) always passes ``False``
    so inference includes all tiles; training still uses the config flag.
    """
    dataset_format = getattr(getattr(config, "dataset", None), "format", None) or "dota"
    data_root = Path(data_root)

    if dataset_format == "airbus_playground":
        ds_config = config.dataset
        if not getattr(ds_config, "annotations_file", None) or not getattr(ds_config, "split_file", None):
            raise ValueError(
                "Airbus Playground format requires dataset.annotations_file and dataset.split_file."
            )
        if data_split not in ("train", "val"):
            raise ValueError(f"Airbus dataset supports only train or val split, got {data_split!r}.")
        airbus_dataset = AirbusPlaygroundCSVDataset(
            data_root=data_root,
            split=data_split,
            annotations_file=ds_config.annotations_file,
            split_file=ds_config.split_file,
            val_split_id=getattr(ds_config, "val_split_id", 0),
            difficult_strategy=ds_config.difficult_strategy,
            allowed_classes=getattr(ds_config, "allowed_classes", None),
            ignore_labels=getattr(ds_config, "ignore_labels", None) or [],
            map_labels=getattr(ds_config, "map_labels", None) or {},
        )
        split_images = [Path(airbus_dataset[idx].image_path) for idx in range(len(airbus_dataset))]
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
