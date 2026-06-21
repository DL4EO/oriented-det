"""Data loading, tiling, transforms, and evaluation for oriented detection."""

from .dota import (
    DOTAAnnotation,
    DOTADataset,
    DOTASample,
    build_dota_loader,
    build_dota_split_dataset,
    collect_dota_image_paths,
    collect_dota_split_image_paths,
    dota_dataset_class_names,
    dota_label_path_for_image,
    format_dota_empty_gt_filter_log,
    format_dota_line,
    resolve_dota_tile_roots,
)
from .airbus_playground import (
    AirbusPlaygroundCSVDataset,
    detect_airbus_split_csv_format,
    format_airbus_empty_gt_filter_log,
    generate_airbus_playground_csvs,
)
from .evaluation import (
    APCalculator,
    ClassEvalMetrics,
    Detection,
    GroundTruth,
    compute_oriented_map,
    format_mmrotate_class_metrics_table,
)
from .tiling import (
    ImageTiler,
    Tile,
    TiledSample,
    visualize_tiles,
)
from .flips import apply_random_train_flips
from .transforms import (
    AlbumentationsTransform,
    Compose,
    DiagonalFlip,
    HorizontalFlip,
    OrientedTransform,
    Rotate,
    VerticalFlip,
    create_albumentations_augmentation,
)

__all__ = [
    # DOTA dataset
    "DOTAAnnotation",
    "DOTADataset",
    "DOTASample",
    "build_dota_loader",
    "build_dota_split_dataset",
    "collect_dota_image_paths",
    "collect_dota_split_image_paths",
    "dota_dataset_class_names",
    "dota_label_path_for_image",
    "format_dota_empty_gt_filter_log",
    "format_dota_line",
    "resolve_dota_tile_roots",
    "AirbusPlaygroundCSVDataset",
    "detect_airbus_split_csv_format",
    "format_airbus_empty_gt_filter_log",
    "generate_airbus_playground_csvs",
    # Tiling
    "ImageTiler",
    "Tile",
    "TiledSample",
    "visualize_tiles",
    # Transforms
    "AlbumentationsTransform",
    "Compose",
    "create_albumentations_augmentation",
    "apply_random_train_flips",
    "HorizontalFlip",
    "DiagonalFlip",
    "OrientedTransform",
    "Rotate",
    "VerticalFlip",
    # Evaluation
    "APCalculator",
    "ClassEvalMetrics",
    "Detection",
    "GroundTruth",
    "compute_oriented_map",
    "format_mmrotate_class_metrics_table",
]
