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
from .hrsc2016 import (
    HRSC2016_CLASSES,
    HRSC2016Dataset,
    export_hrsc2016_to_dota,
    format_hrsc_empty_gt_filter_log,
)
from .build import (
    SUPPORTED_DATASET_FORMATS,
    build_split_dataset,
    dataset_format_name,
    split_class_names,
)
from .lookalike import (
    LOOKALIKE_CLASS_NAME,
    filter_semantic_class_names,
    is_lookalike_class_name,
    resolve_lookalike_label_set,
)
from .evaluation import (
    APCalculator,
    ClassEvalMetrics,
    ClassGtBestIouMetrics,
    Detection,
    GroundTruth,
    GtBestIouAlignmentMetrics,
    compute_gt_best_iou_alignment_metrics,
    compute_oriented_map,
    format_gt_best_iou_alignment_table,
    format_gt_best_iou_alignment_table_from_dict,
    format_mmrotate_class_metrics_table,
    gt_best_iou_alignment_metrics_to_dict,
)
from .tiling import (
    ImageTiler,
    Tile,
    TiledSample,
    visualize_tiles,
)
from .flips import apply_flip_to_image, apply_flip_to_rboxes, apply_random_train_flips
from .rotates import apply_random_train_rotate, apply_rotate_to_image, apply_rotate_to_rboxes
from .transforms import (
    AlbumentationsTransform,
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
    "HRSC2016_CLASSES",
    "HRSC2016Dataset",
    "export_hrsc2016_to_dota",
    "format_hrsc_empty_gt_filter_log",
    "SUPPORTED_DATASET_FORMATS",
    "build_split_dataset",
    "dataset_format_name",
    "split_class_names",
    "LOOKALIKE_CLASS_NAME",
    "filter_semantic_class_names",
    "is_lookalike_class_name",
    "resolve_lookalike_label_set",
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
    "create_albumentations_augmentation",
    "apply_flip_to_image",
    "apply_flip_to_rboxes",
    "apply_random_train_flips",
    "apply_random_train_rotate",
    "apply_rotate_to_image",
    "apply_rotate_to_rboxes",
    # Evaluation
    "APCalculator",
    "ClassEvalMetrics",
    "Detection",
    "GroundTruth",
    "ClassGtBestIouMetrics",
    "GtBestIouAlignmentMetrics",
    "compute_gt_best_iou_alignment_metrics",
    "format_gt_best_iou_alignment_table",
    "format_gt_best_iou_alignment_table_from_dict",
    "gt_best_iou_alignment_metrics_to_dict",
]
