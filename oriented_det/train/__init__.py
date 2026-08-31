"""Training engine and utilities for oriented detection."""

from .engine import (
    evaluate,
    train,
    train_one_epoch,
)
from .utils import (
    CheckpointManager,
    CosineAnnealingWithFixedTailLR,
    MetricTracker,
    OneCycleWrapper,
    WarmupScheduler,
    collate_dota_samples,
    collate_fn_generic,
    create_cosine_with_tail_lr_scheduler,
    create_pytorch_cosine_lr_scheduler,
    format_cosine_with_tail_scheduler_description,
    format_pytorch_cosine_scheduler_description,
    get_best_checkpoint_path,
    resolve_cosine_with_tail_lengths,
    resolve_pytorch_cosine_t_max,
    save_training_config,
)

try:
    from .profiler import TrainingProfiler, profile_training_step
    _PROFILER_AVAILABLE = True
except ImportError:
    TrainingProfiler = None  # type: ignore
    profile_training_step = None  # type: ignore
    _PROFILER_AVAILABLE = False

from .grouped_ce import (
    GroupedCeSpec,
    build_grouped_ce_spec,
    configure_roi_grouped_ce,
    grouped_ce_alpha_for_epoch,
)

from .config import (
    TrainingExperimentConfig,
    DatasetConfig,
    DataLoaderConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ProductionConfig,
    CheckpointConfig,
    AugmentationConfig,
    LossConfig,
    resolve_inference_score_threshold,
    resolve_inference_sliding_window_overlap_pixels,
    resolve_preds_final_nms_iou_threshold,
    effective_eval_metric_thresholds,
    apply_inference_config_to_model,
)

__all__ = [
    # Engine
    "train_one_epoch",
    "evaluate",
    "train",
    # Utils
    "MetricTracker",
    "CheckpointManager",
    "CosineAnnealingWithFixedTailLR",
    "OneCycleWrapper",
    "WarmupScheduler",
    "create_cosine_with_tail_lr_scheduler",
    "create_pytorch_cosine_lr_scheduler",
    "format_cosine_with_tail_scheduler_description",
    "format_pytorch_cosine_scheduler_description",
    "resolve_cosine_with_tail_lengths",
    "resolve_pytorch_cosine_t_max",
    "collate_dota_samples",
    "collate_fn_generic",
    "get_best_checkpoint_path",
    "save_training_config",
    # Profiler
    "TrainingProfiler",
    "profile_training_step",
    # Config
    "TrainingExperimentConfig",
    "DatasetConfig",
    "DataLoaderConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ProductionConfig",
    "resolve_inference_score_threshold",
    "resolve_inference_sliding_window_overlap_pixels",
    "resolve_preds_final_nms_iou_threshold",
    "effective_eval_metric_thresholds",
    "apply_inference_config_to_model",
    "CheckpointConfig",
    "AugmentationConfig",
    "LossConfig",
    # Grouped CE curriculum
    "GroupedCeSpec",
    "build_grouped_ce_spec",
    "configure_roi_grouped_ce",
    "grouped_ce_alpha_for_epoch",
]
