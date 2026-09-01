"""Training experiment configuration (JSON + strict dataclasses).

See docs/user-guide/configuration.md and configs/config.schema.json for every key.
"""

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
import json
import warnings

from oriented_det.utils.config import MUTED_KEY_PREFIX, load_config


def _field_names(dcls: type) -> set[str]:
    return {f.name for f in fields(dcls)}


def _strict_section(
    raw: Any,
    section: str,
    dcls: type,
) -> Dict[str, Any]:
    """Build kwargs for *dcls*; raise with unknown keys (no silent drops)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"Config section {section!r} must be a JSON object, got {type(raw).__name__!r}."
        )
    active_raw = {
        k: v
        for k, v in raw.items()
        if not (isinstance(k, str) and k.startswith(MUTED_KEY_PREFIX))
    }
    allowed = _field_names(dcls)
    extra = set(active_raw) - allowed
    if extra:
        raise ValueError(
            f"Unknown key(s) in config section {section!r}: {sorted(extra)}. "
            f"Valid keys: {sorted(allowed)}"
        )
    return {k: active_raw[k] for k in allowed if k in active_raw}


def _normalize_legacy_loss_type(config_dict: Dict[str, Any]) -> None:
    """Rewrite legacy loss_type=none to explicit cross_entropy when it resolves to CE."""
    loss = config_dict.get("loss")
    if not isinstance(loss, dict) or loss.get("loss_type") != "none":
        return

    model = config_dict.get("model")
    roi_loss_type = "cross_entropy"
    if isinstance(model, dict):
        roi_loss_type = model.get("roi_loss_type", roi_loss_type)

    if roi_loss_type == "cross_entropy":
        loss["loss_type"] = "cross_entropy"


# Removed from ModelConfig; GPU IoU sample count is now env-driven (see ops/README.md).
_LEGACY_MODEL_KEYS = frozenset({"gpu_oriented_iou_samples"})

_LEGACY_ROI_BOX_REG_KEYS = (
    "roi_box_reg_iou_weight",
    "roi_box_reg_iou_loss_type",
    "roi_box_reg_iou_schedule_epochs",
    "roi_box_reg_iou_schedule_values",
    "roi_box_reg_smooth_l1_aux_weight",
)
_NEW_ROI_BOX_REG_AUX_KEYS = (
    "roi_box_reg_aux_weight",
    "roi_box_reg_aux_loss_type",
    "roi_box_reg_aux_schedule_epochs",
    "roi_box_reg_aux_schedule_values",
)
_DECODED_REG_LOSS_TYPES = frozenset({"probiou", "riou", "kfiou"})


def _normalize_legacy_model_keys(config_dict: Dict[str, Any]) -> None:
    """Drop obsolete ``model.*`` keys so older experiment configs still load."""
    model = config_dict.get("model")
    if not isinstance(model, dict):
        return
    for key in _LEGACY_MODEL_KEYS:
        model.pop(key, None)


def _normalize_legacy_roi_box_reg_keys(config_dict: Dict[str, Any]) -> None:
    """Rewrite pre-rename ROI box-reg aux keys onto ``roi_box_reg_aux_*``.

    ``roi_box_reg_iou_weight`` was decoded aux when main is Smooth L1.
    ``roi_box_reg_smooth_l1_aux_weight`` was encoded aux when main is decoded.
    """
    model = config_dict.get("model")
    if not isinstance(model, dict):
        return
    present_legacy = [k for k in _LEGACY_ROI_BOX_REG_KEYS if k in model]
    if not present_legacy:
        return
    present_new = [k for k in _NEW_ROI_BOX_REG_AUX_KEYS if k in model]
    if present_new:
        raise ValueError(
            "Cannot mix legacy ROI box-reg keys "
            f"{present_legacy} with {present_new}. Use roi_box_reg_aux_weight / "
            "roi_box_reg_aux_loss_type only."
        )

    main = str(model.get("roi_box_reg_main_loss_type") or "smooth_l1").strip().lower()
    decoded_main = main in _DECODED_REG_LOSS_TYPES
    iou_w = model.pop("roi_box_reg_iou_weight", None)
    iou_type = model.pop("roi_box_reg_iou_loss_type", None)
    encoded_w = model.pop("roi_box_reg_smooth_l1_aux_weight", None)
    if "roi_box_reg_iou_schedule_epochs" in model:
        model["roi_box_reg_aux_schedule_epochs"] = model.pop(
            "roi_box_reg_iou_schedule_epochs"
        )
    if "roi_box_reg_iou_schedule_values" in model:
        model["roi_box_reg_aux_schedule_values"] = model.pop(
            "roi_box_reg_iou_schedule_values"
        )

    iou_w_f = float(iou_w) if iou_w is not None else 0.0
    encoded_w_f = float(encoded_w) if encoded_w is not None else 0.0
    if iou_w_f > 0.0 and encoded_w_f > 0.0:
        raise ValueError(
            "Legacy ROI box-reg configs cannot enable both "
            "roi_box_reg_iou_weight and roi_box_reg_smooth_l1_aux_weight."
        )

    if decoded_main:
        if iou_w_f > 0.0:
            raise ValueError(
                "roi_box_reg_iou_weight is decoded aux; when "
                "roi_box_reg_main_loss_type is decoded, use "
                "roi_box_reg_aux_weight with roi_box_reg_aux_loss_type "
                "'smooth_l1' (legacy: roi_box_reg_smooth_l1_aux_weight)."
            )
        if encoded_w is not None:
            model["roi_box_reg_aux_weight"] = encoded_w_f
            if encoded_w_f > 0.0:
                model["roi_box_reg_aux_loss_type"] = "smooth_l1"
    else:
        if encoded_w_f > 0.0:
            raise ValueError(
                "roi_box_reg_smooth_l1_aux_weight is encoded aux; when "
                "roi_box_reg_main_loss_type is smooth_l1, use "
                "roi_box_reg_aux_weight with a decoded "
                "roi_box_reg_aux_loss_type (legacy: roi_box_reg_iou_weight)."
            )
        if iou_w is not None:
            model["roi_box_reg_aux_weight"] = iou_w_f
        if iou_type is not None:
            model["roi_box_reg_aux_loss_type"] = iou_type

    warnings.warn(
        "Deprecated ROI box-reg keys "
        f"{present_legacy} were remapped to roi_box_reg_aux_weight / "
        "roi_box_reg_aux_loss_type. Update the config to the new names.",
        DeprecationWarning,
        stacklevel=2,
    )


def _normalize_legacy_cosine_t_max(config_dict: Dict[str, Any]) -> None:
    """Rewrite ``training.lr_scheduler_cosine_t_max`` onto ``lr_scheduler_cosine_epochs``."""
    training = config_dict.get("training")
    if not isinstance(training, dict) or "lr_scheduler_cosine_t_max" not in training:
        return
    t_max = training.pop("lr_scheduler_cosine_t_max")
    if t_max is None:
        return
    t_max_i = int(t_max)
    epochs = training.get("lr_scheduler_cosine_epochs")
    if epochs is not None and int(epochs) != t_max_i:
        raise ValueError(
            "Cannot set both lr_scheduler_cosine_t_max and "
            "lr_scheduler_cosine_epochs to different values. "
            "Use lr_scheduler_cosine_epochs only."
        )
    if epochs is None:
        training["lr_scheduler_cosine_epochs"] = t_max_i
    warnings.warn(
        "Deprecated training.lr_scheduler_cosine_t_max was remapped to "
        "lr_scheduler_cosine_epochs. Update the config to the new name.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_root: Path
    format: str = "dota"  # Options: "dota", "airbus_playground", "hrsc2016"
    train_tiles_dir: Optional[Path] = None
    val_tiles_dir: Optional[Path] = None
    # Optional list of tile roots (MMRotate trainval-style union without on-disk merge).
    train_tiles_dirs: Optional[List[Path]] = None
    val_tiles_dirs: Optional[List[Path]] = None
    same_folder: bool = False  # If True (DOTA), images and labels are in the same directory as train_tiles_dir/val_tiles_dir
    overlap: int = 16  # Tile overlap in pixels (even); deploy uses margin = overlap/2 when production.ignore_margin_pixels is null
    annotations_file: Optional[Path] = None
    split_file: Optional[Path] = None
    # Airbus Playground split.csv: integer fold id used as validation (default 0). Ignored for
    # legacy CSVs whose split column uses only the strings train/val.
    val_split_id: int = 0
    # Airbus Playground fold mode: when true, train split includes all folds (DOTA trainval parity);
    # val split still uses val_split_id for monitoring only.
    train_includes_val: bool = False
    ignore_labels: Optional[List[str]] = None
    # Extra aliases treated as hard-negative lookalikes. Reserved name "lookalike"
    # is always included even when this list is null/empty.
    lookalike_labels: Optional[List[str]] = None
    map_labels: Optional[Dict[str, str]] = None
    # Exact Playground tags (e.g. "Partially Hidden") that set DOTA difficult=1 and are
    # stripped from the semantic class_name at Airbus CSV load / generation time.
    # Not the same as ignore_labels (drop) or lookalike (hard negative).
    difficult_tags: Optional[List[str]] = None
    # How to handle DOTA "difficult" annotations (last field in label lines):
    # - "drop": remove difficult objects at read-time (fast; they never reach training/eval targets)
    # - "ignore": keep difficult objects but route them into ignore targets (MMRotate/MMDet-style)
    # - "keep": treat difficult objects as normal GT
    difficult_strategy: str = "drop"
    # Drop tiles with no GT after difficult_strategy / allowed_classes / ignore_labels (MMRotate parity).
    filter_empty_gt: bool = False
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    # When set with max_train_samples / max_val_samples: pick indices via deterministic shuffle
    # instead of the first N in dataset order (see capped_subset_indices in train.utils).
    max_samples_shuffle_seed: Optional[int] = None
    allowed_classes: Optional[List[str]] = None
    # HRSC2016 / FAIR1M ImageSets name used for the training-loop train/val roles.
    # HRSC defaults (when null): train → trainval, val → test (MMRotate).
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    # Optional CSV from tools/save_predictions (--save-tile-metrics-csv); join on image_id / stem
    tile_metrics_csv: Optional[Path] = None
    hard_tile_metric_column: str = "f1"
    hard_tile_threshold: float = 0.8
    hard_tile_oversample_factor: float = 2.0
    # When tile_metrics_csv is set: drop vacuous true-negative tiles (tp=fp=fn=0) from train
    # before max_train_samples / hard-tile oversampling. Empty tiles with FPs stay and can
    # be oversampled. Requires filter_empty_gt=false in the loader to keep those hard empties.
    drop_easy_empty_tiles: bool = False

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_root = Path(self.data_root)
        if self.train_tiles_dir is not None:
            self.train_tiles_dir = Path(self.train_tiles_dir)
        if self.val_tiles_dir is not None:
            self.val_tiles_dir = Path(self.val_tiles_dir)
        if self.train_tiles_dirs is not None:
            self.train_tiles_dirs = [Path(p) for p in self.train_tiles_dirs]
        if self.val_tiles_dirs is not None:
            self.val_tiles_dirs = [Path(p) for p in self.val_tiles_dirs]
        if self.annotations_file is not None:
            self.annotations_file = Path(self.annotations_file)
        if self.split_file is not None:
            self.split_file = Path(self.split_file)
        if self.tile_metrics_csv is not None:
            self.tile_metrics_csv = Path(self.tile_metrics_csv)

    def has_dota_tiles_config(self) -> bool:
        """True when train and val tile root(s) are configured."""
        train_ok = bool(self.train_tiles_dirs) or self.train_tiles_dir is not None
        val_ok = bool(self.val_tiles_dirs) or self.val_tiles_dir is not None
        return train_ok and val_ok

    def get_train_tile_roots(self) -> List[Path]:
        from ..data.dota import resolve_dota_tile_roots

        return resolve_dota_tile_roots(
            tiles_dirs=self.train_tiles_dirs,
            tiles_dir=self.train_tiles_dir,
            split_label="train",
        )

    def get_val_tile_roots(self) -> List[Path]:
        from ..data.dota import resolve_dota_tile_roots

        return resolve_dota_tile_roots(
            tiles_dirs=self.val_tiles_dirs,
            tiles_dir=self.val_tiles_dir,
            split_label="val",
        )


@dataclass
class AugmentationConfig:
    """Augmentation configuration. Keys ordered: limits first, then p_* (probabilities)."""
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gamma_limit: tuple[int, int] = (80, 120)
    gauss_noise_var_limit: tuple[float, float] = (10.0, 50.0)
    blur_limit: int = 3
    clahe_clip_limit: float = 4.0
    p_brightness_contrast: float = 0.5
    p_gamma: float = 0.3
    p_noise: float = 0.2
    p_blur: float = 0.2
    p_clahe: float = 0.3


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: int = 16
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model configuration. Keys ordered by prefix: backbone, fpn_, anchor_, target_, roi_*, rpn_*, final_nms_*, etc."""
    # Backbone
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    trainable_layers: int = 5
    frozen_stages: Optional[int] = None
    # FPN
    fpn_returned_layers: Optional[List[int]] = None
    # Nominal strides (ROI init / logging); RPN/ROI forward uses strides derived from image_size and feature maps
    fpn_strides: Optional[List[int]] = None
    fpn_extra_level: bool = False
    # Anchors
    anchor_scales: List[int] = field(default_factory=lambda: [8, 16, 32])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_octave_base_scale: Optional[float] = None
    anchor_scales_per_octave: Optional[int] = None
    # Box regression targets [dx, dy, dw, dh, dangle]
    target_means: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    target_stds: tuple[float, float, float, float, float] = (0.1, 0.1, 0.2, 0.2, 0.1)
    # ROI head (switch + params)
    roi_loss_type: str = "cross_entropy"
    roi_focal_alpha: float = 1.0
    roi_focal_gamma: float = 2.0
    roi_norm_factor: Optional[float] = 2.0
    roi_edge_swap: bool = True
    roi_proj_xy: bool = False
    roi_box_reg_angle_weight: float = 1.0
    roi_box_reg_angle_schedule_epochs: Optional[List[int]] = None
    roi_box_reg_angle_schedule_values: Optional[List[float]] = None
    roi_box_reg_aux_weight: float = 0.0
    roi_box_reg_aux_schedule_epochs: Optional[List[int]] = None
    roi_box_reg_aux_schedule_values: Optional[List[float]] = None
    roi_batch_size_per_image: int = 512
    roi_positive_iou_threshold: float = 0.5
    roi_negative_iou_threshold: float = 0.5
    roi_match_low_quality: bool = False
    roi_min_pos_iou: float = 0.5
    roi_box_reg_aux_loss_type: Optional[str] = None
    roi_box_reg_kfiou_fun: Optional[str] = None
    roi_box_reg_probiou_mode: Optional[str] = None
    roi_box_reg_main_loss_type: str = "smooth_l1"
    roi_box_reg_norm: str = "sampled_all"
    # RetinaNet head / regression
    retinanet_stacked_convs: int = 4
    box_reg_loss_type: str = "smooth_l1"
    box_reg_weight: float = 1.0
    # Rotated FCOS
    fcos_stacked_convs: int = 4
    fcos_center_sampling: bool = True
    fcos_center_sample_radius: float = 1.5
    fcos_norm_on_bbox: bool = True
    fcos_centerness_on_reg: bool = True
    fcos_scale_angle: bool = True
    fcos_regress_ranges: Optional[List[List[float]]] = None
    fcos_angle_weight: float = 1.0
    fcos_nms_pre: int = 2000
    aux_loss_type: Optional[str] = None
    aux_loss_weight: float = 0.0
    aux_angle_weight: float = 1.0
    aux_angle_lambda: float = 1.0
    # RPN
    rpn_min_size: float = 0.0
    rpn_pre_nms_top_n: int = 2000
    rpn_post_nms_top_n: int = 1000
    rpn_nms_threshold: float = 0.7
    rpn_batch_size_per_image: int = 256
    rpn_positive_iou_threshold: float = 0.7
    rpn_negative_iou_threshold: float = 0.3
    rpn_min_pos_iou: float = 0.3
    rpn_match_low_quality: bool = True
    # Matching / proposals
    use_hbb_for_matching: bool = True
    roi_use_hbb_for_matching: bool = False
    add_gt_as_proposals: bool = True
    # NMS and inference (after ROI head; distinct from rpn_nms_threshold)
    final_nms_iou_threshold: float = 0.5
    # If True, final ROI NMS runs on all classes together (reduces overlapping car/truck duplicates).
    nms_class_agnostic: bool = False
    # If True, post-ROI final oriented NMS uses exact polygon IoU on CPU (ignores GPU sampling NMS).
    final_nms_use_cpu: bool = False
    final_nms_iou_schedule_epochs: Optional[List[int]] = None
    final_nms_iou_schedule_values: Optional[List[float]] = None
    max_detections_per_image: int = 100
    inference_pre_nms_score_threshold: float = 0.05
    # If True, ROI inference uses argmax foreground class per proposal (MMRotate-style); if False,
    # keep every foreground class above inference_pre_nms_score_threshold (helps weak classifiers).
    roi_inference_top_class_only: bool = False

    def __post_init__(self) -> None:
        aux_w = float(self.roi_box_reg_aux_weight)
        aux_t = self.roi_box_reg_aux_loss_type
        if aux_t is not None:
            aux_t = str(aux_t).strip().lower()
            self.roi_box_reg_aux_loss_type = aux_t or None
            aux_t = self.roi_box_reg_aux_loss_type
        main_t = str(self.roi_box_reg_main_loss_type or "smooth_l1").strip().lower()
        self.roi_box_reg_main_loss_type = main_t
        allowed_aux = {"smooth_l1", "probiou", "riou", "kfiou"}
        if aux_w > 0.0:
            if not aux_t:
                raise ValueError(
                    "roi_box_reg_aux_weight > 0 requires roi_box_reg_aux_loss_type "
                    "(smooth_l1|probiou|riou|kfiou)."
                )
            if aux_t not in allowed_aux:
                raise ValueError(
                    f"roi_box_reg_aux_loss_type must be one of {sorted(allowed_aux)}, got {aux_t!r}."
                )
            if aux_t == main_t:
                raise ValueError(
                    "roi_box_reg_aux_loss_type must differ from roi_box_reg_main_loss_type "
                    f"(both {main_t!r})."
                )
            if main_t == "smooth_l1" and aux_t not in _DECODED_REG_LOSS_TYPES:
                raise ValueError(
                    "When roi_box_reg_main_loss_type is smooth_l1, "
                    "roi_box_reg_aux_loss_type must be probiou, riou, or kfiou."
                )
            if main_t in _DECODED_REG_LOSS_TYPES and aux_t != "smooth_l1":
                raise ValueError(
                    "When roi_box_reg_main_loss_type is decoded "
                    "(probiou/riou/kfiou), roi_box_reg_aux_loss_type must be smooth_l1."
                )
        elif aux_t is not None and aux_t not in allowed_aux:
            raise ValueError(
                f"roi_box_reg_aux_loss_type must be one of {sorted(allowed_aux)}, got {aux_t!r}."
            )


@dataclass
class TrainingConfig:
    """Training hyperparameters. Keys ordered: scheduler switch first, then lr_*, then rest."""
    # LR scheduler: switch first, then all lr_* grouped
    lr_scheduler_type: Optional[str] = None  # None/"multistep"/"step" | "reduce_on_plateau" | "one_cycle" | "cosine_annealing" | "cosine_annealing_with_tail"
    lr_scheduler_step_epochs: int = 8
    lr_scheduler_milestones: Optional[List[int]] = None
    # Scalar: same factor at every MultiStepLR/StepLR drop. List: one factor per milestone.
    lr_scheduler_gamma: Any = 0.1
    lr_scheduler_plateau_metric: str = "total_loss"
    lr_scheduler_plateau_factor: float = 0.1
    lr_scheduler_plateau_patience: int = 5
    lr_scheduler_one_cycle_pct_start: float = 0.3
    lr_scheduler_one_cycle_div_factor: float = 25.0
    lr_scheduler_one_cycle_final_div_factor: float = 1e4
    lr_scheduler_cosine_eta_min: float = 1e-6
    # Cosine phase length (epochs). If set with num_epochs > this value, remaining epochs use a
    # fixed LR tail (see lr_scheduler_cosine_tail_lr). Alternative: set lr_scheduler_cosine_tail_epochs only.
    # Legacy JSON key lr_scheduler_cosine_t_max is remapped onto this field on load.
    lr_scheduler_cosine_epochs: Optional[int] = None
    lr_scheduler_cosine_tail_epochs: int = 0
    lr_scheduler_cosine_tail_lr: Optional[float] = None  # default: lr_scheduler_cosine_eta_min
    lr_warmup_steps: int = 500
    lr_scaling_with_accumulation: str = "linear"
    lr_scale_with_world_size: bool = False
    use_lr_param_groups: bool = False
    lr_mult_backbone: float = 0.5
    lr_mult_rpn: float = 0.5
    lr_mult_roi: float = 2.0
    lr_mult_other: float = 1.0
    # When set: applies to RetinaNet ``head.*``; for two-stage models, the same value applies to
    # ``rpn_head.*`` and ``roi_head.*`` (overrides lr_mult_rpn / lr_mult_roi for those groups).
    # When null: ``head.*`` uses lr_mult_other; RPN/ROI use lr_mult_rpn / lr_mult_roi.
    lr_mult_head: Optional[float] = None
    # Other training
    num_epochs: int = 20
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    loss_weights: Optional[Dict[str, float]] = None
    # Optional partial freeze (0-based training-loop epoch index ``epoch``; same as checkpoint epoch).
    # While ``epoch < N``, the corresponding module does not train (``requires_grad=False``). ROI head
    # always remains trainable. RPN freeze is ignored when the model has no ``rpn_head`` (e.g. RetinaNet).
    freeze_backbone_epochs: int = 0
    freeze_rpn_epochs: int = 0
    # Early stopping (optional). When early_stop_patience is set, stop if monitored metric does not improve.
    early_stop_patience: Optional[int] = None
    early_stop_metric: str = "mAP"
    early_stop_min_delta: float = 0.0
    early_stop_higher_is_better: Optional[bool] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    score_threshold: float = 0.5  # Minimum confidence score for detections
    # Optional class_name -> min score; classes not listed use score_threshold
    per_class_score_threshold: Optional[Dict[str, float]] = None
    iou_threshold: float = 0.5  # IoU threshold for mAP calculation
    # Global score floor for ``odet preds`` / ``make eval-val`` only (published protocol 0.05).
    # When null, those tools use 0.05. Training val uses ``score_threshold`` (often 0.3).
    # Deploy / image_demo use ``production.score_threshold`` (may be 0.3).
    preds_score_threshold: Optional[float] = None
    # Final detection NMS IoU for ``odet preds`` / ``make eval-val`` only (MMRotate test parity).
    # When set, overrides ``production.final_nms_iou_threshold`` for that path. Deploy / image_demo
    # still use ``production.*`` via ``apply_inference_config_to_model``. Training val uses ``model.*``.
    final_nms_iou_threshold: Optional[float] = None
    compute_map_final: bool = True  # After training, load best checkpoint and compute mAP once
    compute_map_every_n_epochs: int = 0  # If >0, compute mAP every N epochs during training (current model)
    # If True, compute expensive GT–IoU histograms (mean/median/buckets, wrong-class counts)
    # and class-agnostic duplicate rate on post-threshold detections (greedy by score vs eval IoU).
    extended_gt_metrics: bool = False
    # mAP matching and GT-cover / accuracy use exact CPU polygon IoU when True (Shapely when
    # installed). When False, use GPU sampling IoU (faster, approximate). Does not affect final
    # detection NMS or production/deploy decode settings.
    use_exact_rotated_iou: bool = True
    # IoU backend for compute_map_final only (best-checkpoint mAP after training). When None,
    # falls back to use_exact_rotated_iou. Set true for publishable exact mAP while keeping
    # periodic in-training mAP on GPU sampling (use_exact_rotated_iou=false).
    use_exact_rotated_iou_for_final_map: Optional[bool] = None


@dataclass
class ProductionConfig:
    """Production overrides (deploy, save_predictions, image_demo).

    ``None`` on a field means: use the same source as before this block existed
    (``evaluation`` / ``model`` / ``dataset``), or deploy defaults for canvas flags (see schema).

    ``final_nms_iou_threshold`` / ``final_nms_use_cpu`` mirror ``model.*`` when set here;
    decode patches apply only when ``apply_inference_config_to_model`` runs on a loaded checkpoint
    (not during ``tools/train.py``).
    """

    score_threshold: Optional[float] = None
    per_class_score_threshold: Optional[Dict[str, float]] = None
    final_nms_iou_threshold: Optional[float] = None
    final_nms_use_cpu: Optional[bool] = None
    inference_pre_nms_score_threshold: Optional[float] = None
    max_detections_per_image: Optional[int] = None
    nms_class_agnostic: Optional[bool] = None
    roi_inference_top_class_only: Optional[bool] = None
    rpn_pre_nms_top_n: Optional[int] = None
    rpn_post_nms_top_n: Optional[int] = None
    rpn_nms_threshold: Optional[float] = None
    # Sliding-window overlap along each axis (pixels). None = 200 (same default as oriented_det.runtime.inference).
    overlap_pixels: Optional[int] = None
    # Interior margin (px): drop detections whose centroid lies in the outer band. None = dataset.overlap/2 (0 = off).
    ignore_margin_pixels: Optional[float] = None
    # Deploy: expand square canvas from first image (fixed resize). None = false.
    use_first_image_canvas: Optional[bool] = None
    # Deploy: always use config target_size / sliding window. None = true.
    stick_to_model_canvas: Optional[bool] = None


def resolve_inference_sliding_window_overlap_pixels(config: "TrainingExperimentConfig") -> int:
    """Sliding-window overlap in pixels per axis for production inference (ratio-free).

    Returns:
        ``production.overlap_pixels`` when set (clamped to ``>= 0``), else **200** (same default as
        :func:`oriented_det.runtime.inference.run_inference_auto`).
    """
    inf = getattr(config, "production", None)
    if inf is not None and getattr(inf, "overlap_pixels", None) is not None:
        return max(0, int(inf.overlap_pixels))
    return 200


def resolve_inference_score_threshold(config: "TrainingExperimentConfig") -> float:
    if getattr(config, "production", None) is not None:
        v = getattr(config.production, "score_threshold", None)
        if v is not None:
            return float(v)
    return float(getattr(config.evaluation, "score_threshold", 0.5))


def resolve_preds_score_threshold(
    config: "TrainingExperimentConfig",
    *,
    cli_score_threshold: Optional[float] = None,
) -> Tuple[float, str]:
    """Global score floor for ``odet preds`` / ``make eval-val``.

    Priority:
    1. CLI ``--score-threshold``
    2. ``evaluation.preds_score_threshold`` when set
    3. ``0.05`` (published eval-val protocol)

    Ignores ``production.score_threshold`` and ``evaluation.score_threshold`` so
    deploy display (often 0.3) and fast train-val (often 0.3) cannot raise the
    published mAP floor. Deploy / ``image_demo`` keep
    :func:`resolve_inference_score_threshold`. Training val keeps
    :func:`effective_eval_metric_thresholds`.
    """
    if cli_score_threshold is not None:
        return float(cli_score_threshold), "CLI --score-threshold"
    ev = getattr(config, "evaluation", None)
    if ev is not None and getattr(ev, "preds_score_threshold", None) is not None:
        return (
            float(ev.preds_score_threshold),
            "evaluation.preds_score_threshold (eval-val / odet preds)",
        )
    return 0.05, "default 0.05 (eval-val / odet preds)"


def resolve_preds_final_nms_iou_threshold(
    config: "TrainingExperimentConfig",
    *,
    cli_nms_threshold: Optional[float] = None,
    model_nms_threshold: Optional[float] = None,
) -> Tuple[float, str]:
    """Final detection NMS IoU for ``odet preds`` / ``make eval-val``.

    Priority:
    1. CLI ``--nms-threshold``
    2. ``evaluation.final_nms_iou_threshold`` (published MMRotate-parity protocol)
    3. ``model_nms_threshold`` (live model attr after ``production`` patch, if provided)
    4. ``production.final_nms_iou_threshold`` then ``model.final_nms_iou_threshold``
    5. ``0.5``

    Deploy / ``image_demo`` do not use this helper; they keep ``production.*`` via
    :func:`apply_inference_config_to_model`.
    """
    if cli_nms_threshold is not None:
        return float(cli_nms_threshold), "CLI --nms-threshold"
    ev = getattr(config, "evaluation", None)
    if ev is not None and getattr(ev, "final_nms_iou_threshold", None) is not None:
        return (
            float(ev.final_nms_iou_threshold),
            "evaluation.final_nms_iou_threshold (eval-val / odet preds)",
        )
    if model_nms_threshold is not None:
        inf = getattr(config, "production", None)
        if inf is not None and getattr(inf, "final_nms_iou_threshold", None) is not None:
            return (
                float(model_nms_threshold),
                "production.final_nms_iou_threshold (patched onto model)",
            )
        return float(model_nms_threshold), "model.final_nms_iou_threshold"
    inf = getattr(config, "production", None)
    if inf is not None and getattr(inf, "final_nms_iou_threshold", None) is not None:
        return float(inf.final_nms_iou_threshold), "production.final_nms_iou_threshold"
    m = getattr(config, "model", None)
    if m is not None and getattr(m, "final_nms_iou_threshold", None) is not None:
        return float(m.final_nms_iou_threshold), "model.final_nms_iou_threshold"
    return 0.5, "default 0.5"


def effective_eval_metric_thresholds(
    config: "TrainingExperimentConfig",
) -> Tuple[float, Optional[Dict[str, float]], float]:
    """Score / per-class / IoU thresholds passed to validation and mAP.

    - ``score_threshold``: ``production.score_threshold`` when set, else ``evaluation.score_threshold``.
    - ``iou_threshold``: always ``evaluation.iou_threshold`` (mAP / val matching).
    - ``per_class_score_threshold``: start from ``evaluation`` entries, then ``production`` updates keys.
    """
    ev = config.evaluation
    sc = float(ev.score_threshold)
    pc: Optional[Dict[str, float]] = None
    if ev.per_class_score_threshold:
        pc = {str(k): float(v) for k, v in ev.per_class_score_threshold.items()}
    iou = float(ev.iou_threshold)
    inf = getattr(config, "production", None)
    if inf is None:
        return sc, pc, iou
    if inf.score_threshold is not None:
        sc = float(inf.score_threshold)
    if inf.per_class_score_threshold is not None:
        merged: Dict[str, float] = dict(pc) if pc else {}
        merged.update({str(k): float(v) for k, v in inf.per_class_score_threshold.items()})
        pc = merged if merged else None
    return sc, pc, iou


def config_use_exact_rotated_iou_for_map(config: "TrainingExperimentConfig") -> bool:
    """Whether periodic mAP and GT-cover matching use exact CPU polygon IoU.

    Reads ``evaluation.use_exact_rotated_iou`` only. Final detection NMS and
    ``production.*`` decode settings are unrelated.
    """
    evaluation = getattr(config, "evaluation", None)
    if evaluation is None:
        return True
    return bool(getattr(evaluation, "use_exact_rotated_iou", True))


def config_use_exact_rotated_iou_for_final_map(config: "TrainingExperimentConfig") -> bool:
    """Whether the post-training ``compute_map_final`` pass uses exact CPU polygon IoU.

    Reads ``evaluation.use_exact_rotated_iou_for_final_map`` when set; otherwise
    ``evaluation.use_exact_rotated_iou``. Does not affect NMS or production decode.
    """
    evaluation = getattr(config, "evaluation", None)
    if evaluation is None:
        return True
    override = getattr(evaluation, "use_exact_rotated_iou_for_final_map", None)
    if override is not None:
        return bool(override)
    return config_use_exact_rotated_iou_for_map(config)


def apply_inference_config_to_model(model: Any, inf: Optional[ProductionConfig]) -> None:
    """Patch decode / NMS-related attributes on ``model`` from non-null ``production.*`` fields.

    Call only for **checkpoint-based inference** outside the training loop — e.g.
    ``tools/save_predictions.load_model_from_checkpoint`` (deploy, ``image_demo``, ``test_single_image``).
    **Not** used by ``tools/train.py``: during training the live module keeps ``model.*`` only so
    deploy-oriented RPN/NMS overrides cannot slow or alter the train forward pass.

    Attributes that do not exist on ``model`` (e.g. RPN keys on RetinaNet) are skipped.
    """
    if inf is None:
        return

    def _set(name: str, value: Any) -> None:
        if value is None:
            return
        if hasattr(model, name):
            setattr(model, name, value)

    _set("inference_pre_nms_score_threshold", inf.inference_pre_nms_score_threshold)
    if inf.inference_pre_nms_score_threshold is not None and not hasattr(
        model, "inference_pre_nms_score_threshold"
    ):
        _set("score_threshold", inf.inference_pre_nms_score_threshold)

    _set("final_nms_iou_threshold", inf.final_nms_iou_threshold)
    _set("final_nms_use_cpu", inf.final_nms_use_cpu)
    _set("max_detections_per_image", inf.max_detections_per_image)
    _set("nms_class_agnostic", inf.nms_class_agnostic)
    _set("roi_inference_top_class_only", inf.roi_inference_top_class_only)
    _set("rpn_pre_nms_top_n", inf.rpn_pre_nms_top_n)
    _set("rpn_post_nms_top_n", inf.rpn_post_nms_top_n)
    _set("rpn_nms_threshold", inf.rpn_nms_threshold)


@dataclass
class CheckpointConfig:
    """Checkpoint loading and best-model selection configuration."""
    load_from_checkpoint: Optional[Path] = None
    load_from_experiment: Optional[Path] = None
    # When True and load_from_experiment is still unset: use newest other run under runs/<model_type>/.
    # When False: never auto-fill experiment dir (null checkpoint + null experiment = no weights loaded).
    discover_previous_run: bool = False
    # True: prefer latest checkpoint_epoch_*.pth, restore optimizer/scheduler per load_* flags, continue epochs.
    # False: prefer best_* checkpoint, typically epoch 0 with fresh optimizer unless config says otherwise.
    resume_from_checkpoint_epoch: bool = True
    load_optimizer_state: bool = True
    # When True (default), restore LR scheduler from checkpoint on resume. Set False to rebuild
    # cosine rebuild on resume when load_scheduler_state=false (see tools/train.py).
    load_scheduler_state: bool = True
    load_include_prefixes: Optional[List[str]] = None  # Load only keys whose names start with these prefixes
    load_exclude_prefixes: Optional[List[str]] = None  # Drop keys whose names start with these prefixes
    start_epoch: int = 0
    # Best checkpoint: metric name used to decide which checkpoint to keep (e.g. "mAP" or "total_loss")
    best_metric: Optional[str] = None  # Default in train: "total_loss"; use "mAP" to save at best validation mAP
    higher_is_better: Optional[bool] = None  # True for mAP, False for total_loss; if None, inferred from best_metric


@dataclass
class LossConfig:
    """Experiment-level ROI loss selection and optional class weighting.

    loss_type chooses the recipe applied by tools/train.py:
    - cross_entropy: plain ROI CE (MMRotate-style)
    - class_weighted: ROI CE with dataset-derived class weights
    - focal: unweighted focal ROI loss
    - focal_weighted: focal ROI loss with dataset-derived class weights
    - none: legacy fallback to model.roi_loss_type
    """
    loss_type: str = "class_weighted"
    class_weight_method: str = "sqrt"
    class_weight_beta: float = 0.9999  # Used by effective_num weighting
    # Optional schedule to ramp class weights in over epochs (e.g. uniform -> inv_freq).
    class_weight_schedule_type: Optional[str] = None  # None | "linear_ramp"
    class_weight_schedule_start_epoch: int = 0
    class_weight_schedule_end_epoch: int = 0
    class_weight_schedule_power: float = 1.0  # 1.0 = linear; >1 eases in
    background_weight: Optional[float] = None
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    class_weight_overrides: Optional[Dict[str, float]] = None  # e.g. {"truck": 0.25} to reduce dominant-class bias
    label_smoothing: float = 0.0  # Label smoothing for ROI classification (e.g. 0.1 to reduce overconfidence)
    # Coarse-to-fine ROI classifier curriculum (single-run alternative to multi-stage training).
    roi_grouped_ce_enabled: bool = False
    roi_grouped_ce_groups: Optional[Dict[str, List[str]]] = None  # e.g. {"plane": ["Bomber Aircraft", ...]}
    roi_grouped_ce_schedule_type: Optional[str] = None  # None | "step" | "linear_ramp"
    roi_grouped_ce_schedule_start_epoch: int = 0
    roi_grouped_ce_schedule_end_epoch: int = 8  # step: first fine epoch; ramp: last grouped-heavy epoch
    roi_grouped_ce_schedule_power: float = 1.0


@dataclass
class TensorboardConfig:
    """TensorBoard and validation logging options."""
    log_debug_anchors_proposals: bool = False  # Log val/debug_anchors and val/debug_proposals
    vis_score_threshold: Optional[float] = None  # Min score for boxes in TB prediction images; None = use evaluation.score_threshold


# Default normalization: MMDetection/MMRotate (on [0,1] scale). Inference must use same as training.
PREPROCESSING_DEFAULT_MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]  # RGB
PREPROCESSING_DEFAULT_STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]  # RGB


@dataclass
class PreprocessingConfig:
    """Preprocessing. Keys ordered: resize_mode (switch) first, then target_size, normalize_*, pad_*, enable_flip_*, enable_random_rotate_*.

    resize_mode: ``fixed`` (stretch to target_size), ``pad`` (scale by large edge + pad to
    target_size), ``keep_ratio`` (scale by large edge; then ``pad_size_divisor``, MMRotate
    style), or ``crop`` (native-res crop/pad to target_size).
    """
    resize_mode: str = "fixed"
    target_size: Any = field(default_factory=lambda: [1024, 1024])
    normalize_mean: List[float] = field(default_factory=lambda: list(PREPROCESSING_DEFAULT_MEAN))
    normalize_std: List[float] = field(default_factory=lambda: list(PREPROCESSING_DEFAULT_STD))
    pad_size_divisor: int = 32
    enable_flip_horizontal: bool = True
    enable_flip_vertical: bool = True
    enable_flip_diagonal: bool = False
    enable_random_rotate: bool = False
    random_rotate_prob: float = 0.5
    random_rotate_angle_range: float = 180.0


def get_preprocessing_params(config: Any) -> Dict[str, Any]:
    """Extract preprocessing params from an experiment config for use in inference.
    Returns dict with: resize_mode, target_size ((h, w) tuple), normalize_mean, normalize_std,
    pad_size_divisor, enable_flip_horizontal, enable_flip_vertical, enable_flip_diagonal.
    """
    prep = getattr(config, "preprocessing", None)
    if prep is None:
        return {
            "resize_mode": "fixed",
            "target_size": (1024, 1024),
            "normalize_mean": list(PREPROCESSING_DEFAULT_MEAN),
            "normalize_std": list(PREPROCESSING_DEFAULT_STD),
            "pad_size_divisor": 32,
            "enable_flip_horizontal": True,
            "enable_flip_vertical": True,
            "enable_flip_diagonal": False,
        }
    from oriented_det.data.preprocessing import parse_canvas_size

    mode = getattr(prep, "resize_mode", "fixed")
    ts = getattr(prep, "target_size", [1024, 1024])
    target_size = parse_canvas_size(mode, ts)
    mean = list(getattr(prep, "normalize_mean", PREPROCESSING_DEFAULT_MEAN))
    std = list(getattr(prep, "normalize_std", PREPROCESSING_DEFAULT_STD))
    return {
        "resize_mode": mode,
        "target_size": target_size,
        "normalize_mean": mean if mean else list(PREPROCESSING_DEFAULT_MEAN),
        "normalize_std": std if std else list(PREPROCESSING_DEFAULT_STD),
        "pad_size_divisor": getattr(prep, "pad_size_divisor", 32),
        "enable_flip_horizontal": getattr(prep, "enable_flip_horizontal", True),
        "enable_flip_vertical": getattr(prep, "enable_flip_vertical", True),
        "enable_flip_diagonal": getattr(prep, "enable_flip_diagonal", False),
    }


@dataclass
class TrainingExperimentConfig:
    """Complete training experiment configuration."""
    # Experiment metadata
    model_type: str = "oriented_rcnn"
    experiment_timestamp: Optional[str] = None
    source_recipe: Optional[str] = None
    # Set at train start (not in hand-written recipes): framework git + package stamp
    source_code_root: Optional[str] = None
    git_commit: Optional[str] = None
    git_describe: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_branch: Optional[str] = None
    git_commit_date: Optional[str] = None
    package_version: Optional[str] = None
    # Switches first
    enable_albumentation: bool = False
    enable_profiling: bool = False
    # Sub-configs (grouped by section)
    dataset: Optional[DatasetConfig] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    # Class information (saved for inference)
    class_map: Optional[Dict[str, int]] = field(default=None, repr=False)
    class_names: Optional[List[str]] = field(default=None, repr=False)
    num_classes: Optional[int] = field(default=None, repr=False)  # Foreground classes only (model output = num_classes + 1 for background)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, including class information."""
        config_dict = asdict(self)
        _normalize_legacy_loss_type(config_dict)
        # Convert Path objects to strings
        return self._convert_paths_to_strings(config_dict)
    
    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_paths_to_strings(item) for item in obj]
        else:
            return obj
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Path to save the configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingExperimentConfig":
        """Load configuration from JSON file with support for _base_ inheritance.
        
        Supports MMRotate-style nested config inheritance. Base configs are loaded
        and merged before converting to dataclass instances.
        
        Args:
            path: Path to the configuration file (can use _base_ field)
            
        Returns:
            TrainingExperimentConfig instance
            
        Examples:
            >>> # Simple config
            >>> config = TrainingExperimentConfig.load(Path("config.json"))
            
            >>> # Config with base inheritance
            >>> # config.json: {"_base_": "../_base_/datasets/dota.json", ...}
            >>> config = TrainingExperimentConfig.load(Path("config.json"))
        """
        path = Path(path)
        
        # Use enhanced config loader that supports _base_ inheritance
        frozen_cfg = load_config(path)
        config_dict = frozen_cfg.to_dict()
        _normalize_legacy_loss_type(config_dict)
        _normalize_legacy_model_keys(config_dict)
        _normalize_legacy_roi_box_reg_keys(config_dict)
        _normalize_legacy_cosine_t_max(config_dict)

        # JSON `null` for a section should fall back to dataclass defaults, not be passed
        # as `None` into non-optional sub-configs.
        for _k in (
            "augmentation", "data_loader", "model", "training", "evaluation",
            "production", "checkpoint", "loss", "tensorboard", "preprocessing", "dataset",
        ):
            if _k in config_dict and config_dict[_k] is None:
                del config_dict[_k]

        valid_root = {f for f in cls.__dataclass_fields__}
        root_extra = set(config_dict) - valid_root
        if root_extra:
            raise ValueError(
                f"Unknown key(s) at config root: {sorted(root_extra)}. "
                f"Valid top-level keys: {sorted(valid_root)}"
            )

        # Convert nested dicts to dataclass instances (strict: no unknown keys per section)
        if "dataset" in config_dict and config_dict["dataset"] is not None:
            config_dict["dataset"] = DatasetConfig(
                **_strict_section(config_dict["dataset"], "dataset", DatasetConfig)
            )
        if "augmentation" in config_dict and config_dict["augmentation"] is not None:
            config_dict["augmentation"] = AugmentationConfig(
                **_strict_section(config_dict["augmentation"], "augmentation", AugmentationConfig)
            )
        if "data_loader" in config_dict and config_dict["data_loader"] is not None:
            config_dict["data_loader"] = DataLoaderConfig(
                **_strict_section(config_dict["data_loader"], "data_loader", DataLoaderConfig)
            )
        if "model" in config_dict and config_dict["model"] is not None:
            config_dict["model"] = ModelConfig(
                **_strict_section(config_dict["model"], "model", ModelConfig)
            )
        if "training" in config_dict and config_dict["training"] is not None:
            config_dict["training"] = TrainingConfig(
                **_strict_section(config_dict["training"], "training", TrainingConfig)
            )
        if "evaluation" in config_dict and config_dict["evaluation"] is not None:
            config_dict["evaluation"] = EvaluationConfig(
                **_strict_section(config_dict["evaluation"], "evaluation", EvaluationConfig)
            )
        if "production" in config_dict and config_dict["production"] is not None:
            config_dict["production"] = ProductionConfig(
                **_strict_section(config_dict["production"], "production", ProductionConfig)
            )
        if "checkpoint" in config_dict and config_dict["checkpoint"] is not None:
            config_dict["checkpoint"] = CheckpointConfig(
                **_strict_section(config_dict["checkpoint"], "checkpoint", CheckpointConfig)
            )
        if "loss" in config_dict and config_dict["loss"] is not None:
            config_dict["loss"] = LossConfig(
                **_strict_section(config_dict["loss"], "loss", LossConfig)
            )
        if "tensorboard" in config_dict and config_dict["tensorboard"] is not None:
            config_dict["tensorboard"] = TensorboardConfig(
                **_strict_section(config_dict["tensorboard"], "tensorboard", TensorboardConfig)
            )
        if "preprocessing" in config_dict and config_dict["preprocessing"] is not None:
            config_dict["preprocessing"] = PreprocessingConfig(
                **_strict_section(config_dict["preprocessing"], "preprocessing", PreprocessingConfig)
            )

        return cls(**config_dict)
    
    def print_summary(self) -> None:
        """Print a summary of the configuration."""
        print("Training Configuration Summary:")
        print("=" * 60)
        print(f"Model Type: {self.model_type}")
        if self.experiment_timestamp:
            print(f"Experiment Timestamp: {self.experiment_timestamp}")
        if self.git_commit:
            dirty = " dirty" if self.git_dirty else ""
            label = self.git_describe or self.git_commit[:12]
            print(f"Git: {label}{dirty}")
            if self.git_branch:
                print(f"Git branch: {self.git_branch}")
            if self.git_commit_date:
                print(f"Git commit date: {self.git_commit_date}")
        if self.package_version:
            print(f"Package: oriented-det {self.package_version}")
        print()
        
        if self.dataset:
            print("Dataset:")
            print(f"  Data Root: {self.dataset.data_root}")
            print(f"  Format: {self.dataset.format}")
            if self.dataset.format == "airbus_playground":
                print(f"  Annotations File: {self.dataset.annotations_file}")
                print(f"  Split File: {self.dataset.split_file}")
                print(f"  Val Split Id: {self.dataset.val_split_id}")
                if getattr(self.dataset, "train_includes_val", False):
                    print("  Train Includes Val: true (all folds in train; val fold for monitoring only)")
                print(f"  Ignore Labels: {self.dataset.ignore_labels}")
                print(f"  Map Labels: {self.dataset.map_labels}")
                if getattr(self.dataset, "difficult_tags", None):
                    print(f"  Difficult Tags: {self.dataset.difficult_tags}")
            elif self.dataset.format == "hrsc2016":
                print(f"  Train Split (ImageSets): {self.dataset.train_split or 'trainval'}")
                print(f"  Val Split (ImageSets): {self.dataset.val_split or 'test'}")
            else:
                if self.dataset.train_tiles_dirs:
                    print(f"  Train Tiles: {self.dataset.train_tiles_dirs}")
                else:
                    print(f"  Train Tiles: {self.dataset.train_tiles_dir}")
                if self.dataset.val_tiles_dirs:
                    print(f"  Val Tiles: {self.dataset.val_tiles_dirs}")
                else:
                    print(f"  Val Tiles: {self.dataset.val_tiles_dir}")
                if getattr(self.dataset, "same_folder", False):
                    print(f"  Same folder: images and labels in train/val tiles dirs")
            if getattr(self.dataset, "lookalike_labels", None):
                print(f"  Lookalike Labels (aliases): {self.dataset.lookalike_labels}")
            print(
                "  Lookalike: reserved name 'lookalike' is always a hard-negative "
                "routing token (excluded from class_map)."
            )
            print(f"  difficult_strategy: {getattr(self.dataset, 'difficult_strategy', 'drop')}")
            print(f"  filter_empty_gt: {getattr(self.dataset, 'filter_empty_gt', False)}")
            print(f"  drop_easy_empty_tiles: {getattr(self.dataset, 'drop_easy_empty_tiles', False)}")
            print(f"  Tile overlap (px): {getattr(self.dataset, 'overlap', 16)}")
            print()
        
        print("DataLoader:")
        print(f"  Batch Size: {self.data_loader.batch_size}")
        print(f"  Num Workers: {self.data_loader.num_workers}")
        print(f"  Shuffle: {self.data_loader.shuffle}")
        print()
        
        print("Model:")
        print(f"  Backbone: {self.model.backbone}")
        print(f"  Pretrained: {self.model.pretrained_backbone}")
        print(f"  Anchor Scales: {self.model.anchor_scales}")
        print(f"  Anchor Ratios: {self.model.anchor_ratios}")
        print(f"  Inference Pre-NMS Score Threshold: {self.model.inference_pre_nms_score_threshold}")
        print()
        
        print("Training:")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Use LR Param Groups: {self.training.use_lr_param_groups}")
        if self.training.use_lr_param_groups:
            lh = getattr(self.training, "lr_mult_head", None)
            head_note = f", head(optional)={lh}" if lh is not None else ""
            print(
                "  LR Multipliers "
                f"(backbone/rpn/roi/other{head_note}): "
                f"{self.training.lr_mult_backbone}/"
                f"{self.training.lr_mult_rpn}/"
                f"{self.training.lr_mult_roi}/"
                f"{self.training.lr_mult_other}"
            )
        print(f"  Batch Size: {self.data_loader.batch_size}")
        print(f"  Gradient Accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {self.data_loader.batch_size * self.training.gradient_accumulation_steps}")
        print(f"  Mixed Precision: {self.training.use_amp}")
        print()
        
        print("Evaluation:")
        print(f"  Score Threshold (train val): {self.evaluation.score_threshold}")
        print(
            f"  preds_score_threshold (odet preds / eval-val): "
            f"{self.evaluation.preds_score_threshold!r}"
        )
        if self.evaluation.per_class_score_threshold:
            print(f"  Per-class score thresholds: {self.evaluation.per_class_score_threshold}")
        print(f"  Extended GT metrics: {self.evaluation.extended_gt_metrics}")
        print(f"  IoU Threshold: {self.evaluation.iou_threshold}")
        print(
            f"  final_nms_iou_threshold (odet preds / eval-val): "
            f"{self.evaluation.final_nms_iou_threshold!r}"
        )
        print(f"  Compute mAP final (best model): {self.evaluation.compute_map_final}")
        print(f"  Compute mAP every N epochs: {self.evaluation.compute_map_every_n_epochs}")
        print(f"  Use exact rotated IoU (mAP / GT cover): {self.evaluation.use_exact_rotated_iou}")
        _final_iou = getattr(self.evaluation, "use_exact_rotated_iou_for_final_map", None)
        if _final_iou is not None:
            print(f"  Use exact rotated IoU (final mAP): {_final_iou}")
        print()

        print("Production (overrides; null = use evaluation/model/dataset):")
        inf = self.production
        ow_px = resolve_inference_sliding_window_overlap_pixels(self)
        _eff_sc, _eff_pc, _eff_iou = effective_eval_metric_thresholds(self)
        _preds_nms, _preds_nms_src = resolve_preds_final_nms_iou_threshold(self)
        _preds_sc, _preds_sc_src = resolve_preds_score_threshold(self)
        print(f"  score_threshold: {inf.score_threshold!r} → train/deploy {_eff_sc}")
        print(f"  odet preds / eval-val score → {_preds_sc} via {_preds_sc_src}")
        print(f"  per_class_score_threshold: {inf.per_class_score_threshold!r} → effective {_eff_pc!r}")
        print(f"  (mAP IoU from evaluation.iou_threshold: {self.evaluation.iou_threshold} → effective {_eff_iou})")
        print(
            f"  final_nms_iou_threshold: {inf.final_nms_iou_threshold!r} "
            f"(deploy/image_demo); odet preds → {_preds_nms} via {_preds_nms_src}"
        )
        print(f"  final_nms_use_cpu: {inf.final_nms_use_cpu!r}")
        print(f"  inference_pre_nms_score_threshold: {inf.inference_pre_nms_score_threshold!r}")
        print(f"  max_detections_per_image: {inf.max_detections_per_image!r}")
        print(f"  nms_class_agnostic: {inf.nms_class_agnostic!r}")
        print(f"  roi_inference_top_class_only: {inf.roi_inference_top_class_only!r}")
        print(f"  rpn_pre_nms_top_n: {inf.rpn_pre_nms_top_n!r}")
        print(f"  rpn_post_nms_top_n: {inf.rpn_post_nms_top_n!r}")
        print(f"  rpn_nms_threshold: {inf.rpn_nms_threshold!r}")
        print(f"  sliding overlap (px): config={inf.overlap_pixels!r} → effective {ow_px}")
        print(f"  ignore_margin_pixels: {inf.ignore_margin_pixels!r}")
        print(f"  use_first_image_canvas: {inf.use_first_image_canvas!r}")
        print(f"  stick_to_model_canvas: {inf.stick_to_model_canvas!r}")
        print()

        print("Checkpoint:")
        print(f"  Discover previous run: {getattr(self.checkpoint, 'discover_previous_run', False)}")
        print(f"  Resume from checkpoint epoch: {self.checkpoint.resume_from_checkpoint_epoch}")
        print(f"  Load Optimizer State: {self.checkpoint.load_optimizer_state}")
        print(f"  Include Prefixes: {self.checkpoint.load_include_prefixes}")
        print(f"  Exclude Prefixes: {self.checkpoint.load_exclude_prefixes}")
        print()

        print("Loss:")
        print(f"  Type: {self.loss.loss_type}")
        print(f"  Class Weight Method: {self.loss.class_weight_method}")
        print(f"  Background Weight: {self.loss.background_weight}")
        print()
        
        if self.num_classes is not None:
            print("Classes:")
            print(f"  Number of classes (foreground): {self.num_classes}")
            if self.class_names:
                print(f"  Class names ({len(self.class_names)}): {', '.join(self.class_names)}")
            if self.class_map:
                print(f"  Class map: {self.class_map}")

