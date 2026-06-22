"""Checkpoint loading and experiment path resolution for inference/deploy/export."""

from __future__ import annotations

from pathlib import Path

import torch

from oriented_det import OrientedRCNN, RotatedFasterRCNN, RotatedRetinaNet
from oriented_det.train.config import TrainingExperimentConfig, apply_inference_config_to_model


def _strip_ddp_prefix(state_dict: dict) -> dict:
    """Normalize DDP checkpoints so head-key inspection matches model keys."""
    if state_dict and next(iter(state_dict.keys()), "").startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_retinanet_num_classes_from_state_dict(state_dict: dict) -> int | None:
    cls_weight_key = "head.conv_cls.weight"
    bbox_weight_key = "head.conv_bbox.weight"
    if cls_weight_key not in state_dict or bbox_weight_key not in state_dict:
        return None

    bbox_channels = state_dict[bbox_weight_key].shape[0]
    if bbox_channels % 5 != 0:
        raise ValueError(
            f"Could not infer RetinaNet anchors: {bbox_weight_key} has {bbox_channels} output channels, not divisible by 5"
        )

    num_anchors = bbox_channels // 5
    cls_channels = state_dict[cls_weight_key].shape[0]
    if cls_channels % num_anchors != 0:
        raise ValueError(
            f"Could not infer RetinaNet classes: {cls_weight_key} has {cls_channels} output channels, "
            f"not divisible by inferred anchors={num_anchors}"
        )
    return cls_channels // num_anchors


def infer_num_classes_from_checkpoint(checkpoint_path: str, model_type: str) -> int:
    """
    Infer the number of classes from the checkpoint state_dict.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model type string
    
    Returns:
        Number of classes (excluding background) - this is what OrientedRCNN expects
    """
    device_obj = torch.device('cpu')  # Load on CPU first for inspection
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    state_dict = _strip_ddp_prefix(checkpoint.get("model_state_dict", checkpoint))
    
    # Try to infer from ROI head classification head (for RCNN models)
    if 'oriented_rcnn' in model_type.lower() or 'rcnn' in model_type.lower():
        cls_weight_key = 'roi_head.cls_head.weight'
        cls_bias_key = 'roi_head.cls_head.bias'
        if cls_weight_key in state_dict:
            # cls_head output is num_classes + 1 (including background)
            # OrientedRCNN expects num_classes (excluding background), so subtract 1
            total_classes = state_dict[cls_weight_key].shape[0]
            num_classes = total_classes - 1
            return num_classes
        elif cls_bias_key in state_dict:
            # cls_head output is num_classes + 1 (including background)
            # OrientedRCNN expects num_classes (excluding background), so subtract 1
            total_classes = state_dict[cls_bias_key].shape[0]
            num_classes = total_classes - 1
            return num_classes
    # For Rotated RetinaNet, cls logits are anchors * foreground classes;
    # bbox logits are anchors * 5 oriented box deltas.
    elif 'retinanet' in model_type.lower():
        num_classes = _infer_retinanet_num_classes_from_state_dict(state_dict)
        if num_classes is not None:
            return num_classes
    
    raise ValueError(f"Could not infer num_classes from checkpoint. Checkpoint keys: {list(state_dict.keys())[:10]}")


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = 'cuda:0'):
    """
    Load model from checkpoint and config.

    After loading weights, applies ``apply_inference_config_to_model`` so ``production.*``
    decode/NMS overrides (e.g. RPN top-k) take effect for **inference-only** callers
    (deploy, ``save_predictions``, ``image_demo``). ``tools/train.py`` does not use this path
    for the live training model.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config.json file
        device: Device to load model on
    
    Returns:
        tuple: (model, config, class_names)
    """
    from oriented_det.pretrained import ensure_checkpoint

    checkpoint_path = str(ensure_checkpoint(checkpoint_path))

    # Load config using the proper load method to convert nested dicts to dataclasses
    config = TrainingExperimentConfig.load(Path(config_path))
    
    # Determine model type
    model_type = config.model_type or 'oriented_rcnn'
    
    # Load checkpoint once for both num_classes inference and model loading
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _strip_ddp_prefix(checkpoint.get("model_state_dict", checkpoint))

    # num_classes = foreground only (config and model API; cls_head/fc_cls has num_classes + 1 for background)
    num_classes_config = config.num_classes
    cls_weight_key = 'roi_head.cls_head.weight'
    if cls_weight_key not in state_dict and 'roi_head.fc_cls.weight' in state_dict:
        cls_weight_key = 'roi_head.fc_cls.weight'
    checkpoint_foreground = None
    if cls_weight_key in state_dict:
        checkpoint_foreground = state_dict[cls_weight_key].shape[0] - 1
    elif 'retinanet' in model_type.lower():
        checkpoint_foreground = _infer_retinanet_num_classes_from_state_dict(state_dict)

    if num_classes_config is None:
        if checkpoint_foreground is not None:
            num_classes = checkpoint_foreground
            print(f"num_classes not in config; inferred from checkpoint: {num_classes} (foreground)")
        else:
            num_classes = infer_num_classes_from_checkpoint(checkpoint_path, model_type)
            print(f"Inferred num_classes from checkpoint: {num_classes} (foreground)")
    elif checkpoint_foreground is not None:
        if num_classes_config == checkpoint_foreground + 1:
            # Legacy: config stored "including background"
            num_classes = checkpoint_foreground
            print(f"Config num_classes={num_classes_config} (legacy including background); using {num_classes} (foreground).")
        elif num_classes_config != checkpoint_foreground:
            num_classes = checkpoint_foreground
            print(f"Warning: config num_classes={num_classes_config} doesn't match checkpoint (foreground={checkpoint_foreground}); using checkpoint.")
        else:
            num_classes = num_classes_config
    else:
        num_classes = num_classes_config
        print(f"Using num_classes={num_classes} from config (cannot verify with checkpoint)")
    
    # Get class names from config (may be None if not saved)
    class_names = config.class_names
    
    # Create model from config with the same critical inference/decode params used in train.py:create_model_from_config.
    model_type_lower = model_type.lower()
    model_kwargs = {
        'num_classes': num_classes,
        'backbone_name': config.model.backbone if config.model else 'resnet50',
        'pretrained_backbone': config.model.pretrained_backbone if config.model else False,
    }
    if config.model:
        # Loss/decode and matching settings (must match training-time model construction)
        loss_config = config.loss
        if loss_config.loss_type == "class_weighted":
            roi_loss_type = "cross_entropy"
        elif loss_config.loss_type in ["focal", "focal_weighted"]:
            roi_loss_type = "focal"
        else:
            roi_loss_type = config.model.roi_loss_type
        if loss_config.loss_type in ["focal", "focal_weighted"]:
            roi_focal_alpha = getattr(loss_config, "focal_alpha", config.model.roi_focal_alpha)
            roi_focal_gamma = getattr(loss_config, "focal_gamma", config.model.roi_focal_gamma)
        else:
            roi_focal_alpha = config.model.roi_focal_alpha
            roi_focal_gamma = config.model.roi_focal_gamma
        roi_label_smoothing = getattr(config.loss, "label_smoothing", 0.0)
        target_means = tuple(config.model.target_means) if isinstance(config.model.target_means, list) else config.model.target_means
        target_stds = tuple(config.model.target_stds) if isinstance(config.model.target_stds, list) else config.model.target_stds
        use_hbb = getattr(config.model, "use_hbb_for_matching", False)
        inference_pre_nms_score_threshold = getattr(config.model, "inference_pre_nms_score_threshold", 0.05)

        if hasattr(config.model, 'anchor_scales') and config.model.anchor_scales:
            model_kwargs['anchor_scales'] = config.model.anchor_scales
        if hasattr(config.model, 'anchor_ratios') and config.model.anchor_ratios:
            model_kwargs['anchor_ratios'] = config.model.anchor_ratios
        # Backbone/FPN must match training or load_state_dict fails (e.g. fpn_returned_layers [1,2,3] has no layer4)
        frozen_stages = getattr(config.model, 'frozen_stages', None)
        if frozen_stages is not None:
            trainable_layers = 5 if frozen_stages == 0 else max(1, 4 - frozen_stages)
        else:
            trainable_layers = getattr(config.model, 'trainable_layers', 5)
        model_kwargs['trainable_layers'] = trainable_layers
        fpn_returned = getattr(config.model, 'fpn_returned_layers', None)
        fpn_strides = getattr(config.model, 'fpn_strides', None)
        if fpn_returned is not None:
            model_kwargs['returned_layers'] = fpn_returned
        if fpn_strides is not None:
            model_kwargs['fpn_strides'] = fpn_strides
        model_kwargs.update({
            'roi_loss_type': roi_loss_type,
            'roi_focal_alpha': roi_focal_alpha,
            'roi_focal_gamma': roi_focal_gamma,
            'roi_label_smoothing': roi_label_smoothing,
            'target_means': target_means,
            'target_stds': target_stds,
            'roi_norm_factor': config.model.roi_norm_factor,
            'roi_edge_swap': config.model.roi_edge_swap,
            'roi_box_reg_angle_weight': getattr(config.model, 'roi_box_reg_angle_weight', 1.0),
            'roi_box_reg_iou_weight': getattr(config.model, 'roi_box_reg_iou_weight', 0.0),
            'roi_box_reg_iou_schedule_epochs': getattr(
                config.model, 'roi_box_reg_iou_schedule_epochs', None
            ),
            'roi_box_reg_iou_schedule_values': getattr(
                config.model, 'roi_box_reg_iou_schedule_values', None
            ),
            'use_hbb_for_matching': use_hbb,
            'inference_pre_nms_score_threshold': inference_pre_nms_score_threshold,
            'rpn_min_size': getattr(config.model, 'rpn_min_size', 0.0),
            'rpn_pre_nms_top_n': getattr(config.model, 'rpn_pre_nms_top_n', 2000),
            'rpn_post_nms_top_n': getattr(config.model, 'rpn_post_nms_top_n', 1000),
            'max_detections_per_image': getattr(config.model, 'max_detections_per_image', 100),
            'rpn_nms_threshold': getattr(config.model, 'rpn_nms_threshold', 0.7),
            'final_nms_iou_threshold': config.model.final_nms_iou_threshold,
            'nms_class_agnostic': getattr(config.model, 'nms_class_agnostic', False),
            'roi_batch_size_per_image': getattr(config.model, 'roi_batch_size_per_image', 512),
            'rpn_batch_size_per_image': getattr(config.model, 'rpn_batch_size_per_image', 256),
            'rpn_min_pos_iou': getattr(config.model, 'rpn_min_pos_iou', 0.3),
            'rpn_match_low_quality': getattr(config.model, 'rpn_match_low_quality', True),
            'roi_match_low_quality': getattr(config.model, 'roi_match_low_quality', False),
            'add_gt_as_proposals': getattr(config.model, 'add_gt_as_proposals', True),
            'rpn_positive_iou_threshold': getattr(config.model, 'rpn_positive_iou_threshold', 0.5),
            'rpn_negative_iou_threshold': getattr(config.model, 'rpn_negative_iou_threshold', 0.2),
            'roi_positive_iou_threshold': getattr(config.model, 'roi_positive_iou_threshold', 0.4),
            'roi_negative_iou_threshold': getattr(config.model, 'roi_negative_iou_threshold', 0.3),
            'final_nms_iou_schedule_epochs': config.model.final_nms_iou_schedule_epochs,
            'final_nms_iou_schedule_values': config.model.final_nms_iou_schedule_values,
            'final_nms_use_cpu': getattr(config.model, 'final_nms_use_cpu', False),
            'roi_inference_top_class_only': getattr(
                config.model, 'roi_inference_top_class_only', False
            ),
        })
    if model_type_lower == 'rotated_faster_rcnn':
        model = RotatedFasterRCNN(**model_kwargs)
    elif 'oriented_rcnn' in model_type_lower:
        oriented_kwargs = dict(model_kwargs)
        # OrientedRCNN constructor does not accept these RotatedFasterRCNN-only args.
        oriented_kwargs.pop('add_gt_as_proposals', None)
        oriented_kwargs.pop('rpn_min_size', None)
        model = OrientedRCNN(**oriented_kwargs)
    elif 'retinanet' in model_type_lower:
        m = config.model
        model = RotatedRetinaNet(
            num_classes=num_classes,
            backbone_name=m.backbone if m else 'resnet50',
            pretrained_backbone=m.pretrained_backbone if m else False,
            trainable_layers=model_kwargs.get('trainable_layers', 5),
            returned_layers=model_kwargs.get('returned_layers', None),
            fpn_strides=model_kwargs.get('fpn_strides', None),
            fpn_extra_level=getattr(m, "fpn_extra_level", False) if m else False,
            anchor_scales=m.anchor_scales if m else None,
            anchor_ratios=m.anchor_ratios if m else None,
            octave_base_scale=getattr(m, "anchor_octave_base_scale", None) if m else None,
            scales_per_octave=getattr(m, "anchor_scales_per_octave", None) if m else None,
            stacked_convs=getattr(m, "retinanet_stacked_convs", 1) if m else 1,
            positive_iou_threshold=getattr(m, "rpn_positive_iou_threshold", 0.5) if m else 0.5,
            negative_iou_threshold=getattr(m, "rpn_negative_iou_threshold", 0.4) if m else 0.4,
            focal_alpha=model_kwargs.get('roi_focal_alpha', 0.25),
            focal_gamma=model_kwargs.get('roi_focal_gamma', 2.0),
            target_means=model_kwargs.get('target_means', None),
            target_stds=model_kwargs.get('target_stds', None),
            norm_factor=m.roi_norm_factor if m else None,
            edge_swap=m.roi_edge_swap if m else True,
            box_reg_weight=getattr(m, "box_reg_weight", 1.0) if m else 1.0,
            box_reg_loss_type=getattr(m, "box_reg_loss_type", "smooth_l1") if m else "smooth_l1",
            box_reg_iou_weight=getattr(m, "roi_box_reg_iou_weight", 0.0) if m else 0.0,
            box_reg_iou_loss_type=getattr(m, "roi_box_reg_iou_loss_type", "riou") if m else "riou",
            box_reg_kfiou_fun=getattr(m, "roi_box_reg_kfiou_fun", None) if m else None,
            box_reg_probiou_mode=getattr(m, "roi_box_reg_probiou_mode", None) if m else None,
            use_hbb_for_matching=getattr(m, "use_hbb_for_matching", False) if m else False,
            score_threshold=model_kwargs.get('inference_pre_nms_score_threshold', 0.05),
            final_nms_iou_threshold=m.final_nms_iou_threshold if m else 0.5,
            max_detections_per_image=getattr(m, "max_detections_per_image", 100) if m else 100,
            final_nms_iou_schedule_epochs=m.final_nms_iou_schedule_epochs if m else None,
            final_nms_iou_schedule_values=m.final_nms_iou_schedule_values if m else None,
            roi_box_reg_iou_schedule_epochs=getattr(m, "roi_box_reg_iou_schedule_epochs", None) if m else None,
            roi_box_reg_iou_schedule_values=getattr(m, "roi_box_reg_iou_schedule_values", None) if m else None,
            final_nms_use_cpu=getattr(m, "final_nms_use_cpu", False) if m else False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint state dict into model (already stripped "module." above if DDP)
    model.load_state_dict(state_dict)
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    apply_inference_config_to_model(model, getattr(config, "production", None))

    print(f"Loaded model from {checkpoint_path}")
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names if class_names else 'Not available (will use generic names)'}")
    
    return model, config, class_names
