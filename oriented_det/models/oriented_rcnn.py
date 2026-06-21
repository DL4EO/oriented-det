"""Complete Oriented R-CNN model for true oriented object detection.

This module implements a full oriented R-CNN that:
- Predicts oriented bounding boxes (5 parameters: cx, cy, w, h, angle)
- Preserves angle information throughout training and inference
- Uses oriented RPN, ROI align, and ROI head
- Supports oriented IoU matching and NMS
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ..geometry import RBox
from .oriented_rpn import (
    OrientedRPNHead,
    generate_oriented_anchors,
    encode_oriented_boxes,
    decode_oriented_boxes,
    match_oriented_anchors_to_gt,
    compute_oriented_rpn_loss,
    generate_oriented_proposals,
    generate_horizontal_proposals,
    compute_midpoint_rpn_loss,
    generate_midpoint_proposals,
)
from .oriented_roi import (
    OrientedROIAlign,
    HorizontalROIAlign,
    OrientedROIHead,
    match_oriented_proposals_to_gt,
    compute_oriented_roi_loss,
    compute_horizontal_roi_loss_mmrotate,
    compute_roi_matching_diagnostics,
)
from .bbox_coder import MidpointOffsetCoder, xyxy_to_obb
from .utils import (
    rboxes_to_tensor,
    tensor_to_rboxes,
    prepare_targets,
    setup_backbone,
    extract_backbone_features,
    setup_anchors,
    ClassWeightsMixin,
    GroupedCeMixin,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
)
from ..ops import obb_to_xyxy_gpu

# Internal debug switch for one-shot first training forward timing prints. Keep
# disabled for normal training; set to True locally when investigating stalls.
TRACE_FIRST_TRAIN_FORWARD_TIMING = False


class RotatedFasterRCNN(ClassWeightsMixin, GroupedCeMixin, nn.Module):
    """Rotated Faster R-CNN (MMRotate-style): horizontal RPN + horizontal RoIAlign + rotated ROI head.
    
    Mimics MMRotate's Rotated Faster R-CNN:
    - Horizontal RPN with 4 reg params (DeltaXYWHBBoxCoder)
    - Horizontal RoIAlign
    - ROI head with 5 params (DeltaXYWHTHBBoxCoder)
    
    Args:
        num_classes: Number of object classes (excluding background)
        backbone: Optional backbone module (if None, creates ResNet+FPN)
        backbone_name: Name of backbone to create ("resnet18", "resnet50", etc.)
        pretrained_backbone: Whether to use pretrained backbone weights
        trainable_layers: Number of backbone layers to keep trainable. For ResNet50,
                         typically 5 layers (conv1 + layer1-4). Set to 5 to train all layers.
                         Default is 5 (all layers trainable).
        anchor_scales: List of anchor scales for RPN
        anchor_ratios: List of anchor aspect ratios for RPN
        rpn_pre_nms_top_n: Number of top proposals before NMS in RPN
        rpn_post_nms_top_n: Number of top proposals after NMS in RPN
        rpn_positive_iou_threshold: IoU threshold for positive RPN anchors
        rpn_negative_iou_threshold: IoU threshold for negative RPN anchors
        roi_positive_iou_threshold: IoU threshold for positive ROI proposals
        roi_negative_iou_threshold: IoU threshold for negative ROI proposals
        roi_output_size: Output size for ROI align (height, width)
        roi_spatial_scales: Spatial scales for each FPN level in ROI align (defaults from ``fpn_strides`` at init)
        fpn_strides: Nominal strides per FPN level for ROI module init; forward pass overrides with strides
            derived from image size and feature map shapes so RPN anchors and ROI align match the grid.
        use_checkpoint: Whether to use gradient checkpointing (trades compute for memory)
        roi_chunk_size: Number of ROIs to process in parallel during ROI align.
                       Lower values use less memory but are slower.
                       Recommended: 16-64 for typical GPU memory (8-24GB).
        roi_use_checkpoint: If True, use gradient checkpointing in ROI align to
                           trade compute for memory (~2x less memory, ~30% slower).
        roi_loss_type: Type of classification loss:
                      - "cross_entropy": Standard cross-entropy loss (default)
                      - "focal": Focal loss for handling class imbalance and hard examples
        roi_class_weights: Optional class weights for loss computation:
                          - Dict[str, float]: Mapping from class name to weight
                            (requires calling set_class_weights() with class_map)
                          - torch.Tensor: Tensor of shape [num_classes + 1] (including background)
                          - None: Equal weights for all classes (default)
        roi_focal_alpha: Alpha parameter for focal loss (default: 1.0)
        roi_focal_gamma: Gamma parameter for focal loss (default: 2.0)
                         - gamma=0: equivalent to cross-entropy
                         - gamma>0: focuses on hard examples
                         - gamma=2: standard value used in literature
    
    Example:
        >>> # Standard cross-entropy loss
        >>> model = OrientedRCNN(num_classes=15)
        
        >>> # Class-weighted cross-entropy loss
        >>> class_weights = {"small-vehicle": 0.5, "plane": 2.0, ...}
        >>> model = OrientedRCNN(num_classes=15, roi_class_weights=class_weights)
        >>> model.set_class_weights(class_map)  # Must call before training
        
        >>> # Focal loss
        >>> model = OrientedRCNN(num_classes=15, roi_loss_type="focal", roi_focal_gamma=2.0)
        
        >>> # Focal loss with class weights
        >>> model = OrientedRCNN(
        ...     num_classes=15,
        ...     roi_loss_type="focal",
        ...     roi_class_weights=class_weights,
        ...     roi_focal_gamma=2.0
        ... )
        >>> model.set_class_weights(class_map)
    """
    
    def __init__(
        self,
        num_classes: int,
        *,
        backbone=None,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = False,
        trainable_layers: int = 5,
        anchor_scales: Optional[List[float]] = None,
        anchor_ratios: Optional[List[float]] = None,
        anchor_angles: Optional[List[float]] = None,
        rpn_pre_nms_top_n: int = 2000,
        rpn_post_nms_top_n: int = 2000,
        rpn_min_size: float = 0.0,  # Minimum proposal side length (pixels); use 2–4 to filter tiny boxes
        max_detections_per_image: int = 2000,  # MMRotate test_cfg rcnn max_per_img
        rpn_nms_threshold: float = 0.7,  # IoU threshold for RPN proposal NMS
        final_nms_iou_threshold: float = 0.1,  # MMRotate rotated NMS iou (DOTA le90 config)
        rpn_positive_iou_threshold: float = 0.7,  # MMRotate RPN MaxIoUAssigner pos_iou_thr
        rpn_negative_iou_threshold: float = 0.3,  # MMRotate RPN neg_iou_thr
        rpn_fg_bg_sampling_ratio: float = 0.5,  # Ratio of foreground to background in sampled RPN anchors
        rpn_batch_size_per_image: int = 256,  # Number of RPN anchors to sample per image (MMRotate: 256)
        rpn_min_pos_iou: float = 0.3,  # Min IoU for best-anchor-per-GT to be positive (MMRotate RPN assigner)
        rpn_match_low_quality: bool = True,  # If True, force best anchor per GT to positive when IoU >= rpn_min_pos_iou
        roi_match_low_quality: bool = False,  # MMRotate RCNN assigner match_low_quality (usually False)
        roi_min_pos_iou: float = 0.5,  # MMRotate RCNN MaxIoUAssigner min_pos_iou
        add_gt_as_proposals: bool = True,  # If True, append GT boxes to RPN proposals in training (MMRotate default)
        roi_positive_iou_threshold: float = 0.5,  # MMRotate RCNN pos_iou_thr
        roi_negative_iou_threshold: float = 0.5,  # MMRotate RCNN neg_iou_thr (same as pos for DOTA config)
        roi_fg_bg_sampling_ratio: float = 0.25,  # Ratio of foreground to background in sampled ROI proposals
        roi_batch_size_per_image: int = 512,  # Number of ROI proposals to sample per image for training
        roi_output_size: Tuple[int, int] = (7, 7),
        roi_spatial_scales: Optional[List[float]] = None,
        fpn_strides: Optional[List[int]] = None,
        returned_layers: Optional[List[int]] = None,
        use_checkpoint: bool = False,
        roi_chunk_size: int = 32,
        roi_use_checkpoint: bool = False,
        # Classification loss configuration
        roi_loss_type: str = "cross_entropy",
        roi_class_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
        roi_focal_alpha: float = 1.0,
        roi_focal_gamma: float = 2.0,
        roi_label_smoothing: float = 0.0,
        # Box regression target normalization (like MMRotate: only ROI head uses small stds)
        target_means: Optional[Tuple[float, float, float, float, float]] = None,
        target_stds: Optional[Tuple[float, float, float, float, float]] = None,
        roi_norm_factor: float = 2.0,
        roi_edge_swap: bool = True,
        roi_box_reg_angle_weight: float = 1.0,
        roi_box_reg_angle_schedule_epochs: Optional[List[int]] = None,
        roi_box_reg_angle_schedule_values: Optional[List[float]] = None,
        roi_box_reg_iou_weight: float = 0.0,
        roi_box_reg_iou_loss_type: str = "riou",
        roi_box_reg_kfiou_fun: Optional[str] = None,
        roi_box_reg_probiou_mode: Optional[str] = None,
        use_hbb_for_matching: bool = False,
        inference_pre_nms_score_threshold: float = 0.05,
        final_nms_iou_schedule_epochs: Optional[List[int]] = None,
        final_nms_iou_schedule_values: Optional[List[float]] = None,
        roi_box_reg_iou_schedule_epochs: Optional[List[int]] = None,
        roi_box_reg_iou_schedule_values: Optional[List[float]] = None,
        nms_class_agnostic: bool = False,
        final_nms_use_cpu: bool = False,
        roi_inference_top_class_only: bool = False,
    ):
        """Initialize Rotated Faster R-CNN model.
        
        Args:
            num_classes: Number of object classes (excluding background)
            backbone: Optional backbone module (if None, creates ResNet+FPN)
            backbone_name: Name of backbone to create ("resnet18", "resnet50", etc.)
            pretrained_backbone: Whether to use pretrained backbone weights
            trainable_layers: Number of backbone layers to keep trainable. For ResNet50,
                             typically 5 layers (conv1 + layer1-4). Set to 5 to train all layers.
                             Default is 5 (all layers trainable).
            anchor_scales: List of anchor scales for RPN
            anchor_ratios: List of anchor aspect ratios for RPN
            anchor_angles: Optional RPN angles in radians (Python API only; not in JSON). ``None`` -> ``[0.0]``.
            rpn_pre_nms_top_n: Number of top proposals before NMS in RPN
            rpn_post_nms_top_n: Number of top proposals after NMS in RPN
            rpn_positive_iou_threshold: IoU threshold for positive RPN anchors
            rpn_negative_iou_threshold: IoU threshold for negative RPN anchors
            rpn_fg_bg_sampling_ratio: Ratio of foreground to background in sampled RPN anchors (default: 0.5)
            roi_positive_iou_threshold: IoU threshold for positive ROI proposals
            roi_negative_iou_threshold: IoU threshold for negative ROI proposals
            roi_fg_bg_sampling_ratio: Ratio of foreground to background in sampled ROI proposals (default: 0.25)
            roi_output_size: Output size for ROI align (height, width)
            roi_spatial_scales: Spatial scales for each FPN level in ROI align (defaults from ``fpn_strides`` at init)
            fpn_strides: Nominal strides for ROI module init; forward uses strides derived from the feature grid.
            use_checkpoint: Whether to use gradient checkpointing (trades compute for memory)
            roi_chunk_size: Number of ROIs to process in parallel during ROI align
            roi_use_checkpoint: If True, use gradient checkpointing in ROI align
            roi_loss_type: Type of classification loss:
                          - "cross_entropy": Standard cross-entropy loss (default)
                          - "focal": Focal loss for handling class imbalance
            roi_class_weights: Optional class weights for loss computation.
                              Can be:
                              - Dict[str, float]: Mapping from class name to weight
                              - torch.Tensor: Tensor of shape [num_classes + 1] (including background)
                              - None: Equal weights for all classes (default)
                              Note: If dict is provided, it will be converted to tensor
                                    using the model's class mapping. Background (index 0)
                                    defaults to weight 1.0 if not specified.
            roi_focal_alpha: Alpha parameter for focal loss (default: 1.0)
            roi_focal_gamma: Gamma parameter for focal loss (default: 2.0)
            roi_label_smoothing: Label smoothing for ROI classification (default: 0.0). Use e.g. 0.1 to reduce overconfidence.
            target_means: Optional means for regression target normalization [dx, dy, dw, dh, dangle].
                         Default: (0.0, 0.0, 0.0, 0.0, 0.0) - no centering (like MMRotate default).
                         Can be set to empirical means computed from dataset for better normalization.
            target_stds: Optional stds for regression target normalization [dx, dy, dw, dh, dangle].
                        Default: (0.1, 0.1, 0.2, 0.2, 0.1) (MMRotate DeltaXYWHTHBBoxCoder).
                        Can be set to empirical stds computed from dataset for better normalization.
            roi_norm_factor: Angle scaling for ROI encode/decode (MMRotate uses 2.0).
            roi_edge_swap: Whether to use edge_swap for ROI (MMRotate uses True).
            roi_box_reg_angle_weight: Weight for the angle (5th) component in ROI box regression loss.
                Use > 1.0 (e.g. 2.0) to improve orientation alignment when predictions stick to anchor angles.
            roi_box_reg_iou_weight: Weight for auxiliary decoded-box loss on ROI positives (``riou`` / ``kfiou``).
            roi_box_reg_iou_loss_type: ``riou`` (default), ``kfiou``, or ``probiou`` for that auxiliary term.
            roi_box_reg_kfiou_fun: Optional KFIoU overlap mapping when using ``kfiou``.
            roi_box_reg_probiou_mode: ``l1`` (default) or ``l2`` when using ``probiou``.
            roi_min_pos_iou: Min IoU for low-quality ROI matches when ``roi_match_low_quality`` is True.
            roi_inference_top_class_only: If True, ROI inference uses argmax fg class per proposal
                (see ``oriented_det/models/README.md``); if False, multiclass thresholding.
            RPN uses fixed (0,0,0,0) and (1,1,1,1) per MMRotate DeltaXYWHBBoxCoder.
        """
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for RotatedFasterRCNN.")
        
        # Initialize ClassWeightsMixin first, then nn.Module
        # ClassWeightsMixin.__init__() sets self.num_classes
        super().__init__(num_classes, roi_class_weights)
        nn.Module.__init__(self)
        self._init_grouped_ce()
        
        # Box regression: ROI head uses target_means/stds/norm_factor; RPN uses identity (MMRotate)
        self.target_means = target_means if target_means is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
        self.target_stds = target_stds if target_stds is not None else (0.1, 0.1, 0.2, 0.2, 0.1)
        self.roi_norm_factor = roi_norm_factor
        self.roi_edge_swap = roi_edge_swap
        self.roi_box_reg_iou_loss_type = roi_box_reg_iou_loss_type
        self.roi_box_reg_kfiou_fun = roi_box_reg_kfiou_fun
        self.roi_box_reg_probiou_mode = roi_box_reg_probiou_mode
        self.use_hbb_for_matching = use_hbb_for_matching
        self.inference_pre_nms_score_threshold = inference_pre_nms_score_threshold
        self._roi_box_reg_angle_schedule_epochs = roi_box_reg_angle_schedule_epochs
        self._roi_box_reg_angle_schedule_values = roi_box_reg_angle_schedule_values
        self._roi_box_reg_angle_weight_default = float(roi_box_reg_angle_weight)
        self.roi_box_reg_angle_weight = float(roi_box_reg_angle_weight)
        self.set_roi_box_reg_angle_weight_for_epoch(0)
        
        # Store loss configuration
        self.roi_loss_type = roi_loss_type
        self.roi_focal_alpha = roi_focal_alpha
        self.roi_focal_gamma = roi_focal_gamma
        self.roi_label_smoothing = roi_label_smoothing

        # Setup backbone using shared utility (returned_layers=[2,3,4] for MMRotate-style C3–C5 only)
        self.backbone, backbone_channels = setup_backbone(
            backbone=backbone,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
        )
        # FPN strides: default [4,8,16,32,64] for C2–C5; [8,16,32,64] when returned_layers=[2,3,4]
        if fpn_strides is None and returned_layers == [2, 3, 4]:
            fpn_strides = [8, 16, 32, 64]
        
        # Default horizontal RPN priors; optional anchor_angles for advanced Python use only.
        self.anchor_scales, self.anchor_ratios, self.anchor_angles, self.num_anchors = setup_anchors(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_angles=anchor_angles,
            default_angles=[0.0],
        )
        
        # RPN head configuration
        # MMRotate format: cls_out_channels=1, reg_out_channels=4 (dx, dy, dw, dh - angle from anchor)
        rpn_cls_out_channels = 1
        rpn_reg_out_channels = 4
        
        # RPN head
        self.rpn_head = OrientedRPNHead(
            backbone_channels, 
            num_anchors=self.num_anchors,
            cls_out_channels=rpn_cls_out_channels,
            reg_out_channels=rpn_reg_out_channels,
        )
        
        # ROI components (MMRotate Rotated Faster R-CNN uses horizontal RoIAlign)
        self.roi_align = HorizontalROIAlign(
            output_size=roi_output_size,
            spatial_scales=roi_spatial_scales,
            fpn_strides=fpn_strides,
            chunk_size=roi_chunk_size,
        )
        
        # ROI head input size: output_size[0] * output_size[1] * backbone_channels
        roi_feature_size = roi_output_size[0] * roi_output_size[1] * backbone_channels
        # Use class-agnostic regression (MMRotate format) - better for initial training when class prediction is weak
        self.roi_class_agnostic_regression = True  # Store flag for inference
        self.roi_head = OrientedROIHead(
            in_channels=roi_feature_size,
            num_classes=num_classes,
            class_agnostic_regression=True,  # MMRotate format: class-agnostic regression
        )
        
        # Configuration
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_min_size = rpn_min_size
        self.max_detections_per_image = max_detections_per_image
        self.rpn_nms_threshold = rpn_nms_threshold
        self.final_nms_iou_threshold = final_nms_iou_threshold
        self.nms_class_agnostic = nms_class_agnostic
        self.final_nms_use_cpu = final_nms_use_cpu
        self.roi_inference_top_class_only = roi_inference_top_class_only
        self.rpn_positive_iou_threshold = rpn_positive_iou_threshold
        self.rpn_negative_iou_threshold = rpn_negative_iou_threshold
        self.rpn_fg_bg_sampling_ratio = rpn_fg_bg_sampling_ratio
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_min_pos_iou = rpn_min_pos_iou
        self.rpn_match_low_quality = rpn_match_low_quality
        self.roi_match_low_quality = roi_match_low_quality
        self.roi_min_pos_iou = roi_min_pos_iou
        self.add_gt_as_proposals = add_gt_as_proposals
        self.roi_positive_iou_threshold = roi_positive_iou_threshold
        self.roi_negative_iou_threshold = roi_negative_iou_threshold
        self.roi_fg_bg_sampling_ratio = roi_fg_bg_sampling_ratio
        self.roi_batch_size_per_image = roi_batch_size_per_image
        self.fpn_strides = fpn_strides or [4, 8, 16, 32, 64]
        self.roi_spatial_scales = roi_spatial_scales or [1.0 / s for s in self.fpn_strides]
        
        # Gradient checkpointing (trades computation for memory)
        # When enabled, recomputes activations during backward pass instead of storing them
        # Reduces memory usage by 2-3x at the cost of ~30% slower backward pass
        self.use_checkpoint = use_checkpoint
        
        # NMS IoU schedule for final detection NMS (optional; essential for RetinaNet, optional for R-CNN)
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values
        self._roi_box_reg_iou_schedule_epochs = roi_box_reg_iou_schedule_epochs
        self._roi_box_reg_iou_schedule_values = roi_box_reg_iou_schedule_values
        self._roi_box_reg_iou_weight_default = float(roi_box_reg_iou_weight)
        self.roi_box_reg_iou_weight = float(roi_box_reg_iou_weight)
        self.set_roi_box_reg_iou_weight_for_epoch(0)
    
    def set_final_nms_iou_for_epoch(self, epoch: int) -> None:
        """Update final detection NMS IoU threshold from schedule. Lower = more aggressive suppression."""
        if self._final_nms_iou_schedule_epochs is None or self._final_nms_iou_schedule_values is None:
            return
        if not self._final_nms_iou_schedule_epochs or not self._final_nms_iou_schedule_values:
            return
        idx = 0
        for boundary in self._final_nms_iou_schedule_epochs:
            if epoch < boundary:
                break
            idx += 1
        idx = min(idx, len(self._final_nms_iou_schedule_values) - 1)
        self.final_nms_iou_threshold = self._final_nms_iou_schedule_values[idx]

    def set_roi_box_reg_iou_weight_for_epoch(self, epoch: int) -> None:
        """Update ROI auxiliary IoU loss weight from schedule (0-based epoch index)."""
        from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule

        self.roi_box_reg_iou_weight = resolve_piecewise_schedule(
            epoch,
            self._roi_box_reg_iou_schedule_epochs,
            self._roi_box_reg_iou_schedule_values,
            self._roi_box_reg_iou_weight_default,
        )

    def set_roi_box_reg_angle_weight_for_epoch(self, epoch: int) -> None:
        """Update ROI angle (5th dim) SmoothL1 weight from schedule (0-based epoch index)."""
        from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule

        self.roi_box_reg_angle_weight = resolve_piecewise_schedule(
            epoch,
            self._roi_box_reg_angle_schedule_epochs,
            self._roi_box_reg_angle_schedule_values,
            self._roi_box_reg_angle_weight_default,
        )
    
    def forward(
        self,
        images: Sequence[torch.Tensor],
        targets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Forward pass through oriented R-CNN.
        
        Args:
            images: List of image tensors (C, H, W) in [0, 1] range
            targets: Optional list of target dicts for training.
                    Each dict must contain:
                    - "rboxes" (List[RBox] or tensor [N, 5] with format [cx, cy, w, h, angle])
                    - "labels" (tensor [N])
        
        Returns:
            - Training: Dict with loss keys:
                - "loss_objectness": RPN classification loss
                - "loss_rpn_box_reg": RPN box regression loss
                - "loss_classifier": ROI classification loss
                - "loss_box_reg": ROI box regression loss
            - Inference: List of dicts, one per image, containing:
                - "rboxes": List[RBox] with oriented boxes
                - "rboxes": List[RBox] or Tensor [N, 5] with oriented boxes [cx, cy, w, h, angle]
                - "labels": Tensor [N] with class labels
                - "scores": Tensor [N] with confidence scores
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        # Get image sizes
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        timing_enabled = bool(
            TRACE_FIRST_TRAIN_FORWARD_TIMING
            and self.training
            and not getattr(self, "_timed_first_train_forward", False)
        )
        if timing_enabled:
            self._timed_first_train_forward = True
            timing_start = time.perf_counter()
            timing_last = timing_start

            def _timing_mark(label: str) -> None:
                nonlocal timing_last
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                now = time.perf_counter()
                print(f"[first-batch timing] {label}: +{now - timing_last:.3f}s total={now - timing_start:.3f}s", flush=True)
                timing_last = now

            print("[first-batch timing] begin RotatedFasterRCNN.forward", flush=True)
        else:
            def _timing_mark(label: str) -> None:
                return None
        
        # Extract features using shared utility. include_pool_level keeps torchvision's
        # P6 (stride-64 max-pool level) so the RPN sees 5 levels like MMRotate;
        # horizontal_roi_align still restricts ROI extraction to the first 4 levels.
        feature_list = extract_backbone_features(
            self.backbone,
            images,
            use_checkpoint=self.use_checkpoint,
            training=self.training,
            include_pool_level=True,
        )
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
        fpn_strides_live = derive_fpn_strides_from_grid(image_sizes[0], feature_map_sizes)
        warn_if_fpn_strides_mismatch(self.fpn_strides, fpn_strides_live)
        roi_spatial_scales_live = [1.0 / s for s in fpn_strides_live]
        _timing_mark("backbone_fpn")
        
        # Get device from first feature map
        images_tensor = torch.stack(images, dim=0)  # [B, C, H, W] - needed for device reference
        
        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            
            # Prepare targets (preserve oriented boxes)
            gt_boxes_list, gt_labels_list, gt_boxes_ignore_list = prepare_targets(targets, device=images_tensor.device)
            _timing_mark("prepare_targets")
            
            # RPN forward
            objectness_logits, bbox_regression = self.rpn_head(feature_list)
            _timing_mark("rpn_head")
            
            # Generate anchors (strides from actual grid — see fpn_strides_live)
            img_h, img_w = image_sizes[0]
            anchors = generate_oriented_anchors(
                image_size=(img_h, img_w),
                feature_map_sizes=feature_map_sizes,
                anchor_scales=self.anchor_scales,
                anchor_ratios=self.anchor_ratios,
                anchor_angles=self.anchor_angles,
                stride_per_level=fpn_strides_live,
            )
            _timing_mark("generate_anchors")
            
            # Compute RPN loss (4 params, no angle - MMRotate style)
            rpn_losses = compute_oriented_rpn_loss(
                objectness_logits=objectness_logits,
                bbox_regression=bbox_regression,
                anchors=anchors,
                gt_boxes=gt_boxes_list,
                gt_boxes_ignore=gt_boxes_ignore_list,
                image_sizes=image_sizes,
                positive_iou_threshold=self.rpn_positive_iou_threshold,
                negative_iou_threshold=self.rpn_negative_iou_threshold,
                box_reg_weight=1.0,
                fg_bg_sampling_ratio=self.rpn_fg_bg_sampling_ratio,
                batch_size_per_image=self.rpn_batch_size_per_image,
                min_pos_iou=self.rpn_min_pos_iou,
                match_low_quality=self.rpn_match_low_quality,
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
                use_horizontal_targets=True,
                use_hbb_for_matching=self.use_hbb_for_matching,
            )
            _timing_mark("rpn_loss")
            
            # Horizontal proposals (xyxy) — MMRotate semantics. No objectness score
            # threshold: MMRotate's RPN keeps top-k proposals by score only;
            # inference_pre_nms_score_threshold applies to ROI-head scores, not the RPN.
            proposals_xyxy = generate_horizontal_proposals(
                objectness_logits=objectness_logits,
                bbox_regression=bbox_regression,
                anchors=anchors,
                image_sizes=image_sizes,
                score_threshold=0.0,
                nms_threshold=self.rpn_nms_threshold,
                pre_nms_top_n=self.rpn_pre_nms_top_n,
                post_nms_top_n=self.rpn_post_nms_top_n,
                min_size=self.rpn_min_size,
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            )
            _timing_mark("generate_proposals")
            if timing_enabled:
                proposal_counts = [int(len(p)) for p in proposals_xyxy]
                print(
                    f"[first-batch timing] proposals_after_cap: "
                    f"images={len(proposal_counts)} total={sum(proposal_counts)} "
                    f"min={min(proposal_counts) if proposal_counts else 0} "
                    f"max={max(proposal_counts) if proposal_counts else 0}",
                    flush=True,
                )
            # RPN-only ROI assignment stats for logs (matches inference-time proposals; not inflated by add_gt)
            roi_diag_list: List[Dict[str, float]] = []
            for img_idx in range(len(images)):
                if len(gt_boxes_list[img_idx]) == 0:
                    continue
                roi_diag_list.append(
                    compute_roi_matching_diagnostics(
                        xyxy_to_obb(proposals_xyxy[img_idx]),
                        gt_boxes_list[img_idx],
                        gt_labels_list[img_idx],
                        device=images_tensor.device,
                        positive_iou_threshold=self.roi_positive_iou_threshold,
                        negative_iou_threshold=self.roi_negative_iou_threshold,
                        use_hbb_for_matching=self.use_hbb_for_matching,
                        match_low_quality=self.roi_match_low_quality,
                        roi_min_pos_iou=self.roi_min_pos_iou,
                        fg_bg_sampling_ratio=self.roi_fg_bg_sampling_ratio,
                        batch_size_per_image=self.roi_batch_size_per_image,
                    )
                )
            _timing_mark("roi_matching_diagnostics")
            # Optionally add GT boxes as proposals (MMRotate-style) so RoI head always sees positive samples
            if self.add_gt_as_proposals:
                for img_idx in range(len(proposals_xyxy)):
                    gt_boxes = gt_boxes_list[img_idx]
                    if len(gt_boxes) > 0:
                        gt_xyxy = obb_to_xyxy_gpu(
                            gt_boxes.to(device=proposals_xyxy[img_idx].device, dtype=proposals_xyxy[img_idx].dtype)
                        )
                        proposals_xyxy[img_idx] = torch.cat([proposals_xyxy[img_idx], gt_xyxy], dim=0)
            _timing_mark("add_gt_proposals")
            
            # ROI forward
            # Track which image each proposal belongs to
            box_to_image = []
            all_proposals = []
            for img_idx, img_proposals in enumerate(proposals_xyxy):
                num_proposals = len(img_proposals)
                box_to_image.extend([img_idx] * num_proposals)
                all_proposals.append(img_proposals)
            
            if len(all_proposals) > 0:
                all_proposals_tensor = torch.cat(all_proposals, dim=0).detach()  # [total,4] xyxy
                box_to_image_tensor = torch.tensor(box_to_image, dtype=torch.long, device=all_proposals_tensor.device)
                
                # ROI align
                roi_features = self.roi_align(
                    feature_maps=feature_list,
                    boxes_xyxy=all_proposals_tensor,
                    image_sizes=image_sizes,
                    box_to_image=box_to_image_tensor,
                    fpn_strides_override=fpn_strides_live,
                    spatial_scales_override=roi_spatial_scales_live,
                )  # [total_proposals, C, roi_h, roi_w]
                _timing_mark("roi_align")
                
                # Flatten ROI features
                roi_features_flat = roi_features.view(roi_features.shape[0], -1)  # [total_proposals, C*roi_h*roi_w]
                
                # ROI head
                class_logits, box_regression = self.roi_head(roi_features_flat)
                _timing_mark("roi_head")
                
                # Compute ROI loss
                # Group proposals by image
                roi_losses_list = []
                start_idx = 0
                for img_idx in range(len(images)):
                    # Find proposals for this image
                    img_mask = box_to_image_tensor == img_idx
                    img_proposal_indices = img_mask.nonzero(as_tuple=True)[0]
                    
                    if len(img_proposal_indices) > 0:
                        img_proposals = all_proposals_tensor[img_proposal_indices]
                        img_class_logits = class_logits[img_proposal_indices]
                        img_box_regression = box_regression[img_proposal_indices]
                        img_gt_boxes = gt_boxes_list[img_idx]
                        img_gt_labels = gt_labels_list[img_idx]
                        
                        if len(img_gt_boxes) > 0:
                            roi_losses = compute_horizontal_roi_loss_mmrotate(
                                class_logits=img_class_logits,
                                box_regression=img_box_regression,
                                proposals_xyxy=img_proposals,
                                gt_boxes=img_gt_boxes,
                                gt_labels=img_gt_labels,
                                gt_boxes_ignore=gt_boxes_ignore_list[img_idx],
                                positive_iou_threshold=self.roi_positive_iou_threshold,
                                negative_iou_threshold=self.roi_negative_iou_threshold,
                                box_reg_weight=1.0,
                                box_reg_angle_weight=self.roi_box_reg_angle_weight,
                                box_reg_iou_weight=self.roi_box_reg_iou_weight,
                                box_reg_iou_loss_type=self.roi_box_reg_iou_loss_type,
                                box_reg_kfiou_fun=self.roi_box_reg_kfiou_fun,
                                box_reg_probiou_mode=self.roi_box_reg_probiou_mode,
                                fg_bg_sampling_ratio=self.roi_fg_bg_sampling_ratio,
                                batch_size_per_image=self.roi_batch_size_per_image,
                                num_classes=self.num_classes,
                                loss_type=self.roi_loss_type,
                                class_weights=self.roi_class_weights_tensor,
                                focal_alpha=self.roi_focal_alpha,
                                focal_gamma=self.roi_focal_gamma,
                                means=self.target_means,
                                stds=self.target_stds,
                                norm_factor=self.roi_norm_factor,
                                edge_swap=self.roi_edge_swap,
                                label_smoothing=self.roi_label_smoothing,
                                match_low_quality=self.roi_match_low_quality,
                                roi_min_pos_iou=self.roi_min_pos_iou,
                                include_assignment_diagnostics=False,
                                **self.roi_grouped_ce_kwargs(),
                            )
                            roi_losses_list.append(roi_losses)
                _timing_mark("roi_loss")
                
                # Aggregate ROI losses
                if roi_losses_list:
                    roi_losses = {}
                    for key in roi_losses_list[0].keys():
                        values = [l[key] for l in roi_losses_list if key in l]
                        if not values:
                            continue
                        if torch.is_tensor(values[0]):
                            roi_losses[key] = torch.stack(values).mean()
                        else:
                            roi_losses[key] = float(sum(values) / len(values))
                else:
                    # Maintain gradient flow: compute zero losses from model outputs
                    # Use ROI head outputs to maintain connection to computation graph
                    if len(all_proposals) > 0 and len(all_proposals[0]) > 0 and class_logits.numel() > 0:
                        # If we have proposals but no matches, use ROI head outputs
                        dummy_loss_cls = (class_logits[0:1] * 0.0).sum()
                        dummy_loss_reg = (box_regression[0:1] * 0.0).sum() if box_regression.numel() > 0 else dummy_loss_cls
                        roi_losses = {
                            "loss_classifier": dummy_loss_cls,
                            "loss_box_reg": dummy_loss_reg,
                        }
                    else:
                        # No proposals at all - use RPN outputs to maintain gradient flow
                        if len(objectness_logits) > 0 and objectness_logits[0].numel() > 0:
                            dummy_loss = (objectness_logits[0] * 0.0).sum()
                        else:
                            dummy_loss = torch.tensor(0.0, device=images_tensor.device, requires_grad=True)
                        roi_losses = {
                            "loss_classifier": dummy_loss,
                            "loss_box_reg": dummy_loss,
                        }
            else:
                # No proposals generated - maintain gradient flow using RPN outputs
                if len(objectness_logits) > 0 and objectness_logits[0].numel() > 0:
                    dummy_loss = (objectness_logits[0] * 0.0).sum()
                else:
                    dummy_loss = torch.tensor(0.0, device=images_tensor.device, requires_grad=True)
                roi_losses = {
                    "loss_classifier": dummy_loss,
                    "loss_box_reg": dummy_loss,
                }
            
            if roi_diag_list:
                diag_agg = {
                    k: float(sum(d[k] for d in roi_diag_list) / len(roi_diag_list))
                    for k in roi_diag_list[0]
                }
                roi_losses.update(diag_agg)

            # Combine losses
            return {
                **rpn_losses,
                **roi_losses,
            }
        
        else:
            from .faster_rcnn_inference import faster_rcnn_inference

            return faster_rcnn_inference(self, images)


class OrientedRCNN(ClassWeightsMixin, GroupedCeMixin, nn.Module):
    """Oriented R-CNN (MMRotate-style): horizontal RPN + MidpointOffset (6 params) -> oriented ROI head.

    RPN ``anchor_angles`` is optional on the constructor only (not in training JSON); default is horizontal ``[0.0]``.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        backbone=None,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = False,
        trainable_layers: int = 5,
        anchor_scales: Optional[List[float]] = None,
        anchor_ratios: Optional[List[float]] = None,
        anchor_angles: Optional[List[float]] = None,
        rpn_pre_nms_top_n: int = 2000,
        rpn_post_nms_top_n: int = 1000,
        max_detections_per_image: int = 100,  # Maximum number of detections per image after NMS
        rpn_nms_threshold: float = 0.7,  # IoU threshold for RPN proposal NMS
        final_nms_iou_threshold: float = 0.5,  # Final detection NMS (after ROI head); not rpn_nms_threshold
        rpn_positive_iou_threshold: float = 0.5,
        rpn_negative_iou_threshold: float = 0.2,
        rpn_fg_bg_sampling_ratio: float = 0.5,
        rpn_batch_size_per_image: int = 256,  # Number of RPN anchors to sample per image (MMRotate: 256)
        rpn_min_pos_iou: float = 0.3,
        rpn_match_low_quality: bool = True,
        roi_match_low_quality: bool = False,
        roi_positive_iou_threshold: float = 0.4,
        roi_negative_iou_threshold: float = 0.3,
        roi_fg_bg_sampling_ratio: float = 0.25,
        roi_batch_size_per_image: int = 512,  # Number of ROI proposals to sample per image for training
        roi_output_size: Tuple[int, int] = (7, 7),
        roi_spatial_scales: Optional[List[float]] = None,
        fpn_strides: Optional[List[int]] = None,
        returned_layers: Optional[List[int]] = None,
        use_checkpoint: bool = False,
        roi_chunk_size: int = 32,
        roi_use_checkpoint: bool = False,
        roi_loss_type: str = "cross_entropy",
        roi_class_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
        roi_focal_alpha: float = 1.0,
        roi_focal_gamma: float = 2.0,
        roi_label_smoothing: float = 0.0,
        target_means: Optional[Tuple[float, float, float, float, float]] = None,
        target_stds: Optional[Tuple[float, float, float, float, float]] = None,
        roi_norm_factor: Optional[float] = None,
        roi_edge_swap: bool = True,
        roi_proj_xy: bool = True,
        roi_box_reg_angle_weight: float = 1.0,
        roi_box_reg_angle_schedule_epochs: Optional[List[int]] = None,
        roi_box_reg_angle_schedule_values: Optional[List[float]] = None,
        roi_box_reg_iou_weight: float = 0.0,
        roi_box_reg_iou_loss_type: str = "riou",
        roi_box_reg_kfiou_fun: Optional[str] = None,
        roi_box_reg_probiou_mode: Optional[str] = None,
        use_hbb_for_matching: bool = False,
        inference_pre_nms_score_threshold: float = 0.05,
        final_nms_iou_schedule_epochs: Optional[List[int]] = None,
        final_nms_iou_schedule_values: Optional[List[float]] = None,
        roi_box_reg_iou_schedule_epochs: Optional[List[int]] = None,
        roi_box_reg_iou_schedule_values: Optional[List[float]] = None,
        nms_class_agnostic: bool = False,
        final_nms_use_cpu: bool = False,
        roi_inference_top_class_only: bool = False,
        add_gt_as_proposals: bool = True,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for OrientedRCNN.")
        # Initialize ClassWeightsMixin first, then nn.Module
        # ClassWeightsMixin.__init__() sets self.num_classes
        super().__init__(num_classes, roi_class_weights)
        nn.Module.__init__(self)
        self._init_grouped_ce()
        self.target_means = target_means if target_means is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
        self.target_stds = target_stds if target_stds is not None else (1.0, 1.0, 1.0, 1.0, 1.0)
        self.roi_norm_factor = roi_norm_factor
        self.roi_edge_swap = roi_edge_swap
        self.roi_proj_xy = roi_proj_xy
        self._roi_box_reg_angle_schedule_epochs = roi_box_reg_angle_schedule_epochs
        self._roi_box_reg_angle_schedule_values = roi_box_reg_angle_schedule_values
        self._roi_box_reg_angle_weight_default = float(roi_box_reg_angle_weight)
        self.roi_box_reg_angle_weight = float(roi_box_reg_angle_weight)
        self.set_roi_box_reg_angle_weight_for_epoch(0)
        self.roi_box_reg_iou_loss_type = roi_box_reg_iou_loss_type
        self.roi_box_reg_kfiou_fun = roi_box_reg_kfiou_fun
        self.roi_box_reg_probiou_mode = roi_box_reg_probiou_mode
        self.use_hbb_for_matching = use_hbb_for_matching
        self.inference_pre_nms_score_threshold = inference_pre_nms_score_threshold
        self.roi_loss_type = roi_loss_type
        self.roi_focal_alpha = roi_focal_alpha
        self.roi_focal_gamma = roi_focal_gamma
        self.roi_label_smoothing = roi_label_smoothing

        # Setup backbone using shared utility (returned_layers=[2,3,4] for MMRotate-style C3–C5 only)
        self.backbone, backbone_channels = setup_backbone(
            backbone=backbone,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
        )
        if fpn_strides is None and returned_layers == [2, 3, 4]:
            fpn_strides = [8, 16, 32, 64]
        
        self.anchor_scales, self.anchor_ratios, self.anchor_angles, self.num_anchors = setup_anchors(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_angles=anchor_angles,
            default_angles=[0.0],
        )
        self.rpn_head = OrientedRPNHead(
            backbone_channels, num_anchors=self.num_anchors, cls_out_channels=1, reg_out_channels=6
        )
        self.roi_align = OrientedROIAlign(
            output_size=roi_output_size,
            spatial_scales=roi_spatial_scales,
            fpn_strides=fpn_strides,
            chunk_size=roi_chunk_size,
            use_checkpoint=roi_use_checkpoint,
        )
        roi_feature_size = roi_output_size[0] * roi_output_size[1] * backbone_channels
        self.roi_class_agnostic_regression = True
        self.roi_head = OrientedROIHead(
            in_channels=roi_feature_size,
            num_classes=num_classes,
            class_agnostic_regression=True,
        )
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.max_detections_per_image = max_detections_per_image
        self.rpn_nms_threshold = rpn_nms_threshold
        self.final_nms_iou_threshold = final_nms_iou_threshold
        self.nms_class_agnostic = nms_class_agnostic
        self.final_nms_use_cpu = final_nms_use_cpu
        self.roi_inference_top_class_only = roi_inference_top_class_only
        self.rpn_positive_iou_threshold = rpn_positive_iou_threshold
        self.rpn_negative_iou_threshold = rpn_negative_iou_threshold
        self.rpn_fg_bg_sampling_ratio = rpn_fg_bg_sampling_ratio
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_min_pos_iou = rpn_min_pos_iou
        self.rpn_match_low_quality = rpn_match_low_quality
        self.roi_match_low_quality = roi_match_low_quality
        self.roi_positive_iou_threshold = roi_positive_iou_threshold
        self.roi_negative_iou_threshold = roi_negative_iou_threshold
        self.roi_fg_bg_sampling_ratio = roi_fg_bg_sampling_ratio
        self.roi_batch_size_per_image = roi_batch_size_per_image
        self.fpn_strides = fpn_strides or [4, 8, 16, 32, 64]
        self.roi_spatial_scales = roi_spatial_scales or [1.0 / s for s in self.fpn_strides]
        self.use_checkpoint = use_checkpoint
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values
        self.add_gt_as_proposals = add_gt_as_proposals
        self._roi_box_reg_iou_schedule_epochs = roi_box_reg_iou_schedule_epochs
        self._roi_box_reg_iou_schedule_values = roi_box_reg_iou_schedule_values
        self._roi_box_reg_iou_weight_default = float(roi_box_reg_iou_weight)
        self.roi_box_reg_iou_weight = float(roi_box_reg_iou_weight)
        self.set_roi_box_reg_iou_weight_for_epoch(0)

    def set_final_nms_iou_for_epoch(self, epoch: int) -> None:
        """Update final detection NMS IoU threshold from schedule. Lower = more aggressive suppression."""
        if self._final_nms_iou_schedule_epochs is None or self._final_nms_iou_schedule_values is None:
            return
        if not self._final_nms_iou_schedule_epochs or not self._final_nms_iou_schedule_values:
            return
        idx = 0
        for boundary in self._final_nms_iou_schedule_epochs:
            if epoch < boundary:
                break
            idx += 1
        idx = min(idx, len(self._final_nms_iou_schedule_values) - 1)
        self.final_nms_iou_threshold = self._final_nms_iou_schedule_values[idx]

    def set_roi_box_reg_iou_weight_for_epoch(self, epoch: int) -> None:
        """Update ROI auxiliary IoU loss weight from schedule (0-based epoch index)."""
        from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule

        self.roi_box_reg_iou_weight = resolve_piecewise_schedule(
            epoch,
            self._roi_box_reg_iou_schedule_epochs,
            self._roi_box_reg_iou_schedule_values,
            self._roi_box_reg_iou_weight_default,
        )

    def set_roi_box_reg_angle_weight_for_epoch(self, epoch: int) -> None:
        """Update ROI angle (5th dim) SmoothL1 weight from schedule (0-based epoch index)."""
        from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule

        self.roi_box_reg_angle_weight = resolve_piecewise_schedule(
            epoch,
            self._roi_box_reg_angle_schedule_epochs,
            self._roi_box_reg_angle_schedule_values,
            self._roi_box_reg_angle_weight_default,
        )

    def forward(
        self,
        images: Sequence[torch.Tensor],
        targets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # Extract features using shared utility. include_pool_level keeps torchvision's
        # P6 (stride-64 max-pool level) so the RPN sees 5 levels like MMRotate.
        feature_list = extract_backbone_features(
            self.backbone,
            images,
            use_checkpoint=self.use_checkpoint,
            training=self.training,
            include_pool_level=True,
        )
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
        fpn_strides_live = derive_fpn_strides_from_grid(image_sizes[0], feature_map_sizes)
        warn_if_fpn_strides_mismatch(self.fpn_strides, fpn_strides_live)
        roi_spatial_scales_live = [1.0 / s for s in fpn_strides_live]
        
        # Get device from first feature map (needed for device reference)
        images_tensor = torch.stack(images, dim=0)

        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            gt_boxes_list, gt_labels_list, gt_boxes_ignore_list = prepare_targets(targets, device=images_tensor.device)
            obj_logits, bbox_reg = self.rpn_head(feature_list)
            img_h, img_w = image_sizes[0]
            anchors = generate_oriented_anchors(
                image_size=(img_h, img_w),
                feature_map_sizes=feature_map_sizes,
                anchor_scales=self.anchor_scales,
                anchor_ratios=self.anchor_ratios,
                anchor_angles=self.anchor_angles,
                stride_per_level=fpn_strides_live,
            )
            rpn_losses = compute_midpoint_rpn_loss(
                objectness_logits=obj_logits,
                bbox_regression=bbox_reg,
                anchors=anchors,
                gt_boxes=gt_boxes_list,
                gt_boxes_ignore=gt_boxes_ignore_list,
                image_sizes=image_sizes,
                positive_iou_threshold=self.rpn_positive_iou_threshold,
                negative_iou_threshold=self.rpn_negative_iou_threshold,
                box_reg_weight=1.0,
                fg_bg_sampling_ratio=self.rpn_fg_bg_sampling_ratio,
                batch_size_per_image=self.rpn_batch_size_per_image,
                use_hbb_for_matching=self.use_hbb_for_matching,
                min_pos_iou=self.rpn_min_pos_iou,
                match_low_quality=self.rpn_match_low_quality,
                sample_from_all_levels=True,
            )
            proposals_list = generate_midpoint_proposals(
                obj_logits,
                bbox_reg,
                anchors,
                image_sizes,
                score_threshold=self.inference_pre_nms_score_threshold,
                nms_threshold=self.rpn_nms_threshold,
                pre_nms_top_n=self.rpn_pre_nms_top_n,
                post_nms_top_n=self.rpn_post_nms_top_n,
                min_size=0.0,
            )
            # Optionally add GT boxes as proposals (MMRotate default).
            if self.add_gt_as_proposals:
                for img_idx in range(len(proposals_list)):
                    gt = gt_boxes_list[img_idx]
                    if gt.numel() > 0:
                        proposals_list[img_idx] = torch.cat([proposals_list[img_idx], gt.to(proposals_list[img_idx].device)], dim=0)
            box_to_img = torch.cat(
                [torch.full((p.shape[0],), i, dtype=torch.long, device=p.device) for i, p in enumerate(proposals_list)]
            ) if any(p.shape[0] > 0 for p in proposals_list) else torch.zeros(0, dtype=torch.long, device=images_tensor.device)
            oriented_proposals = torch.cat(proposals_list, dim=0).detach() if box_to_img.numel() > 0 else torch.zeros((0, 5), dtype=torch.float32, device=images_tensor.device)
            # Clamp proposal w/h to image scale so ROI loss does not explode on degenerate boxes
            min_side = 1.0
            for img_idx in range(len(image_sizes)):
                img_h, img_w = image_sizes[img_idx]
                max_side = float(max(img_w, img_h)) * 2.0
                mask = box_to_img == img_idx
                if mask.any():
                    oriented_proposals[mask, 2] = oriented_proposals[mask, 2].clamp(min=min_side, max=max_side)
                    oriented_proposals[mask, 3] = oriented_proposals[mask, 3].clamp(min=min_side, max=max_side)
            if oriented_proposals.shape[0] == 0:
                roi_losses = {"loss_classifier": torch.tensor(0.0, device=images_tensor.device), "loss_box_reg": torch.tensor(0.0, device=images_tensor.device)}
                return {**rpn_losses, **roi_losses}
            box_to_img = box_to_img.to(device=oriented_proposals.device)
            roi_feats = self.roi_align(
                feature_maps=feature_list,
                boxes=oriented_proposals,
                image_sizes=image_sizes,
                box_to_image=box_to_img,
                fpn_strides_override=fpn_strides_live,
                spatial_scales_override=roi_spatial_scales_live,
            )
            class_logits, box_reg = self.roi_head(roi_feats.view(roi_feats.shape[0], -1))
            roi_losses_list = []
            for img_idx in range(len(images)):
                mask = box_to_img == img_idx
                if mask.sum() == 0:
                    continue
                img_proposals = oriented_proposals[mask]
                img_logits = class_logits[mask]
                img_reg = box_reg[mask]
                img_gt = gt_boxes_list[img_idx]
                img_labels = gt_labels_list[img_idx]
                if len(img_gt) > 0:
                    roi_losses_list.append(
                        compute_oriented_roi_loss(
                            class_logits=img_logits, box_regression=img_reg, proposals=img_proposals,
                            gt_boxes=img_gt, gt_labels=img_labels,
                            gt_boxes_ignore=gt_boxes_ignore_list[img_idx],
                            positive_iou_threshold=self.roi_positive_iou_threshold,
                            negative_iou_threshold=self.roi_negative_iou_threshold,
                            box_reg_weight=1.0,
                            box_reg_angle_weight=self.roi_box_reg_angle_weight,
                            fg_bg_sampling_ratio=self.roi_fg_bg_sampling_ratio,
                            batch_size_per_image=self.roi_batch_size_per_image,
                            num_classes=self.num_classes, loss_type=self.roi_loss_type,
                            class_weights=self.roi_class_weights_tensor,
                            focal_alpha=self.roi_focal_alpha, focal_gamma=self.roi_focal_gamma,
                            target_means=self.target_means, target_stds=self.target_stds,
                            norm_factor=self.roi_norm_factor, edge_swap=self.roi_edge_swap,
                            proj_xy=self.roi_proj_xy,
                            box_reg_iou_weight=self.roi_box_reg_iou_weight,
                            box_reg_iou_loss_type=self.roi_box_reg_iou_loss_type,
                            box_reg_kfiou_fun=self.roi_box_reg_kfiou_fun,
                            box_reg_probiou_mode=self.roi_box_reg_probiou_mode,
                            use_hbb_for_matching=self.use_hbb_for_matching,
                            match_low_quality=self.roi_match_low_quality,
                            label_smoothing=self.roi_label_smoothing,
                            **self.roi_grouped_ce_kwargs(),
                        )
                    )
            if roi_losses_list:
                roi_losses = {}
                for key in roi_losses_list[0].keys():
                    values = [l[key] for l in roi_losses_list if key in l]
                    if not values:
                        continue
                    if torch.is_tensor(values[0]):
                        roi_losses[key] = torch.stack(values).mean()
                    else:
                        roi_losses[key] = float(sum(values) / len(values))
            else:
                roi_losses = {"loss_classifier": torch.tensor(0.0, device=images_tensor.device), "loss_box_reg": torch.tensor(0.0, device=images_tensor.device)}
            return {**rpn_losses, **roi_losses}

        obj_logits, bbox_reg = self.rpn_head(feature_list)
        img_h, img_w = image_sizes[0]
        anchors = generate_oriented_anchors(
            image_size=(img_h, img_w),
            feature_map_sizes=feature_map_sizes,
            anchor_scales=self.anchor_scales,
            anchor_ratios=self.anchor_ratios,
            anchor_angles=self.anchor_angles,
            stride_per_level=fpn_strides_live,
        )
        proposals_list = generate_midpoint_proposals(
            obj_logits,
            bbox_reg,
            anchors,
            image_sizes,
            score_threshold=0.0,
            nms_threshold=self.rpn_nms_threshold,
            pre_nms_top_n=self.rpn_pre_nms_top_n,
            post_nms_top_n=self.rpn_post_nms_top_n,
            min_size=0.0,
        )
        return_debug = getattr(self, '_return_anchors_proposals', False)
        if return_debug:
            all_anchors = torch.cat(anchors, dim=0).detach().cpu()
        outputs = []
        from ..ops.rotated_ops import rotated_nms
        for img_idx, oriented in enumerate(proposals_list):
            if oriented.shape[0] == 0:
                out = {"rboxes": [], "labels": torch.zeros(0, dtype=torch.int64, device=images_tensor.device), "scores": torch.zeros(0, dtype=torch.float32, device=images_tensor.device)}
                if return_debug:
                    out["anchors"] = all_anchors
                    out["proposals"] = torch.zeros((0, 5), dtype=torch.float32)
                outputs.append(out)
                continue
            # Clamp proposal w/h to image scale (same as training) so refined boxes stay sane
            ih, iw = image_sizes[img_idx]
            max_side = float(max(iw, ih)) * 2.0
            oriented[:, 2] = oriented[:, 2].clamp(min=1.0, max=max_side)
            oriented[:, 3] = oriented[:, 3].clamp(min=1.0, max=max_side)
            box_to_img = torch.full((oriented.shape[0],), img_idx, dtype=torch.long, device=oriented.device)
            roi_feats = self.roi_align(
                feature_maps=feature_list,
                boxes=oriented,
                image_sizes=image_sizes,
                box_to_image=box_to_img,
                fpn_strides_override=fpn_strides_live,
                spatial_scales_override=roi_spatial_scales_live,
            )
            class_logits, box_reg = self.roi_head(roi_feats.view(roi_feats.shape[0], -1))
            class_probs = F.softmax(class_logits, dim=1)
            fg_probs = class_probs[:, 1:]
            score_threshold = self.inference_pre_nms_score_threshold
            if self.roi_inference_top_class_only:
                max_scores, argmax_cls = fg_probs.max(dim=1)
                keep_prop = max_scores > score_threshold
                if keep_prop.any():
                    proposal_indices = keep_prop.nonzero(as_tuple=True)[0]
                    filtered_proposals = oriented[proposal_indices]
                    filtered_scores = max_scores[proposal_indices]
                    filtered_class_indices = argmax_cls[proposal_indices]
                    filtered_labels = filtered_class_indices + 1
                    filtered_box_reg = box_reg[proposal_indices]
                else:
                    filtered_proposals = None
            else:
                candidate_mask = fg_probs > score_threshold  # [N, C]
                if candidate_mask.any():
                    proposal_indices, class_indices = candidate_mask.nonzero(as_tuple=True)
                    filtered_proposals = oriented[proposal_indices]
                    filtered_scores = fg_probs[proposal_indices, class_indices]
                    filtered_labels = class_indices + 1
                    filtered_box_reg = box_reg[proposal_indices]
                    filtered_class_indices = class_indices
                else:
                    filtered_proposals = None
            total_candidates = int(fg_probs.numel())
            from ..utils.logging import logger
            _roi_mode = "top_class" if self.roi_inference_top_class_only else "multiclass"
            logger.trace(
                "ROI inference candidates: mode={}, pre_nms_thr={:.3f}, total={}, kept_after_thr={}",
                _roi_mode,
                score_threshold,
                total_candidates,
                0 if filtered_proposals is None else int(filtered_proposals.shape[0]),
            )
            if filtered_proposals is not None:
                if self.roi_class_agnostic_regression:
                    selected_reg = filtered_box_reg
                else:
                    filtered_box_reg = filtered_box_reg.view(len(filtered_proposals), self.num_classes, 5)
                    selected_reg = filtered_box_reg[
                        torch.arange(len(filtered_proposals), device=filtered_proposals.device),
                        filtered_class_indices
                    ]
                refined = decode_oriented_boxes(
                    filtered_proposals, selected_reg,
                    target_means=self.target_means, target_stds=self.target_stds,
                    norm_factor=self.roi_norm_factor, edge_swap=self.roi_edge_swap,
                    proj_xy=self.roi_proj_xy,
                )
                if self.nms_class_agnostic:
                    keep = rotated_nms(
                        refined,
                        filtered_scores,
                        self.final_nms_iou_threshold,
                        self.max_detections_per_image,
                        force_cpu=self.final_nms_use_cpu,
                    )
                    if len(keep) > 0:
                        k_scores = filtered_scores[keep]
                        _, order = k_scores.sort(descending=True)
                        keep = keep[order]
                    kept_per_class = {"agnostic": int(len(keep))}
                else:
                    # Per-class cap is an optimization; final per-image cap applied after merge.
                    keep_per_class = []
                    kept_per_class = {}
                    for cls in filtered_labels.unique():
                        cls_mask = filtered_labels == cls
                        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
                        cls_keep = rotated_nms(
                            refined[cls_mask],
                            filtered_scores[cls_mask],
                            self.final_nms_iou_threshold,
                            self.max_detections_per_image,
                            force_cpu=self.final_nms_use_cpu,
                        )
                        kept_per_class[int(cls.item())] = int(len(cls_keep))
                        keep_per_class.append(cls_indices[cls_keep])
                    if keep_per_class:
                        keep = torch.cat(keep_per_class, dim=0)
                        keep_scores = filtered_scores[keep]
                        _, keep_order = keep_scores.sort(descending=True)
                        keep = keep[keep_order]
                        if self.max_detections_per_image is not None:
                            keep = keep[:self.max_detections_per_image]
                    else:
                        keep = torch.zeros((0,), dtype=torch.long, device=filtered_scores.device)
                final_boxes = refined[keep]
                final_scores = filtered_scores[keep]
                final_labels = filtered_labels[keep]
                logger.trace(
                    "ROI inference kept after {} NMS: total={}, per_class={}",
                    "class-agnostic" if self.nms_class_agnostic else "class-wise",
                    int(len(keep)),
                    kept_per_class,
                )
                valid = (final_boxes[:, 2] >= 1.0) & (final_boxes[:, 3] >= 1.0)
                final_boxes, final_scores, final_labels = final_boxes[valid], final_scores[valid], final_labels[valid]
                out = {"rboxes": tensor_to_rboxes(final_boxes), "labels": final_labels, "scores": final_scores}
                if return_debug:
                    out["anchors"] = all_anchors
                    out["proposals"] = oriented.detach().cpu()
                outputs.append(out)
            else:
                out = {"rboxes": [], "labels": torch.zeros(0, dtype=torch.int64, device=images_tensor.device), "scores": torch.zeros(0, dtype=torch.float32, device=images_tensor.device)}
                if return_debug:
                    out["anchors"] = all_anchors
                    out["proposals"] = oriented.detach().cpu()
                outputs.append(out)
        return outputs


__all__ = ["RotatedFasterRCNN", "OrientedRCNN"]