"""Complete Rotated RetinaNet model for true oriented object detection.

This module implements a full single-stage oriented detector that:
- Predicts oriented bounding boxes with 5 parameters (cx, cy, w, h, angle)
- Uses oriented anchors, oriented IoU matching, and oriented NMS
- Preserves angle information throughout training and inference
- Uses sigmoid focal loss for classification (MMRotate FocalLoss use_sigmoid=True)
"""

from __future__ import annotations

import math
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
from ..ops import nms
from ..ops.kfiou import mean_auxiliary_box_reg_loss
from ..ops.rotated_ops import rotated_nms
from .oriented_rpn import (
    generate_oriented_anchors,
    encode_oriented_boxes,
    decode_oriented_boxes,
    match_oriented_anchors_to_gt,
)
from .utils import (
    rboxes_to_tensor,
    tensor_to_rboxes,
    prepare_targets,
    setup_backbone,
    extract_backbone_features,
    setup_anchors,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
)


class OrientedRetinaNetHead(nn.Module):
    """Rotated RetinaNet head (MMRotate-style separate cls/reg subnets).

    Two independent 3x3 conv towers (``stacked_convs`` each) feed 3x3 prediction
    heads. Classification uses sigmoid focal loss (``use_sigmoid=True``); regression
    outputs 5 parameters per anchor.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int,
        stacked_convs: int = 4,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for OrientedRetinaNetHead.")
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.stacked_convs = max(1, int(stacked_convs))

        self.cls_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(self.stacked_convs)
        ])
        self.reg_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(self.stacked_convs)
        ])

        # MMRotate: 3x3 prediction convs (not 1x1).
        self.conv_cls = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.conv_bbox = nn.Conv2d(
            in_channels, num_anchors * 5, kernel_size=3, stride=1, padding=1
        )

        for layer in list(self.cls_convs) + list(self.reg_convs) + [self.conv_bbox]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.conv_cls.weight, std=0.01)
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.conv_cls.bias, bias_init)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        classification_logits = []
        bbox_regression = []

        for feat in features:
            cls_x = feat
            for conv in self.cls_convs:
                cls_x = F.relu(conv(cls_x))
            reg_x = feat
            for conv in self.reg_convs:
                reg_x = F.relu(conv(reg_x))

            classification_logits.append(self.conv_cls(cls_x))
            bbox_regression.append(self.conv_bbox(reg_x))

        return classification_logits, bbox_regression



def sigmoid_focal_loss_sum(
    logits: torch.Tensor,
    targets_onehot: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss (RetinaNet / MMRotate ``FocalLoss(use_sigmoid=True)``), sum reduction.
    
    Each class is an independent binary classifier; ``alpha`` weights the positive
    (target=1) entries and ``1 - alpha`` the negative entries. The caller is expected
    to normalize the returned sum by the number of positive anchors (MMDet avg_factor).
    
    Args:
        logits: [N, num_classes] raw classification logits.
        targets_onehot: [N, num_classes] binary targets (1.0 at the GT class of
            positive anchors, all zeros for background anchors).
        alpha: Balance weight for positive entries (default 0.25).
        gamma: Focusing parameter (default 2.0).
    
    Returns:
        Scalar tensor: sum of focal loss over all entries.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    pt = p * targets_onehot + (1 - p) * (1 - targets_onehot)
    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
    loss = alpha_t * (1 - pt).pow(gamma) * ce
    return loss.sum()


def compute_oriented_retinanet_loss(
    classification_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    num_classes: int,
    gt_boxes_ignore: Optional[List[torch.Tensor]] = None,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.4,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    box_reg_weight: float = 1.0,
    target_means: Optional[Tuple[float, float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float, float]] = None,
    target_norm_factor: Optional[float] = None,
    norm_factor: Optional[float] = None,
    edge_swap: bool = False,
    box_reg_iou_weight: float = 0.0,
    box_reg_iou_loss_type: str = "riou",
    box_reg_kfiou_fun: Optional[str] = None,
    box_reg_probiou_mode: Optional[str] = None,
    use_hbb_for_matching: bool = False,
    box_reg_loss_type: str = "smooth_l1",
) -> Dict[str, torch.Tensor]:
    """Compute Rotated RetinaNet losses for oriented object detection.
    
    This function processes anchors per-level to avoid memory issues from concatenating
    millions of anchors. Each level is processed independently, and losses are accumulated.
    
    Args:
        classification_logits: List of classification logits from Rotated RetinaNet head
        bbox_regression: List of box regression predictions from Rotated RetinaNet head
        anchors: List of anchor tensors for each level
        gt_boxes: List of ground truth boxes per image (each as [M, 5] tensor)
        gt_labels: List of ground truth labels per image (each as [M] tensor, 1-indexed)
        image_sizes: List of (height, width) for each image
        num_classes: Number of object classes (excluding background)
        positive_iou_threshold: IoU threshold for positive anchors
        negative_iou_threshold: IoU threshold for negative anchors
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        box_reg_weight: Weight for box regression loss
        target_means: Optional means for target normalization
        target_stds: Optional stds for target normalization
        target_norm_factor: Optional angle scale for loss (when norm_factor used in encode; default None)
        norm_factor: Optional angle scaling for encode (MMRotate Rotated RetinaNet uses None)
        edge_swap: Whether to use edge_swap in bbox encode/decode (MMRotate uses True)
        box_reg_iou_weight: Extra weight for auxiliary loss on decoded positive anchors.
        box_reg_iou_loss_type: ``\"riou\"``, ``\"kfiou\"``, or ``\"probiou\"`` (see :func:`~oriented_det.ops.kfiou.mean_auxiliary_box_reg_loss`).
        box_reg_kfiou_fun: Optional KFIoU overlap transform when using ``kfiou``.
        box_reg_probiou_mode: ``l1`` (default) or ``l2`` when using ``probiou``.
        use_hbb_for_matching: If True, use HBB (axis-aligned) IoU for anchor-GT matching (recommended for single angle).
        box_reg_loss_type: ``smooth_l1`` (default) or ``l1`` (MMRotate RetinaNet).
    
    Returns:
        Dictionary with loss values:
        - "loss_classifier": Classification loss (sigmoid focal loss, normalized by the number
          of positive anchors); same TensorBoard name as two-stage detectors
        - "loss_box_reg": Box regression loss (smooth L1)
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation.")
    
    device = classification_logits[0].device if classification_logits else torch.device('cpu')
    num_images = len(gt_boxes)
    num_levels = len(classification_logits)
    
    # Process each level independently (memory-efficient approach).
    # Classification follows MMDet/MMRotate: sum of sigmoid focal loss over all levels/images,
    # normalized once by the total number of positive anchors in the batch (avg_factor).
    cls_loss_sums = []
    num_pos_total = 0
    reg_loss_sums: List[torch.Tensor] = []
    
    for level_idx in range(num_levels):
        # Get shapes for this level
        B, C_cls, H, W = classification_logits[level_idx].shape
        B_reg, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
        
        # Extract num_anchors from regression channels (5 params per anchor)
        if C_reg % 5 != 0:
            raise RuntimeError(
                f"Invalid Rotated RetinaNet format: C_reg={C_reg} must be divisible by 5"
            )
        num_anchors = C_reg // 5
        
        # Classification head has K sigmoid outputs per anchor (no background channel)
        if C_cls != num_anchors * num_classes:
            raise RuntimeError(
                f"Rotated RetinaNet format mismatch: C_cls={C_cls} but expected "
                f"num_anchors*num_classes={num_anchors}*{num_classes}"
            )
        
        # Reshape predictions: [B, num_anchors*num_classes, H, W] -> [B*H*W*num_anchors, num_classes]
        cls_logits = classification_logits[level_idx].view(B, num_anchors, num_classes, H, W)
        cls_logits = cls_logits.permute(0, 3, 4, 1, 2).contiguous().view(-1, num_classes)
        
        # Reshape regression: [B, num_anchors*5, H, W] -> [B*H*W*num_anchors, 5]
        bbox_pred = bbox_regression[level_idx].view(B, num_anchors, 5, H, W)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous().view(-1, 5)
        
        # Process anchors for this level (no gradients needed)
        with torch.no_grad():
            level_anchors_raw = anchors[level_idx]
            if isinstance(level_anchors_raw, torch.Tensor):
                level_anchors_raw = level_anchors_raw.detach().to(device)
            else:
                level_anchors_raw = torch.tensor(
                    level_anchors_raw, dtype=torch.float32, device=device, requires_grad=False
                )
            
            # Expand anchors to match batch size: [N, 5] -> [B*N, 5]
            anchors_per_image = len(level_anchors_raw)
            level_anchors = level_anchors_raw.unsqueeze(0).repeat(B, 1, 1).view(-1, 5).clone()
            level_anchors = level_anchors.detach()
            level_anchors.requires_grad_(False)
        
        # Process each image in the batch for this level
        for img_idx in range(B):
            # Get anchors for this image at this level
            start_idx = img_idx * anchors_per_image
            end_idx = (img_idx + 1) * anchors_per_image
            img_anchors = level_anchors[start_idx:end_idx]
            img_cls_logits = cls_logits[start_idx:end_idx]  # [N, num_classes]
            img_bbox_pred = bbox_pred[start_idx:end_idx]  # [N, 5]
            
            # Get ground truth boxes and labels for this image
            if isinstance(gt_boxes[img_idx], torch.Tensor):
                img_gt_boxes = gt_boxes[img_idx].to(device).detach()
            else:
                img_gt_boxes = torch.tensor(
                    gt_boxes[img_idx], dtype=torch.float32, device=device, requires_grad=False
                )
            
            if isinstance(gt_labels[img_idx], torch.Tensor):
                img_gt_labels = gt_labels[img_idx].to(device).detach()
            else:
                img_gt_labels = torch.tensor(
                    gt_labels[img_idx], dtype=torch.int64, device=device, requires_grad=False
                )
            
            # Initialize class_labels and regression_targets
            class_labels = torch.zeros(len(img_anchors), dtype=torch.int64, device=device)
            regression_targets = torch.zeros((len(img_anchors), 5), dtype=torch.float32, device=device)
            
            if len(img_gt_boxes) == 0:
                # No ground truth - all anchors are background (class 0)
                labels = torch.zeros(len(img_anchors), dtype=torch.int64, device=device)
            else:
                # Match anchors to GT (oriented IoU or HBB IoU when use_hbb_for_matching)
                img_gt_ignore = None
                if gt_boxes_ignore is not None and img_idx < len(gt_boxes_ignore):
                    img_gt_ignore = gt_boxes_ignore[img_idx].to(device).detach()
                labels, matched_indices = match_oriented_anchors_to_gt(
                    img_anchors,
                    img_gt_boxes,
                    positive_iou_threshold,
                    negative_iou_threshold,
                    device,
                    use_hbb_for_matching=use_hbb_for_matching,
                    # MMRotate RetinaNet MaxIoUAssigner: min_pos_iou=0 so every GT gets its
                    # best-overlapping anchor as positive (low-quality match), even below 0.5.
                    min_pos_iou=0.0,
                    match_low_quality=True,
                    gt_boxes_ignore=img_gt_ignore,
                    ignore_iou_threshold=positive_iou_threshold,
                )
                
                # Class labels: 0 = background, 1..K = foreground (1-indexed GT labels);
                # converted to one-hot sigmoid targets (class k -> column k-1) below.
                positive_mask = labels == 1
                if positive_mask.any():
                    matched_gt_labels = img_gt_labels[matched_indices[positive_mask]]
                    class_labels[positive_mask] = matched_gt_labels  # 1-indexed: 1..num_classes
                    
                    # Compute regression targets for positive anchors
                    matched_gt = img_gt_boxes[matched_indices[positive_mask]]
                    matched_anchors = img_anchors[positive_mask].detach()
                    regression_targets[positive_mask] = encode_oriented_boxes(
                        matched_anchors,
                        matched_gt,
                        target_means=target_means,
                        target_stds=target_stds,
                        norm_factor=norm_factor,
                        edge_swap=edge_swap,
                    )
            
            # Compute classification loss (sigmoid focal loss, sum reduction)
            # Only compute on non-ignored anchors (labels != -1)
            valid_mask = labels != -1
            if valid_mask.any():
                valid_cls_logits = img_cls_logits[valid_mask]  # [V, num_classes]
                valid_class_labels = class_labels[valid_mask]  # [V], 0 = bg, 1..K = fg
                
                # Binary one-hot targets: positive anchors get 1.0 at their GT class column,
                # background anchors are all zeros (no background channel with sigmoid).
                cls_targets = torch.zeros_like(valid_cls_logits)
                fg_mask = valid_class_labels > 0
                if fg_mask.any():
                    cls_targets[fg_mask, valid_class_labels[fg_mask] - 1] = 1.0
                
                cls_loss_sums.append(
                    sigmoid_focal_loss_sum(
                        valid_cls_logits,
                        cls_targets,
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                )
            num_pos_total += int((labels == 1).sum())
            
            # Compute regression loss on positive anchors (encoded space, MMDet avg_factor)
            positive_mask = labels == 1
            if positive_mask.any():
                positive_bbox_pred = img_bbox_pred[positive_mask]
                positive_regression_targets = regression_targets[positive_mask]
                if box_reg_loss_type == "l1":
                    reg_elem = F.l1_loss(
                        positive_bbox_pred,
                        positive_regression_targets,
                        reduction="none",
                    )
                else:
                    reg_elem = F.smooth_l1_loss(
                        positive_bbox_pred,
                        positive_regression_targets,
                        beta=1.0 / 9.0,
                        reduction="none",
                    )
                reg_loss = reg_elem.sum()
                if box_reg_iou_weight > 0.0:
                    matched_gt = img_gt_boxes[matched_indices[positive_mask]]
                    decoded_boxes = decode_oriented_boxes(
                        img_anchors[positive_mask],
                        positive_bbox_pred,
                        target_means=target_means,
                        target_stds=target_stds,
                        normalize_le90=True,
                        norm_factor=norm_factor,
                        edge_swap=edge_swap,
                    )
                    loss_iou = mean_auxiliary_box_reg_loss(
                        decoded_boxes,
                        matched_gt,
                        loss_type=box_reg_iou_loss_type,
                        kfiou_fun=box_reg_kfiou_fun,
                        probiou_mode=box_reg_probiou_mode,
                    )
                    reg_loss = reg_loss + (box_reg_iou_weight * loss_iou)
                reg_loss_sums.append(reg_loss * box_reg_weight)
        
        # Accumulate losses for this level (classification only; reg summed globally below)
    # Aggregate losses across all levels.
    # Classification: total focal sum / num positive anchors (MMDet avg_factor, clamped to >= 1).
    if cls_loss_sums:
        loss_classification = torch.stack(cls_loss_sums).sum() / max(num_pos_total, 1)
    else:
        # Maintain gradient flow: compute zero loss from model outputs
        # Use first level's classification logits to maintain connection
        if len(classification_logits) > 0 and classification_logits[0].numel() > 0:
            loss_classification = (classification_logits[0] * 0.0).sum()
        else:
            # Fallback: create a small constant loss from device
            loss_classification = torch.tensor(0.0, device=device, requires_grad=True)
    
    if reg_loss_sums:
        loss_box_reg = torch.stack(reg_loss_sums).sum() / max(num_pos_total, 1)
    else:
        # Maintain gradient flow: compute zero loss from model outputs
        # Use first level's bbox regression to maintain connection
        if len(bbox_regression) > 0 and bbox_regression[0].numel() > 0:
            loss_box_reg = (bbox_regression[0] * 0.0).sum()
        else:
            # Fallback: create a small constant loss from device
            loss_box_reg = torch.tensor(0.0, device=device, requires_grad=True)
    
    return {
        "loss_classifier": loss_classification,
        "loss_box_reg": loss_box_reg,
    }


class RotatedRetinaNet(nn.Module):
    """Complete Rotated RetinaNet model for true oriented object detection.
    
    This model implements a full single-stage oriented detector that:
    - Uses MMRotate-style anchor priors: horizontal boxes as (cx, cy, w, h, theta) with fixed theta=0 at init
    - Predicts oriented bounding boxes with 5 parameters (cx, cy, w, h, angle)
    - Uses oriented IoU for matching and oriented NMS for post-processing
    - Preserves angle information throughout training and inference
    - Uses sigmoid focal loss for classification (MMRotate FocalLoss use_sigmoid=True)
    
    Args:
        num_classes: Number of object classes (excluding background)
        backbone: Optional backbone module (if None, creates ResNet+FPN)
        backbone_name: Name of backbone to create ("resnet18", "resnet50", etc.)
        pretrained_backbone: Whether to use pretrained backbone weights
        trainable_layers: Number of backbone layers to keep trainable
        anchor_scales: List of anchor scales for Rotated RetinaNet
        anchor_ratios: List of anchor aspect ratios for Rotated RetinaNet
        anchor_angles: Optional RPN anchor angles in radians (advanced, **not** in `ModelConfig` / JSON).
            ``None`` uses horizontal priors ``[0.0]`` (recommended). Non-default multi-angle banks can hurt accuracy.
        positive_iou_threshold: IoU threshold for positive anchors
        negative_iou_threshold: IoU threshold for negative anchors
        focal_alpha: Alpha parameter for focal loss (default: 0.25, Rotated RetinaNet standard)
        focal_gamma: Gamma parameter for focal loss (default: 2.0, Rotated RetinaNet standard)
        box_reg_weight: Weight for box regression loss (default: 1.0)
        score_threshold: Score threshold for inference (default: 0.05)
        final_nms_iou_threshold: IoU threshold for final oriented NMS (default: 0.5); not RPN NMS
        max_detections_per_image: Maximum number of detections per image (default: 100)
        final_nms_iou_schedule_epochs: Optional epoch boundaries for final NMS IoU schedule (e.g. [50, 150, 250])
        final_nms_iou_schedule_values: Final NMS IoU per segment (e.g. [0.6, 0.45, 0.35, 0.25]); lower = more suppression
        target_means: Optional means for target normalization (MMRotate compatibility)
        target_stds: Optional stds for target normalization (MMRotate compatibility)
        norm_factor: Optional angle scaling for encode/decode (MMRotate Rotated RetinaNet uses None)
        edge_swap: Whether to use edge_swap in bbox coder (MMRotate uses True)
        use_hbb_for_matching: If True, use HBB IoU for anchor-GT matching (optional; priors use a single reference angle).
        final_nms_use_cpu: If True, skip GPU sampling NMS and run final class-aware NMS with exact polygon IoU on CPU.
    
    Example:
        >>> model = RotatedRetinaNet(num_classes=15, backbone_name="resnet50")
        >>> model.train()
        >>> losses = model(images, targets)
        >>> model.eval()
        >>> predictions = model(images)
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
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_reg_weight: float = 1.0,
        score_threshold: float = 0.05,
        final_nms_iou_threshold: float = 0.5,
        max_detections_per_image: int = 100,
        final_nms_iou_schedule_epochs: Optional[List[int]] = None,
        final_nms_iou_schedule_values: Optional[List[float]] = None,
        roi_box_reg_iou_schedule_epochs: Optional[List[int]] = None,
        roi_box_reg_iou_schedule_values: Optional[List[float]] = None,
        target_means: Optional[Tuple[float, float, float, float, float]] = None,
        target_stds: Optional[Tuple[float, float, float, float, float]] = None,
        norm_factor: Optional[float] = None,
        edge_swap: bool = True,
        box_reg_iou_weight: float = 0.0,
        box_reg_iou_loss_type: str = "riou",
        box_reg_kfiou_fun: Optional[str] = None,
        box_reg_probiou_mode: Optional[str] = None,
        use_hbb_for_matching: bool = False,
        final_nms_use_cpu: bool = False,
        returned_layers: Optional[List[int]] = None,
        fpn_strides: Optional[List[int]] = None,
        fpn_extra_level: bool = False,
        octave_base_scale: Optional[float] = None,
        scales_per_octave: Optional[int] = None,
        stacked_convs: int = 1,
        box_reg_loss_type: str = "smooth_l1",
    ) -> None:
        from .backbones.utils import require_torch
        require_torch()
        super().__init__()
        
        self.num_classes = num_classes
        
        # Bbox coder options (MMRotate: norm_factor=None, edge_swap=True for Rotated RetinaNet)
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.box_reg_iou_loss_type = box_reg_iou_loss_type
        self.box_reg_kfiou_fun = box_reg_kfiou_fun
        self.box_reg_probiou_mode = box_reg_probiou_mode
        self.use_hbb_for_matching = use_hbb_for_matching
        self.final_nms_use_cpu = final_nms_use_cpu
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.box_reg_loss_type = box_reg_loss_type
        
        # Setup backbone (P6/P7 convs on C5 when fpn_extra_level, MMRotate on_input)
        self.backbone, backbone_channels = setup_backbone(
            backbone=backbone,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
            use_p6p7_extra_levels=fpn_extra_level,
        )
        
        # Default: horizontal priors (theta=0), MMRotate-style. Optional anchor_angles is Python-only (not in JSON).
        self.anchor_scales, self.anchor_ratios, self.anchor_angles, self.num_anchors = setup_anchors(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_angles=anchor_angles,
            default_angles=[0.0],
            octave_base_scale=octave_base_scale,
            scales_per_octave=scales_per_octave,
        )
        
        # FPN strides (nominal; forward uses grid-derived strides)
        if fpn_strides is not None:
            self.fpn_strides = list(fpn_strides)
        elif returned_layers == [2, 3, 4]:
            self.fpn_strides = [8, 16, 32, 64, 128] if fpn_extra_level else [8, 16, 32, 64]
        else:
            self.fpn_strides = [4, 8, 16, 32, 64]
        
        self.fpn_extra_level = fpn_extra_level
        
        # Create Rotated RetinaNet head
        self.head = OrientedRetinaNetHead(
            in_channels=backbone_channels,
            num_classes=num_classes,
            num_anchors=self.num_anchors,
            stacked_convs=stacked_convs,
        )
        
        # Loss and inference parameters
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_reg_weight = box_reg_weight
        self.score_threshold = score_threshold
        self.final_nms_iou_threshold = final_nms_iou_threshold
        self.max_detections_per_image = max_detections_per_image
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values
        self._roi_box_reg_iou_schedule_epochs = roi_box_reg_iou_schedule_epochs
        self._roi_box_reg_iou_schedule_values = roi_box_reg_iou_schedule_values
        self._box_reg_iou_weight_default = float(box_reg_iou_weight)
        self.box_reg_iou_weight = float(box_reg_iou_weight)
        self.set_box_reg_iou_weight_for_epoch(0)
        
        # Target normalization (MMRotate compatibility)
        self.target_means = target_means
        self.target_stds = target_stds
    
    def set_final_nms_iou_for_epoch(self, epoch: int) -> None:
        """Update final detection NMS IoU threshold from schedule for the given epoch.
        Lower threshold = more aggressive suppression of overlapping boxes.
        """
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

    def set_box_reg_iou_weight_for_epoch(self, epoch: int) -> None:
        """Update anchor auxiliary IoU loss weight from schedule (0-based epoch index)."""
        from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule

        self.box_reg_iou_weight = resolve_piecewise_schedule(
            epoch,
            self._roi_box_reg_iou_schedule_epochs,
            self._roi_box_reg_iou_schedule_values,
            self._box_reg_iou_weight_default,
        )

    def forward(
        self,
        images: Sequence[torch.Tensor],
        targets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Forward pass through Rotated RetinaNet.
        
        Args:
            images: List of image tensors (C, H, W) in [0, 1] range
            targets: Optional list of target dicts for training.
                    Each dict must contain:
                    - "rboxes" (List[RBox] or tensor [N, 5] with format [cx, cy, w, h, angle])
                    - "labels" (tensor [N], 1-indexed)
        
        Returns:
            - Training: Dict with loss keys:
                - "loss_classifier": Classification loss (focal loss)
                - "loss_box_reg": Box regression loss
            - Inference: List of dicts, one per image, containing:
                - "rboxes": List[RBox] with oriented boxes (with predicted angles)
                - "labels": Tensor [N] with class labels (1-indexed)
                - "scores": Tensor [N] with confidence scores
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        # Get image sizes
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # Extract features using shared utility
        feature_list = extract_backbone_features(
            self.backbone,
            images,
            use_checkpoint=False,  # Rotated RetinaNet doesn't use checkpointing currently
            training=self.training,
            include_pool_level=not self.fpn_extra_level,
        )
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
        fpn_strides_live = derive_fpn_strides_from_grid(image_sizes[0], feature_map_sizes)
        warn_if_fpn_strides_mismatch(self.fpn_strides, fpn_strides_live)
        
        # Get device from first feature map (needed for device reference)
        images_tensor = torch.stack(images, dim=0)
        
        # Forward through Rotated RetinaNet head
        classification_logits, bbox_regression = self.head(feature_list)
        
        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            
            # Prepare targets using shared utility
            gt_boxes_list, gt_labels_list, gt_boxes_ignore_list = prepare_targets(targets, device=images_tensor.device)
            
            # Generate anchors for all FPN levels
            img_h, img_w = image_sizes[0]
            anchors = generate_oriented_anchors(
                image_size=(img_h, img_w),
                feature_map_sizes=feature_map_sizes,
                anchor_scales=self.anchor_scales,
                anchor_ratios=self.anchor_ratios,
                anchor_angles=self.anchor_angles,
                stride_per_level=fpn_strides_live,
                octave_base_scale=self.octave_base_scale,
                scales_per_octave=self.scales_per_octave,
            )
            
            # Compute losses
            losses = compute_oriented_retinanet_loss(
                classification_logits=classification_logits,
                bbox_regression=bbox_regression,
                anchors=anchors,
                gt_boxes=gt_boxes_list,
                gt_labels=gt_labels_list,
                image_sizes=image_sizes,
                gt_boxes_ignore=gt_boxes_ignore_list,
                num_classes=self.num_classes,
                positive_iou_threshold=self.positive_iou_threshold,
                negative_iou_threshold=self.negative_iou_threshold,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                box_reg_weight=self.box_reg_weight,
                target_means=self.target_means,
                target_stds=self.target_stds,
                target_norm_factor=self.norm_factor,
                norm_factor=self.norm_factor,
                edge_swap=self.edge_swap,
                box_reg_iou_weight=self.box_reg_iou_weight,
                box_reg_iou_loss_type=self.box_reg_iou_loss_type,
                box_reg_kfiou_fun=self.box_reg_kfiou_fun,
                box_reg_probiou_mode=self.box_reg_probiou_mode,
                use_hbb_for_matching=self.use_hbb_for_matching,
                box_reg_loss_type=self.box_reg_loss_type,
            )
            
            return losses
        
        else:
            # Inference
            # Generate anchors
            img_h, img_w = image_sizes[0]
            anchors = generate_oriented_anchors(
                image_size=(img_h, img_w),
                feature_map_sizes=feature_map_sizes,
                anchor_scales=self.anchor_scales,
                anchor_ratios=self.anchor_ratios,
                anchor_angles=self.anchor_angles,
                stride_per_level=fpn_strides_live,
                octave_base_scale=self.octave_base_scale,
                scales_per_octave=self.scales_per_octave,
            )
            # Optional debug: expose anchors and decoded boxes (pre-NMS) for TensorBoard
            return_debug = getattr(self, '_return_anchors_proposals', False)
            if return_debug:
                all_anchors_cat = torch.cat(anchors, dim=0).detach().cpu()
            # Process each image
            outputs = []
            for img_idx in range(len(images)):
                # Collect predictions from all levels for this image
                all_boxes = []
                all_scores = []
                all_labels = []
                
                num_classes = self.num_classes  # K sigmoid outputs per anchor (no background channel)
                for level_idx in range(len(feature_list)):
                    B, C_cls, H, W = classification_logits[level_idx].shape
                    num_anchors = self.num_anchors
                    
                    # Get predictions for this image at this level
                    cls_logits = classification_logits[level_idx][img_idx]  # [num_anchors*K, H, W]
                    bbox_pred = bbox_regression[level_idx][img_idx]  # [num_anchors*5, H, W]
                    
                    # Reshape: [num_anchors*K, H, W] -> [H*W*num_anchors, num_classes]
                    cls_logits_flat = cls_logits.view(num_anchors, num_classes, H, W)
                    cls_logits_flat = cls_logits_flat.permute(2, 3, 0, 1).contiguous().view(-1, num_classes)
                    
                    # Reshape: [num_anchors*5, H, W] -> [H*W*num_anchors, 5]
                    bbox_pred_flat = bbox_pred.view(num_anchors, 5, H, W)
                    bbox_pred_flat = bbox_pred_flat.permute(2, 3, 0, 1).contiguous().view(-1, 5)
                    
                    # Get anchors for this level
                    level_anchors = anchors[level_idx]
                    if isinstance(level_anchors, torch.Tensor):
                        level_anchors = level_anchors.to(cls_logits.device)
                    else:
                        level_anchors = torch.tensor(
                            level_anchors, dtype=torch.float32, device=cls_logits.device
                        )
                    
                    # Decode boxes
                    decoded_boxes = decode_oriented_boxes(
                        level_anchors,
                        bbox_pred_flat,
                        target_means=self.target_means,
                        target_stds=self.target_stds,
                        normalize_le90=True,
                        norm_factor=self.norm_factor,
                        edge_swap=self.edge_swap,
                    )
                    
                    # Sigmoid per class (MMRotate use_sigmoid=True); take best foreground class per anchor
                    class_scores = torch.sigmoid(cls_logits_flat)  # [N, num_classes]
                    max_scores, fg_indices = class_scores.max(dim=1)  # [N], [N] in 0..K-1
                    # 1-indexed class label = 1 + fg_indices (downstream pipeline uses 1-indexed labels)
                    class_indices_1idx = fg_indices + 1  # [N], values 1..num_classes
                    
                    # Pre-NMS score filter (MMRotate nms_pre). Keeps ~10³ candidates instead of
                    # ~2×10⁵ decoded anchors per image; eval still applies evaluation.score_threshold.
                    if self.training:
                        score_thresh = 0.0
                    else:
                        score_thresh = self.score_threshold
                    
                    # Filter by score threshold
                    valid_mask = max_scores >= score_thresh
                    if valid_mask.any():
                        all_boxes.append(decoded_boxes[valid_mask])
                        all_scores.append(max_scores[valid_mask])
                        all_labels.append(class_indices_1idx[valid_mask])
                
                if not all_boxes:
                    out = {
                        "rboxes": [],
                        "labels": torch.zeros((0,), dtype=torch.int64, device=images_tensor.device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=images_tensor.device),
                    }
                    if return_debug:
                        out["anchors"] = all_anchors_cat
                        out["proposals"] = torch.zeros((0, 5), dtype=torch.float32)
                    outputs.append(out)
                    continue
                
                # Concatenate predictions from all levels
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                # Filter out degenerate boxes (zero or near-zero width/height) before RBox conversion
                min_size = 1.0
                valid = (all_boxes[:, 2] >= min_size) & (all_boxes[:, 3] >= min_size) & torch.isfinite(all_boxes).all(dim=1)
                all_boxes = all_boxes[valid]
                all_scores = all_scores[valid]
                all_labels = all_labels[valid]
                
                # Save decoded boxes before NMS for debug logging (RetinaNet "proposals" analogue)
                proposals_for_debug = all_boxes.detach().cpu() if return_debug else None
                
                # Apply oriented NMS (class-aware)
                # Use GPU-accelerated NMS when available (much faster)
                # Use GPU NMS on CUDA or MPS (Apple Silicon)
                device_type = images_tensor.device.type
                use_gpu_nms = (
                    not self.final_nms_use_cpu
                    and torch is not None
                    and len(all_boxes) > 0
                    and (
                        (device_type == "cuda" and torch.cuda.is_available())
                        or (
                            device_type == "mps"
                            and getattr(torch.backends, "mps", None) is not None
                            and torch.backends.mps.is_available()
                        )
                    )
                )
                
                keep_indices_tensor = None
                if use_gpu_nms:
                    # GPU-accelerated NMS: handle class-aware NMS by grouping by class
                    # Group boxes by class and run NMS per class
                    unique_labels = torch.unique(all_labels)
                    all_keep_indices = []
                    
                    for label in unique_labels:
                        class_mask = all_labels == label
                        class_boxes = all_boxes[class_mask]
                        class_scores = all_scores[class_mask]
                        class_indices = torch.where(class_mask)[0]
                        
                        if len(class_boxes) > 0:
                            # Pre-filter: limit to top boxes per class to avoid memory issues
                            # Use a reasonable limit (e.g., 2000 boxes) before NMS to prevent OOM
                            # This is much larger than the final global limit but prevents huge IoU matrices
                            max_boxes_per_class = 2000
                            if len(class_boxes) > max_boxes_per_class:
                                # Sort by score and take top N
                                _, top_indices = class_scores.sort(descending=True)
                                top_indices = top_indices[:max_boxes_per_class]
                                class_boxes = class_boxes[top_indices]
                                class_scores = class_scores[top_indices]
                                class_indices = class_indices[top_indices]
                            
                            # Run GPU NMS for this class
                            # Pass None to let NMS keep all results, we'll apply global limit later
                            class_keep = rotated_nms(
                                class_boxes,
                                class_scores,
                                iou_threshold=self.final_nms_iou_threshold,
                                max_detections=None,  # No per-class limit - apply globally later
                            )
                            # Map back to original indices
                            all_keep_indices.append(class_indices[class_keep])
                    
                    if all_keep_indices:
                        # Filter out empty tensors before concatenating
                        non_empty_indices = [idx for idx in all_keep_indices if len(idx) > 0]
                        if non_empty_indices:
                            keep_indices_tensor = torch.cat(non_empty_indices)
                            # Sort by score to maintain order
                            keep_scores = all_scores[keep_indices_tensor]
                            _, sort_order = keep_scores.sort(descending=True)
                            keep_indices_tensor = keep_indices_tensor[sort_order]
                            
                            # Apply max_detections limit across all classes
                            if self.max_detections_per_image is not None:
                                keep_indices_tensor = keep_indices_tensor[:self.max_detections_per_image]
                        else:
                            # All classes had empty results after NMS
                            keep_indices_tensor = None
                else:
                    # Fall back to CPU NMS (class-aware). Pre-filter per class like the GPU path.
                    keep_indices: list[int] = []
                    unique_labels = torch.unique(all_labels)
                    max_boxes_per_class = 2000
                    for label in unique_labels:
                        class_mask = all_labels == label
                        class_boxes = all_boxes[class_mask]
                        class_scores = all_scores[class_mask]
                        class_indices = torch.where(class_mask)[0]
                        if len(class_boxes) == 0:
                            continue
                        if len(class_boxes) > max_boxes_per_class:
                            _, top_indices = class_scores.sort(descending=True)
                            top_indices = top_indices[:max_boxes_per_class]
                            class_boxes = class_boxes[top_indices]
                            class_scores = class_scores[top_indices]
                            class_indices = class_indices[top_indices]
                        rboxes = tensor_to_rboxes(class_boxes)
                        class_keep = nms.oriented_nms(
                            boxes=rboxes,
                            scores=class_scores.cpu().tolist(),
                            labels=[int(label.item())] * len(rboxes),
                            iou_threshold=self.final_nms_iou_threshold,
                            max_detections=None,
                        )
                        keep_indices.extend(class_indices[class_keep].tolist())
                    if keep_indices:
                        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=all_boxes.device)
                        keep_scores = all_scores[keep_tensor]
                        _, sort_order = keep_scores.sort(descending=True)
                        keep_indices = keep_tensor[sort_order].tolist()
                        if self.max_detections_per_image is not None:
                            keep_indices = keep_indices[: self.max_detections_per_image]
                    else:
                        keep_indices = []
                
                # Filter results
                if use_gpu_nms and keep_indices_tensor is not None and len(keep_indices_tensor) > 0:
                    # GPU path: use tensor indices directly
                    output_rboxes = tensor_to_rboxes(all_boxes[keep_indices_tensor])
                    output_labels = all_labels[keep_indices_tensor]
                    output_scores = all_scores[keep_indices_tensor]
                elif not use_gpu_nms and keep_indices:
                    # CPU path: convert from list indices
                    rboxes = tensor_to_rboxes(all_boxes)
                    output_rboxes = [rboxes[i] for i in keep_indices]
                    output_labels = all_labels[keep_indices]
                    output_scores = all_scores[keep_indices]
                else:
                    output_rboxes = []
                    output_labels = torch.zeros((0,), dtype=torch.int64, device=images_tensor.device)
                    output_scores = torch.zeros((0,), dtype=torch.float32, device=images_tensor.device)
                
                out = {
                    "rboxes": output_rboxes,
                    "labels": output_labels,
                    "scores": output_scores,
                }
                if return_debug:
                    out["anchors"] = all_anchors_cat
                    out["proposals"] = proposals_for_debug if proposals_for_debug is not None else torch.zeros((0, 5), dtype=torch.float32)
                outputs.append(out)
            
            return outputs


__all__ = ["RotatedRetinaNet", "OrientedRetinaNetHead"]
