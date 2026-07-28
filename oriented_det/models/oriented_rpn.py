"""Oriented Region Proposal Network (RPN) for oriented object detection.

This module implements an RPN that predicts oriented bounding boxes with 5 parameters:
(cx, cy, width, height, angle) instead of axis-aligned boxes with 4 parameters.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


from ..geometry import RBox
from ..ops import iou, nms
from ..ops.gpu_ops import (
    generate_oriented_anchors_gpu,
    hbb_nms_for_oriented_boxes_gpu,
    match_anchors_to_gt_gpu,
    obb_to_xyxy_gpu,
)
from ..ops.rotated_ops import pairwise_rotated_iou
from ..utils.logging import logger

# Global flag to use GPU-accelerated operations
# Uses sampling-based IoU approximation that is fully vectorized on GPU
USE_GPU_OPS = True

# Internal debug switch for one-shot RPN loss timing prints. Keep disabled for
# normal training; set to True locally when investigating first-batch stalls.
TRACE_RPN_LOSS_TIMING = False


class OrientedRPNHead(nn.Module if nn is not None else object):  # type: ignore
    """RPN head that predicts oriented proposals (MMRotate format).
    
    Args:
        in_channels: Number of input channels from backbone/FPN
        num_anchors: Number of anchors per spatial location
        cls_out_channels: Number of classification output channels per anchor (1 for objectness score).
        reg_out_channels: Number of regression output channels per anchor (6 parameters).
    """
    
    def __init__(
        self, 
        in_channels: int, 
        num_anchors: int = 3,
        cls_out_channels: int = 2,
        reg_out_channels: int = 5,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for OrientedRPNHead.")
        super().__init__()
        self.num_anchors = num_anchors
        self.cls_out_channels = cls_out_channels
        self.reg_out_channels = reg_out_channels
        
        # Shared 3x3 conv layer before cls and reg heads (matches MMRotate architecture)
        # This is critical for MMRotate weight compatibility
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # Objectness classification head: num_anchors * cls_out_channels (typically 1 for objectness score)
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * cls_out_channels, kernel_size=1, stride=1)
        
        # Box regression head: num_anchors * reg_out_channels (4 params: dx, dy, dw, dh - angle from anchor)
        self.conv_bbox = nn.Conv2d(in_channels, num_anchors * reg_out_channels, kernel_size=1, stride=1)
        
        # Initialize weights
        for layer in [self.rpn_conv, self.conv_cls, self.conv_bbox]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through RPN head.
        
        Args:
            features: List of feature maps from FPN, each of shape [B, C, H, W]
        
        Returns:
            Tuple of (objectness_logits, bbox_regression):
            - objectness_logits: List of tensors, each [B, num_anchors*cls_out_channels, H, W]
            - bbox_regression: List of tensors, each [B, num_anchors*reg_out_channels, H, W]
        """
        objectness_logits = []
        bbox_regression = []
        
        for feat in features:
            # Apply shared 3x3 conv with ReLU (matches MMRotate architecture)
            x = F.relu(self.rpn_conv(feat))
            
            # Classification: [B, C, H, W] -> [B, num_anchors*cls_out_channels, H, W]
            cls_logits = self.conv_cls(x)
            objectness_logits.append(cls_logits)
            
            # Regression: [B, C, H, W] -> [B, num_anchors*reg_out_channels, H, W]
            bbox_pred = self.conv_bbox(x)
            bbox_regression.append(bbox_pred)
        
        return objectness_logits, bbox_regression


def generate_oriented_anchors(
    image_size: Tuple[int, int],
    feature_map_sizes: List[Tuple[int, int]],
    anchor_scales: List[float],
    anchor_ratios: List[float],
    anchor_angles: List[float],
    stride_per_level: List[int],
    device: Optional[torch.device] = None,
    octave_base_scale: Optional[float] = None,
    scales_per_octave: Optional[int] = None,
) -> List[torch.Tensor]:
    """Generate oriented anchors for all feature map levels.
    
    Uses GPU-accelerated vectorized generation when USE_GPU_OPS is True (default).
    This is 10-100x faster than the Python loop version for large feature maps.
    
    Args:
        image_size: (height, width) of input image
        feature_map_sizes: List of (height, width) for each FPN level
        anchor_scales: List of anchor scales (e.g., [8, 16, 32])
        anchor_ratios: List of aspect ratios (e.g., [0.5, 1.0, 2.0])
        anchor_angles: List of anchor angles in radians (e.g., [-π/6, 0, π/6])
        stride_per_level: List of strides for each FPN level
        device: Device to create tensors on (default: CPU, will be moved to GPU later)
    
    Returns:
        List of anchor tensors, each of shape [N, 5] where N = H * W * num_anchors
        Format: [cx, cy, width, height, angle]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for anchor generation.")
    
    if device is None:
        device = torch.device('cpu')
    
    # Use GPU-accelerated vectorized anchor generation (10-100x faster)
    if USE_GPU_OPS:
        return generate_oriented_anchors_gpu(
            image_size=image_size,
            feature_map_sizes=feature_map_sizes,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_angles=anchor_angles,
            stride_per_level=stride_per_level,
            device=device,
            octave_base_scale=octave_base_scale,
            scales_per_octave=scales_per_octave,
        )
    
    # Fallback to Python loop version (slow, for debugging only)
    with logger.trace_block(
        "generate_oriented_anchors(image_size={}, levels={}, scales={}, ratios={}, angles={})",
        image_size, len(feature_map_sizes), len(anchor_scales), len(anchor_ratios), len(anchor_angles)
    ):
        logger.trace_value("Image size", image_size)
        logger.trace_value("Feature map sizes", feature_map_sizes)
        logger.trace_value("Anchor scales", anchor_scales)
        logger.trace_value("Anchor ratios", anchor_ratios)
        logger.trace_value("Anchor angles", [f"{a*180/math.pi:.1f}°" for a in anchor_angles])
        
        anchors_per_level = []
        total_anchors = 0
        
        for level_idx, (feat_h, feat_w) in enumerate(feature_map_sizes):
            stride = stride_per_level[level_idx]
            scale = anchor_scales[level_idx] if level_idx < len(anchor_scales) else anchor_scales[-1]
            if octave_base_scale is not None and scales_per_octave is not None:
                scale_factors = [
                    2.0 ** (i / float(scales_per_octave)) for i in range(int(scales_per_octave))
                ]
                effective_scales = [stride * float(octave_base_scale) * f for f in scale_factors]
            else:
                effective_scales = [scale * stride]
            
            with logger.trace_block(
                "Level {}: feature_map={}x{}, stride={}, scale={}",
                level_idx, feat_h, feat_w, stride, scale
            ):
                # Generate anchors for this level
                level_anchors = []
                num_locations = feat_h * feat_w
                num_anchors_per_loc = len(effective_scales) * len(anchor_ratios) * len(anchor_angles)
                expected_anchors = num_locations * num_anchors_per_loc
                
                logger.trace_value("Locations", num_locations)
                logger.trace_value("Anchors per location", num_anchors_per_loc)
                logger.trace_value("Expected total anchors", expected_anchors)
                
                # Show progress for large feature maps
                if num_locations > 10000 and logger.is_enabled():
                    logger.trace("Generating anchors (this may take a moment for large feature maps)...")
                
                for y in range(feat_h):
                    for x in range(feat_w):
                        # Center of anchor in image coordinates
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        
                        # Generate anchors with different scales, ratios, and angles
                        for effective_scale in effective_scales:
                            for ratio in anchor_ratios:
                                for angle in anchor_angles:
                                    base_w = effective_scale * math.sqrt(ratio)
                                    base_h = effective_scale / math.sqrt(ratio)
                                    
                                    # Create anchor as [cx, cy, w, h, angle]
                                    # CRITICAL: Anchors don't need gradients - they're just coordinates
                                    anchor = torch.tensor([cx, cy, base_w, base_h, angle], dtype=torch.float32, requires_grad=False)
                                    level_anchors.append(anchor)
                
                if level_anchors:
                    anchors_tensor = torch.stack(level_anchors)  # [N, 5]
                    # CRITICAL: Ensure anchors are explicitly non-differentiable
                    # Anchors are just coordinates and don't need gradients
                    anchors_tensor = anchors_tensor.detach()
                    anchors_per_level.append(anchors_tensor)
                    level_count = len(anchors_tensor)
                    total_anchors += level_count
                    logger.trace_value("Generated anchors", level_count)
                else:
                    # Empty anchors for this level
                    anchors_per_level.append(torch.zeros((0, 5), dtype=torch.float32, requires_grad=False))
        
        logger.trace_value("Total anchors across all levels", total_anchors)
        return anchors_per_level


def normalize_angle_delta(dangle: torch.Tensor) -> torch.Tensor:
    """Normalize angle deltas to [-π, π] range.
    
    This ensures angle differences are properly handled for periodic angles.
    Used when computing loss between predicted and target angle deltas.
    
    Args:
        dangle: Angle delta tensor (can be any shape)
    
    Returns:
        Normalized angle delta in [-π, π] range
    """
    return torch.atan2(torch.sin(dangle), torch.cos(dangle))


def norm_angle_le90(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-π/2, π/2) for le90 convention. Used in edge_swap."""
    return torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2


def encode_rpn_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
) -> torch.Tensor:
    """Encode ground truth boxes relative to anchors for RPN (4 params, no angle).
    
    Alias: DeltaXYWHBBoxCoder (MMRotate Rotated Faster R-CNN RPN).
    RPN predicts only dx, dy, dw, dh; angle stays from anchor.
    
    Args:
        anchors: Tensor [N, 5] with format [cx, cy, w, h, angle]
        gt_boxes: Tensor [N, 5] with format [cx, cy, w, h, angle]
        target_means: Optional means [dx, dy, dw, dh]. Default: (0, 0, 0, 0)
        target_stds: Optional stds [dx, dy, dw, dh]. Default: (1, 1, 1, 1)
    
    Returns:
        Encoded targets [N, 4] with format [dx, dy, dw, dh]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box encoding.")
    
    if target_means is None:
        target_means = (0.0, 0.0, 0.0, 0.0)
    if target_stds is None:
        target_stds = (1.0, 1.0, 1.0, 1.0)
    
    means = torch.tensor(target_means, dtype=torch.float32, device=anchors.device)
    stds = torch.tensor(target_stds, dtype=torch.float32, device=anchors.device)
    
    with torch.no_grad():
        if isinstance(anchors, torch.Tensor):
            anchors = anchors.detach()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.detach()
        
        anchor_cx, anchor_cy, anchor_w, anchor_h, _ = anchors.unbind(dim=1)
        gt_cx, gt_cy, gt_w, gt_h, _ = gt_boxes.unbind(dim=1)

        # Robustness: clamp widths/heights to avoid zero / negative sizes that
        # would produce -inf or NaN regression targets and cause training to explode.
        eps = 1e-6
        anchor_w = torch.clamp(anchor_w, min=eps)
        anchor_h = torch.clamp(anchor_h, min=eps)
        gt_w = torch.clamp(gt_w, min=eps)
        gt_h = torch.clamp(gt_h, min=eps)
        
        dx_raw = (gt_cx - anchor_cx) / anchor_w
        dy_raw = (gt_cy - anchor_cy) / anchor_h
        dw_raw = torch.log(gt_w / anchor_w)
        dh_raw = torch.log(gt_h / anchor_h)
        
        targets_raw = torch.stack([dx_raw, dy_raw, dw_raw, dh_raw], dim=1)
        targets = (targets_raw - means) / (stds + 1e-8)
    
    return targets


def decode_rpn_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
    wh_ratio_clip: Optional[float] = 16.0,
) -> torch.Tensor:
    """Decode RPN predicted deltas (4 params) back to oriented boxes.
    
    Angle is copied from anchor. Supports target_means/target_stds like MMRotate.
    Clamps dw/dh so exp(dw), exp(dh) stay in [1/wh_ratio_clip, wh_ratio_clip] to avoid
    degenerate (sliver or huge) proposals that prevent ROI head convergence.
    
    Args:
        anchors: Tensor [N, 5] with format [cx, cy, w, h, angle]
        deltas: Tensor [N, 4] with normalized [dx, dy, dw, dh]
        target_means: Optional means [dx, dy, dw, dh]. Default: (0, 0, 0, 0)
        target_stds: Optional stds [dx, dy, dw, dh]. Default: (1, 1, 1, 1)
        wh_ratio_clip: If set, clamp dw/dh so w,h stay within anchor * [1/wh_ratio_clip, wh_ratio_clip].
                      Default 16.0 prevents explosion/sliver proposals. None = no clamp.
    
    Returns:
        Decoded boxes [N, 5] with format [cx, cy, w, h, angle]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box decoding.")
    
    if target_means is None:
        target_means = (0.0, 0.0, 0.0, 0.0)
    if target_stds is None:
        target_stds = (1.0, 1.0, 1.0, 1.0)
    
    means = torch.tensor(target_means, dtype=torch.float32, device=anchors.device)
    stds = torch.tensor(target_stds, dtype=torch.float32, device=anchors.device)
    deltas_raw = deltas * stds + means
    
    anchor_cx, anchor_cy, anchor_w, anchor_h, anchor_angle = anchors.unbind(dim=1)
    dx, dy, dw, dh = deltas_raw[:, 0], deltas_raw[:, 1], deltas_raw[:, 2], deltas_raw[:, 3]
    
    if wh_ratio_clip is not None:
        max_log_ratio = math.log(wh_ratio_clip)
        dw = torch.clamp(dw, -max_log_ratio, max_log_ratio)
        dh = torch.clamp(dh, -max_log_ratio, max_log_ratio)
    
    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)
    
    boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h, anchor_angle], dim=1)
    
    # Check for invalid boxes (single-line warning for debugging)
    invalid_before_norm = (boxes[:, 2] <= 0) | (boxes[:, 3] <= 0) | ~torch.isfinite(boxes).all(dim=1)
    if invalid_before_norm.any():
        num_invalid = invalid_before_norm.sum().item()
        _log.warning("decode_rpn_boxes: %s/%s boxes invalid before le90 normalization", num_invalid, len(boxes))
    
    normalized = normalize_boxes_to_le90(boxes)
    
    invalid_after_norm = (normalized[:, 2] <= 0) | (normalized[:, 3] <= 0) | ~torch.isfinite(normalized).all(dim=1)
    if invalid_after_norm.any():
        num_invalid = invalid_after_norm.sum().item()
        _log.warning("decode_rpn_boxes: %s/%s boxes invalid after le90 normalization", num_invalid, len(normalized))
    
    return normalized


def decode_rpn_boxes_xyxy(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
    wh_ratio_clip: Optional[float] = 16.0,
) -> torch.Tensor:
    """Decode 4D RPN deltas to horizontal xyxy boxes without le90 normalization.

    This is the MMRotate Rotated Faster R-CNN RPN/proposal path: anchors are
    horizontal and proposals must stay horizontal. Do not call
    :func:`normalize_boxes_to_le90` here because it can swap ``w/h`` and distort
    tall horizontal boxes.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box decoding.")

    if target_means is None:
        target_means = (0.0, 0.0, 0.0, 0.0)
    if target_stds is None:
        target_stds = (1.0, 1.0, 1.0, 1.0)

    means = torch.tensor(target_means, dtype=torch.float32, device=anchors.device)
    stds = torch.tensor(target_stds, dtype=torch.float32, device=anchors.device)
    deltas_raw = deltas * stds + means

    anchor_cx, anchor_cy, anchor_w, anchor_h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    dx, dy, dw, dh = deltas_raw[:, 0], deltas_raw[:, 1], deltas_raw[:, 2], deltas_raw[:, 3]
    if wh_ratio_clip is not None:
        max_log_ratio = math.log(wh_ratio_clip)
        dw = torch.clamp(dw, -max_log_ratio, max_log_ratio)
        dh = torch.clamp(dh, -max_log_ratio, max_log_ratio)

    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)
    x1 = pred_cx - pred_w * 0.5
    y1 = pred_cy - pred_h * 0.5
    x2 = pred_cx + pred_w * 0.5
    y2 = pred_cy + pred_h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def encode_oriented_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    target_means: Optional[Tuple[float, ...]] = None,
    target_stds: Optional[Tuple[float, ...]] = None,
    norm_factor: Optional[float] = None,
    edge_swap: bool = False,
    proj_xy: bool = False,
) -> torch.Tensor:
    """Encode ground truth boxes relative to anchors (regression targets).
    
    Alias: DeltaXYWHAHBBoxCoder (MMRotate ROI head).
    Uses standard Faster R-CNN parameterization adapted for oriented boxes.
    
    Args:
        anchors: Tensor [N, 5] with format [cx, cy, w, h, angle]
        gt_boxes: Tensor [N, 5] with format [cx, cy, w, h, angle]
        target_means: Optional means for normalization [dx, dy, dw, dh, dangle].
                     If None, uses (0.0, 0.0, 0.0, 0.0, 0.0) - no centering.
                     Default: None
        target_stds: Optional stds for normalization [dx, dy, dw, dh, dangle].
                    If None, uses (1.0, 1.0, 1.0, 1.0, 1.0) - no scaling.
                    Default: None
        norm_factor: If set, da = dangle_raw / (norm_factor * π) to scale angle into [-0.5, 0.5].
                     MMRotate uses norm_factor=2 for ROI.
        edge_swap: If True, pick (w,h) vs (h,w) representation to minimize |dtheta| (MMRotate style).
        proj_xy: If True, express dx/dy in the anchor local frame (MMRotate ``proj_xy`` for
            ``DeltaXYWHTRBBoxCoder``). Default False (legacy global dx/dy).
    
    Returns:
        Encoded targets [N, 5] with format [dx, dy, dw, dh, dangle] where:
        - dx = ((gt_cx - anchor_cx) / anchor_w - mean_dx) / std_dx
        - dy = ((gt_cy - anchor_cy) / anchor_h - mean_dy) / std_dy
        - dw = (log(gt_w / anchor_w) - mean_dw) / std_dw
        - dh = (log(gt_h / anchor_h) - mean_dh) / std_dh
        - dangle = (angle_delta - mean_dangle) / std_dangle (with norm_factor and edge_swap if set)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box encoding.")
    
    # Default means/stds (no normalization by default, like MMRotate default)
    if target_means is None:
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
    if target_stds is None:
        target_stds = (1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Convert to tensors for broadcasting
    means = torch.tensor(target_means, dtype=torch.float32, device=anchors.device)
    stds = torch.tensor(target_stds, dtype=torch.float32, device=anchors.device)
    
    # These are ground truth targets - no gradients needed
    with torch.no_grad():
        if isinstance(anchors, torch.Tensor):
            anchors = anchors.detach()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.detach()
        
        anchor_cx, anchor_cy, anchor_w, anchor_h, anchor_angle = anchors.unbind(dim=1)
        gt_cx, gt_cy, gt_w, gt_h, gt_angle = gt_boxes.unbind(dim=1)

        # Clamp w/h to avoid zero or negative (would produce -inf or inf in log/div)
        eps = 1e-6
        anchor_w = torch.clamp(anchor_w, min=eps)
        anchor_h = torch.clamp(anchor_h, min=eps)
        gt_w = torch.clamp(gt_w, min=eps)
        gt_h = torch.clamp(gt_h, min=eps)

        # Compute raw targets for center and size
        if proj_xy:
            cos_t = torch.cos(anchor_angle)
            sin_t = torch.sin(anchor_angle)
            dcx = gt_cx - anchor_cx
            dcy = gt_cy - anchor_cy
            dx_raw = (cos_t * dcx + sin_t * dcy) / anchor_w
            dy_raw = (-sin_t * dcx + cos_t * dcy) / anchor_h
        else:
            dx_raw = (gt_cx - anchor_cx) / anchor_w
            dy_raw = (gt_cy - anchor_cy) / anchor_h

        # Angle and size with optional edge_swap (pick representation that minimizes |dtheta|)
        # MMRotate bbox2delta wraps both candidates with the π-periodic le90 norm
        # ([-π/2, π/2)), so the chosen dtheta is always in [-π/4, π/4]. Decoding stays
        # consistent: decode's edge_swap inverse (w<h comparison + π/2 correction +
        # le90 re-norm) recovers the same box for π-shifted angles.
        if edge_swap:
            dtheta1 = norm_angle_le90(gt_angle - anchor_angle)
            dtheta2 = norm_angle_le90(gt_angle + math.pi / 2 - anchor_angle)
            abs1 = torch.abs(dtheta1)
            abs2 = torch.abs(dtheta2)
            use_swap = abs2 < abs1
            gw_regular = torch.where(use_swap, gt_h, gt_w)
            gh_regular = torch.where(use_swap, gt_w, gt_h)
            dangle_raw = torch.where(use_swap, dtheta2, dtheta1)
        else:
            dangle_raw = gt_angle - anchor_angle
            dangle_raw = normalize_angle_delta(dangle_raw)
            gw_regular = gt_w
            gh_regular = gt_h

        # Clamp ratios for log to avoid -inf/inf (e.g. degenerate boxes)
        max_log_ratio = 12.0  # exp(12) ~ 1.6e5
        ratio_w = torch.clamp(gw_regular / anchor_w, min=1e-6, max=1e6)
        ratio_h = torch.clamp(gh_regular / anchor_h, min=1e-6, max=1e6)
        dw_raw = torch.clamp(torch.log(ratio_w), -max_log_ratio, max_log_ratio)
        dh_raw = torch.clamp(torch.log(ratio_h), -max_log_ratio, max_log_ratio)
        
        # Apply norm_factor to angle (MMRotate: da = ga / (norm_factor * π))
        if norm_factor is not None:
            dangle_raw = dangle_raw / (norm_factor * math.pi)
        
        # Stack raw targets
        targets_raw = torch.stack([dx_raw, dy_raw, dw_raw, dh_raw, dangle_raw], dim=1)
        
        # Apply mean/std normalization (like MMRotate)
        # normalized = (target - mean) / std
        targets = (targets_raw - means) / (stds + 1e-8)
    
    return targets


def normalize_boxes_to_le90(boxes: torch.Tensor) -> torch.Tensor:
    """Normalize boxes to le90 convention (width >= height, angle in [-π/2, π/2)).
    
    This ensures all boxes follow the le90 convention used in training.
    Fully tensorized implementation for GPU efficiency - no CPU sync or gradient breaks.
    
    Args:
        boxes: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
    
    Returns:
        Normalized boxes of shape [N, 5] with width >= height and angle in [-π/2, π/2)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box normalization.")
    
    # Extract components
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    angle = boxes[:, 4]
    
    # Step 1: Normalize angle to [-π/2, π/2) by adding/subtracting integer multiples of π.
    # Important: a π rotation does NOT swap width/height for rectangles.
    pi = math.pi
    pi_half = pi / 2
    
    # Normalize angle to [-π/2, π/2) using floor-based wrap.
    angle_offset = angle + pi_half
    k = torch.floor(angle_offset / pi)
    angle_normalized = angle - k * pi

    # Step 2: Enforce width >= height by optionally swapping once and adding π/2.
    needs_swap3 = w < h
    w_final = torch.where(needs_swap3, h, w)
    h_final = torch.where(needs_swap3, w, h)

    # Adjust angle: if we swapped, add π/2 and normalize again to [-π/2, π/2)
    angle_adjusted = torch.where(needs_swap3, angle_normalized + pi_half, angle_normalized)
    angle_final = torch.remainder(angle_adjusted + pi_half, pi) - pi_half
    
    # Stack back into [N, 5] format
    normalized_boxes = torch.stack([cx, cy, w_final, h_final, angle_final], dim=1)
    
    return normalized_boxes


def decode_oriented_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    target_means: Optional[Tuple[float, ...]] = None,
    target_stds: Optional[Tuple[float, ...]] = None,
    normalize_le90: bool = True,
    norm_factor: Optional[float] = None,
    edge_swap: bool = False,
    proj_xy: bool = False,
) -> torch.Tensor:
    """Decode predicted deltas back to oriented boxes.
    
    This is the inverse of encode_oriented_boxes.
    
    Args:
        anchors: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
        deltas: Tensor of shape [N, 5] with normalized [dx, dy, dw, dh, dangle]
        target_means: Optional means used during encoding [dx, dy, dw, dh, dangle].
                     Must match those used in encode_oriented_boxes.
                     Default: None (assumes (0.0, 0.0, 0.0, 0.0, 0.0))
        target_stds: Optional stds used during encoding [dx, dy, dw, dh, dangle].
                    Must match those used in encode_oriented_boxes.
                    Default: None (assumes (1.0, 1.0, 1.0, 1.0, 1.0))
        normalize_le90: If True, normalize decoded boxes to le90 convention.
                       Default: True (matches training convention)
        norm_factor: If set, dangle was encoded as da/(norm_factor*π); reverse that here.
        edge_swap: If True, apply edge_swap inverse (w_regular, h_regular, theta_regular).
        proj_xy: If True, invert anchor-local dx/dy (must match ``encode_oriented_boxes``).
    
    Returns:
        Decoded boxes of shape [N, 5] with format [cx, cy, w, h, angle]
        (normalized to le90 if normalize_le90=True)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for box decoding.")
    
    # Default means/stds (no normalization by default, like MMRotate default)
    if target_means is None:
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
    if target_stds is None:
        target_stds = (1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Convert to tensors for broadcasting
    means = torch.tensor(target_means, dtype=torch.float32, device=anchors.device)
    stds = torch.tensor(target_stds, dtype=torch.float32, device=anchors.device)
    
    # Denormalize: raw = normalized * std + mean
    deltas_raw = deltas * stds + means
    
    # Reverse norm_factor for angle
    if norm_factor is not None:
        deltas_raw = deltas_raw.clone()
        deltas_raw[:, 4] = deltas_raw[:, 4] * (norm_factor * math.pi)
    
    # Extract components
    anchor_cx, anchor_cy, anchor_w, anchor_h, anchor_angle = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3], anchors[:, 4]
    dx, dy, dw, dh, dangle = deltas_raw[:, 0], deltas_raw[:, 1], deltas_raw[:, 2], deltas_raw[:, 3], deltas_raw[:, 4]
    
    # Decode boxes
    if proj_xy:
        cos_t = torch.cos(anchor_angle)
        sin_t = torch.sin(anchor_angle)
        pred_cx = anchor_cx + dx * anchor_w * cos_t - dy * anchor_h * sin_t
        pred_cy = anchor_cy + dx * anchor_w * sin_t + dy * anchor_h * cos_t
    else:
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)
    pred_angle = anchor_angle + dangle
    
    # Normalize angle to [-π, π]
    pred_angle = torch.atan2(torch.sin(pred_angle), torch.cos(pred_angle))
    
    # Edge_swap inverse: w_regular = max(gw, gh), h_regular = min(gw, gh), theta_regular
    if edge_swap:
        w_regular = torch.where(pred_w > pred_h, pred_w, pred_h)
        h_regular = torch.where(pred_w > pred_h, pred_h, pred_w)
        theta_regular = torch.where(pred_w > pred_h, pred_angle, norm_angle_le90(pred_angle + math.pi / 2))
        pred_w = w_regular
        pred_h = h_regular
        pred_angle = theta_regular
    
    # Stack into [N, 5] tensor
    boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_angle], dim=1)
    
    # Normalize to le90 convention (width >= height, angle in [-π/2, π/2))
    if normalize_le90:
        boxes = normalize_boxes_to_le90(boxes)
    
    return boxes


def _obb_to_hbb_obb(obb: torch.Tensor) -> torch.Tensor:
    """Convert OBB [N, 5] to HBB as [N, 5] (cx, cy, w, h, angle=0). Used for horizontal RPN targets."""
    xyxy = _obb_to_xyxy_hbb(obb)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    angle = torch.zeros_like(cx, device=obb.device)
    return torch.stack([cx, cy, w, h, angle], dim=1)


def _obb_to_xyxy_hbb(obb: torch.Tensor) -> torch.Tensor:
    """Convert obb [N, 5] (cx, cy, w, h, angle) to axis-aligned xyxy [N, 4] (HBB).

    Thin wrapper around obb_to_xyxy_gpu (works on CPU and GPU).
    """
    return obb_to_xyxy_gpu(obb)


def _box_iou_xyxy_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix [N, M] for axis-aligned boxes in xyxy format. boxes1 [N,4], boxes2 [M,4]."""
    try:
        from torchvision.ops import box_iou
        return box_iou(boxes1, boxes2)
    except ImportError:
        pass
    # Fallback: manual IoU for xyxy
    N, M = boxes1.shape[0], boxes2.shape[0]
    device = boxes1.device
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.maximum(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    iy1 = torch.maximum(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    ix2 = torch.minimum(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    iy2 = torch.minimum(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return (inter / (union + 1e-8)).to(device)


def match_oriented_anchors_to_gt(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    positive_iou_threshold: float = 0.7,
    negative_iou_threshold: float = 0.3,
    device: Optional[torch.device] = None,
    use_hbb_for_matching: bool = False,
    min_pos_iou: float = 0.3,
    match_low_quality: bool = True,
    gt_boxes_ignore: Optional[torch.Tensor] = None,
    ignore_iou_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match oriented anchors to ground truth boxes using oriented IoU or HBB IoU.
    
    When use_hbb_for_matching is True, matching uses the axis-aligned (horizontal)
    bounding box of each rotated box for IoU. This is recommended when using a single
    anchor angle (e.g. -90° for le90): otherwise only nearly vertical/horizontal GT
    get matched well; HBB matching gives better coverage for all orientations.
    
    Uses GPU-accelerated vectorized matching when USE_GPU_OPS is True (both oriented
    and HBB paths). This is 10-100x faster than the Python fallback versions.
    
    Args:
        anchors: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
        gt_boxes: Tensor of shape [M, 5] with format [cx, cy, w, h, angle]
        positive_iou_threshold: IoU threshold for positive matches
        negative_iou_threshold: IoU threshold for negative matches
        device: Optional device for computation
        use_hbb_for_matching: If True, use HBB (axis-aligned) IoU for matching instead of rotated IoU.
    
    Returns:
        Tuple of (labels, matched_gt_indices):
        - labels: Tensor of shape [N] with values: -1 (ignore), 0 (background), 1 (foreground)
        - matched_gt_indices: Tensor of shape [N] with index of matched GT box (-1 if no match)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for anchor matching.")
    
    if device is None:
        device = anchors.device if hasattr(anchors, 'device') else torch.device('cpu')
    
    # Ensure tensors are on device
    anchors = anchors.to(device).detach()
    gt_boxes = gt_boxes.to(device).detach()
    
    # Use GPU-accelerated vectorized matching when available (10-100x faster)
    if USE_GPU_OPS:
        labels, matched_gt_indices = match_anchors_to_gt_gpu(
            anchors=anchors,
            gt_boxes=gt_boxes,
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold,
            use_hbb_for_assignment=use_hbb_for_matching,
            min_pos_iou=min_pos_iou,
            match_low_quality=match_low_quality,
        )
        if gt_boxes_ignore is not None and gt_boxes_ignore.numel() > 0:
            thr = float(ignore_iou_threshold) if ignore_iou_threshold is not None else float(positive_iou_threshold)
            ign = gt_boxes_ignore.to(device).detach()
            try:
                from ..ops.gpu_ops import oriented_box_hbb_iou_gpu
                iou_ign = oriented_box_hbb_iou_gpu(anchors, ign) if use_hbb_for_matching else pairwise_rotated_iou(anchors, ign)
            except Exception:
                # Fallback: CPU IoU only for ignore regions; normal matching above stays on GPU.
                from ..ops.iou import batch_rbox_iou
                a_list = [RBox(*a.tolist()) for a in anchors.detach().cpu()]
                g_list = [RBox(*g.tolist()) for g in ign.detach().cpu()]
                iou_ign = torch.tensor(batch_rbox_iou(a_list, g_list, device=device), device=device)
            max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((anchors.shape[0],), device=device)
            ign_mask = (labels <= 0) & (max_iou_ign >= thr)
            labels[ign_mask] = -1
            matched_gt_indices[ign_mask] = -1
        return labels, matched_gt_indices
    else:
        labels = None
        matched_gt_indices = None
    
    # Fallback: HBB matching (Python) - convert to xyxy and use axis-aligned IoU
    if labels is None and use_hbb_for_matching:
        with torch.no_grad():
            anchors_xyxy = _obb_to_xyxy_hbb(anchors)
            gt_xyxy = _obb_to_xyxy_hbb(gt_boxes)
            iou_matrix = _box_iou_xyxy_matrix(anchors_xyxy, gt_xyxy)
        N, M = anchors.shape[0], gt_boxes.shape[0]
        labels = torch.full((N,), -1, dtype=torch.int64, device=device)
        matched_gt_indices = torch.full((N,), -1, dtype=torch.int64, device=device)
        if M == 0:
            labels.fill_(0)
            return labels, matched_gt_indices
        max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)
        best_anchor_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if match_low_quality:
            for gt_idx in range(M):
                if max_iou_per_gt[gt_idx] >= min_pos_iou:
                    anchor_idx = best_anchor_per_gt[gt_idx].item()
                    labels[anchor_idx] = 1
                    matched_gt_indices[anchor_idx] = gt_idx
                    best_anchor_mask[anchor_idx] = True
        positive_mask = max_iou_per_anchor >= positive_iou_threshold
        labels[positive_mask] = 1
        matched_gt_indices[positive_mask] = best_gt_per_anchor[positive_mask]
        negative_mask = (max_iou_per_anchor < negative_iou_threshold) & (~best_anchor_mask)
        labels[negative_mask] = 0
        # Apply ignore GT regions (if any) after assignment (do not override positives).
        if gt_boxes_ignore is not None and gt_boxes_ignore.numel() > 0:
            thr = float(ignore_iou_threshold) if ignore_iou_threshold is not None else float(positive_iou_threshold)
            with torch.no_grad():
                ign_xyxy = _obb_to_xyxy_hbb(gt_boxes_ignore.to(device).detach())
                iou_ign = _box_iou_xyxy_matrix(anchors_xyxy, ign_xyxy)
                max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((N,), device=device)
                ign_mask = (labels <= 0) & (max_iou_ign >= thr)
                labels[ign_mask] = -1
                matched_gt_indices[ign_mask] = -1
        return labels, matched_gt_indices
    
    # Fallback: Python loop version for oriented IoU (slow, for debugging only)
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    
    # Initialize labels and matched indices
    labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)
    matched_gt_indices = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)
    
    if num_gt == 0:
        labels.fill_(0)
        return labels, matched_gt_indices
    
    # Matching doesn't need gradients
    if labels is None:
        labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)
        matched_gt_indices = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)

    with torch.no_grad():
        # Use spatial filtering to reduce computation - only compute IoU for nearby anchors
        anchor_centers = anchors[:, :2]  # [N, 2] (cx, cy)
        anchor_max_size = torch.max(anchors[:, 2:4], dim=1)[0]  # [N]
        gt_centers = gt_boxes[:, :2]  # [M, 2]
        gt_max_size = torch.max(gt_boxes[:, 2:4], dim=1)[0]  # [M]
        
        iou_matrix = torch.zeros((num_anchors, num_gt), device=device)
        
        # Process GT boxes in chunks
        chunk_size = min(10, num_gt)
        if chunk_size <= 0:
            labels.fill_(0)
            return labels, matched_gt_indices
        
        for chunk_start in range(0, num_gt, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_gt)
            
            for gt_idx in range(chunk_start, chunk_end):
                gt_center = gt_centers[gt_idx:gt_idx+1]  # [1, 2]
                gt_size = gt_max_size[gt_idx].item()
                
                # Find anchors within reasonable distance (3x max size)
                distance_threshold = gt_size * 3.0
                distances = torch.norm(anchor_centers - gt_center, dim=1)
                combined_threshold = distance_threshold + anchor_max_size
                
                candidate_mask = distances < combined_threshold
                candidate_indices = torch.where(candidate_mask)[0]
                del distances  # Free memory
                
                if len(candidate_indices) == 0:
                    continue
                
                # Limit candidates to avoid memory issues (shouldn't happen often)
                max_candidates_per_gt = 5000
                if len(candidate_indices) > max_candidates_per_gt:
                    # Recompute distances only for candidates (more memory efficient)
                    candidate_distances = torch.norm(anchor_centers[candidate_indices] - gt_center, dim=1)
                    _, closest_indices = torch.topk(candidate_distances, max_candidates_per_gt, largest=False)
                    candidate_indices = candidate_indices[closest_indices]
                    del candidate_distances
                
                # Convert candidate anchors to RBox format
                candidate_anchors = anchors[candidate_indices].cpu()
                candidate_anchor_list = [RBox(*a.tolist()) for a in candidate_anchors]
                gt_rbox = RBox(*gt_boxes[gt_idx].cpu().tolist())
                gt_list = [gt_rbox]
                
                # Use batch IoU computation
                from ..ops.iou import batch_rbox_iou
                try:
                    iou_values = batch_rbox_iou(candidate_anchor_list, gt_list, device=device, intersection_backend="auto")
                    # iou_values is a list of lists: [[iou1], [iou2], ...]
                    for i, anchor_idx in enumerate(candidate_indices):
                        if iou_values and len(iou_values) > i and len(iou_values[i]) > 0:
                            iou_matrix[anchor_idx, gt_idx] = iou_values[i][0]
                except Exception:
                    # Fallback: compute individually
                    for anchor_idx in candidate_indices:
                        anchor_rbox = RBox(*anchors[anchor_idx].cpu().tolist())
                        try:
                            iou_val = iou.rbox_iou(anchor_rbox, gt_rbox)
                            iou_matrix[anchor_idx, gt_idx] = iou_val
                        except (ValueError, AttributeError):
                            pass
                
                # Clear intermediate tensors to free memory
                del candidate_anchors, candidate_anchor_list
        
        # For each ground truth box, find the anchor with highest IoU
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)  # [M]
        
        # Mark best anchors for each GT as positive (if match_low_quality and IoU >= min_pos_iou)
        if match_low_quality:
            for gt_idx in range(num_gt):
                if max_iou_per_gt[gt_idx] >= min_pos_iou:
                    anchor_idx = best_anchor_per_gt[gt_idx].item()
                    labels[anchor_idx] = 1
                    matched_gt_indices[anchor_idx] = gt_idx
        
        # For each anchor, find the GT with highest IoU
        max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)  # [N]
        
        # Mark anchors above positive threshold as positive
        positive_mask = max_iou_per_anchor >= positive_iou_threshold
        labels[positive_mask] = 1
        matched_gt_indices[positive_mask] = best_gt_per_anchor[positive_mask]
        
        # Mark anchors below negative threshold as negative
        negative_mask = max_iou_per_anchor < negative_iou_threshold
        labels[negative_mask] = 0
    
    # Anchors between thresholds are ignored (label = -1)
    
    # Apply ignore GT regions (if any) after assignment (do not override positives).
    if gt_boxes_ignore is not None and gt_boxes_ignore.numel() > 0:
        thr = float(ignore_iou_threshold) if ignore_iou_threshold is not None else float(positive_iou_threshold)
        ign = gt_boxes_ignore.to(device).detach()
        if USE_GPU_OPS:
            # Labels may come from the GPU fast-path above; we still need GPU IoU here.
            try:
                from ..ops.gpu_ops import oriented_box_hbb_iou_gpu
                iou_ign = oriented_box_hbb_iou_gpu(anchors, ign) if use_hbb_for_matching else pairwise_rotated_iou(anchors, ign)
            except Exception:
                # Fallback: CPU IoU (slower, but keeps training running)
                from ..ops.iou import batch_rbox_iou
                a_list = [RBox(*a.tolist()) for a in anchors.detach().cpu()]
                g_list = [RBox(*g.tolist()) for g in ign.detach().cpu()]
                iou_ign = torch.tensor(batch_rbox_iou(a_list, g_list, device=device), device=device)
            max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((num_anchors,), device=device)
        else:
            from ..ops.iou import batch_rbox_iou
            a_list = [RBox(*a.tolist()) for a in anchors.detach().cpu()]
            g_list = [RBox(*g.tolist()) for g in ign.detach().cpu()]
            iou_ign = torch.tensor(batch_rbox_iou(a_list, g_list, device=device), device=device)
            max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((num_anchors,), device=device)
        ign_mask = (labels <= 0) & (max_iou_ign >= thr)
        labels[ign_mask] = -1
        matched_gt_indices[ign_mask] = -1

    return labels, matched_gt_indices


def compute_oriented_rpn_loss(
    objectness_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    gt_boxes_ignore: Optional[List[torch.Tensor]] = None,
    positive_iou_threshold: float = 0.7,
    negative_iou_threshold: float = 0.3,
    box_reg_weight: float = 1.0,
    fg_bg_sampling_ratio: float = 0.5,
    batch_size_per_image: int = 256,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
    use_horizontal_targets: bool = False,
    use_hbb_for_matching: bool = False,
    min_pos_iou: float = 0.3,
    match_low_quality: bool = True,
    sample_from_all_levels: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute RPN losses for oriented object detection.
    
    When sample_from_all_levels is True (default, MMDet-style): anchors from all FPN
    levels are pooled per image, then batch_size_per_image (e.g. 256) samples are
    drawn from this pool. This gives better use of positives and matches MMDetection.
    When False: batch_size_per_image is split across levels (256/num_levels per level)
    and loss is averaged over levels (older behavior; can cause GT coverage plateau).
    
    Args:
        objectness_logits: List of classification logits from RPN head
        bbox_regression: List of box regression predictions from RPN head
        anchors: List of anchor tensors for each level
        gt_boxes: List of ground truth boxes per image (each as [M, 5] tensor)
        gt_boxes_ignore: Optional list of ignored GT boxes per image (each as [I, 5] tensor).
        image_sizes: List of (height, width) for each image
        positive_iou_threshold: IoU threshold for positive anchors
        negative_iou_threshold: IoU threshold for negative anchors
        box_reg_weight: Weight for box regression loss
        fg_bg_sampling_ratio: Ratio of foreground to background in sampled anchors
        batch_size_per_image: Number of anchors to sample per image (total when sample_from_all_levels=True)
        target_means: Optional normalization means for box regression targets (cx, cy, w, h, angle)
        target_stds: Optional normalization stds for box regression targets
        use_horizontal_targets: If True, encode RPN targets as horizontal boxes
        use_hbb_for_matching: If True, use HBB IoU for anchor-GT assignment
        min_pos_iou: Min IoU for best-anchor-per-GT to be positive
        match_low_quality: If True, force best anchor per GT to positive when IoU >= min_pos_iou
        sample_from_all_levels: If True, sample batch_size_per_image from all levels combined (MMDet-style).
    
    Returns:
        Dictionary with loss values:
        - "loss_objectness": Classification loss (foreground/background)
        - "loss_rpn_box_reg": Box regression loss
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation.")
    
    device = objectness_logits[0].device if objectness_logits else torch.device('cpu')
    num_images = len(gt_boxes)
    num_levels = len(objectness_logits)
    B = num_images
    timing_enabled = TRACE_RPN_LOSS_TIMING and not getattr(compute_oriented_rpn_loss, "_timed_first_call", False)
    if timing_enabled:
        setattr(compute_oriented_rpn_loss, "_timed_first_call", True)
        timing_start = time.perf_counter()

        def _sync_and_time() -> float:
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            return time.perf_counter()

        print(
            f"[rpn-loss timing] begin: images={B} levels={num_levels} "
            f"use_hbb_for_matching={use_hbb_for_matching}",
            flush=True,
        )
    else:
        def _sync_and_time() -> float:
            return time.perf_counter()
    
    # Per-level batch size when not sampling from all levels (legacy)
    batch_size_per_level = max(1, batch_size_per_image // num_levels)
    
    # First pass: per level, compute labels and targets and reshape logits/pred
    # Store per-level tensors and anchors_per_image for each level
    level_logits: List[torch.Tensor] = []
    level_pred: List[torch.Tensor] = []
    level_labels: List[torch.Tensor] = []
    level_targets: List[torch.Tensor] = []
    anchors_per_image_per_level: List[int] = []
    all_loss_objectness: List[torch.Tensor] = []
    all_loss_rpn_box_reg: List[torch.Tensor] = []
    
    for level_idx in range(num_levels):
        level_t0 = _sync_and_time() if timing_enabled else 0.0
        # Get shapes for this level
        B, C_cls, H, W = objectness_logits[level_idx].shape
        B_reg, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
        
        # RPN format: C_cls = num_anchors * 1, C_reg = num_anchors * 4 (dx, dy, dw, dh - no angle)
        if C_reg % 4 != 0:
            raise RuntimeError(
                f"Invalid RPN format: C_reg={C_reg} must be divisible by 4 (4 reg params per anchor)"
            )
        num_anchors = C_reg // 4
        if num_anchors != C_cls:
            raise RuntimeError(
                f"RPN format mismatch: C_cls={C_cls} but C_reg={C_reg} suggests num_anchors={num_anchors}. "
                f"Expected C_cls==num_anchors for RPN format (1 cls output per anchor)"
            )
        
        # Single objectness score per anchor
        cls_out_channels = 1
        obj_logits = objectness_logits[level_idx].view(B, num_anchors, 1, H, W)
        obj_logits = obj_logits.permute(0, 3, 4, 1, 2).contiguous().view(-1, 1)
        # Convert to 2 channels for binary classification loss (bg/fg)
        obj_logits = torch.cat([torch.zeros_like(obj_logits), obj_logits], dim=1)  # [N, 2] logits: [0, z]
        
        # Bbox regression: 4 params (dx, dy, dw, dh)
        bbox_pred = bbox_regression[level_idx].view(B_reg, num_anchors, 4, H_reg, W_reg)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous().view(-1, 4)
        
        # Process anchors for this level (no gradients needed)
        # CRITICAL: All anchor operations must be in no_grad() to prevent tracking
        # Expanding anchors creates [B*N, 5] tensors which can be huge (B*783K for large images)
        with torch.no_grad():
            level_anchors_raw = anchors[level_idx]  # [N, 5] where N = H*W*num_anchors
            if isinstance(level_anchors_raw, torch.Tensor):
                level_anchors_raw = level_anchors_raw.detach().to(device)
            else:
                level_anchors_raw = torch.tensor(level_anchors_raw, dtype=torch.float32, device=device, requires_grad=False)
            
            # Debug: verify anchor count matches expected num_anchors
            expected_anchors_per_image = H * W * num_anchors
            if len(level_anchors_raw) != expected_anchors_per_image:
                raise RuntimeError(
                    f"Anchor count mismatch at level {level_idx}: "
                    f"anchors[level_idx] has {len(level_anchors_raw)} anchors, "
                    f"but expected H*W*num_anchors = {H}*{W}*{num_anchors} = {expected_anchors_per_image}. "
                    f"This suggests anchor generation created {len(level_anchors_raw) / expected_anchors_per_image:.1f}x the expected anchors."
                )
            
            # Expand anchors to match batch size: [N, 5] -> [B*N, 5]
            # CRITICAL: Ensure expansion happens entirely in no_grad context
            # Use clone() to ensure we get a fresh tensor that's not tracked
            level_anchors = level_anchors_raw.unsqueeze(0).repeat(B, 1, 1).view(-1, 5).clone()
            level_anchors = level_anchors.detach()
            level_anchors.requires_grad_(False)
        
        # Process each image in the batch for this level
        anchors_per_image = len(level_anchors) // B
        if timing_enabled:
            print(
                f"[rpn-loss timing] level {level_idx} start: "
                f"grid={H}x{W} anchors_per_loc={num_anchors} "
                f"anchors_per_image={anchors_per_image}",
                flush=True,
            )
        
        img_labels_list: List[torch.Tensor] = []
        img_regression_targets_list: List[torch.Tensor] = []
        
        # Track match statistics across all images for this level
        total_positive = 0
        total_negative = 0
        total_gt_boxes = 0
        matched_gt_count = 0
        
        for img_idx in range(B):
            # Get anchors for this image at this level
            img_anchors = level_anchors[img_idx * anchors_per_image:(img_idx + 1) * anchors_per_image]
            img_anchors = img_anchors.to(device).detach()
            
            # Get ground truth boxes for this image
            if isinstance(gt_boxes[img_idx], torch.Tensor):
                img_gt_boxes = gt_boxes[img_idx].to(device).detach()
            else:
                img_gt_boxes = torch.tensor(gt_boxes[img_idx], dtype=torch.float32, device=device, requires_grad=False)
            
            total_gt_boxes += len(img_gt_boxes)
            
            if len(img_gt_boxes) == 0:
                # No ground truth - all anchors are background
                labels = torch.zeros(len(img_anchors), dtype=torch.int64, device=device)
                regression_targets = torch.zeros((len(img_anchors), 4), dtype=torch.float32, device=device)
                total_negative += len(img_anchors)
            else:
                # Match anchors to GT for this level
                img_gt_ignore = None
                if gt_boxes_ignore is not None and img_idx < len(gt_boxes_ignore):
                    img_gt_ignore = gt_boxes_ignore[img_idx].to(device).detach()
                img_t0 = _sync_and_time() if timing_enabled else 0.0
                if timing_enabled and (img_idx < 3 or img_idx % 10 == 0):
                    print(
                        f"[rpn-loss timing] level {level_idx} img {img_idx} match start: "
                        f"anchors={len(img_anchors)} gt={len(img_gt_boxes)} "
                        f"ignore={0 if img_gt_ignore is None else len(img_gt_ignore)}",
                        flush=True,
                    )
                labels, matched_indices = match_oriented_anchors_to_gt(
                    img_anchors,
                    img_gt_boxes,
                    positive_iou_threshold,
                    negative_iou_threshold,
                    device,
                    use_hbb_for_matching=use_hbb_for_matching,
                    min_pos_iou=min_pos_iou,
                    match_low_quality=match_low_quality,
                    gt_boxes_ignore=img_gt_ignore,
                    ignore_iou_threshold=positive_iou_threshold,
                )
                if timing_enabled and (img_idx < 3 or img_idx % 10 == 0):
                    img_t1 = _sync_and_time()
                    print(
                        f"[rpn-loss timing] level {level_idx} img {img_idx} match done: "
                        f"{img_t1 - img_t0:.3f}s",
                        flush=True,
                    )
                
                # Count matches
                positive_mask = labels == 1
                negative_mask = labels == 0
                total_positive += positive_mask.sum().item()
                total_negative += negative_mask.sum().item()
                
                # Count how many GT boxes were matched
                if positive_mask.any():
                    matched_gt_indices_unique = matched_indices[positive_mask].unique()
                    matched_gt_count += len(matched_gt_indices_unique)
                
                # Compute regression targets for positive anchors (4 params: dx, dy, dw, dh)
                regression_targets = torch.zeros((len(img_anchors), 4), dtype=torch.float32, device=device)
                if positive_mask.any():
                    matched_gt = img_gt_boxes[matched_indices[positive_mask]]
                    if use_horizontal_targets:
                        matched_gt = _obb_to_hbb_obb(matched_gt)
                    matched_anchors = img_anchors[positive_mask].detach()
                    regression_targets[positive_mask] = encode_rpn_boxes(matched_anchors, matched_gt)
            
            img_labels_list.append(labels)
            img_regression_targets_list.append(regression_targets)
        if timing_enabled:
            level_t1 = _sync_and_time()
            print(
                f"[rpn-loss timing] level {level_idx} labels/targets done: "
                f"{level_t1 - level_t0:.3f}s total={level_t1 - timing_start:.3f}s",
                flush=True,
            )
        
        # Concatenate labels and targets for this level (across batch)
        labels_level = torch.cat(img_labels_list, dim=0)  # [B*anchors_per_image]
        regression_targets_level = torch.cat(img_regression_targets_list, dim=0)  # [B*anchors_per_image, 4]
        
        # Log match statistics for this level (after labels_level is created)
        match_rate = matched_gt_count / total_gt_boxes if total_gt_boxes > 0 else 0.0
        logger.trace(
            "RPN level {} match stats: total_anchors={}, positive={}, negative={}, "
            "gt_boxes={}, matched_gt={}, match_rate={:.1%}",
            level_idx, len(labels_level), total_positive, total_negative,
            total_gt_boxes, matched_gt_count, match_rate
        )
        
        # Verify shapes match between obj_logits and labels_level
        if obj_logits.shape[0] != len(labels_level):
            raise RuntimeError(
                f"Shape mismatch in RPN loss computation at level {level_idx}: "
                f"obj_logits.shape[0]={obj_logits.shape[0]} but len(labels_level)={len(labels_level)}. "
                f"This suggests a mismatch between the number of anchors in the model output and anchor generation. "
                f"Expected {obj_logits.shape[0]} anchors but got {len(labels_level)} labels."
            )
        
        # Store for all-levels sampling (or use now for per-level sampling)
        level_logits.append(obj_logits)
        level_pred.append(bbox_pred)
        level_labels.append(labels_level)
        level_targets.append(regression_targets_level)
        anchors_per_image_per_level.append(anchors_per_image)
        
        if not sample_from_all_levels:
            # Legacy: sample from this level only (256/num_levels per level)
            fg_indices = (labels_level == 1).nonzero(as_tuple=True)[0]
            bg_indices = (labels_level == 0).nonzero(as_tuple=True)[0]
            num_fg = min(len(fg_indices), int(batch_size_per_level * fg_bg_sampling_ratio * B))
            num_bg = min(len(bg_indices), int(batch_size_per_level * (1 - fg_bg_sampling_ratio) * B))
            logger.trace(
                "RPN level {}: anchors={}, fg_candidates={}, bg_candidates={}, sampled_fg={}, sampled_bg={}",
                level_idx, len(labels_level), len(fg_indices), len(bg_indices), num_fg, num_bg
            )
            if num_fg > 0:
                sampled_fg = fg_indices[torch.randperm(len(fg_indices), device=device)[:num_fg]]
            else:
                sampled_fg = torch.tensor([], dtype=torch.int64, device=device)
            if num_bg > 0:
                sampled_bg = bg_indices[torch.randperm(len(bg_indices), device=device)[:num_bg]]
            else:
                sampled_bg = torch.tensor([], dtype=torch.int64, device=device)
            sampled_indices = torch.cat([sampled_fg, sampled_bg], dim=0)
            if len(sampled_indices) > 0:
                valid_mask = labels_level[sampled_indices] >= 0
                if valid_mask.any():
                    loss_objectness_level = F.cross_entropy(
                        obj_logits[sampled_indices][valid_mask].clone(),
                        labels_level[sampled_indices][valid_mask].long(),
                        reduction='mean',
                    )
                else:
                    loss_objectness_level = (obj_logits[0:1] * 0.0).sum()
                positive_sampled = sampled_indices[torch.isin(sampled_indices, fg_indices)]
                if len(positive_sampled) > 0:
                    loss_rpn_box_reg_level = F.smooth_l1_loss(
                        bbox_pred[positive_sampled].clone(),
                        regression_targets_level[positive_sampled],
                        beta=1.0 / 9.0,
                        reduction='mean',
                    )
                else:
                    loss_rpn_box_reg_level = (bbox_pred[0:1] * 0.0).sum()
            else:
                loss_objectness_level = (obj_logits[0:1] * 0.0).sum()
                loss_rpn_box_reg_level = (bbox_pred[0:1] * 0.0).sum()
            all_loss_objectness.append(loss_objectness_level)
            all_loss_rpn_box_reg.append(loss_rpn_box_reg_level)
    
    if sample_from_all_levels:
        # MMDet-style: sample batch_size_per_image from all levels combined per image
        all_loss_objectness = []
        all_loss_rpn_box_reg = []
        for img_idx in range(B):
            logits_i = torch.cat(
                [level_logits[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]]
                for l in range(num_levels)],
                dim=0,
            )
            pred_i = torch.cat(
                [level_pred[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]]
                for l in range(num_levels)],
                dim=0,
            )
            labels_i = torch.cat(
                [level_labels[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]]
                for l in range(num_levels)],
                dim=0,
            )
            targets_i = torch.cat(
                [level_targets[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]]
                for l in range(num_levels)],
                dim=0,
            )
            fg_i = (labels_i == 1).nonzero(as_tuple=True)[0]
            bg_i = (labels_i == 0).nonzero(as_tuple=True)[0]
            num_fg = min(len(fg_i), int(batch_size_per_image * fg_bg_sampling_ratio))
            num_bg = min(len(bg_i), batch_size_per_image - num_fg)
            if num_fg > 0:
                sampled_fg = fg_i[torch.randperm(len(fg_i), device=device)[:num_fg]]
            else:
                sampled_fg = torch.tensor([], dtype=torch.int64, device=device)
            if num_bg > 0:
                sampled_bg = bg_i[torch.randperm(len(bg_i), device=device)[:num_bg]]
            else:
                sampled_bg = torch.tensor([], dtype=torch.int64, device=device)
            sampled_idx = torch.cat([sampled_fg, sampled_bg], dim=0)
            if len(sampled_idx) > 0:
                valid = labels_i[sampled_idx] >= 0
                if valid.any():
                    loss_obj_i = F.cross_entropy(
                        logits_i[sampled_idx][valid].clone(),
                        labels_i[sampled_idx][valid].long(),
                        reduction='mean',
                    )
                else:
                    loss_obj_i = (logits_i[0:1] * 0.0).sum()
                pos_sampled = sampled_idx[torch.isin(sampled_idx, fg_i)]
                if len(pos_sampled) > 0:
                    # MMDet/MMRotate normalization: sum over positive anchors' 4 coords
                    # divided by the total number of sampled anchors (pos + neg), not by
                    # the positive count. Keeps the cls/reg balance independent of how
                    # many positives an image happens to have.
                    loss_reg_i = F.smooth_l1_loss(
                        pred_i[pos_sampled].clone(),
                        targets_i[pos_sampled],
                        beta=1.0 / 9.0,
                        reduction='sum',
                    ) / float(len(sampled_idx))
                else:
                    loss_reg_i = (pred_i[0:1] * 0.0).sum()
            else:
                loss_obj_i = (logits_i[0:1] * 0.0).sum()
                loss_reg_i = (pred_i[0:1] * 0.0).sum()
            all_loss_objectness.append(loss_obj_i)
            all_loss_rpn_box_reg.append(loss_reg_i)
    
    # Aggregate losses (over levels when not sample_from_all_levels, over images when sample_from_all_levels)
    if all_loss_objectness:
        loss_objectness = torch.stack(all_loss_objectness).mean()
    else:
        # Maintain gradient flow: compute zero loss from model outputs
        # Use first level's objectness logits to maintain connection
        if len(objectness_logits) > 0 and objectness_logits[0].numel() > 0:
            loss_objectness = (objectness_logits[0] * 0.0).sum()
        else:
            # Fallback: create a small constant loss from device
            loss_objectness = torch.tensor(0.0, device=device, requires_grad=True)
    
    if all_loss_rpn_box_reg:
        loss_rpn_box_reg = torch.stack(all_loss_rpn_box_reg).mean()
    else:
        # Maintain gradient flow: compute zero loss from model outputs
        # Use first level's bbox regression to maintain connection
        if len(bbox_regression) > 0 and bbox_regression[0].numel() > 0:
            loss_rpn_box_reg = (bbox_regression[0] * 0.0).sum()
        else:
            # Fallback: create a small constant loss from device
            loss_rpn_box_reg = torch.tensor(0.0, device=device, requires_grad=True)

    # Ensure returned losses are finite (avoid nan/inf from edge cases)
    def _make_finite(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            return t
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t, device=t.device))

    loss_objectness = _make_finite(loss_objectness)
    loss_rpn_box_reg = _make_finite(box_reg_weight * loss_rpn_box_reg)

    return {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }


def generate_oriented_proposals(
    objectness_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    score_threshold: float = 0.0,
    nms_threshold: float = 0.7,
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 1000,
    min_size: float = 0.0,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
) -> List[torch.Tensor]:
    """Generate oriented proposals from RPN predictions.
    
    Args:
        objectness_logits: List of classification logits from RPN head
        bbox_regression: List of box regression predictions from RPN head
        anchors: List of anchor tensors for each level
        image_sizes: List of (height, width) for each image
        score_threshold: Minimum objectness score to keep proposals
        nms_threshold: IoU threshold for NMS
        pre_nms_top_n: Number of top proposals before NMS
        post_nms_top_n: Number of top proposals after NMS
        min_size: Minimum box size (in pixels) to keep proposals
        target_means: Optional normalization means for box decoding (cx, cy, w, h, angle)
        target_stds: Optional normalization stds for box decoding
        target_norm_factor: Optional angle scale (e.g. 2.0 for le90); accepted for API consistency, RPN typically uses None.
    
    Returns:
        List of proposal tensors, one per image, each of shape [N, 5] with format [cx, cy, w, h, angle]
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for proposal generation.")
    
    with logger.trace_block(
        "generate_oriented_proposals(num_images={}, levels={}, pre_nms_top_n={}, post_nms_top_n={})",
        len(image_sizes), len(objectness_logits), pre_nms_top_n, post_nms_top_n
    ):
        device = objectness_logits[0].device if objectness_logits else torch.device('cpu')
        num_images = len(image_sizes)
        
        all_proposals = []
        
        for img_idx in range(num_images):
            # Collect proposals from all levels for this image
            img_proposals = []
            img_scores = []
            
            for level_idx in range(len(objectness_logits)):
                # Get shapes
                B, C_cls, H, W = objectness_logits[level_idx].shape
                B_reg, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
                
                # RPN format: C_cls = num_anchors * 1, C_reg = num_anchors * 4
                if C_reg % 4 != 0:
                    raise RuntimeError(
                        f"Invalid RPN format in proposals: C_reg={C_reg} must be divisible by 4"
                    )
                num_anchors = C_reg // 4
                if num_anchors != C_cls:
                    raise RuntimeError(
                        f"RPN format mismatch in proposals: C_cls={C_cls} but C_reg={C_reg} suggests num_anchors={num_anchors}"
                    )
                
                obj_logits = objectness_logits[level_idx][img_idx:img_idx+1].view(num_anchors, 1, H, W)
                obj_logits = obj_logits.permute(2, 3, 0, 1).contiguous().view(-1, 1)
                scores = torch.sigmoid(obj_logits[:, 0])  # Objectness scores
                
                # Bbox regression: 4 params (dx, dy, dw, dh)
                bbox_pred = bbox_regression[level_idx][img_idx:img_idx+1].view(num_anchors, 4, H_reg, W_reg)
                bbox_pred = bbox_pred.permute(2, 3, 0, 1).contiguous().view(-1, 4)
                
                level_anchors = anchors[level_idx].to(device)  # [H*W*num_anchors, 5]
                level_anchors = level_anchors.detach()
                
                valid_mask = scores >= score_threshold
                
                if valid_mask.sum() > pre_nms_top_n:
                    top_k = min(pre_nms_top_n * 2, len(scores))
                    # Break ties randomly so we don't always select top-left anchors when objectness is ~uniform (early training)
                    scores_for_topk = scores + 1e-5 * (torch.rand_like(scores, device=scores.device) * 2 - 1)
                    _, top_indices = torch.topk(scores_for_topk, top_k)
                    valid_mask = torch.zeros_like(scores, dtype=torch.bool)
                    valid_mask[top_indices] = True
                    valid_mask = valid_mask & (scores >= score_threshold)
                
                if valid_mask.any():
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    
                    with torch.no_grad():
                        decoded_boxes = decode_rpn_boxes(
                            level_anchors[valid_indices], 
                            bbox_pred[valid_indices].detach(),
                            target_means=target_means, target_stds=target_stds,
                        ).detach()
                    
                    img_proposals.append(decoded_boxes)
                    img_scores.append(scores[valid_indices])
            
            if not img_proposals:
                # No proposals for this image
                logger.trace(f"Image {img_idx}: No proposals generated")
                all_proposals.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                continue
            
            # Concatenate proposals from all levels
            proposals = torch.cat(img_proposals, dim=0)  # [total, 5]
            scores = torch.cat(img_scores, dim=0)  # [total]
            logger.trace(f"Image {img_idx}: {len(proposals)} proposals before filtering/NMS")
            
            # Ensure proposals and scores are on the correct device
            proposals = proposals.to(device)
            scores = scores.to(device)
            
            # Clip boxes to image boundaries and cap w/h to avoid huge proposals
            img_h, img_w = image_sizes[img_idx]
            proposals[:, 0] = torch.clamp(proposals[:, 0], min=0, max=img_w)  # cx
            proposals[:, 1] = torch.clamp(proposals[:, 1], min=0, max=img_h)  # cy
            max_side = float(max(img_w, img_h)) * 2.0  # allow up to 2x image size
            proposals[:, 2] = torch.clamp(proposals[:, 2], min=min_size, max=max_side)  # w
            proposals[:, 3] = torch.clamp(proposals[:, 3], min=min_size, max=max_side)  # h
            
            # Filter out invalid proposals:
            # - Width and height must be positive and >= min_size
            # - No NaN or Inf values
            # - Width and height must be > 0 (not just >= min_size, but actually > 0)
            valid_mask = (
                (proposals[:, 2] > min_size) &  # width > min_size
                (proposals[:, 3] > min_size) &   # height > min_size
                torch.isfinite(proposals).all(dim=1)  # No NaN or Inf
            )
            proposals = proposals[valid_mask]
            scores = scores[valid_mask]
            
            if len(proposals) == 0:
                all_proposals.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                continue
            
            # Sort by score and take top pre_nms_top_n (break ties randomly to avoid top-left bias in early training)
            if len(scores) > pre_nms_top_n:
                scores_for_topk = scores + 1e-5 * (torch.rand_like(scores, device=scores.device) * 2 - 1)
                top_indices = torch.topk(scores_for_topk, pre_nms_top_n).indices
                proposals = proposals[top_indices]
                scores = scores[top_indices]
            
            # RPN proposal pruning: use HBB NMS on enclosing boxes for speed, then keep
            # the original oriented proposals. Final ROI detections can still use OBB NMS.
            if USE_GPU_OPS:
                keep_indices = hbb_nms_for_oriented_boxes_gpu(
                    proposals, scores, iou_threshold=nms_threshold, max_detections=post_nms_top_n
                )
                proposals = proposals[keep_indices]
            else:
                # Fallback to Python implementation (slow, for debugging only)
                # Filter out invalid proposals before converting to RBox
                # (width and height must be > 0, no NaN/Inf)
                valid_proposal_mask = (
                    (proposals[:, 2] > 0) &  # width > 0
                    (proposals[:, 3] > 0) &  # height > 0
                    torch.isfinite(proposals).all(dim=1)  # No NaN or Inf
                )
                valid_proposals = proposals[valid_proposal_mask]
                valid_scores = scores[valid_proposal_mask]
                
                if len(valid_proposals) == 0:
                    all_proposals.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                    continue
                
                # Convert to RBox objects, filtering out any that fail conversion
                # Track mapping from proposals_list index to valid_proposals index
                proposals_list = []
                scores_list = []
                list_to_valid_map = []  # Maps proposals_list index to valid_proposals index
                
                for valid_idx, (p, s) in enumerate(zip(valid_proposals, valid_scores)):
                    try:
                        rbox = RBox(*p.tolist())
                        # Validate by checking corners
                        corners = rbox.corners()
                        if len(corners) >= 3:
                            proposals_list.append(rbox)
                            scores_list.append(s.item())
                            list_to_valid_map.append(valid_idx)
                    except (ValueError, AttributeError, TypeError):
                        # Skip invalid boxes
                        continue
                
                if not proposals_list:
                    all_proposals.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                    continue
                
                keep_list_indices = nms.oriented_nms(proposals_list, scores_list, nms_threshold)
                if not keep_list_indices:
                    all_proposals.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                    continue
                
                # Map back: proposals_list index -> valid_proposals index -> original proposals index
                # First map to valid_proposals indices
                valid_proposals_indices = torch.where(valid_proposal_mask)[0]
                keep_valid_indices = [list_to_valid_map[idx] for idx in keep_list_indices]
                # Then map to original proposals indices
                keep_indices = valid_proposals_indices[torch.tensor(keep_valid_indices, device=device, dtype=torch.int64)]
                proposals = proposals[keep_indices]
            
            # Take top post_nms_top_n
            if len(proposals) > post_nms_top_n:
                proposals = proposals[:post_nms_top_n]
            
            # Log proposal statistics for debugging
            if len(proposals) > 0:
                logger.trace(
                    "RPN proposals (img {}): count={}, "
                    "cx_range=[{:.1f}, {:.1f}], cy_range=[{:.1f}, {:.1f}], "
                    "w_range=[{:.1f}, {:.1f}], h_range=[{:.1f}, {:.1f}], "
                    "angle_range=[{:.3f}, {:.3f}]",
                    img_idx, len(proposals),
                    proposals[:, 0].min().item(), proposals[:, 0].max().item(),
                    proposals[:, 1].min().item(), proposals[:, 1].max().item(),
                    proposals[:, 2].min().item(), proposals[:, 2].max().item(),
                    proposals[:, 3].min().item(), proposals[:, 3].max().item(),
                    proposals[:, 4].min().item(), proposals[:, 4].max().item()
                )
                logger.trace(
                    "RPN proposals (img {}): mean_cx={:.1f}, mean_cy={:.1f}, "
                    "mean_w={:.1f}, mean_h={:.1f}, mean_angle={:.3f}",
                    img_idx,
                    proposals[:, 0].mean().item(),
                    proposals[:, 1].mean().item(),
                    proposals[:, 2].mean().item(),
                    proposals[:, 3].mean().item(),
                    proposals[:, 4].mean().item()
                )
            else:
                logger.trace("RPN proposals (img {}): count=0 (no proposals generated!)", img_idx)
            
            all_proposals.append(proposals)
    
        # Log final proposal counts
        total_proposals = sum(len(p) for p in all_proposals)
        logger.trace_value("Total proposals generated", total_proposals)
        for img_idx, proposals in enumerate(all_proposals):
            logger.trace(f"Image {img_idx}: {len(proposals)} proposals")
    
    return all_proposals


def _obb_cxcywha_to_xyxy(obb: torch.Tensor) -> torch.Tensor:
    """Convert obb [N, 5] (cx, cy, w, h, angle) to xyxy [N, 4]. Works for any angle."""
    cx, cy, w, h = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3]
    w2, h2 = w / 2, h / 2
    x1 = cx - w2
    y1 = cy - h2
    x2 = cx + w2
    y2 = cy + h2
    return torch.stack([x1, y1, x2, y2], dim=1)


def generate_horizontal_proposals(
    objectness_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    score_threshold: float = 0.0,
    nms_threshold: float = 0.7,
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 1000,
    min_size: float = 0.0,
    target_means: Optional[Tuple[float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float]] = None,
    deterministic: bool = False,
) -> List[torch.Tensor]:
    """Generate axis-aligned (horizontal) proposals in xyxy format.
    
    Used by Rotated Faster R-CNN (MMRotate): the RPN outputs horizontal
    proposals [N, 4] (x1, y1, x2, y2), and the ROI head regresses final
    oriented boxes from those horizontal RoIs.
    
    Args:
        objectness_logits: List of classification logits from RPN head
        bbox_regression: List of box regression predictions from RPN head (4 params)
        anchors: List of anchor tensors (horizontal: angle=0) for each level
        image_sizes: List of (height, width) for each image
        score_threshold: Minimum objectness score
        nms_threshold: IoU threshold for NMS (axis-aligned)
        pre_nms_top_n: Number of top proposals before NMS
        post_nms_top_n: Number of top proposals after NMS
        min_size: Minimum box size
        target_means: Optional means for 4-param decode. Default: (0,0,0,0)
        target_stds: Optional stds for 4-param decode. Default: (1,1,1,1)
        deterministic: If True, use score-only top-k (no random tie-break) for reproducible export.
    
    Returns:
        List of proposal tensors, one per image, each [N, 4] xyxy (x1, y1, x2, y2).
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for proposal generation.")
    try:
        from torchvision.ops import nms as torchvision_nms
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for horizontal RPN NMS.") from exc
    
    device = objectness_logits[0].device if objectness_logits else torch.device("cpu")
    num_images = len(image_sizes)
    all_proposals_xyxy = []
    
    for img_idx in range(num_images):
        img_proposals = []
        img_scores = []
        
        for level_idx in range(len(objectness_logits)):
            B, C_cls, H, W = objectness_logits[level_idx].shape
            B_reg, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
            if C_reg % 4 != 0:
                raise RuntimeError(f"Invalid RPN format: C_reg={C_reg} must be divisible by 4")
            num_anchors = C_reg // 4
            if num_anchors != C_cls:
                raise RuntimeError(f"RPN format mismatch: C_cls={C_cls} vs num_anchors={num_anchors}")
            
            obj_logits = objectness_logits[level_idx][img_idx : img_idx + 1].view(num_anchors, 1, H, W)
            obj_logits = obj_logits.permute(2, 3, 0, 1).contiguous().view(-1, 1)
            scores = torch.sigmoid(obj_logits[:, 0])
            
            bbox_pred = bbox_regression[level_idx][img_idx : img_idx + 1].view(num_anchors, 4, H_reg, W_reg)
            bbox_pred = bbox_pred.permute(2, 3, 0, 1).contiguous().view(-1, 4)
            
            level_anchors = anchors[level_idx].to(device).detach()
            valid_mask = scores >= score_threshold
            if valid_mask.sum() > pre_nms_top_n:
                top_k = min(pre_nms_top_n * 2, len(scores))
                if deterministic:
                    scores_for_topk = scores
                else:
                    # Break ties randomly so we don't always select top-left anchors when objectness is ~uniform (early training)
                    scores_for_topk = scores + 1e-5 * (torch.rand_like(scores, device=scores.device) * 2 - 1)
                _, top_indices = torch.topk(scores_for_topk, top_k)
                valid_mask = torch.zeros_like(scores, dtype=torch.bool)
                valid_mask[top_indices] = True
                valid_mask = valid_mask & (scores >= score_threshold)
            
            if valid_mask.any():
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    decoded = decode_rpn_boxes_xyxy(
                        level_anchors[valid_indices],
                        bbox_pred[valid_indices].detach(),
                        target_means=target_means,
                        target_stds=target_stds,
                    ).detach()
                img_proposals.append(decoded)
                img_scores.append(scores[valid_indices])
        
        if not img_proposals:
            all_proposals_xyxy.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
            continue
        
        proposals = torch.cat(img_proposals, dim=0)
        scores = torch.cat(img_scores, dim=0)
        proposals = proposals.to(device)
        scores = scores.to(device)
        
        img_h, img_w = image_sizes[img_idx]
        proposals[:, 0] = torch.clamp(proposals[:, 0], min=0, max=img_w)
        proposals[:, 1] = torch.clamp(proposals[:, 1], min=0, max=img_h)
        proposals[:, 2] = torch.clamp(proposals[:, 2], min=0, max=img_w)
        proposals[:, 3] = torch.clamp(proposals[:, 3], min=0, max=img_h)
        valid_mask = (
            ((proposals[:, 2] - proposals[:, 0]) > min_size)
            & ((proposals[:, 3] - proposals[:, 1]) > min_size)
            & torch.isfinite(proposals).all(dim=1)
        )
        proposals = proposals[valid_mask]
        scores = scores[valid_mask]
        
        if len(proposals) == 0:
            all_proposals_xyxy.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
            continue
        
        if len(scores) > pre_nms_top_n:
            if deterministic:
                scores_for_topk = scores
            else:
                scores_for_topk = scores + 1e-5 * (torch.rand_like(scores, device=scores.device) * 2 - 1)
            top_indices = torch.topk(scores_for_topk, pre_nms_top_n).indices
            proposals = proposals[top_indices]
            scores = scores[top_indices]
        
        keep_indices = torchvision_nms(proposals, scores, nms_threshold)
        if post_nms_top_n is not None:
            keep_indices = keep_indices[:post_nms_top_n]
        proposals = proposals[keep_indices]
        
        if len(proposals) > post_nms_top_n:
            proposals = proposals[: post_nms_top_n]
        
        all_proposals_xyxy.append(proposals)
    
    return all_proposals_xyxy


def _cxcywh_to_xyxy(obb: torch.Tensor) -> torch.Tensor:
    """Convert boxes [N,5] (cx,cy,w,h,angle) to xyxy ignoring angle (assumes horizontal anchors)."""
    cx, cy, w, h = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_midpoint_rpn_loss(
    objectness_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    gt_boxes_ignore: Optional[List[torch.Tensor]] = None,
    *,
    positive_iou_threshold: float = 0.7,
    negative_iou_threshold: float = 0.3,
    box_reg_weight: float = 1.0,
    fg_bg_sampling_ratio: float = 0.5,
    batch_size_per_image: int = 256,
    use_hbb_for_matching: bool = True,
    min_pos_iou: float = 0.3,
    match_low_quality: bool = True,
    sample_from_all_levels: bool = True,
    beta: float = 1.0 / 9.0,
    target_stds: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
) -> Dict[str, torch.Tensor]:
    """Oriented R-CNN RPN loss: 6D MidpointOffsetCoder deltas supervised at the RPN stage."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation.")

    from .bbox_coder import MidpointOffsetCoder

    device = objectness_logits[0].device if objectness_logits else torch.device("cpu")
    num_images = len(gt_boxes)
    num_levels = len(objectness_logits)
    B = num_images
    coder = MidpointOffsetCoder(target_stds=target_stds)

    level_logits: List[torch.Tensor] = []
    level_pred: List[torch.Tensor] = []
    level_labels: List[torch.Tensor] = []
    level_targets: List[torch.Tensor] = []
    anchors_per_image_per_level: List[int] = []
    all_loss_objectness: List[torch.Tensor] = []
    all_loss_rpn_box_reg: List[torch.Tensor] = []

    for level_idx in range(num_levels):
        B0, C_cls, H, W = objectness_logits[level_idx].shape
        B_reg, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
        if C_reg % 6 != 0:
            raise RuntimeError(f"Invalid RPN format: C_reg={C_reg} must be divisible by 6")
        num_anchors = C_reg // 6
        if num_anchors != C_cls:
            raise RuntimeError(f"RPN format mismatch: C_cls={C_cls} vs num_anchors={num_anchors}")

        obj_logits = objectness_logits[level_idx].view(B0, num_anchors, 1, H, W)
        obj_logits = obj_logits.permute(0, 3, 4, 1, 2).contiguous().view(-1, 1)
        obj_logits = torch.cat([torch.zeros_like(obj_logits), obj_logits], dim=1)  # [N,2]

        bbox_pred = bbox_regression[level_idx].view(B_reg, num_anchors, 6, H_reg, W_reg)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous().view(-1, 6)

        with torch.no_grad():
            level_anchors_raw = anchors[level_idx]
            level_anchors_raw = level_anchors_raw.detach().to(device)
            expected = H * W * num_anchors
            if len(level_anchors_raw) != expected:
                raise RuntimeError(
                    f"Anchor count mismatch at level {level_idx}: got {len(level_anchors_raw)} expected {expected}"
                )
            level_anchors = level_anchors_raw.unsqueeze(0).repeat(B, 1, 1).view(-1, 5).clone().detach()
            level_anchors.requires_grad_(False)

        anchors_per_image = len(level_anchors) // B
        img_labels_list: List[torch.Tensor] = []
        img_targets_list: List[torch.Tensor] = []

        for img_idx in range(B):
            img_anchors = level_anchors[img_idx * anchors_per_image : (img_idx + 1) * anchors_per_image].detach()
            img_gt = gt_boxes[img_idx].to(device).detach()

            if img_gt.numel() == 0:
                labels = torch.zeros((len(img_anchors),), dtype=torch.int64, device=device)
                targets = torch.zeros((len(img_anchors), 6), dtype=torch.float32, device=device)
            else:
                img_gt_ignore = None
                if gt_boxes_ignore is not None and img_idx < len(gt_boxes_ignore):
                    img_gt_ignore = gt_boxes_ignore[img_idx].to(device).detach()
                labels, matched = match_oriented_anchors_to_gt(
                    img_anchors,
                    img_gt,
                    positive_iou_threshold,
                    negative_iou_threshold,
                    device,
                    use_hbb_for_matching=use_hbb_for_matching,
                    min_pos_iou=min_pos_iou,
                    match_low_quality=match_low_quality,
                    gt_boxes_ignore=img_gt_ignore,
                    ignore_iou_threshold=positive_iou_threshold,
                )
                pos = labels == 1
                targets = torch.zeros((len(img_anchors), 6), dtype=torch.float32, device=device)
                if pos.any():
                    matched_gt = img_gt[matched[pos]]
                    rois_xyxy = _cxcywh_to_xyxy(img_anchors[pos])
                    targets[pos] = coder.encode(rois_xyxy, matched_gt)

            img_labels_list.append(labels)
            img_targets_list.append(targets)

        labels_level = torch.cat(img_labels_list, dim=0)
        targets_level = torch.cat(img_targets_list, dim=0)
        if obj_logits.shape[0] != labels_level.shape[0]:
            raise RuntimeError(f"Shape mismatch in RPN loss at level {level_idx}")

        level_logits.append(obj_logits)
        level_pred.append(bbox_pred)
        level_labels.append(labels_level)
        level_targets.append(targets_level)
        anchors_per_image_per_level.append(anchors_per_image)

    if sample_from_all_levels:
        for img_idx in range(B):
            logits_i = torch.cat(
                [level_logits[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]] for l in range(num_levels)],
                dim=0,
            )
            pred_i = torch.cat(
                [level_pred[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]] for l in range(num_levels)],
                dim=0,
            )
            labels_i = torch.cat(
                [level_labels[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]] for l in range(num_levels)],
                dim=0,
            )
            targets_i = torch.cat(
                [level_targets[l][img_idx * anchors_per_image_per_level[l] : (img_idx + 1) * anchors_per_image_per_level[l]] for l in range(num_levels)],
                dim=0,
            )
            fg_i = (labels_i == 1).nonzero(as_tuple=True)[0]
            bg_i = (labels_i == 0).nonzero(as_tuple=True)[0]
            num_fg = min(len(fg_i), int(batch_size_per_image * fg_bg_sampling_ratio))
            num_bg = min(len(bg_i), batch_size_per_image - num_fg)
            sampled_fg = fg_i[torch.randperm(len(fg_i), device=device)[:num_fg]] if num_fg > 0 else torch.zeros((0,), dtype=torch.long, device=device)
            sampled_bg = bg_i[torch.randperm(len(bg_i), device=device)[:num_bg]] if num_bg > 0 else torch.zeros((0,), dtype=torch.long, device=device)
            sampled = torch.cat([sampled_fg, sampled_bg], dim=0)
            if sampled.numel() > 0:
                valid = labels_i[sampled] >= 0
                if valid.any():
                    loss_obj_i = F.cross_entropy(logits_i[sampled][valid].clone(), labels_i[sampled][valid].long(), reduction="mean")
                else:
                    loss_obj_i = (logits_i[0:1] * 0.0).sum()
                if sampled_fg.numel() > 0:
                    loss_reg_i = F.smooth_l1_loss(
                        pred_i[sampled_fg].clone(),
                        targets_i[sampled_fg],
                        beta=beta,
                        reduction="sum",
                    ) / float(max(1, len(sampled)))
                else:
                    loss_reg_i = (pred_i[0:1] * 0.0).sum()
            else:
                loss_obj_i = (logits_i[0:1] * 0.0).sum()
                loss_reg_i = (pred_i[0:1] * 0.0).sum()
            all_loss_objectness.append(loss_obj_i)
            all_loss_rpn_box_reg.append(loss_reg_i)
    else:  # pragma: no cover (keep parity with older API)
        raise RuntimeError("compute_midpoint_rpn_loss requires sample_from_all_levels=True")

    loss_objectness = torch.stack(all_loss_objectness).mean() if all_loss_objectness else (objectness_logits[0] * 0.0).sum()
    loss_rpn_box_reg = torch.stack(all_loss_rpn_box_reg).mean() if all_loss_rpn_box_reg else (bbox_regression[0] * 0.0).sum()
    loss_objectness = torch.where(torch.isfinite(loss_objectness), loss_objectness, torch.zeros_like(loss_objectness))
    loss_rpn_box_reg = torch.where(torch.isfinite(loss_rpn_box_reg), loss_rpn_box_reg, torch.zeros_like(loss_rpn_box_reg))
    return {"loss_objectness": loss_objectness, "loss_rpn_box_reg": box_reg_weight * loss_rpn_box_reg}


def generate_midpoint_proposals(
    objectness_logits: List[torch.Tensor],
    bbox_regression: List[torch.Tensor],
    anchors: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    *,
    score_threshold: float = 0.0,
    nms_threshold: float = 0.8,
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 2000,
    min_size: float = 0.0,
    target_stds: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
    deterministic: bool = False,
) -> List[torch.Tensor]:
    """Generate oriented proposals from 6D midpoint RPN deltas, then prune with HBB NMS.

    Args:
        deterministic: If True, use score-only top-k (no random tie-break) for reproducible export.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for proposal generation.")
    from .bbox_coder import MidpointOffsetCoder

    device = objectness_logits[0].device if objectness_logits else torch.device("cpu")
    coder = MidpointOffsetCoder(target_stds=target_stds)
    num_images = len(image_sizes)
    out: List[torch.Tensor] = []

    for img_idx in range(num_images):
        img_props: List[torch.Tensor] = []
        img_scores: List[torch.Tensor] = []
        for level_idx in range(len(objectness_logits)):
            B, C_cls, H, W = objectness_logits[level_idx].shape
            _, C_reg, H_reg, W_reg = bbox_regression[level_idx].shape
            if C_reg % 6 != 0:
                raise RuntimeError(f"Invalid midpoint RPN format: C_reg={C_reg}")
            num_anchors = C_reg // 6
            if num_anchors != C_cls:
                raise RuntimeError("Midpoint RPN format mismatch")

            obj = objectness_logits[level_idx][img_idx : img_idx + 1].view(num_anchors, 1, H, W)
            obj = obj.permute(2, 3, 0, 1).contiguous().view(-1, 1)
            scores = torch.sigmoid(obj[:, 0])

            pred = bbox_regression[level_idx][img_idx : img_idx + 1].view(num_anchors, 6, H_reg, W_reg)
            pred = pred.permute(2, 3, 0, 1).contiguous().view(-1, 6)

            level_anchors = anchors[level_idx].to(device).detach()
            valid = scores >= score_threshold
            if valid.sum() > pre_nms_top_n:
                top_k = min(pre_nms_top_n * 2, int(scores.numel()))
                if deterministic:
                    scores_for_topk = scores
                else:
                    scores_for_topk = scores + 1e-5 * (torch.rand_like(scores) * 2 - 1)
                _, top_idx = torch.topk(scores_for_topk, top_k)
                mask = torch.zeros_like(valid)
                mask[top_idx] = True
                valid = mask & (scores >= score_threshold)
            if not valid.any():
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                rois_xyxy = _cxcywh_to_xyxy(level_anchors[idx])
                decoded = coder.decode(rois_xyxy, pred[idx].detach()).detach()
            img_props.append(decoded)
            img_scores.append(scores[idx])

        if not img_props:
            out.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
            continue

        props = torch.cat(img_props, dim=0)
        scores = torch.cat(img_scores, dim=0)
        img_h, img_w = image_sizes[img_idx]
        props[:, 0] = props[:, 0].clamp(min=0, max=img_w)
        props[:, 1] = props[:, 1].clamp(min=0, max=img_h)
        max_side = float(max(img_h, img_w)) * 2.0
        props[:, 2] = props[:, 2].clamp(min=min_size, max=max_side)
        props[:, 3] = props[:, 3].clamp(min=min_size, max=max_side)
        valid2 = (props[:, 2] > min_size) & (props[:, 3] > min_size) & torch.isfinite(props).all(dim=1)
        props = props[valid2]
        scores = scores[valid2]
        if props.numel() == 0:
            out.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
            continue

        if scores.numel() > pre_nms_top_n:
            if deterministic:
                scores_for_topk = scores
            else:
                scores_for_topk = scores + 1e-5 * (torch.rand_like(scores) * 2 - 1)
            top_idx = torch.topk(scores_for_topk, pre_nms_top_n).indices
            props = props[top_idx]
            scores = scores[top_idx]

        keep = hbb_nms_for_oriented_boxes_gpu(props, scores, iou_threshold=nms_threshold, max_detections=post_nms_top_n)
        props = props[keep]
        if props.shape[0] > post_nms_top_n:
            props = props[:post_nms_top_n]
        out.append(props)

    return out


__all__ = [
    "OrientedRPNHead",
    "generate_oriented_anchors",
    "encode_rpn_boxes",
    "decode_rpn_boxes",
    "decode_rpn_boxes_xyxy",
    "encode_oriented_boxes",
    "decode_oriented_boxes",
    "match_oriented_anchors_to_gt",
    "compute_oriented_rpn_loss",
    "generate_oriented_proposals",
    "generate_horizontal_proposals",
    "compute_midpoint_rpn_loss",
    "generate_midpoint_proposals",
]

