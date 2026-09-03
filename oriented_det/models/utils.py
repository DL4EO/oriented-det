"""Shared utilities for oriented object detection models.

This module provides common helper functions and mixins used across
different model implementations to reduce code duplication.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from ..geometry import RBox
from .backbones import build_resnet_fpn_backbone


# ============================================================================
# RBox Conversion Utilities
# ============================================================================

def rboxes_to_tensor(
    rboxes: Sequence[RBox], 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert RBoxes to tensor format [N, 5] with [cx, cy, w, h, angle].
    
    Args:
        rboxes: Sequence of RBox objects
        device: Optional device to create tensor on
        
    Returns:
        Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    
    if not rboxes:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)
    
    data = []
    for rbox in rboxes:
        data.append([rbox.cx, rbox.cy, rbox.width, rbox.height, rbox.angle])
    
    return torch.tensor(data, dtype=torch.float32, device=device)


def tensor_to_rboxes(boxes: torch.Tensor) -> List[RBox]:
    """Convert tensor [N, 5] with [cx, cy, w, h, angle] to RBoxes.
    
    Args:
        boxes: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
        
    Returns:
        List of RBox objects
    """
    rboxes = []
    for box in boxes.detach().cpu():
        cx, cy, w, h, angle = box.tolist()
        # Validate coordinates are reasonable (not NaN/inf and within expected range)
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and 
                math.isfinite(h) and math.isfinite(angle)):
            continue  # Skip invalid boxes
        rboxes.append(RBox(cx, cy, w, h, angle))
    return rboxes


# ============================================================================
# Target Preparation Utility
# ============================================================================

def prepare_targets(
    targets: Sequence[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Prepare targets for training, preserving oriented boxes.
    
    Args:
        targets: List of target dicts, each containing:
            - "rboxes" (List[RBox] or tensor [N, 5] with format [cx, cy, w, h, angle])
            - "labels" (tensor [N] or list, 1-indexed)
            - optional "rboxes_ignore" / "rboxes_lookalike"
        device: Optional device to move tensors to
        
    Returns:
        Tuple of (gt_boxes_list, gt_labels_list, gt_boxes_ignore_list, gt_boxes_lookalike_list):
        - gt_boxes_list: List of tensors [M, 5] with oriented boxes
        - gt_labels_list: List of tensors [M] with class labels (1-indexed)
        - gt_boxes_ignore_list: List of tensors [I, 5] with ignored GT boxes (e.g. DOTA difficult)
        - gt_boxes_lookalike_list: List of tensors [L, 5] hard-negative lookalike boxes
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    
    gt_boxes_list = []
    gt_labels_list = []
    gt_boxes_ignore_list: List[torch.Tensor] = []
    gt_boxes_lookalike_list: List[torch.Tensor] = []
    
    for target in targets:
        target = dict(target)
        
        # Get oriented boxes
        if "rboxes" in target:
            rboxes = target["rboxes"]
            if isinstance(rboxes, torch.Tensor):
                gt_boxes = rboxes
                if device is not None:
                    gt_boxes = gt_boxes.to(device)
            else:
                # rboxes is a list of RBox objects
                gt_boxes = rboxes_to_tensor(rboxes, device=device)
        else:
            raise ValueError("Targets must include 'rboxes'.")
        
        # Get labels
        labels = target.get("labels")
        if labels is None:
            raise ValueError("Targets must include 'labels'.")
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.int64, device=device)
        elif device is not None:
            labels = labels.to(device)
        
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(labels)

        # Optional ignore GTs (MMDet/MMRotate style): used to mark anchors/proposals as ignore
        # when they overlap with difficult / don't-care regions.
        if "rboxes_ignore" in target:
            rboxes_ign = target["rboxes_ignore"]
            if isinstance(rboxes_ign, torch.Tensor):
                gt_ign = rboxes_ign.to(device) if device is not None else rboxes_ign
            else:
                gt_ign = rboxes_to_tensor(rboxes_ign, device=device)
        else:
            gt_ign = torch.zeros((0, 5), dtype=torch.float32, device=device)
        gt_boxes_ignore_list.append(gt_ign)

        # Optional lookalike hard-negatives: force overlapping non-positives to background.
        if "rboxes_lookalike" in target:
            rboxes_look = target["rboxes_lookalike"]
            if isinstance(rboxes_look, torch.Tensor):
                gt_look = rboxes_look.to(device) if device is not None else rboxes_look
            else:
                gt_look = rboxes_to_tensor(rboxes_look, device=device)
        else:
            gt_look = torch.zeros((0, 5), dtype=torch.float32, device=device)
        gt_boxes_lookalike_list.append(gt_look)
    
    return gt_boxes_list, gt_labels_list, gt_boxes_ignore_list, gt_boxes_lookalike_list


def max_iou_vs_boxes(
    boxes: "torch.Tensor",
    gt_boxes: "torch.Tensor",
    *,
    use_hbb_for_matching: bool = False,
) -> "torch.Tensor":
    """Max IoU of each box in ``boxes`` against any box in ``gt_boxes`` ([N])."""
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    n = boxes.shape[0]
    device = boxes.device
    if gt_boxes is None or gt_boxes.numel() == 0 or n == 0:
        return torch.zeros((n,), dtype=torch.float32, device=device)
    boxes = boxes.to(device).detach()
    gt_boxes = gt_boxes.to(device).detach()
    try:
        from ..ops.gpu_ops import oriented_box_hbb_iou_gpu
        from ..ops.rotated_ops import pairwise_rotated_iou

        if use_hbb_for_matching:
            iou = oriented_box_hbb_iou_gpu(boxes, gt_boxes)
        else:
            iou = pairwise_rotated_iou(boxes, gt_boxes)
    except Exception:
        from ..ops.iou import batch_rbox_iou

        a_list = [RBox(*a.tolist()) for a in boxes.detach().cpu()]
        g_list = [RBox(*g.tolist()) for g in gt_boxes.detach().cpu()]
        iou = torch.tensor(batch_rbox_iou(a_list, g_list, device=device), device=device)
    if iou.numel() == 0:
        return torch.zeros((n,), dtype=torch.float32, device=device)
    return iou.max(dim=1).values


def force_lookalike_to_background(
    labels: "torch.Tensor",
    boxes: "torch.Tensor",
    gt_boxes_lookalike: Optional["torch.Tensor"],
    *,
    iou_threshold: float,
    use_hbb_for_matching: bool = False,
    matched_gt_indices: Optional["torch.Tensor"] = None,
    matched_gt_boxes: Optional["torch.Tensor"] = None,
    positive_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Force non-positive boxes overlapping lookalikes to background label ``0``.

    Applied after normal assignment and ignore regions so lookalike wins over
    don't-care (``-1``). Never overrides real-object positives.

    Args:
        labels: Per-sample labels (RPN: -1/0/1; ROI: -1/0/1..K). Modified in place.
        boxes: Anchors or proposals [N, 5].
        gt_boxes_lookalike: Lookalike OBBs [L, 5] or empty/None.
        iou_threshold: IoU threshold (typically the positive assign threshold).
        use_hbb_for_matching: Use HBB IoU instead of rotated IoU.
        matched_gt_indices: Optional matched GT index tensor to clear for forced bg.
        matched_gt_boxes: Optional matched GT boxes tensor to zero for forced bg.
        positive_mask: Optional mask of positives to protect. Default: ``labels > 0``.

    Returns:
        Boolean mask of samples forced to background (for preferential sampling).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    n = labels.shape[0]
    device = labels.device
    forced = torch.zeros((n,), dtype=torch.bool, device=device)
    if gt_boxes_lookalike is None or gt_boxes_lookalike.numel() == 0 or n == 0:
        return forced
    if positive_mask is None:
        positive_mask = labels > 0
    max_iou = max_iou_vs_boxes(
        boxes, gt_boxes_lookalike, use_hbb_for_matching=use_hbb_for_matching
    )
    look_mask = (~positive_mask) & (max_iou >= float(iou_threshold))
    if not look_mask.any():
        return forced
    labels[look_mask] = 0
    if matched_gt_indices is not None:
        matched_gt_indices[look_mask] = -1
    if matched_gt_boxes is not None:
        matched_gt_boxes[look_mask] = 0.0
    forced[look_mask] = True
    return forced


def sample_fg_bg_indices(
    labels: "torch.Tensor",
    *,
    num_fg: int,
    num_bg: int,
    device: "torch.device",
    fg_selector=None,
    bg_selector=None,
    boxes: Optional["torch.Tensor"] = None,
    gt_boxes_lookalike: Optional["torch.Tensor"] = None,
    lookalike_iou_threshold: Optional[float] = None,
    use_hbb_for_matching: bool = False,
    prefer_lookalike_bg_fraction: float = 0.5,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Sample foreground/background indices with optional lookalike-negative preference.

    When ``gt_boxes_lookalike`` is set, up to ``prefer_lookalike_bg_fraction`` of the
    background quota is filled first from bg samples overlapping lookalikes.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    if fg_selector is None:
        fg_selector = labels == 1
    if bg_selector is None:
        bg_selector = labels == 0
    fg_indices = fg_selector.nonzero(as_tuple=True)[0]
    bg_indices = bg_selector.nonzero(as_tuple=True)[0]
    num_fg = min(int(num_fg), len(fg_indices))
    num_bg = min(int(num_bg), len(bg_indices))
    if num_fg > 0:
        sampled_fg = fg_indices[torch.randperm(len(fg_indices), device=device)[:num_fg]]
    else:
        sampled_fg = torch.tensor([], dtype=torch.int64, device=device)

    if num_bg <= 0:
        sampled_bg = torch.tensor([], dtype=torch.int64, device=device)
        return sampled_fg, sampled_bg

    lookalike_bg = torch.tensor([], dtype=torch.int64, device=device)
    other_bg = bg_indices
    if (
        boxes is not None
        and gt_boxes_lookalike is not None
        and gt_boxes_lookalike.numel() > 0
        and len(bg_indices) > 0
    ):
        thr = float(lookalike_iou_threshold) if lookalike_iou_threshold is not None else 0.5
        max_iou = max_iou_vs_boxes(
            boxes[bg_indices],
            gt_boxes_lookalike,
            use_hbb_for_matching=use_hbb_for_matching,
        )
        look_mask = max_iou >= thr
        lookalike_bg = bg_indices[look_mask]
        other_bg = bg_indices[~look_mask]

    prefer_n = min(len(lookalike_bg), max(0, int(num_bg * float(prefer_lookalike_bg_fraction))))
    if len(other_bg) == 0:
        prefer_n = min(len(lookalike_bg), num_bg)
    rest_n = num_bg - prefer_n
    parts = []
    if prefer_n > 0:
        parts.append(lookalike_bg[torch.randperm(len(lookalike_bg), device=device)[:prefer_n]])
    if rest_n > 0 and len(other_bg) > 0:
        take = min(rest_n, len(other_bg))
        parts.append(other_bg[torch.randperm(len(other_bg), device=device)[:take]])
        rest_n -= take
    # If ordinary bg was short, fill remaining from leftover lookalikes.
    if rest_n > 0 and len(lookalike_bg) > prefer_n:
        leftover = lookalike_bg[torch.randperm(len(lookalike_bg), device=device)]
        already = set(int(x) for x in parts[0].tolist()) if parts else set()
        fill = [int(i) for i in leftover.tolist() if int(i) not in already][:rest_n]
        if fill:
            parts.append(torch.tensor(fill, dtype=torch.int64, device=device))
    if parts:
        sampled_bg = torch.cat(parts, dim=0)
    else:
        sampled_bg = bg_indices[torch.randperm(len(bg_indices), device=device)[:num_bg]]
    return sampled_fg, sampled_bg


# ============================================================================
# Backbone Setup Utility
# ============================================================================

def setup_backbone(
    backbone: Optional[nn.Module],
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = False,
    trainable_layers: int = 5,
    returned_layers: Optional[List[int]] = None,
    use_p6p7_extra_levels: bool = False,
) -> Tuple[nn.Module, int]:
    """Setup backbone and return backbone module and output channels.

    Args:
        backbone: Optional backbone module (if None, creates ResNet+FPN)
        backbone_name: Name of backbone to create ("resnet18", "resnet50", etc.)
        pretrained_backbone: Whether to use pretrained backbone weights
        trainable_layers: Number of backbone layers to keep trainable
        returned_layers: ResNet stages for FPN (e.g. [2,3,4] for MMRotate C3–C5 only)
        use_p6p7_extra_levels: RetinaNet-style P6/P7 convs on C5 (MMRotate on_input).

    Returns:
        Tuple of (backbone, output_channels):
        - backbone: The backbone module
        - output_channels: Number of output channels (typically 256 for FPN)
    """
    if nn is None:
        raise RuntimeError("PyTorch is required.")

    if backbone is None:
        backbone = build_resnet_fpn_backbone(
            backbone_name,
            pretrained=pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
            use_p6p7_extra_levels=use_p6p7_extra_levels,
        )
    
    # Get backbone output channels (typically 256 for FPN)
    if hasattr(backbone, 'out_channels'):
        backbone_channels = backbone.out_channels
    else:
        # Try to infer from FPN if available
        if hasattr(backbone, 'fpn') and hasattr(backbone.fpn, 'out_channels'):
            backbone_channels = backbone.fpn.out_channels
        else:
            # Default for FPN
            backbone_channels = 256
    
    return backbone, backbone_channels


# ============================================================================
# Feature Extraction Utility
# ============================================================================

def _ordered_fpn_feature_keys(keys: Sequence[str]) -> List[str]:
    """Stable FPN dict key order: numeric levels, fpn* keys, pN extras, then pool."""
    key_list = list(keys)
    numeric = sorted((k for k in key_list if k.isdigit()), key=int)
    fpn_named = sorted(k for k in key_list if k.startswith("fpn"))
    p_levels = sorted(
        (k for k in key_list if len(k) > 1 and k[0] == "p" and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
    ordered = numeric + fpn_named + p_levels
    if "pool" in key_list:
        ordered.append("pool")
    return ordered


def _is_fpn_feature_key(key: str) -> bool:
    """True for torchvision FPN outputs we should feed to detection heads."""
    if key.isdigit() or key.startswith("fpn"):
        return True
    return len(key) > 1 and key[0] == "p" and key[1:].isdigit()


def extract_backbone_features(
    backbone: nn.Module,
    images: Sequence[torch.Tensor],
    use_checkpoint: bool = False,
    training: bool = False,
    include_pool_level: bool = False,
) -> List[torch.Tensor]:
    """Extract features from backbone and convert to list format.
    
    Handles:
    - Stacking images into batch
    - Optional gradient checkpointing
    - Converting OrderedDict/dict outputs to list format
    
    Args:
        backbone: Backbone module
        images: Sequence of image tensors (C, H, W) in [0, 1] range
        use_checkpoint: Whether to use gradient checkpointing
        training: Whether model is in training mode
        include_pool_level: If True, also keep torchvision FPN's ``LastLevelMaxPool``
            output (dict key ``"pool"``) as the last pyramid level. RetinaNet needs it
            (P6, stride 2x the last conv level); the two-stage models ignore it.
        
    Returns:
        List of feature maps from FPN, each of shape [B, C, H, W]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    
    # Stack images into batch
    images_tensor = torch.stack(images, dim=0)  # [B, C, H, W]
    
    # Extract features using backbone
    # OPTIONAL: Use gradient checkpointing to reduce memory usage
    # This trades computation (recomputes activations) for memory (2-3x reduction)
    if use_checkpoint and training:
        try:
            from torch.utils.checkpoint import checkpoint
            # Wrap backbone forward in checkpoint - will recompute during backward
            features = checkpoint(backbone, images_tensor, use_reentrant=False)
        except Exception:
            # Fallback if checkpointing fails
            features = backbone(images_tensor)
    else:
        features = backbone(images_tensor)
    
    # Convert OrderedDict/dict to list (FPN outputs)
    if isinstance(features, dict):
        # FPN returns OrderedDict with keys like "0", "1", "2", "p6", "p7", or "pool".
        feature_list = []
        for key in _ordered_fpn_feature_keys(features.keys()):
            if key == "pool":
                if include_pool_level:
                    feature_list.append(features[key])
                continue
            if _is_fpn_feature_key(key):
                feature_list.append(features[key])
        # If no recognized keys, just use all values
        if not feature_list:
            feature_list = list(features.values())
    else:
        # Single feature map
        feature_list = [features]
    
    return feature_list


def derive_fpn_strides_from_grid(
    image_size: Tuple[int, int],
    feature_map_sizes: Sequence[Tuple[int, int]],
    configured: Optional[Sequence[int]] = None,
) -> List[int]:
    """Compute FPN strides from the real input size and backbone feature map shapes.

    For each level, stride is ``image_size / feature_map_size`` when H and W agree
    on a positive integer. Otherwise, if ``configured`` is set (MMRotate
    ``[4, 8, 16, 32, 64]``), use that level's nominal stride. That is required for
    ``Pad(size_divisor=32)`` + torchvision P6: ``max_pool(kernel=1, stride=2)`` on
    a map whose size is odd (e.g. 800→25→13) so H/W no longer share one stride.

    Args:
        image_size: ``(H, W)`` of the image tensor fed to the backbone (same as anchor generation).
        feature_map_sizes: ``(h, w)`` per FPN level, from ``feature.shape[2:]``.
        configured: Optional nominal strides (same length as ``feature_map_sizes``).
            Used when the grid does not imply a unique integer isotropic stride.

    Returns:
        One integer stride per level.

    Raises:
        ValueError: If sizes are invalid, or H/W imply different strides and no
            configured fallback is available.
    """
    img_h, img_w = int(image_size[0]), int(image_size[1])
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    cfg = [int(x) for x in configured] if configured else None
    strides: List[int] = []
    for i, (fh, fw) in enumerate(feature_map_sizes):
        fh, fw = int(fh), int(fw)
        if fh <= 0 or fw <= 0:
            raise ValueError(f"feature_map_sizes entries must be positive, got {(fh, fw)}")
        sh = img_h / fh
        sw = img_w / fw
        tol = max(1e-2, 1e-3 * max(sh, sw))
        isotropic = abs(sh - sw) <= tol
        mid = (sh + sw) / 2.0
        s_round = int(round(mid))
        clean = isotropic and s_round > 0 and abs(mid - s_round) <= tol
        if clean:
            strides.append(s_round)
            continue
        if cfg is not None and i < len(cfg) and cfg[i] > 0:
            strides.append(cfg[i])
            continue
        if not isotropic:
            raise ValueError(
                f"Anisotropic FPN stride for image {img_h}x{img_w} and feature map {fh}x{fw}: "
                f"stride_h={sh:.6f}, stride_w={sw:.6f}"
            )
        if s_round <= 0:
            raise ValueError(f"Non-positive derived stride {s_round} for image {image_size}, feature {(fh, fw)}")
        strides.append(s_round)
    return strides


def warn_if_fpn_strides_mismatch(
    configured: Optional[Sequence[int]],
    derived: Sequence[int],
) -> None:
    """Emit a single warning if config strides disagree with grid-derived strides."""
    if not configured:
        return
    cfg_full = [int(x) for x in configured]
    der = [int(x) for x in derived]
    if len(cfg_full) != len(der):
        warnings.warn(
            f"fpn_strides from config has {len(cfg_full)} levels {cfg_full} but the model "
            f"produced {len(der)} feature levels (derived strides {der}); using derived strides. "
            "Check FPN level configuration (returned_layers / fpn_extra_level).",
            UserWarning,
            stacklevel=3,
        )
        return
    if cfg_full == der:
        return
    warnings.warn(
        f"fpn_strides from config {cfg_full} does not match grid-derived strides {der}; "
        "using derived strides. Fix fpn_strides or training image size to remove this warning.",
        UserWarning,
        stacklevel=3,
    )


# ============================================================================
# Anchor Setup Utility
# ============================================================================

def setup_anchors(
    anchor_scales: Optional[List[float]],
    anchor_ratios: Optional[List[float]],
    anchor_angles: Optional[List[float]],
    default_angles: List[float],
    octave_base_scale: Optional[float] = None,
    scales_per_octave: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float], int]:
    """Setup anchor configuration and return scales, ratios, angles, and num_anchors.
    
    Args:
        anchor_scales: Optional list of anchor scales (default: [8])
        anchor_ratios: Optional list of anchor ratios (default: [0.5, 1.0, 2.0])
        anchor_angles: Optional list of anchor angles (default: provided)
        default_angles: Default angles to use if anchor_angles is None
        octave_base_scale: MMRotate RetinaNet ``octave_base_scale`` (with ``scales_per_octave``)
        scales_per_octave: MMRotate RetinaNet ``scales_per_octave`` (e.g. 3)
        
    Returns:
        Tuple of (scales, ratios, angles, num_anchors):
        - scales: List of anchor scales
        - ratios: List of anchor ratios
        - angles: List of anchor angles
        - num_anchors: Total number of anchors per location
    """
    scales = anchor_scales or [8]  # MMRotate standard: single scale
    ratios = anchor_ratios or [0.5, 1.0, 2.0]  # MMRotate standard
    # Handle empty list case: use default_angles if anchor_angles is None or empty
    if anchor_angles is None or (isinstance(anchor_angles, list) and len(anchor_angles) == 0):
        angles = default_angles
    else:
        angles = anchor_angles
    per_loc = len(ratios) * len(angles)
    if octave_base_scale is not None and scales_per_octave is not None:
        num_anchors = per_loc * int(scales_per_octave)
    else:
        num_anchors = per_loc
    
    return scales, ratios, angles, num_anchors


# ============================================================================
# Class Weights Mixin
# ============================================================================

class ClassWeightsMixin:
    """Mixin class for models that support class weights.
    
    Provides common functionality for managing class weights in two-stage
    detectors (RotatedFasterRCNN, OrientedRCNN).
    """
    
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
    ):
        """Initialize class weights mixin.
        
        Args:
            num_classes: Number of object classes (excluding background)
            class_weights: Optional class weights:
                - Dict[str, float]: Mapping from class name to weight
                - torch.Tensor: Tensor of shape [num_classes + 1] (including background)
                - None: Equal weights for all classes
        """
        self.num_classes = num_classes
        self._roi_class_weights = class_weights  # Store raw input
        self.roi_class_weights_tensor: Optional[torch.Tensor] = None  # Will be set when class mapping is known
        
        # Validate tensor if provided directly
        if class_weights is not None and isinstance(class_weights, torch.Tensor):
            if class_weights.shape[0] != num_classes + 1:
                raise ValueError(
                    f"Class weights tensor must have shape [num_classes + 1] = [{num_classes + 1}], "
                    f"but got shape {class_weights.shape}"
                )
            self.roi_class_weights_tensor = class_weights
    
    def set_class_weights(
        self,
        class_map: Dict[str, int],
        device: Optional[torch.device] = None,
    ) -> None:
        """Set class weights tensor from class mapping.
        
        Converts class weights dictionary (class_name -> weight) to tensor format
        using the provided class mapping. Must be called before training if using
        class weights with a dictionary.
        
        Args:
            class_map: Mapping from class names to class IDs (1-indexed, 0 is background)
            device: Device to create tensor on (defaults to model's device)
        
        Example:
            >>> class_map = {"plane": 1, "ship": 2, "small-vehicle": 3}
            >>> class_weights = {"plane": 2.0, "ship": 1.5, "small-vehicle": 1.0}
            >>> model = OrientedRCNN(num_classes=3, roi_class_weights=class_weights)
            >>> model.set_class_weights(class_map)
        """
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        
        if self._roi_class_weights is None:
            return
        
        if not isinstance(self._roi_class_weights, dict):
            return  # Already a tensor or None
        
        if device is None:
            # Try to get device from model parameters
            if hasattr(self, 'parameters'):
                device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            else:
                device = torch.device("cpu")
        
        # Create weights tensor: [num_classes + 1] (background + object classes)
        weights = torch.ones(self.num_classes + 1, dtype=torch.float32, device=device)
        
        # Set background weight (index 0) - default to 1.0 if not specified
        if "background" in self._roi_class_weights:
            weights[0] = self._roi_class_weights["background"]
        elif "__background__" in self._roi_class_weights:
            weights[0] = self._roi_class_weights["__background__"]
        
        # Set class weights (indices 1, 2, ..., num_classes)
        for class_name, class_id in class_map.items():
            if class_id < 1 or class_id > self.num_classes:
                continue  # Skip invalid class IDs
            if class_name in self._roi_class_weights:
                weights[class_id] = self._roi_class_weights[class_name]
        
        self.roi_class_weights_tensor = weights

    def set_class_weights_tensor(self, weights: "torch.Tensor") -> None:
        """Set ROI class weights directly as a tensor.

        Args:
            weights: Tensor of shape [num_classes + 1] (including background).
        """
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if weights.ndim != 1 or weights.shape[0] != self.num_classes + 1:
            raise ValueError(
                f"weights must have shape [{self.num_classes + 1}], got {tuple(weights.shape)}"
            )
        self.roi_class_weights_tensor = weights
    
    @property
    def roi_class_weights(self) -> Optional[torch.Tensor]:
        """Get the current class weights tensor.
        
        Returns the class weights tensor if it has been set (via `set_class_weights` or
        passed directly as a tensor during initialization). Returns None if no class
        weights are configured.
        
        Returns:
            Class weights tensor of shape [num_classes + 1] (including background),
            or None if not configured.
        
        Example:
            >>> model = OrientedRCNN(num_classes=15)
            >>> weights = model.roi_class_weights  # Returns None if not set
            >>> model.set_class_weights(class_map)
            >>> weights = model.roi_class_weights  # Returns tensor after setting
        """
        return self.roi_class_weights_tensor


class SigmoidFocalClassWeightsMixin:
    """Class weights for one-stage sigmoid focal loss (FCOS / RetinaNet).

    The stored tensor is ``[num_classes]``, aligned with one-hot columns / FCOS
    labels ``0 .. C-1``. There is no background class; ``background`` keys in a
    weights dict are ignored. ``set_class_weights_tensor`` also accepts the
    ``[C+1]`` ROI schedule tensor and drops index 0 so the existing epoch ramp
    works unchanged.
    """

    def _init_sigmoid_focal_class_weights(
        self,
        class_weights: Optional[Union[Dict[str, float], "torch.Tensor"]] = None,
    ) -> None:
        self._roi_class_weights = class_weights
        self.roi_class_weights_tensor: Optional["torch.Tensor"] = None
        if class_weights is not None and torch is not None and isinstance(class_weights, torch.Tensor):
            self.set_class_weights_tensor(class_weights)

    def set_class_weights(
        self,
        class_map: Dict[str, int],
        device: Optional["torch.device"] = None,
    ) -> None:
        """Map a name→weight dict onto FCOS/RetinaNet columns ``0 .. C-1``."""
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        if self._roi_class_weights is None:
            return
        if not isinstance(self._roi_class_weights, dict):
            return
        if device is None:
            if hasattr(self, "parameters"):
                params = list(self.parameters())
                device = params[0].device if params else torch.device("cpu")
            else:
                device = torch.device("cpu")
        num_classes = int(self.num_classes)
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        for class_name, class_id in class_map.items():
            if class_name in ("background", "__background__"):
                continue
            cid = int(class_id)
            if cid < 1 or cid > num_classes:
                continue
            if class_name in self._roi_class_weights:
                weights[cid - 1] = float(self._roi_class_weights[class_name])
        self.roi_class_weights_tensor = weights

    def set_class_weights_tensor(self, weights: "torch.Tensor") -> None:
        """Set per-class focal weights.

        Accepts ``[num_classes]`` (one-hot columns) or ``[num_classes + 1]``
        (ROI schedule tensor; background at index 0 is dropped).
        """
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        num_classes = int(self.num_classes)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {tuple(weights.shape)}")
        if weights.shape[0] == num_classes + 1:
            weights = weights[1:]
        elif weights.shape[0] != num_classes:
            raise ValueError(
                f"weights must have shape [{num_classes}] or [{num_classes + 1}], "
                f"got {tuple(weights.shape)}"
            )
        self.roi_class_weights_tensor = weights

    @property
    def roi_class_weights(self) -> Optional["torch.Tensor"]:
        """Per-class focal weights ``[num_classes]``, or None if unweighted."""
        return self.roi_class_weights_tensor


class GroupedCeMixin:
    """Optional coarse-to-fine ROI classification curriculum (grouped cross-entropy)."""

    def _init_grouped_ce(self) -> None:
        self._roi_grouped_ce_enabled = False
        self.roi_grouped_ce_alpha: float = 0.0
        self._roi_grouped_ce_group_index_lists: List[List[int]] = []
        self._roi_grouped_ce_class_in_group_id: Optional["torch.Tensor"] = None
        self._roi_grouped_ce_schedule_type: Optional[str] = None
        self._roi_grouped_ce_schedule_start_epoch: int = 0
        self._roi_grouped_ce_schedule_end_epoch: int = 0
        self._roi_grouped_ce_schedule_power: float = 1.0

    def clear_roi_grouped_ce(self) -> None:
        self._init_grouped_ce()

    def set_roi_grouped_ce(
        self,
        *,
        group_index_lists: List[List[int]],
        class_in_group_id: "torch.Tensor",
        schedule_type: Optional[str],
        schedule_start_epoch: int,
        schedule_end_epoch: int,
        schedule_power: float = 1.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        self._roi_grouped_ce_enabled = True
        self._roi_grouped_ce_group_index_lists = [list(g) for g in group_index_lists]
        self._roi_grouped_ce_class_in_group_id = class_in_group_id
        self._roi_grouped_ce_schedule_type = schedule_type
        self._roi_grouped_ce_schedule_start_epoch = int(schedule_start_epoch)
        self._roi_grouped_ce_schedule_end_epoch = int(schedule_end_epoch)
        self._roi_grouped_ce_schedule_power = float(schedule_power)
        self.set_grouped_ce_alpha_for_epoch(0)

    def set_grouped_ce_alpha_for_epoch(self, epoch: int) -> None:
        if not self._roi_grouped_ce_enabled:
            self.roi_grouped_ce_alpha = 0.0
            return
        from oriented_det.train.grouped_ce import grouped_ce_alpha_for_epoch

        self.roi_grouped_ce_alpha = grouped_ce_alpha_for_epoch(
            epoch,
            enabled=True,
            schedule_type=self._roi_grouped_ce_schedule_type,
            start_epoch=self._roi_grouped_ce_schedule_start_epoch,
            end_epoch=self._roi_grouped_ce_schedule_end_epoch,
            power=self._roi_grouped_ce_schedule_power,
        )

    def roi_grouped_ce_kwargs(self) -> Dict[str, Any]:
        """Keyword args for :func:`oriented_det.models.oriented_roi.roi_classification_loss`."""
        if not self._roi_grouped_ce_enabled or self.roi_grouped_ce_alpha <= 0.0:
            return {
                "grouped_alpha": 0.0,
                "group_index_lists": None,
                "class_in_group_id": None,
            }
        return {
            "grouped_alpha": self.roi_grouped_ce_alpha,
            "group_index_lists": self._roi_grouped_ce_group_index_lists,
            "class_in_group_id": self._roi_grouped_ce_class_in_group_id,
        }


__all__ = [
    "rboxes_to_tensor",
    "tensor_to_rboxes",
    "prepare_targets",
    "max_iou_vs_boxes",
    "force_lookalike_to_background",
    "sample_fg_bg_indices",
    "setup_backbone",
    "extract_backbone_features",
    "setup_anchors",
    "ClassWeightsMixin",
    "SigmoidFocalClassWeightsMixin",
    "GroupedCeMixin",
]
