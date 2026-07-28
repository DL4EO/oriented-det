"""Oriented Region of Interest (ROI) head and loss functions for oriented object detection.

This module implements ROI head components that work with oriented bounding boxes:
- ROI pooling/align for oriented boxes
- Classification head
- Box regression head (5 parameters: cx, cy, w, h, angle)
- Loss functions for ROI predictions
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

RegNorm = Literal["sampled_all", "positives_only"]
MainRegLossType = Literal["smooth_l1", "probiou", "riou", "kfiou"]

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ..geometry import RBox
from ..ops import iou
from ..ops.gpu_ops import oriented_box_hbb_iou_gpu
from ..ops.kfiou import mean_auxiliary_box_reg_loss
from ..ops.rotated_ops import pairwise_rotated_iou
from .oriented_rpn import encode_oriented_boxes, decode_oriented_boxes, normalize_angle_delta
from .horizontal_roi_coder import encode_delta_xywh_th, decode_delta_xywh_th
from ..utils.logging import logger

# Global flag to use GPU-accelerated operations
# Uses sampling-based IoU approximation that is fully vectorized on GPU
USE_GPU_OPS = True

# Import gradient checkpointing utility
try:
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    _CHECKPOINTING_AVAILABLE = True
except ImportError:
    grad_checkpoint = None
    _CHECKPOINTING_AVAILABLE = False


def assign_roi_fpn_levels_mmrotate(
    boxes: torch.Tensor,
    fpn_strides: List[int],
    finest_scale: float = 56.0,
    box_format: str = "obb",
) -> torch.Tensor:
    """MMDet/MMRotate-style FPN level from ``sqrt(w*h)`` (``finest_scale`` default 56).

    Returns feature-map indices in ``[0, len(fpn_strides) - 1]`` (finest stride → index 0).

    Args:
        boxes: ``[N, 5]`` oriented ``(cx,cy,w,h,a)`` if ``box_format='obb'``, else ``[N,4]`` xyxy.
        fpn_strides: Strides per level (e.g. ``[8, 16, 32, 64]``).
        finest_scale: Reference scale in pixels (MMRotate/MMDet default 56).
        box_format: ``\"obb\"`` or ``\"xyxy\"``.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    num_levels = len(fpn_strides)
    if num_levels == 0:
        raise ValueError("fpn_strides must be non-empty")
    if box_format == "xyxy":
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    else:
        w = boxes[:, 2].clamp(min=1e-6)
        h = boxes[:, 3].clamp(min=1e-6)
    scale = torch.sqrt(w * h)
    lvl = torch.floor(torch.log2(scale / finest_scale + 1e-6)).long()
    return torch.clamp(lvl, min=0, max=num_levels - 1)


def _create_rotated_grid(
    box: torch.Tensor,
    grid_local: torch.Tensor,
    spatial_scale: float,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a rotated sampling grid for a single box.
    
    Args:
        box: Single box tensor [5] with format [cx, cy, w, h, angle]
        grid_local: Base grid in box-local coordinates [grid_h, grid_w, 2]
        spatial_scale: Scale factor to convert image coords to feature map coords
        H: Feature map height
        W: Feature map width
        device: Device to create tensors on
    
    Returns:
        Sampling grid of shape [1, grid_h, grid_w, 2] in normalized [-1, 1] coords
    """
    return _create_rotated_grids(
        box.unsqueeze(0), grid_local, spatial_scale, H, W
    )


def _create_rotated_grids(
    boxes: torch.Tensor,
    grid_local: torch.Tensor,
    spatial_scale: float,
    H: int,
    W: int,
) -> torch.Tensor:
    """Batched rotated sampling grids for ``grid_sample``.

    Args:
        boxes: ``[N, 5]`` oriented boxes ``(cx, cy, w, h, angle)`` in image coords.
        grid_local: Base grid in box-local coords ``[grid_h, grid_w, 2]``.
        spatial_scale: Image → feature scale.
        H, W: Feature map spatial size.

    Returns:
        ``[N, grid_h, grid_w, 2]`` normalized ``[-1, 1]`` grids.
    """
    grid_h, grid_w = grid_local.shape[:2]
    dtype = boxes.dtype
    device = boxes.device
    cx = boxes[:, 0] * spatial_scale
    cy = boxes[:, 1] * spatial_scale
    w = boxes[:, 2] * spatial_scale
    h = boxes[:, 3] * spatial_scale
    angle = boxes[:, 4]

    gx = grid_local[..., 0].to(dtype=dtype, device=device).view(1, grid_h, grid_w) * (
        w * 0.5
    ).view(-1, 1, 1)
    gy = grid_local[..., 1].to(dtype=dtype, device=device).view(1, grid_h, grid_w) * (
        h * 0.5
    ).view(-1, 1, 1)
    cos_a = torch.cos(angle).view(-1, 1, 1)
    sin_a = torch.sin(angle).view(-1, 1, 1)
    rx = gx * cos_a - gy * sin_a
    ry = gx * sin_a + gy * cos_a
    grid_x = (rx + cx.view(-1, 1, 1)) / (float(W) * 0.5) - 1.0
    grid_y = (ry + cy.view(-1, 1, 1)) / (float(H) * 0.5) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _checkpointable_grid_sample(
    feature_map: torch.Tensor,
    grids: torch.Tensor,
) -> torch.Tensor:
    """Grid sample wrapped for gradient checkpointing.
    
    This function is designed to be used with torch.utils.checkpoint.checkpoint()
    which recomputes the forward pass during backward instead of storing activations.
    
    Args:
        feature_map: Feature map tensor [1, C, H, W] or [N, C, H, W]
        grids: Sampling grids [N, grid_h, grid_w, 2]
    
    Returns:
        Sampled features [N, C, grid_h, grid_w]
    """
    return F.grid_sample(
        feature_map,
        grids,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )


def oriented_roi_align(
    feature_maps: List[torch.Tensor],
    boxes: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    box_to_image: Optional[torch.Tensor] = None,
    output_size: Tuple[int, int] = (7, 7),
    spatial_scales: Optional[List[float]] = None,
    fpn_strides: Optional[List[int]] = None,
    chunk_size: int = 32,
    use_checkpoint: bool = False,
    *,
    finest_scale: float = 56.0,
) -> torch.Tensor:
    """Extract features from oriented bounding boxes using rotated ROI align.
    
    This function extracts fixed-size features from oriented bounding boxes by:
    1. Assigning boxes to appropriate FPN levels (if multiple levels provided)
    2. Creating a rotated sampling grid for each box
    3. Using bilinear interpolation to sample features
    
    Memory Efficiency:
        Processes boxes in small chunks to avoid memory explosion during backward pass.
        Peak memory is limited to roughly chunk_size * C * H * W per grid_sample call.
    
    Args:
        feature_maps: List of feature maps from FPN, each of shape [B, C, H, W]
        boxes: Oriented boxes [N, 5] with format [cx, cy, w, h, angle]
        image_sizes: List of (height, width) for each image in the batch
        box_to_image: Optional tensor [N] mapping each box to image index.
                     If None, assumes all boxes are from the first image.
        output_size: Output feature size (height, width), e.g., (7, 7)
        spatial_scales: Spatial scales for each FPN level (default: 1/stride)
        fpn_strides: Strides for each FPN level (for level assignment)
        chunk_size: Boxes to process in parallel (higher = faster, more memory)
        use_checkpoint: Use gradient checkpointing (~2x less memory, ~30% slower)
        finest_scale: Passed to MMRotate-style assignment (default 56).
    
    Returns:
        Extracted features of shape [N, C, output_h, output_w]
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for oriented ROI align.")
    
    if len(feature_maps) == 0:
        raise ValueError("At least one feature map is required.")

  # MMRotate RotatedSingleRoIExtractor: ROI uses first 4 FPN levels (e.g. strides 4–32).
    num_roi_levels = min(len(feature_maps), 4)
    if fpn_strides is not None:
        roi_strides = [int(s) for s in fpn_strides[:num_roi_levels]]
    else:
        roi_strides = None
    roi_feature_maps = feature_maps[:num_roi_levels]
    
    with logger.trace_block(
        "oriented_roi_align(num_boxes={}, num_levels={}, output_size={}, chunk_size={})",
        boxes.shape[0], len(roi_feature_maps), output_size, chunk_size
    ):
        device = roi_feature_maps[0].device
        num_boxes = boxes.shape[0]
        
        if num_boxes == 0:
            # Return empty tensor with correct shape
            C = roi_feature_maps[0].shape[1]
            logger.trace("No boxes provided, returning empty tensor")
            return torch.zeros((0, C, output_size[0], output_size[1]), device=device)
    
    # Determine which feature map to use for each box
    if len(roi_feature_maps) > 1 and roi_strides is not None:
        feature_map_indices = assign_roi_fpn_levels_mmrotate(
            boxes, roi_strides, finest_scale=finest_scale, box_format="obb"
        )
    else:
        # Use first feature map for all boxes
        feature_map_indices = torch.zeros(num_boxes, dtype=torch.long, device=device)
    
    # Determine spatial scales
    if spatial_scales is None:
        # Default: compute from FPN level
        if roi_strides is not None:
            spatial_scales = [1.0 / stride for stride in roi_strides]
        else:
            # Default scale for single feature map
            spatial_scales = [1.0 / 16.0]
    else:
        spatial_scales = spatial_scales[:num_roi_levels]
    
    # Determine which image each box belongs to
    if box_to_image is None:
        box_to_image = torch.zeros(num_boxes, dtype=torch.long, device=device)
    
    B = roi_feature_maps[0].shape[0]
    C = roi_feature_maps[0].shape[1]
    
    # Create base sampling grid (in box-local coordinates)
    # Grid coordinates in normalized [-1, 1] range
    grid_h, grid_w = output_size
    y_coords = torch.linspace(-1, 1, grid_h, device=device)
    x_coords = torch.linspace(-1, 1, grid_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Grid is in box-local coordinates (centered at origin, aligned with box)
    # Shape: [grid_h, grid_w, 2] where last dim is [x, y] in box-local coords
    grid_local = torch.stack([grid_x, grid_y], dim=-1)  # [grid_h, grid_w, 2]
    
    dtype = roi_feature_maps[0].dtype
    _onnx_export = bool(
        torch.onnx.is_in_onnx_export() if hasattr(torch.onnx, "is_in_onnx_export") else False
    )

    if _onnx_export:
        # All-N grid_sample per level with masks (avoids ScatterND from indexed writes).
        out = torch.zeros(
            (num_boxes, C, grid_h, grid_w), device=device, dtype=dtype
        )
        for feat_idx in range(len(roi_feature_maps)):
            feature_map = roi_feature_maps[feat_idx]
            _, _, H, W = feature_map.shape
            spatial_scale = spatial_scales[min(feat_idx, len(spatial_scales) - 1)]
            level_mask = (feature_map_indices == feat_idx).to(dtype=dtype).view(
                num_boxes, 1, 1, 1
            )
            grids = _create_rotated_grids(
                boxes, grid_local, spatial_scale, H, W
            )
            for img_idx in range(B):
                img_mask = (box_to_image == img_idx).to(dtype=dtype).view(
                    num_boxes, 1, 1, 1
                )
                feature_input = feature_map[img_idx : img_idx + 1].expand(
                    num_boxes, -1, -1, -1
                )
                sampled = F.grid_sample(
                    feature_input,
                    grids,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                out = out + sampled * level_mask * img_mask
        return out

    # Pre-allocate output tensor for efficiency
    output_features = torch.zeros((num_boxes, C, grid_h, grid_w), device=device, dtype=dtype)
    
    # Process boxes by feature map level and image
    # MEMORY-EFFICIENT: Process boxes sequentially to avoid expand() memory explosion
    for feat_idx in range(len(roi_feature_maps)):
        feature_map = roi_feature_maps[feat_idx]  # [B, C, H, W]
        _, _, H, W = feature_map.shape
        spatial_scale = spatial_scales[min(feat_idx, len(spatial_scales) - 1)]
        
        # Find boxes assigned to this feature map
        mask = feature_map_indices == feat_idx
        if not mask.any():
            continue
        
        box_indices = mask.nonzero(as_tuple=True)[0]
        level_boxes = boxes[box_indices]
        
        # Process boxes by image
        for img_idx in range(B):
            img_mask = (box_to_image[box_indices] == img_idx)
            if not img_mask.any():
                continue
            
            img_box_indices = box_indices[img_mask]
            img_boxes = level_boxes[img_mask]
            num_img_boxes = len(img_boxes)
            
            if num_img_boxes == 0:
                continue
            
            # Get the feature map for this image (NOT expanded)
            feature_map_img = feature_map[img_idx:img_idx+1]  # [1, C, H, W]
            
            # MEMORY-EFFICIENT: Process boxes in chunks
            # Each chunk processes up to chunk_size boxes together
            # This limits peak memory while still allowing some parallelism
            for chunk_start in range(0, num_img_boxes, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_img_boxes)
                chunk_boxes = img_boxes[chunk_start:chunk_end]
                chunk_indices = img_box_indices[chunk_start:chunk_end]
                num_chunk_boxes = len(chunk_boxes)
                
                # Create grids for this chunk of boxes
                chunk_grids = _create_rotated_grids(
                    chunk_boxes, grid_local, spatial_scale, H, W
                )
                
                # Sample features - expand only for small chunk to limit memory
                feature_input = feature_map_img
                if num_chunk_boxes > 1:
                    feature_input = feature_map_img.expand(num_chunk_boxes, -1, -1, -1)
                
                # Use gradient checkpointing if enabled (recomputes forward during backward)
                if use_checkpoint and _CHECKPOINTING_AVAILABLE and feature_input.requires_grad:
                    sampled_chunk = grad_checkpoint(
                        _checkpointable_grid_sample,
                        feature_input,
                        chunk_grids,
                        use_reentrant=False,
                    )
                else:
                    sampled_chunk = F.grid_sample(
                        feature_input,
                        chunk_grids,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False,
                    )
                
                # Store results in pre-allocated output
                output_features[chunk_indices] = sampled_chunk
    
    return output_features


def horizontal_roi_align(
    feature_maps: List[torch.Tensor],
    boxes_xyxy: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    box_to_image: Optional[torch.Tensor],
    output_size: Tuple[int, int],
    spatial_scales: Optional[List[float]],
    fpn_strides: Optional[List[int]],
    chunk_size: int = 32,
    finest_scale: float = 56.0,
) -> torch.Tensor:
    """Horizontal RoIAlign (``torchvision.ops.roi_align``) with MMRotate-style FPN level assignment.

    Args:
        feature_maps: FPN levels ``[B, C, H_l, W_l]``.
        boxes_xyxy: ``[N, 4]`` in **image pixel** coordinates ``(x1, y1, x2, y2)``.
        image_sizes: Per-image ``(H, W)`` (reserved for API parity with oriented path).
        box_to_image: ``[N]`` batch indices into ``feature_maps[*].shape[0]``.
        output_size: ``(pool_h, pool_w)``.
        spatial_scales: Optional ``1/stride`` per level; inferred from ``fpn_strides`` if omitted.
        fpn_strides: Stride per FPN level (same length as ``feature_maps``).
        chunk_size: Chunk size for ``roi_align`` calls (memory).
        finest_scale: Passed to :func:`assign_roi_fpn_levels_mmrotate`.

    MMRotate alignment:
        For two-stage detectors with 5 FPN outputs (e.g. strides ``[4,8,16,32,64]``),
        ROI extraction uses only the first 4 levels (``[4,8,16,32]``). The coarsest
        extra level is kept for RPN but excluded from ROI extraction.

    Returns:
        Tensor ``[N, C, pool_h, pool_w]``.
    """
    del image_sizes  # boxes are already in image space; clipping is the caller's responsibility
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for horizontal ROI align.")
    try:
        from torchvision.ops import roi_align as tv_roi_align
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for horizontal_roi_align.") from exc

    if len(feature_maps) == 0:
        raise ValueError("At least one feature map is required.")
    device = feature_maps[0].device
    dtype = feature_maps[0].dtype
    num_boxes = int(boxes_xyxy.shape[0])
    if num_boxes == 0:
        c = int(feature_maps[0].shape[1])
        return torch.zeros((0, c, output_size[0], output_size[1]), device=device, dtype=dtype)

    if box_to_image is None:
        box_to_image = torch.zeros((num_boxes,), dtype=torch.long, device=device)
    if fpn_strides is None:
        raise ValueError("horizontal_roi_align requires fpn_strides when using FPN feature maps.")
    eff_strides = [int(s) for s in fpn_strides]
    # MMRotate ROI extractor consumes at most 4 levels (typically [4,8,16,32]).
    # Keep any extra level (e.g. stride 64) for RPN, but do not use it for ROIAlign.
    num_roi_levels = min(len(feature_maps), len(eff_strides), 4)
    if num_roi_levels <= 0:
        raise ValueError("horizontal_roi_align requires at least one ROI feature level.")
    roi_feature_maps = feature_maps[:num_roi_levels]
    roi_strides = eff_strides[:num_roi_levels]

    if num_roi_levels > 1:
        level_idx = assign_roi_fpn_levels_mmrotate(
            boxes_xyxy, roi_strides, finest_scale=finest_scale, box_format="xyxy"
        )
    else:
        level_idx = torch.zeros((num_boxes,), dtype=torch.long, device=device)

    c = int(feature_maps[0].shape[1])
    out = torch.zeros((num_boxes, c, output_size[0], output_size[1]), device=device, dtype=dtype)
    _onnx_export = bool(
        torch.onnx.is_in_onnx_export() if hasattr(torch.onnx, "is_in_onnx_export") else False
    )

    if _onnx_export:
        # RoIAlign on all N boxes per level with a mask (avoids trace-time empty batches).
        all_rois = torch.cat(
            [box_to_image.unsqueeze(1).to(dtype=dtype), boxes_xyxy],
            dim=1,
        )
        for lev in range(num_roi_levels):
            feat = roi_feature_maps[lev]
            scale = 1.0 / float(roi_strides[lev])
            level_mask = (level_idx == lev).to(dtype=dtype).view(num_boxes, 1, 1, 1)
            rois = torch.cat([all_rois[:, :1], boxes_xyxy * scale], dim=1)
            sampled = tv_roi_align(
                feat,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                aligned=True,  # mmcv RoIAlign default (half-pixel aligned), per MMRotate
            )
            out = out + sampled * level_mask
        return out

    for lev in range(num_roi_levels):
        mask = level_idx == lev
        idx_all = mask.nonzero(as_tuple=True)[0]
        if idx_all.numel() == 0:
            continue
        feat = roi_feature_maps[lev]
        stride = float(roi_strides[lev])
        scale = 1.0 / stride
        if idx_all.numel() <= chunk_size:
            rois = torch.cat(
                [box_to_image[idx_all].unsqueeze(1).to(dtype=dtype), boxes_xyxy[idx_all] * scale],
                dim=1,
            )
            out[idx_all] = tv_roi_align(
                feat,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                aligned=True,  # mmcv RoIAlign default (half-pixel aligned), per MMRotate
            )
        else:
            for start in range(0, int(idx_all.numel()), chunk_size):
                sl = idx_all[start : start + chunk_size]
                rois_chunk = torch.cat(
                    [box_to_image[sl].unsqueeze(1).to(dtype=dtype), boxes_xyxy[sl] * scale],
                    dim=1,
                )
                out[sl] = tv_roi_align(
                    feat,
                    rois_chunk,
                    output_size,
                    spatial_scale=1.0,
                    sampling_ratio=0,
                    aligned=True,  # mmcv RoIAlign default (half-pixel aligned), per MMRotate
                )
    return out


class OrientedROIAlign(nn.Module if nn is not None else object):  # type: ignore
    """Oriented ROI Align module for extracting features from oriented bounding boxes.
    
    Uses chunked processing to avoid memory explosion during backward pass.
    
    Args:
        output_size: Output feature size (height, width), e.g., (7, 7)
        spatial_scales: Spatial scales for each FPN level
        fpn_strides: Strides for each FPN level
        chunk_size: Boxes to process in parallel (16-64 recommended for 8-24GB GPU)
        use_checkpoint: Use gradient checkpointing (~2x less memory, ~30% slower)
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (7, 7),
        spatial_scales: Optional[List[float]] = None,
        fpn_strides: Optional[List[int]] = None,
        chunk_size: int = 32,
        use_checkpoint: bool = False,
        finest_scale: float = 56.0,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for OrientedROIAlign.")
        super().__init__()
        
        self.output_size = output_size
        self.spatial_scales = spatial_scales
        self.fpn_strides = fpn_strides
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.finest_scale = finest_scale
    
    def forward(
        self,
        feature_maps: List[torch.Tensor],
        boxes: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        box_to_image: Optional[torch.Tensor] = None,
        *,
        fpn_strides_override: Optional[List[int]] = None,
        spatial_scales_override: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Extract features from oriented boxes.
        
        Args:
            feature_maps: List of feature maps from FPN
            boxes: Oriented boxes [N, 5] with format [cx, cy, w, h, angle]
            image_sizes: List of (height, width) for each image
            box_to_image: Optional tensor mapping each box to image index
            fpn_strides_override: If set, strides from the actual feature grid (overrides init config).
            spatial_scales_override: If set, scales for ROI sampling (default ``1/stride`` from override).
        
        Returns:
            Extracted features [N, C, output_h, output_w]
        """
        eff_fpn = fpn_strides_override if fpn_strides_override is not None else self.fpn_strides
        eff_spatial = spatial_scales_override if spatial_scales_override is not None else self.spatial_scales
        return oriented_roi_align(
            feature_maps,
            boxes,
            image_sizes,
            box_to_image,
            self.output_size,
            eff_spatial,
            eff_fpn,
            self.chunk_size,
            self.use_checkpoint,
            finest_scale=self.finest_scale,
        )


class HorizontalROIAlign(nn.Module if nn is not None else object):  # type: ignore
    """Axis-aligned RoIAlign via ``torchvision.ops.roi_align`` (MMRotate Rotated Faster R-CNN ROI)."""

    def __init__(
        self,
        output_size: Tuple[int, int] = (7, 7),
        spatial_scales: Optional[List[float]] = None,
        fpn_strides: Optional[List[int]] = None,
        chunk_size: int = 32,
        finest_scale: float = 56.0,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for HorizontalROIAlign.")
        super().__init__()
        self.output_size = output_size
        self.spatial_scales = spatial_scales
        self.fpn_strides = fpn_strides
        self.chunk_size = chunk_size
        self.finest_scale = finest_scale

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        boxes_xyxy: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        box_to_image: Optional[torch.Tensor] = None,
        *,
        fpn_strides_override: Optional[List[int]] = None,
        spatial_scales_override: Optional[List[float]] = None,
    ) -> torch.Tensor:
        eff_fpn = fpn_strides_override if fpn_strides_override is not None else self.fpn_strides
        eff_spatial = spatial_scales_override if spatial_scales_override is not None else self.spatial_scales
        return horizontal_roi_align(
            feature_maps,
            boxes_xyxy,
            image_sizes,
            box_to_image,
            self.output_size,
            eff_spatial,
            eff_fpn,
            self.chunk_size,
            self.finest_scale,
        )


def match_oriented_proposals_to_gt(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.5,
    device: Optional[torch.device] = None,
    use_hbb_for_matching: bool = False,
    match_low_quality: bool = False,
    min_pos_iou: float = 0.5,
    gt_boxes_ignore: Optional[torch.Tensor] = None,
    ignore_iou_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match oriented proposals to ground truth boxes using IoU.
    
    Assignment order (MMDet-style): (1) proposals with IoU >= positive_iou_threshold
    are positive; (2) when match_low_quality is True, for each GT the best proposal
    is forced positive if its IoU >= min_pos_iou; (3) proposals with IoU <
    negative_iou_threshold are background (unless they are the protected best for
    some GT). Proposals in [neg, pos) are ignore (-1).
    
    When use_hbb_for_matching is False, uses oriented (OBB) IoU. When True, uses
    axis-aligned (HBB) IoU so that proposals with good location/size but wrong
    angle still get positive labels; the ROI head can then learn angle refinement.
    
    Args:
        proposals: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
        gt_boxes: Tensor of shape [M, 5] with format [cx, cy, w, h, angle]
        gt_labels: Tensor of shape [M] with class labels (1-indexed: 1, 2, ..., num_classes)
        positive_iou_threshold: IoU threshold for positive matches
        negative_iou_threshold: IoU threshold for negative matches (typically same as positive)
        device: Optional device for computation
        use_hbb_for_matching: If True, use axis-aligned (HBB) IoU for assignment instead of OBB IoU.
            Recommended when angle regression is weak so more proposals get positive labels.
        match_low_quality: If True, force best proposal per GT to positive when IoU >= min_pos_iou.
            MMRotate Rotated Faster R-CNN ROI uses False.
        min_pos_iou: When match_low_quality is True, only force best proposal to positive if
            max_iou_per_gt >= min_pos_iou. MMRotate RCNN assigner uses 0.5.
    
    Returns:
        Tuple of (labels, matched_gt_indices, matched_gt_boxes):
        - labels: Tensor of shape [N] with class labels
            (-1 = ignore, 0 = background, 1,2,... = object classes)
        - matched_gt_indices: Tensor of shape [N] with index of matched GT box (-1 if bg/ignore)
        - matched_gt_boxes: Tensor of shape [N, 5] with matched GT boxes (zeros for bg/ignore)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for proposal matching.")
    
    if device is None:
        device = proposals.device if hasattr(proposals, 'device') else torch.device('cpu')
    
    num_proposals = proposals.shape[0]
    num_gt = gt_boxes.shape[0]
    
    # Initialize outputs
    labels = torch.full((num_proposals,), -1, dtype=torch.int64, device=device)
    matched_gt_indices = torch.full((num_proposals,), -1, dtype=torch.int64, device=device)
    matched_gt_boxes = torch.zeros((num_proposals, 5), dtype=torch.float32, device=device)
    
    if num_gt == 0:
        # No ground truth - all proposals are background
        labels.fill_(0)
        return labels, matched_gt_indices, matched_gt_boxes
    
    # Detach inputs - matching doesn't need gradients
    if isinstance(proposals, torch.Tensor):
        proposals = proposals.to(device).detach()
    else:
        proposals = torch.tensor(proposals, dtype=torch.float32, device=device)
    
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.to(device).detach()
    else:
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
    
    # Compute IoU matrix (no gradients needed for matching)
    with torch.no_grad():
        # Use GPU-accelerated IoU (10-100x faster)
        if USE_GPU_OPS:
            if use_hbb_for_matching:
                iou_matrix = oriented_box_hbb_iou_gpu(proposals, gt_boxes)
            else:
                iou_matrix = pairwise_rotated_iou(proposals, gt_boxes)
        else:
            # Fallback to Python implementation (slow, for debugging only)
            try:
                from ..ops.iou import batch_rbox_iou
                
                # Convert to RBox format for IoU computation
                proposal_list = [RBox(*p.tolist()) for p in proposals]
                gt_list = [RBox(*g.tolist()) for g in gt_boxes]
                
                # Use batch IoU computation
                iou_matrix_list = batch_rbox_iou(proposal_list, gt_list, device=device)
                iou_matrix = torch.tensor(iou_matrix_list, device=device, requires_grad=False)  # [N, M]
            except Exception:
                # Fall back to Python implementation
                iou_matrix = torch.zeros((num_proposals, num_gt), device=device, requires_grad=False)
                for i, proposal in enumerate(proposals):
                    proposal_rbox = RBox(*proposal.tolist())
                    for j, gt_box in enumerate(gt_boxes):
                        gt_rbox = RBox(*gt_box.tolist())
                        iou_matrix[i, j] = iou.rbox_iou(proposal_rbox, gt_rbox)
    
    # For each proposal, find the GT with highest IoU
    max_iou_per_proposal, best_gt_per_proposal = iou_matrix.max(dim=1)  # [N]
    
    # Mark proposals above positive threshold as positive
    positive_mask = max_iou_per_proposal >= positive_iou_threshold
    labels[positive_mask] = gt_labels[best_gt_per_proposal[positive_mask]]  # Use actual class labels
    matched_gt_indices[positive_mask] = best_gt_per_proposal[positive_mask]
    matched_gt_boxes[positive_mask] = gt_boxes[best_gt_per_proposal[positive_mask]]
    
    # For each ground truth box, find the proposal with highest IoU.
    # Force best proposals as positive only when overlap is meaningful.
    max_iou_per_gt, best_proposal_per_gt = iou_matrix.max(dim=0)  # [M]
    
    # Track which proposals are assigned as "best" for a GT (protected from becoming background)
    best_proposal_mask = torch.zeros(num_proposals, dtype=torch.bool, device=device)
    
    # Debug: Log IoU statistics and proposal/GT comparison
    logger.trace(
        "ROI IoU stats: max_iou_per_gt={}, num_gt_with_iou>0={}, "
        "mean_max_iou={:.4f}, min_max_iou={:.4f}",
        max_iou_per_gt.tolist(),
        (max_iou_per_gt > 0).sum().item(),
        max_iou_per_gt.mean().item() if num_gt > 0 else 0.0,
        max_iou_per_gt.min().item() if num_gt > 0 else 0.0
    )
    
    # Log proposal statistics
    if len(proposals) > 0:
        logger.trace(
            "ROI proposals: count={}, "
            "cx_range=[{:.1f}, {:.1f}], cy_range=[{:.1f}, {:.1f}], "
            "w_range=[{:.1f}, {:.1f}], h_range=[{:.1f}, {:.1f}], "
            "angle_range=[{:.3f}, {:.3f}]",
            len(proposals),
            proposals[:, 0].min().item(), proposals[:, 0].max().item(),
            proposals[:, 1].min().item(), proposals[:, 1].max().item(),
            proposals[:, 2].min().item(), proposals[:, 2].max().item(),
            proposals[:, 3].min().item(), proposals[:, 3].max().item(),
            proposals[:, 4].min().item(), proposals[:, 4].max().item()
        )
    
    # Log GT box statistics for comparison
    if len(gt_boxes) > 0:
        logger.trace(
            "ROI GT boxes: count={}, "
            "cx_range=[{:.1f}, {:.1f}], cy_range=[{:.1f}, {:.1f}], "
            "w_range=[{:.1f}, {:.1f}], h_range=[{:.1f}, {:.1f}], "
            "angle_range=[{:.3f}, {:.3f}]",
            len(gt_boxes),
            gt_boxes[:, 0].min().item(), gt_boxes[:, 0].max().item(),
            gt_boxes[:, 1].min().item(), gt_boxes[:, 1].max().item(),
            gt_boxes[:, 2].min().item(), gt_boxes[:, 2].max().item(),
            gt_boxes[:, 3].min().item(), gt_boxes[:, 3].max().item(),
            gt_boxes[:, 4].min().item(), gt_boxes[:, 4].max().item()
        )
        
        # Log individual GT boxes for detailed comparison
        for gt_idx in range(len(gt_boxes)):
            gt_box = gt_boxes[gt_idx]
            max_iou = max_iou_per_gt[gt_idx].item()
            best_proposal_idx = best_proposal_per_gt[gt_idx].item() if max_iou > 0 else -1
            
            logger.trace(
                "GT box {}: cx={:.1f}, cy={:.1f}, w={:.1f}, h={:.1f}, angle={:.3f}, max_iou={:.4f}",
                gt_idx,
                gt_box[0].item(), gt_box[1].item(),
                gt_box[2].item(), gt_box[3].item(),
                gt_box[4].item(), max_iou
            )
            
            # If max_iou is 0, find the closest proposal by center distance
            if max_iou == 0 and len(proposals) > 0:
                gt_cx, gt_cy = gt_box[0].item(), gt_box[1].item()
                proposal_centers = proposals[:, :2]  # [N, 2] with cx, cy
                distances = torch.sqrt(((proposal_centers - torch.tensor([gt_cx, gt_cy], device=proposals.device))**2).sum(dim=1))
                closest_idx = distances.argmin().item()
                closest_proposal = proposals[closest_idx]
                closest_dist = distances[closest_idx].item()
                logger.trace(
                    "  → Closest proposal (idx {}): cx={:.1f}, cy={:.1f}, w={:.1f}, h={:.1f}, angle={:.3f}, "
                    "center_dist={:.1f}px",
                    closest_idx,
                    closest_proposal[0].item(), closest_proposal[1].item(),
                    closest_proposal[2].item(), closest_proposal[3].item(),
                    closest_proposal[4].item(), closest_dist
                )
    
    if match_low_quality:
        for gt_idx in range(num_gt):
            if max_iou_per_gt[gt_idx] >= min_pos_iou:
                proposal_idx = best_proposal_per_gt[gt_idx].item()
                labels[proposal_idx] = gt_labels[gt_idx]
                matched_gt_indices[proposal_idx] = gt_idx
                matched_gt_boxes[proposal_idx] = gt_boxes[gt_idx]
                best_proposal_mask[proposal_idx] = True
    
    # Mark proposals below negative threshold as background,
    # but exclude proposals that are the protected best match for some GT.
    # Proposals in [neg, pos) stay -1 (ignore) and are not sampled for ROI loss.
    negative_mask = (max_iou_per_proposal < negative_iou_threshold) & (~best_proposal_mask)
    labels[negative_mask] = 0

    # Optional: ignore GT regions (MMDet/MMRotate behavior). Apply after normal GT assignment,
    # but do not override positives.
    if gt_boxes_ignore is not None and gt_boxes_ignore.numel() > 0:
        thr = float(ignore_iou_threshold) if ignore_iou_threshold is not None else float(positive_iou_threshold)
        with torch.no_grad():
            ign = gt_boxes_ignore.to(device).detach()
            if USE_GPU_OPS:
                if use_hbb_for_matching:
                    iou_ign = oriented_box_hbb_iou_gpu(proposals, ign)
                else:
                    iou_ign = pairwise_rotated_iou(proposals, ign)
                max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((num_proposals,), device=device)
            else:
                from ..ops.iou import batch_rbox_iou
                prop_list = [RBox(*p.tolist()) for p in proposals.detach().cpu()]
                ign_list = [RBox(*g.tolist()) for g in ign.detach().cpu()]
                iou_ign = torch.tensor(batch_rbox_iou(prop_list, ign_list, device=device), device=device)
                max_iou_ign = iou_ign.max(dim=1).values if iou_ign.numel() > 0 else torch.zeros((num_proposals,), device=device)
            ign_mask = (labels <= 0) & (max_iou_ign >= thr)
            labels[ign_mask] = -1
            matched_gt_indices[ign_mask] = -1
            matched_gt_boxes[ign_mask] = 0.0
    
    return labels, matched_gt_indices, matched_gt_boxes


def grouped_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    grouped_alpha: float,
    group_index_lists: Sequence[Sequence[int]],
    class_in_group_id: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with optional coarse group targets for foreground classes.

    For foreground samples in a group, the grouped term is ``-log sum_{k in group} p(k)``.
    Background (label 0) always uses fine CE. When ``grouped_alpha`` is in (0, 1), grouped
    and fine per-sample losses are linearly mixed for grouped foreground samples.

    Args:
        logits: [N, num_classes + 1]
        targets: [N] class indices (0 = background)
        grouped_alpha: 1.0 = grouped only (for grouped fg), 0.0 = fine CE only
        group_index_lists: per-group 1-indexed foreground class ids
        class_in_group_id: [num_classes + 1] long; -1 for bg/unmapped fg, else group index
        class_weights: optional [num_classes + 1] tensor
        label_smoothing: passed to fine CE branch
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for grouped cross-entropy.")

    if grouped_alpha <= 0.0 or not group_index_lists:
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

    fine = F.cross_entropy(
        logits,
        targets,
        weight=None,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if class_weights is not None:
        fine = fine * class_weights[targets]

    log_probs = F.log_softmax(logits, dim=-1)
    grouped_vec = torch.zeros_like(fine)
    num_groups = len(group_index_lists)
    if class_in_group_id.numel() != logits.shape[1]:
        raise ValueError(
            f"class_in_group_id length {class_in_group_id.numel()} != num logits classes {logits.shape[1]}"
        )
    cig = class_in_group_id[targets]
    for g_idx, class_ids in enumerate(group_index_lists):
        if not class_ids:
            continue
        cols = torch.tensor(class_ids, device=logits.device, dtype=torch.long)
        log_p_group = torch.logsumexp(log_probs[:, cols], dim=1)
        in_group = (targets > 0) & (cig == g_idx)
        grouped_vec = torch.where(in_group, -log_p_group, grouped_vec)

    has_group = (targets > 0) & (cig >= 0)
    alpha = float(grouped_alpha)
    per_sample = torch.where(
        targets == 0,
        fine,
        torch.where(
            has_group,
            alpha * grouped_vec + (1.0 - alpha) * fine,
            fine,
        ),
    )
    return per_sample.mean()


def roi_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    grouped_alpha: float = 0.0,
    group_index_lists: Optional[Sequence[Sequence[int]]] = None,
    class_in_group_id: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """ROI classifier loss (CE, focal, or grouped CE curriculum)."""
    use_grouped = (
        grouped_alpha > 0.0
        and group_index_lists
        and class_in_group_id is not None
        and loss_type == "cross_entropy"
    )
    if use_grouped:
        return grouped_cross_entropy_loss(
            logits,
            targets,
            grouped_alpha=grouped_alpha,
            group_index_lists=group_index_lists,
            class_in_group_id=class_in_group_id,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type == "focal":
        return focal_loss(
            logits,
            targets,
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type == "cross_entropy":
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="mean",
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}. Must be one of ['cross_entropy', 'focal']")


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute focal loss for classification.
    
    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard examples. It combines the benefits of
    class weighting with hard example mining.
    
    Args:
        logits: Classification logits of shape [N, num_classes]
        targets: Ground truth class labels of shape [N] (0-indexed)
        alpha: Weighting factor for rare class vs common class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
               - gamma=0: equivalent to cross-entropy
               - gamma>0: focuses on hard examples
               - gamma=2: standard value used in literature
        class_weights: Optional class weights tensor of shape [num_classes]
                      Applied in addition to focal loss weighting
        label_smoothing: Label smoothing factor (default: 0.0). Use e.g. 0.1 to reduce overconfidence.
    
    Returns:
        Focal loss scalar tensor
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for focal loss computation.")
    
    # Compute cross-entropy loss for each sample (not reduced)
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=class_weights, label_smoothing=label_smoothing)

    # Clamp ce_loss to avoid -inf from log(0) and to keep exp(-ce_loss) in valid range
    ce_loss = torch.clamp(ce_loss, min=1e-7, max=50.0)

    # Compute p_t: probability of true class
    # For correctly classified examples, p_t is high (easy examples)
    # For incorrectly classified examples, p_t is low (hard examples)
    pt = torch.exp(-ce_loss)
    pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # Avoid (1-pt)**gamma or log(0) edge cases

    # Focal term: (1 - p_t)^gamma
    # This down-weights easy examples (high p_t) and focuses on hard examples (low p_t)
    focal_term = (1 - pt) ** gamma

    # Focal loss = alpha * focal_term * ce_loss
    focal_loss_value = alpha * focal_term * ce_loss
    focal_loss_value = torch.where(
        torch.isfinite(focal_loss_value),
        focal_loss_value,
        torch.zeros_like(focal_loss_value, device=focal_loss_value.device),
    )

    return focal_loss_value.mean()


def _roi_assignment_stats(
    matched_labels: torch.Tensor,
    matched_gt_indices: torch.Tensor,
    num_gt: int,
    fg_bg_sampling_ratio: float,
    batch_size_per_image: int,
) -> Dict[str, float]:
    """Aggregate ROI assignment counts (shared by loss and RPN-only diagnostics)."""
    fg_indices = (matched_labels > 0).nonzero(as_tuple=True)[0]
    bg_indices = (matched_labels == 0).nonzero(as_tuple=True)[0]
    ignore_indices = (matched_labels == -1).nonzero(as_tuple=True)[0]

    matched_gt_count = 0
    if len(fg_indices) > 0:
        matched_gt_indices_unique = matched_gt_indices[fg_indices].unique()
        matched_gt_count = len(matched_gt_indices_unique[matched_gt_indices_unique >= 0])

    num_fg = min(len(fg_indices), int(batch_size_per_image * fg_bg_sampling_ratio))
    num_bg = min(len(bg_indices), batch_size_per_image - num_fg)
    match_rate = matched_gt_count / num_gt if num_gt > 0 else 0.0

    return {
        "roi_num_pos": float(len(fg_indices)),
        "roi_num_bg": float(len(bg_indices)),
        "roi_num_ignore": float(len(ignore_indices)),
        "roi_sampled_fg": float(num_fg),
        "roi_sampled_bg": float(num_bg),
        "roi_matched_gt": float(matched_gt_count),
        "roi_match_rate": float(match_rate),
    }


def compute_roi_matching_diagnostics(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.5,
    device: Optional[torch.device] = None,
    use_hbb_for_matching: bool = False,
    match_low_quality: bool = False,
    roi_min_pos_iou: float = 0.5,
    fg_bg_sampling_ratio: float = 0.25,
    batch_size_per_image: int = 512,
) -> Dict[str, float]:
    """ROI assignment stats from a proposal tensor (e.g. RPN-only, before ``add_gt_as_proposals``).

    Uses the same matching rules as :func:`compute_oriented_roi_loss`. Cheap: no ROI head forward.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for proposal matching diagnostics.")
    if proposals.shape[0] == 0:
        num_gt = int(gt_boxes.shape[0]) if gt_boxes is not None else 0
        return {
            "roi_num_pos": 0.0,
            "roi_num_bg": 0.0,
            "roi_num_ignore": 0.0,
            "roi_sampled_fg": 0.0,
            "roi_sampled_bg": 0.0,
            "roi_matched_gt": 0.0,
            "roi_match_rate": 0.0 if num_gt > 0 else 0.0,
        }
    matched_labels, matched_gt_indices, _ = match_oriented_proposals_to_gt(
        proposals,
        gt_boxes,
        gt_labels,
        positive_iou_threshold,
        negative_iou_threshold,
        device,
        use_hbb_for_matching=use_hbb_for_matching,
        match_low_quality=match_low_quality,
        min_pos_iou=roi_min_pos_iou,
    )
    return _roi_assignment_stats(
        matched_labels,
        matched_gt_indices,
        len(gt_boxes),
        fg_bg_sampling_ratio,
        batch_size_per_image,
    )


def compute_oriented_roi_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes_ignore: Optional[torch.Tensor] = None,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.5,
    box_reg_weight: float = 1.0,
    box_reg_angle_weight: float = 1.0,
    fg_bg_sampling_ratio: float = 0.25,
    batch_size_per_image: int = 512,
    num_classes: int = 1,
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    target_means: Optional[Tuple[float, float, float, float, float]] = None,
    target_stds: Optional[Tuple[float, float, float, float, float]] = None,
    norm_factor: Optional[float] = 2.0,
    edge_swap: bool = True,
    proj_xy: bool = False,
    box_reg_iou_weight: float = 0.0,
    box_reg_iou_loss_type: str = "riou",
    box_reg_kfiou_fun: Optional[str] = None,
    box_reg_probiou_mode: Optional[str] = None,
    use_hbb_for_matching: bool = False,
    match_low_quality: bool = False,
    roi_min_pos_iou: float = 0.5,
    label_smoothing: float = 0.0,
    grouped_alpha: float = 0.0,
    group_index_lists: Optional[Sequence[Sequence[int]]] = None,
    class_in_group_id: Optional[torch.Tensor] = None,
    reg_norm: RegNorm = "sampled_all",
    include_assignment_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Compute ROI losses for oriented object detection.
    
    Args:
        class_logits: Classification logits of shape [N, num_classes + 1]
                      (including background class)
        box_regression: Box regression predictions of shape [N, num_classes * 5]
                       (5 parameters per class: cx, cy, w, h, angle)
        proposals: Oriented proposals of shape [N, 5] with format [cx, cy, w, h, angle]
        gt_boxes: Ground truth boxes of shape [M, 5] with format [cx, cy, w, h, angle]
        gt_labels: Ground truth class labels of shape [M] (1-indexed, 0 is background)
        positive_iou_threshold: IoU threshold for positive matches
        negative_iou_threshold: IoU threshold for negative matches
        box_reg_weight: Weight for box regression loss
        box_reg_angle_weight: Scales the angle (5th encoded dim) SmoothL1 term; at 1.0 the
            mean loss matches an unweighted mean over all five encoded dimensions (same as
            :func:`compute_horizontal_roi_loss_mmrotate`).
        fg_bg_sampling_ratio: Ratio of foreground to background in sampled proposals
        batch_size_per_image: Number of proposals to sample per image
        num_classes: Number of object classes (excluding background)
        loss_type: Type of classification loss to use:
                  - "cross_entropy": Standard cross-entropy loss (default)
                  - "focal": Focal loss for handling class imbalance and hard examples
        class_weights: Optional tensor of shape [num_classes + 1] (including background)
                      with class weights for weighted cross-entropy or focal loss.
                      If None, all classes have equal weight (1.0).
                      Typically computed as inverse frequency or square root weighting.
        focal_alpha: Alpha parameter for focal loss (default: 1.0)
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
                    - gamma=0: equivalent to cross-entropy
                    - gamma>0: focuses on hard examples
        target_means: Optional normalization means for box regression targets (cx, cy, w, h, angle)
        target_stds: Optional normalization stds for box regression targets
        norm_factor: Angle scaling for encode (MMRotate uses 2.0).
        edge_swap: Whether to use edge_swap in bbox encode (MMRotate uses True).
        proj_xy: If True, encode/decode dx/dy in the proposal local frame (MMRotate DeltaXYWHTRBBoxCoder).
        box_reg_iou_weight: Extra weight for auxiliary loss on decoded positive ROI boxes,
            added to the SmoothL1 regression loss. Ignored when this weight is 0.
        box_reg_iou_loss_type: ``\"riou\"`` (default): ``mean(1 - rIoU)`` via
            :func:`~oriented_det.ops.rotated_ops.pairwise_rotated_iou`. ``\"kfiou\"``: Kalman-filter
            IoU surrogate (Gaussian overlap + center Smooth L1). ``\"probiou\"``: mean ProbIoU.
        box_reg_kfiou_fun: When ``box_reg_iou_loss_type=\"kfiou\"``, optional ``ln`` / ``exp``
            transform on the overlap term (see :func:`~oriented_det.ops.kfiou.kfiou_loss`).
        box_reg_probiou_mode: When ``box_reg_iou_loss_type=\"probiou\"``, ``\"l1\"`` (default) or ``\"l2\"``.
        use_hbb_for_matching: If True, match proposals to GT using axis-aligned (HBB) IoU
            instead of oriented IoU. Gives more positive labels when proposal angles are wrong,
            so the classifier and angle regression can learn; recommended when angle regression is weak.
        match_low_quality: If True, force best proposal per GT to positive when IoU >= roi_min_pos_iou.
        roi_min_pos_iou: When match_low_quality is True, min IoU to force best proposal to positive (MMRotate: 0.5).
        label_smoothing: Label smoothing for classification loss (default: 0.0). Use e.g. 0.1 to reduce overconfidence.
        include_assignment_diagnostics: If True, include ``roi_*`` keys for logging (default True).
            Set False when the caller supplies RPN-only diagnostics (e.g. Rotated Faster R-CNN with ``add_gt_as_proposals``).
    
    Returns:
        Dictionary with loss values:
        - "loss_classifier": Classification loss
        - "loss_box_reg": Box regression loss (only on positive proposals)
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation.")
    
    device = class_logits.device
    
    # Match proposals to ground truth
    matched_labels, matched_gt_indices, matched_gt_boxes = match_oriented_proposals_to_gt(
        proposals,
        gt_boxes,
        gt_labels,
        positive_iou_threshold,
        negative_iou_threshold,
        device,
        use_hbb_for_matching=use_hbb_for_matching,
        match_low_quality=match_low_quality,
        min_pos_iou=roi_min_pos_iou,
        gt_boxes_ignore=gt_boxes_ignore,
        ignore_iou_threshold=positive_iou_threshold,
    )
    
    # Sample proposals for training (balance foreground/background)
    fg_indices = (matched_labels > 0).nonzero(as_tuple=True)[0]
    bg_indices = (matched_labels == 0).nonzero(as_tuple=True)[0]

    assign_stats = _roi_assignment_stats(
        matched_labels,
        matched_gt_indices,
        len(gt_boxes),
        fg_bg_sampling_ratio,
        batch_size_per_image,
    )
    matched_gt_count = int(assign_stats["roi_matched_gt"])
    match_rate = assign_stats["roi_match_rate"]
    num_fg = int(assign_stats["roi_sampled_fg"])
    num_bg = int(assign_stats["roi_sampled_bg"])

    # Log match statistics
    logger.trace(
        "ROI match stats: proposals={}, positive={}, negative={}, "
        "gt_boxes={}, matched_gt={}, match_rate={:.1%}, "
        "sampled_fg={}, sampled_bg={}",
        len(proposals), len(fg_indices), len(bg_indices),
        len(gt_boxes), matched_gt_count, match_rate,
        num_fg, num_bg
    )
    if len(fg_indices) == 0 and len(gt_boxes) > 0:
        logger.trace(
            "WARNING: No positive ROI matches found! "
            "IoU threshold may be too high (pos={}, neg={})",
            positive_iou_threshold, negative_iou_threshold
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
    
    # Classification loss (clone to break view relationship for memory efficiency)
    sampled_logits = class_logits[sampled_indices].clone()  # [sampled, num_classes + 1]
    sampled_labels = matched_labels[sampled_indices]  # [sampled]
    
    # Labels and logits are already aligned:
    # - Logits: index 0 = background, indices 1,2,... = object classes
    # - Labels: 0 = background, 1,2,... = object classes
    # No conversion needed - direct use for cross-entropy
    
    loss_classifier = roi_classification_loss(
        sampled_logits,
        sampled_labels,
        loss_type=loss_type,
        grouped_alpha=grouped_alpha,
        group_index_lists=group_index_lists,
        class_in_group_id=class_in_group_id,
        class_weights=class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    # Box regression loss (only on positive proposals)
    if len(sampled_fg) > 0:
        # Get matched boxes (detach - these are ground truth, not predictions)
        positive_proposals = proposals[sampled_fg].detach()
        positive_matched_gt = matched_gt_boxes[sampled_fg].detach()
        positive_labels = matched_labels[sampled_fg]
        
        # Get regression predictions (clone for memory efficiency)
        positive_regression = box_regression[sampled_fg].clone()
        
        # Handle both class-agnostic and class-specific regression
        if positive_regression.shape[1] == 5:
            # Class-agnostic regression (MMRotate format): [N, 5]
            selected_regression = positive_regression
        else:
            # Class-specific regression: [N, num_classes * 5] -> [N, num_classes, 5]
            positive_regression = positive_regression.view(len(sampled_fg), num_classes, 5)
            # Select regression for the predicted class (convert 1-indexed to 0-indexed)
            class_indices = (positive_labels - 1).long()
            selected_regression = positive_regression[torch.arange(len(sampled_fg), device=device), class_indices]
        
        # Compute targets with norm_factor and edge_swap (MMRotate ROI style)
        regression_targets = encode_oriented_boxes(
            positive_proposals, positive_matched_gt,
            target_means=target_means, target_stds=target_stds,
            norm_factor=norm_factor, edge_swap=edge_swap, proj_xy=proj_xy,
        )
        # Replace any nan/inf in targets (e.g. from degenerate boxes) so loss stays finite
        regression_targets = torch.where(
            torch.isfinite(regression_targets),
            regression_targets,
            torch.zeros_like(regression_targets, device=regression_targets.device),
        )

        # MMRotate: Smooth L1 on all 5 encoded channels; avg_factor over sampled RoIs.
        loss_box_reg = _smooth_l1_encoded_regression_loss(
            selected_regression,
            regression_targets,
            angle_weight=box_reg_angle_weight,
            reg_norm=_normalize_reg_norm(reg_norm),
            num_total_samples=max(1, int(len(sampled_indices))),
        )
        if box_reg_iou_weight > 0.0:
            decoded_boxes = decode_oriented_boxes(
                positive_proposals,
                selected_regression,
                target_means=target_means,
                target_stds=target_stds,
                normalize_le90=True,
                norm_factor=norm_factor,
                edge_swap=edge_swap,
                proj_xy=proj_xy,
            )
            loss_iou = mean_auxiliary_box_reg_loss(
                decoded_boxes,
                positive_matched_gt,
                loss_type=box_reg_iou_loss_type,
                kfiou_fun=box_reg_kfiou_fun,
                probiou_mode=box_reg_probiou_mode,
            )
            loss_box_reg = loss_box_reg + (box_reg_iou_weight * loss_iou)
    else:
        # Maintain gradient flow: compute zero loss from model outputs
        # Use box_regression to maintain connection to computation graph
        if box_regression.numel() > 0:
            loss_box_reg = (box_regression[0:1] * 0.0).sum()
        else:
            # Fallback: use class_logits if box_regression is empty
            loss_box_reg = (class_logits[0:1] * 0.0).sum() if class_logits.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    
    # Ensure loss tensors are finite (avoid nan/inf from edge cases)
    def _make_finite(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            return t
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t, device=t.device))

    loss_classifier = _make_finite(loss_classifier)
    loss_box_reg = _make_finite(box_reg_weight * loss_box_reg)

    out: Dict[str, Any] = {
        "loss_classifier": loss_classifier,
        "loss_box_reg": loss_box_reg,
    }
    if include_assignment_diagnostics:
        out.update(assign_stats)
    return out


def _normalize_reg_norm(reg_norm: str) -> RegNorm:
    norm = (reg_norm or "positives_only").strip().lower()
    if norm not in ("sampled_all", "positives_only"):
        raise ValueError(f"reg_norm must be 'sampled_all' or 'positives_only', got {reg_norm!r}")
    return norm  # type: ignore[return-value]


def _normalize_main_reg_loss_type(main_loss_type: str) -> MainRegLossType:
    lt = (main_loss_type or "smooth_l1").strip().lower()
    if lt not in ("smooth_l1", "probiou", "riou", "kfiou"):
        raise ValueError(
            f"main_loss_type must be smooth_l1|probiou|riou|kfiou, got {main_loss_type!r}"
        )
    return lt  # type: ignore[return-value]


def _smooth_l1_encoded_regression_loss(
    selected_regression: torch.Tensor,
    regression_targets: torch.Tensor,
    *,
    angle_weight: float = 1.0,
    reg_norm: RegNorm = "sampled_all",
    num_total_samples: int = 1,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 on all 5 encoded channels (MMRotate / MMDet L1Loss on bbox targets).

    Predictions and targets are already coder-normalized (edge_swap, norm_factor,
  stds applied at encode time). The 5th channel uses ``angle_weight`` as a scalar
    multiplier on its element losses.
    """
    loss_per_dim = F.smooth_l1_loss(
        selected_regression,
        regression_targets,
        beta=beta,
        reduction="none",
    )
    if angle_weight != 1.0:
        loss_per_dim = loss_per_dim.clone()
        loss_per_dim[:, 4] = loss_per_dim[:, 4] * angle_weight
    if reg_norm == "sampled_all":
        return loss_per_dim.sum() / float(max(1, num_total_samples))
    return loss_per_dim.mean()


def _horizontal_decoded_reg_loss(
    decoded_boxes: torch.Tensor,
    matched_gt: torch.Tensor,
    *,
    loss_type: str,
    kfiou_fun: Optional[str] = None,
    probiou_mode: Optional[str] = None,
) -> torch.Tensor:
    return mean_auxiliary_box_reg_loss(
        decoded_boxes,
        matched_gt,
        loss_type=loss_type,
        kfiou_fun=kfiou_fun,
        probiou_mode=probiou_mode,
    )


def compute_horizontal_roi_loss(
    *,
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    proposals_xyxy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes_ignore: Optional[torch.Tensor] = None,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.5,
    box_reg_weight: float = 1.0,
    box_reg_angle_weight: float = 1.0,
    main_loss_type: str = "smooth_l1",
    reg_norm: RegNorm = "positives_only",
    box_reg_iou_weight: float = 0.0,
    box_reg_iou_loss_type: str = "riou",
    box_reg_kfiou_fun: Optional[str] = None,
    box_reg_probiou_mode: Optional[str] = None,
    smooth_l1_aux_weight: float = 0.0,
    fg_bg_sampling_ratio: float = 0.25,
    batch_size_per_image: int = 512,
    num_classes: int = 1,
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    means: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, float, float, float, float] = (0.1, 0.1, 0.2, 0.2, 0.1),
    norm_factor: Optional[float] = 2.0,
    edge_swap: bool = True,
    proj_xy: bool = False,
    label_smoothing: float = 0.0,
    grouped_alpha: float = 0.0,
    group_index_lists: Optional[Sequence[Sequence[int]]] = None,
    class_in_group_id: Optional[torch.Tensor] = None,
    match_low_quality: bool = False,
    roi_min_pos_iou: float = 0.5,
    include_assignment_diagnostics: bool = True,
) -> Dict[str, Any]:
    """ROI loss for Rotated Faster R-CNN: horizontal proposals (xyxy) + 5D regression.

    - Proposals are axis-aligned xyxy.
    - Matching uses HBB-style IoU between ``xyxy_to_obb(proposals_xyxy)`` and ``gt_boxes``.
    - Regression targets use DeltaXYWHTHBBoxCoder math (see ``horizontal_roi_coder.py``).
    - ``main_loss_type``: ``smooth_l1`` (encoded deltas, default) or decoded ``probiou`` /
      ``riou`` / ``kfiou``.
    - ``reg_norm``: ``sampled_all`` (MMDet avg_factor: divide by pos+neg sample count) or
      ``positives_only`` (mean over positive RoIs).
    - When main is ``smooth_l1``, ``box_reg_iou_weight`` adds decoded aux loss
      (``box_reg_iou_loss_type``). When main is decoded, ``smooth_l1_aux_weight`` adds
      encoded Smooth L1 aux.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation.")
    device = class_logits.device

    # Convert proposals to obb(angle=0) for matching diagnostics/assignment.
    from .bbox_coder import xyxy_to_obb

    proposals_obb = xyxy_to_obb(proposals_xyxy).to(device)

    matched_labels, matched_gt_indices, matched_gt_boxes = match_oriented_proposals_to_gt(
        proposals_obb,
        gt_boxes,
        gt_labels,
        positive_iou_threshold,
        negative_iou_threshold,
        device,
        use_hbb_for_matching=True,
        match_low_quality=match_low_quality,
        min_pos_iou=roi_min_pos_iou,
        gt_boxes_ignore=gt_boxes_ignore,
        ignore_iou_threshold=positive_iou_threshold,
    )

    fg_indices = (matched_labels > 0).nonzero(as_tuple=True)[0]
    bg_indices = (matched_labels == 0).nonzero(as_tuple=True)[0]

    assign_stats = _roi_assignment_stats(
        matched_labels,
        matched_gt_indices,
        len(gt_boxes),
        fg_bg_sampling_ratio,
        batch_size_per_image,
    )
    num_fg = int(assign_stats["roi_sampled_fg"])
    num_bg = int(assign_stats["roi_sampled_bg"])

    if num_fg > 0:
        sampled_fg = fg_indices[torch.randperm(len(fg_indices), device=device)[:num_fg]]
    else:
        sampled_fg = torch.tensor([], dtype=torch.int64, device=device)
    if num_bg > 0:
        sampled_bg = bg_indices[torch.randperm(len(bg_indices), device=device)[:num_bg]]
    else:
        sampled_bg = torch.tensor([], dtype=torch.int64, device=device)
    sampled_indices = torch.cat([sampled_fg, sampled_bg], dim=0)

    sampled_logits = class_logits[sampled_indices].clone()
    sampled_labels = matched_labels[sampled_indices]
    loss_classifier = roi_classification_loss(
        sampled_logits,
        sampled_labels,
        loss_type=loss_type,
        grouped_alpha=grouped_alpha,
        group_index_lists=group_index_lists,
        class_in_group_id=class_in_group_id,
        class_weights=class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    if len(sampled_fg) > 0:
        positive_rois = proposals_xyxy[sampled_fg].detach()
        positive_matched_gt = matched_gt_boxes[sampled_fg].detach()
        positive_labels = matched_labels[sampled_fg]
        positive_roi_angle = (
            proposals_obb[sampled_fg, 4].detach() if proj_xy else None
        )

        positive_regression = box_regression[sampled_fg].clone()
        if positive_regression.shape[1] == 5:
            selected_regression = positive_regression
        else:
            positive_regression = positive_regression.view(len(sampled_fg), num_classes, 5)
            class_indices = (positive_labels - 1).long()
            selected_regression = positive_regression[torch.arange(len(sampled_fg), device=device), class_indices]

        regression_targets = encode_delta_xywh_th(
            positive_rois,
            positive_matched_gt,
            means=means,
            stds=stds,
            norm_factor=norm_factor,
            edge_swap=edge_swap,
            proj_xy=proj_xy,
            roi_angle=positive_roi_angle,
        )
        regression_targets = torch.where(
            torch.isfinite(regression_targets),
            regression_targets,
            torch.zeros_like(regression_targets, device=device),
        )
        num_total_samples = max(1, int(len(sampled_indices)))
        norm = _normalize_reg_norm(reg_norm)
        main_lt = _normalize_main_reg_loss_type(main_loss_type)
        smooth_l1_loss = _smooth_l1_encoded_regression_loss(
            selected_regression,
            regression_targets,
            angle_weight=box_reg_angle_weight,
            reg_norm=norm,
            num_total_samples=num_total_samples,
        )
        decoded_boxes = decode_delta_xywh_th(
            positive_rois,
            selected_regression,
            means=means,
            stds=stds,
            norm_factor=norm_factor,
            edge_swap=edge_swap,
            proj_xy=proj_xy,
            roi_angle=positive_roi_angle,
        )
        if main_lt == "smooth_l1":
            loss_box_reg = smooth_l1_loss
            if box_reg_iou_weight > 0.0:
                loss_iou = _horizontal_decoded_reg_loss(
                    decoded_boxes,
                    positive_matched_gt,
                    loss_type=box_reg_iou_loss_type,
                    kfiou_fun=box_reg_kfiou_fun,
                    probiou_mode=box_reg_probiou_mode,
                )
                loss_box_reg = loss_box_reg + (box_reg_iou_weight * loss_iou)
        else:
            loss_box_reg = _horizontal_decoded_reg_loss(
                decoded_boxes,
                positive_matched_gt,
                loss_type=main_lt,
                kfiou_fun=box_reg_kfiou_fun,
                probiou_mode=box_reg_probiou_mode,
            )
            if smooth_l1_aux_weight > 0.0:
                loss_box_reg = loss_box_reg + (smooth_l1_aux_weight * smooth_l1_loss)
    else:
        loss_box_reg = (box_regression[0:1] * 0.0).sum() if box_regression.numel() > 0 else (class_logits[0:1] * 0.0).sum()

    def _finite(t: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t, device=t.device))

    loss_classifier = _finite(loss_classifier)
    loss_box_reg = _finite(box_reg_weight * loss_box_reg)

    out: Dict[str, Any] = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    if include_assignment_diagnostics:
        out.update(assign_stats)
    return out


def compute_horizontal_roi_loss_mmrotate(
    **kwargs: Any,
) -> Dict[str, Any]:
    """MMRotate-style wrapper: ``reg_norm='sampled_all'`` (avg_factor over pos+neg)."""
    return compute_horizontal_roi_loss(**kwargs, reg_norm="sampled_all")


class OrientedROIHead(nn.Module if nn is not None else object):  # type: ignore
    """ROI head for oriented object detection (MMRotate RotatedShared2FCBBoxHead compatible).
    
    This head takes ROI features and predicts:
    - Object class scores
    - Oriented box refinements (5 parameters: cx, cy, w, h, angle)
    
    Structure matches MMRotate's RotatedShared2FCBBoxHead:
    - Two shared FC layers (fc6, fc7) with ReLU and optional dropout
    - Classification head (fc_cls)
    - Regression head (fc_reg) with class-agnostic or class-specific output
    
    Args:
        in_channels: Number of input channels from ROI features
        num_classes: Number of object classes (excluding background)
        representation_size: Size of the representation before classification/regression heads (default: 1024)
        class_agnostic_regression: If True, use class-agnostic regression (MMRotate default)
        dropout: Dropout probability for shared FC layers (default: 0.0, MMRotate typically uses 0.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        representation_size: int = 1024,
        class_agnostic_regression: bool = True,  # MMRotate format: class-agnostic regression
        dropout: float = 0.0,  # Dropout probability (MMRotate typically uses 0.0, but configurable)
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for OrientedROIHead.")
        
        super().__init__()
        self.num_classes = num_classes
        self.class_agnostic_regression = class_agnostic_regression
        
        # Shared representation layers (matches MMRotate's shared_fcs Sequential)
        # Structure: fc6 -> ReLU -> Dropout -> fc7 -> ReLU -> Dropout
        shared_fcs = []
        shared_fcs.append(nn.Linear(in_channels, representation_size))
        shared_fcs.append(nn.ReLU(inplace=True))
        if dropout > 0:
            shared_fcs.append(nn.Dropout(dropout))
        shared_fcs.append(nn.Linear(representation_size, representation_size))
        shared_fcs.append(nn.ReLU(inplace=True))
        if dropout > 0:
            shared_fcs.append(nn.Dropout(dropout))
        
        # Store as Sequential for MMRotate weight compatibility
        self.shared_fcs = nn.Sequential(*shared_fcs)
        
        # Also expose fc6 and fc7 individually for backward compatibility and weight loading
        # These reference the same layers as in shared_fcs
        self.fc6 = shared_fcs[0]
        # fc7 is at index 3 if dropout > 0 (fc6, ReLU, Dropout, fc7), else at index 2 (fc6, ReLU, fc7)
        self.fc7 = shared_fcs[3] if dropout > 0 else shared_fcs[2]
        
        # Classification head (MMRotate naming: fc_cls)
        # Output: num_classes + 1 (including background)
        self.fc_cls = nn.Linear(representation_size, num_classes + 1)
        # Keep cls_head as alias for backward compatibility
        self.cls_head = self.fc_cls
        
        # Box regression head (MMRotate naming: fc_reg)
        # MMRotate format: class-agnostic regression (5 params shared across all classes)
        # Standard format: class-specific regression (num_classes * 5 params)
        if class_agnostic_regression:
            # Class-agnostic: single set of 5 params for all classes (MMRotate format)
            reg_output_size = 5
        else:
            # Class-specific: 5 params per class
            reg_output_size = num_classes * 5
        self.fc_reg = nn.Linear(representation_size, reg_output_size)
        # Keep bbox_head as alias for backward compatibility
        self.bbox_head = self.fc_reg
        
        # Initialize weights (MMRotate style: std=0.01 for cls, std=0.001 for reg)
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        nn.init.normal_(self.fc_reg.weight, std=0.001)
        nn.init.constant_(self.fc_reg.bias, 0)
        # Shared FC layers use std=0.01 (MMRotate default)
        nn.init.normal_(self.fc6.weight, std=0.01)
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.normal_(self.fc7.weight, std=0.01)
        nn.init.constant_(self.fc7.bias, 0)
    
    def forward(self, roi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ROI head.
        
        Args:
            roi_features: ROI features of shape [N, in_channels]
                         (e.g., from ROI pooling/align)
        
        Returns:
            Tuple of (class_logits, box_regression):
            - class_logits: [N, num_classes + 1] classification logits
            - box_regression: [N, 5] (class-agnostic) or [N, num_classes * 5] (class-specific) box regression predictions
        """
        # Shared representation (using Sequential for MMRotate compatibility)
        x = self.shared_fcs(roi_features)
        
        # Classification (using fc_cls for MMRotate weight compatibility)
        class_logits = self.fc_cls(x)  # [N, num_classes + 1]
        
        # Box regression (using fc_reg for MMRotate weight compatibility)
        box_regression = self.fc_reg(x)  # [N, 5] (class-agnostic) or [N, num_classes * 5] (class-specific)
        
        return class_logits, box_regression


__all__ = [
    "oriented_roi_align",
    "OrientedROIAlign",
    "match_oriented_proposals_to_gt",
    "focal_loss",
    "grouped_cross_entropy_loss",
    "roi_classification_loss",
    "compute_roi_matching_diagnostics",
    "compute_oriented_roi_loss",
    "compute_horizontal_roi_loss",
    "compute_horizontal_roi_loss_mmrotate",
    "OrientedROIHead",
]

