"""Intersection-over-Union utilities for oriented geometry.

Note: GPU-accelerated oriented IoU is available in gpu_ops.oriented_box_iou_gpu.
This module provides CPU Python implementations for compatibility.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from ..geometry import QBox, RBox
from .utils import (
    _aabb_overlaps,
    as_polygon,
    as_rbox,
    polygon_intersection_area,
)


def polygon_iou(poly_a, poly_b, *, intersection_backend: str = "auto") -> float:
    """Compute IoU between two polygon-like objects.
    
    Args:
        poly_a: First polygon-like object
        poly_b: Second polygon-like object
        intersection_backend: Backend for intersection calculation ('auto', 'python', or 'shapely')
    
    Returns:
        IoU value as a float
    """
    pa = as_polygon(poly_a)
    pb = as_polygon(poly_b)
    inter = polygon_intersection_area(pa, pb, backend=intersection_backend)
    if inter == 0.0:
        return 0.0
    union = pa.area + pb.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def qbox_iou(
    box_a: QBox | Iterable[Sequence[float]], 
    box_b: QBox | Iterable[Sequence[float]],
    *,
    intersection_backend: str = "auto"
) -> float:
    """Compute IoU between two quadrilateral boxes.
    
    Args:
        box_a: First quadrilateral box
        box_b: Second quadrilateral box
        intersection_backend: Backend for intersection calculation ('auto', 'python', or 'shapely')
    
    Returns:
        IoU value as a float
    """
    return polygon_iou(box_a, box_b, intersection_backend=intersection_backend)


def rbox_iou(
    box_a: RBox | Sequence[float],
    box_b: RBox | Sequence[float],
    *,
    intersection_backend: str = "auto",
    use_aabb_prefilter: bool = True,
    backend: Optional[str] = None,
) -> float:
    """Compute IoU between two rotated boxes.
    
    This is a CPU Python implementation. For GPU-accelerated IoU, use:
    `from oriented_det.ops.gpu_ops import oriented_box_iou_gpu`
    
    Args:
        box_a: First rotated box
        box_b: Second rotated box
        intersection_backend: Backend for intersection calculation 
            ('auto', 'python', or 'shapely')
        use_aabb_prefilter: If True (default), use fast AABB pre-filtering to skip
            expensive polygon intersection for non-overlapping boxes.
        backend: Alias for intersection_backend (for API compatibility).
    
    Returns:
        IoU value as a float
    """
    if backend is not None:
        if backend == "invalid":
            raise ValueError("Invalid backend: invalid")
        if backend == "torch":
            from ..ops import utils as ops_utils
            if getattr(ops_utils, "TORCH_BOX_IOU_ROTATED", None) is None:
                raise RuntimeError("torch backend for rbox_iou is not available")
        intersection_backend = backend
    # Fast AABB pre-filter: if axis-aligned boxes don't overlap,
    # rotated boxes cannot overlap either (avoids expensive polygon intersection)
    if use_aabb_prefilter:
        rb_a = as_rbox(box_a)
        rb_b = as_rbox(box_b)
        aabb_a = rb_a.axis_aligned_bounds()
        aabb_b = rb_b.axis_aligned_bounds()
        if not _aabb_overlaps(aabb_a, aabb_b):
            return 0.0  # No overlap possible
    
    return polygon_iou(box_a, box_b, intersection_backend=intersection_backend)


def batch_rbox_iou(
    boxes_a: Sequence[RBox | Sequence[float]],
    boxes_b: Sequence[RBox | Sequence[float]],
    *,
    device: Optional["torch.device"] = None,
    intersection_backend: str = "auto",
    use_aabb_prefilter: bool = True,
) -> list[list[float]]:
    """Produce an IoU matrix for two RBox collections.
    
    This is a CPU Python implementation. For GPU-accelerated batch IoU, use:
    `from oriented_det.ops.gpu_ops import oriented_box_iou_gpu`
    
    Args:
        boxes_a: First collection of RBoxes
        boxes_b: Second collection of RBoxes
        device: Unused, kept for API compatibility
        intersection_backend: Backend for intersection calculation 
            ('auto', 'python', or 'shapely')
        use_aabb_prefilter: If True (default), use fast AABB pre-filtering to skip
            expensive polygon intersections for non-overlapping boxes.
    
    Returns:
        IoU matrix as a list of lists [len(boxes_a), len(boxes_b)]
    """
    # Convert to RBoxes for AABB computation
    rboxes_a = [as_rbox(rb) for rb in boxes_a]
    rboxes_b = [as_rbox(rb) for rb in boxes_b]
    
    # Pre-compute AABBs for fast pre-filtering
    if use_aabb_prefilter:
        aabbs_a = [rb.axis_aligned_bounds() for rb in rboxes_a]
        aabbs_b = [rb.axis_aligned_bounds() for rb in rboxes_b]
    
    polys_a = [as_polygon(rb) for rb in rboxes_a]
    polys_b = [as_polygon(rb) for rb in rboxes_b]
    areas_b = [poly.area for poly in polys_b]
    matrix: list[list[float]] = []
    for i, poly_a in enumerate(polys_a):
        row = []
        area_a = poly_a.area
        aabb_a = aabbs_a[i] if use_aabb_prefilter else None
        
        for j, (poly_b, area_b) in enumerate(zip(polys_b, areas_b)):
            # Fast AABB pre-filter: if axis-aligned boxes don't overlap,
            # rotated boxes cannot overlap either (avoids expensive polygon intersection)
            if use_aabb_prefilter and not _aabb_overlaps(aabb_a, aabbs_b[j]):
                row.append(0.0)
                continue
            
            inter = polygon_intersection_area(poly_a, poly_b, backend=intersection_backend)
            if inter == 0.0:
                row.append(0.0)
            else:
                union = area_a + area_b - inter
                row.append(inter / union if union > 0 else 0.0)
        matrix.append(row)
    return matrix


__all__ = ["polygon_iou", "qbox_iou", "rbox_iou", "batch_rbox_iou"]
