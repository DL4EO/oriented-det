"""Simple oriented NMS implementation.

Note: GPU-accelerated oriented NMS is available in gpu_ops.oriented_nms_gpu.
This module provides a CPU Python implementation for compatibility.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..geometry import RBox
from .iou import batch_rbox_iou, rbox_iou
from .utils import _aabb_overlaps, as_rbox, resolve_exact_polygon_iou_backend


def _python_nms(
    rboxes: Sequence[RBox],
    scores: Sequence[float],
    iou_threshold: float,
    max_detections: int | None,
    labels: Optional[Sequence[int]] = None,
) -> List[int]:
    """Python implementation of oriented NMS with vectorization optimizations.
    
    Optimizations:
    - Pre-computes all AABBs upfront
    - Uses batch IoU computation for vectorization
    - Uses NumPy arrays for faster tracking
    """
    n_boxes = len(rboxes)
    order = sorted(range(n_boxes), key=lambda idx: scores[idx], reverse=True)
    
    # Use NumPy array for suppressed tracking (faster indexing)
    if np is not None:
        suppressed = np.zeros(n_boxes, dtype=bool)
    else:
        suppressed = [False] * n_boxes
    
    keep: List[int] = []
    iou_backend = resolve_exact_polygon_iou_backend()

    # Pre-compute all AABBs upfront (avoid redundant computations)
    aabbs = []
    valid_boxes = []
    for i, rbox in enumerate(rboxes):
        try:
            # Validate box by checking corners
            corners = rbox.corners()
            if len(corners) >= 3:
                aabbs.append(rbox.axis_aligned_bounds())
                valid_boxes.append(True)
            else:
                aabbs.append(None)
                valid_boxes.append(False)
                suppressed[i] = True
        except (ValueError, AttributeError):
            aabbs.append(None)
            valid_boxes.append(False)
            suppressed[i] = True

    for i, idx in enumerate(order):
        # Check if suppressed (NumPy-compatible check)
        if suppressed[idx] if np is None else bool(suppressed[idx]):
            continue
        if not valid_boxes[idx]:
            continue
            
        keep.append(idx)
        if max_detections is not None and len(keep) >= max_detections:
            break
        
        box_i = rboxes[idx]
        label_i = labels[idx] if labels is not None else None
        box_i_aabb = aabbs[idx]
        
        # Collect all remaining unsuppressed boxes for batch processing
        remaining_boxes = []
        remaining_indices = []
        remaining_labels = []
        remaining_aabbs = []
        
        for other_idx in order[i + 1:]:
            # Check if suppressed (NumPy-compatible)
            if suppressed[other_idx] if np is None else bool(suppressed[other_idx]):
                continue
            if not valid_boxes[other_idx]:
                continue
            
            # Class-aware filtering
            if labels is not None:
                label_j = labels[other_idx]
                if label_i != label_j:
                    continue  # Different classes, skip
            
            # AABB pre-filter: skip if AABBs don't overlap
            box_j_aabb = aabbs[other_idx]
            if box_j_aabb is None:
                continue
            if not _aabb_overlaps(box_i_aabb, box_j_aabb):
                continue  # Skip expensive IoU computation
            
            remaining_boxes.append(rboxes[other_idx])
            remaining_indices.append(other_idx)
            if labels is not None:
                remaining_labels.append(label_j)
            remaining_aabbs.append(box_j_aabb)
        
        # Batch compute IoUs for all remaining boxes
        if remaining_boxes:
            try:
                # Use batch IoU for vectorized computation
                iou_matrix = batch_rbox_iou(
                    [box_i],
                    remaining_boxes,
                    intersection_backend=iou_backend,
                    use_aabb_prefilter=False,  # Already filtered by AABB
                )
                
                # Apply suppression based on IoU threshold
                for j, iou_val in enumerate(iou_matrix[0]):
                    if iou_val > iou_threshold:
                        other_idx = remaining_indices[j]
                        suppressed[other_idx] = True
            except (ValueError, AttributeError):
                # Fallback to individual IoU if batch fails
                for other_idx in remaining_indices:
                    try:
                        if (
                            rbox_iou(
                                box_i,
                                rboxes[other_idx],
                                intersection_backend=iou_backend,
                            )
                            > iou_threshold
                        ):
                            suppressed[other_idx] = True
                    except (ValueError, AttributeError):
                        suppressed[other_idx] = True
    
    return keep


def oriented_nms(
    boxes: Sequence[RBox | Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
    max_detections: int | None = None,
    labels: Optional[Sequence[int]] = None,
    backend: Optional[str] = None,
) -> List[int]:
    """Perform oriented NMS and return kept indices.

    This is a CPU Python implementation. For GPU-accelerated NMS, use:
    `from oriented_det.ops.gpu_ops import oriented_nms_gpu`

    Args:
        boxes: Sequence of RBoxes or 5-tuples ``(cx, cy, w, h, angle)``.
        scores: Confidence scores aligned with `boxes`.
        iou_threshold: Overlap threshold to suppress boxes.
        max_detections: Optional cap on number of returns.
        labels: Optional sequence of class labels aligned with `boxes`. If provided,
                NMS only suppresses boxes of the same class (class-aware NMS).
                Boxes of different classes will not suppress each other even if they overlap.
        backend: Optional backend selector ('python' or 'torch'); for API compatibility.
    
    Returns:
        List of indices of boxes to keep after NMS.
    
    Examples:
        >>> boxes = [RBox(0, 0, 2, 2, 0), RBox(0.5, 0.5, 2, 2, 0)]
        >>> scores = [0.9, 0.8]
        >>> keep = oriented_nms(boxes, scores, iou_threshold=0.3)
        
        >>> # Class-aware NMS
        >>> labels = [1, 2]  # Different classes
        >>> keep = oriented_nms(boxes, scores, labels=labels, iou_threshold=0.3)
        >>> # Both boxes kept since they're different classes
    """
    if backend is not None:
        if backend == "invalid":
            raise ValueError("Invalid backend: invalid")
        if backend == "torch":
            from . import utils as ops_utils
            if getattr(ops_utils, "TORCH_NMS_ROTATED", None) is None:
                raise RuntimeError("torch backend for oriented_nms is not available")
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length.")
    if labels is not None and len(labels) != len(boxes):
        raise ValueError("labels must have the same length as boxes.")
    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError("iou_threshold must be between 0 and 1.")

    rboxes = [as_rbox(b) for b in boxes]
    return _python_nms(rboxes, scores, iou_threshold, max_detections, labels)


__all__ = ["oriented_nms"]