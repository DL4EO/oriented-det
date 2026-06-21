"""GPU-accelerated operations for oriented bounding boxes.

This module provides fully vectorized, GPU-native implementations of:
- Oriented IoU computation (using sampling-based approximation)
- Oriented NMS  
- Anchor generation
- Anchor-to-GT matching

All operations are fully vectorized without Python loops, providing
significant speedup over CPU implementations.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.ops import nms as torchvision_nms


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _sample_by_max_side_enabled(env_var: str) -> bool:
    """Return whether geometry-based sample sizing is used for IoU/NMS.

    **Default: enabled** when the variable is unset or empty. Set to ``0`` /
    ``false`` / ``no`` / ``off`` to use a flat count from the corresponding
    ``*_IOU_SAMPLES`` env only. Set to ``1`` / ``true`` / ``yes`` / ``on`` to
    enable explicitly.
    """
    val = os.environ.get(env_var)
    if val is None or not val.strip():
        return True
    v = val.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return v in ("1", "true", "yes", "on")


def _read_env_perfect_square(name: str, default: int) -> int:
    """Read integer env; must be a perfect square (grid cell count per box)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        n = default
    else:
        try:
            n = int(raw.strip(), 10)
        except ValueError as e:
            raise ValueError(
                f"{name} must be a non-negative integer, got {raw!r}"
            ) from e
    if n < 1:
        raise ValueError(f"{name} must be >= 1, got {n}")
    root = int(math.sqrt(n))
    if root * root != n:
        raise ValueError(
            f"{name}={n} must be a perfect square "
            f"(e.g. 25, 36, 49, 64, 81, 100, 121, ...)."
        )
    return n


# Geometry-based sampling: grid side is capped so num_samples <= 1024 (32×32).
_MAX_GRID_SIDE = 32
_DEFAULT_TARGET_SPACING_PX = 2.0
_DEFAULT_MIN_SAMPLES = 25  # 5×5 floor (cars/trucks ~10–25 px)
_DEFAULT_MAX_SAMPLES = 1024  # 32×32
_MIN_POINTS_ALONG_SHORT_SIDE = 3
_FLAT_IOU_SAMPLES = 100  # when geometry sizing is disabled (debug)


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {raw!r}") from e
    if val <= 0.0:
        raise ValueError(f"{name} must be > 0, got {val}")
    return val


def _read_env_perfect_square_optional(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return _read_env_perfect_square(name, 100)


def _clamp_perfect_square(n: int, *, lo: int, hi: int) -> int:
    n = max(lo, min(hi, n))
    root = int(math.isqrt(n))
    if root * root == n:
        return n
    root = min(int(math.ceil(math.sqrt(n))), int(math.isqrt(hi)))
    return max(lo, root * root)


def _validate_explicit_num_samples(explicit: int) -> int:
    root = int(math.sqrt(explicit))
    if root * root != explicit:
        raise ValueError(
            f"num_samples must be a perfect square, got {explicit}. "
            f"Valid values include: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, etc."
        )
    return explicit


def _grid_side_for_box_wh(
    w: Tensor,
    h: Tensor,
    *,
    target_spacing_px: float,
    min_points_short: int,
    max_grid_side: int,
) -> Tensor:
    """Per-box square grid side from width/height (image px), aspect-ratio aware."""
    w = w.abs().clamp(min=1e-6)
    h = h.abs().clamp(min=1e-6)
    short = torch.minimum(w, h)
    long = torch.maximum(w, h)
    aspect = long / short

    nx = (0.99 * w / target_spacing_px).ceil().clamp(min=2.0)
    ny = (0.99 * h / target_spacing_px).ceil().clamp(min=2.0)

    short_target = torch.maximum(
        torch.full_like(short, target_spacing_px),
        short / float(min_points_short),
    )
    n_short = (0.99 * short / short_target).ceil().clamp(min=float(min_points_short))

    # Elongated boxes: tighten spacing along the long axis (ships, bridges, …).
    eff_long_spacing = target_spacing_px * torch.clamp(2.0 / aspect, min=0.5, max=1.0)
    n_long = (0.99 * long / eff_long_spacing).ceil().clamp(min=2.0)
    nx = torch.where(w >= h, torch.maximum(nx, n_long), nx)
    ny = torch.where(h >= w, torch.maximum(ny, n_long), ny)

    grid = torch.maximum(torch.maximum(nx, ny), n_short).ceil().to(torch.int64)
    return grid.clamp(min=2, max=max_grid_side)


def _geometry_sample_count_for_boxes(
    boxes: Tensor,
    *,
    target_spacing_px: float,
    min_samples: int,
    max_samples: int,
    min_points_short: int = _MIN_POINTS_ALONG_SHORT_SIDE,
) -> int:
    """Sample count (perfect square) from box geometry; one grid for the whole batch."""
    if boxes.numel() == 0:
        return min_samples
    w = boxes[:, 2]
    h = boxes[:, 3]
    max_grid = int(math.isqrt(max_samples))
    max_grid = max(2, max_grid)
    grid_side = int(
        _grid_side_for_box_wh(
            w,
            h,
            target_spacing_px=target_spacing_px,
            min_points_short=min_points_short,
            max_grid_side=max_grid,
        ).max().item()
    )
    count = grid_side * grid_side
    return _clamp_perfect_square(count, lo=min_samples, hi=max_samples)


def geometry_sample_count_for_boxes(
    boxes: Tensor,
    *,
    target_spacing_px: float = _DEFAULT_TARGET_SPACING_PX,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    min_points_short: int = _MIN_POINTS_ALONG_SHORT_SIDE,
) -> int:
    """Public geometry resolver (for benchmarks / tuning). Returns a perfect square."""
    return _geometry_sample_count_for_boxes(
        boxes,
        target_spacing_px=target_spacing_px,
        min_samples=min_samples,
        max_samples=max_samples,
        min_points_short=min_points_short,
    )


def _resolve_oriented_iou_sample_count(
    boxes1: Tensor,
    boxes2: Tensor,
    explicit: Optional[int],
) -> int:
    if explicit is not None:
        return _validate_explicit_num_samples(explicit)

    if not _sample_by_max_side_enabled("ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE"):
        return _FLAT_IOU_SAMPLES

    target_px = _read_env_float(
        "ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX", _DEFAULT_TARGET_SPACING_PX
    )
    min_samples = _read_env_perfect_square(
        "ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES", _DEFAULT_MIN_SAMPLES
    )
    max_samples = _read_env_perfect_square(
        "ORIENTED_DET_GPU_ORIENTED_IOU_MAX_SAMPLES", _DEFAULT_MAX_SAMPLES
    )
    combined = boxes1 if boxes2 is boxes1 else torch.cat([boxes1, boxes2], dim=0)
    return _geometry_sample_count_for_boxes(
        combined,
        target_spacing_px=target_px,
        min_samples=min_samples,
        max_samples=max_samples,
    )


def _iou_num_samples(
    boxes1: Tensor,
    boxes2: Tensor,
    explicit: Optional[int],
) -> int:
    return _resolve_oriented_iou_sample_count(boxes1, boxes2, explicit)


def _nms_num_samples_for_boxes(boxes: Tensor) -> int:
    flat_env = _read_env_perfect_square_optional("ORIENTED_DET_GPU_NMS_IOU_SAMPLES")
    if not _sample_by_max_side_enabled("ORIENTED_DET_GPU_NMS_IOU_SAMPLE_BY_MAX_SIDE"):
        return flat_env if flat_env is not None else _FLAT_IOU_SAMPLES

    target_px = _read_env_float(
        "ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX", _DEFAULT_TARGET_SPACING_PX
    )
    min_samples = _read_env_perfect_square(
        "ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES", _DEFAULT_MIN_SAMPLES
    )
    max_samples = _read_env_perfect_square(
        "ORIENTED_DET_GPU_ORIENTED_IOU_MAX_SAMPLES", _DEFAULT_MAX_SAMPLES
    )
    return _geometry_sample_count_for_boxes(
        boxes,
        target_spacing_px=target_px,
        min_samples=min_samples,
        max_samples=max_samples,
    )


def resolve_oriented_iou_sample_count(
    boxes1: Tensor,
    boxes2: Tensor,
    *,
    num_samples: Optional[int] = None,
) -> int:
    """Return the sampling grid size (perfect square) for ``oriented_box_iou_gpu``.

    Public helper for configs, tests, and debugging. See ``_resolve_oriented_iou_sample_count``.
    """
    return _resolve_oriented_iou_sample_count(boxes1, boxes2, num_samples)


@torch.jit.script
def _box_vertices(boxes: Tensor) -> Tensor:
    """Compute the 4 corner vertices of oriented boxes (JIT compiled).
    
    Args:
        boxes: Tensor of shape [N, 5] with format [cx, cy, w, h, angle]
    
    Returns:
        Vertices tensor of shape [N, 4, 2] with corner coordinates
    """
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    angle = boxes[:, 4]
    
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Half dimensions
    hw = w / 2.0
    hh = h / 2.0
    
    # Local corner offsets (before rotation)
    # Order: top-left, top-right, bottom-right, bottom-left
    dx = torch.stack([-hw, hw, hw, -hw], dim=-1)  # [N, 4]
    dy = torch.stack([-hh, -hh, hh, hh], dim=-1)  # [N, 4]
    
    # Rotate corners
    cos_a_exp = cos_a.unsqueeze(-1)  # [N, 1]
    sin_a_exp = sin_a.unsqueeze(-1)  # [N, 1]
    rot_dx = dx * cos_a_exp - dy * sin_a_exp  # [N, 4]
    rot_dy = dx * sin_a_exp + dy * cos_a_exp  # [N, 4]
    
    # Translate to center
    vx = cx.unsqueeze(-1) + rot_dx  # [N, 4]
    vy = cy.unsqueeze(-1) + rot_dy  # [N, 4]
    
    # Stack to [N, 4, 2]
    vertices = torch.stack([vx, vy], dim=-1)
    return vertices


# NOTE: Removed @torch.jit.script to force fresh compilation after fix
def _point_in_box(points: Tensor, box_vertices: Tensor) -> Tensor:
    """Check if points are inside a convex polygon (box) using cross products.
    
    Fully vectorized, no Python loops.
    Updated: v2 - supports both CW and CCW winding orders.
    
    Args:
        points: [P, 2] points to check
        box_vertices: [4, 2] vertices of a single box
    
    Returns:
        [P] boolean tensor, True if point is inside
    """
    P = points.shape[0]
    
    # Get edges: v[i] -> v[(i+1)%4]
    v0 = box_vertices[0]  # [2]
    v1 = box_vertices[1]  # [2]
    v2 = box_vertices[2]  # [2]
    v3 = box_vertices[3]  # [2]
    
    # Edge vectors
    e0 = v1 - v0  # [2]
    e1 = v2 - v1  # [2]
    e2 = v3 - v2  # [2]
    e3 = v0 - v3  # [2]
    
    # Vector from vertex to each point
    d0 = points - v0  # [P, 2]
    d1 = points - v1  # [P, 2]
    d2 = points - v2  # [P, 2]
    d3 = points - v3  # [P, 2]
    
    # Cross products (2D cross product = x1*y2 - y1*x2)
    cross0 = e0[0] * d0[:, 1] - e0[1] * d0[:, 0]  # [P]
    cross1 = e1[0] * d1[:, 1] - e1[1] * d1[:, 0]  # [P]
    cross2 = e2[0] * d2[:, 1] - e2[1] * d2[:, 0]  # [P]
    cross3 = e3[0] * d3[:, 1] - e3[1] * d3[:, 0]  # [P]
    
    # Point is inside if all cross products have same sign
    # Works for both CCW (all >= 0) and CW (all <= 0) winding
    inside_ccw = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0)
    inside_cw = (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0)
    inside = inside_ccw | inside_cw
    return inside


# NOTE: Removed @torch.jit.script to force fresh compilation after fix
# The JIT cache was retaining the old buggy version
def _points_in_boxes_batch(points: Tensor, boxes_vertices: Tensor) -> Tensor:
    """Check if points are inside multiple boxes.
    
    Works for both clockwise and counter-clockwise vertex winding.
    Updated: v2 - supports both CW and CCW winding orders.
    
    Args:
        points: [P, 2] points to check
        boxes_vertices: [M, 4, 2] vertices of M boxes
    
    Returns:
        [P, M] boolean tensor
    """
    P = points.shape[0]
    M = boxes_vertices.shape[0]
    
    # Expand points: [P, 1, 2] and boxes: [1, M, 4, 2]
    points_exp = points.unsqueeze(1)  # [P, 1, 2]
    
    # Get all vertices
    v0 = boxes_vertices[:, 0, :]  # [M, 2]
    v1 = boxes_vertices[:, 1, :]  # [M, 2]
    v2 = boxes_vertices[:, 2, :]  # [M, 2]
    v3 = boxes_vertices[:, 3, :]  # [M, 2]
    
    # Edge vectors [M, 2]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v0 - v3
    
    # Expand vertices for broadcasting: [1, M, 2]
    v0_exp = v0.unsqueeze(0)
    v1_exp = v1.unsqueeze(0)
    v2_exp = v2.unsqueeze(0)
    v3_exp = v3.unsqueeze(0)
    
    # Vector from vertex to each point: [P, M, 2]
    d0 = points_exp - v0_exp
    d1 = points_exp - v1_exp
    d2 = points_exp - v2_exp
    d3 = points_exp - v3_exp
    
    # Cross products [P, M]
    e0_exp = e0.unsqueeze(0)  # [1, M, 2]
    e1_exp = e1.unsqueeze(0)
    e2_exp = e2.unsqueeze(0)
    e3_exp = e3.unsqueeze(0)
    
    cross0 = e0_exp[:, :, 0] * d0[:, :, 1] - e0_exp[:, :, 1] * d0[:, :, 0]
    cross1 = e1_exp[:, :, 0] * d1[:, :, 1] - e1_exp[:, :, 1] * d1[:, :, 0]
    cross2 = e2_exp[:, :, 0] * d2[:, :, 1] - e2_exp[:, :, 1] * d2[:, :, 0]
    cross3 = e3_exp[:, :, 0] * d3[:, :, 1] - e3_exp[:, :, 1] * d3[:, :, 0]
    
    # Point is inside if all cross products have the same sign
    # This works for both CCW (all >= 0) and CW (all <= 0) winding
    # The box vertices from _box_vertices are in CW order (in image coords)
    inside_ccw = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0)
    inside_cw = (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0)
    inside = inside_ccw | inside_cw
    return inside


def _points_in_paired_boxes(points: Tensor, boxes_vertices: Tensor) -> Tensor:
    """Check if each pair's points are inside that pair's box (elementwise over pairs).

    Unlike ``_points_in_boxes_batch`` (all points vs all boxes), this tests the
    points of pair p only against the box of pair p, so the cost is O(P*S)
    instead of O(P*S*M).

    Args:
        points: [P, S, 2] sample points (S samples per pair)
        boxes_vertices: [P, 4, 2] vertices of one box per pair

    Returns:
        [P, S] boolean tensor
    """
    v0 = boxes_vertices[:, 0, :].unsqueeze(1)  # [P, 1, 2]
    v1 = boxes_vertices[:, 1, :].unsqueeze(1)
    v2 = boxes_vertices[:, 2, :].unsqueeze(1)
    v3 = boxes_vertices[:, 3, :].unsqueeze(1)

    e0 = v1 - v0  # [P, 1, 2]
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v0 - v3

    d0 = points - v0  # [P, S, 2]
    d1 = points - v1
    d2 = points - v2
    d3 = points - v3

    cross0 = e0[..., 0] * d0[..., 1] - e0[..., 1] * d0[..., 0]  # [P, S]
    cross1 = e1[..., 0] * d1[..., 1] - e1[..., 1] * d1[..., 0]
    cross2 = e2[..., 0] * d2[..., 1] - e2[..., 1] * d2[..., 0]
    cross3 = e3[..., 0] * d3[..., 1] - e3[..., 1] * d3[..., 0]

    inside_ccw = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0)
    inside_cw = (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0)
    return inside_ccw | inside_cw


def _generate_box_samples(boxes: Tensor, num_samples: int = 16) -> Tensor:
    """Generate sample points inside oriented boxes.
    
    Args:
        boxes: [N, 5] boxes in format [cx, cy, w, h, angle]
        num_samples: Number of samples per box (must be a perfect square)
    
    Returns:
        [N, num_samples, 2] sample points
    
    Raises:
        ValueError: If num_samples is not a perfect square
    """
    device = boxes.device
    N = boxes.shape[0]
    
    # Validate that num_samples is a perfect square
    sqrt_samples = int(math.sqrt(num_samples))
    if sqrt_samples * sqrt_samples != num_samples:
        raise ValueError(
            f"num_samples must be a perfect square, got {num_samples}. "
            f"Valid values include: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, etc."
        )
    
    # Grid in local box coords: ±0.495 of half-extent per axis (99% coverage;
    # slightly inset from ±0.5 to reduce corner-only numerical noise).
    lin = torch.linspace(-0.495, 0.495, sqrt_samples, device=device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_samples, 2]
    
    # Scale by box dimensions
    cx = boxes[:, 0:1]  # [N, 1]
    cy = boxes[:, 1:2]  # [N, 1]
    w = boxes[:, 2:3]   # [N, 1]
    h = boxes[:, 3:4]   # [N, 1]
    angle = boxes[:, 4]  # [N]
    
    # Scale grid to box size: [N, num_samples, 2]
    samples_local = grid.unsqueeze(0) * torch.stack([w, h], dim=-1)  # [N, num_samples, 2]
    
    # Rotate samples
    cos_a = torch.cos(angle).view(N, 1, 1)  # [N, 1, 1]
    sin_a = torch.sin(angle).view(N, 1, 1)  # [N, 1, 1]
    
    x_local = samples_local[:, :, 0:1]  # [N, num_samples, 1]
    y_local = samples_local[:, :, 1:2]  # [N, num_samples, 1]
    
    x_rot = x_local * cos_a - y_local * sin_a
    y_rot = x_local * sin_a + y_local * cos_a
    
    samples_rot = torch.cat([x_rot, y_rot], dim=-1)  # [N, num_samples, 2]
    
    # Translate to box center
    center = torch.stack([cx, cy], dim=-1).squeeze(-2)  # [N, 2]
    samples = samples_rot + center.unsqueeze(1)  # [N, num_samples, 2]
    
    return samples


def oriented_box_iou_gpu(
    boxes1: Tensor,
    boxes2: Tensor,
    num_samples: Optional[int] = None,
) -> Tensor:
    """Compute oriented IoU matrix using sampling-based approximation.
    
    This is a fully vectorized GPU implementation that approximates IoU
    by sampling points inside boxes and checking containment.
    
    Args:
        boxes1: [N, 5] boxes in format [cx, cy, w, h, angle]
        boxes2: [M, 5] boxes in format [cx, cy, w, h, angle]
        num_samples: Samples per box (perfect square, e.g. 100 = 10×10).
            If ``None``, sample count follows geometry-based sizing (default):
            grid side from each box ``w``/``h`` and aspect ratio with target spacing
            ``ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX`` (default ``2`` px),
            clamped to ``[ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES, ORIENTED_DET_GPU_ORIENTED_IOU_MAX_SAMPLES]``
            (defaults ``25`` … ``1024``). Disable geometry with ``ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE=0``
            (flat ``100``-sample grid, debug only).
    
    Returns:
        [N, M] IoU matrix
    """
    device = boxes1.device
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), device=device)

    num_samples = _iou_num_samples(boxes1, boxes2, num_samples)
    
    # Compute areas
    area1 = boxes1[:, 2] * boxes1[:, 3]  # [N]
    area2 = boxes2[:, 2] * boxes2[:, 3]  # [M]
    
    # AABB pre-filtering
    verts1 = _box_vertices(boxes1)  # [N, 4, 2]
    verts2 = _box_vertices(boxes2)  # [M, 4, 2]
    
    aabb1_min = verts1.min(dim=1).values  # [N, 2]
    aabb1_max = verts1.max(dim=1).values  # [N, 2]
    aabb2_min = verts2.min(dim=1).values  # [M, 2]
    aabb2_max = verts2.max(dim=1).values  # [M, 2]
    
    # Check AABB overlap
    aabb_overlap = (
        (aabb1_min[:, 0:1] <= aabb2_max[None, :, 0]) &
        (aabb1_max[:, 0:1] >= aabb2_min[None, :, 0]) &
        (aabb1_min[:, 1:2] <= aabb2_max[None, :, 1]) &
        (aabb1_max[:, 1:2] >= aabb2_min[None, :, 1])
    )  # [N, M]
    
    # Initialize IoU matrix
    iou_matrix = torch.zeros((N, M), device=device)
    
    # Find overlapping pairs
    overlap_mask = aabb_overlap
    if not overlap_mask.any():
        return iou_matrix
    
    # Generate samples for boxes1
    samples1 = _generate_box_samples(boxes1, num_samples)  # [N, S, 2]
    S = samples1.shape[1]
    
    # Also generate samples for boxes2 (for symmetric IoU)
    samples2 = _generate_box_samples(boxes2, num_samples)  # [M, S, 2]
    
    # For each box in boxes1, count how many of its samples are in each box of boxes2.
    # This is done in chunks to manage memory. _points_in_boxes_batch builds [P, M, 2]
    # float tensors; with P = chunk_n*S and M = num boxes, we need chunk_n*S*M*2*4 bytes.
    # Cap so that allocation stays under ~1.5 GB to avoid CUDA OOM when M is large.
    target_max_bytes = 1.5 * (1024 ** 3)
    max_elements = int(target_max_bytes / (4 * 2))  # float32, 2 coords per element
    max_inner_chunk = max(1, max_elements // (S * M))
    chunk_size = min(1000, max_inner_chunk)
    
    # Track total samples inside for debugging
    _total_samples_in_box2 = 0
    _total_samples_in_box1 = 0
    _num_chunks = 0
    
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_samples = samples1[i_start:i_end]  # [chunk, S, 2]
        chunk_n = chunk_samples.shape[0]
        
        # Flatten samples: [chunk * S, 2]
        flat_samples = chunk_samples.view(-1, 2)
        
        # Check which samples are in each box2: [chunk * S, M]
        in_boxes2 = _points_in_boxes_batch(flat_samples, verts2)
        
        # Reshape: [chunk, S, M]
        in_boxes2 = in_boxes2.view(chunk_n, S, M)
        
        # Count samples in intersection: [chunk, M]
        count_in_box2 = in_boxes2.sum(dim=1).float()
        _total_samples_in_box2 += int(count_in_box2.sum().item())
        
        # Approximate intersection area from boxes1 perspective
        # intersection ≈ (count_in_box2 / S) * area1
        inter_approx1 = (count_in_box2 / S) * area1[i_start:i_end, None]
        
        # Also compute intersection from boxes2 perspective for symmetry
        # For each box2, check how many of its samples are in boxes1[i_start:i_end]
        flat_samples2 = samples2.view(-1, 2)  # [M * S, 2]
        in_boxes1_chunk = _points_in_boxes_batch(flat_samples2, verts1[i_start:i_end])  # [M*S, chunk]
        in_boxes1_chunk = in_boxes1_chunk.view(M, S, chunk_n)  # [M, S, chunk]
        count_in_box1 = in_boxes1_chunk.sum(dim=1).float().T  # [chunk, M]
        _total_samples_in_box1 += int(count_in_box1.sum().item())
        
        inter_approx2 = (count_in_box1 / S) * area2[None, :]
        
        # Use MAX of both estimates (not geometric mean)
        # Geometric mean fails when one box is much smaller than the other:
        # - Small proposal inside large GT: inter_approx1 is good, inter_approx2 is ~0
        # - sqrt(good * 0) = 0, which is wrong!
        # MAX gives a better estimate in asymmetric size scenarios
        inter_approx = torch.maximum(inter_approx1, inter_approx2)
        
        # Clamp intersection to theoretical maximum (can't exceed smaller box area)
        # This prevents IoU > 1.0 due to sampling approximation errors
        max_possible_inter = torch.minimum(area1[i_start:i_end, None], area2[None, :])
        inter_approx = torch.minimum(inter_approx, max_possible_inter)
        
        _num_chunks += 1
        
        # Compute union (ensure union >= intersection to prevent IoU > 1.0)
        union_approx = area1[i_start:i_end, None] + area2[None, :] - inter_approx
        union_approx = torch.maximum(union_approx, inter_approx + 1e-8)
        
        # Compute IoU
        iou_chunk = inter_approx / (union_approx + 1e-8)
        
        # Final clamp to [0, 1] to handle any remaining numerical errors
        iou_chunk = torch.clamp(iou_chunk, 0.0, 1.0)
        
        # Apply AABB mask
        iou_chunk = iou_chunk * overlap_mask[i_start:i_end].float()
        
        iou_matrix[i_start:i_end] = iou_chunk
    
    return iou_matrix


def generate_oriented_anchors_gpu(
    image_size: Tuple[int, int],
    feature_map_sizes: List[Tuple[int, int]],
    anchor_scales: List[float],
    anchor_ratios: List[float],
    anchor_angles: List[float],
    stride_per_level: List[int],
    device: torch.device = torch.device('cpu'),
    octave_base_scale: Optional[float] = None,
    scales_per_octave: Optional[int] = None,
) -> List[Tensor]:
    """Generate oriented anchors using vectorized GPU operations.
    
    Fully vectorized, no Python loops for anchor generation.
    
    Args:
        image_size: (height, width) of input image
        feature_map_sizes: List of (height, width) for each FPN level
        anchor_scales: List of anchor scales (one per level or shared)
        anchor_ratios: List of aspect ratios
        anchor_angles: List of anchor angles in radians
        stride_per_level: List of strides for each FPN level
        device: Device to create tensors on
    
    Returns:
        List of anchor tensors, each [H*W*A, 5] in format [cx, cy, w, h, angle]
    """
    anchors_per_level = []
    
    ratios = torch.tensor(anchor_ratios, device=device, dtype=torch.float32)
    angles = torch.tensor(anchor_angles, device=device, dtype=torch.float32)
    
    for level_idx, (feat_h, feat_w) in enumerate(feature_map_sizes):
        stride = stride_per_level[level_idx]
        
        # Grid of centers
        shift_x = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # Flatten: [H*W]
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        num_positions = grid_x.shape[0]
        
        # Anchor dimensions
        sqrt_ratios = torch.sqrt(ratios)
        if octave_base_scale is not None and scales_per_octave is not None:
            factors = torch.tensor(
                [2.0 ** (i / float(scales_per_octave)) for i in range(int(scales_per_octave))],
                device=device,
                dtype=torch.float32,
            )
            base_sizes = stride * float(octave_base_scale) * factors  # [S]
            base_w = (base_sizes.unsqueeze(1) * sqrt_ratios.unsqueeze(0)).reshape(-1)
            base_h = (base_sizes.unsqueeze(1) / sqrt_ratios.unsqueeze(0)).reshape(-1)
        else:
            scale = anchor_scales[level_idx] if level_idx < len(anchor_scales) else anchor_scales[-1]
            effective_scale = scale * stride
            base_w = effective_scale * sqrt_ratios  # [R]
            base_h = effective_scale / sqrt_ratios  # [R]
        
        num_angles = len(angles)
        num_templates = base_w.numel()
        A = num_templates * num_angles
        
        # Expand templates x angles -> [A]
        base_w_exp = base_w.unsqueeze(1).expand(-1, num_angles).reshape(-1)
        base_h_exp = base_h.unsqueeze(1).expand(-1, num_angles).reshape(-1)
        angles_exp = angles.unsqueeze(0).expand(num_templates, -1).reshape(-1)
        
        # Create all anchors: [num_positions, A, 5]
        cx_all = grid_x.unsqueeze(1).expand(-1, A)  # [P, A]
        cy_all = grid_y.unsqueeze(1).expand(-1, A)  # [P, A]
        w_all = base_w_exp.unsqueeze(0).expand(num_positions, -1)  # [P, A]
        h_all = base_h_exp.unsqueeze(0).expand(num_positions, -1)  # [P, A]
        angle_all = angles_exp.unsqueeze(0).expand(num_positions, -1)  # [P, A]
        
        anchors = torch.stack([cx_all, cy_all, w_all, h_all, angle_all], dim=-1)  # [P, A, 5]
        anchors = anchors.view(-1, 5).detach()  # [P*A, 5]
        anchors.requires_grad_(False)
        
        anchors_per_level.append(anchors)
    
    return anchors_per_level


def obb_to_xyxy_gpu(boxes: Tensor) -> Tensor:
    """Convert oriented boxes [N, 5] (cx, cy, w, h, angle) to axis-aligned xyxy [N, 4].
    
    Uses the 4 corner vertices of each OBB and takes min/max to get the HBB.
    Fully vectorized and GPU-compatible.
    
    Args:
        boxes: [N, 5] in format [cx, cy, w, h, angle]
    
    Returns:
        [N, 4] in format [x1, y1, x2, y2]
    """
    verts = _box_vertices(boxes)  # [N, 4, 2]
    xy_min = verts.min(dim=1).values   # [N, 2]
    xy_max = verts.max(dim=1).values   # [N, 2]
    return torch.cat([xy_min, xy_max], dim=1)  # [N, 4]


def hbb_iou_gpu(boxes1_xyxy: Tensor, boxes2_xyxy: Tensor) -> Tensor:
    """Compute axis-aligned (horizontal) IoU matrix between two sets of xyxy boxes.
    
    Args:
        boxes1_xyxy: [N, 4] (x1, y1, x2, y2)
        boxes2_xyxy: [M, 4] (x1, y1, x2, y2)
    
    Returns:
        [N, M] IoU matrix
    """
    N = boxes1_xyxy.shape[0]
    M = boxes2_xyxy.shape[0]
    device = boxes1_xyxy.device
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=device)
    # Broadcast: [N, 1, 4] vs [1, M, 4]
    x1_1 = boxes1_xyxy[:, 0].unsqueeze(1)   # [N, 1]
    y1_1 = boxes1_xyxy[:, 1].unsqueeze(1)
    x2_1 = boxes1_xyxy[:, 2].unsqueeze(1)
    y2_1 = boxes1_xyxy[:, 3].unsqueeze(1)
    x1_2 = boxes2_xyxy[:, 0].unsqueeze(0)   # [1, M]
    y1_2 = boxes2_xyxy[:, 1].unsqueeze(0)
    x2_2 = boxes2_xyxy[:, 2].unsqueeze(0)
    y2_2 = boxes2_xyxy[:, 3].unsqueeze(0)
    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x2 = torch.minimum(x2_1, x2_2)
    inter_y2 = torch.minimum(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # [N, 1]
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # [1, M]
    union = area1 + area2 - inter_area
    iou = inter_area / (union + 1e-8)
    return iou


def oriented_box_hbb_iou_gpu(anchors: Tensor, gt_boxes: Tensor) -> Tensor:
    """Compute HBB (axis-aligned) IoU matrix between oriented boxes.
    
    Each OBB is converted to its axis-aligned bounding box (min/max of corners),
    then standard xyxy IoU is computed. Fully GPU-compatible.
    
    Args:
        anchors: [N, 5] (cx, cy, w, h, angle)
        gt_boxes: [M, 5] (cx, cy, w, h, angle)
    
    Returns:
        [N, M] IoU matrix
    """
    xyxy_anchors = obb_to_xyxy_gpu(anchors)
    xyxy_gt = obb_to_xyxy_gpu(gt_boxes)
    return hbb_iou_gpu(xyxy_anchors, xyxy_gt)


def match_anchors_to_gt_gpu(
    anchors: Tensor,
    gt_boxes: Tensor,
    positive_iou_threshold: float = 0.7,
    negative_iou_threshold: float = 0.3,
    use_hbb_for_assignment: bool = False,
    min_pos_iou: float = 0.3,
    match_low_quality: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Match anchors to GT using GPU-accelerated IoU.
    
    Args:
        anchors: [N, 5] anchors
        gt_boxes: [M, 5] ground truth boxes
        positive_iou_threshold: IoU threshold for positive
        negative_iou_threshold: IoU threshold for negative
        use_hbb_for_assignment: If True, use axis-aligned (HBB) overlap for assignment
                               instead of oriented IoU. Each box is converted to its
                               horizontal bounding box (min/max of corners) then
                               standard IoU is computed. Can yield more positive
                               anchors when GT is rotated.
    
    Returns:
        labels: [N] with -1=ignore, 0=background, 1=foreground
        matched_gt_indices: [N] index of matched GT (-1 if none)
    """
    device = anchors.device
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    
    labels = torch.full((N,), -1, dtype=torch.long, device=device)
    matched_gt_indices = torch.full((N,), -1, dtype=torch.long, device=device)
    
    if M == 0:
        labels.fill_(0)
        return labels, matched_gt_indices
    
    if use_hbb_for_assignment:
        # HBB assignment only needs the best GT per anchor and best anchor per GT.
        # Process large chunks against all GTs at once so dense P2 grids do not
        # create thousands of tiny GPU launches or materialize a full N x M matrix.
        anchor_xyxy = obb_to_xyxy_gpu(anchors)
        gt_xyxy = obb_to_xyxy_gpu(gt_boxes)

        max_iou_per_anchor = torch.zeros((N,), device=device)
        best_gt_per_anchor = torch.zeros((N,), dtype=torch.long, device=device)
        max_iou_per_gt = torch.zeros((M,), device=device)
        best_anchor_per_gt = torch.zeros((M,), dtype=torch.long, device=device)

        target_pairs_per_chunk = 8_000_000
        chunk_size = max(1, min(N, target_pairs_per_chunk // max(1, M)))
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            ious = hbb_iou_gpu(anchor_xyxy[start:end], gt_xyxy)

            chunk_anchor_iou, chunk_best_gt = ious.max(dim=1)
            max_iou_per_anchor[start:end] = chunk_anchor_iou
            best_gt_per_anchor[start:end] = chunk_best_gt

            chunk_gt_iou, chunk_best_anchor = ious.max(dim=0)
            better_gt = chunk_gt_iou > max_iou_per_gt
            if better_gt.any():
                max_iou_per_gt[better_gt] = chunk_gt_iou[better_gt]
                best_anchor_per_gt[better_gt] = chunk_best_anchor[better_gt] + start

        # Mark anchors above positive threshold as positive
        positive_mask = max_iou_per_anchor >= positive_iou_threshold
        labels[positive_mask] = 1
        matched_gt_indices[positive_mask] = best_gt_per_anchor[positive_mask]

        # Best anchor per GT - mark as positive when match_low_quality and IoU >= min_pos_iou (MMRotate-style)
        best_anchor_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if match_low_quality:
            for gt_idx in range(M):
                if max_iou_per_gt[gt_idx] >= min_pos_iou:
                    anchor_idx = best_anchor_per_gt[gt_idx]
                    labels[anchor_idx] = 1
                    matched_gt_indices[anchor_idx] = gt_idx
                    best_anchor_mask[anchor_idx] = True

        # Mark anchors below negative threshold as negative
        negative_mask = (max_iou_per_anchor < negative_iou_threshold) & (~best_anchor_mask)
        labels[negative_mask] = 0

        return labels, matched_gt_indices
    else:
        # Oriented IoU in chunks to manage memory
        chunk_size = 5000
        iou_matrix = torch.zeros((N, M), device=device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            iou_matrix[start:end] = oriented_box_iou_gpu(anchors[start:end], gt_boxes)
    
    # Best GT per anchor
    max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)
    
    # Mark anchors above positive threshold as positive
    positive_mask = max_iou_per_anchor >= positive_iou_threshold
    labels[positive_mask] = 1
    matched_gt_indices[positive_mask] = best_gt_per_anchor[positive_mask]
    
    # Best anchor per GT - mark as positive when match_low_quality and IoU >= min_pos_iou (MMRotate-style)
    max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)
    best_anchor_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if match_low_quality:
        for gt_idx in range(M):
            if max_iou_per_gt[gt_idx] >= min_pos_iou:
                anchor_idx = best_anchor_per_gt[gt_idx]
                labels[anchor_idx] = 1
                matched_gt_indices[anchor_idx] = gt_idx
                best_anchor_mask[anchor_idx] = True
    
    # Mark anchors below negative threshold as negative
    # BUT protect "best anchor per GT" from being overwritten
    negative_mask = (max_iou_per_anchor < negative_iou_threshold) & (~best_anchor_mask)
    labels[negative_mask] = 0
    
    return labels, matched_gt_indices


# oriented_box_iou_gpu uses [P,M,2]-style batches; chunk_size scales down as S or M grows.
# With S=100 and M=4000, peak temps stay within the module's ~1.5 GiB target.
MAX_BOXES_FOR_NMS = 4000


def oriented_nms_gpu(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.5,
    max_detections: Optional[int] = None,
) -> Tensor:
    """GPU-accelerated oriented NMS using sampling-based IoU.

    Uses ``ORIENTED_DET_GPU_NMS_IOU_SAMPLES`` as an optional floor when geometry
    sizing is enabled; otherwise a flat count (default ``100``). Geometry uses the
    same rules as oriented IoU (see ``ops/README.md``).

    Args:
        boxes: [N, 5] boxes
        scores: [N] confidence scores
        iou_threshold: Suppression threshold
        max_detections: Maximum detections to keep

    Returns:
        Indices of kept boxes
    """
    device = boxes.device
    N = boxes.shape[0]
    
    if N == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    
    # Sort by score
    _, order = scores.sort(descending=True)
    
    # ALWAYS limit boxes before NxN IoU computation to prevent OOM.
    # Rotated RetinaNet can produce 100k+ proposals per class; full pairwise IoU would need ~1TB.
    max_candidates = (
        min(max_detections * 3, MAX_BOXES_FOR_NMS) if max_detections is not None
        else MAX_BOXES_FOR_NMS
    )
    order = order[:min(len(order), max_candidates)]
    
    ordered_boxes = boxes[order]
    M = ordered_boxes.shape[0]

    if M == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    # Candidate pair pruning: rotated IoU is only needed where it can plausibly exceed
    # the threshold. Upper-bound IoU with the AABB intersection over the rotated-box
    # union: iou_rot <= I_aabb / (area_i + area_j - I_aabb). A 0.5 safety factor on the
    # threshold absorbs sampling-grid noise in the estimate below.
    verts = _box_vertices(ordered_boxes)  # [M, 4, 2]
    xy_min = verts.min(dim=1).values
    xy_max = verts.max(dim=1).values
    inter_wh = (
        torch.minimum(xy_max.unsqueeze(1), xy_max.unsqueeze(0))
        - torch.maximum(xy_min.unsqueeze(1), xy_min.unsqueeze(0))
    ).clamp(min=0)  # [M, M, 2]
    inter_aabb = inter_wh[..., 0] * inter_wh[..., 1]  # [M, M]
    areas = ordered_boxes[:, 2] * ordered_boxes[:, 3]  # [M]
    union_lb = (areas.unsqueeze(1) + areas.unsqueeze(0) - inter_aabb).clamp(min=1e-8)
    iou_upper_bound = inter_aabb / union_lb
    # Strictly upper triangular: boxes are score-sorted, suppression flows i -> j (i < j).
    cand = torch.triu(iou_upper_bound > iou_threshold * 0.5, diagonal=1)
    pair_i, pair_j = cand.nonzero(as_tuple=True)  # [P], [P]

    suppress_matrix = torch.zeros((M, M), dtype=torch.bool, device=device)
    if pair_i.numel() > 0:
        # Sampled rotated IoU computed per pair (O(P*S)) instead of as a dense
        # [M, M] matrix (O(M^2 * S)) — same estimator as oriented_box_iou_gpu:
        # max of both containment perspectives, clamped by the smaller box area.
        num_s = _nms_num_samples_for_boxes(ordered_boxes)
        target_max_bytes = 1.5 * (1024 ** 3)
        pair_chunk = max(1, int(target_max_bytes / (4 * 2 * num_s)))
        for p_start in range(0, pair_i.numel(), pair_chunk):
            p_end = min(p_start + pair_chunk, pair_i.numel())
            pi = pair_i[p_start:p_end]
            pj = pair_j[p_start:p_end]
            boxes_i = ordered_boxes[pi]
            boxes_j = ordered_boxes[pj]
            area_i = areas[pi]
            area_j = areas[pj]

            samples_i = _generate_box_samples(boxes_i, num_s)  # [p, S, 2]
            samples_j = _generate_box_samples(boxes_j, num_s)
            in_j = _points_in_paired_boxes(samples_i, verts[pj])  # [p, S]
            in_i = _points_in_paired_boxes(samples_j, verts[pi])

            inter_1 = in_j.sum(dim=1).float() / float(num_s) * area_i
            inter_2 = in_i.sum(dim=1).float() / float(num_s) * area_j
            inter = torch.maximum(inter_1, inter_2)
            inter = torch.minimum(inter, torch.minimum(area_i, area_j))
            union = torch.maximum(area_i + area_j - inter, inter + 1e-8)
            iou = torch.clamp(inter / (union + 1e-8), 0.0, 1.0)

            sup = iou > iou_threshold
            suppress_matrix[pi[sup], pj[sup]] = True

    # Greedy NMS via iterative matrix suppression (Cluster-NMS, Zheng et al. 2020).
    # Iterating keep[j] = not any_i(suppress_matrix[i, j] and keep[i]) converges to the
    # unique fixpoint, which equals the sequential greedy NMS result. Convergence takes
    # at most the longest suppression-chain depth (typically < 10 iterations), so this
    # replaces M kernel launches + M host syncs of the previous per-row Python loop
    # (the dominant validation cost for large weakly-suppressing candidate pools).
    keep_mask = torch.ones(M, dtype=torch.bool, device=device)
    for _ in range(M):
        new_keep = ~(suppress_matrix & keep_mask.unsqueeze(1)).any(dim=0)
        if torch.equal(new_keep, keep_mask):
            break
        keep_mask = new_keep

    kept_indices = order[keep_mask]
    
    # Apply max_detections limit after processing all candidates
    if max_detections is not None:
        kept_indices = kept_indices[:max_detections]
    
    return kept_indices


def hbb_nms_for_oriented_boxes_gpu(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.7,
    max_detections: Optional[int] = None,
) -> Tensor:
    """Fast HBB NMS for oriented boxes.

    This is intended for RPN proposal filtering, where MMRotate-style Rotated
    Faster R-CNN uses horizontal overlaps for assignment/proposal pruning. It
    avoids the Python greedy loop in oriented_nms_gpu, which can become
    CPU-launch bound for large weakly suppressing proposal pools.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    xyxy = obb_to_xyxy_gpu(boxes)
    keep = torchvision_nms(xyxy, scores, iou_threshold)
    if max_detections is not None:
        keep = keep[:max_detections]
    return keep


__all__ = [
    "oriented_box_iou_gpu",
    "obb_to_xyxy_gpu",
    "hbb_iou_gpu",
    "oriented_box_hbb_iou_gpu",
    "generate_oriented_anchors_gpu",
    "match_anchors_to_gt_gpu",
    "oriented_nms_gpu",
    "hbb_nms_for_oriented_boxes_gpu",
    "resolve_oriented_iou_sample_count",
    "geometry_sample_count_for_boxes",
]
