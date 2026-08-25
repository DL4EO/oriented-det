"""Differentiable rotated IoU for matched OBB pairs (train loss).

Ports the pure-PyTorch path of mmcv ``diff_iou_rotated_2d`` / lilanxiao
Rotated_IoU (convex intersection via edge crossings + contained corners,
then shoelace). **Not** Monte-Carlo :func:`~oriented_det.ops.rotated_ops.pairwise_rotated_iou`.

Boxes are ``[..., 5]`` in this repo's ``(cx, cy, w, h, angle_rad)``.
Runs on CPU or CUDA (vectorized PyTorch; no custom kernel). Intersection
math is fp32 with AMP off, same reason as KFIoU.

Vertex order around the intersection centroid uses ``atan2`` (mmcv dropped
its CUDA sort for stability). Invalid slots gather a zeroed dummy so extra
shoelace terms are 0.
"""

from __future__ import annotations

import contextlib
from typing import Tuple

import torch
from torch import Tensor

_EPSILON = 1e-8
_INVALID_ANGLE = 1.0e6


def _fp32_no_autocast_ctx(tensor: Tensor):
    device_type = tensor.device.type
    if device_type in ("cuda", "cpu"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        except (AttributeError, TypeError):
            if device_type == "cuda":
                return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def _as_bn5(boxes: Tensor) -> Tuple[Tensor, bool]:
    if boxes.ndim == 2:
        if boxes.size(-1) != 5:
            raise ValueError(f"Expected last dim 5 (xywhr), got {tuple(boxes.shape)}")
        return boxes.unsqueeze(0), True
    if boxes.ndim == 3 and boxes.size(-1) == 5:
        return boxes, False
    raise ValueError(f"Expected [N, 5] or [B, N, 5], got {tuple(boxes.shape)}")


def _box2corners(box: Tensor) -> Tensor:
    """``(B, N, 5)`` xywhr → ``(B, N, 4, 2)`` corners (same rotation as ``RBox.corners``)."""
    bsz = box.size(0)
    x, y, w, h, alpha = box.split(1, dim=-1)
    w = w.clamp(min=_EPSILON)
    h = h.clamp(min=_EPSILON)
    x4 = box.new_tensor([0.5, -0.5, -0.5, 0.5]) * w
    y4 = box.new_tensor([0.5, 0.5, -0.5, -0.5]) * h
    corners = torch.stack([x4, y4], dim=-1)
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    rot_t = torch.stack(
        [torch.cat([cos_a, sin_a], dim=-1), torch.cat([-sin_a, cos_a], dim=-1)],
        dim=-2,
    )
    rotated = torch.bmm(corners.reshape(-1, 4, 2), rot_t.reshape(-1, 2, 2))
    rotated = rotated.view(bsz, -1, 4, 2)
    rotated = rotated + torch.cat([x, y], dim=-1).unsqueeze(2)
    return rotated


def _box_intersection(corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Edge crossings. Collinear edges count as no intersection (mmcv convention)."""
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1, y1, x2, y2 = line1_ext.split(1, dim=-1)
    x3, y3, x4, y4 = line2_ext.split(1, dim=-1)
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / numerator
    t = torch.where(numerator == 0, t.new_full(t.shape, -1.0), t)
    mask_t = (t > 0) & (t < 1)
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / numerator
    u = torch.where(numerator == 0, u.new_full(u.shape, -1.0), u)
    mask_u = (u > 0) & (u < 1)
    mask = mask_t & mask_u
    t = den_t / (numerator + _EPSILON)
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    intersections = intersections * mask.to(dtype=intersections.dtype).unsqueeze(-1)
    return intersections, mask


def _box1_in_box2(corners1: Tensor, corners2: Tensor) -> Tensor:
    a = corners2[:, :, 0:1, :]
    b = corners2[:, :, 1:2, :]
    d = corners2[:, :, 3:4, :]
    ab = b - a
    am = corners1 - a
    ad = d - a
    prod_ab = torch.sum(ab * am, dim=-1)
    norm_ab = torch.sum(ab * ab, dim=-1).clamp(min=_EPSILON)
    prod_ad = torch.sum(ad * am, dim=-1)
    norm_ad = torch.sum(ad * ad, dim=-1).clamp(min=_EPSILON)
    t_ab = prod_ab / norm_ab
    t_ad = prod_ad / norm_ad
    cond1 = (t_ab > -1e-6) & (t_ab < 1.0 + 1e-6)
    cond2 = (t_ad > -1e-6) & (t_ad < 1.0 + 1e-6)
    return cond1 & cond2


def _sort_indices(vertices: Tensor, mask: Tensor) -> Tensor:
    """``(B, N, 24, 2)`` + validity → ``(B, N, 9)`` hull indices (wrap + dummy pad)."""
    mask_b = mask.bool()
    mask_f = mask_b.to(dtype=vertices.dtype)
    num_valid = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (vertices * mask_f.unsqueeze(-1)).sum(dim=2) / num_valid
    rel = vertices - mean.unsqueeze(2)
    ang = torch.atan2(rel[..., 1], rel[..., 0])
    ang = torch.where(mask_b, ang, ang.new_full(ang.shape, _INVALID_ANGLE))
    sorted_idx = torch.argsort(ang, dim=-1)
    first8 = sorted_idx[..., :8]
    wrap = first8[..., :1]
    dummy_idx = torch.argsort(mask_b.to(dtype=vertices.dtype), dim=-1)[..., :1]
    k = mask_b.to(dtype=torch.int64).sum(dim=-1).clamp(max=8)
    slots = torch.arange(9, device=vertices.device).view(1, 1, 9)
    k_exp = k.unsqueeze(-1)
    is_vertex = slots < k_exp
    is_wrap = (slots == k_exp) & (k_exp > 0)
    first8_pad = torch.cat([first8, first8[..., :1]], dim=-1)
    dummy_exp = dummy_idx.expand(-1, -1, 9)
    wrap_exp = wrap.expand(-1, -1, 9)
    out = torch.where(is_vertex, first8_pad, dummy_exp)
    return torch.where(is_wrap, wrap_exp, out)


def _intersection_area(corners1: Tensor, corners2: Tensor) -> Tensor:
    intersections, valid_mask = _box_intersection(corners1, corners2)
    c12 = _box1_in_box2(corners1, corners2)
    c21 = _box1_in_box2(corners2, corners1)
    bsz, nbox = corners1.shape[:2]
    vertices = torch.cat(
        [corners1, corners2, intersections.reshape(bsz, nbox, -1, 2)],
        dim=2,
    )
    mask = torch.cat([c12, c21, valid_mask.reshape(bsz, nbox, -1)], dim=2)
    idx_sorted = _sort_indices(vertices, mask)
    idx_ext = idx_sorted.unsqueeze(-1).expand(-1, -1, -1, 2)
    selected = torch.gather(vertices, 2, idx_ext)
    total = (
        selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1]
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    )
    return torch.abs(total.sum(dim=2)) * 0.5


def diff_iou_rotated_2d(box1: Tensor, box2: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Paired differentiable rotated IoU.

    Args:
        box1: ``[N, 5]`` or ``[B, N, 5]`` predicted boxes (differentiable).
        box2: same shape, typically detached GT.
        eps: union stabilizer.

    Returns:
        IoU with shape ``[N]`` or ``[B, N]``, in ``float32`` (autograd casts back).
    """
    if box1.shape != box2.shape:
        raise ValueError(f"Shape mismatch: {tuple(box1.shape)} vs {tuple(box2.shape)}")
    b1, squeezed = _as_bn5(box1)
    b2, _ = _as_bn5(box2)
    if b1.numel() == 0:
        out = b1.new_zeros(b1.shape[:2])
        return out.squeeze(0) if squeezed else out
    with _fp32_no_autocast_ctx(b1):
        p = b1.float()
        t = b2.float()
        inter = _intersection_area(_box2corners(p), _box2corners(t))
        area1 = (p[..., 2] * p[..., 3]).clamp(min=0.0)
        area2 = (t[..., 2] * t[..., 3]).clamp(min=0.0)
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=eps)
        iou = iou.clamp(0.0, 1.0)
        iou = torch.where(torch.isfinite(iou), iou, torch.zeros_like(iou))
    return iou.squeeze(0) if squeezed else iou


def riou_loss_per_box(
    pred_decode: Tensor,
    targets_decode: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Linear rotated-IoU loss ``1 - IoU`` per matched pair (``[N]``)."""
    iou = diff_iou_rotated_2d(pred_decode, targets_decode, eps=eps)
    return (1.0 - iou).clamp(min=0.0)
