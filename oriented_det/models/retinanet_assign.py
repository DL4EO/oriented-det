"""Rotated RetinaNet MaxIoU assignment (dense FPN priors).

Two-stage RPN/ROI matching stays on ``match_oriented_anchors_to_gt`` /
``oriented_box_iou_gpu``. This module is used only by Rotated RetinaNet so
P3-scale grids (especially 27 priors/location) do not run the shared dense
sampling IoU path (geometry-sized grids up to 1024 samples plus per-chunk
``.item()`` syncs).
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = None  # type: ignore

from ..ops.gpu_ops import (
    _box_vertices,
    _generate_box_samples,
    _points_in_paired_boxes,
)
from .oriented_rpn import match_oriented_anchors_to_gt

# Same estimator as ``oriented_box_iou_gpu``, but one fixed grid for assignment
# ranking (0.5 / 0.4). Not the shared geometry-max-of-batch sample count.
_ASSIGN_IOU_SAMPLES = 100
_ANCHOR_CHUNK = 8192
_PAIR_CHUNK = 65536
_MAX_DENSE_ELEMENTS = 16_000_000  # ~64 MiB float32 [chunk, M]


def match_retinanet_anchors_to_gt(
    anchors: Tensor,
    gt_boxes: Tensor,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.4,
    device: Optional[torch.device] = None,
    use_hbb_for_matching: bool = False,
    min_pos_iou: float = 0.0,
    match_low_quality: bool = True,
    gt_boxes_ignore: Optional[Tensor] = None,
    ignore_iou_threshold: Optional[float] = None,
    gt_boxes_lookalike: Optional[Tensor] = None,
    lookalike_iou_threshold: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """MaxIoU assign RetinaNet anchors. HBB still uses the shared matcher."""
    if torch is None:
        raise RuntimeError("PyTorch is required for RetinaNet assignment.")
    if use_hbb_for_matching:
        return match_oriented_anchors_to_gt(
            anchors,
            gt_boxes,
            positive_iou_threshold,
            negative_iou_threshold,
            device,
            use_hbb_for_matching=True,
            min_pos_iou=min_pos_iou,
            match_low_quality=match_low_quality,
            gt_boxes_ignore=gt_boxes_ignore,
            ignore_iou_threshold=ignore_iou_threshold,
            gt_boxes_lookalike=gt_boxes_lookalike,
            lookalike_iou_threshold=lookalike_iou_threshold,
        )

    if device is None:
        device = anchors.device
    anchors = anchors.to(device).detach()
    gt_boxes = gt_boxes.to(device).detach()
    n = int(anchors.shape[0])
    m = int(gt_boxes.shape[0])

    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    matched_gt_indices = torch.full((n,), -1, dtype=torch.long, device=device)

    if m == 0:
        labels.fill_(0)
    else:
        max_iou_per_anchor, best_gt_per_anchor, max_iou_per_gt, best_anchor_per_gt = (
            _maxiou_rotated_stats(anchors, gt_boxes)
        )
        positive_mask = max_iou_per_anchor >= positive_iou_threshold
        labels[positive_mask] = 1
        matched_gt_indices[positive_mask] = best_gt_per_anchor[positive_mask]

        best_anchor_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if match_low_quality:
            for gt_idx in range(m):
                if max_iou_per_gt[gt_idx] >= min_pos_iou:
                    anchor_idx = best_anchor_per_gt[gt_idx]
                    labels[anchor_idx] = 1
                    matched_gt_indices[anchor_idx] = gt_idx
                    best_anchor_mask[anchor_idx] = True

        negative_mask = (max_iou_per_anchor < negative_iou_threshold) & (~best_anchor_mask)
        labels[negative_mask] = 0

    if gt_boxes_ignore is not None and gt_boxes_ignore.numel() > 0:
        thr = (
            float(ignore_iou_threshold)
            if ignore_iou_threshold is not None
            else float(positive_iou_threshold)
        )
        max_iou_ign = _max_rotated_iou_per_anchor(anchors, gt_boxes_ignore.to(device).detach())
        ign_mask = (labels <= 0) & (max_iou_ign >= thr)
        labels[ign_mask] = -1
        matched_gt_indices[ign_mask] = -1

    if gt_boxes_lookalike is not None and gt_boxes_lookalike.numel() > 0:
        look_thr = (
            float(lookalike_iou_threshold)
            if lookalike_iou_threshold is not None
            else float(positive_iou_threshold)
        )
        max_iou_look = _max_rotated_iou_per_anchor(
            anchors, gt_boxes_lookalike.to(device).detach()
        )
        look_mask = (labels != 1) & (max_iou_look >= look_thr)
        labels[look_mask] = 0
        matched_gt_indices[look_mask] = -1

    return labels, matched_gt_indices


def _aabb_overlap(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    verts1 = _box_vertices(boxes1)
    verts2 = _box_vertices(boxes2)
    min1 = verts1.min(dim=1).values
    max1 = verts1.max(dim=1).values
    min2 = verts2.min(dim=1).values
    max2 = verts2.max(dim=1).values
    return (
        (min1[:, 0:1] <= max2[None, :, 0])
        & (max1[:, 0:1] >= min2[None, :, 0])
        & (min1[:, 1:2] <= max2[None, :, 1])
        & (max1[:, 1:2] >= min2[None, :, 1])
    )


def _paired_rotated_iou(boxes_a: Tensor, boxes_b: Tensor, num_samples: int) -> Tensor:
    """Sampling IoU for aligned pairs (same estimator as ``oriented_box_iou_gpu``)."""
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    samples_a = _generate_box_samples(boxes_a, num_samples)
    samples_b = _generate_box_samples(boxes_b, num_samples)
    verts_a = _box_vertices(boxes_a)
    verts_b = _box_vertices(boxes_b)
    count_in_b = _points_in_paired_boxes(samples_a, verts_b).sum(dim=1).float()
    count_in_a = _points_in_paired_boxes(samples_b, verts_a).sum(dim=1).float()
    inter = torch.maximum(
        (count_in_b / num_samples) * area_a,
        (count_in_a / num_samples) * area_b,
    )
    inter = torch.minimum(inter, torch.minimum(area_a, area_b))
    union = (area_a + area_b - inter).clamp(min=1e-8)
    return torch.clamp(inter / union, 0.0, 1.0)


def _anchor_chunk_size(num_gt: int) -> int:
    denom = max(1, int(num_gt))
    return max(1, min(_ANCHOR_CHUNK, _MAX_DENSE_ELEMENTS // denom))


def _fill_pairwise_rotated_iou(chunk: Tensor, gt_boxes: Tensor, iou_dense: Tensor) -> None:
    overlap = _aabb_overlap(chunk, gt_boxes)
    if not overlap.any():
        return
    pair_i, pair_j = overlap.nonzero(as_tuple=True)
    for p_start in range(0, pair_i.numel(), _PAIR_CHUNK):
        p_end = min(p_start + _PAIR_CHUNK, pair_i.numel())
        pi = pair_i[p_start:p_end]
        pj = pair_j[p_start:p_end]
        iou_dense[pi, pj] = _paired_rotated_iou(
            chunk[pi], gt_boxes[pj], _ASSIGN_IOU_SAMPLES
        )


def _maxiou_rotated_stats(
    anchors: Tensor, gt_boxes: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = anchors.device
    n = anchors.shape[0]
    m = gt_boxes.shape[0]
    max_iou_per_anchor = torch.zeros((n,), device=device)
    best_gt_per_anchor = torch.zeros((n,), dtype=torch.long, device=device)
    max_iou_per_gt = torch.zeros((m,), device=device)
    best_anchor_per_gt = torch.zeros((m,), dtype=torch.long, device=device)
    chunk_size = _anchor_chunk_size(m)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = anchors[start:end]
        iou_dense = torch.zeros((chunk.shape[0], m), device=device)
        _fill_pairwise_rotated_iou(chunk, gt_boxes, iou_dense)
        chunk_anchor_iou, chunk_best_gt = iou_dense.max(dim=1)
        max_iou_per_anchor[start:end] = chunk_anchor_iou
        best_gt_per_anchor[start:end] = chunk_best_gt
        chunk_gt_iou, chunk_best_anchor = iou_dense.max(dim=0)
        better_gt = chunk_gt_iou > max_iou_per_gt
        if better_gt.any():
            max_iou_per_gt[better_gt] = chunk_gt_iou[better_gt]
            best_anchor_per_gt[better_gt] = chunk_best_anchor[better_gt] + start
    return max_iou_per_anchor, best_gt_per_anchor, max_iou_per_gt, best_anchor_per_gt


def _max_rotated_iou_per_anchor(anchors: Tensor, other_boxes: Tensor) -> Tensor:
    device = anchors.device
    n = anchors.shape[0]
    m = other_boxes.shape[0]
    out = torch.zeros((n,), device=device)
    if n == 0 or m == 0:
        return out
    chunk_size = _anchor_chunk_size(m)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = anchors[start:end]
        overlap = _aabb_overlap(chunk, other_boxes)
        if not overlap.any():
            continue
        pair_i, pair_j = overlap.nonzero(as_tuple=True)
        chunk_max = torch.zeros((chunk.shape[0],), device=device)
        for p_start in range(0, pair_i.numel(), _PAIR_CHUNK):
            p_end = min(p_start + _PAIR_CHUNK, pair_i.numel())
            pi = pair_i[p_start:p_end]
            pj = pair_j[p_start:p_end]
            ious = _paired_rotated_iou(chunk[pi], other_boxes[pj], _ASSIGN_IOU_SAMPLES)
            chunk_max.scatter_reduce_(0, pi, ious, reduce="amax", include_self=True)
        out[start:end] = chunk_max
    return out
