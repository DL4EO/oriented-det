"""Bbox coders with MMRotate-compatible naming for user familiarity.

Mapping to MMRotate:
- DeltaXYWHBBoxCoder: 4 params (dx, dy, dw, dh). Rotated Faster R-CNN RPN.
- DeltaXYWHAHBBoxCoder: 5 params (dx, dy, dw, dh, da) with norm_factor, edge_swap. ROI head.
- DistanceAnglePointCoder: 5 params (left, top, right, bottom, angle) from FPN points. Rotated FCOS.
- MidpointOffsetCoder: 6 params (dx, dy, dw, dh, da, db). Oriented R-CNN RPN (midpoint offset).
  Expects axis-aligned xyxy proposals (x1, y1, x2, y2) because it transforms horizontal boxes
  into oriented boxes; (da, db) are defined relative to the proposal center and size.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .oriented_rpn import (
    encode_rpn_boxes,
    decode_rpn_boxes,
    encode_oriented_boxes,
    decode_oriented_boxes,
    normalize_boxes_to_le90,
    norm_angle_le90,
)
from ..ops import obb_to_xyxy_gpu
from .oriented_rpn import _obb_to_hbb_obb as obb_to_hbb_obb


class DeltaXYWHBBoxCoder:
    """4 params: dx, dy, dw, dh. Angle from anchor. MMRotate Rotated Faster R-CNN RPN.

    Same as encode_rpn_boxes / decode_rpn_boxes.
    """

    def __init__(
        self,
        target_means: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        target_stds: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        self.target_means = target_means
        self.target_stds = target_stds

    def encode(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode gt boxes relative to anchors. anchors/gt: [N, 5] (cx, cy, w, h, angle)."""
        return encode_rpn_boxes(anchors, gt_boxes, self.target_means, self.target_stds)

    def decode(
        self,
        anchors: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Decode deltas [N, 4] to boxes [N, 5]. Angle from anchor."""
        return decode_rpn_boxes(anchors, deltas, self.target_means, self.target_stds)


class DeltaXYWHAHBBoxCoder:
    """5 params: dx, dy, dw, dh, da. norm_factor, edge_swap. MMRotate ROI head."""

    def __init__(
        self,
        target_means: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        norm_factor: Optional[float] = 2.0,
        edge_swap: bool = True,
        angle_range: str = "le90",
    ):
        self.target_means = target_means
        self.target_stds = target_stds
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.angle_range = angle_range

    def encode(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode gt boxes relative to anchors. anchors/gt: [N, 5] (cx, cy, w, h, angle)."""
        return encode_oriented_boxes(
            anchors,
            gt_boxes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            norm_factor=self.norm_factor,
            edge_swap=self.edge_swap,
        )

    def decode(
        self,
        anchors: torch.Tensor,
        deltas: torch.Tensor,
        normalize_le90: bool = True,
    ) -> torch.Tensor:
        """Decode deltas [N, 5] to boxes [N, 5]."""
        return decode_oriented_boxes(
            anchors,
            deltas,
            target_means=self.target_means,
            target_stds=self.target_stds,
            normalize_le90=normalize_le90,
            norm_factor=self.norm_factor,
            edge_swap=self.edge_swap,
        )


def _obb_to_xyxy_tensor(obb: torch.Tensor) -> torch.Tensor:
    """Convert obb [N, 5] (cx, cy, w, h, angle) to axis-aligned xyxy [N, 4].

    Thin wrapper around obb_to_xyxy_gpu (works on CPU and GPU).
    """
    return obb_to_xyxy_gpu(obb)


obb_to_xyxy = obb_to_xyxy_gpu  # Public alias for __all__


def _obb_to_poly_tensor(obb: torch.Tensor) -> torch.Tensor:
    """Convert obb [N, 5] (cx, cy, w, h, angle) to poly [N, 8] (x1,y1,x2,y2,x3,y3,x4,y4)."""
    cx, cy, w, h, angle = obb.unbind(dim=1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    w2, h2 = w / 2, h / 2
    # Local corners: (-w2,-h2), (w2,-h2), (w2,h2), (-w2,h2)
    dx = torch.stack([-w2, w2, w2, -w2], dim=1)
    dy = torch.stack([-h2, -h2, h2, h2], dim=1)
    rx = cx.unsqueeze(1) + dx * cos_a.unsqueeze(1) - dy * sin_a.unsqueeze(1)
    ry = cy.unsqueeze(1) + dx * sin_a.unsqueeze(1) + dy * cos_a.unsqueeze(1)
    return torch.stack(
        [rx[:, 0], ry[:, 0], rx[:, 1], ry[:, 1], rx[:, 2], ry[:, 2], rx[:, 3], ry[:, 3]],
        dim=1,
    )


def _poly_to_obb_tensor(poly: torch.Tensor) -> torch.Tensor:
    """Convert poly [N, 8] (x1,y1,...,x4,y4) to obb [N, 5] (cx, cy, w, h, angle) le90."""
    x_coords = poly[:, 0::2]  # [N, 4]
    y_coords = poly[:, 1::2]  # [N, 4]
    cx = x_coords.mean(dim=1)
    cy = y_coords.mean(dim=1)
    # Edge vectors
    dx1 = x_coords[:, 1] - x_coords[:, 0]
    dy1 = y_coords[:, 1] - y_coords[:, 0]
    dx2 = x_coords[:, 2] - x_coords[:, 1]
    dy2 = y_coords[:, 2] - y_coords[:, 1]
    w = torch.sqrt(dx1 * dx1 + dy1 * dy1 + 1e-8)
    h = torch.sqrt(dx2 * dx2 + dy2 * dy2 + 1e-8)
    angle = torch.atan2(dy1, dx1 + 1e-8)
    # le90: ensure width >= height
    swap = w < h
    w = torch.where(swap, h, w)
    h = torch.where(swap, torch.sqrt(dx1 * dx1 + dy1 * dy1 + 1e-8), h)
    angle = torch.where(swap, angle + math.pi / 2, angle)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([cx, cy, w, h, angle], dim=1)


def xyxy_to_obb(xyxy: torch.Tensor) -> torch.Tensor:
    """Convert axis-aligned boxes [N, 4] (x1, y1, x2, y2) to oriented boxes [N, 5] (cx, cy, w, h, angle).
    
    For axis-aligned boxes, the angle is 0. This is used in OrientedRCNN to convert
    horizontal proposals to oriented format before ROI align.
    
    Args:
        xyxy: Tensor [N, 4] with format [x1, y1, x2, y2]
        
    Returns:
        Tensor [N, 5] with format [cx, cy, w, h, angle] where angle=0
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    
    x1, y1, x2, y2 = xyxy.unbind(dim=1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    angle = torch.zeros_like(cx)  # Axis-aligned boxes have angle=0
    
    return torch.stack([cx, cy, w, h, angle], dim=1)


class DistanceAnglePointCoder:
    """Encode OBB as distances from a point: (left, top, right, bottom, angle).

    MMRotate ``DistanceAnglePointCoder`` (channel order follows their *code*, not the
    outdated docstring that said top/bottom/left/right). Used by Rotated FCOS.
    """

    def __init__(self, angle_version: str = "le90", clip_border: bool = False):
        if angle_version != "le90":
            raise ValueError(f"Only angle_version='le90' is supported, got {angle_version!r}")
        self.angle_version = angle_version
        self.clip_border = clip_border

    def encode(
        self,
        points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        max_dis: Optional[float] = None,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """Encode GT OBBs relative to points.

        Args:
            points: [N, 2] (x, y)
            gt_bboxes: [N, 5] (cx, cy, w, h, angle)
            max_dis: optional upper clamp on distances
            eps: ensure target < max_dis when clamping

        Returns:
            [N, 5] (left, top, right, bottom, angle)
        """
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2 and gt_bboxes.size(-1) == 5
        return _obb2distance(points, gt_bboxes, max_dis=max_dis, eps=eps)

    def decode(
        self,
        points: torch.Tensor,
        pred: torch.Tensor,
        max_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Decode (left, top, right, bottom, angle) to OBB (cx, cy, w, h, angle) le90.

        Args:
            points: [N, 2] or [B, N, 2]
            pred: [N, 5] or [B, N, 5]
            max_shape: unused unless clip_border (kept for API parity)
        """
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        if self.clip_border is False:
            max_shape = None
        return _distance2obb(points, pred, max_shape=max_shape)


def _obb2distance(
    points: torch.Tensor,
    gt_bboxes: torch.Tensor,
    max_dis: Optional[float] = None,
    eps: float = 0.1,
) -> torch.Tensor:
    """Point + OBB → (left, top, right, bottom, angle). Matches MMRotate ``obb2distance``."""
    ctr, wh, angle = torch.split(gt_bboxes, [2, 2, 1], dim=-1)
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    # Local frame: rotate (point - ctr) by R(θ) = [[c, s], [-s, c]]
    rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(
        -1, 2, 2
    )
    offset = points - ctr
    offset = torch.matmul(rot_matrix, offset.unsqueeze(-1)).squeeze(-1)
    w, h = wh[..., 0], wh[..., 1]
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    left = w / 2 + offset_x
    right = w / 2 - offset_x
    top = h / 2 + offset_y
    bottom = h / 2 - offset_y
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack((left, top, right, bottom, angle.squeeze(-1)), dim=-1)


def _distance2obb(
    points: torch.Tensor,
    distance: torch.Tensor,
    max_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """(left, top, right, bottom, angle) + point → OBB. Matches MMRotate ``distance2obb``."""
    flat = distance.dim() == 2
    if flat:
        points_ = points
        distance_ = distance
    else:
        b, n, _ = distance.shape
        points_ = points.reshape(b * n, 2)
        distance_ = distance.reshape(b * n, 5)

    dist4, angle = distance_.split([4, 1], dim=1)
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    # Inverse of encode rotation: R^T = [[c, -s], [s, c]]
    rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle], dim=1).reshape(
        -1, 2, 2
    )
    wh = dist4[:, :2] + dist4[:, 2:]
    offset_t = ((dist4[:, 2:] - dist4[:, :2]) / 2).unsqueeze(2)
    offset = torch.bmm(rot_matrix, offset_t).squeeze(2)
    ctr = points_ + offset
    angle_regular = norm_angle_le90(angle.squeeze(-1)).unsqueeze(-1)
    obb = torch.cat([ctr, wh, angle_regular], dim=-1)
    obb = normalize_boxes_to_le90(obb)

    if max_shape is not None:
        # Optional border clip kept as no-op for API parity (clip_border=False by default).
        pass

    if flat:
        return obb
    return obb.reshape(b, n, 5)


class MidpointOffsetCoder:
    """6 params: dx, dy, dw, dh, da, db. MMRotate Oriented R-CNN RPN (MidpointOffsetCoder).

    Encodes/decodes axis-aligned proposals (xyxy) relative to oriented GT using midpoint offsets.
    Proposals must be axis-aligned xyxy [N, 4] (x1, y1, x2, y2) so that center (px, py) and
    size (pw, ph) are uniquely defined; the coder then predicts deltas to transform each
    horizontal box into an oriented box (da, db = offsets of the oriented box's top/right
    edge midpoints relative to the horizontal box). Used when the RPN outputs horizontal
    proposals and a second stage predicts orientation (e.g. Oriented R-CNN).
    """

    def __init__(
        self,
        target_means: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
        angle_range: str = "le90",
    ):
        self.target_means = target_means
        self.target_stds = target_stds
        self.angle_range = angle_range

    def encode(
        self,
        proposals: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode GT obb relative to axis-aligned proposals.

        proposals: [N, 4] xyxy (x1, y1, x2, y2); must be axis-aligned so (px, py, pw, ph)
        are well-defined. gt_bboxes: [N, 5] (cx, cy, w, h, angle). Returns [N, 6] deltas.
        """
        return _midpoint_offset_encode(proposals, gt_bboxes, self.target_means, self.target_stds)

    def decode(
        self,
        proposals: torch.Tensor,
        deltas: torch.Tensor,
        wh_ratio_clip: float = 16 / 1000,
    ) -> torch.Tensor:
        """Decode deltas to oriented boxes.

        proposals: [N, 4] xyxy (axis-aligned); deltas: [N, 6]. Returns [N, 5] obb (cx, cy, w, h, angle).
        """
        return _midpoint_offset_decode(
            proposals, deltas, self.target_means, self.target_stds, wh_ratio_clip
        )


def _midpoint_offset_encode(
    proposals: torch.Tensor,
    gt: torch.Tensor,
    means: Tuple[float, ...],
    stds: Tuple[float, ...],
) -> torch.Tensor:
    """Encode GT obb relative to axis-aligned proposals (xyxy).

    proposals must be xyxy so center/size (px, py, pw, ph) are defined. Returns [N, 6] (dx, dy, dw, dh, da, db).
    """
    if torch is None:
        raise RuntimeError("PyTorch required.")
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    hbb = _obb_to_xyxy_tensor(gt)
    gx = (hbb[:, 0] + hbb[:, 2]) * 0.5
    gy = (hbb[:, 1] + hbb[:, 3]) * 0.5
    gw = hbb[:, 2] - hbb[:, 0]
    gh = hbb[:, 3] - hbb[:, 1]

    poly = _obb_to_poly_tensor(gt)  # [N, 8]
    x_coor = poly[:, 0::2]  # [N, 4]
    y_coor = poly[:, 1::2]  # [N, 4]
    y_min, _ = y_coor.min(dim=1, keepdim=True)
    x_max, _ = x_coor.max(dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor - y_min) > 0.1] = -1000
    ga, _ = _x_coor.max(dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor - x_max) > 0.1] = -1000
    gb, _ = _y_coor.max(dim=1)

    dx = (gx - px) / (pw + 1e-8)
    dy = (gy - py) / (ph + 1e-8)
    dw = torch.log(gw / (pw + 1e-8))
    dh = torch.log(gh / (ph + 1e-8))
    da = (ga - gx) / (gw + 1e-8)
    db = (gb - gy) / (gh + 1e-8)
    deltas = torch.stack([dx, dy, dw, dh, da, db], dim=1)
    means_t = deltas.new_tensor(means)
    stds_t = deltas.new_tensor(stds)
    deltas = (deltas - means_t) / (stds_t + 1e-8)
    return deltas


def _midpoint_offset_decode(
    rois: torch.Tensor,
    deltas: torch.Tensor,
    means: Tuple[float, ...],
    stds: Tuple[float, ...],
    wh_ratio_clip: float = 16 / 1000,
) -> torch.Tensor:
    """Decode deltas [N, 6] to obb [N, 5]. rois must be [N, 4] axis-aligned xyxy."""
    if torch is None:
        raise RuntimeError("PyTorch required.")
    means_t = deltas.new_tensor(means)
    stds_t = deltas.new_tensor(stds)
    denorm = deltas * stds_t + means_t
    dx = denorm[:, 0]
    dy = denorm[:, 1]
    dw = denorm[:, 2]
    dh = denorm[:, 3]
    da = denorm[:, 4]
    db = denorm[:, 5]
    max_ratio = abs(math.log(wh_ratio_clip))
    dw = torch.clamp(dw, -max_ratio, max_ratio)
    dh = torch.clamp(dh, -max_ratio, max_ratio)

    px = (rois[:, 0] + rois[:, 2]) * 0.5
    py = (rois[:, 1] + rois[:, 3]) * 0.5
    pw = rois[:, 2] - rois[:, 0]
    ph = rois[:, 3] - rois[:, 1]

    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    gx = px + pw * dx
    gy = py + ph * dy

    da = torch.clamp(da, -0.5, 0.5)
    db = torch.clamp(db, -0.5, 0.5)
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=1)
    # Rectify quad to rectangle (MMRotate: diag_scale_factor)
    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=1)
    center_polys = polys - center
    diag_len = torch.sqrt(
        center_polys[:, 0::2] ** 2 + center_polys[:, 1::2] ** 2 + 1e-8
    )
    max_diag_len, _ = diag_len.max(dim=1, keepdim=True)
    diag_scale_factor = max_diag_len / (diag_len + 1e-8)
    center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=1)
    rectpolys = center_polys + center
    obb = _poly_to_obb_tensor(rectpolys)
    return normalize_boxes_to_le90(obb)


__all__ = [
    "DeltaXYWHBBoxCoder",
    "DeltaXYWHAHBBoxCoder",
    "DistanceAnglePointCoder",
    "MidpointOffsetCoder",
    "obb_to_hbb_obb",
    "obb_to_xyxy",
    "xyxy_to_obb",
]
