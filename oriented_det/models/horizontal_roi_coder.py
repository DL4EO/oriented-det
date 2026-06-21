"""DeltaXYWHTHBBoxCoder — horizontal RoI (xyxy) to rotated GT (MMRotate-style).

Pure PyTorch; no custom CUDA. Matches the math in MMRotate
``delta_xywht_hbbox_coder.py`` for ``angle_version='le90'``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .oriented_rpn import normalize_boxes_to_le90


def _norm_angle_le90(angle: torch.Tensor) -> torch.Tensor:
    """Map angle to [-pi/2, pi/2) (le90)."""
    pi = math.pi
    return torch.remainder(angle + pi / 2, pi) - pi / 2


def encode_delta_xywh_th(
    rois_xyxy: torch.Tensor,
    gt_rboxes: torch.Tensor,
    means: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    norm_factor: Optional[float] = 2.0,
    edge_swap: bool = True,
) -> torch.Tensor:
    """Encode rotated GT w.r.t. horizontal RoIs [N,4] xyxy -> deltas [N,5]."""
    if torch is None:
        raise RuntimeError("torch required")
    rois = rois_xyxy.float()
    gts = normalize_boxes_to_le90(gt_rboxes.float())
    px = (rois[:, 0] + rois[:, 2]) * 0.5
    py = (rois[:, 1] + rois[:, 3]) * 0.5
    pw = rois[:, 2] - rois[:, 0]
    ph = rois[:, 3] - rois[:, 1]
    pw = torch.clamp(pw, min=1e-6)
    ph = torch.clamp(ph, min=1e-6)
    gx, gy, gw, gh, gt = gts.unbind(dim=-1)
    if edge_swap:
        dtheta1 = _norm_angle_le90(gt)
        dtheta2 = _norm_angle_le90(gt + math.pi / 2)
        abs_dtheta1 = torch.abs(dtheta1)
        abs_dtheta2 = torch.abs(dtheta2)
        gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        gt_enc = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dw = torch.log(gw_regular / pw)
        dh = torch.log(gh_regular / ph)
    else:
        gt_enc = _norm_angle_le90(gt)
        dw = torch.log(torch.clamp(gw, min=1e-6) / pw)
        dh = torch.log(torch.clamp(gh, min=1e-6) / ph)
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    if norm_factor is not None:
        dt = gt_enc / (norm_factor * math.pi)
    else:
        dt = gt_enc
    deltas = torch.stack([dx, dy, dw, dh, dt], dim=-1)
    means_t = deltas.new_tensor(means).unsqueeze(0)
    stds_t = deltas.new_tensor(stds).unsqueeze(0)
    return (deltas - means_t) / (stds_t + 1e-8)


def decode_delta_xywh_th(
    rois_xyxy: torch.Tensor,
    deltas: torch.Tensor,
    means: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    norm_factor: Optional[float] = 2.0,
    edge_swap: bool = True,
    wh_ratio_clip: float = 16.0 / 1000.0,
) -> torch.Tensor:
    """Decode [N,5] deltas to rotated boxes [N,5] (cx,cy,w,h,a) le90."""
    if torch is None:
        raise RuntimeError("torch required")
    rois = rois_xyxy.float()
    means_t = rois.new_tensor(means).view(1, -1)
    stds_t = rois.new_tensor(stds).view(1, -1)
    denorm = deltas * stds_t + means_t
    dx, dy, dw, dh, dt = denorm[:, 0], denorm[:, 1], denorm[:, 2], denorm[:, 3], denorm[:, 4]
    if norm_factor is not None:
        dt = dt * (norm_factor * math.pi)
    px = (rois[:, 0] + rois[:, 2]) * 0.5
    py = (rois[:, 1] + rois[:, 3]) * 0.5
    pw = rois[:, 2] - rois[:, 0]
    ph = rois[:, 3] - rois[:, 1]
    pw = torch.clamp(pw, min=1e-6)
    ph = torch.clamp(ph, min=1e-6)
    max_ratio = abs(math.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    gx = px + pw * dx
    gy = py + ph * dy
    gt = _norm_angle_le90(dt)
    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, gt, _norm_angle_le90(gt + math.pi / 2))
        out = torch.stack([gx, gy, w_regular, h_regular, theta_regular], dim=-1)
    else:
        out = torch.stack([gx, gy, gw, gh, gt], dim=-1)
    return normalize_boxes_to_le90(out)


class DeltaXYWHTHBBoxCoder:
    """Horizontal RoI (xyxy) <-> rotated box deltas (MMRotate ROI for Rotated Faster R-CNN)."""

    def __init__(
        self,
        target_means: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds: Tuple[float, float, float, float, float] = (0.1, 0.1, 0.2, 0.2, 0.1),
        norm_factor: Optional[float] = 2.0,
        edge_swap: bool = True,
    ):
        self.target_means = target_means
        self.target_stds = target_stds
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap

    def encode(self, rois_xyxy: torch.Tensor, gt_rboxes: torch.Tensor) -> torch.Tensor:
        return encode_delta_xywh_th(
            rois_xyxy,
            gt_rboxes,
            means=self.target_means,
            stds=self.target_stds,
            norm_factor=self.norm_factor,
            edge_swap=self.edge_swap,
        )

    def decode(self, rois_xyxy: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return decode_delta_xywh_th(
            rois_xyxy,
            deltas,
            means=self.target_means,
            stds=self.target_stds,
            norm_factor=self.norm_factor,
            edge_swap=self.edge_swap,
        )


__all__ = [
    "encode_delta_xywh_th",
    "decode_delta_xywh_th",
    "DeltaXYWHTHBBoxCoder",
]
