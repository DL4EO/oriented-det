"""Probabilistic IoU (ProbIoU) loss for oriented boxes.

Gaussian bounding-box formulation from Lv et al., *Probabilistic IoU for Oriented
Object Detection* ([reference implementation](https://github.com/ProbIOU/probiou-sample/blob/main/probiou_pytorch.py)).

Box tensors are ``[cx, cy, w, h, angle_rad]`` (same convention as the rest of this
codebase).
"""

from __future__ import annotations

import contextlib
from typing import Literal, Optional

import torch
from torch import Tensor

ProbIoUMode = Literal["l1", "l2"]


def _fp32_no_autocast_ctx(tensor: Tensor):
    device_type = tensor.device.type
    if device_type in ("cuda", "cpu"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        except (AttributeError, TypeError):
            if device_type == "cuda":
                return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def gbb_form(boxes: Tensor) -> Tensor:
    """Oriented box to Gaussian parameters ``(cx, cy, var_w, var_h, angle)``."""
    wh = boxes[:, 2:4].clamp(min=1e-7)
    return torch.cat((boxes[:, :2], torch.pow(wh, 2) / 12.0, boxes[:, 4:]), dim=1)


def rotated_form(a_: Tensor, b_: Tensor, angles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Rotate Gaussian variances by ``angles`` (radians)."""
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    cos2 = torch.pow(cos_a, 2.0)
    sin2 = torch.pow(sin_a, 2.0)
    a = a_ * cos2 + b_ * sin2
    b = a_ * sin2 + b_ * cos2
    c = a_ * cos_a * sin_a - b_ * sin_a * cos_a
    return a, b, c


def probiou_loss_per_box(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-3,
    mode: ProbIoUMode = "l1",
) -> Tensor:
    """Per-instance ProbIoU loss ``[N]`` (lower is better overlap).

    Args:
        pred: ``[N, 5]`` predicted boxes ``(cx, cy, w, h, angle_rad)``.
        target: ``[N, 5]`` target boxes (same layout).
        eps: Stabilizer for divisions and logarithms.
        mode: ``\"l1\"`` → bounded metric in ``[0, 1]``; ``\"l2\"`` → unbounded
            ``-log(1 - l1^2)`` variant from the paper.
    """
    if pred.shape != target.shape or pred.shape[-1] != 5:
        raise ValueError(f"Expected matching [N, 5] tensors, got {pred.shape} vs {target.shape}")
    mode_norm = (mode or "l1").strip().lower()
    if mode_norm not in ("l1", "l2"):
        raise ValueError(f"mode must be 'l1' or 'l2', got {mode!r}")

    with _fp32_no_autocast_ctx(pred):
        pred_f = pred.float()
        target_f = target.float()

        gbboxes1 = gbb_form(pred_f)
        gbboxes2 = gbb_form(target_f)

        x1, y1, a1_, b1_, c1_ = (
            gbboxes1[:, 0],
            gbboxes1[:, 1],
            gbboxes1[:, 2],
            gbboxes1[:, 3],
            gbboxes1[:, 4],
        )
        x2, y2, a2_, b2_, c2_ = (
            gbboxes2[:, 0],
            gbboxes2[:, 1],
            gbboxes2[:, 2],
            gbboxes2[:, 3],
            gbboxes2[:, 4],
        )

        a1, b1, c1 = rotated_form(a1_, b1_, c1_)
        a2, b2, c2 = rotated_form(a2_, b2_, c2_)

        denom_base = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2) + eps
        t1 = (
            ((a1 + a2) * torch.pow(y1 - y2, 2) + (b1 + b2) * torch.pow(x1 - x2, 2))
            / denom_base
        ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom_base) * 0.5
        det1 = (a1 * b1 - torch.pow(c1, 2)).clamp(min=eps)
        det2 = (a2 * b2 - torch.pow(c2, 2)).clamp(min=eps)
        t3 = (
            torch.log(
                denom_base
                / (4.0 * torch.sqrt(det1 * det2) + eps)
                + eps
            )
            * 0.5
        )

        b_d = torch.clamp(t1 + t2 + t3, min=eps, max=100.0)
        l1 = torch.sqrt(1.0 - torch.exp(-b_d) + eps)
        if mode_norm == "l1":
            probiou = l1
        else:
            l_i = torch.pow(l1, 2.0)
            probiou = -torch.log(1.0 - l_i + eps)

    return probiou.to(dtype=pred.dtype)


def probiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-3,
    mode: ProbIoUMode = "l1",
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Reduced ProbIoU loss (scalar unless ``reduction=\"none\"``)."""
    loss = probiou_loss_per_box(pred, target, eps=eps, mode=mode)
    if weight is not None:
        if weight.dim() > 1:
            weight = weight.mean(dim=-1)
        loss = loss * weight
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if weight is not None:
            denom = weight.sum().clamp(min=eps)
            return loss.sum() / denom
        return loss.mean()
    raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")


def _normalize_probiou_mode(mode: Optional[str]) -> ProbIoUMode:
    s = (mode or "l1").strip().lower()
    if s not in ("l1", "l2"):
        raise ValueError(f"probiou mode must be 'l1' or 'l2', got {mode!r}")
    return s  # type: ignore[return-value]


__all__ = [
    "gbb_form",
    "rotated_form",
    "probiou_loss_per_box",
    "probiou_loss",
    "_normalize_probiou_mode",
]
