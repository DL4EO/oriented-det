"""Aspect-gated angle loss for Gaussian OBB surrogates (YOLO26 / MKIoU-style).

Restores a heading gradient on near-square boxes where KFIoU / ProbIoU are
rotation-invariant (isotropic covariance). Elongated GT get ``ω ≈ 0`` so the
term does not fight the Gaussian overlap loss.

``ω = exp(-log²(w*/h*) / λ²)``, ``ℓ = ω · sin²(2 Δθ)`` with
``Δθ = norm_le90(θ_p − θ_t)``. The ``sin²(2Δθ)`` period is 90° (correct for
squares under le90). ``ω`` uses detached GT ``w*, h*``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _norm_angle_le90(angle: Tensor) -> Tensor:
    """Wrap to ``[-π/2, π/2)`` without importing models (ops must stay leaf)."""
    return torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2


def aspect_gated_angle_loss_per_box(
    pred: Tensor,
    target: Tensor,
    *,
    lam: float = 1.0,
    eps: float = 1e-7,
) -> Tensor:
    """Per-box heading term ``[N]`` (lower is better).

    Args:
        pred: ``[N, 5]`` predicted boxes ``(cx, cy, w, h, angle_rad)``.
        target: ``[N, 5]`` GT boxes (same layout). Aspect gate uses GT ``w, h``.
        lam: Positive scale for the log-aspect Gaussian. ``λ = 1`` → ``ω ≈ 0.62``
            at AR=2, ``ω ≈ 0.30`` at AR=3.
        eps: Clamp on GT width/height before ``log(w/h)``.
    """
    if pred.shape != target.shape or pred.shape[-1] != 5:
        raise ValueError(
            f"Expected matching [N, 5] tensors, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if float(lam) <= 0.0:
        raise ValueError(f"lam must be > 0, got {lam!r}")

    pred_f = pred.float()
    target_f = target.detach().float()
    w = target_f[:, 2].clamp(min=eps)
    h = target_f[:, 3].clamp(min=eps)
    log_ar = torch.log(w / h)
    omega = torch.exp(-(log_ar * log_ar) / (float(lam) * float(lam)))

    dtheta = _norm_angle_le90(pred_f[:, 4] - target_f[:, 4])
    angle_term = torch.sin(2.0 * dtheta).square()
    return (omega * angle_term).to(dtype=pred.dtype)


__all__ = ["aspect_gated_angle_loss_per_box"]
