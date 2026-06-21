"""Kalman Filter IoU (KFIoU) surrogate for oriented boxes.

Models each OBB as a 2D Gaussian and uses the Kalman-style fused covariance to
define a differentiable overlap ratio (see Y et al., *The KFIoU Loss for Rotated
Object Detection*). Box tensors are ``[cx, cy, w, h, angle_rad]`` (same as the
rest of this codebase).

This follows the structure of MMRotate's ``kf_iou_loss`` (center Smooth L1 +
overlap term) without MMCV/MMDet dependencies.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch
from torch import Tensor


def _fp32_no_autocast_ctx(tensor: Tensor):
    """Context that disables AMP autocast on the tensor's device.

    KFIoU calls ``torch.det`` / ``torch.linalg.pinv`` on 2x2 covariances. On
    CUDA those fall back to LU factorization which has no fp16/bf16 kernel
    (``"lu_factor_cublas" not implemented for 'Half'``); fp16 also has too
    narrow a dynamic range to hold ``(w/2)**2`` for ~1024 px boxes. We run the
    Gaussian-overlap math in fp32 with autocast off, then cast back.
    """
    device_type = tensor.device.type
    if device_type in ("cuda", "cpu"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        except (AttributeError, TypeError):
            if device_type == "cuda":
                return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def xy_wh_r_to_xy_sigma(xywhr: Tensor) -> Tuple[Tensor, Tensor]:
    """Map oriented boxes to Gaussian mean ``xy`` and covariance ``sigma``.

    Args:
        xywhr: ``[..., 5]`` with ``(cx, cy, w, h, angle)``.

    Returns:
        ``xy`` with shape ``[..., 2]``, ``sigma`` with shape ``[..., 2, 2]``.
    """
    shape = xywhr.shape
    if shape[-1] != 5:
        raise ValueError(f"Expected last dim 5 (xywhr), got {shape[-1]}")
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4].reshape(-1)
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    rot = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    s = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(s.square()).bmm(rot.transpose(-2, -1)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def kfiou_overlap_ratio(
    pred_decode: Tensor,
    targets_decode: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Per-row KFIoU overlap ratio ``Vb / (Vb_p + Vb_t - Vb + eps)`` in ``[0, 1]`` (typically).

    Args:
        pred_decode: ``[N, 5]`` predicted decoded boxes (differentiable).
        targets_decode: ``[N, 5]`` matched GT boxes (usually detached).
        eps: Small constant for numerical stability in the denominator.

    Returns:
        Tensor ``[N]`` overlap ratios. Always returned in ``float32`` so that
        downstream ``-log(kfiou + eps)`` / ``exp(1 - kfiou)`` stay numerically
        stable under AMP (``eps=1e-6`` underflows to 0 in fp16). Autograd
        casts gradients back to the input dtype for backward.
    """
    with _fp32_no_autocast_ctx(pred_decode):
        # Cast inputs (not just sigmas) so that wh.square() in
        # xy_wh_r_to_xy_sigma cannot overflow fp16 for ~1024 px boxes.
        pred_f = pred_decode.float()
        targets_f = targets_decode.float()
        _, sigma_p = xy_wh_r_to_xy_sigma(pred_f)
        _, sigma_t = xy_wh_r_to_xy_sigma(targets_f)

        det_p = sigma_p.det().clamp(min=eps)
        det_t = sigma_t.det().clamp(min=eps)
        vb_p = 4.0 * det_p.sqrt()
        vb_t = 4.0 * det_t.sqrt()

        sigma_sum = sigma_p + sigma_t
        # Kalman gain K = sigma_p @ sigma_sum^{-1}. The sum can be singular or
        # badly conditioned when both covariances are (near) rank-1 with the same
        # kernel (degenerate skinny boxes, same orientation) or under unstable
        # predictions (e.g. LR finder at very high LR). Diagonal jitter helps
        # conditioning; ``pinv`` avoids ``linalg.solve`` hard failures on CUDA when
        # LAPACK still reports a singular/invalid factorization.
        solve_jitter = max(float(eps), 1e-8)
        eye2 = torch.eye(2, device=sigma_sum.device, dtype=sigma_sum.dtype)
        eye2 = eye2.reshape(*([1] * (sigma_sum.dim() - 2)), 2, 2).expand_as(sigma_sum)
        sigma_sum_reg = sigma_sum + solve_jitter * eye2
        pinv_sum = torch.linalg.pinv(sigma_sum_reg)
        k_mat = torch.matmul(sigma_p, pinv_sum)
        sigma_fused = sigma_p - k_mat.bmm(sigma_p)
        det_f = sigma_fused.det()
        vb = 4.0 * det_f.clamp(min=0.0).sqrt()
        vb = torch.where(torch.isfinite(vb), vb, torch.zeros_like(vb))

        denom = vb_p + vb_t - vb + eps
        kfiou = vb / denom.clamp(min=eps)
        kfiou = kfiou.clamp(0.0, 1.0)
    return kfiou


def _normalize_kfiou_fun(fun: Optional[str]) -> Optional[str]:
    if fun is None:
        return None
    s = str(fun).strip().lower()
    if s in ("none", "", "null"):
        return None
    return s


def kfiou_loss_per_box(
    pred_decode: Tensor,
    targets_decode: Tensor,
    *,
    pred: Optional[Tensor] = None,
    target: Optional[Tensor] = None,
    fun: Optional[str] = None,
    beta: float = 1.0 / 9.0,
    eps: float = 1e-6,
) -> Tensor:
    """KFIoU loss terms per box (MMRotate ``kfiou_loss`` without reduction).

    Args:
        pred_decode: ``[N, 5]`` predicted decoded boxes.
        targets_decode: ``[N, 5]`` target decoded boxes.
        pred: Optional ``[N, *]`` whose first two columns are predicted centers;
            defaults to ``pred_decode``.
        target: Optional ``[N, *]`` whose first two columns are GT centers;
            defaults to ``targets_decode``.
        fun: ``None`` / ``\"none\"`` → ``1 - KFIoU``; ``\"ln\"`` → ``-log(KFIoU + eps)``;
            ``\"exp\"`` → ``exp(1 - KFIoU) - 1``.
        beta: Smooth L1 beta for the center term.
        eps: Numerical stabilizer.

    Returns:
        ``[N]`` tensor of per-instance losses (non-negative).
    """
    fun = _normalize_kfiou_fun(fun)
    if pred is None:
        pred = pred_decode
    if target is None:
        target = targets_decode
    xy_p = pred[:, :2]
    xy_t = target[:, :2]

    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta,
    ).sum(dim=-1)

    kfiou = kfiou_overlap_ratio(pred_decode, targets_decode, eps=eps)
    if fun == "ln":
        kf_loss = -torch.log(kfiou + eps)
    elif fun == "exp":
        kf_loss = torch.exp(1.0 - kfiou) - 1.0
    else:
        kf_loss = 1.0 - kfiou

    return (xy_loss + kf_loss).clamp(min=0.0)


def kfiou_loss(
    pred_decode: Tensor,
    targets_decode: Tensor,
    *,
    pred: Optional[Tensor] = None,
    target: Optional[Tensor] = None,
    fun: Optional[str] = None,
    beta: float = 1.0 / 9.0,
    eps: float = 1e-6,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Reduced KFIoU loss (scalar unless ``reduction=\"none\"``)."""
    loss = kfiou_loss_per_box(
        pred_decode,
        targets_decode,
        pred=pred,
        target=target,
        fun=fun,
        beta=beta,
        eps=eps,
    )
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


def mean_auxiliary_box_reg_loss(
    decoded_boxes: Tensor,
    matched_gt: Tensor,
    *,
    loss_type: str = "riou",
    kfiou_fun: Optional[str] = None,
    probiou_mode: Optional[str] = None,
) -> Tensor:
    """Scalar auxiliary loss on decoded positives (mean over boxes).

    Args:
        decoded_boxes: ``[N, 5]`` predictions (typically require grad).
        matched_gt: ``[N, 5]`` matched ground truth (typically detached).
        loss_type: ``\"riou\"`` for ``mean(1 - rIoU)`` (sampling/GPU backend via
            :func:`rotated_ops.pairwise_rotated_iou`); ``\"kfiou\"`` for reduced
            KFIoU loss (Gaussian overlap + center Smooth L1); ``\"probiou\"`` for
            mean ProbIoU (see :func:`probiou.probiou_loss`).
        kfiou_fun: When ``loss_type=\"kfiou\"``, optional ``ln`` / ``exp`` (see
            :func:`kfiou_loss`); ``none`` uses ``1 - KFIoU``.
        probiou_mode: When ``loss_type=\"probiou\"``, ``\"l1\"`` (default) or ``\"l2\"``.
    """
    lt = (loss_type or "riou").strip().lower()
    if lt == "riou":
        from .rotated_ops import pairwise_rotated_iou

        iou_matrix = pairwise_rotated_iou(decoded_boxes, matched_gt)
        pairwise_iou = torch.diagonal(iou_matrix, offset=0).clamp(0.0, 1.0)
        return 1.0 - pairwise_iou.mean()
    if lt == "kfiou":
        return kfiou_loss(
            decoded_boxes,
            matched_gt,
            fun=_normalize_kfiou_fun(kfiou_fun),
            reduction="mean",
        )
    if lt == "probiou":
        from .probiou import _normalize_probiou_mode, probiou_loss

        return probiou_loss(
            decoded_boxes,
            matched_gt,
            mode=_normalize_probiou_mode(probiou_mode),
            reduction="mean",
        )
    raise ValueError(
        f"Unknown loss_type {loss_type!r} for auxiliary box loss; "
        "use 'riou', 'kfiou', or 'probiou'"
    )


__all__ = [
    "xy_wh_r_to_xy_sigma",
    "kfiou_overlap_ratio",
    "kfiou_loss_per_box",
    "kfiou_loss",
    "mean_auxiliary_box_reg_loss",
]
