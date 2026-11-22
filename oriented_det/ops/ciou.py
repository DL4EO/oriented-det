"""Complete IoU loss for horizontal boxes.

Zheng et al., *Distance-IoU Loss* (https://arxiv.org/abs/1911.08287).
Boxes are ``(cx, cy, w, h)``. The returned loss is ``1 - CIoU``. The aspect
weight is detached, matching the usual training implementation.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor


def _fp32_no_autocast(tensor: Tensor):
    device_type = tensor.device.type
    if device_type in ("cuda", "cpu"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        except (AttributeError, TypeError):
            if device_type == "cuda":
                return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def ciou_loss_per_box(pred: Tensor, target: Tensor, *, eps: float = 1e-7) -> Tensor:
    """Per-box CIoU loss ``[N]``.

    Args:
        pred: ``[N, 4]`` or ``[N, 5+]`` predicted ``(cx, cy, w, h)``. Extra columns are ignored.
        target: matching target boxes. Gradients flow to ``pred`` only when ``target`` requires them.
        eps: Stabilizer for the union, the enclosing diagonal, and the aspect weight.
    """
    if pred.shape != target.shape or pred.shape[-1] < 4:
        raise ValueError(f"Expected matching [N, >=4] tensors, got {pred.shape} vs {target.shape}")

    with _fp32_no_autocast(pred):
        pred_f = pred[..., :4].float()
        target_f = target[..., :4].float()
        px, py, pw, ph = pred_f.unbind(dim=-1)
        gx, gy, gw, gh = target_f.unbind(dim=-1)
        pw = pw.clamp(min=eps)
        ph = ph.clamp(min=eps)
        gw = gw.clamp(min=eps)
        gh = gh.clamp(min=eps)

        b1_x1, b1_x2 = px - pw / 2, px + pw / 2
        b1_y1, b1_y2 = py - ph / 2, py + ph / 2
        b2_x1, b2_x2 = gx - gw / 2, gx + gw / 2
        b2_y1, b2_y2 = gy - gh / 2, gy + gh / 2

        inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
        inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
        inter = inter_w * inter_h
        union = pw * ph + gw * gh - inter + eps
        iou = inter / union

        cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
        ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = (px - gx).pow(2) + (py - gy).pow(2)
        v = (4.0 / math.pi**2) * (torch.atan(pw / ph) - torch.atan(gw / gh)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1.0 + eps))
        ciou = iou - (rho2 / c2 + v * alpha)
        loss = 1.0 - ciou

    return loss.to(dtype=pred.dtype)
