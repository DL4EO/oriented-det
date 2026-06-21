"""Tests for Probabilistic IoU (ProbIoU) helpers."""

import pytest
import torch

from oriented_det.ops.kfiou import mean_auxiliary_box_reg_loss
from oriented_det.ops.probiou import gbb_form, probiou_loss, probiou_loss_per_box


def test_gbb_form_variance_is_w_squared_over_12():
    boxes = torch.tensor([[1.0, 2.0, 6.0, 12.0, 0.25]], dtype=torch.float32)
    gbb = gbb_form(boxes)
    assert gbb[0, 2].item() == pytest.approx(3.0)
    assert gbb[0, 3].item() == pytest.approx(12.0)
    assert gbb[0, 4].item() == pytest.approx(0.25)


def test_probiou_small_when_identical_boxes():
    """Identical OBBs still have non-zero l1 (Gaussian surrogate, not polygon IoU)."""
    pred = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.5]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach()
    per_box = probiou_loss_per_box(pred, gt, mode="l1")
    assert 0.0 <= per_box.item() < 0.1
    mean_loss = probiou_loss(pred, gt, mode="l1", reduction="mean")
    assert mean_loss.item() == per_box.item()


def test_probiou_increases_when_misaligned():
    pred = torch.tensor([[50.0, 50.0, 10.0, 20.0, 0.3]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach().clone()
    loss_aligned = probiou_loss(pred, gt, reduction="mean")

    pred_far = pred.clone().detach().requires_grad_(True)
    pred_far.data[0, 0] += 40.0
    loss_far = probiou_loss(pred_far, gt, reduction="mean")
    assert loss_far.item() > loss_aligned.item()


def test_mean_auxiliary_probiou_backward():
    pred = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach()
    loss = mean_auxiliary_box_reg_loss(pred, gt, loss_type="probiou")
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_probiou_l2_mode_finite():
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.1]], dtype=torch.float32, requires_grad=True)
    gt = torch.tensor([[2.0, 3.0, 3.0, 4.0, 0.2]], dtype=torch.float32)
    loss = probiou_loss(pred, gt, mode="l2", reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_probiou_fp16_inputs_finite():
    device = "cuda"
    pred = torch.tensor(
        [[512.0, 512.0, 800.0, 80.0, 0.6]],
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    gt = torch.tensor([[512.0, 512.0, 800.0, 80.0, 0.7]], dtype=torch.float16, device=device)
    loss = probiou_loss(pred, gt, reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
