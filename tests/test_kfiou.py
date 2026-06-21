"""Tests for Kalman Filter IoU (KFIoU) helpers."""

import pytest
import torch

from oriented_det.ops.kfiou import (
    kfiou_loss,
    kfiou_overlap_ratio,
    mean_auxiliary_box_reg_loss,
    xy_wh_r_to_xy_sigma,
)


def test_xy_wh_r_to_xy_sigma_shape_and_symmetric_positive_definite():
    b = torch.tensor([[10.0, 20.0, 4.0, 8.0, 0.5]], dtype=torch.float32)
    xy, sigma = xy_wh_r_to_xy_sigma(b)
    assert xy.shape == (1, 2)
    assert sigma.shape == (1, 2, 2)
    assert torch.all(torch.linalg.eigvalsh(sigma[0]) > 0)


def test_kfiou_surrogate_increases_loss_when_misaligned():
    """KFIoU is a Gaussian surrogate, not polygon IoU; it stays in [0, 1] and rises when centers diverge."""
    pred = torch.tensor([[50.0, 50.0, 10.0, 20.0, 0.3]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach().clone()
    overlap = kfiou_overlap_ratio(pred, gt)
    assert 0.0 <= overlap.item() <= 1.0
    loss_aligned = kfiou_loss(pred, gt, reduction="mean")

    pred_far = pred.clone().detach().requires_grad_(True)
    pred_far.data[0, 0] += 40.0
    loss_far = kfiou_loss(pred_far, gt, reduction="mean")
    assert loss_far.item() > loss_aligned.item()


def test_mean_auxiliary_riou_zero_when_polygon_iou_perfect():
    pred = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach()
    lr = mean_auxiliary_box_reg_loss(pred, gt, loss_type="riou")
    assert lr.item() == pytest.approx(0.0, abs=1e-5)


def test_mean_auxiliary_kfiou_backward():
    pred = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach()
    lk = mean_auxiliary_box_reg_loss(pred, gt, loss_type="kfiou")
    assert torch.isfinite(lk)
    lk.backward()
    assert pred.grad is not None


def test_mean_auxiliary_unknown_loss_type_raises():
    pred = torch.zeros((1, 5))
    gt = pred.clone()
    with pytest.raises(ValueError, match="Unknown loss_type"):
        mean_auxiliary_box_reg_loss(pred, gt, loss_type="bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 dispatch")
def test_kfiou_overlap_fp16_runs_without_lu_factor_error():
    """Regression test for `lu_factor_cublas` not implemented for `Half`.

    KFIoU's ``det`` / ``pinv`` have no fp16 CUDA kernel, and fp16's
    dynamic range can overflow ``(w/2)**2`` for ~1024 px boxes. The op must
    upcast internally and stay finite under AMP-like fp16 inputs.
    """
    device = "cuda"
    pred = torch.tensor(
        [[512.0, 512.0, 800.0, 80.0, 0.6], [200.0, 300.0, 60.0, 30.0, -0.3]],
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    gt = torch.tensor(
        [[512.0, 512.0, 800.0, 80.0, 0.7], [200.0, 300.0, 60.0, 30.0, 0.0]],
        dtype=torch.float16,
        device=device,
    )
    overlap = kfiou_overlap_ratio(pred, gt)
    assert torch.isfinite(overlap).all()
    assert overlap.dtype == torch.float32

    loss = kfiou_loss(pred, gt, fun="ln", reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_kfiou_overlap_near_degenerate_boxes_finite():
    """sigma_p + sigma_t can be singular for collinear rank-1 covariances; must not crash."""
    pred = torch.tensor(
        [[100.0, 100.0, 1e-7, 50.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gt = torch.tensor([[100.0, 100.0, 1e-7, 50.0, 0.0]], dtype=torch.float32)
    overlap = kfiou_overlap_ratio(pred, gt)
    assert torch.isfinite(overlap).all()
    loss = kfiou_loss(pred, gt, reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_kfiou_overlap_dtype_invariance_cpu():
    """Overlap ratio in fp32 should match between fp32 and fp16 inputs (CPU)."""
    pred32 = torch.tensor(
        [[100.0, 100.0, 80.0, 30.0, 0.4]], dtype=torch.float32
    )
    gt32 = torch.tensor(
        [[100.0, 100.0, 80.0, 30.0, 0.5]], dtype=torch.float32
    )
    o32 = kfiou_overlap_ratio(pred32, gt32)
    o16 = kfiou_overlap_ratio(pred32.half(), gt32.half())
    assert torch.allclose(o32, o16, atol=1e-3)
