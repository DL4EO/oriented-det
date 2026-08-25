"""Tests for aspect-gated heading term used in FCOS decoded aux."""

import math

import pytest
import torch

from oriented_det.ops.gaussian_angle import aspect_gated_angle_loss_per_box
from oriented_det.models.rotated_fcos import _fcos_decoded_aux_per_box


def test_identical_boxes_near_zero():
    pred = torch.tensor([[10.0, 20.0, 8.0, 8.0, 0.3]], dtype=torch.float32, requires_grad=True)
    gt = pred.detach()
    loss = aspect_gated_angle_loss_per_box(pred, gt)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_square_small_angle_error_has_theta_grad():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.2]], dtype=torch.float32, requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]], dtype=torch.float32)
    loss = aspect_gated_angle_loss_per_box(pred, gt)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert abs(float(pred.grad[0, 4])) > 0.0
    assert pred.grad[0, :4].abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_square_90_degree_period_is_near_zero():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0, math.pi / 2]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]], dtype=torch.float32)
    loss = aspect_gated_angle_loss_per_box(pred, gt)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_elongated_gate_is_much_smaller_than_square():
    dtheta = 0.25
    square = aspect_gated_angle_loss_per_box(
        torch.tensor([[0.0, 0.0, 10.0, 10.0, dtheta]]),
        torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]]),
    )
    long_box = aspect_gated_angle_loss_per_box(
        torch.tensor([[0.0, 0.0, 80.0, 10.0, dtheta]]),
        torch.tensor([[0.0, 0.0, 80.0, 10.0, 0.0]]),
    )
    assert square.item() > 0.0
    assert long_box.item() < 0.05 * square.item()


def test_rejects_nonpositive_lambda():
    pred = torch.zeros((1, 5))
    with pytest.raises(ValueError, match="lam"):
        aspect_gated_angle_loss_per_box(pred, pred, lam=0.0)


def test_fcos_aux_angle_weight_zero_matches_gaussian_only():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.3]], dtype=torch.float32, requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]], dtype=torch.float32)
    gauss_only = _fcos_decoded_aux_per_box("kfiou", pred, gt, aux_angle_weight=0.0)
    with_heading = _fcos_decoded_aux_per_box("kfiou", pred, gt, aux_angle_weight=1.0)
    heading = aspect_gated_angle_loss_per_box(pred, gt)
    assert with_heading.item() == pytest.approx(
        (gauss_only + heading).item(), rel=1e-5, abs=1e-5
    )
    assert with_heading.item() > gauss_only.item()
