"""Tests for differentiable rotated IoU (train-loss op, not sampling)."""

import math

import pytest
import torch

from oriented_det.geometry.rbox import RBox
from oriented_det.ops.diff_iou_rotated import diff_iou_rotated_2d, riou_loss_per_box
from oriented_det.ops.iou import rbox_iou


def _exact_iou(a, b) -> float:
    return float(
        rbox_iou(
            RBox(*[float(x) for x in a]),
            RBox(*[float(x) for x in b]),
            intersection_backend="auto",
        )
    )


def test_identical_boxes_iou_one_loss_zero():
    box = torch.tensor([[10.0, 20.0, 30.0, 12.0, 0.4]], dtype=torch.float32, requires_grad=True)
    iou = diff_iou_rotated_2d(box, box.detach())
    loss = riou_loss_per_box(box, box.detach())
    assert float(iou) == pytest.approx(1.0, abs=2e-3)
    assert float(loss) == pytest.approx(0.0, abs=2e-3)


def test_axis_aligned_partial_overlap_matches_exact():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], dtype=torch.float32)
    b = torch.tensor([[1.0, 0.0, 2.0, 2.0, 0.0]], dtype=torch.float32)
    iou = float(diff_iou_rotated_2d(a, b))
    assert iou == pytest.approx(_exact_iou(a[0], b[0]), abs=3e-3)
    assert iou == pytest.approx(1.0 / 3.0, abs=3e-3)


def test_rotated_pairs_close_to_exact_polygon_iou():
    torch.manual_seed(0)
    pred = torch.tensor(
        [
            [50.0, 40.0, 20.0, 8.0, 0.3],
            [10.0, 10.0, 16.0, 16.0, math.pi / 4],
            [100.0, 80.0, 40.0, 10.0, -0.7],
        ],
        dtype=torch.float32,
    )
    tgt = torch.tensor(
        [
            [52.0, 41.0, 18.0, 9.0, 0.35],
            [10.0, 10.0, 16.0, 16.0, 0.0],
            [130.0, 80.0, 40.0, 10.0, -0.7],
        ],
        dtype=torch.float32,
    )
    ious = diff_iou_rotated_2d(pred, tgt)
    for i in range(pred.size(0)):
        exact = _exact_iou(pred[i], tgt[i])
        assert float(ious[i]) == pytest.approx(exact, abs=2e-2)


def test_non_overlap_and_degenerate_are_finite():
    far = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.2]], dtype=torch.float32)
    other = torch.tensor([[100.0, 100.0, 4.0, 2.0, 0.2]], dtype=torch.float32)
    iou = diff_iou_rotated_2d(far, other)
    assert torch.isfinite(iou).all()
    assert float(iou) == pytest.approx(0.0, abs=1e-4)
    skinny = torch.tensor([[0.0, 0.0, 1e-8, 10.0, 0.1]], dtype=torch.float32)
    loss = riou_loss_per_box(skinny, other)
    assert torch.isfinite(loss).all()


def test_backward_has_xywh_and_angle_grads():
    pred = torch.tensor([[0.0, 0.0, 20.0, 8.0, 0.25]], dtype=torch.float32, requires_grad=True)
    gt = torch.tensor([[1.0, 0.5, 18.0, 9.0, 0.0]], dtype=torch.float32)
    loss = riou_loss_per_box(pred, gt)
    assert float(loss) > 0.0
    loss.backward()
    assert pred.grad is not None
    assert float(pred.grad[0, :4].norm()) > 0.0
    assert abs(float(pred.grad[0, 4])) > 0.0


def test_empty_pairs():
    empty = torch.zeros((0, 5), dtype=torch.float32, requires_grad=True)
    iou = diff_iou_rotated_2d(empty, empty.detach())
    assert tuple(iou.shape) == (0,)
    loss = riou_loss_per_box(empty, empty.detach())
    assert tuple(loss.shape) == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu_and_backward():
    pred_cpu = torch.tensor([[12.0, 8.0, 16.0, 6.0, 0.4]], dtype=torch.float32, requires_grad=True)
    gt = torch.tensor([[13.0, 8.5, 15.0, 7.0, 0.2]], dtype=torch.float32)
    iou_cpu = diff_iou_rotated_2d(pred_cpu, gt)
    pred_gpu = pred_cpu.detach().cuda().requires_grad_(True)
    iou_gpu = diff_iou_rotated_2d(pred_gpu, gt.cuda())
    assert float(iou_gpu.cpu()) == pytest.approx(float(iou_cpu.detach()), abs=2e-4)
    riou_loss_per_box(pred_gpu, gt.cuda()).backward()
    assert pred_gpu.grad is not None
    assert float(pred_gpu.grad.norm()) > 0.0
