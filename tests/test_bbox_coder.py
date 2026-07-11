"""Regression tests for bbox coder primitives and round-trips."""

import math

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.models.bbox_coder import (
    DeltaXYWHBBoxCoder,
    DeltaXYWHAHBBoxCoder,
    MidpointOffsetCoder,
    xyxy_to_obb,
)
from oriented_det.models.horizontal_roi_coder import DeltaXYWHTHBBoxCoder
from oriented_det.models.oriented_rpn import encode_oriented_boxes, decode_oriented_boxes
from oriented_det.models.oriented_rpn import normalize_boxes_to_le90


def _rand_le90_boxes(n: int, scale: float = 256.0) -> torch.Tensor:
    cx = torch.rand(n) * scale
    cy = torch.rand(n) * scale
    w0 = torch.rand(n) * 80.0 + 4.0
    h0 = torch.rand(n) * 80.0 + 4.0
    w = torch.maximum(w0, h0)
    h = torch.minimum(w0, h0)
    a = (torch.rand(n) - 0.5) * math.pi
    return torch.stack([cx, cy, w, h, a], dim=1)


def test_delta_xywh_bbox_coder_round_trip():
    torch.manual_seed(10)
    n = 512
    anchors = _rand_le90_boxes(n)
    # Keep GT close to anchors so decode_rpn_boxes wh_ratio_clip does not saturate.
    gt = anchors.clone()
    gt[:, 0:2] = gt[:, 0:2] + torch.randn(n, 2) * 3.0
    scale = 0.5 + 1.5 * torch.rand(n, 2)  # [0.5, 2.0]
    gt[:, 2:4] = gt[:, 2:4] * scale
    gt = normalize_boxes_to_le90(gt)

    coder = DeltaXYWHBBoxCoder()
    deltas = coder.encode(anchors, gt)
    decoded = coder.decode(anchors, deltas)

    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-4)
    assert torch.isfinite(decoded).all()
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-3)
    # RPN 4-parameter coder keeps angle from anchors (not GT), so only enforce le90 invariants.
    assert torch.all(decoded[:, 2] >= decoded[:, 3])
    assert torch.all(decoded[:, 4] >= -math.pi / 2)
    assert torch.all(decoded[:, 4] < math.pi / 2)


@pytest.mark.parametrize("edge_swap", [True, False])
def test_delta_xywhah_bbox_coder_round_trip(edge_swap: bool):
    torch.manual_seed(11 if edge_swap else 12)
    n = 512
    anchors = _rand_le90_boxes(n)
    gt = _rand_le90_boxes(n)

    coder = DeltaXYWHAHBBoxCoder(norm_factor=2.0, edge_swap=edge_swap)
    deltas = coder.encode(anchors, gt)
    decoded = coder.decode(anchors, deltas, normalize_le90=True)

    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-4)
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-4)
    dtheta = torch.atan2(torch.sin(decoded[:, 4] - gt_norm[:, 4]), torch.cos(decoded[:, 4] - gt_norm[:, 4]))
    assert dtheta.abs().max() < 1e-4


def test_xyxy_to_obb_axis_aligned_conversion():
    xyxy = torch.tensor(
        [
            [0.0, 0.0, 10.0, 20.0],
            [5.0, -3.0, 15.0, 9.0],
        ],
        dtype=torch.float32,
    )
    obb = xyxy_to_obb(xyxy)
    assert obb.shape == (2, 5)
    assert torch.allclose(obb[:, 0], torch.tensor([5.0, 10.0]))
    assert torch.allclose(obb[:, 1], torch.tensor([10.0, 3.0]))
    assert torch.allclose(obb[:, 2], torch.tensor([10.0, 10.0]))
    assert torch.allclose(obb[:, 3], torch.tensor([20.0, 12.0]))
    assert torch.allclose(obb[:, 4], torch.zeros(2))


def test_midpoint_offset_coder_round_trip_for_axis_aligned_gt():
    """Midpoint coder should decode back to axis-aligned GT in a simple stable case."""
    proposals = torch.tensor(
        [
            [0.0, 0.0, 20.0, 10.0],
            [10.0, 10.0, 40.0, 30.0],
            [100.0, 80.0, 140.0, 100.0],
        ],
        dtype=torch.float32,
    )
    gt = xyxy_to_obb(proposals)  # angle=0, exact axis-aligned target

    coder = MidpointOffsetCoder()
    deltas = coder.encode(proposals, gt)
    decoded = coder.decode(proposals, deltas)

    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-4)
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-4)
    dtheta = torch.atan2(torch.sin(decoded[:, 4] - gt_norm[:, 4]), torch.cos(decoded[:, 4] - gt_norm[:, 4]))
    assert dtheta.abs().max() < 1e-4


def test_midpoint_offset_decode_respects_wh_ratio_clip():
    """wh_ratio_clip must clamp dw/dh growth and shrink factors."""
    rois = torch.tensor([[0.0, 0.0, 10.0, 20.0]], dtype=torch.float32)
    # Large dw / dh should be clamped by decode.
    deltas = torch.tensor([[0.0, 0.0, 10.0, -10.0, 0.0, 0.0]], dtype=torch.float32)
    coder = MidpointOffsetCoder()
    decoded = coder.decode(rois, deltas, wh_ratio_clip=0.5)

    # max_ratio=abs(log(0.5)) => exp(clamp) in [0.5, 2.0]
    # roi width=10 -> [5,20], roi height=20 -> [10,40]
    assert 5.0 - 1e-5 <= decoded[0, 2].item() <= 20.0 + 1e-5
    assert 10.0 - 1e-5 <= decoded[0, 3].item() <= 40.0 + 1e-5


def test_delta_xywhth_bbox_coder_round_trip():
    torch.manual_seed(20)
    n = 512
    # Horizontal RoIs (xyxy)
    x1y1 = torch.rand(n, 2) * 256.0
    wh = torch.rand(n, 2) * 80.0 + 4.0
    rois = torch.cat([x1y1, x1y1 + wh], dim=1)
    gt = _rand_le90_boxes(n)
    coder = DeltaXYWHTHBBoxCoder(norm_factor=2.0, edge_swap=True)
    deltas = coder.encode(rois, gt)
    decoded = coder.decode(rois, deltas)
    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-4)
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-3)
    dtheta = torch.atan2(torch.sin(decoded[:, 4] - gt_norm[:, 4]), torch.cos(decoded[:, 4] - gt_norm[:, 4]))
    assert dtheta.abs().max() < 2e-3


def test_delta_xywhth_bbox_coder_proj_xy_with_roi_angle_round_trip():
    torch.manual_seed(23)
    n = 64
    x1y1 = torch.rand(n, 2) * 200.0 + 32.0
    wh = torch.rand(n, 2) * 60.0 + 20.0
    rois = torch.cat([x1y1, x1y1 + wh], dim=1)
    gt = _rand_le90_boxes(n)
    roi_angle = (torch.rand(n) - 0.5) * math.pi

    coder = DeltaXYWHTHBBoxCoder(norm_factor=2.0, edge_swap=True, proj_xy=True)
    deltas = coder.encode(rois, gt, roi_angle=roi_angle)
    decoded = coder.decode(rois, deltas, roi_angle=roi_angle)
    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-3)
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-3)
    dtheta = torch.atan2(
        torch.sin(decoded[:, 4] - gt_norm[:, 4]),
        torch.cos(decoded[:, 4] - gt_norm[:, 4]),
    )
    assert dtheta.abs().max() < 5e-3


def test_delta_xywhth_bbox_coder_proj_xy_matches_global_for_horizontal_rois():
    torch.manual_seed(22)
    n = 128
    x1y1 = torch.rand(n, 2) * 256.0
    wh = torch.rand(n, 2) * 80.0 + 4.0
    rois = torch.cat([x1y1, x1y1 + wh], dim=1)
    gt = _rand_le90_boxes(n)

    coder_global = DeltaXYWHTHBBoxCoder(norm_factor=2.0, edge_swap=True, proj_xy=False)
    coder_local = DeltaXYWHTHBBoxCoder(norm_factor=2.0, edge_swap=True, proj_xy=True)

    deltas_global = coder_global.encode(rois, gt)
    deltas_local = coder_local.encode(rois, gt)
    assert torch.allclose(deltas_global, deltas_local, atol=1e-6)

    decoded_global = coder_global.decode(rois, deltas_global)
    decoded_local = coder_local.decode(rois, deltas_local)
    assert torch.allclose(decoded_global, decoded_local, atol=1e-6)


def test_proj_xy_encode_decode_round_trip():
    torch.manual_seed(21)
    n = 256
    anchors = _rand_le90_boxes(n)
    # Force non-trivial angle so proj_xy differs from global dx/dy.
    anchors[:, 4] = (torch.rand(n) - 0.5) * math.pi
    gt = _rand_le90_boxes(n)
    # Keep GT near anchors for stable dx/dy.
    gt[:, 0:2] = anchors[:, 0:2] + torch.randn(n, 2) * 3.0
    gt = normalize_boxes_to_le90(gt)
    deltas = encode_oriented_boxes(
        anchors,
        gt,
        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        norm_factor=None,
        edge_swap=True,
        proj_xy=True,
    )
    decoded = decode_oriented_boxes(
        anchors,
        deltas,
        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        normalize_le90=True,
        norm_factor=None,
        edge_swap=True,
        proj_xy=True,
    )
    gt_norm = normalize_boxes_to_le90(gt)
    assert torch.allclose(decoded[:, :2], gt_norm[:, :2], atol=1e-3)
    assert torch.allclose(decoded[:, 2:4], gt_norm[:, 2:4], atol=1e-3)
