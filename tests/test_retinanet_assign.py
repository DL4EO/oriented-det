"""Rotated RetinaNet assignment (not the shared two-stage matcher)."""

from unittest.mock import patch

import torch

from oriented_det.models.retinanet_assign import match_retinanet_anchors_to_gt


def _box(cx, cy, w, h, angle=0.0):
    return torch.tensor([[cx, cy, w, h, angle]], dtype=torch.float32)


def test_identical_anchor_is_positive_far_anchor_is_background():
    anchors = torch.tensor(
        [
            [64.0, 64.0, 32.0, 16.0, 0.0],
            [400.0, 400.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = _box(64.0, 64.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert labels[0].item() == 1
    assert matched[0].item() == 0
    assert labels[1].item() == 0
    assert matched[1].item() == -1


def test_low_quality_match_assigns_best_anchor_when_all_ious_below_pos():
    """min_pos_iou=0 still promotes the best-overlapping anchor per GT."""
    anchors = torch.tensor(
        [
            [80.0, 64.0, 32.0, 16.0, 0.0],
            [400.0, 400.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = _box(64.0, 64.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors, gt, positive_iou_threshold=0.99, negative_iou_threshold=0.4
    )
    assert labels[0].item() == 1
    assert matched[0].item() == 0
    assert labels[1].item() == 0


def test_empty_gt_is_all_background():
    anchors = torch.zeros((8, 5), dtype=torch.float32)
    gt = torch.zeros((0, 5), dtype=torch.float32)
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert torch.equal(labels, torch.zeros(8, dtype=torch.long))
    assert torch.equal(matched, torch.full((8,), -1, dtype=torch.long))


def test_ignore_region_marks_non_positive_as_ignore():
    anchors = torch.tensor(
        [
            [64.0, 64.0, 32.0, 16.0, 0.0],
            [200.0, 200.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.zeros((0, 5), dtype=torch.float32)
    ign = _box(200.0, 200.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors, gt, gt_boxes_ignore=ign, ignore_iou_threshold=0.5
    )
    assert labels[0].item() == 0
    assert labels[1].item() == -1
    assert matched[1].item() == -1


def test_lookalike_overrides_ignore_to_background():
    anchors = torch.tensor(
        [
            [200.0, 200.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.zeros((0, 5), dtype=torch.float32)
    region = _box(200.0, 200.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors,
        gt,
        gt_boxes_ignore=region,
        ignore_iou_threshold=0.5,
        gt_boxes_lookalike=region,
        lookalike_iou_threshold=0.5,
    )
    assert labels[0].item() == 0
    assert matched[0].item() == -1


def test_hbb_path_delegates_to_shared_matcher():
    anchors = _box(64.0, 64.0, 32.0, 16.0, 0.3)
    gt = _box(64.0, 64.0, 32.0, 16.0, 1.0)
    sentinel_labels = torch.tensor([1], dtype=torch.long)
    sentinel_matched = torch.tensor([0], dtype=torch.long)
    with patch(
        "oriented_det.models.retinanet_assign.match_oriented_anchors_to_gt",
        return_value=(sentinel_labels, sentinel_matched),
    ) as mocked:
        labels, matched = match_retinanet_anchors_to_gt(
            anchors, gt, use_hbb_for_matching=True
        )
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["use_hbb_for_matching"] is True
    assert labels is sentinel_labels
    assert matched is sentinel_matched


def test_dense_p3_grid_finishes_without_materializing_full_sampling():
    """27-anchor P3-scale grid vs a handful of GTs must stay cheap on CPU."""
    h = w = 64
    num_anchors = 27
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx = (xx + 0.5) * 8.0
    cy = (yy + 0.5) * 8.0
    n_loc = h * w
    anchors = torch.stack(
        [
            cx.reshape(-1).repeat(num_anchors),
            cy.reshape(-1).repeat(num_anchors),
            torch.full((n_loc * num_anchors,), 32.0),
            torch.full((n_loc * num_anchors,), 16.0),
            torch.zeros(n_loc * num_anchors),
        ],
        dim=1,
    )
    gt = torch.tensor(
        [
            [64.0, 64.0, 40.0, 18.0, 0.4],
            [200.0, 180.0, 28.0, 12.0, -0.3],
            [400.0, 400.0, 48.0, 20.0, 0.7],
        ],
        dtype=torch.float32,
    )
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert labels.shape == (n_loc * num_anchors,)
    assert (labels == 1).any()
    assert (labels == 0).any()
    assert matched[labels == 1].min().item() >= 0
