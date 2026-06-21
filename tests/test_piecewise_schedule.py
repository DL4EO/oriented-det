"""Tests for piecewise epoch schedules."""

import pytest

from oriented_det.train.piecewise_schedule import resolve_piecewise_schedule


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 0.1),
        (19, 0.1),
        (20, 0.05),
        (23, 0.05),
        (24, 0.0),
        (100, 0.0),
    ],
)
def test_resolve_piecewise_schedule_three_segments(epoch: int, expected: float):
    value = resolve_piecewise_schedule(
        epoch,
        boundaries=[20, 24],
        values=[0.1, 0.05, 0.0],
        default=0.0,
    )
    assert value == pytest.approx(expected)


def test_resolve_piecewise_schedule_uses_default_when_unset():
    assert resolve_piecewise_schedule(5, None, None, 0.2) == pytest.approx(0.2)
    assert resolve_piecewise_schedule(5, [], [0.1], 0.2) == pytest.approx(0.2)


def test_rotated_faster_rcnn_iou_weight_schedule():
    from oriented_det.models.oriented_rcnn import RotatedFasterRCNN

    model = RotatedFasterRCNN(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=3,
        roi_box_reg_iou_weight=0.1,
        roi_box_reg_iou_schedule_epochs=[24, 28],
        roi_box_reg_iou_schedule_values=[0.1, 0.05, 0.0],
    )
    model.set_roi_box_reg_iou_weight_for_epoch(0)
    assert model.roi_box_reg_iou_weight == pytest.approx(0.1)
    model.set_roi_box_reg_iou_weight_for_epoch(24)
    assert model.roi_box_reg_iou_weight == pytest.approx(0.05)
    model.set_roi_box_reg_iou_weight_for_epoch(28)
    assert model.roi_box_reg_iou_weight == pytest.approx(0.0)


def test_rotated_faster_rcnn_angle_weight_schedule():
    from oriented_det.models.oriented_rcnn import RotatedFasterRCNN

    model = RotatedFasterRCNN(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=3,
        roi_box_reg_angle_weight=2.0,
        roi_box_reg_angle_schedule_epochs=[12, 24],
        roi_box_reg_angle_schedule_values=[2.0, 2.75, 3.5],
    )
    model.set_roi_box_reg_angle_weight_for_epoch(0)
    assert model.roi_box_reg_angle_weight == pytest.approx(2.0)
    model.set_roi_box_reg_angle_weight_for_epoch(12)
    assert model.roi_box_reg_angle_weight == pytest.approx(2.75)
    model.set_roi_box_reg_angle_weight_for_epoch(24)
    assert model.roi_box_reg_angle_weight == pytest.approx(3.5)
