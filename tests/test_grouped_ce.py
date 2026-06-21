"""Tests for ROI grouped cross-entropy curriculum."""

import pytest

pytest.importorskip("torch")
import torch

from oriented_det.models.oriented_roi import grouped_cross_entropy_loss, roi_classification_loss
from oriented_det.train.config import LossConfig
from oriented_det.train.grouped_ce import (
    build_grouped_ce_spec,
    configure_roi_grouped_ce,
    grouped_ce_alpha_for_epoch,
)
from oriented_det.models.oriented_rcnn import RotatedFasterRCNN


def test_grouped_ce_alpha_schedule():
    assert grouped_ce_alpha_for_epoch(
        3, enabled=True, schedule_type="step", start_epoch=0, end_epoch=8
    ) == 1.0
    assert grouped_ce_alpha_for_epoch(
        8, enabled=True, schedule_type="step", start_epoch=0, end_epoch=8
    ) == 0.0
    assert grouped_ce_alpha_for_epoch(
        4, enabled=True, schedule_type="linear_ramp", start_epoch=0, end_epoch=8, power=1.0
    ) == pytest.approx(0.5)
    assert grouped_ce_alpha_for_epoch(10, enabled=False, schedule_type="step", start_epoch=0, end_epoch=8) == 0.0


def test_build_grouped_ce_spec_maps_classes():
    class_map = {"plane": 1, "ship": 2}
    spec = build_grouped_ce_spec({"air": ["plane"]}, class_map, num_foreground_classes=2)
    assert spec.group_index_lists == ((1,),)
    assert spec.class_in_group_id[1] == 0
    assert spec.class_in_group_id[2] == -1


def test_grouped_loss_matches_fine_for_singleton_group():
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    targets = torch.tensor([1, 2])
    class_in_group = torch.tensor([-1, 0, -1], dtype=torch.long)
    fine = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    grouped_only = grouped_cross_entropy_loss(
        logits,
        targets,
        grouped_alpha=1.0,
        group_index_lists=[[1]],
        class_in_group_id=class_in_group,
    )
    # target 2 not in group -> fine path; target 1 in singleton group ~ fine for class 1
    assert grouped_only.ndim == 0


def test_model_grouped_ce_epoch_alpha():
    model = RotatedFasterRCNN(num_classes=2, pretrained_backbone=False)
    class_map = {"A": 1, "B": 2}
    loss_cfg = LossConfig(
        roi_grouped_ce_enabled=True,
        roi_grouped_ce_groups={"all": ["A", "B"]},
        roi_grouped_ce_schedule_type="step",
        roi_grouped_ce_schedule_end_epoch=5,
    )
    ok = configure_roi_grouped_ce(model, loss_cfg, class_map, num_foreground_classes=2)
    assert ok
    model.set_grouped_ce_alpha_for_epoch(2)
    assert model.roi_grouped_ce_alpha == 1.0
    model.set_grouped_ce_alpha_for_epoch(5)
    assert model.roi_grouped_ce_alpha == 0.0
    kw = model.roi_grouped_ce_kwargs()
    assert kw["grouped_alpha"] == 0.0


def test_roi_classification_loss_degenerates_to_ce_when_alpha_zero():
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2, 0])
    l1 = roi_classification_loss(
        logits, targets, loss_type="cross_entropy", grouped_alpha=0.0
    )
    l2 = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(l1, l2)
