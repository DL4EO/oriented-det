"""Tests for tools.train.build_optimizer_param_groups (lr_mult_head, head.* grouping)."""

import pytest
import torch.nn as nn

from oriented_det.train.config import TrainingConfig, TrainingExperimentConfig
from tools.train import build_optimizer_param_groups


class _FakeRetinaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


class _FakeFRCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.rpn_head = nn.Linear(2, 2)
        self.roi_head = nn.Linear(2, 2)


def _exp(**training_kw) -> TrainingExperimentConfig:
    t = TrainingConfig(use_lr_param_groups=True, learning_rate=0.1, **training_kw)
    return TrainingExperimentConfig(training=t)


def test_lr_mult_head_applies_to_retinanet_head():
    m = _FakeRetinaNet()
    cfg = _exp(lr_mult_head=2.0, lr_mult_backbone=0.5, lr_mult_other=1.0)
    _, summary = build_optimizer_param_groups(m, 0.1, 1e-4, cfg)
    assert summary["backbone"]["lr"] == pytest.approx(0.05)
    assert summary["head"]["lr"] == pytest.approx(0.2)
    assert summary["head"]["multiplier"] == pytest.approx(2.0)


def test_head_uses_lr_mult_other_when_lr_mult_head_unset():
    m = _FakeRetinaNet()
    cfg = _exp(lr_mult_head=None, lr_mult_other=3.0, lr_mult_backbone=1.0)
    _, summary = build_optimizer_param_groups(m, 0.1, 1e-4, cfg)
    assert summary["head"]["multiplier"] == pytest.approx(3.0)
    assert summary["head"]["lr"] == pytest.approx(0.3)


def test_lr_mult_head_overrides_rpn_and_roi_multipliers():
    m = _FakeFRCNN()
    cfg = _exp(
        lr_mult_head=1.5,
        lr_mult_rpn=0.1,
        lr_mult_roi=10.0,
        lr_mult_backbone=0.5,
    )
    _, summary = build_optimizer_param_groups(m, 1.0, 1e-4, cfg)
    assert summary["rpn"]["multiplier"] == pytest.approx(1.5)
    assert summary["roi"]["multiplier"] == pytest.approx(1.5)
    assert summary["rpn"]["lr"] == pytest.approx(1.5)
    assert summary["roi"]["lr"] == pytest.approx(1.5)


def test_without_lr_mult_head_frcnn_uses_rpn_roi_defaults():
    m = _FakeFRCNN()
    cfg = _exp(lr_mult_head=None, lr_mult_rpn=0.5, lr_mult_roi=2.0, lr_mult_backbone=1.0)
    _, summary = build_optimizer_param_groups(m, 0.1, 1e-4, cfg)
    assert summary["rpn"]["lr"] == pytest.approx(0.05)
    assert summary["roi"]["lr"] == pytest.approx(0.2)
