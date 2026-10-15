"""Rotated RTMDet-S 3× recipe: late cosine, EMA, and decay groups."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from oriented_det.train.config import TrainingConfig, TrainingExperimentConfig
from oriented_det.train.ema import ExpMomentumEMA, exp_momentum
from oriented_det.train.utils import (
    CheckpointManager,
    apply_weight_decay_multipliers,
    create_late_cosine_lr_scheduler,
)


def _recipe():
    root = Path(__file__).resolve().parents[1]
    return TrainingExperimentConfig.load(
        root / "configs" / "rotated_rtmdet" / "dota_le90_3x_s.json"
    )


def test_rr_recipe_loads_rotation_schedule():
    cfg = _recipe()
    assert cfg.model.backbone == "cspnext_s"
    assert cfg.data_loader.batch_size == 8
    assert cfg.training.num_epochs == 36
    assert cfg.training.optimizer == "adamw"
    assert cfg.training.learning_rate == pytest.approx(2.5e-4)
    assert cfg.training.weight_decay == pytest.approx(0.05)
    assert cfg.training.use_amp is False
    assert cfg.training.max_grad_norm is None
    assert cfg.training.lr_warmup_steps == 1000
    assert cfg.training.lr_scheduler_type == "cosine_annealing_late"
    assert cfg.training.lr_scheduler_cosine_start_epoch == 18
    assert cfg.training.lr_scheduler_cosine_eta_min == pytest.approx(1.25e-5)
    assert cfg.training.use_ema is True
    assert cfg.training.ema_momentum == pytest.approx(0.0002)
    assert cfg.training.ema_gamma == pytest.approx(2000)
    assert cfg.training.weight_decay_norm_mult == pytest.approx(0.0)
    assert cfg.training.weight_decay_bias_mult == pytest.approx(0.0)
    assert cfg.preprocessing.enable_mosaic is False
    assert cfg.preprocessing.enable_random_rotate is True
    assert cfg.preprocessing.random_rotate_prob == pytest.approx(0.5)
    assert cfg.preprocessing.random_rotate_angle_range == pytest.approx(180)
    assert cfg.preprocessing.random_rotate_rect_labels == ["storage-tank", "roundabout"]
    assert cfg.model.rtmdet_angle_weight == pytest.approx(0.0)
    assert cfg.model.box_reg_weight == pytest.approx(2.0)
    assert cfg.model.rtmdet_nms_pre == 30000
    assert cfg.production.score_threshold == pytest.approx(0.45)


def test_late_cosine_holds_then_reaches_eta_min():
    base_lr = 2.5e-4
    eta_min = 1.25e-5
    module = nn.Linear(1, 1)
    opt = optim.SGD(module.parameters(), lr=base_lr)
    sch, start, resolved_eta = create_late_cosine_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=36,
            lr_scheduler_cosine_start_epoch=18,
            lr_scheduler_cosine_eta_min=eta_min,
        ),
    )
    assert start == 18
    assert resolved_eta == pytest.approx(eta_min)
    lrs = []
    for _ in range(36):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[16] == pytest.approx(base_lr)
    assert lrs[35] == pytest.approx(eta_min, rel=1e-6, abs=1e-12)
    assert lrs[17] == pytest.approx(base_lr)
    assert lrs[18] < base_lr


def test_exp_momentum_at_zero_and_large_step():
    m = 0.0002
    gamma = 2000.0
    assert exp_momentum(0, m, gamma) == pytest.approx(
        (1.0 - m) * math.exp(-1.0 / gamma) + m
    )
    assert exp_momentum(0, m, gamma) > 0.99
    assert exp_momentum(100_000, m, gamma) == pytest.approx(m, abs=1e-8)


def test_norm_and_bias_weight_decay_groups():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.bn = nn.BatchNorm2d(1)

    module = Tiny()
    groups = apply_weight_decay_multipliers(
        [{"params": [p for p in module.parameters() if p.requires_grad]}],
        module,
        weight_decay=0.05,
        norm_mult=0.0,
        bias_mult=0.0,
    )
    by_id = {}
    for group in groups:
        for param in group["params"]:
            by_id[id(param)] = group["weight_decay"]
    assert by_id[id(module.conv.weight)] == pytest.approx(0.05)
    assert by_id[id(module.conv.bias)] == pytest.approx(0.0)
    assert by_id[id(module.bn.weight)] == pytest.approx(0.0)
    assert by_id[id(module.bn.bias)] == pytest.approx(0.0)

    unchanged = apply_weight_decay_multipliers(
        [{"params": list(module.parameters()), "weight_decay": 0.05}],
        module,
        weight_decay=0.05,
        norm_mult=1.0,
        bias_mult=1.0,
    )
    assert len(unchanged) == 1
    assert unchanged[0]["weight_decay"] == pytest.approx(0.05)


def test_best_checkpoint_holds_ema_weights(tmp_path):
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()
    ema = ExpMomentumEMA(module, momentum=0.0002, gamma=2000.0)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema.update(module)
    with torch.no_grad():
        module.weight.fill_(5.0)
    manager = CheckpointManager(tmp_path, best_metric="mAP", higher_is_better=True)
    epoch_path = manager.save(
        module,
        epoch=0,
        metrics={"mAP": 0.1},
        ema_state_dict=ema.state_dict(),
    )
    backup = ema.apply_to_model(module)
    try:
        best_path = manager.save_best(module, epoch=0, metrics={"mAP": 0.5})
    finally:
        ema.restore_model(module, backup)
    assert best_path is not None
    epoch_ckpt = torch.load(epoch_path, map_location="cpu")
    best_ckpt = torch.load(best_path, map_location="cpu")
    assert epoch_ckpt["model_state_dict"]["weight"][0, 0].item() == pytest.approx(5.0)
    assert "ema_state_dict" in epoch_ckpt
    assert epoch_ckpt["ema_state_dict"]["step"] == 1
    assert best_ckpt["model_state_dict"]["weight"][0, 0].item() == pytest.approx(
        ema.shadow["weight"][0, 0].item()
    )
    assert best_ckpt["model_state_dict"]["weight"][0, 0].item() != pytest.approx(5.0)
