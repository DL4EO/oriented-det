"""Tests for linear LR decay (Ultralytics-style)."""

from __future__ import annotations

import pytest
import torch
from torch import optim

from pathlib import Path

from oriented_det.train.config import TrainingConfig, TrainingExperimentConfig
from oriented_det.train.utils import (
    WarmupScheduler,
    apply_weight_decay_multipliers,
    create_linear_lr_scheduler,
    format_linear_scheduler_description,
    linear_lr_factor,
)


def test_linear_lr_factor_endpoints():
    assert linear_lr_factor(0, 500, 0.01) == pytest.approx(1.0)
    assert linear_lr_factor(500, 500, 0.01) == pytest.approx(0.01)
    assert linear_lr_factor(250, 500, 0.01) == pytest.approx(0.505)


def test_linear_scheduler_decays_to_end_factor():
    base_lr = 0.01
    end_factor = 0.01
    num_epochs = 10
    model = torch.nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=base_lr)
    sch, resolved_epochs, resolved_factor = create_linear_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=base_lr,
            lr_scheduler_linear_end_factor=end_factor,
        ),
    )
    assert resolved_epochs == num_epochs
    assert resolved_factor == pytest.approx(end_factor)
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr)

    for epoch in range(num_epochs):
        sch.step()
        expected = base_lr * linear_lr_factor(epoch + 1, num_epochs, end_factor)
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6, abs=1e-12)

    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * end_factor)


def test_linear_scheduler_rejects_negative_end_factor():
    model = torch.nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="lr_scheduler_linear_end_factor"):
        create_linear_lr_scheduler(
            opt,
            TrainingConfig(num_epochs=10, lr_scheduler_linear_end_factor=-0.1),
        )


def test_linear_warmup_keeps_ramp_and_ends_at_end_factor():
    base_lr = 0.01
    end_factor = 0.01
    num_epochs = 10
    steps_per_epoch = 3
    warmup_steps = 6
    model = torch.nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=base_lr)
    base, _, _ = create_linear_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=base_lr,
            lr_scheduler_linear_end_factor=end_factor,
        ),
    )
    sch = WarmupScheduler(
        opt,
        base,
        warmup_steps,
        advance_base_during_warmup=True,
    )
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    for epoch in range(num_epochs):
        for _ in range(steps_per_epoch):
            sch.step()
            if sch.current_step <= warmup_steps:
                progress = sch.current_step / warmup_steps
                assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * progress)
        sch.step_epoch()
        if sch.current_step <= warmup_steps:
            progress = sch.current_step / warmup_steps
            assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * progress)

    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * end_factor)


def test_warmup_momentum_ramps_to_sgd_momentum():
    model = torch.nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, nesterov=True)
    base, _, _ = create_linear_lr_scheduler(
        opt,
        TrainingConfig(num_epochs=4, learning_rate=0.01),
    )
    sch = WarmupScheduler(opt, base, warmup_steps=4, warmup_momentum=0.8)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.8)
    sch.step()
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.8 + 0.25 * (0.937 - 0.8))
    for _ in range(3):
        sch.step()
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.937)


def test_warmup_bias_lr_ramps_from_fixed_start():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)
            self.bn = torch.nn.BatchNorm2d(1)

    module = Tiny()
    groups = apply_weight_decay_multipliers(
        [{"params": [p for p in module.parameters()]}],
        module,
        weight_decay=0.0005,
        norm_mult=1.0,
        bias_mult=1.0,
        force_split=True,
    )
    by_name = {}
    for group in groups:
        for param in group["params"]:
            by_name.setdefault(group["name"], []).append(id(param))
    assert id(module.conv.weight) in by_name["decay"]
    assert id(module.conv.bias) in by_name["bias"]
    assert id(module.bn.bias) in by_name["norm"]

    ref_groups = apply_weight_decay_multipliers(
        [{"params": [p for p in module.parameters()]}],
        module,
        weight_decay=0.0005,
        norm_mult=0.0,
        bias_mult=0.0,
        norm_bias_in_bias_group=True,
    )
    ref_by_name = {g["name"]: {id(p) for p in g["params"]} for g in ref_groups}
    assert id(module.bn.bias) in ref_by_name["bias"]
    assert id(module.bn.weight) in ref_by_name["norm"]
    assert id(module.conv.bias) in ref_by_name["bias"]
    assert id(module.conv.weight) in ref_by_name["decay"]

    base_lr = 0.01
    warmup_steps = 4
    opt = optim.SGD(groups, lr=base_lr, momentum=0.937)
    base, _, _ = create_linear_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=10,
            learning_rate=base_lr,
            lr_scheduler_linear_end_factor=0.01,
        ),
    )
    sch = WarmupScheduler(
        opt,
        base,
        warmup_steps,
        advance_base_during_warmup=True,
        warmup_bias_lr=0.1,
    )

    def lr_of(name: str) -> float:
        for group in opt.param_groups:
            if group.get("name") == name:
                return float(group["lr"])
        raise AssertionError(name)

    assert lr_of("bias") == pytest.approx(0.1)
    assert lr_of("decay") == pytest.approx(0.0)
    assert lr_of("norm") == pytest.approx(0.0)

    sch.step()
    sch.step_epoch()
    progress = 1 / warmup_steps
    assert lr_of("decay") == pytest.approx(base_lr * progress)
    assert lr_of("norm") == pytest.approx(base_lr * progress)
    assert lr_of("bias") == pytest.approx(0.1 + progress * (base_lr - 0.1))

    for _ in range(warmup_steps - 1):
        sch.step()
    assert sch.current_step == warmup_steps
    assert sch.warmup_complete is False
    assert lr_of("bias") == pytest.approx(base_lr)
    assert lr_of("decay") == pytest.approx(base_lr)
    sch.step_epoch()
    assert lr_of("bias") == pytest.approx(base_lr)
    assert lr_of("decay") == pytest.approx(base_lr)


def test_linear_scheduler_description_includes_warmup():
    text = format_linear_scheduler_description(500, 0.01, 0.01, 22179)
    assert "LinearLR" in text
    assert "end_factor=0.01" in text
    assert "final_lr=0.0001" in text
    assert "warmup 22179 steps" in text


def test_committed_coco_300e_recipe():
    root = Path(__file__).resolve().parents[1]
    coco = TrainingExperimentConfig.load(root / "configs/yolo_obb/coco_300e_s.json")

    assert coco.dataset.format == "coco"
    assert coco.data_loader.batch_size == 64
    assert coco.training.num_epochs == 300
    assert coco.training.optimizer == "sgd"
    assert coco.training.learning_rate == pytest.approx(0.01)
    assert coco.training.momentum == pytest.approx(0.937)
    assert coco.training.nesterov is True
    assert coco.training.weight_decay == pytest.approx(0.0005)
    assert coco.training.weight_decay_norm_mult == pytest.approx(0.0)
    assert coco.training.weight_decay_bias_mult == pytest.approx(0.0)
    assert coco.training.use_ema is True
    assert coco.training.ema_momentum == pytest.approx(0.0001)
    assert coco.training.ema_gamma == pytest.approx(2000.0)
    assert coco.training.lr_scheduler_type == "linear"
    assert coco.training.lr_scheduler_linear_end_factor == pytest.approx(0.01)
    assert coco.training.lr_warmup_steps == 5547
    assert coco.training.warmup_momentum == pytest.approx(0.8)
    assert coco.training.warmup_bias_lr == pytest.approx(0.1)
    assert coco.training.norm_bias_in_bias_group is True
    assert coco.training.max_grad_norm == pytest.approx(10.0)
    assert coco.model.yolo_obb_tal_topk == 10
    assert coco.model.yolo_obb_stride_cls_prior is True
    assert coco.model.yolo_obb_min_gt_side == pytest.approx(16.0)
    assert coco.training.use_amp is True
    assert coco.enable_albumentation is True
    assert coco.augmentation.p_hsv == pytest.approx(1.0)
    assert coco.augmentation.hsv_h == pytest.approx(0.015)
    assert coco.augmentation.hsv_s == pytest.approx(0.7)
    assert coco.augmentation.hsv_v == pytest.approx(0.4)
    assert coco.augmentation.p_brightness_contrast == pytest.approx(0.0)
    assert coco.augmentation.p_gamma == 0
    assert coco.model.yolo_obb_force_horizontal is True
    assert coco.model.inference_pre_nms_score_threshold == pytest.approx(0.001)
    assert coco.model.final_nms_iou_threshold == pytest.approx(0.7)
    assert coco.model.max_detections_per_image == 300
    assert coco.preprocessing.resize_mode == "pad"
    assert list(coco.preprocessing.target_size) == [640, 640]
    assert coco.preprocessing.enable_flip_horizontal is True
    assert coco.preprocessing.enable_flip_vertical is False
    assert coco.preprocessing.enable_flip_diagonal is False
    assert coco.preprocessing.enable_mosaic is True
    assert list(coco.preprocessing.mosaic_ratio_range) == [1, 1]
    assert coco.preprocessing.mosaic_disable_last_n_epochs == 10
    assert coco.preprocessing.random_scale == pytest.approx(0.5)
    assert coco.preprocessing.random_translate == pytest.approx(0.1)
    assert coco.preprocessing.enable_random_rotate is False
    assert list(coco.preprocessing.normalize_mean) == pytest.approx(
        [0.470113, 0.446900, 0.407686]
    )
    assert coco.evaluation.val_score_threshold == pytest.approx(0.001)
    assert coco.evaluation.final_nms_iou_threshold == pytest.approx(0.7)
    assert coco.evaluation.iou_threshold == pytest.approx(0.5)
    assert list(coco.evaluation.map_iou_thresholds) == [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    ]
    assert coco.evaluation.val_map_every_n_epochs == 10
    assert coco.production.score_threshold == pytest.approx(0.2)
