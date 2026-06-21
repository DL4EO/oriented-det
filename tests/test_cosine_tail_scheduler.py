"""Tests for PyTorch cosine vs cosine + fixed-LR tail schedulers."""

from __future__ import annotations

import math

import pytest
import torch
from torch import optim

from oriented_det.train.config import TrainingConfig
from oriented_det.train.utils import (
    CosineAnnealingWithFixedTailLR,
    create_cosine_with_tail_lr_scheduler,
    create_pytorch_cosine_lr_scheduler,
    resolve_cosine_with_tail_lengths,
    resolve_pytorch_cosine_t_max,
)


def _closed_cosine_lr(epoch: int, base: float, t_max: int, eta_min: float) -> float:
    if epoch <= 0:
        return base
    if epoch >= t_max:
        return eta_min
    return eta_min + (base - eta_min) * (1.0 + math.cos(math.pi * epoch / t_max)) / 2.0


def test_resolve_pytorch_t_max_legacy():
    t = TrainingConfig(num_epochs=12, lr_scheduler_cosine_t_max=12)
    assert resolve_pytorch_cosine_t_max(t) == 12


def test_resolve_with_tail_lengths():
    t = TrainingConfig(
        num_epochs=24,
        lr_scheduler_cosine_epochs=20,
        lr_scheduler_cosine_tail_epochs=4,
    )
    assert resolve_cosine_with_tail_lengths(t) == (20, 4)


def test_resolve_with_tail_auto_tail():
    t = TrainingConfig(
        num_epochs=24,
        lr_scheduler_cosine_epochs=20,
        lr_scheduler_cosine_tail_epochs=0,
    )
    assert resolve_cosine_with_tail_lengths(t) == (20, 4)


def test_pytorch_cosine_single_cycle():
    base_lr = 0.002
    eta_min = 1e-5
    t_max = 12
    m = torch.nn.Linear(2, 1)
    opt = optim.SGD(m.parameters(), lr=base_lr)
    sch, resolved_t_max, _ = create_pytorch_cosine_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=12,
            lr_scheduler_cosine_t_max=t_max,
            lr_scheduler_cosine_eta_min=eta_min,
        ),
    )
    assert resolved_t_max == 12
    for epoch in range(12):
        sch.step()
        expected = _closed_cosine_lr(epoch + 1, base_lr, t_max, eta_min)
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_pytorch_cosine_restarts_after_t_max():
    """num_epochs > T_max: LR rises again (PyTorch default), not a fixed tail."""
    base_lr = 0.0025
    eta_min = 1e-5
    t_max = 20
    m = torch.nn.Linear(2, 1)
    opt = optim.SGD(m.parameters(), lr=base_lr)
    sch, _, _ = create_pytorch_cosine_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=24,
            lr_scheduler_cosine_epochs=t_max,
            lr_scheduler_cosine_tail_epochs=4,
            lr_scheduler_cosine_eta_min=eta_min,
        ),
    )
    lrs = []
    for _ in range(24):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[19] == pytest.approx(eta_min, rel=1e-6, abs=1e-10)
    assert lrs[20] > lrs[19]
    assert lrs[23] > lrs[20]


def test_with_tail_matches_pytorch_cosine_then_constant():
    base_lr = 0.0025
    eta_min = 1e-5
    tail_lr = 5e-5
    cosine_epochs = 20

    m = torch.nn.Linear(2, 1)
    opt = optim.SGD(m.parameters(), lr=base_lr)
    sch, _, _, _ = create_cosine_with_tail_lr_scheduler(
        opt,
        TrainingConfig(
            num_epochs=24,
            lr_scheduler_cosine_epochs=cosine_epochs,
            lr_scheduler_cosine_tail_epochs=4,
            lr_scheduler_cosine_eta_min=eta_min,
            lr_scheduler_cosine_tail_lr=tail_lr,
        ),
    )
    assert isinstance(sch, CosineAnnealingWithFixedTailLR)

    ref = optim.SGD(m.parameters(), lr=base_lr)
    ref_sch = optim.lr_scheduler.CosineAnnealingLR(
        ref, T_max=cosine_epochs, eta_min=eta_min, last_epoch=-1,
    )

    for epoch in range(24):
        sch.step()
        ref_sch.step()
        if epoch < cosine_epochs:
            expected = ref.param_groups[0]["lr"]
        else:
            expected = tail_lr
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_with_tail_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must equal num_epochs"):
        resolve_cosine_with_tail_lengths(
            TrainingConfig(
                num_epochs=12,
                lr_scheduler_cosine_epochs=10,
                lr_scheduler_cosine_tail_epochs=5,
            )
        )
