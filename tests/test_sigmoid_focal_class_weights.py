"""Sigmoid focal class weights for Rotated FCOS and Rotated RetinaNet."""

from pathlib import Path
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from oriented_det.models.rotated_fcos import RotatedFCOS
from oriented_det.models.rotated_retinanet import (
    RotatedRetinaNet,
    sigmoid_focal_loss_sum,
)
from oriented_det.train.config import LossConfig


def _load_train_module():
    root = Path(__file__).resolve().parents[1]
    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return importlib.import_module("train")


def _unweighted_focal_sum(logits, targets, alpha=0.25, gamma=2.0):
    """Reference copy of the original unweighted sigmoid_focal_loss_sum body."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt).pow(gamma) * ce).sum()


def test_unweighted_sigmoid_focal_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.zeros_like(logits)
    targets[0, 1] = 1.0
    targets[3, 0] = 1.0
    expected = _unweighted_focal_sum(logits, targets)
    got = sigmoid_focal_loss_sum(logits, targets)
    assert torch.equal(got, expected)
    got_none = sigmoid_focal_loss_sum(logits, targets, class_weights=None)
    assert torch.equal(got_none, expected)


def test_weighted_doubling_one_class_doubles_that_column():
    torch.manual_seed(1)
    logits = torch.randn(6, 3)
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1.0
    targets[2, 1] = 1.0
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1 - targets)
    per_elem = alpha_t * (1 - pt).pow(2.0) * ce
    col0 = per_elem[:, 0].sum()

    ones = torch.ones(3)
    doubled = torch.tensor([2.0, 1.0, 1.0])
    base = sigmoid_focal_loss_sum(logits, targets, class_weights=ones)
    weighted = sigmoid_focal_loss_sum(logits, targets, class_weights=doubled)
    assert torch.allclose(weighted - base, col0)
    assert torch.allclose(weighted, base + col0)


def test_roi_schedule_tensor_drops_background_column():
    torch.manual_seed(2)
    logits = torch.randn(4, 2)
    targets = torch.zeros_like(logits)
    fg = torch.tensor([2.0, 0.5])
    with_bg = torch.tensor([9.0, 2.0, 0.5])
    a = sigmoid_focal_loss_sum(logits, targets, class_weights=fg)
    b = sigmoid_focal_loss_sum(logits, targets, class_weights=with_bg)
    assert torch.equal(a, b)


@pytest.mark.parametrize("model_cls", [RotatedFCOS, RotatedRetinaNet])
def test_set_class_weights_maps_foreground_ids_and_ignores_background(model_cls):
    model = model_cls(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
        roi_class_weights={
            "alpha": 2.0,
            "background": 0.1,
        },
    )
    model.set_class_weights({"alpha": 1, "beta": 2})
    weights = model.roi_class_weights
    assert weights is not None
    assert weights.shape == (2,)
    assert weights[0].item() == pytest.approx(2.0)
    assert weights[1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("model_cls", [RotatedFCOS, RotatedRetinaNet])
def test_set_class_weights_tensor_accepts_roi_schedule_shape(model_cls):
    model = model_cls(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
    )
    model.set_class_weights_tensor(torch.tensor([0.3, 2.0, 1.5]))
    assert torch.allclose(model.roi_class_weights, torch.tensor([2.0, 1.5]))


def _minimal_one_stage_cfg(model_type: str, loss_type: str) -> "TrainingExperimentConfig":
    from oriented_det.train.config import TrainingExperimentConfig, ModelConfig, TrainingConfig

    return TrainingExperimentConfig(
        model_type=model_type,
        dataset=None,
        model=ModelConfig(
            backbone="resnet18",
            pretrained_backbone=False,
        ),
        training=TrainingConfig(),
        loss=LossConfig(loss_type=loss_type),
    )


@pytest.mark.parametrize(
    "model_type, model_attr",
    [
        ("rotated_fcos", "RotatedFCOS"),
        ("rotated_retinanet", "RotatedRetinaNet"),
    ],
)
def test_create_model_focal_weighted_sets_weights_focal_leaves_none(model_type, model_attr):
    train = _load_train_module()
    weights = {"alpha": 2.0, "beta": 0.5}
    device = torch.device("cpu")

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg_w = _minimal_one_stage_cfg(model_type, "focal_weighted")
    with patch.object(train, model_attr, mock_cls):
        train.create_model_from_config(
            cfg_w, num_classes=2, device=device, roi_class_weights=weights
        )
    assert mock_cls.call_args.kwargs["roi_class_weights"] == weights

    mock_cls.reset_mock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst
    cfg_u = _minimal_one_stage_cfg(model_type, "focal")
    with patch.object(train, model_attr, mock_cls):
        train.create_model_from_config(
            cfg_u, num_classes=2, device=device, roi_class_weights=None
        )
    assert mock_cls.call_args.kwargs["roi_class_weights"] is None


@pytest.mark.parametrize(
    "model_type, model_cls",
    [
        ("rotated_fcos", RotatedFCOS),
        ("rotated_retinanet", RotatedRetinaNet),
    ],
)
def test_create_model_from_config_applies_weights_on_real_models(model_type, model_cls):
    train = _load_train_module()
    device = torch.device("cpu")
    class_map = {"alpha": 1, "beta": 2}
    weights = {"alpha": 2.0, "beta": 0.5}

    cfg_w = _minimal_one_stage_cfg(model_type, "focal_weighted")
    model_w, _ = train.create_model_from_config(
        cfg_w, num_classes=2, device=device, roi_class_weights=weights
    )
    assert isinstance(model_w, model_cls)
    model_w.set_class_weights(class_map, device=device)
    assert model_w.roi_class_weights is not None
    assert torch.allclose(model_w.roi_class_weights, torch.tensor([2.0, 0.5]))

    cfg_u = _minimal_one_stage_cfg(model_type, "focal")
    model_u, _ = train.create_model_from_config(
        cfg_u, num_classes=2, device=device, roi_class_weights=None
    )
    assert isinstance(model_u, model_cls)
    assert model_u.roi_class_weights is None
    model_u.set_class_weights(class_map, device=device)
    assert model_u.roi_class_weights is None


@pytest.mark.parametrize("model_cls", [RotatedFCOS, RotatedRetinaNet])
def test_class_weight_overrides_win(model_cls):
    train = _load_train_module()
    counts = {"alpha": 1000, "beta": 10}
    overrides = {"alpha": 0.25}
    final, computed = train.compute_class_weights(
        counts, None, "sqrt", class_weight_overrides=overrides
    )
    assert final["alpha"] == pytest.approx(0.25)
    assert computed["alpha"] != pytest.approx(0.25)

    model = model_cls(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
        roi_class_weights=final,
    )
    model.set_class_weights({"alpha": 1, "beta": 2})
    assert model.roi_class_weights[0].item() == pytest.approx(0.25)
    assert model.roi_class_weights[1].item() == pytest.approx(final["beta"])
