"""Behavioral tests: values set in `TrainingExperimentConfig` reach the code paths that consume them (todo7)."""

from pathlib import Path
import sys
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from oriented_det.train.config import (
    TrainingExperimentConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
    PreprocessingConfig,
    get_preprocessing_params,
)


def _load_train_module():
    root = Path(__file__).resolve().parents[1]
    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return importlib.import_module("train")


def _minimal_config(model_type: str) -> TrainingExperimentConfig:
    return TrainingExperimentConfig(
        model_type=model_type,
        dataset=None,
        model=ModelConfig(
            backbone="resnet18",
            pretrained_backbone=False,
            trainable_layers=3,
            anchor_scales=[4, 8],
            anchor_ratios=[0.5, 2.0],
            roi_norm_factor=3.0,
            roi_edge_swap=False,
            roi_box_reg_angle_weight=1.7,
            roi_box_reg_iou_weight=0.2,
            use_hbb_for_matching=False,
            inference_pre_nms_score_threshold=0.13,
            rpn_pre_nms_top_n=111,
            rpn_post_nms_top_n=77,
            max_detections_per_image=33,
            rpn_nms_threshold=0.41,
            final_nms_iou_threshold=0.31,
            final_nms_iou_schedule_epochs=[1, 2],
            final_nms_iou_schedule_values=[0.6, 0.4, 0.2],
            roi_inference_top_class_only=True,
        ),
        training=TrainingConfig(),
        loss=LossConfig(loss_type="class_weighted"),
    )


def test_get_preprocessing_params_reads_pad_and_flips_from_config():
    cfg = TrainingExperimentConfig(
        model_type="rotated_faster_rcnn",
        preprocessing=PreprocessingConfig(
            resize_mode="pad",
            target_size=[800],
            pad_size_divisor=64,
            enable_flip_horizontal=False,
            enable_flip_vertical=True,
        ),
    )
    p = get_preprocessing_params(cfg)
    assert p["resize_mode"] == "pad"
    assert p["pad_size_divisor"] == 64
    assert p["enable_flip_horizontal"] is False
    assert p["enable_flip_vertical"] is True
    assert p["target_size"] == (800, 800)


def test_get_preprocessing_params_fixed_mode_tuple_target():
    cfg = TrainingExperimentConfig(
        preprocessing=PreprocessingConfig(
            resize_mode="fixed",
            target_size=[640, 480],
            pad_size_divisor=32,
        ),
    )
    p = get_preprocessing_params(cfg)
    assert p["resize_mode"] == "fixed"
    assert p["target_size"] == (640, 480)


def test_create_model_oriented_rcnn_passes_inference_pre_nms_score_from_config():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("oriented_rcnn")
    device = torch.device("cpu")
    with patch.object(train, "OrientedRCNN", mock_cls):
        model, _ = train.create_model_from_config(
            cfg, num_classes=2, device=device, roi_class_weights=None
        )
    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["backbone_name"] == "resnet18"
    assert call_kw["inference_pre_nms_score_threshold"] == pytest.approx(0.13)
    assert call_kw["anchor_scales"] == [4, 8]
    assert call_kw["anchor_ratios"] == [0.5, 2.0]
    # Oriented R-CNN branch does not take anchor_angles; guard against silent regression
    assert "anchor_angles" not in call_kw
    assert call_kw["roi_inference_top_class_only"] is True


def test_create_model_uses_loss_focal_alpha_for_focal_loss_type():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_faster_rcnn")
    cfg.loss = LossConfig(
        loss_type="focal",
        focal_alpha=0.66,
        focal_gamma=1.5,
    )
    with patch.object(train, "RotatedFasterRCNN", mock_cls):
        _, roi_loss_type = train.create_model_from_config(
            cfg, num_classes=2, device=torch.device("cpu"), roi_class_weights=None
        )
    assert roi_loss_type == "focal"
    k = mock_cls.call_args.kwargs
    assert k["roi_focal_alpha"] == pytest.approx(0.66)
    assert k["roi_focal_gamma"] == pytest.approx(1.5)
    assert k["roi_loss_type"] == "focal"


def test_create_model_uses_loss_cross_entropy_mapping():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_faster_rcnn")
    cfg.loss = LossConfig(loss_type="cross_entropy")
    with patch.object(train, "RotatedFasterRCNN", mock_cls):
        _, roi_loss_type = train.create_model_from_config(
            cfg, num_classes=2, device=torch.device("cpu"), roi_class_weights=None
        )
    assert roi_loss_type == "cross_entropy"
    assert mock_cls.call_args.kwargs["roi_loss_type"] == "cross_entropy"


def test_label_smoothing_from_loss_reaches_model_constructor():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_faster_rcnn")
    cfg.loss = LossConfig(loss_type="class_weighted", label_smoothing=0.07)
    with patch.object(train, "RotatedFasterRCNN", mock_cls):
        train.create_model_from_config(
            cfg, num_classes=2, device=torch.device("cpu"), roi_class_weights=None
        )
    assert mock_cls.call_args.kwargs["roi_label_smoothing"] == pytest.approx(0.07)


def test_oriented_rcnn_dota_3x_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(root / "configs/oriented_rcnn/dota_le90_3x.json")
    assert cfg.model_type == "oriented_rcnn"
    assert cfg.training.num_epochs == 36
    assert cfg.training.lr_scheduler_milestones == [24, 33]
    assert cfg.model.rpn_nms_threshold == pytest.approx(0.8)