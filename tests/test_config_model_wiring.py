"""Regression: config fields must reach model constructors (no silent defaults masking bugs)."""

from pathlib import Path
import sys
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from oriented_det.train.config import TrainingExperimentConfig, ModelConfig, TrainingConfig, LossConfig


def _load_train_module():
    """Import tools/train.py through the same top-level path used by tools."""
    root = Path(__file__).resolve().parents[1]
    tools_dir = str(root / "tools")
    # tools/train.py is imported as top-level "train" with tools/ on sys.path.
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
            roi_proj_xy=True,
            roi_box_reg_angle_weight=1.7,
            roi_box_reg_aux_weight=0.15,
            roi_min_pos_iou=0.37,
            use_hbb_for_matching=False,
            inference_pre_nms_score_threshold=0.13,
            rpn_pre_nms_top_n=111,
            rpn_post_nms_top_n=77,
            max_detections_per_image=33,
            rpn_nms_threshold=0.41,
            final_nms_iou_threshold=0.31,
            final_nms_iou_schedule_epochs=[1, 2],
            final_nms_iou_schedule_values=[0.6, 0.4, 0.2],
            roi_box_reg_aux_schedule_epochs=[10, 20],
            roi_box_reg_aux_schedule_values=[0.2, 0.1, 0.0],
            roi_inference_top_class_only=True,
            roi_box_reg_aux_loss_type="smooth_l1",
            roi_box_reg_kfiou_fun="ln",
            roi_box_reg_main_loss_type="probiou",
            roi_box_reg_norm="positives_only",
            rpn_positive_iou_threshold=0.51,
            rpn_negative_iou_threshold=0.39,
            fpn_returned_layers=[2, 3, 4],
            fpn_strides=[8, 16, 32, 64, 128],
            fpn_extra_level=True,
            anchor_octave_base_scale=4.0,
            anchor_scales_per_octave=3,
            retinanet_stacked_convs=4,
            box_reg_loss_type="l1",
            box_reg_weight=0.9,
            fcos_stacked_convs=4,
            fcos_center_sampling=True,
            fcos_center_sample_radius=1.5,
            fcos_norm_on_bbox=True,
            fcos_centerness_on_reg=True,
            fcos_scale_angle=True,
            fcos_angle_weight=1.25,
            fcos_nms_pre=1500,
            fcos_regress_ranges=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, 1e8]],
            aux_loss_type="kfiou",
            aux_loss_weight=0.1,
            aux_angle_weight=0.5,
            aux_angle_lambda=1.5,
        ),
        training=TrainingConfig(),
        loss=LossConfig(loss_type="class_weighted"),
    )


def test_create_model_retinanet_passes_pre_nms_score_from_config():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_retinanet")
    cfg.model.nms_class_agnostic = True
    device = torch.device("cpu")
    with patch.object(train, "RotatedRetinaNet", mock_cls):
        model, _ = train.create_model_from_config(
            cfg, num_classes=2, device=device, roi_class_weights=None
        )
    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["backbone_name"] == "resnet18"
    assert call_kw["pretrained_backbone"] is False
    assert call_kw["trainable_layers"] == 3
    assert "anchor_angles" not in call_kw
    assert call_kw["anchor_scales"] == [4, 8]
    assert call_kw["anchor_ratios"] == [0.5, 2.0]
    assert call_kw["norm_factor"] == pytest.approx(3.0)
    assert call_kw["edge_swap"] is False
    assert call_kw["box_reg_aux_weight"] == pytest.approx(0.15)
    assert call_kw["box_reg_aux_loss_type"] == "smooth_l1"
    assert call_kw["box_reg_kfiou_fun"] == "ln"
    assert call_kw["use_hbb_for_matching"] is False
    assert call_kw["positive_iou_threshold"] == pytest.approx(0.51)
    assert call_kw["negative_iou_threshold"] == pytest.approx(0.39)
    assert call_kw["returned_layers"] == [2, 3, 4]
    assert call_kw["fpn_strides"] == [8, 16, 32, 64, 128]
    assert call_kw["fpn_extra_level"] is True
    assert call_kw["octave_base_scale"] == pytest.approx(4.0)
    assert call_kw["scales_per_octave"] == 3
    assert call_kw["stacked_convs"] == 4
    assert call_kw["box_reg_loss_type"] == "l1"
    assert call_kw["box_reg_main_loss_type"] == "probiou"
    assert call_kw["reg_sample_size_per_image"] == 512
    assert call_kw["box_reg_weight"] == pytest.approx(0.9)
    assert call_kw["score_threshold"] == pytest.approx(0.13)
    assert call_kw["final_nms_iou_threshold"] == pytest.approx(0.31)
    assert call_kw["max_detections_per_image"] == 33
    assert call_kw["nms_class_agnostic"] is True
    assert call_kw["final_nms_iou_schedule_epochs"] == [1, 2]
    assert call_kw["final_nms_iou_schedule_values"] == [0.6, 0.4, 0.2]
    assert call_kw["roi_box_reg_aux_schedule_epochs"] == [10, 20]
    assert call_kw["roi_box_reg_aux_schedule_values"] == [0.2, 0.1, 0.0]


def test_create_model_fcos_passes_config_fields():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_fcos")
    cfg.model.nms_class_agnostic = True
    cfg.loss = LossConfig(loss_type="focal", focal_alpha=0.25, focal_gamma=2.0)
    device = torch.device("cpu")
    with patch.object(train, "RotatedFCOS", mock_cls):
        model, _ = train.create_model_from_config(
            cfg, num_classes=3, device=device, roi_class_weights=None
        )
    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["backbone_name"] == "resnet18"
    assert call_kw["returned_layers"] == [2, 3, 4]
    assert call_kw["fpn_strides"] == [8, 16, 32, 64, 128]
    assert call_kw["fpn_extra_level"] is True
    assert call_kw["stacked_convs"] == 4
    assert call_kw["center_sampling"] is True
    assert call_kw["center_sample_radius"] == pytest.approx(1.5)
    assert call_kw["norm_on_bbox"] is True
    assert call_kw["centerness_on_reg"] is True
    assert call_kw["scale_angle"] is True
    assert call_kw["angle_weight"] == pytest.approx(1.25)
    assert call_kw["nms_pre"] == 1500
    assert call_kw["box_reg_loss_type"] == "l1"
    assert call_kw["aux_loss_type"] == "kfiou"
    assert call_kw["aux_loss_weight"] == pytest.approx(0.1)
    assert call_kw["aux_angle_weight"] == pytest.approx(0.5)
    assert call_kw["aux_angle_lambda"] == pytest.approx(1.5)
    assert call_kw["box_reg_weight"] == pytest.approx(0.9)
    assert call_kw["score_threshold"] == pytest.approx(0.13)
    assert call_kw["final_nms_iou_threshold"] == pytest.approx(0.31)
    assert call_kw["max_detections_per_image"] == 33
    assert call_kw["nms_class_agnostic"] is True
    assert call_kw["focal_alpha"] == pytest.approx(0.25)
    assert call_kw["focal_gamma"] == pytest.approx(2.0)
    assert call_kw["regress_ranges"] == [
        (-1, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1e8),
    ]


def test_fcos_riou_1x_recipe_loads_lr_and_loss_type():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(root / "configs" / "rotated_fcos" / "dota_le90_1x.json")
    assert cfg.model_type == "rotated_fcos"
    assert cfg.model.box_reg_loss_type == "riou"
    assert cfg.training.learning_rate == pytest.approx(0.0025)
    assert float(cfg.model.aux_loss_weight or 0.0) == pytest.approx(0.0)


def test_fcos_riou_3x_recipe_keeps_lr_and_extends_schedule():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(root / "configs" / "rotated_fcos" / "dota_le90_3x.json")
    assert cfg.model_type == "rotated_fcos"
    assert cfg.model.box_reg_loss_type == "riou"
    assert cfg.training.learning_rate == pytest.approx(0.0025)
    assert cfg.training.num_epochs == 36
    assert cfg.training.lr_scheduler_milestones == [24, 33]
    assert cfg.training.lr_warmup_steps == 2000
    assert float(cfg.model.aux_loss_weight or 0.0) == pytest.approx(0.0)


def test_fcos_l1_kfiou_aux_1x_recipe_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(
        root / "configs" / "rotated_fcos" / "dota_le90_1x_l1_kfiou_aux.json"
    )
    assert cfg.model_type == "rotated_fcos"
    assert cfg.model.box_reg_loss_type == "l1"
    assert cfg.model.aux_loss_type == "kfiou"
    assert cfg.model.aux_loss_weight == pytest.approx(0.1)
    assert cfg.training.learning_rate == pytest.approx(0.00025)


def test_load_model_from_checkpoint_retinanet_passes_fpn_and_anchor_config():
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_retinanet")
    cfg.num_classes = 2
    cfg.class_names = ["a", "b"]

    fake_state = {
        "head.conv_cls.weight": torch.zeros(4, 256, 1, 1),
        "head.conv_cls.bias": torch.zeros(4),
        "head.conv_bbox.weight": torch.zeros(10, 256, 1, 1),
        "head.conv_bbox.bias": torch.zeros(10),
    }

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "RotatedRetinaNet", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", side_effect=lambda p: p), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        model, _, _ = ckpt_mod.load_model_from_checkpoint("/tmp/fake.pth", "/tmp/config.json", "cpu")

    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["backbone_name"] == "resnet18"
    assert call_kw["returned_layers"] == [2, 3, 4]
    assert call_kw["fpn_strides"] == [8, 16, 32, 64, 128]
    assert call_kw["fpn_extra_level"] is True
    assert call_kw["octave_base_scale"] == pytest.approx(4.0)
    assert call_kw["scales_per_octave"] == 3
    assert call_kw["stacked_convs"] == 4
    assert call_kw["norm_factor"] == pytest.approx(3.0)


def test_load_model_from_checkpoint_prefers_pretrained_sidecar_for_canonical_config(tmp_path):
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_retinanet")
    cfg.num_classes = 2
    fake_state = {
        "head.conv_cls.weight": torch.zeros(4, 256, 1, 1),
        "head.conv_bbox.weight": torch.zeros(10, 256, 1, 1),
    }
    canonical = tmp_path / "configs" / "rotated_retinanet" / "dota_le90_1x.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "pretrained" / "rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json"
    sidecar.parent.mkdir()
    sidecar.write_text("{}", encoding="utf-8")
    ckpt = sidecar.with_suffix(".pth")
    ckpt.write_bytes(b"x")

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "RotatedRetinaNet", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", return_value=ckpt), patch(
        "oriented_det.pretrained.resolve_checkpoint_sidecar_config", return_value=sidecar
    ), patch("oriented_det.pretrained.resolve_checkpoint_source_recipe", return_value=str(canonical)), patch.object(
        ckpt_mod, "_config_matches_source_recipe", return_value=True
    ), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        ckpt_mod.load_model_from_checkpoint(str(ckpt), str(canonical), "cpu")

    mock_cfg_cls.load.assert_called_once_with(sidecar)


def test_load_model_from_checkpoint_keeps_different_explicit_config(tmp_path):
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_retinanet")
    cfg.num_classes = 2
    fake_state = {
        "head.conv_cls.weight": torch.zeros(4, 256, 1, 1),
        "head.conv_bbox.weight": torch.zeros(10, 256, 1, 1),
    }
    explicit_config = tmp_path / "configs" / "rotated_retinanet" / "custom.json"
    explicit_config.parent.mkdir(parents=True)
    explicit_config.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "pretrained" / "rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json"
    sidecar.parent.mkdir()
    sidecar.write_text("{}", encoding="utf-8")
    ckpt = sidecar.with_suffix(".pth")
    ckpt.write_bytes(b"x")

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "RotatedRetinaNet", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", return_value=ckpt), patch(
        "oriented_det.pretrained.resolve_checkpoint_sidecar_config", return_value=sidecar
    ), patch("oriented_det.pretrained.resolve_checkpoint_source_recipe", return_value="configs/rotated_retinanet/dota_le90_1x.json"), patch.object(
        ckpt_mod, "_config_matches_source_recipe", return_value=False
    ), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        ckpt_mod.load_model_from_checkpoint(str(ckpt), str(explicit_config), "cpu")

    mock_cfg_cls.load.assert_called_once_with(explicit_config)


def test_infer_num_classes_from_checkpoint_retinanet_uses_head_shapes(tmp_path):
    from oriented_det.runtime.checkpoint import infer_num_classes_from_checkpoint

    checkpoint_path = tmp_path / "retinanet.pth"
    torch.save(
        {
            "model_state_dict": {
                "head.conv_bbox.weight": torch.zeros(45, 256, 3, 3),
                "head.conv_cls.weight": torch.zeros(135, 256, 3, 3),
            }
        },
        checkpoint_path,
    )

    assert infer_num_classes_from_checkpoint(str(checkpoint_path), "rotated_retinanet") == 15


def test_load_model_from_checkpoint_retinanet_infers_missing_num_classes():
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_retinanet")
    cfg.num_classes = None

    fake_state = {
        "head.conv_cls.weight": torch.zeros(135, 256, 1, 1),
        "head.conv_cls.bias": torch.zeros(135),
        "head.conv_bbox.weight": torch.zeros(45, 256, 1, 1),
        "head.conv_bbox.bias": torch.zeros(45),
    }

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "RotatedRetinaNet", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", side_effect=lambda p: p), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        model, _, _ = ckpt_mod.load_model_from_checkpoint("/tmp/fake.pth", "/tmp/config.json", "cpu")

    assert model is inst
    assert mock_cls.call_args.kwargs["num_classes"] == 15


def test_create_model_rcnn_passes_inference_pre_nms_score_from_config():
    train = _load_train_module()
    mock_cls = MagicMock()
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_faster_rcnn")
    device = torch.device("cpu")
    with patch.object(train, "RotatedFasterRCNN", mock_cls):
        model, _ = train.create_model_from_config(
            cfg, num_classes=2, device=device, roi_class_weights=None
        )
    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["backbone_name"] == "resnet18"
    assert call_kw["pretrained_backbone"] is False
    assert call_kw["trainable_layers"] == 3
    assert "anchor_angles" not in call_kw
    assert call_kw["anchor_scales"] == [4, 8]
    assert call_kw["anchor_ratios"] == [0.5, 2.0]
    assert call_kw["roi_norm_factor"] == pytest.approx(3.0)
    assert call_kw["roi_edge_swap"] is False
    assert call_kw["roi_proj_xy"] is True
    assert call_kw["roi_box_reg_angle_weight"] == pytest.approx(1.7)
    assert call_kw["roi_box_reg_aux_weight"] == pytest.approx(0.15)
    assert call_kw["roi_box_reg_aux_loss_type"] == "smooth_l1"
    assert call_kw["roi_box_reg_kfiou_fun"] == "ln"
    assert call_kw["roi_box_reg_main_loss_type"] == "probiou"
    assert call_kw["roi_box_reg_norm"] == "positives_only"
    assert call_kw["roi_min_pos_iou"] == pytest.approx(0.37)
    assert call_kw["use_hbb_for_matching"] is False
    assert call_kw["inference_pre_nms_score_threshold"] == pytest.approx(0.13)
    assert call_kw["rpn_pre_nms_top_n"] == 111
    assert call_kw["rpn_post_nms_top_n"] == 77
    assert call_kw["max_detections_per_image"] == 33
    assert call_kw["rpn_nms_threshold"] == pytest.approx(0.41)
    assert call_kw["final_nms_iou_threshold"] == pytest.approx(0.31)
    assert call_kw["final_nms_iou_schedule_epochs"] == [1, 2]
    assert call_kw["final_nms_iou_schedule_values"] == [0.6, 0.4, 0.2]
    assert call_kw["roi_box_reg_aux_schedule_epochs"] == [10, 20]
    assert call_kw["roi_box_reg_aux_schedule_values"] == [0.2, 0.1, 0.0]
    assert call_kw["roi_inference_top_class_only"] is True


def test_load_model_from_checkpoint_rotated_faster_rcnn_passes_roi_proj_xy():
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("rotated_faster_rcnn")
    cfg.num_classes = 2
    cfg.model.roi_proj_xy = True
    fake_state = {
        "roi_head.fc_cls.weight": torch.zeros(4, 1024),
        "roi_head.fc_cls.bias": torch.zeros(4),
    }

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "RotatedFasterRCNN", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", side_effect=lambda p: p), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        model, _, _ = ckpt_mod.load_model_from_checkpoint("/tmp/fake.pth", "/tmp/config.json", "cpu")

    assert model is inst
    assert mock_cls.call_args.kwargs["roi_proj_xy"] is True
    assert "roi_use_hbb_for_matching" not in mock_cls.call_args.kwargs


def test_load_model_from_checkpoint_oriented_rcnn_passes_roi_use_hbb_for_matching():
    from oriented_det.runtime import checkpoint as ckpt_mod

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.load_state_dict = MagicMock()
    inst.to = MagicMock(return_value=inst)
    inst.eval = MagicMock()
    mock_cls.return_value = inst

    cfg = _minimal_config("oriented_rcnn")
    cfg.num_classes = 2
    cfg.model.roi_use_hbb_for_matching = True
    fake_state = {
        "roi_head.fc_cls.weight": torch.zeros(4, 1024),
        "roi_head.fc_cls.bias": torch.zeros(4),
    }

    with patch.object(ckpt_mod, "TrainingExperimentConfig") as mock_cfg_cls, patch.object(
        ckpt_mod, "OrientedRCNN", mock_cls
    ), patch("oriented_det.pretrained.ensure_checkpoint", side_effect=lambda p: p), patch.object(
        ckpt_mod.torch, "load", return_value={"model_state_dict": fake_state}
    ), patch.object(ckpt_mod, "apply_inference_config_to_model"):
        mock_cfg_cls.load.return_value = cfg
        ckpt_mod.load_model_from_checkpoint("/tmp/fake.pth", "/tmp/config.json", "cpu")

    assert mock_cls.call_args.kwargs["roi_use_hbb_for_matching"] is True
