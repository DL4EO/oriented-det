"""Tests for the YOLO-OBB backbone, DFL head, and detector."""

import math

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.geometry import RBox


def _dummy_inputs(size: int = 256):
    image = torch.rand(3, size, size)
    target = {
        "rboxes": [RBox(size / 2, size / 2, 48.0, 24.0, 0.2)],
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    return image, target


class TestYoloOBBShapes:
    def test_n_p3_p5_feature_sizes(self):
        from oriented_det.models.backbones.yolo_obb import build_yolo_obb_backbone

        bb = build_yolo_obb_backbone("yolo_obb_n", pretrained=False)
        outs = bb(torch.rand(1, 3, 256, 256))
        assert list(outs.keys()) == ["0", "1", "2"]
        assert outs["0"].shape == (1, 64, 32, 32)
        assert outs["1"].shape == (1, 128, 16, 16)
        assert outs["2"].shape == (1, 256, 8, 8)

    def test_s_p3_p5_feature_sizes(self):
        from oriented_det.models.backbones.yolo_obb import build_yolo_obb_backbone

        bb = build_yolo_obb_backbone("yolo_obb_s", pretrained=False)
        outs = bb(torch.rand(1, 3, 256, 256))
        assert outs["0"].shape == (1, 128, 32, 32)
        assert outs["1"].shape == (1, 256, 16, 16)
        assert outs["2"].shape == (1, 512, 8, 8)


class TestDFL:
    def test_expectation_of_one_hot(self):
        from oriented_det.models.yolo_obb import dfl_expectation

        reg_max = 8
        logits = torch.full((2, 4 * reg_max), -20.0)
        logits[:, 3] = 20.0
        logits[:, reg_max + 1] = 20.0
        logits[:, 2 * reg_max + 0] = 20.0
        logits[:, 3 * reg_max + 5] = 20.0
        dist = dfl_expectation(logits, reg_max)
        assert dist.shape == (2, 4)
        assert torch.allclose(dist[0], torch.tensor([3.0, 1.0, 0.0, 5.0]), atol=1e-4)


class TestTaskAlignedAssigner:
    def test_assigns_interior_point(self):
        from oriented_det.models.yolo_obb import task_aligned_assign_single

        ys = (torch.arange(8) + 0.5) * 8
        xs = (torch.arange(8) + 0.5) * 8
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        gt = torch.tensor([[32.0, 32.0, 40.0, 20.0, 0.0]])
        labels_1idx = torch.tensor([1], dtype=torch.long)
        decoded = gt.expand(points.size(0), -1).clone()
        cls_logits = torch.zeros(points.size(0), 2)
        cls_logits[:, 0] = 2.0
        labels, soft, bbox_t, pos = task_aligned_assign_single(
            points,
            cls_logits,
            decoded,
            gt,
            labels_1idx,
            num_classes=2,
            topk=13,
            alpha=0.5,
            beta=6.0,
        )
        assert pos.any()
        assert (labels[pos] == 0).all()
        assert (soft[pos, 0] > 0).all()
        assert (bbox_t[pos, :4] > 0).any()

    def test_soft_targets_normalize_per_object(self):
        from oriented_det.models.yolo_obb import task_aligned_assign_single
        from oriented_det.ops.diff_iou_rotated import diff_iou_rotated_2d

        points = torch.tensor([[10.0, 10.0], [13.0, 10.0], [80.0, 80.0]])
        gt = torch.tensor(
            [
                [10.0, 10.0, 20.0, 10.0, 0.0],
                [80.0, 80.0, 12.0, 8.0, 0.0],
            ]
        )
        decoded = torch.tensor(
            [
                [10.0, 10.0, 20.0, 10.0, 0.0],
                [16.0, 10.0, 20.0, 10.0, 0.0],
                [80.0, 80.0, 12.0, 8.0, 0.0],
            ]
        )
        cls = torch.full((3, 1), -20.0)
        cls[0, 0] = 20.0
        cls[1, 0] = math.log(0.25 / 0.75)
        cls[2, 0] = math.log(0.25 / 0.75)
        _labels, soft, _bbox, pos = task_aligned_assign_single(
            points,
            cls,
            decoded,
            gt,
            torch.tensor([1, 1], dtype=torch.long),
            num_classes=1,
            topk=13,
            alpha=0.5,
            beta=6.0,
        )
        assert pos.tolist() == [True, True, True]
        scores = torch.sigmoid(cls[:, 0])
        ious = []
        aligns = []
        for point_i, gt_i in ((0, 0), (1, 0), (2, 1)):
            iou = float(diff_iou_rotated_2d(decoded[point_i : point_i + 1], gt[gt_i : gt_i + 1]).clamp(0, 1))
            ious.append(iou)
            aligns.append(float(scores[point_i] ** 0.5 * (iou ** 6)))
        max_align = max(aligns[0], aligns[1])
        max_iou = max(ious[0], ious[1])
        assert soft[0, 0].item() == pytest.approx(aligns[0] / max_align * max_iou, rel=1e-4)
        assert soft[1, 0].item() == pytest.approx(aligns[1] / max_align * max_iou, rel=1e-4)
        assert soft[2, 0].item() == pytest.approx(ious[2], rel=1e-4)
        assert soft[2, 0].item() > 0.9

    def test_min_gt_side_floor_keeps_real_width(self):
        from oriented_det.models.yolo_obb import task_aligned_assign_single

        ys = (torch.arange(16) + 0.5) * 8
        xs = (torch.arange(16) + 0.5) * 8
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        # Midway between stride-8 centers, so a 4 px width contains none of them.
        gt = torch.tensor([[8.0, 32.0, 4.0, 60.0, 0.0]])
        decoded = gt.expand(points.size(0), -1).clone()
        cls_logits = torch.full((points.size(0), 1), 2.0)
        labels_1idx = torch.tensor([1], dtype=torch.long)
        common = dict(
            points=points,
            cls_logits=cls_logits,
            decoded_boxes=decoded,
            gt_bboxes=gt,
            gt_labels=labels_1idx,
            num_classes=1,
            topk=10,
        )
        _labels, _soft, _bbox, pos0 = task_aligned_assign_single(**common, min_gt_side=0.0)
        assert not bool(pos0.any())
        _labels, _soft, bbox_t, pos = task_aligned_assign_single(**common, min_gt_side=16.0)
        assert bool(pos.any())
        width = bbox_t[pos, 0] + bbox_t[pos, 2]
        assert torch.allclose(width, torch.full_like(width, 4.0), atol=1e-4)


def test_stride_cls_prior_per_level():
    from oriented_det.models.yolo_obb import YoloOBBHead

    num_classes = 80
    strides = (8.0, 16.0, 32.0)
    head = YoloOBBHead([16, 16, 16], num_classes, strides=strides, stride_cls_prior=True)
    for level, stride in enumerate(strides):
        expected = math.log(5 / num_classes / (640 / stride) ** 2)
        bias = head.cls_heads[level][-1].bias.detach()
        assert torch.allclose(bias, torch.full_like(bias, expected))
    plain = YoloOBBHead([16, 16, 16], num_classes, strides=strides)
    flat = math.log(0.01 / 0.99)
    for level in range(3):
        bias = plain.cls_heads[level][-1].bias.detach()
        assert torch.allclose(bias, torch.full_like(bias, flat))


def test_elongated_angle_weight_is_zero_for_squares():
    from oriented_det.models.yolo_obb import elongated_angle_weight

    gate = elongated_angle_weight(
        torch.tensor([4.0, 8.0, 16.0]),
        torch.tensor([4.0, 4.0, 4.0]),
    )
    assert gate[0].item() == pytest.approx(0.0, abs=1e-6)
    assert gate[2].item() > gate[1].item() > gate[0].item()
    assert gate[2].item() == pytest.approx(1.0 - math.exp(-(math.log(4.0) ** 2)), rel=1e-5)


def test_box_and_dfl_follow_soft_targets_and_square_angle_is_ignored(monkeypatch):
    import oriented_det.models.yolo_obb as yolo_mod
    from oriented_det.models.bbox_coder import DistanceAnglePointCoder

    def fake_assign(points, cls_logits, decoded_boxes, gt_bboxes, gt_labels, num_classes, **kwargs):
        n = points.shape[0]
        labels = torch.full((n,), num_classes, dtype=torch.long)
        soft = torch.zeros(n, num_classes)
        bbox = torch.zeros(n, 5)
        pos = torch.zeros(n, dtype=torch.bool)
        pos[:2] = True
        labels[:2] = 0
        soft[0, 0] = 0.2
        soft[1, 0] = 0.8
        # Square with a large angle error, then a 4:1 box with a 0.4 rad error.
        bbox[0] = torch.tensor([4.0, 4.0, 4.0, 4.0, 1.2])
        bbox[1] = torch.tensor([8.0, 2.0, 8.0, 2.0, 0.4])
        return labels, soft, bbox, pos

    def fake_box(pred, target, **kwargs):
        return torch.tensor([0.0, 1.0], device=pred.device, dtype=pred.dtype)

    def fake_dfl(pred, target, reg_max):
        return torch.tensor([0.0, 1.0], device=pred.device, dtype=pred.dtype)

    monkeypatch.setattr(yolo_mod, "task_aligned_assign_single", fake_assign)
    monkeypatch.setattr(yolo_mod, "ciou_loss_per_box", fake_box)
    monkeypatch.setattr(yolo_mod, "distribution_focal_loss", fake_dfl)

    reg_max = 16
    losses = yolo_mod.compute_yolo_obb_loss(
        cls_scores=[torch.zeros(1, 1, 1, 2)],
        bbox_preds=[torch.zeros(1, 4 * reg_max, 1, 2)],
        angle_preds=[torch.zeros(1, 1, 1, 2)],
        points=[torch.tensor([[4.0, 4.0], [12.0, 4.0]])],
        strides=[8.0],
        gt_boxes=[torch.tensor([[4.0, 4.0, 8.0, 8.0, 0.0]])],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        num_classes=1,
        reg_max=reg_max,
        bbox_coder=DistanceAnglePointCoder(angle_version="le90"),
        box_reg_weight=7.5,
        dfl_weight=1.5,
        angle_weight=0.2,
        assign_topk=13,
        tal_alpha=0.5,
        tal_beta=6.0,
        use_ciou=True,
    )
    # (0 * 0.2 + 1 * 0.8) / 1.0
    assert losses["loss_box_reg"].item() == pytest.approx(7.5 * 0.8)
    assert losses["loss_dfl"].item() == pytest.approx(1.5 * 0.8)
    gate = 1.0 - math.exp(-(math.log(4.0) ** 2))
    assert losses["loss_angle"].item() == pytest.approx(0.2 * gate * 0.4)
    # Two zero-logit anchors, BCE = log(2) each, gain 0.5, soft sum 1.
    assert losses["loss_classifier"].item() == pytest.approx(0.5 * 2.0 * math.log(2.0))


class TestYoloOBB:
    def test_train_forward_backward(self):
        from oriented_det.models import YoloOBB

        image, target = _dummy_inputs(256)
        model = YoloOBB(num_classes=2, backbone_name="yolo_obb_n", pretrained_backbone=False)
        model.train()
        losses = model([image], [target])
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        assert "loss_dfl" in losses
        assert "loss_angle" in losses
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_train_forward_under_autocast(self):
        from oriented_det.models import YoloOBB

        image, target = _dummy_inputs(256)
        model = YoloOBB(num_classes=2, backbone_name="yolo_obb_n", pretrained_backbone=False)
        model.train()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype):
            losses = model([image], [target])
            total = sum(losses.values())
        assert torch.isfinite(total.float())
        total.backward()

    def test_eval_forward_format(self):
        from oriented_det.models import YoloOBB

        image, _ = _dummy_inputs(256)
        model = YoloOBB(num_classes=2, backbone_name="yolo_obb_n", pretrained_backbone=False)
        model.eval()
        with torch.no_grad():
            outs = model([image])
        assert isinstance(outs, list) and len(outs) == 1
        assert "rboxes" in outs[0]
        assert "labels" in outs[0]
        assert "scores" in outs[0]

    def test_empty_gt_train(self):
        from oriented_det.models import YoloOBB

        image = torch.rand(3, 128, 128)
        target = {"rboxes": [], "labels": torch.zeros((0,), dtype=torch.int64)}
        model = YoloOBB(num_classes=2, backbone_name="yolo_obb_n", pretrained_backbone=False)
        model.train()
        losses = model([image], [target])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_riou_box_loss_trains(self):
        from oriented_det.models import YoloOBB

        image, target = _dummy_inputs(128)
        model = YoloOBB(
            num_classes=2,
            backbone_name="yolo_obb_n",
            pretrained_backbone=False,
            box_reg_loss_type="riou",
        )
        assert model.box_reg_loss_type == "riou"
        model.train()
        losses = model([image], [target])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_qfl_cls_loss_trains(self):
        from oriented_det.models import YoloOBB

        image, target = _dummy_inputs(128)
        torch.manual_seed(0)
        bce = YoloOBB(num_classes=2, backbone_name="yolo_obb_n", pretrained_backbone=False)
        qfl = YoloOBB(
            num_classes=2,
            backbone_name="yolo_obb_n",
            pretrained_backbone=False,
            cls_loss_type="qfl",
        )
        qfl.load_state_dict(bce.state_dict())
        bce.train()
        qfl.train()
        bce_losses = bce([image], [target])
        qfl_losses = qfl([image], [target])
        # Same weights and targets: only the classification term differs.
        assert torch.allclose(qfl_losses["loss_box_reg"], bce_losses["loss_box_reg"])
        assert torch.allclose(qfl_losses["loss_dfl"], bce_losses["loss_dfl"])
        assert not torch.allclose(qfl_losses["loss_classifier"], bce_losses["loss_classifier"])
        total = sum(qfl_losses.values())
        assert torch.isfinite(total)
        total.backward()
        cls_grad = qfl.head.cls_heads[0][2].weight.grad
        assert cls_grad is not None and float(cls_grad.abs().sum()) > 0

    def test_dynamic_soft_assigner_trains(self):
        from oriented_det.models import YoloOBB

        image, target = _dummy_inputs(128)
        model = YoloOBB(
            num_classes=2,
            backbone_name="yolo_obb_n",
            pretrained_backbone=False,
            cls_loss_type="qfl",
            assigner="dynamic_soft",
        )
        model.train()
        losses = model([image], [target])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_unknown_assigner_rejected(self):
        from oriented_det.models import YoloOBB

        with pytest.raises(ValueError, match="yolo_obb_assigner"):
            YoloOBB(
                num_classes=2,
                backbone_name="yolo_obb_n",
                pretrained_backbone=False,
                assigner="simota",
            )

    def test_unknown_cls_loss_rejected(self):
        from oriented_det.models import YoloOBB

        with pytest.raises(ValueError, match="yolo_obb_cls_loss_type"):
            YoloOBB(
                num_classes=2,
                backbone_name="yolo_obb_n",
                pretrained_backbone=False,
                cls_loss_type="focal",
            )

    def test_riou_rejected_when_horizontal(self):
        from oriented_det.models import YoloOBB

        with pytest.raises(ValueError, match="force_horizontal"):
            YoloOBB(
                num_classes=2,
                backbone_name="yolo_obb_n",
                pretrained_backbone=False,
                force_horizontal=True,
                box_reg_loss_type="riou",
            )


def test_dynamic_soft_loss_runs_when_predictions_are_fp16():
    from oriented_det.models.bbox_coder import DistanceAnglePointCoder
    from oriented_det.models.yolo_obb import compute_yolo_obb_loss

    reg_max = 16
    losses = compute_yolo_obb_loss(
        cls_scores=[torch.zeros(1, 1, 1, 2, dtype=torch.float16)],
        bbox_preds=[torch.zeros(1, 4 * reg_max, 1, 2, dtype=torch.float16)],
        angle_preds=[torch.zeros(1, 1, 1, 2, dtype=torch.float16)],
        points=[torch.tensor([[32.0, 32.0], [40.0, 32.0]], dtype=torch.float16)],
        strides=[8.0],
        gt_boxes=[torch.tensor([[36.0, 32.0, 24.0, 16.0, 0.2]])],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        num_classes=1,
        reg_max=reg_max,
        bbox_coder=DistanceAnglePointCoder(angle_version="le90"),
        box_reg_weight=7.5,
        dfl_weight=1.5,
        angle_weight=0.2,
        assign_topk=13,
        tal_alpha=0.5,
        tal_beta=6.0,
        assigner="dynamic_soft",
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_dynamic_soft_assigner_keeps_the_center_and_drops_the_far_point():
    from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single
    from oriented_det.models.yolo_obb import task_aligned_assign_single

    points = torch.tensor([[64.0, 64.0], [140.0, 64.0], [64.0, 200.0]])
    gt = torch.tensor([[64.0, 64.0, 200.0, 40.0, 0.0]])
    decoded = gt.expand(3, -1).contiguous()
    logits = torch.full((3, 1), 10.0)
    labels = torch.tensor([1])
    strides = torch.full((3,), 8.0)
    common = dict(
        points=points,
        cls_logits=logits,
        decoded_boxes=decoded,
        gt_bboxes=gt,
        gt_labels=labels,
        num_classes=1,
        topk=13,
        strides=strides,
    )
    _labels, _soft, _bbox, pos_tal = task_aligned_assign_single(**common, alpha=0.5, beta=6.0)
    _labels, soft, _bbox, pos = dynamic_soft_label_assign_single(
        **common, soft_center_radius=3.0, iou_weight=3.0
    )
    assert pos_tal[:2].tolist() == [True, True]
    assert pos.tolist() == [True, False, False]
    assert soft[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert soft[1, 0].item() == pytest.approx(0.0)


def test_dota_12x_recipe_stands_on_the_shared_bases():
    import json
    from pathlib import Path

    from oriented_det.models.yolo_obb import build_yolo_obb
    from oriented_det.train.config import TrainingExperimentConfig

    root = Path(__file__).resolve().parents[1]
    path = root / "configs/yolo_obb/dota_le90_12x_s_coco.json"
    raw = json.loads(path.read_text())
    assert raw["_base_"] == [
        "../_base_/datasets/dota_le90.json",
        "../_base_/models/yolo_obb_s.json",
        "../_base_/schedules/12x.json",
        "../_base_/preprocessing.json",
    ]
    recipe = TrainingExperimentConfig.load(path)
    assert recipe.training.num_epochs == 144
    assert recipe.training.learning_rate == pytest.approx(1.25e-4)
    assert recipe.training.lr_scheduler_type == "cosine_annealing_late"
    assert recipe.training.lr_scheduler_cosine_start_epoch == 72
    assert recipe.training.lr_scheduler_cosine_eta_min == pytest.approx(1.25e-5)
    assert recipe.training.weight_decay == pytest.approx(0.05)
    assert recipe.model.yolo_obb_cls_loss_type == "bce"
    assert recipe.model.yolo_obb_assigner == "tal"
    assert recipe.model.yolo_obb_dfl_weight == pytest.approx(0.75)
    assert recipe.model.box_reg_weight == pytest.approx(7.5)
    assert recipe.model.box_reg_loss_type == "smooth_l1"
    assert recipe.preprocessing.enable_mosaic is True
    assert recipe.preprocessing.random_scale == pytest.approx(0.9)
    assert recipe.preprocessing.random_translate == pytest.approx(0.1)
    assert recipe.preprocessing.mosaic_disable_last_n_epochs == 0
    assert recipe.enable_albumentation is True
    assert recipe.augmentation.p_hsv == pytest.approx(1.0)
    assert recipe.augmentation.p_brightness_contrast == pytest.approx(0.0)
    assert recipe.evaluation.test_map is False
    assert recipe.evaluation.val_map_every_n_epochs == 12
    assert str(recipe.checkpoint.load_from_checkpoint) == "hf://yolo_obb_coco_300e"
    model = build_yolo_obb(recipe.model, 15)
    assert model.assigner == "tal"
    assert model.cls_loss_type == "bce"
    assert model.dfl_weight == pytest.approx(0.75)
    assert model.box_reg_loss_type == "probiou"
