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
