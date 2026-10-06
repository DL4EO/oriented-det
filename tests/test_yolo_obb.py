"""Tests for the YOLO-OBB backbone, DFL head, and detector."""

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
