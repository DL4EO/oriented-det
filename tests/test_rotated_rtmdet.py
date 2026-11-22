"""Tests for CSPNeXt backbone and Rotated RTMDet."""

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


class TestCSPNeXtShapes:
    def test_tiny_p3_p5_feature_sizes(self):
        from oriented_det.models.backbones.cspnext import build_cspnext_pafpn_backbone

        bb = build_cspnext_pafpn_backbone("cspnext_tiny", pretrained=False)
        x = torch.rand(1, 3, 256, 256)
        outs = bb(x)
        assert list(outs.keys()) == ["0", "1", "2"]
        assert outs["0"].shape == (1, 96, 32, 32)
        assert outs["1"].shape == (1, 96, 16, 16)
        assert outs["2"].shape == (1, 96, 8, 8)
        assert bb.out_channels == 96

    def test_s_p3_p5_feature_sizes(self):
        from oriented_det.models.backbones.cspnext import build_cspnext_pafpn_backbone

        bb = build_cspnext_pafpn_backbone("cspnext_s", pretrained=False)
        x = torch.rand(1, 3, 256, 256)
        outs = bb(x)
        assert outs["0"].shape == (1, 128, 32, 32)
        assert outs["1"].shape == (1, 128, 16, 16)
        assert outs["2"].shape == (1, 128, 8, 8)
        assert bb.out_channels == 128

    def test_imagenet_alias_loads_depthwise_strict(self, monkeypatch):
        from oriented_det.models.backbones.cspnext import CSPNeXt, _load_cspnext_imagenet

        body = CSPNeXt(deepen_factor=0.33, widen_factor=0.5)
        ckpt_sd = {}
        for key, value in body.state_dict().items():
            aliased = "backbone." + key
            aliased = aliased.replace(".depthwise.", ".depthwise_conv.")
            aliased = aliased.replace(".pointwise.", ".pointwise_conv.")
            ckpt_sd[aliased] = value.detach().clone()
        ckpt_sd["head.fc.weight"] = torch.zeros(4, 4)
        ckpt_sd["head.fc.bias"] = torch.zeros(4)
        dw_key = (
            "backbone.stage1.1.blocks.0.conv2.depthwise_conv.conv.weight"
        )
        assert dw_key in ckpt_sd

        def _fake_load(*_args, **_kwargs):
            return {"state_dict": ckpt_sd}

        monkeypatch.setattr("torch.hub.load_state_dict_from_url", _fake_load)
        with torch.no_grad():
            ckpt_sd[dw_key].fill_(1.25)
            body.stage1[1].blocks[0].conv2.depthwise.conv.weight.zero_()
        assert _load_cspnext_imagenet(body, "cspnext_s") is True
        loaded = body.stage1[1].blocks[0].conv2.depthwise.conv.weight
        assert torch.allclose(loaded, torch.full_like(loaded, 1.25))

    def test_body_channel_list(self):
        from oriented_det.models.backbones.cspnext import CSPNeXt

        body = CSPNeXt(deepen_factor=0.167, widen_factor=0.375, out_indices=(2, 3, 4))
        assert body.out_channels_list == [96, 192, 384]


class TestQualityFocalLoss:
    def test_zero_when_perfect(self):
        from oriented_det.models.rotated_rtmdet import quality_focal_loss

        # logits → sigmoid ≈ target when target is 0 or 1 and logits extreme
        pred = torch.tensor([[20.0, -20.0]])
        target = torch.tensor([[1.0, 0.0]])
        loss = quality_focal_loss(pred, target, beta=2.0)
        assert float(loss) < 1e-3

    def test_positive_finite(self):
        from oriented_det.models.rotated_rtmdet import quality_focal_loss

        pred = torch.zeros(4, 3)
        target = torch.zeros(4, 3)
        target[0, 1] = 0.7
        loss = quality_focal_loss(pred, target, beta=2.0)
        assert torch.isfinite(loss)
        assert float(loss) > 0

    def test_valid_mask_drops_ignore_rows_and_keeps_graph(self):
        from oriented_det.models.rotated_rtmdet import quality_focal_loss

        pred = torch.tensor([[8.0, -3.0], [0.2, -0.4], [1.5, 0.5]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.4]])
        weight = torch.tensor([1.0, 2.0])
        # Row 0 is a don't-care point (label -1). Its confident logit must not add loss.
        valid = torch.tensor([False, True, True])
        loss = quality_focal_loss(pred, target, beta=2.0, weight=weight, valid=valid)
        expected = quality_focal_loss(pred[1:], target[1:], beta=2.0, weight=weight)
        assert torch.allclose(loss, expected)

        pred_ign = torch.randn(5, 2, requires_grad=True)
        loss_ign = quality_focal_loss(
            pred_ign,
            torch.rand(5, 2),
            valid=torch.zeros(5, dtype=torch.bool),
        )
        assert float(loss_ign) == 0.0
        loss_ign.backward()
        assert pred_ign.grad is not None
        assert torch.equal(pred_ign.grad, torch.zeros_like(pred_ign))


class TestDynamicSoftLabelAssigner:
    def test_assigns_center_point(self):
        from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single

        # 8x8 grid at stride 8 covering 64x64
        ys = (torch.arange(8) + 0.5) * 8
        xs = (torch.arange(8) + 0.5) * 8
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        gt = torch.tensor([[32.0, 32.0, 40.0, 20.0, 0.0]])
        labels_1idx = torch.tensor([1], dtype=torch.long)
        # Perfect decode = GT
        decoded = gt.expand(points.size(0), -1).clone()
        cls_logits = torch.zeros(points.size(0), 2)
        cls_logits[:, 0] = 2.0
        strides = points.new_full((points.size(0),), 8.0)
        labels, soft, bbox_t, pos = dynamic_soft_label_assign_single(
            points,
            cls_logits,
            decoded,
            gt,
            labels_1idx,
            num_classes=2,
            soft_center_radius=3.0,
            topk=13,
            strides=strides,
        )
        assert pos.any()
        assert (labels[pos] == 0).all()
        assert (soft[pos, 0] > 0).all()
        enc = bbox_t[pos]
        assert enc.size(0) > 0
        assert (enc[:, :4] >= 0).all()
        # At least one assigned point is strictly inside (all distances > 0)
        assert (enc[:, :4] > 0).any()

    def test_ignore_marks_background_on_empty_gt_and_no_prior(self):
        from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single

        points = torch.tensor([[50.0, 50.0], [0.0, 0.0], [200.0, 200.0]])
        ignore = torch.tensor([[50.0, 50.0, 30.0, 30.0, 0.0]])
        decoded = torch.zeros(points.size(0), 5)
        cls_logits = torch.zeros(points.size(0), 2)

        labels, _, _, pos = dynamic_soft_label_assign_single(
            points,
            cls_logits,
            decoded,
            torch.zeros((0, 5)),
            torch.tensor([], dtype=torch.long),
            num_classes=2,
            gt_bboxes_ignore=ignore,
        )
        assert labels[0].item() == -1
        assert labels[1].item() == 2
        assert labels[2].item() == 2
        assert not pos[0]

        # GT exists, but no point falls in its prior. The ignore point is still -1.
        far_gt = torch.tensor([[400.0, 400.0, 10.0, 10.0, 0.0]])
        labels_np, _, _, pos_np = dynamic_soft_label_assign_single(
            points,
            cls_logits,
            far_gt.expand(points.size(0), -1).clone(),
            far_gt,
            torch.tensor([1], dtype=torch.long),
            num_classes=2,
            strides=points.new_full((points.size(0),), 8.0),
            gt_bboxes_ignore=ignore,
        )
        assert labels_np[0].item() == -1
        assert labels_np[1].item() == 2
        assert not pos_np.any()

    def test_ignore_does_not_override_positive_and_empty_tensors_are_noop(self):
        from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single

        points = torch.tensor([[32.0, 32.0], [0.0, 0.0]])
        gt = torch.tensor([[32.0, 32.0, 40.0, 20.0, 0.0]])
        decoded = gt.expand(points.size(0), -1).clone()
        cls_logits = torch.zeros(points.size(0), 2)
        cls_logits[:, 0] = 2.0
        strides = points.new_full((points.size(0),), 8.0)
        labels_gt = torch.tensor([1], dtype=torch.long)
        common = dict(
            points=points,
            cls_logits=cls_logits,
            decoded_boxes=decoded,
            gt_bboxes=gt,
            gt_labels=labels_gt,
            num_classes=2,
            soft_center_radius=3.0,
            topk=13,
            strides=strides,
        )
        labels, _, _, pos = dynamic_soft_label_assign_single(
            **common,
            gt_bboxes_ignore=gt.clone(),
        )
        assert labels[0].item() == 0
        assert bool(pos[0])
        assert labels[1].item() == 2

        empty = torch.zeros((0, 5))
        plain, _, _, plain_pos = dynamic_soft_label_assign_single(**common)
        kept, _, _, kept_pos = dynamic_soft_label_assign_single(
            **common,
            gt_bboxes_ignore=empty,
            gt_bboxes_lookalike=empty,
        )
        assert torch.equal(plain, kept)
        assert torch.equal(plain_pos, kept_pos)
        assert (kept >= 0).all()

    def test_lookalike_wins_over_ignore(self):
        from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single

        points = torch.tensor([[50.0, 50.0], [0.0, 0.0]])
        region = torch.tensor([[50.0, 50.0, 30.0, 30.0, 0.0]])
        labels, _, _, pos = dynamic_soft_label_assign_single(
            points,
            torch.zeros(points.size(0), 2),
            torch.zeros(points.size(0), 5),
            torch.zeros((0, 5)),
            torch.tensor([], dtype=torch.long),
            num_classes=2,
            gt_bboxes_ignore=region,
            gt_bboxes_lookalike=region,
        )
        assert labels[0].item() == 2
        assert labels[1].item() == 2
        assert not pos[0]

    def test_fp16_logits_with_fp32_points(self):
        from oriented_det.models.rotated_rtmdet import dynamic_soft_label_assign_single

        ys = (torch.arange(8) + 0.5) * 8
        xs = (torch.arange(8) + 0.5) * 8
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        assert points.dtype == torch.float32
        gt = torch.tensor([[32.0, 32.0, 40.0, 20.0, 0.0]])
        labels_1idx = torch.tensor([1], dtype=torch.long)
        decoded = gt.expand(points.size(0), -1).clone()
        cls_logits = torch.zeros(points.size(0), 2, dtype=torch.float16)
        cls_logits[:, 0] = 2.0
        strides = points.new_full((points.size(0),), 8.0)
        labels, soft, bbox_t, pos = dynamic_soft_label_assign_single(
            points,
            cls_logits,
            decoded,
            gt,
            labels_1idx,
            num_classes=2,
            soft_center_radius=3.0,
            topk=13,
            strides=strides,
        )
        assert pos.any()
        assert (labels[pos] == 0).all()
        assert torch.isfinite(soft.float()).all()
        assert torch.isfinite(bbox_t).all()


class TestRotatedRTMDet:
    def test_train_forward_backward(self):
        from oriented_det.models import RotatedRTMDet

        image, target = _dummy_inputs(256)
        model = RotatedRTMDet(
            num_classes=2,
            backbone_name="cspnext_tiny",
            pretrained_backbone=False,
            fpn_strides=[8, 16, 32],
        )
        model.train()
        losses = model([image], [target])
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        assert "loss_angle" in losses
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_eval_forward_format(self):
        from oriented_det.models import RotatedRTMDet

        image, _ = _dummy_inputs(256)
        model = RotatedRTMDet(
            num_classes=2,
            backbone_name="cspnext_tiny",
            pretrained_backbone=False,
        )
        model.eval()
        with torch.no_grad():
            outs = model([image])
        assert isinstance(outs, list)
        assert len(outs) == 1
        assert "rboxes" in outs[0]
        assert "labels" in outs[0]
        assert "scores" in outs[0]

    def test_empty_gt_train(self):
        from oriented_det.models import RotatedRTMDet

        image = torch.rand(3, 256, 256)
        target = {
            "rboxes": [],
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
        model = RotatedRTMDet(
            num_classes=2,
            backbone_name="cspnext_tiny",
            pretrained_backbone=False,
        )
        model.train()
        losses = model([image], [target])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

    def test_ignore_only_and_lookalike_only_train(self):
        from oriented_det.models import RotatedRTMDet

        model = RotatedRTMDet(
            num_classes=2,
            backbone_name="cspnext_tiny",
            pretrained_backbone=False,
        )
        model.train()
        image = torch.rand(3, 256, 256)
        ignore_only = {
            "rboxes": [],
            "labels": torch.zeros((0,), dtype=torch.int64),
            "rboxes_ignore": [RBox(128.0, 128.0, 48.0, 24.0, 0.2)],
        }
        losses = model([image], [ignore_only])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()

        look_only = {
            "rboxes": [],
            "labels": torch.zeros((0,), dtype=torch.int64),
            "rboxes_lookalike": [RBox(128.0, 128.0, 48.0, 24.0, 0.2)],
        }
        losses = model([image], [look_only])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
