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
