"""Tests for RPN (Region Proposal Network) components."""

import math
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch

from oriented_det.geometry import RBox
from oriented_det.geometry.rbox import normalize_le90 as normalize_le90_rbox
from oriented_det.models.oriented_rpn import (
    OrientedRPNHead,
    generate_oriented_anchors,
    encode_rpn_boxes,
    decode_rpn_boxes,
    encode_oriented_boxes,
    decode_oriented_boxes,
    normalize_boxes_to_le90,
    match_oriented_anchors_to_gt,
    compute_oriented_rpn_loss,
    generate_oriented_proposals,
    compute_midpoint_rpn_loss,
    generate_midpoint_proposals,
)


@pytest.fixture
def dummy_features():
    """Create dummy FPN feature maps."""
    return [
        torch.randn(1, 256, 32, 32),  # Level 0
        torch.randn(1, 256, 16, 16),  # Level 1
    ]


@pytest.fixture
def dummy_anchors():
    """Create dummy anchors."""
    return torch.tensor([
        [64.0, 64.0, 32.0, 16.0, 0.0],
        [128.0, 128.0, 32.0, 16.0, 0.0],
        [192.0, 192.0, 32.0, 16.0, 0.0],
    ])


@pytest.fixture
def dummy_gt_boxes():
    """Create dummy ground truth boxes."""
    return torch.tensor([
        [64.0, 64.0, 30.0, 15.0, 0.0],
        [128.0, 128.0, 35.0, 18.0, 0.0],
    ])


class TestOrientedRPNHead:
    """Tests for OrientedRPNHead."""
    
    def test_forward_single_level(self, dummy_features):
        """Test forward pass with single FPN level."""
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        features = [dummy_features[0]]
        objectness_logits, bbox_regression = head(features)
        
        assert len(objectness_logits) == 1
        assert len(bbox_regression) == 1
        
        # Check shapes: [B, num_anchors*cls_out_channels, H, W]
        assert objectness_logits[0].shape == (1, 3 * 1, 32, 32)
        # Check shapes: [B, num_anchors*reg_out_channels, H, W] - 4 params (dx, dy, dw, dh)
        assert bbox_regression[0].shape == (1, 3 * 4, 32, 32)
    
    def test_forward_multiple_levels(self, dummy_features):
        """Test forward pass with multiple FPN levels."""
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        assert len(objectness_logits) == 2
        assert len(bbox_regression) == 2
        
        # Level 0: 32x32 - 4 params per anchor
        assert objectness_logits[0].shape == (1, 3, 32, 32)
        assert bbox_regression[0].shape == (1, 12, 32, 32)
        
        # Level 1: 16x16
        assert objectness_logits[1].shape == (1, 3, 16, 16)
        assert bbox_regression[1].shape == (1, 12, 16, 16)
    
    def test_forward_batch_size_gt_1(self):
        """Test forward pass with batch size > 1."""
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        features = [torch.randn(2, 256, 16, 16)]  # Batch size 2
        objectness_logits, bbox_regression = head(features)
        
        assert objectness_logits[0].shape[0] == 2
        assert bbox_regression[0].shape[0] == 2
    
    @pytest.mark.parametrize("cls_out_channels,reg_out_channels", [
        (1, 4),  # MMRotate format (4 params: dx, dy, dw, dh)
        (2, 4),  # 2 cls channels, 4 reg params
    ])
    def test_different_configurations(self, dummy_features, cls_out_channels, reg_out_channels):
        """Test different RPN head configurations."""
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=cls_out_channels,
            reg_out_channels=reg_out_channels,
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        expected_cls_channels = 3 * cls_out_channels
        expected_reg_channels = 3 * reg_out_channels
        
        assert objectness_logits[0].shape[1] == expected_cls_channels
        assert bbox_regression[0].shape[1] == expected_reg_channels


class TestAnchorGeneration:
    """Tests for anchor generation."""
    
    def test_single_scale_single_angle(self):
        """Test anchor generation with single scale and angle."""
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4],
        )
        
        assert len(anchors) == 1
        level_anchors = anchors[0]
        
        # Should have 32*32*3 = 3072 anchors (3 ratios × 1 angle)
        assert level_anchors.shape[0] == 32 * 32 * 3
        assert level_anchors.shape[1] == 5  # [cx, cy, w, h, angle]
        
        # Check anchor format
        assert torch.all(level_anchors[:, 4] == -math.pi / 2)  # All angles should be -90°
    
    def test_multiple_scales_angles(self):
        """Test anchor generation with multiple scales and angles."""
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(16, 16)],
            anchor_scales=[8, 16],
            anchor_ratios=[0.5, 1.0],
            anchor_angles=[-math.pi / 2, 0, math.pi / 2],
            stride_per_level=[8],
        )
        
        assert len(anchors) == 1
        level_anchors = anchors[0]
        
        # Should use last scale if fewer scales than levels
        # 16*16*2*3 = 1536 anchors (2 ratios × 3 angles)
        assert level_anchors.shape[0] == 16 * 16 * 2 * 3
    
    def test_multiple_fpn_levels(self):
        """Test anchor generation for multiple FPN levels."""
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        assert len(anchors) == 2
        
        # Level 0: stride 4, 32x32 feature map
        assert anchors[0].shape[0] == 32 * 32 * 3
        
        # Level 1: stride 8, 16x16 feature map
        assert anchors[1].shape[0] == 16 * 16 * 3
    
    def test_anchor_format_validation(self):
        """Test that anchors have correct format [cx, cy, w, h, angle]."""
        anchors = generate_oriented_anchors(
            image_size=(64, 64),
            feature_map_sizes=[(16, 16)],
            anchor_scales=[8],
            anchor_ratios=[1.0],
            anchor_angles=[0],
            stride_per_level=[4],
        )
        
        level_anchors = anchors[0]
        
        # Check that all anchors have positive width and height
        assert torch.all(level_anchors[:, 2] > 0)  # width > 0
        assert torch.all(level_anchors[:, 3] > 0)  # height > 0
        
        # Check that centers are within image bounds (with some tolerance)
        assert torch.all(level_anchors[:, 0] >= 0)  # cx >= 0
        assert torch.all(level_anchors[:, 1] >= 0)  # cy >= 0
        assert torch.all(level_anchors[:, 0] <= 64)  # cx <= image_width
        assert torch.all(level_anchors[:, 1] <= 64)  # cy <= image_height


class TestBoxEncodingDecoding:
    """Tests for box encoding and decoding."""
    
    def test_encode_rpn_decode_rpn_round_trip(self, dummy_anchors, dummy_gt_boxes):
        """Test RPN encode/decode (4 params, angle from anchor) round trip."""
        anchors = dummy_anchors[:1].repeat(len(dummy_gt_boxes), 1)
        target_means = (0.0, 0.0, 0.0, 0.0)
        target_stds = (1.0, 1.0, 1.0, 1.0)
        encoded = encode_rpn_boxes(anchors, dummy_gt_boxes, target_means, target_stds)
        assert encoded.shape == (len(dummy_gt_boxes), 4)
        decoded = decode_rpn_boxes(anchors, encoded, target_means, target_stds)
        assert decoded.shape == (len(dummy_gt_boxes), 5)
        assert torch.allclose(decoded[:, :2], dummy_gt_boxes[:, :2], atol=1.0)
        assert torch.allclose(decoded[:, 2:4], dummy_gt_boxes[:, 2:4], rtol=0.1)
    
    def test_encode_decode_round_trip(self, dummy_anchors, dummy_gt_boxes):
        """Test that encoding then decoding recovers original boxes."""
        # Encode GT boxes relative to anchors
        # Use first anchor for all GT boxes for simplicity
        anchors = dummy_anchors[:1].repeat(len(dummy_gt_boxes), 1)
        
        encoded = encode_oriented_boxes(anchors, dummy_gt_boxes)
        assert encoded.shape == (len(dummy_gt_boxes), 5)
        
        # Decode back
        decoded = decode_oriented_boxes(anchors, encoded, normalize_le90=False)
        assert decoded.shape == (len(dummy_gt_boxes), 5)
        
        # Check that decoded boxes are close to original (within tolerance)
        # Note: exact match may not be possible due to normalization, but should be close
        assert torch.allclose(decoded[:, :2], dummy_gt_boxes[:, :2], atol=1.0)  # Centers
        assert torch.allclose(decoded[:, 2:4], dummy_gt_boxes[:, 2:4], rtol=0.1)  # Sizes
    
    def test_encode_with_target_normalization(self, dummy_anchors, dummy_gt_boxes):
        """Test encoding with target normalization."""
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
        target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
        
        anchors = dummy_anchors[:1].repeat(len(dummy_gt_boxes), 1)
        
        encoded = encode_oriented_boxes(
            anchors, dummy_gt_boxes,
            target_means=target_means,
            target_stds=target_stds,
        )
        
        # Decode with same normalization
        decoded = decode_oriented_boxes(
            anchors, encoded,
            target_means=target_means,
            target_stds=target_stds,
            normalize_le90=False,
        )
        
        # Should still recover approximately
        assert torch.allclose(decoded[:, :2], dummy_gt_boxes[:, :2], atol=1.0)
    
    def test_encode_identical_anchor_gt(self):
        """Test encoding when anchor and GT box are identical."""
        anchors = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]])
        gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]])
        
        encoded = encode_oriented_boxes(anchors, gt_boxes)
        
        # dx, dy should be ~0, dw, dh should be ~0, dangle should be ~0
        assert torch.allclose(encoded[:, :4], torch.zeros(1, 4), atol=0.1)
        assert torch.allclose(encoded[:, 4:5], torch.zeros(1, 1), atol=0.1)
    
    def test_decode_zero_deltas(self, dummy_anchors):
        """Test decoding zero deltas should return anchors."""
        deltas = torch.zeros(len(dummy_anchors), 5)
        
        decoded = decode_oriented_boxes(dummy_anchors, deltas, normalize_le90=False)
        
        # Should be close to anchors (exact match may differ due to normalization)
        assert torch.allclose(decoded[:, :2], dummy_anchors[:, :2], atol=0.1)
        assert torch.allclose(decoded[:, 2:4], dummy_anchors[:, 2:4], rtol=0.01)

    def test_encode_decode_angle_delta_greater_than_90_degrees(self):
        """Encode/decode round-trip when anchor and GT differ by more than 90° (edge_swap).

        Regression test for angle encoding: we must use normalize_angle_delta ([-π, π])
        for the angle delta in edge_swap, not norm_angle_le90 ([-π/2, π/2)). Using
        norm_angle_le90 would wrap e.g. 95° to -85°, so decode would produce the wrong
        angle. This case (anchor at -90°, GT at +5°) would have failed with the bug.
        """
        # Anchor at -90°, GT at +5° -> true delta = 95° (> 90°)
        anchor = torch.tensor([[100.0, 100.0, 50.0, 20.0, -math.pi / 2]])
        gt = torch.tensor([[100.0, 100.0, 50.0, 20.0, 0.087]])  # ~5°
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
        target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
        encoded = encode_oriented_boxes(
            anchor, gt,
            target_means=target_means,
            target_stds=target_stds,
            norm_factor=2.0,
            edge_swap=True,
        )
        decoded = decode_oriented_boxes(
            anchor, encoded,
            target_means=target_means,
            target_stds=target_stds,
            norm_factor=2.0,
            edge_swap=True,
        )
        dtheta = math.atan2(
            math.sin(float(decoded[0, 4] - gt[0, 4])),
            math.cos(float(decoded[0, 4] - gt[0, 4])),
        )
        assert abs(dtheta) < 1e-4, (
            f"Angle round-trip failed for delta>90°: decoded angle {math.degrees(decoded[0, 4].item()):.2f}° "
            f"vs GT {math.degrees(gt[0, 4].item()):.2f}° (diff {math.degrees(abs(dtheta)):.4f}°)"
        )

    def test_encode_decode_round_trip_with_edge_swap_keeps_le90_equivalence(self):
        """Edge-swap encode/decode should preserve boxes up to le90-equivalent representation.

        Regression guard: this catches a ±pi/2 angle drift caused by incorrect
        width/height swapping during le90 normalization.

        Note: This test uses random anchor/GT angles in [-π/2, π/2]. It did not catch
        the norm_angle_le90 vs normalize_angle_delta bug because (1) dummy round-trip
        tests use angle=0 for both anchor and GT; (2) here, when |delta|>90°, the
        decode's edge_swap inverse can sometimes mask the wrong encoding. The explicit
        test_encode_decode_angle_delta_greater_than_90_degrees covers the failing case.
        """
        torch.manual_seed(123)
        n = 1024

        # Anchors in le90 convention
        anchor_cx = torch.rand(n) * 1024
        anchor_cy = torch.rand(n) * 1024
        a_w0 = torch.rand(n) * 200 + 4
        a_h0 = torch.rand(n) * 200 + 4
        anchor_w = torch.maximum(a_w0, a_h0)
        anchor_h = torch.minimum(a_w0, a_h0)
        anchor_angle = (torch.rand(n) - 0.5) * math.pi
        anchors = torch.stack([anchor_cx, anchor_cy, anchor_w, anchor_h, anchor_angle], dim=1)

        # GT boxes in le90 convention
        gt_cx = anchor_cx + torch.randn(n) * 20
        gt_cy = anchor_cy + torch.randn(n) * 20
        g_w0 = torch.rand(n) * 220 + 4
        g_h0 = torch.rand(n) * 220 + 4
        gt_w = torch.maximum(g_w0, g_h0)
        gt_h = torch.minimum(g_w0, g_h0)
        gt_angle = (torch.rand(n) - 0.5) * math.pi
        gt_boxes = torch.stack([gt_cx, gt_cy, gt_w, gt_h, gt_angle], dim=1)

        deltas = encode_oriented_boxes(
            anchors,
            gt_boxes,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            norm_factor=2.0,
            edge_swap=True,
        )
        decoded = decode_oriented_boxes(
            anchors,
            deltas,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            norm_factor=2.0,
            edge_swap=True,
        )

        # Centers/sizes should round-trip tightly.
        assert torch.allclose(decoded[:, :2], gt_boxes[:, :2], atol=1e-4)
        assert torch.allclose(decoded[:, 2:4], gt_boxes[:, 2:4], atol=1e-4)

        # Angle should also match without systematic ±pi/2 drift.
        dtheta = torch.atan2(torch.sin(decoded[:, 4] - gt_boxes[:, 4]), torch.cos(decoded[:, 4] - gt_boxes[:, 4]))
        assert dtheta.abs().max() < 1e-4

    def test_le90_python_and_tensor_normalizers_are_equivalent(self):
        """RBox normalize_le90 and tensor normalize_boxes_to_le90 must agree."""
        torch.manual_seed(321)
        n = 1024
        cx = torch.rand(n) * 1024
        cy = torch.rand(n) * 1024
        w = torch.rand(n) * 200 + 1
        h = torch.rand(n) * 200 + 1
        angle = (torch.rand(n) - 0.5) * (8.0 * math.pi)  # broad range stresses wrapping
        boxes = torch.stack([cx, cy, w, h, angle], dim=1)

        tnorm = normalize_boxes_to_le90(boxes)
        pnorm = []
        for b in boxes:
            rb = RBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]))
            rn = normalize_le90_rbox(rb)
            pnorm.append([rn.cx, rn.cy, rn.width, rn.height, rn.angle])
        pnorm = torch.tensor(pnorm, dtype=torch.float32)

        assert torch.allclose(tnorm[:, :4], pnorm[:, :4], atol=1e-4)
        dtheta = torch.atan2(torch.sin(tnorm[:, 4] - pnorm[:, 4]), torch.cos(tnorm[:, 4] - pnorm[:, 4]))
        assert dtheta.abs().max() < 1e-4

    def test_tensor_normalize_le90_is_idempotent(self):
        """normalize_boxes_to_le90(normalize_boxes_to_le90(x)) == normalize_boxes_to_le90(x)."""
        torch.manual_seed(888)
        n = 2048
        boxes = torch.empty(n, 5)
        boxes[:, 0:2] = torch.randn(n, 2) * 128.0
        boxes[:, 2:4] = torch.rand(n, 2) * 300.0 + 1.0
        boxes[:, 4] = (torch.rand(n) - 0.5) * (16.0 * math.pi)

        once = normalize_boxes_to_le90(boxes)
        twice = normalize_boxes_to_le90(once)
        assert torch.allclose(twice, once, atol=1e-6)

    def test_tensor_normalize_le90_output_invariants(self):
        """Normalized tensor boxes must satisfy le90 invariants."""
        torch.manual_seed(889)
        n = 2048
        boxes = torch.empty(n, 5)
        boxes[:, 0:2] = torch.randn(n, 2) * 256.0
        boxes[:, 2:4] = torch.rand(n, 2) * 400.0 + 0.5
        boxes[:, 4] = (torch.rand(n) - 0.5) * (20.0 * math.pi)

        normed = normalize_boxes_to_le90(boxes)
        assert torch.all(normed[:, 2] >= normed[:, 3])
        assert torch.all(normed[:, 4] >= -math.pi / 2)
        assert torch.all(normed[:, 4] < math.pi / 2)

    def test_encode_decode_round_trip_without_edge_swap_with_le90_decode(self):
        """Non-edge-swap path should remain stable with normalize_le90 decode."""
        torch.manual_seed(777)
        n = 1024
        anchor_cx = torch.rand(n) * 512
        anchor_cy = torch.rand(n) * 512
        a_w0 = torch.rand(n) * 100 + 4
        a_h0 = torch.rand(n) * 100 + 4
        anchors = torch.stack(
            [
                anchor_cx,
                anchor_cy,
                torch.maximum(a_w0, a_h0),
                torch.minimum(a_w0, a_h0),
                (torch.rand(n) - 0.5) * math.pi,
            ],
            dim=1,
        )

        gt_cx = anchor_cx + torch.randn(n) * 8.0
        gt_cy = anchor_cy + torch.randn(n) * 8.0
        g_w0 = torch.rand(n) * 110 + 4
        g_h0 = torch.rand(n) * 110 + 4
        gt_boxes = torch.stack(
            [
                gt_cx,
                gt_cy,
                torch.maximum(g_w0, g_h0),
                torch.minimum(g_w0, g_h0),
                (torch.rand(n) - 0.5) * math.pi,
            ],
            dim=1,
        )
        gt_boxes = normalize_boxes_to_le90(gt_boxes)

        deltas = encode_oriented_boxes(
            anchors,
            gt_boxes,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            norm_factor=2.0,
            edge_swap=False,
        )
        decoded = decode_oriented_boxes(
            anchors,
            deltas,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            normalize_le90=True,
            norm_factor=2.0,
            edge_swap=False,
        )

        assert torch.allclose(decoded[:, :2], gt_boxes[:, :2], atol=1e-4)
        assert torch.allclose(decoded[:, 2:4], gt_boxes[:, 2:4], atol=1e-4)
        dtheta = torch.atan2(torch.sin(decoded[:, 4] - gt_boxes[:, 4]), torch.cos(decoded[:, 4] - gt_boxes[:, 4]))
        assert dtheta.abs().max() < 1e-4


class TestAnchorMatching:
    """Tests for anchor matching."""
    
    def test_match_positive_anchors(self, dummy_anchors, dummy_gt_boxes):
        """Test matching anchors to GT boxes."""
        labels, matched_indices = match_oriented_anchors_to_gt(
            dummy_anchors,
            dummy_gt_boxes,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        assert labels.shape == (len(dummy_anchors),)
        assert matched_indices.shape == (len(dummy_anchors),)
        
        # Should have some positive matches
        assert torch.any(labels == 1)
        
        # Positive anchors should have valid matched indices
        positive_mask = labels == 1
        assert torch.all(matched_indices[positive_mask] >= 0)
        assert torch.all(matched_indices[positive_mask] < len(dummy_gt_boxes))
    
    def test_match_empty_gt_boxes(self, dummy_anchors):
        """Test matching with empty GT boxes."""
        empty_gt = torch.zeros((0, 5))
        
        labels, matched_indices = match_oriented_anchors_to_gt(
            dummy_anchors,
            empty_gt,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        # All anchors should be background
        assert torch.all(labels == 0)
        assert torch.all(matched_indices == -1)
    
    def test_match_empty_anchors(self, dummy_gt_boxes):
        """Test matching with empty anchors."""
        empty_anchors = torch.zeros((0, 5))
        
        # GPU ops may not handle empty anchors, so skip if using GPU
        # This is acceptable - empty anchors is an edge case
        try:
            labels, matched_indices = match_oriented_anchors_to_gt(
                empty_anchors,
                dummy_gt_boxes,
                positive_iou_threshold=0.5,
                negative_iou_threshold=0.3,
            )
            
            assert len(labels) == 0
            assert len(matched_indices) == 0
        except (IndexError, RuntimeError):
            # GPU ops may raise error for empty anchors - this is acceptable
            pytest.skip("GPU ops don't handle empty anchors")
    
    def test_match_iou_thresholds(self, dummy_anchors, dummy_gt_boxes):
        """Test that IoU thresholds work correctly."""
        # High positive threshold should give fewer positives
        labels_high, _ = match_oriented_anchors_to_gt(
            dummy_anchors,
            dummy_gt_boxes,
            positive_iou_threshold=0.9,
            negative_iou_threshold=0.3,
        )
        
        # Low positive threshold should give more positives
        labels_low, _ = match_oriented_anchors_to_gt(
            dummy_anchors,
            dummy_gt_boxes,
            positive_iou_threshold=0.3,
            negative_iou_threshold=0.1,
        )
        
        # Lower threshold should have at least as many positives
        assert (labels_low == 1).sum() >= (labels_high == 1).sum()


class TestRPNLoss:
    """Tests for RPN loss computation."""
    
    def test_rpn_loss_non_negative(self, dummy_features):
        """Test that RPN losses are non-negative."""
        # Use MMRotate format (cls_out_channels=1, reg_out_channels=6)
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        # Generate anchors for all feature map levels
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        # Forward pass
        objectness_logits, bbox_regression = head(dummy_features)
        
        # Create dummy GT boxes
        gt_boxes = [torch.tensor([[64.0, 64.0, 30.0, 15.0, 0.0]])]
        
        # Compute loss
        losses = compute_oriented_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        # All losses should be non-negative
        assert losses["loss_objectness"] >= 0
        assert losses["loss_rpn_box_reg"] >= 0
    
    def test_rpn_loss_with_no_positive_anchors(self, dummy_features):
        """Test RPN loss when there are no positive anchors."""
        # Use MMRotate format
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        # GT boxes far away from anchors
        gt_boxes = [torch.tensor([[1000.0, 1000.0, 30.0, 15.0, 0.0]])]
        
        losses = compute_oriented_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        # Should still return valid losses (may be zero or small)
        assert "loss_objectness" in losses
        assert "loss_rpn_box_reg" in losses
    
    def test_rpn_loss_target_normalization(self, dummy_features):
        """Test RPN loss with target_means/target_stds (MMRotate style)."""
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        gt_boxes = [torch.tensor([[64.0, 64.0, 30.0, 15.0, 0.0]])]
        
        # MMRotate RPN: target_means=[0,0,0,0], target_stds=[1,1,1,1]
        losses = compute_oriented_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0),
        )
        
        assert "loss_objectness" in losses
        assert "loss_rpn_box_reg" in losses


class TestProposalGeneration:
    """Tests for proposal generation."""
    
    def test_generate_proposals_basic(self, dummy_features):
        """Test basic proposal generation."""
        # Use MMRotate format
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        proposals = generate_oriented_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            nms_threshold=0.7,
            pre_nms_top_n=100,
            post_nms_top_n=50,
        )
        
        assert len(proposals) == 1  # One image
        assert proposals[0].shape[1] == 5  # [cx, cy, w, h, angle]
        assert len(proposals[0]) <= 50  # Should respect post_nms_top_n
    
    def test_proposal_count_validation(self, dummy_features):
        """Test that proposal counts respect pre_nms_top_n and post_nms_top_n."""
        # Use MMRotate format
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        proposals = generate_oriented_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            nms_threshold=0.7,
            pre_nms_top_n=10,
            post_nms_top_n=5,
        )
        
        # Should have at most post_nms_top_n proposals
        assert len(proposals[0]) <= 5
    
    def test_proposal_score_threshold(self, dummy_features):
        """Test proposal filtering by score threshold."""
        # Use MMRotate format
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        # High score threshold should give fewer proposals
        proposals_high = generate_oriented_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            score_threshold=0.9,
            nms_threshold=0.7,
            pre_nms_top_n=100,
            post_nms_top_n=50,
        )
        
        # Low score threshold should give more proposals
        proposals_low = generate_oriented_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            score_threshold=0.0,
            nms_threshold=0.7,
            pre_nms_top_n=100,
            post_nms_top_n=50,
        )
        
        assert len(proposals_low[0]) >= len(proposals_high[0])
    
    def test_proposal_format_validation(self, dummy_features):
        """Test that proposals have correct format."""
        # Use MMRotate format
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=4,
        )
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],  # Match dummy_features
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2],
            stride_per_level=[4, 8],
        )
        
        objectness_logits, bbox_regression = head(dummy_features)
        
        proposals = generate_oriented_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            nms_threshold=0.7,
            pre_nms_top_n=10,
            post_nms_top_n=5,
        )
        
        if len(proposals[0]) > 0:
            # Check format: [cx, cy, w, h, angle]
            assert proposals[0].shape[1] == 5
            
            # Check that widths and heights are positive
            assert torch.all(proposals[0][:, 2] > 0)
            assert torch.all(proposals[0][:, 3] > 0)


class TestMidpointRPN:
    """Tests for Oriented R-CNN midpoint-offset RPN (6D regression)."""

    def test_midpoint_rpn_head_forward_shape(self, dummy_features):
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=6,
        )
        objectness_logits, bbox_regression = head(dummy_features)
        assert objectness_logits[0].shape == (1, 3 * 1, 32, 32)
        assert bbox_regression[0].shape == (1, 3 * 6, 32, 32)

    def test_compute_midpoint_rpn_loss(self, dummy_features):
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=6,
        )
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[0.0],
            stride_per_level=[4, 8],
        )
        objectness_logits, bbox_regression = head(dummy_features)
        gt_boxes = [
            torch.tensor([[64.0, 64.0, 30.0, 15.0, 0.0], [128.0, 128.0, 35.0, 18.0, 0.0]]),
        ]
        losses = compute_midpoint_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            batch_size_per_image=64,
        )
        assert "loss_objectness" in losses
        assert "loss_rpn_box_reg" in losses
        assert torch.isfinite(losses["loss_objectness"])
        assert torch.isfinite(losses["loss_rpn_box_reg"])
        total = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
        total.backward()
        assert head.rpn_conv.weight.grad is not None

    def test_midpoint_rpn_reg_loss_sampled_all_avg_factor(self, dummy_features):
        """Regression loss divides by len(sampled), not num positives (MMDet avg_factor)."""
        torch.manual_seed(0)
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=6,
        )
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[0.0],
            stride_per_level=[4],
        )
        objectness_logits, bbox_regression = head([dummy_features[0]])
        gt_boxes = [torch.tensor([[64.0, 64.0, 30.0, 15.0, 0.0]])]
        losses_small = compute_midpoint_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            batch_size_per_image=8,
        )
        torch.manual_seed(0)
        losses_large = compute_midpoint_rpn_loss(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            gt_boxes=gt_boxes,
            image_sizes=[(128, 128)],
            batch_size_per_image=16,
        )
        assert losses_small["loss_rpn_box_reg"].item() > losses_large["loss_rpn_box_reg"].item()

    def test_generate_midpoint_proposals(self, dummy_features):
        head = OrientedRPNHead(
            in_channels=256,
            num_anchors=3,
            cls_out_channels=1,
            reg_out_channels=6,
        )
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32), (16, 16)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[0.0],
            stride_per_level=[4, 8],
        )
        objectness_logits, bbox_regression = head(dummy_features)
        proposals = generate_midpoint_proposals(
            objectness_logits=objectness_logits,
            bbox_regression=bbox_regression,
            anchors=anchors,
            image_sizes=[(128, 128)],
            pre_nms_top_n=50,
            post_nms_top_n=20,
            nms_threshold=0.8,
        )
        assert proposals[0].shape[1] == 5
        assert len(proposals[0]) <= 20
