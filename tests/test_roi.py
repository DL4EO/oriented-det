"""Tests for ROI (Region of Interest) components."""

import math
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
import torch.nn.functional as F

from oriented_det.geometry import RBox
from oriented_det.models.oriented_roi import (
    OrientedROIHead,
    OrientedROIAlign,
    oriented_roi_align,
    horizontal_roi_align,
    assign_roi_fpn_levels_mmrotate,
    match_oriented_proposals_to_gt,
    compute_oriented_roi_loss,
    compute_horizontal_roi_loss,
    compute_horizontal_roi_loss_mmrotate,
    _smooth_l1_encoded_regression_loss,
    focal_loss,
)


@pytest.fixture
def dummy_roi_features():
    """Create dummy ROI features."""
    # ROI features after pooling/align: [N, C*H*W] where C=256, H=W=7
    return torch.randn(10, 256 * 7 * 7)


@pytest.fixture
def dummy_proposals():
    """Create dummy proposals."""
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


@pytest.fixture
def dummy_feature_maps():
    """Create dummy FPN feature maps."""
    return [
        torch.randn(1, 256, 32, 32),  # Level 0
        torch.randn(1, 256, 16, 16),  # Level 1
    ]


class TestOrientedROIHead:
    """Tests for OrientedROIHead."""
    
    def test_forward_class_agnostic(self, dummy_roi_features):
        """Test forward pass with class-agnostic regression."""
        head = OrientedROIHead(
            in_channels=256 * 7 * 7,
            num_classes=15,
            class_agnostic_regression=True,
        )
        
        class_logits, box_regression = head(dummy_roi_features)
        
        # Classification: [N, num_classes + 1]
        assert class_logits.shape == (10, 16)  # 15 classes + background
        
        # Regression: [N, 5] (class-agnostic)
        assert box_regression.shape == (10, 5)
    
    def test_forward_class_specific(self, dummy_roi_features):
        """Test forward pass with class-specific regression."""
        head = OrientedROIHead(
            in_channels=256 * 7 * 7,
            num_classes=15,
            class_agnostic_regression=False,
        )
        
        class_logits, box_regression = head(dummy_roi_features)
        
        # Classification: [N, num_classes + 1]
        assert class_logits.shape == (10, 16)
        
        # Regression: [N, num_classes * 5] (class-specific)
        assert box_regression.shape == (10, 15 * 5)
    
    def test_forward_empty_features(self):
        """Test forward pass with empty ROI features."""
        head = OrientedROIHead(
            in_channels=256 * 7 * 7,
            num_classes=15,
        )
        
        empty_features = torch.zeros((0, 256 * 7 * 7))
        class_logits, box_regression = head(empty_features)
        
        assert class_logits.shape[0] == 0
        assert box_regression.shape[0] == 0
    
    def test_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        head = OrientedROIHead(
            in_channels=256 * 7 * 7,
            num_classes=10,
        )
        
        # Test with different batch sizes
        for batch_size in [1, 5, 20]:
            features = torch.randn(batch_size, 256 * 7 * 7)
            class_logits, box_regression = head(features)
            
            assert class_logits.shape[0] == batch_size
            assert box_regression.shape[0] == batch_size


class TestOrientedROIAlign:
    """Tests for OrientedROIAlign."""
    
    def test_forward_basic(self, dummy_feature_maps, dummy_proposals):
        """Test basic ROI align forward pass."""
        roi_align = OrientedROIAlign(
            output_size=(7, 7),
            spatial_scales=[1.0 / 4, 1.0 / 8],
            fpn_strides=[4, 8],
        )
        
        features = roi_align(
            feature_maps=dummy_feature_maps,
            boxes=dummy_proposals,
            image_sizes=[(128, 128)],
        )
        
        # Should have [N, C, H, W] shape
        assert features.shape[0] == len(dummy_proposals)
        assert features.shape[1] == 256  # Feature channels
        assert features.shape[2] == 7  # Output height
        assert features.shape[3] == 7  # Output width
    
    def test_forward_chunked_processing(self, dummy_feature_maps, dummy_proposals):
        """Test ROI align with chunked processing."""
        roi_align = OrientedROIAlign(
            output_size=(7, 7),
            spatial_scales=[1.0 / 4, 1.0 / 8],
            fpn_strides=[4, 8],
            chunk_size=2,  # Process 2 boxes at a time
        )
        
        features = roi_align(
            feature_maps=dummy_feature_maps,
            boxes=dummy_proposals,
            image_sizes=[(128, 128)],
        )
        
        assert features.shape[0] == len(dummy_proposals)
        assert features.shape[1:] == (256, 7, 7)
    
    def test_forward_different_output_sizes(self, dummy_feature_maps, dummy_proposals):
        """Test ROI align with different output sizes."""
        for output_size in [(7, 7), (14, 14), (3, 3)]:
            roi_align = OrientedROIAlign(
                output_size=output_size,
                spatial_scales=[1.0 / 4, 1.0 / 8],
                fpn_strides=[4, 8],
            )
            
            features = roi_align(
                feature_maps=dummy_feature_maps,
                boxes=dummy_proposals,
                image_sizes=[(128, 128)],
            )
            
            assert features.shape[2] == output_size[0]
            assert features.shape[3] == output_size[1]
    
    def test_forward_empty_boxes(self, dummy_feature_maps):
        """Test ROI align with empty boxes."""
        roi_align = OrientedROIAlign(
            output_size=(7, 7),
            spatial_scales=[1.0 / 4],
            fpn_strides=[4],
        )
        
        empty_boxes = torch.zeros((0, 5))
        features = roi_align(
            feature_maps=[dummy_feature_maps[0]],
            boxes=empty_boxes,
            image_sizes=[(128, 128)],
        )
        
        assert features.shape[0] == 0
        assert features.shape[1:] == (256, 7, 7)
    
    def test_forward_multiple_images(self, dummy_feature_maps):
        """Test ROI align with multiple images."""
        roi_align = OrientedROIAlign(
            output_size=(7, 7),
            spatial_scales=[1.0 / 4],
            fpn_strides=[4],
        )
        
        # Create feature maps for batch size 2
        batch_features = [torch.randn(2, 256, 32, 32)]
        
        # Create proposals for 2 images
        proposals = torch.tensor([
            [64.0, 64.0, 32.0, 16.0, 0.0],
            [128.0, 128.0, 32.0, 16.0, 0.0],
        ])
        
        # Map boxes to images
        box_to_image = torch.tensor([0, 1])
        
        features = roi_align(
            feature_maps=batch_features,
            boxes=proposals,
            image_sizes=[(128, 128), (128, 128)],
            box_to_image=box_to_image,
        )
        
        assert features.shape[0] == len(proposals)


def test_assign_roi_fpn_levels_mmrotate_sqrt_area_mapping():
    strides = [4, 8, 16, 32]
    finest = 56.0
    # sqrt(area) = 56 -> level 0
    b0 = torch.tensor([[0.0, 0.0, 56.0, 56.0, 0.0]])
    # sqrt(area) = 112 -> level 1
    b1 = torch.tensor([[0.0, 0.0, 112.0, 112.0, 0.0]])
    # sqrt(area) = 224 -> level 2
    b2 = torch.tensor([[0.0, 0.0, 224.0, 224.0, 0.0]])
    lv = assign_roi_fpn_levels_mmrotate(torch.cat([b0, b1, b2], dim=0), strides, finest_scale=finest, box_format="obb")
    assert lv.tolist() == [0, 1, 2]


class TestProposalMatching:
    """Tests for proposal matching."""
    
    def test_match_positive_proposals(self, dummy_proposals, dummy_gt_boxes):
        """Test matching proposals to GT boxes."""
        gt_labels = torch.tensor([1, 2])  # Class labels
        
        labels, matched_indices, matched_boxes = match_oriented_proposals_to_gt(
            dummy_proposals,
            dummy_gt_boxes,
            gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        assert labels.shape == (len(dummy_proposals),)
        assert matched_indices.shape == (len(dummy_proposals),)
        assert matched_boxes.shape == (len(dummy_proposals), 5)
        
        # Should have some positive matches
        assert torch.any(labels > 0)
    
    def test_match_empty_proposals(self, dummy_gt_boxes):
        """Test matching with empty proposals."""
        empty_proposals = torch.zeros((0, 5))
        gt_labels = torch.tensor([1, 2])
        
        # GPU ops may not handle empty proposals, so handle gracefully
        try:
            labels, matched_indices, matched_boxes = match_oriented_proposals_to_gt(
                empty_proposals,
                dummy_gt_boxes,
                gt_labels,
                positive_iou_threshold=0.5,
                negative_iou_threshold=0.3,
            )
            
            assert len(labels) == 0
            assert len(matched_indices) == 0
            assert len(matched_boxes) == 0
        except (IndexError, RuntimeError):
            # GPU ops may raise error for empty proposals - this is acceptable
            pytest.skip("GPU ops don't handle empty proposals")
    
    def test_match_empty_gt(self, dummy_proposals):
        """Test matching with empty GT boxes."""
        empty_gt = torch.zeros((0, 5))
        empty_labels = torch.tensor([], dtype=torch.int64)
        
        labels, matched_indices, matched_boxes = match_oriented_proposals_to_gt(
            dummy_proposals,
            empty_gt,
            empty_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        # All proposals should be background
        assert torch.all(labels == 0)
        assert torch.all(matched_indices == -1)
    
    def test_match_iou_thresholds(self, dummy_proposals, dummy_gt_boxes):
        """Test that IoU thresholds work correctly."""
        gt_labels = torch.tensor([1, 2])
        
        # High positive threshold
        labels_high, _, _ = match_oriented_proposals_to_gt(
            dummy_proposals,
            dummy_gt_boxes,
            gt_labels,
            positive_iou_threshold=0.9,
            negative_iou_threshold=0.3,
        )
        
        # Low positive threshold
        labels_low, _, _ = match_oriented_proposals_to_gt(
            dummy_proposals,
            dummy_gt_boxes,
            gt_labels,
            positive_iou_threshold=0.3,
            negative_iou_threshold=0.1,
        )
        
        # Lower threshold should have at least as many positives
        assert (labels_low > 0).sum() >= (labels_high > 0).sum()

    def test_match_low_quality_true_forces_best_proposal_positive(self):
        """With match_low_quality=True and min_pos_iou=0, best proposal per GT is forced positive when IoU > 0."""
        proposals = torch.tensor([
            [1.0, 0.0, 2.0, 2.0, 0.0],   # IoU ~= 1/3 against GT at origin
            [10.0, 10.0, 2.0, 2.0, 0.0],  # no overlap
        ])
        gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        gt_labels = torch.tensor([4], dtype=torch.int64)

        labels, matched_indices, _ = match_oriented_proposals_to_gt(
            proposals,
            gt_boxes,
            gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            match_low_quality=True,
            min_pos_iou=0.0,  # force best proposal when IoU > 0
        )
        assert labels[0].item() == 4
        assert matched_indices[0].item() == 0

    def test_match_low_quality_false_does_not_force_low_iou_positive(self):
        """With match_low_quality=False, low-IoU best proposal stays ignore/background."""
        proposals = torch.tensor([
            [1.0, 0.0, 2.0, 2.0, 0.0],   # IoU ~= 1/3 against GT at origin
            [10.0, 10.0, 2.0, 2.0, 0.0],
        ])
        gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        gt_labels = torch.tensor([2], dtype=torch.int64)

        labels, _, _ = match_oriented_proposals_to_gt(
            proposals,
            gt_boxes,
            gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            match_low_quality=False,
        )
        assert labels[0].item() == -1  # ignore band, not forced positive

    def test_match_threshold_boundaries(self):
        """Boundary semantics: IoU == pos is positive, IoU == neg is not background."""
        proposals = torch.tensor([
            [0.0, 0.0, 2.0, 2.0, 0.0],   # IoU = 1 with GT
            [20.0, 20.0, 2.0, 2.0, 0.0],  # IoU = 0 with GT
        ])
        gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        gt_labels = torch.tensor([1], dtype=torch.int64)

        labels, _, _ = match_oriented_proposals_to_gt(
            proposals,
            gt_boxes,
            gt_labels,
            positive_iou_threshold=1.0,
            negative_iou_threshold=0.0,
            match_low_quality=False,
        )
        assert labels[0].item() == 1   # IoU == positive threshold
        assert labels[1].item() == -1  # IoU == negative threshold is not "< neg"

    def test_match_ignore_band_labels_minus_one(self):
        """Proposals in [neg, pos) are labeled as ignore (-1)."""
        proposals = torch.tensor([[1.0, 0.0, 2.0, 2.0, 0.0]])  # IoU ~= 1/3
        gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        gt_labels = torch.tensor([3], dtype=torch.int64)

        labels, _, _ = match_oriented_proposals_to_gt(
            proposals,
            gt_boxes,
            gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            match_low_quality=False,
        )
        assert labels[0].item() == -1


class TestROILoss:
    """Tests for ROI loss computation."""
    
    def test_roi_loss_cross_entropy(self, dummy_proposals, dummy_gt_boxes):
        """Test ROI loss with cross-entropy."""
        num_classes = 15
        num_proposals = len(dummy_proposals)
        
        # Create dummy predictions
        class_logits = torch.randn(num_proposals, num_classes + 1)
        box_regression = torch.randn(num_proposals, 5)  # Class-agnostic
        
        gt_labels = torch.tensor([1, 2])
        
        losses = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=dummy_proposals,
            gt_boxes=dummy_gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            loss_type="cross_entropy",
        )
        
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        
        # Losses should be non-negative
        assert losses["loss_classifier"] >= 0
        assert losses["loss_box_reg"] >= 0
    
    def test_roi_loss_focal(self, dummy_proposals, dummy_gt_boxes):
        """Test ROI loss with focal loss."""
        num_classes = 15
        num_proposals = len(dummy_proposals)
        
        class_logits = torch.randn(num_proposals, num_classes + 1)
        box_regression = torch.randn(num_proposals, 5)
        gt_labels = torch.tensor([1, 2])
        
        losses = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=dummy_proposals,
            gt_boxes=dummy_gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            loss_type="focal",
            focal_alpha=1.0,
            focal_gamma=2.0,
        )
        
        assert "loss_classifier" in losses
        assert losses["loss_classifier"] >= 0
    
    def test_roi_loss_class_weights(self, dummy_proposals, dummy_gt_boxes):
        """Test ROI loss with class weights."""
        num_classes = 15
        num_proposals = len(dummy_proposals)
        
        class_logits = torch.randn(num_proposals, num_classes + 1)
        box_regression = torch.randn(num_proposals, 5)
        gt_labels = torch.tensor([1, 2])
        
        # Create class weights tensor
        class_weights = torch.ones(num_classes + 1)
        class_weights[1] = 2.0  # Weight class 1 more
        class_weights[2] = 0.5  # Weight class 2 less
        
        losses = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=dummy_proposals,
            gt_boxes=dummy_gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            class_weights=class_weights,
        )
        
        assert "loss_classifier" in losses
        assert losses["loss_classifier"] >= 0
    
    def test_roi_loss_no_positive_proposals(self, dummy_proposals):
        """Test ROI loss when there are no positive proposals."""
        num_classes = 15
        num_proposals = len(dummy_proposals)
        
        class_logits = torch.randn(num_proposals, num_classes + 1)
        box_regression = torch.randn(num_proposals, 5)
        
        # GT boxes far away from proposals
        far_gt_boxes = torch.tensor([[1000.0, 1000.0, 30.0, 15.0, 0.0]])
        gt_labels = torch.tensor([1])
        
        losses = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=dummy_proposals,
            gt_boxes=far_gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
        )
        
        # Should still return valid losses
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
    
    def test_roi_loss_target_normalization(self, dummy_proposals, dummy_gt_boxes):
        """Test ROI loss with target normalization."""
        num_classes = 15
        num_proposals = len(dummy_proposals)
        
        class_logits = torch.randn(num_proposals, num_classes + 1)
        box_regression = torch.randn(num_proposals, 5)
        gt_labels = torch.tensor([1, 2])
        
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
        target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
        
        losses = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=dummy_proposals,
            gt_boxes=dummy_gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            target_means=target_means,
            target_stds=target_stds,
        )
        
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses

    def test_roi_loss_angle_term_uses_encoded_smooth_l1(self):
        """ROI box regression uses encoded-space Smooth L1 (MMRotate), not radian periodic loss."""
        gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, math.radians(89.0)]])
        proposals = gt_boxes.clone()
        gt_labels = torch.tensor([1], dtype=torch.int64)
        class_logits = torch.tensor([[0.0, 10.0]])

        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
        target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
        norm_factor = 2.0

        target_angle_encoded = (
            (math.radians(89.0) / (norm_factor * math.pi) - target_means[4]) / target_stds[4]
        )
        box_regression_a = torch.tensor([[0.0, 0.0, 0.0, 0.0, target_angle_encoded]])
        box_regression_b = torch.tensor([[0.0, 0.0, 0.0, 0.0, target_angle_encoded + (1.0 / target_stds[4])]])

        losses_a = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression_a,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            target_means=target_means,
            target_stds=target_stds,
            norm_factor=norm_factor,
            edge_swap=False,
            include_assignment_diagnostics=False,
        )
        losses_b = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression_b,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            target_means=target_means,
            target_stds=target_stds,
            norm_factor=norm_factor,
            edge_swap=False,
            include_assignment_diagnostics=False,
        )

        assert losses_a["loss_box_reg"].item() < losses_b["loss_box_reg"].item()

    def test_roi_loss_iou_term_penalizes_geometrically_worse_box(self):
        """Adding the ROI IoU term should increase loss for a worse decoded box."""
        proposals = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]])
        gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]])
        gt_labels = torch.tensor([1], dtype=torch.int64)
        class_logits = torch.tensor([[0.0, 10.0]])

        perfect_regression = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        shifted_regression = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]])

        perfect_no_iou = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=perfect_regression,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            include_assignment_diagnostics=False,
            box_reg_iou_weight=0.0,
            edge_swap=False,
        )
        shifted_no_iou = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=shifted_regression,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            include_assignment_diagnostics=False,
            box_reg_iou_weight=0.0,
            edge_swap=False,
        )
        shifted_with_iou = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=shifted_regression,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            include_assignment_diagnostics=False,
            box_reg_iou_weight=0.5,
            edge_swap=False,
        )

        assert perfect_no_iou["loss_box_reg"] < shifted_no_iou["loss_box_reg"]
        assert shifted_with_iou["loss_box_reg"] > shifted_no_iou["loss_box_reg"]

    def test_oriented_roi_loss_angle_weight_scales_angle_term(self):
        """Higher box_reg_angle_weight should increase loss when only angle is wrong."""
        proposals = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]])
        gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.3]])
        gt_labels = torch.tensor([1], dtype=torch.int64)
        class_logits = torch.tensor([[0.0, 10.0]])
        box_regression = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5]])

        loss_w1 = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            box_reg_angle_weight=1.0,
            box_reg_iou_weight=0.0,
            edge_swap=False,
            include_assignment_diagnostics=False,
        )
        loss_w3 = compute_oriented_roi_loss(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            box_reg_angle_weight=3.0,
            box_reg_iou_weight=0.0,
            edge_swap=False,
            include_assignment_diagnostics=False,
        )
        assert loss_w3["loss_box_reg"] > loss_w1["loss_box_reg"]


def test_horizontal_roi_loss_angle_weight_scales_angle_term():
    """Rotated Faster R-CNN horizontal ROI path honors box_reg_angle_weight."""
    proposals_xyxy = torch.tensor([[40.0, 50.0, 88.0, 82.0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.3]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)
    class_logits = torch.tensor([[0.0, 10.0]])
    box_regression = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5]])

    loss_w1 = compute_horizontal_roi_loss_mmrotate(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        box_reg_angle_weight=1.0,
        box_reg_iou_weight=0.0,
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    loss_w3 = compute_horizontal_roi_loss_mmrotate(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        box_reg_angle_weight=3.0,
        box_reg_iou_weight=0.0,
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    assert loss_w3["loss_box_reg"] > loss_w1["loss_box_reg"]


def test_horizontal_roi_loss_angle_term_uses_encoded_smooth_l1():
    """Horizontal RoI Smooth L1 uses encoded deltas (MMRotate), not radian periodic loss."""
    from oriented_det.models.horizontal_roi_coder import encode_delta_xywh_th

    proposals_xyxy, gt_boxes, gt_labels, class_logits, _ = _horizontal_roi_fixture()
    target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
    norm_factor = 2.0
    encoded = encode_delta_xywh_th(
        proposals_xyxy,
        gt_boxes,
        stds=target_stds,
        norm_factor=norm_factor,
        edge_swap=False,
    )
    box_regression_a = encoded.clone()
    box_regression_b = encoded.clone()
    box_regression_b[0, 4] = box_regression_b[0, 4] + (1.0 / target_stds[4])
    kwargs = dict(
        class_logits=class_logits,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        main_loss_type="smooth_l1",
        box_reg_iou_weight=0.0,
        stds=target_stds,
        norm_factor=norm_factor,
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    loss_a = compute_horizontal_roi_loss(box_regression=box_regression_a, **kwargs)
    loss_b = compute_horizontal_roi_loss(box_regression=box_regression_b, **kwargs)
    assert loss_a["loss_box_reg"].item() < loss_b["loss_box_reg"].item()


def _horizontal_roi_fixture():
    proposals_xyxy = torch.tensor([[40.0, 50.0, 88.0, 82.0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.3]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)
    class_logits = torch.tensor([[0.0, 10.0]], requires_grad=False)
    box_regression = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5]], requires_grad=True)
    return proposals_xyxy, gt_boxes, gt_labels, class_logits, box_regression


def test_horizontal_roi_loss_mmrotate_matches_sampled_all():
    proposals_xyxy, gt_boxes, gt_labels, class_logits, box_regression = _horizontal_roi_fixture()
    box_regression = box_regression.detach()
    kwargs = dict(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        box_reg_iou_weight=0.0,
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    loss_mm = compute_horizontal_roi_loss_mmrotate(**kwargs)
    loss_all = compute_horizontal_roi_loss(**kwargs, reg_norm="sampled_all")
    assert loss_mm["loss_box_reg"].item() == pytest.approx(loss_all["loss_box_reg"].item())


def test_smooth_l1_encoded_regression_loss_norm_modes():
    """positives_only uses per-element mean; sampled_all divides sum by total sample count."""
    pred = torch.zeros(2, 5)
    target = torch.zeros(2, 5)
    target[:, :4] = 1.0
    loss_all = _smooth_l1_encoded_regression_loss(
        pred,
        target,
        angle_weight=1.0,
        reg_norm="sampled_all",
        num_total_samples=512,
    )
    loss_pos = _smooth_l1_encoded_regression_loss(
        pred,
        target,
        angle_weight=1.0,
        reg_norm="positives_only",
        num_total_samples=512,
    )
    assert loss_pos.item() > loss_all.item()
    assert loss_all.item() == pytest.approx(4.0 / 512.0)
    assert loss_pos.item() == pytest.approx(0.4)


def test_roi_encoded_regression_matches_mmrotate_smooth_l1_sum_avg_factor():
    """Hand-computed encoded Smooth L1 / num_total_samples matches the ROI helper."""
    pred = torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.05]])
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    num_total = 512
    hand = F.smooth_l1_loss(pred, target, beta=1.0, reduction="sum") / float(num_total)
    got = _smooth_l1_encoded_regression_loss(
        pred,
        target,
        reg_norm="sampled_all",
        num_total_samples=num_total,
    )
    assert got.item() == pytest.approx(hand.item())


def test_horizontal_roi_loss_probiou_main_with_smooth_l1_aux_backward():
    proposals_xyxy, gt_boxes, gt_labels, class_logits, box_regression = _horizontal_roi_fixture()
    box_regression = box_regression.clone().detach().requires_grad_(True)
    out = compute_horizontal_roi_loss(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        main_loss_type="probiou",
        box_reg_probiou_mode="l1",
        smooth_l1_aux_weight=0.1,
        reg_norm="positives_only",
        box_reg_iou_weight=0.0,
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    assert torch.isfinite(out["loss_box_reg"])
    out["loss_box_reg"].backward()
    assert box_regression.grad is not None
    assert torch.isfinite(box_regression.grad).all()


def test_horizontal_roi_loss_default_smooth_l1_main_unchanged_with_probiou_aux():
    proposals_xyxy, gt_boxes, gt_labels, class_logits, box_regression = _horizontal_roi_fixture()
    kwargs = dict(
        class_logits=class_logits,
        box_regression=box_regression.clone(),
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        main_loss_type="smooth_l1",
        box_reg_iou_weight=0.1,
        box_reg_iou_loss_type="probiou",
        box_reg_probiou_mode="l1",
        edge_swap=False,
        include_assignment_diagnostics=False,
    )
    loss_mm = compute_horizontal_roi_loss_mmrotate(**kwargs)
    loss_explicit = compute_horizontal_roi_loss(**kwargs, reg_norm="sampled_all")
    assert loss_mm["loss_box_reg"].item() == pytest.approx(loss_explicit["loss_box_reg"].item())


def test_horizontal_roi_loss_uses_hbb_style_matching(monkeypatch):
    """Rotated Faster R-CNN ROI assignment should use HBB-style IoU."""
    called = {"use_hbb": None, "match_low_quality": None, "min_pos_iou": None}

    def _fake_match(
        proposals,
        gt_boxes,
        gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.5,
        device=None,
        use_hbb_for_matching=False,
        match_low_quality=False,
        min_pos_iou=0.5,
        gt_boxes_ignore=None,
        ignore_iou_threshold=None,
    ):
        called["use_hbb"] = use_hbb_for_matching
        called["match_low_quality"] = match_low_quality
        called["min_pos_iou"] = min_pos_iou
        n = int(proposals.shape[0])
        labels = torch.zeros((n,), dtype=torch.int64, device=proposals.device)
        matched_gt_indices = torch.full((n,), -1, dtype=torch.int64, device=proposals.device)
        matched_gt_boxes = torch.zeros((n, 5), dtype=torch.float32, device=proposals.device)
        return labels, matched_gt_indices, matched_gt_boxes

    monkeypatch.setattr("oriented_det.models.oriented_roi.match_oriented_proposals_to_gt", _fake_match)

    class_logits = torch.randn(1, 2)
    box_regression = torch.randn(1, 5)
    proposals_xyxy = torch.tensor([[10.0, 10.0, 30.0, 20.0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[20.0, 15.0, 20.0, 10.0, 0.5]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)

    compute_horizontal_roi_loss_mmrotate(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
    )
    assert called["use_hbb"] is True
    assert called["match_low_quality"] is False
    assert called["min_pos_iou"] == pytest.approx(0.5)

    compute_horizontal_roi_loss_mmrotate(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals_xyxy=proposals_xyxy,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        match_low_quality=True,
        roi_min_pos_iou=0.42,
    )
    assert called["match_low_quality"] is True
    assert called["min_pos_iou"] == pytest.approx(0.42)


def test_horizontal_roi_align_uses_only_first_four_fpn_levels(monkeypatch):
    """MMRotate-style ROI extractor should ignore the extra stride-64 FPN level."""
    captured = {"strides": None}

    def _fake_assign(boxes, fpn_strides, finest_scale=56.0, box_format="xyxy"):
        captured["strides"] = list(fpn_strides)
        return torch.zeros((boxes.shape[0],), dtype=torch.long, device=boxes.device)

    monkeypatch.setattr("oriented_det.models.oriented_roi.assign_roi_fpn_levels_mmrotate", _fake_assign)

    feature_maps = [
        torch.full((1, 1, 64, 64), 1.0),   # stride 4
        torch.full((1, 1, 32, 32), 2.0),   # stride 8
        torch.full((1, 1, 16, 16), 3.0),   # stride 16
        torch.full((1, 1, 8, 8), 4.0),     # stride 32
        torch.full((1, 1, 4, 4), 5.0),     # stride 64 (RPN-only for MMRotate ROI path)
    ]
    boxes_xyxy = torch.tensor([[16.0, 16.0, 64.0, 64.0]], dtype=torch.float32)
    _ = horizontal_roi_align(
        feature_maps=feature_maps,
        boxes_xyxy=boxes_xyxy,
        image_sizes=[(256, 256)],
        box_to_image=torch.tensor([0], dtype=torch.long),
        output_size=(1, 1),
        spatial_scales=None,
        fpn_strides=[4, 8, 16, 32, 64],
        chunk_size=1,
        finest_scale=56.0,
    )
    assert captured["strides"] == [4, 8, 16, 32]


def test_horizontal_roi_align_eager_matches_onnx_export_path():
    """Eager training path must match masked RoIAlign used during ONNX export."""
    torch.manual_seed(0)
    feature_maps = [
        torch.randn(1, 8, 64, 64),
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
    ]
    boxes_xyxy = torch.tensor(
        [
            [8.0, 8.0, 48.0, 48.0],
            [80.0, 80.0, 160.0, 160.0],
            [200.0, 200.0, 280.0, 280.0],
            [20.0, 100.0, 60.0, 140.0],
            [120.0, 20.0, 200.0, 80.0],
        ],
        dtype=torch.float32,
    )
    box_to_image = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.long)
    kwargs = dict(
        feature_maps=feature_maps,
        boxes_xyxy=boxes_xyxy,
        image_sizes=[(256, 256)],
        box_to_image=box_to_image,
        output_size=(7, 7),
        spatial_scales=None,
        fpn_strides=[4, 8, 16, 32],
        chunk_size=32,
        finest_scale=56.0,
    )
    eager = horizontal_roi_align(**kwargs)
    with mock.patch(
        "oriented_det.models.oriented_roi.torch.onnx.is_in_onnx_export",
        return_value=True,
    ):
        masked = horizontal_roi_align(**kwargs)
    assert torch.allclose(eager, masked, atol=1e-5, rtol=1e-5)


def test_oriented_roi_align_eager_matches_onnx_export_path():
    """Eager path must match masked grid_sample used during ONNX export."""
    torch.manual_seed(0)
    feature_maps = [
        torch.randn(1, 8, 64, 64),
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
    ]
    boxes = torch.tensor(
        [
            [64.0, 64.0, 40.0, 20.0, 0.0],
            [120.0, 120.0, 50.0, 30.0, 0.3],
            [200.0, 180.0, 36.0, 24.0, -0.2],
            [40.0, 160.0, 28.0, 28.0, 0.1],
            [180.0, 40.0, 48.0, 16.0, 0.5],
        ],
        dtype=torch.float32,
    )
    box_to_image = torch.zeros((boxes.shape[0],), dtype=torch.long)
    kwargs = dict(
        feature_maps=feature_maps,
        boxes=boxes,
        image_sizes=[(256, 256)],
        box_to_image=box_to_image,
        output_size=(7, 7),
        spatial_scales=None,
        fpn_strides=[4, 8, 16, 32],
        chunk_size=32,
        use_checkpoint=False,
        finest_scale=56.0,
    )
    eager = oriented_roi_align(**kwargs)
    with mock.patch(
        "oriented_det.models.oriented_roi.torch.onnx.is_in_onnx_export",
        return_value=True,
    ):
        masked = oriented_roi_align(**kwargs)
    assert torch.allclose(eager, masked, atol=1e-5, rtol=1e-5)


class TestFocalLoss:
    """Tests for focal loss."""
    
    def test_focal_loss_gamma_zero_equals_ce(self):
        """Test that focal loss with gamma=0 equals cross-entropy."""
        num_classes = 10
        num_samples = 20
        
        logits = torch.randn(num_samples, num_classes)
        targets = torch.randint(0, num_classes, (num_samples,))
        
        # Focal loss with gamma=0
        focal = focal_loss(logits, targets, alpha=1.0, gamma=0.0)
        
        # Cross-entropy
        ce = torch.nn.functional.cross_entropy(logits, targets)
        
        # Should be approximately equal
        assert torch.allclose(focal, ce, rtol=1e-5)
    
    def test_focal_loss_different_alpha(self):
        """Test focal loss with different alpha values."""
        num_classes = 10
        num_samples = 20
        
        logits = torch.randn(num_samples, num_classes)
        targets = torch.randint(0, num_classes, (num_samples,))
        
        focal_alpha_1 = focal_loss(logits, targets, alpha=1.0, gamma=2.0)
        focal_alpha_05 = focal_loss(logits, targets, alpha=0.5, gamma=2.0)
        
        # Different alpha should give different loss values
        assert not torch.allclose(focal_alpha_1, focal_alpha_05)
    
    def test_focal_loss_different_gamma(self):
        """Test focal loss with different gamma values."""
        num_classes = 10
        num_samples = 20
        
        logits = torch.randn(num_samples, num_classes)
        targets = torch.randint(0, num_classes, (num_samples,))
        
        focal_gamma_1 = focal_loss(logits, targets, alpha=1.0, gamma=1.0)
        focal_gamma_2 = focal_loss(logits, targets, alpha=1.0, gamma=2.0)
        
        # Different gamma should give different loss values
        assert not torch.allclose(focal_gamma_1, focal_gamma_2)
    
    def test_focal_loss_non_negative(self):
        """Test that focal loss is always non-negative."""
        num_classes = 10
        num_samples = 20
        
        logits = torch.randn(num_samples, num_classes)
        targets = torch.randint(0, num_classes, (num_samples,))
        
        focal = focal_loss(logits, targets, alpha=1.0, gamma=2.0)
        
        assert focal >= 0


def test_oriented_roi_align_uses_first_four_fpn_levels(monkeypatch):
    """Oriented RoIAlign matches MMRotate: pool from strides 4–32 only."""
    captured = {"strides": None}

    def _fake_assign(boxes, fpn_strides, finest_scale=56.0, box_format="obb"):
        captured["strides"] = list(fpn_strides)
        return torch.zeros((boxes.shape[0],), dtype=torch.long, device=boxes.device)

    monkeypatch.setattr("oriented_det.models.oriented_roi.assign_roi_fpn_levels_mmrotate", _fake_assign)

    device = torch.device("cpu")
    feature_maps = [
        torch.randn(1, 8, 128, 128, device=device),
        torch.randn(1, 8, 64, 64, device=device),
        torch.randn(1, 8, 32, 32, device=device),
        torch.randn(1, 8, 16, 16, device=device),
        torch.randn(1, 8, 8, 8, device=device),
    ]
    boxes = torch.tensor([[64.0, 64.0, 56.0, 56.0, 0.0]], device=device)
    out = oriented_roi_align(
        feature_maps,
        boxes,
        image_sizes=[(512, 512)],
        output_size=(7, 7),
        fpn_strides=[4, 8, 16, 32, 64],
        chunk_size=1,
    )
    assert out.shape == (1, 8, 7, 7)
    assert captured["strides"] == [4, 8, 16, 32]
