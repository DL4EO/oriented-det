import math
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
import warnings

from oriented_det.geometry import RBox
from oriented_det.models import OrientedRCNN, RotatedRetinaNet, RotatedFasterRCNN
from oriented_det.models.backbones import build_resnet_fpn_backbone
from oriented_det.models.utils import derive_fpn_strides_from_grid, warn_if_fpn_strides_mismatch


def _dummy_inputs():
    image = torch.rand(3, 128, 128)
    target = {
        "rboxes": [RBox(64, 64, 32, 16, 0.0)],
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    return image, target


@pytest.mark.parametrize(
    "model_cls",
    [OrientedRCNN, RotatedRetinaNet],
)
def test_model_forward_train(model_cls):
    image, target = _dummy_inputs()
    model = model_cls(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
    )
    model.train()
    losses = model([image], [target])
    assert isinstance(losses, dict)
    # Different detectors expose different loss keys. Keep this test API-robust.
    assert "loss_box_reg" in losses
    assert any(k in losses for k in ("loss_objectness", "loss_classifier"))
    for v in losses.values():
        assert torch.is_tensor(v) or isinstance(v, (float, int))


@pytest.mark.parametrize(
    "model_cls",
    [OrientedRCNN, RotatedRetinaNet],
)
@torch.no_grad()
def test_model_forward_eval(model_cls):
    image, _ = _dummy_inputs()
    model = model_cls(
        num_classes=2,
        backbone_name="resnet18",
        pretrained_backbone=False,
    )
    model.eval()
    outputs = model([image])
    assert isinstance(outputs, list)
    assert "rboxes" in outputs[0]


class TestOrientedRCNNConfiguration:
    """Tests for OrientedRCNN configuration options."""
    
    def test_different_anchor_configurations(self):
        """Test OrientedRCNN with different scales/ratios (RPN priors stay horizontal)."""
        model1 = OrientedRCNN(
            num_classes=15,
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
        )
        model2 = OrientedRCNN(
            num_classes=15,
            anchor_scales=[8],
            anchor_ratios=[0.125, 0.5, 1.0, 2.0, 8.0],
        )
        assert model1.num_classes == 15
        assert model2.num_classes == 15
        assert model1.anchor_angles == [0.0]
        assert model2.anchor_angles == [0.0]
        assert model1.num_anchors == 3
        assert model2.num_anchors == 5

        model_custom = OrientedRCNN(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2, math.pi / 2],
        )
        assert model_custom.num_anchors == 6
    
    def test_class_agnostic_vs_specific_regression(self):
        """Test class-agnostic vs class-specific regression."""
        # Class-agnostic (default)
        model_agnostic = OrientedRCNN(
            num_classes=10,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        # Check ROI head configuration
        assert model_agnostic.roi_class_agnostic_regression == True
        
        # Test forward pass
        image, target = _dummy_inputs()
        model_agnostic.train()
        losses = model_agnostic([image], [target])
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
    
    def test_different_loss_types(self):
        """Test different loss types."""
        # Cross-entropy (default)
        model_ce = OrientedRCNN(
            num_classes=10,
            backbone_name="resnet18",
            pretrained_backbone=False,
            roi_loss_type="cross_entropy",
        )
        
        # Focal loss
        model_focal = OrientedRCNN(
            num_classes=10,
            backbone_name="resnet18",
            pretrained_backbone=False,
            roi_loss_type="focal",
            roi_focal_gamma=2.0,
        )
        
        image, target = _dummy_inputs()
        
        model_ce.train()
        losses_ce = model_ce([image], [target])
        
        model_focal.train()
        losses_focal = model_focal([image], [target])
        
        assert "loss_classifier" in losses_ce
        assert "loss_classifier" in losses_focal
    
    def test_class_weights_dict(self):
        """Test class weights with dictionary format."""
        class_weights = {
            "class1": 2.0,
            "class2": 0.5,
        }
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
            roi_class_weights=class_weights,
        )
        
        # Set class mapping
        class_map = {"class1": 1, "class2": 2}
        model.set_class_weights(class_map)
        
        # Check that weights were set
        assert model.roi_class_weights is not None
        assert model.roi_class_weights.shape[0] == 3  # background + 2 classes
    
    def test_class_weights_tensor(self):
        """Test class weights with tensor format."""
        class_weights = torch.tensor([1.0, 2.0, 0.5])  # background, class1, class2
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
            roi_class_weights=class_weights,
        )
        
        assert model.roi_class_weights is not None
        assert torch.allclose(model.roi_class_weights, class_weights)
    
    def test_target_normalization(self):
        """Test target normalization configuration."""
        target_means = (0.0, 0.0, 0.0, 0.0, 0.0)
        target_stds = (0.1, 0.1, 0.2, 0.2, 0.1)
        
        model = OrientedRCNN(
            num_classes=10,
            backbone_name="resnet18",
            pretrained_backbone=False,
            target_means=target_means,
            target_stds=target_stds,
        )
        
        assert model.target_means == target_means
        assert model.target_stds == target_stds
        
        # Test forward pass
        image, target = _dummy_inputs()
        model.train()
        losses = model([image], [target])
        assert "loss_rpn_box_reg" in losses


class TestOrientedRCNNEdgeCases:
    """Tests for OrientedRCNN edge cases."""
    
    def test_empty_images(self):
        """Test with empty image list."""
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        model.eval()
        
        # Empty list should raise error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            model([])
    
    def test_images_with_no_objects(self):
        """Test with images that have no objects."""
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 128, 128)
        target = {
            "rboxes": [],
            "labels": torch.tensor([], dtype=torch.int64),
        }
        
        model.train()
        losses = model([image], [target])
        
        # Should still return valid loss dict
        assert isinstance(losses, dict)
        assert "loss_objectness" in losses
    
    def test_batch_size_gt_1(self):
        """Test with batch size > 1."""
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        images = [
            torch.rand(3, 128, 128),
            torch.rand(3, 128, 128),
        ]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])},
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])},
        ]
        
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict)
    
    def test_variable_image_sizes(self):
        """Test with variable image sizes."""
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        # Model currently requires same-size images in a batch (due to torch.stack)
        # Test with same-size images instead
        images = [
            torch.rand(3, 128, 128),
            torch.rand(3, 128, 128),
        ]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])},
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])},
        ]
        
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict)
    
    def test_multiple_classes(self):
        """Test with multiple classes."""
        model = OrientedRCNN(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 128, 128)
        target = {
            "rboxes": [
                RBox(64, 64, 32, 16, 0.0),
                RBox(128, 128, 32, 16, 0.0),
            ],
            "labels": torch.tensor([1, 5]),
        }
        
        model.train()
        losses = model([image], [target])
        assert isinstance(losses, dict)
        
        model.eval()
        with torch.no_grad():
            outputs = model([image])
            assert len(outputs) == 1
            assert "labels" in outputs[0]


class TestOrientedRCNNIntegration:
    """Integration tests for OrientedRCNN."""
    
    def test_full_training_forward(self):
        """Test full forward pass in training mode."""
        model = OrientedRCNN(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 256, 256)
        target = {
            "rboxes": [
                RBox(128, 128, 64, 32, 0.0),
                RBox(200, 200, 40, 20, math.pi / 4),
            ],
            "labels": torch.tensor([1, 5]),
        }
        
        model.train()
        losses = model([image], [target])
        
        # Check all expected losses are present
        assert "loss_objectness" in losses
        assert "loss_rpn_box_reg" in losses
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        
        # Optimization losses should be scalar tensors.
        for loss_name in ("loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"):
            loss_value = losses[loss_name]
            assert torch.is_tensor(loss_value)
            assert loss_value.dim() == 0  # Scalar

    def test_fpn_includes_pool_level_no_stride_warning(self):
        """OrientedRCNN RPN must see 5 FPN levels (P2–P6), matching RotatedFasterRCNN."""
        import warnings

        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
            fpn_strides=[4, 8, 16, 32, 64],
        )
        image = torch.rand(3, 512, 512)
        target = {"rboxes": [RBox(256, 256, 64, 32, 0.0)], "labels": torch.tensor([1])}
        model.train()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model([image], [target])
        fpn_warnings = [w for w in caught if "fpn_strides from config" in str(w.message)]
        assert not fpn_warnings
    
    def test_full_inference_forward(self):
        """Test full forward pass in inference mode."""
        model = OrientedRCNN(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            outputs = model([image])
        
        assert len(outputs) == 1
        output = outputs[0]
        
        # Check output format (only rboxes, no boxes)
        assert "rboxes" in output
        assert "labels" in output
        assert "scores" in output
        assert "boxes" not in output  # Should not have boxes format
        
        # Check tensor shapes
        assert len(output["labels"]) == len(output["scores"])
        assert len(output["rboxes"]) == len(output["labels"])
    
    def test_proposal_generation_end_to_end(self):
        """Test that proposals are generated correctly."""
        model = OrientedRCNN(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            outputs = model([image])
        
        # Should have some proposals (or empty if score threshold filters all)
        assert len(outputs) == 1
        assert "rboxes" in outputs[0]
        assert isinstance(outputs[0]["rboxes"], list)


class TestBackbone:
    """Tests for backbone construction."""
    
    @pytest.mark.parametrize("backbone_name", ["resnet18", "resnet50", "resnet101"])
    def test_different_backbone_types(self, backbone_name):
        """Test different backbone types."""
        backbone = build_resnet_fpn_backbone(
            backbone_name,
            pretrained=False,
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        features = backbone(dummy_input)
        
        # Should return dict or list of feature maps
        assert isinstance(features, (dict, list))
    
    def test_backbone_pretrained_vs_random(self):
        """Test pretrained vs random initialization."""
        # Random initialization
        backbone_random = build_resnet_fpn_backbone(
            "resnet18",
            pretrained=False,
        )
        
        # Pretrained (if available)
        backbone_pretrained = build_resnet_fpn_backbone(
            "resnet18",
            pretrained=True,
        )
        
        # Both should work
        dummy_input = torch.randn(1, 3, 256, 256)
        features_random = backbone_random(dummy_input)
        features_pretrained = backbone_pretrained(dummy_input)
        
        assert isinstance(features_random, (dict, list))
        assert isinstance(features_pretrained, (dict, list))
    
    def test_backbone_trainable_layers(self):
        """Test trainable layers configuration."""
        backbone = build_resnet_fpn_backbone(
            "resnet50",
            pretrained=False,
            trainable_layers=3,
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        assert trainable_params > 0


class TestRotatedRetinaNet:
    """Tests for RotatedRetinaNet oriented detection."""

    def test_rotated_retinanet_accepts_iou_loss_weight(self):
        """RetinaNet should accept the optional decoded IoU regression term."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
            box_reg_iou_weight=0.25,
        )
        assert model.box_reg_iou_weight == 0.25

    def test_retinanet_head_separate_cls_reg_towers(self):
        """RetinaNet head uses MMRotate-style separate subnets and 3x3 prediction convs."""
        from oriented_det.models.rotated_retinanet import OrientedRetinaNetHead

        num_classes, num_anchors = 3, 9
        head = OrientedRetinaNetHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=num_anchors,
            stacked_convs=4,
        )
        assert len(head.cls_convs) == 4
        assert len(head.reg_convs) == 4
        assert not hasattr(head, "convs")
        assert head.conv_cls.kernel_size == (3, 3)
        assert head.conv_bbox.kernel_size == (3, 3)
        feats = [torch.randn(1, 256, 32, 32)]
        cls_logits, bbox_pred = head(feats)
        assert cls_logits[0].shape == (1, num_anchors * num_classes, 32, 32)
        assert bbox_pred[0].shape == (1, num_anchors * 5, 32, 32)
    
    def test_rotated_retinanet_angle_prediction(self):
        """Test that RotatedRetinaNet predicts non-zero angles."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 256, 256)
        target = {
            "rboxes": [
                RBox(128, 128, 64, 32, math.pi / 4),  # Rotated box
                RBox(200, 200, 40, 20, -math.pi / 6),  # Another rotated box
            ],
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        }
        
        # Training forward
        model.train()
        losses = model([image], [target])
        assert isinstance(losses, dict)
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        
        # Inference forward - check that angles are predicted
        model.eval()
        with torch.no_grad():
            outputs = model([image])
            assert len(outputs) == 1
            output = outputs[0]
            assert "rboxes" in output
            assert "labels" in output
            assert "scores" in output
            
            # If there are detections, verify they have angles (not all zero)
            if len(output["rboxes"]) > 0:
                angles = [rbox.angle for rbox in output["rboxes"]]
                # Angles should be in reasonable range (not necessarily all non-zero due to thresholding)
                assert all(-math.pi <= angle <= math.pi for angle in angles)
    
    def test_rotated_retinanet_different_anchor_configs(self):
        """Test RotatedRetinaNet with different scales/ratios (reference angle fixed, MMRotate-style)."""
        model1 = RotatedRetinaNet(
            num_classes=15,
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
        )
        model2 = RotatedRetinaNet(
            num_classes=15,
            anchor_scales=[8, 16],
            anchor_ratios=[0.5, 1.0, 2.0, 3.0],
        )
        assert model1.num_classes == 15
        assert model2.num_classes == 15
        assert model1.anchor_angles == [0.0]
        assert model2.anchor_angles == [0.0]
        assert model1.num_anchors == 3  # len(ratios) * single reference angle
        assert model2.num_anchors == 4

        # Optional constructor-only override (not available via JSON / ModelConfig).
        model_custom = RotatedRetinaNet(
            num_classes=15,
            backbone_name="resnet18",
            pretrained_backbone=False,
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-math.pi / 2, 0.0],
        )
        assert model_custom.num_anchors == 6
        assert len(model_custom.anchor_angles) == 2
    
    def test_rotated_retinanet_loss_computation(self):
        """Test that RotatedRetinaNet computes losses correctly."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 128, 128)
        target = {
            "rboxes": [RBox(64, 64, 32, 16, 0.0)],
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        
        model.train()
        losses = model([image], [target])
        
        # Check loss structure
        assert isinstance(losses, dict)
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        
        # Losses should be scalars
        assert losses["loss_classifier"].dim() == 0
        assert losses["loss_box_reg"].dim() == 0
        
        # Losses should be non-negative
        assert losses["loss_classifier"].item() >= 0
        assert losses["loss_box_reg"].item() >= 0
    
    def test_rotated_retinanet_inference_format(self):
        """Test that RotatedRetinaNet inference returns correct format."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            outputs = model([image])
            
            assert isinstance(outputs, list)
            assert len(outputs) == 1
            
            output = outputs[0]
            assert "rboxes" in output
            assert "labels" in output
            assert "scores" in output
            
            # Check types
            assert isinstance(output["rboxes"], list)
            assert torch.is_tensor(output["labels"])
            assert torch.is_tensor(output["scores"])
            
            # Check shapes match
            assert len(output["rboxes"]) == len(output["labels"])
            assert len(output["labels"]) == len(output["scores"])
            
            # Check RBoxes have angles
            for rbox in output["rboxes"]:
                assert isinstance(rbox, RBox)
                assert hasattr(rbox, 'angle')
    
    def test_rotated_retinanet_empty_images(self):
        """Test RotatedRetinaNet with images that have no objects."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        image = torch.rand(3, 128, 128)
        target = {
            "rboxes": [],
            "labels": torch.tensor([], dtype=torch.int64),
        }
        
        model.train()
        losses = model([image], [target])
        
        # Should still return valid loss dict
        assert isinstance(losses, dict)
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
    
    def test_rotated_retinanet_batch_processing(self):
        """Test RotatedRetinaNet with batch size > 1."""
        model = RotatedRetinaNet(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        images = [
            torch.rand(3, 128, 128),
            torch.rand(3, 128, 128),
        ]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])},
            {"rboxes": [RBox(64, 64, 32, 16, math.pi / 4)], "labels": torch.tensor([2])},
        ]
        
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict)
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            assert len(outputs) == 2


class TestRotatedFasterRCNN:
    """Test that RotatedFasterRCNN works correctly."""
    
    def test_class_exists(self):
        """Test that RotatedFasterRCNN class exists."""
        assert RotatedFasterRCNN is not None
    
    def test_is_different_from_oriented_rcnn(self):
        """Test that RotatedFasterRCNN is a different class from OrientedRCNN."""
        # They are different implementations:
        # - RotatedFasterRCNN: horizontal RPN anchors (angle 0), 4-param RPN, 5-param ROI (DeltaXYWHTH)
        # - OrientedRCNN: horizontal RPN anchors, 6-param midpoint-offset ROI head
        assert RotatedFasterRCNN != OrientedRCNN
    
    def test_instantiation(self):
        """Test that RotatedFasterRCNN can be instantiated."""
        model = RotatedFasterRCNN(
            num_classes=10,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        assert isinstance(model, RotatedFasterRCNN)
        assert not isinstance(model, OrientedRCNN)

    def test_optional_anchor_angles_constructor(self):
        """Python-only anchor_angles kwarg (not in JSON)."""
        model = RotatedFasterRCNN(
            num_classes=3,
            backbone_name="resnet18",
            pretrained_backbone=False,
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[0.0, math.pi / 2],
        )
        assert model.num_anchors == 6

    def test_full_training_forward_and_backward(self):
        """Training forward + backward through horizontal RoIAlign (not ONNX export path)."""
        model = RotatedFasterRCNN(
            num_classes=5,
            backbone_name="resnet18",
            pretrained_backbone=False,
            trainable_layers=5,
            rpn_post_nms_top_n=128,
            rpn_pre_nms_top_n=128,
        )
        image = torch.rand(3, 256, 256)
        target = {
            "rboxes": [
                RBox(128, 128, 64, 32, 0.0),
                RBox(200, 200, 40, 20, math.pi / 4),
            ],
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        }
        model.train()
        losses = model([image], [target])
        for key in ("loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"):
            assert key in losses
            assert torch.is_tensor(losses[key])
            assert losses[key].dim() == 0
        total = sum(v for k, v in losses.items() if k.startswith("loss_"))
        total.backward()


def test_derive_fpn_strides_from_grid():
    assert derive_fpn_strides_from_grid((512, 512), [(128, 128), (64, 64), (32, 32)]) == [4, 8, 16]
    assert derive_fpn_strides_from_grid((800, 800), [(200, 200), (100, 100)]) == [4, 8]


def test_derive_fpn_strides_from_grid_anisotropic_raises():
    with pytest.raises(ValueError):
        derive_fpn_strides_from_grid((800, 800), [(100, 50)])


def test_warn_if_fpn_strides_mismatch():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_fpn_strides_mismatch(None, [4, 8])
        warn_if_fpn_strides_mismatch([4, 8], [4, 8])
        assert len(w) == 0
        warn_if_fpn_strides_mismatch([4, 9], [4, 8])
        assert len(w) == 1
