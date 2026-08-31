"""Model registry for oriented detection baselines."""

from .oriented_rcnn import RotatedFasterRCNN, OrientedRCNN
from .rotated_retinanet import RotatedRetinaNet
from .rotated_fcos import RotatedFCOS
from .backbones import build_resnet_fpn_backbone
from .bbox_coder import (
    DeltaXYWHBBoxCoder,
    DeltaXYWHAHBBoxCoder,
    DistanceAnglePointCoder,
    MidpointOffsetCoder,
)
from .utils import (
    rboxes_to_tensor,
    tensor_to_rboxes,
    prepare_targets,
    force_lookalike_to_background,
    setup_backbone,
    extract_backbone_features,
    setup_anchors,
    ClassWeightsMixin,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
)
__all__ = [
    "RotatedFasterRCNN",
    "OrientedRCNN",
    "RotatedRetinaNet",
    "RotatedFCOS",
    "build_resnet_fpn_backbone",
    "DeltaXYWHBBoxCoder",
    "DeltaXYWHAHBBoxCoder",
    "DistanceAnglePointCoder",
    "MidpointOffsetCoder",
    # Shared utilities
    "rboxes_to_tensor",
    "tensor_to_rboxes",
    "prepare_targets",
    "force_lookalike_to_background",
    "setup_backbone",
    "extract_backbone_features",
    "setup_anchors",
    "ClassWeightsMixin",
    "derive_fpn_strides_from_grid",
    "warn_if_fpn_strides_mismatch",
]
