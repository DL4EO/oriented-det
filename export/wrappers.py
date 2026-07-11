"""ONNX-friendly ``nn.Module`` wrappers around oriented-det PyTorch models.

Export targets **tensor-in / tensor-out** subgraphs. Post-decode NMS and RBox
assembly stay in Python or a separate TF/TFLite graph (see export/PARITY.md).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from oriented_det.models.faster_rcnn_inference import faster_rcnn_inference_pre_nms_padded
from oriented_det.models.oriented_rcnn import RotatedFasterRCNN
from oriented_det.models.oriented_rpn import generate_oriented_anchors
from oriented_det.models.rotated_retinanet import RotatedRetinaNet
from oriented_det.models.utils import derive_fpn_strides_from_grid, extract_backbone_features


def _ordered_fpn_values(features: object) -> Tuple[torch.Tensor, ...]:
    """Turn backbone output into a tuple of tensors in a stable order."""
    if isinstance(features, dict):
        keys = []
        for k in sorted(features.keys(), key=lambda x: str(x)):
            ks = str(k)
            if ks.isdigit() or ks.startswith("fpn"):
                keys.append(k)
        if not keys:
            keys = list(features.keys())
        return tuple(features[k] for k in keys)
    if isinstance(features, (list, tuple)):
        return tuple(features)
    return (features,)  # type: ignore[return-value]


class BackboneExportWrapper(nn.Module):
    """Exports ``backbone(images)`` as a flat tuple of FPN tensors.

    Works for any model that exposes a ``backbone`` attribute (ResNet+FPN).

    Args:
        backbone: Module accepting ``[B, 3, H, W]`` float tensor.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats = self.backbone(images)
        return _ordered_fpn_values(feats)


class RetinaNetBackboneHeadExportWrapper(nn.Module):
    """Exports Rotated RetinaNet **backbone + classification/regression head** only.

    Outputs alternate per FPN level: ``cls_lvl0``, ``bbox_lvl0``, ``cls_lvl1``, …
    Classification tensors are sigmoid logits ``[B, A*K, H, W]``; bbox tensors are
    ``[B, A*5, H, W]`` in model order. Decoding, anchor generation, thresholding,
    and oriented NMS are **not** included (run in PyTorch inference or downstream TF).
    """

    def __init__(self, model: RotatedRetinaNet) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.fpn_extra_level = model.fpn_extra_level
        self.head = model.head

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # List of (C, H, W) — use unbind so batch size stays symbolic for ONNX.
        img_list: Sequence[torch.Tensor] = torch.unbind(images, dim=0)
        feat_list = extract_backbone_features(
            self.backbone,
            img_list,
            use_checkpoint=False,
            training=False,
            include_pool_level=not self.fpn_extra_level,
        )
        cls_list, bbox_list = self.head(feat_list)
        out: List[torch.Tensor] = []
        for c, b in zip(cls_list, bbox_list):
            out.append(c)
            out.append(b)
        return tuple(out)


class RotatedFasterRCNNPreNmsExportWrapper(nn.Module):
    """Rotated Faster R-CNN through ROI decode; outputs padded pre-NMS tensors.

    Input: ``images`` ``[1, 3, H, W]`` float32 in [0, 1] RGB.
    Outputs: ``pre_nms_boxes``, ``pre_nms_scores``, ``pre_nms_labels``, ``pre_nms_count``.
    Final rotated NMS and production score filters run in the TF SavedModel wrapper.
    """

    def __init__(
        self,
        model: RotatedFasterRCNN,
        height: int,
        width: int,
        max_candidates: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.height = int(height)
        self.width = int(width)
        self.max_candidates = int(max_candidates or model.rpn_post_nms_top_n)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.height, self.width, dtype=torch.float32)
            feat_list = extract_backbone_features(
                model.backbone,
                [dummy[0]],
                use_checkpoint=False,
                training=False,
                include_pool_level=True,  # P6 for the RPN, matching RotatedFasterRCNN.forward
            )
            feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feat_list]
            fpn_strides_live = derive_fpn_strides_from_grid(
                (self.height, self.width), feature_map_sizes
            )
            anchors = generate_oriented_anchors(
                image_size=(self.height, self.width),
                feature_map_sizes=feature_map_sizes,
                anchor_scales=model.anchor_scales,
                anchor_ratios=model.anchor_ratios,
                anchor_angles=model.anchor_angles,
                stride_per_level=fpn_strides_live,
            )

        self._num_anchor_levels = len(anchors)
        for i, anchor_tensor in enumerate(anchors):
            self.register_buffer(f"_anchor_{i}", anchor_tensor)
        self._fpn_strides_list = [int(s) for s in fpn_strides_live]

    def _anchor_list(self) -> List[torch.Tensor]:
        return [getattr(self, f"_anchor_{i}") for i in range(self._num_anchor_levels)]

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return faster_rcnn_inference_pre_nms_padded(
            self.model,
            images,
            self.max_candidates,
            anchors=self._anchor_list(),
            fpn_strides_live=self._fpn_strides_list,
            deterministic_rpn=True,
        )


__all__ = [
    "BackboneExportWrapper",
    "RetinaNetBackboneHeadExportWrapper",
    "RotatedFasterRCNNPreNmsExportWrapper",
    "_ordered_fpn_values",
]
