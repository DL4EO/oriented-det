"""ResNet + FPN backbone factory with graceful fallbacks."""

from __future__ import annotations

from typing import Iterable, List, Optional
import inspect

try:
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone as tv_resnet_fpn_backbone
except Exception:  # pragma: no cover
    tv_resnet_fpn_backbone = None  # type: ignore

from .utils import make_single_feature_backbone, require_torch


def build_resnet_fpn_backbone(
    backbone_name: str = "resnet50",
    *,
    pretrained: bool = False,
    trainable_layers: int = 3,
    norm_layer=None,
    returned_layers: Optional[List[int]] = None,
    use_p6p7_extra_levels: bool = False,
) -> object:
    """Create a ResNet+FPN backbone or a minimal fallback if torchvision is missing.

    Args:
        backbone_name: ResNet variant ("resnet50", etc.).
        pretrained: Whether to load ImageNet backbone weights.
        trainable_layers: Number of backbone stages to train.
        norm_layer: Norm layer override. Default None keeps torchvision's
            ``FrozenBatchNorm2d`` (frozen BN statistics), matching MMRotate's
            ``norm_eval=True`` detection recipe. Pass ``torch.nn.BatchNorm2d``
            explicitly to train with live batch statistics.
        returned_layers: ResNet stage indices to use for FPN (1–4). Default None
            uses torchvision default [1,2,3,4] (C2–C5, 5 levels with P6).
            Use [2,3,4] for MMRotate-style FPN (C3–C5 only, 4 levels with P6).
        use_p6p7_extra_levels: If True, attach ``LastLevelP6P7`` (MMRotate RetinaNet
            ``add_extra_convs='on_input'``) instead of ``LastLevelMaxPool``.
    """
    require_torch()
    if tv_resnet_fpn_backbone is None:
        return make_single_feature_backbone()

    # Do not pass norm_layer=None through: torchvision would then fall back to live
    # BatchNorm2d. Omitting it keeps torchvision's FrozenBatchNorm2d default.
    kwargs = dict(trainable_layers=trainable_layers)
    if norm_layer is not None:
        kwargs["norm_layer"] = norm_layer
    signature = inspect.signature(tv_resnet_fpn_backbone)
    if "weights" in signature.parameters:
        kwargs["weights"] = None if not pretrained else "DEFAULT"
    if "weights_backbone" in signature.parameters:
        kwargs["weights_backbone"] = "DEFAULT" if pretrained else None
    if "returned_layers" in signature.parameters and returned_layers is not None:
        kwargs["returned_layers"] = returned_layers
    if use_p6p7_extra_levels and "extra_blocks" in signature.parameters:
        from torchvision.ops.feature_pyramid_network import LastLevelP6P7

        c5_channels = 512 if any(x in backbone_name.lower() for x in ("18", "34")) else 2048
        kwargs["extra_blocks"] = LastLevelP6P7(c5_channels, 256)

    try:
        backbone = tv_resnet_fpn_backbone(backbone_name=backbone_name, **kwargs)
        return backbone
    except Exception as exc:  # pragma: no cover
        # Offline / restricted environments (CI, sandboxes) may block weight downloads.
        # Fall back to random init while still returning a functional backbone.
        if pretrained:
            import warnings

            warnings.warn(
                f"Failed to load pretrained backbone weights for {backbone_name!r} ({exc}). "
                "Falling back to random initialization.",
                RuntimeWarning,
            )
            kwargs2 = dict(kwargs)
            if "weights" in kwargs2:
                kwargs2["weights"] = None
            if "weights_backbone" in kwargs2:
                kwargs2["weights_backbone"] = None
            backbone = tv_resnet_fpn_backbone(backbone_name=backbone_name, **kwargs2)
            return backbone
        raise


__all__ = ["build_resnet_fpn_backbone"]
