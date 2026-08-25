"""OrientedDet: rotated object detection for aerial imagery.

Exports geometry (``RBox``, ``QBox``, ``Polygon``), ops (``iou``, ``nms``), data loaders,
and detectors (``OrientedRCNN``, ``RotatedFasterRCNN``, ``RotatedRetinaNet``, ``RotatedFCOS``).
Training configs: ``oriented_det.train.config.TrainingExperimentConfig``.
"""

from __future__ import annotations

from .geometry import Polygon, QBox, RBox, transforms as geometry_transforms

# Alias for convenience (geometry_transforms is the canonical name)
transforms = geometry_transforms

__all__ = [
    "Polygon",
    "QBox",
    "RBox",
    "transforms",
    "geometry_transforms",
    "iou",
    "nms",
    "DOTADataset",
    "DOTASample",
    "DOTAAnnotation",
    "build_dota_loader",
    "ImageTiler",
    "compute_oriented_map",
    "OrientedRCNN",
    "RotatedFasterRCNN",
    "RotatedRetinaNet",
    "RotatedFCOS",
    "DeltaXYWHBBoxCoder",   # 4 params, RPN (MMRotate Rotated Faster R-CNN)
    "DeltaXYWHAHBBoxCoder", # 5 params, ROI (MMRotate)
    "DistanceAnglePointCoder",  # 5 params, FCOS (MMRotate)
    "MidpointOffsetCoder",  # 6 params, RPN (MMRotate Oriented R-CNN)
]


_LAZY_ATTRS = {
    # ops
    "iou": (".ops", "iou"),
    "nms": (".ops", "nms"),
    # data
    "DOTADataset": (".data", "DOTADataset"),
    "DOTASample": (".data", "DOTASample"),
    "DOTAAnnotation": (".data", "DOTAAnnotation"),
    "build_dota_loader": (".data", "build_dota_loader"),
    "ImageTiler": (".data", "ImageTiler"),
    "compute_oriented_map": (".data", "compute_oriented_map"),
    # models
    "OrientedRCNN": (".models", "OrientedRCNN"),
    "RotatedFasterRCNN": (".models", "RotatedFasterRCNN"),
    "RotatedRetinaNet": (".models", "RotatedRetinaNet"),
    "RotatedFCOS": (".models", "RotatedFCOS"),
    "DeltaXYWHBBoxCoder": (".models", "DeltaXYWHBBoxCoder"),
    "DeltaXYWHAHBBoxCoder": (".models", "DeltaXYWHAHBBoxCoder"),
    "DistanceAnglePointCoder": (".models", "DistanceAnglePointCoder"),
    "MidpointOffsetCoder": (".models", "MidpointOffsetCoder"),
}


def __getattr__(name: str):
    """Lazy import torch-heavy modules on first access.

    This keeps lightweight utilities (e.g. `oriented_det.pretrained`) usable in
    environments where importing PyTorch/torchvision kernels is undesirable.
    """
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _LAZY_ATTRS[name]
    import importlib

    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
