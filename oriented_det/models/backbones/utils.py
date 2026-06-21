"""Utility helpers for backbone modules."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Iterator, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for model backbones. Please install torch>=2.1.")


def freeze_layers(module: "nn.Module", trainable_layers: Iterable[str] | int) -> None:
    """Freeze parameters except the specified layers.

    Args:
        module: Backbone module.
        trainable_layers: Either an integer representing how many trailing child
            modules remain trainable, or an iterable of child names to keep.
    """
    require_torch()
    children = list(module.named_children())

    if isinstance(trainable_layers, int):
        if trainable_layers < 0:
            trainable_layers = 0
        keep = {name for name, _ in children[-trainable_layers:]}
    else:
        keep = set(trainable_layers)

    for name, child in children:
        requires_grad = name in keep
        for param in child.parameters():
            param.requires_grad = requires_grad


def count_trainable_parameters(module: "nn.Module") -> int:
    require_torch()
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_single_feature_backbone(out_channels: int = 256) -> "nn.Module":
    """Create a lightweight CNN backbone for environments without torchvision."""
    require_torch()

    class TinyBackbone(nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.out_channels = out_channels

        def forward(self, x: "torch.Tensor") -> "OrderedDict[str, torch.Tensor]":
            return OrderedDict({"0": self.body(x)})

    return TinyBackbone(out_channels)


__all__ = [
    "freeze_layers",
    "count_trainable_parameters",
    "make_single_feature_backbone",
    "require_torch",
]
