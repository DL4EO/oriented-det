"""C2f CSP backbone + PAFPN for YOLO-OBB (clean-room).

Published YOLOv8-style stages: stride-2 Conv stem, C2f blocks, SPPF, then a
path-aggregation neck. Presets ``yolo_obb_n`` and ``yolo_obb_s``. No pretrained
weights (no Apache backbone checkpoint to load).
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


# depth, width, max_channels
YOLO_OBB_PRESETS: Dict[str, Tuple[float, float, int]] = {
    "yolo_obb_n": (0.33, 0.25, 1024),
    "yolo_obb_s": (0.33, 0.50, 1024),
}


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for YOLO-OBB.")


def _make_divisible(value: float, divisor: int = 8) -> int:
    return int(math.ceil(float(value) / divisor) * divisor)


def _channels(base: int, width: float, max_channels: int) -> int:
    return min(_make_divisible(base * width), int(max_channels))


def _depth(repeats: int, depth: float) -> int:
    return max(round(repeats * depth), 1)


def _make_norm(num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(num_features, eps=1e-3, momentum=0.03)


class Conv(nn.Module):
    """Conv2d + BN + SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        _require_torch()
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = _make_norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two 3×3 convs. Residual add when ``shortcut`` and channels match."""

    def __init__(self, channels: int, *, shortcut: bool = True):
        _require_torch()
        super().__init__()
        self.cv1 = Conv(channels, channels, 3, 1)
        self.cv2 = Conv(channels, channels, 3, 1)
        self.add = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.add:
            return out + x
        return out


class C2f(nn.Module):
    """Cross-stage partial block: split, stack bottlenecks, concat."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        *,
        shortcut: bool = False,
        expansion: float = 0.5,
    ):
        _require_torch()
        super().__init__()
        hidden = max(int(out_channels * expansion), 1)
        n = max(int(num_blocks), 1)
        self.cv1 = Conv(in_channels, 2 * hidden, kernel_size=1, stride=1)
        self.cv2 = Conv((2 + n) * hidden, out_channels, kernel_size=1, stride=1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, shortcut=shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast: three successive k=5 pools."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        _require_torch()
        super().__init__()
        hidden = max(in_channels // 2, 1)
        self.cv1 = Conv(in_channels, hidden, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden * 4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.pool(y2)], dim=1))


class YoloOBBBackbone(nn.Module):
    """P5 C2f backbone. Returns P3, P4, P5 (strides 8, 16, 32)."""

    def __init__(
        self,
        *,
        depth: float = 0.33,
        width: float = 0.50,
        max_channels: int = 1024,
        frozen_stages: int = -1,
    ):
        _require_torch()
        super().__init__()
        self.depth = float(depth)
        self.width = float(width)
        self.max_channels = int(max_channels)
        self.frozen_stages = int(frozen_stages)

        def ch(base: int) -> int:
            return _channels(base, self.width, self.max_channels)

        def n(repeats: int) -> int:
            return _depth(repeats, self.depth)

        c64, c128, c256, c512, c1024 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        self.stem = Conv(3, c64, 3, 2)
        self.stage2 = nn.Sequential(Conv(c64, c128, 3, 2), C2f(c128, c128, n(3), shortcut=True))
        self.stage3 = nn.Sequential(Conv(c128, c256, 3, 2), C2f(c256, c256, n(6), shortcut=True))
        self.stage4 = nn.Sequential(Conv(c256, c512, 3, 2), C2f(c512, c512, n(6), shortcut=True))
        self.stage5 = nn.Sequential(
            Conv(c512, c1024, 3, 2),
            C2f(c1024, c1024, n(3), shortcut=True),
            SPPF(c1024, c1024),
        )
        self.out_channels_list = [c256, c512, c1024]
        self._freeze()

    def _freeze(self) -> None:
        if self.frozen_stages < 0:
            return
        stages = (self.stem, self.stage2, self.stage3, self.stage4, self.stage5)
        for idx, stage in enumerate(stages):
            if idx <= self.frozen_stages:
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class YoloOBBPAFPN(nn.Module):
    """Top-down then bottom-up C2f neck. Outputs P3, P4, P5 channels."""

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        depth: float,
        width: float,
        max_channels: int,
    ):
        _require_torch()
        super().__init__()
        c3, c4, c5 = (int(c) for c in in_channels)
        n = _depth(3, depth)
        td4 = _channels(512, width, max_channels)
        out3 = _channels(256, width, max_channels)
        out4 = _channels(512, width, max_channels)
        out5 = _channels(1024, width, max_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_td4 = C2f(c5 + c4, td4, n, shortcut=False)
        self.c2f_out3 = C2f(td4 + c3, out3, n, shortcut=False)
        self.down4 = Conv(out3, out3, 3, 2)
        self.c2f_out4 = C2f(out3 + td4, out4, n, shortcut=False)
        self.down5 = Conv(out4, out4, 3, 2)
        self.c2f_out5 = C2f(out4 + c5, out5, n, shortcut=False)
        self.out_channels_list = [out3, out4, out5]

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features
        up5 = self.upsample(p5)
        if up5.shape[-2:] != p4.shape[-2:]:
            up5 = nn.functional.interpolate(up5, size=p4.shape[-2:], mode="nearest")
        td4 = self.c2f_td4(torch.cat([up5, p4], dim=1))
        up4 = self.upsample(td4)
        if up4.shape[-2:] != p3.shape[-2:]:
            up4 = nn.functional.interpolate(up4, size=p3.shape[-2:], mode="nearest")
        out3 = self.c2f_out3(torch.cat([up4, p3], dim=1))
        down4 = self.down4(out3)
        if down4.shape[-2:] != td4.shape[-2:]:
            down4 = nn.functional.interpolate(down4, size=td4.shape[-2:], mode="nearest")
        out4 = self.c2f_out4(torch.cat([down4, td4], dim=1))
        down5 = self.down5(out4)
        if down5.shape[-2:] != p5.shape[-2:]:
            down5 = nn.functional.interpolate(down5, size=p5.shape[-2:], mode="nearest")
        out5 = self.c2f_out5(torch.cat([down5, p5], dim=1))
        return out3, out4, out5


class YoloOBBWithPAFPN(nn.Module):
    """Backbone + neck. Forward returns an OrderedDict with keys ``\"0\"``, ``\"1\"``, ``\"2\"``."""

    def __init__(
        self,
        preset: str = "yolo_obb_s",
        *,
        pretrained: bool = False,
        frozen_stages: int = -1,
    ):
        _require_torch()
        super().__init__()
        key = preset.lower().replace("-", "_")
        if key not in YOLO_OBB_PRESETS:
            raise ValueError(
                f"Unknown YOLO-OBB preset {preset!r}; known: {sorted(YOLO_OBB_PRESETS)}."
            )
        depth, width, max_channels = YOLO_OBB_PRESETS[key]
        if pretrained:
            warnings.warn(
                "YOLO-OBB has no pretrained backbone weights; using random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.preset = key
        self.body = YoloOBBBackbone(
            depth=depth,
            width=width,
            max_channels=max_channels,
            frozen_stages=frozen_stages,
        )
        self.neck = YoloOBBPAFPN(
            self.body.out_channels_list,
            depth=depth,
            width=width,
            max_channels=max_channels,
        )
        self.out_channels_list = list(self.neck.out_channels_list)
        self.fpn_strides = [8, 16, 32]

    def forward(self, x: torch.Tensor) -> OrderedDict:
        outs = self.neck(self.body(x))
        return OrderedDict((str(i), t) for i, t in enumerate(outs))


def build_yolo_obb_backbone(
    backbone_name: str = "yolo_obb_s",
    *,
    pretrained: bool = False,
    frozen_stages: int = -1,
) -> YoloOBBWithPAFPN:
    """Factory matching the CSPNeXt builder style."""
    return YoloOBBWithPAFPN(
        preset=backbone_name,
        pretrained=pretrained,
        frozen_stages=frozen_stages,
    )


__all__ = [
    "YOLO_OBB_PRESETS",
    "Conv",
    "C2f",
    "YoloOBBBackbone",
    "YoloOBBPAFPN",
    "YoloOBBWithPAFPN",
    "build_yolo_obb_backbone",
]
