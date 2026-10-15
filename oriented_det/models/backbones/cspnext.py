"""CSPNeXt backbone + CSPNeXt-PAFPN neck for RTMDet-R (clean-room).

Architecture follows the public RTMDet / CSPNeXt design (P5 stages, path-
aggregation FPN). No MMDetection / MMCV runtime dependency.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# P5 arch: (in_ch, out_ch, num_blocks, add_identity, use_spp)
_ARCH_P5: List[Tuple[int, int, int, bool, bool]] = [
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, True, True),
]

# Named presets: (deepen, widen, pafpn_out_channels)
CSPNEXT_PRESETS: Dict[str, Tuple[float, float, int]] = {
    "cspnext_tiny": (0.167, 0.375, 96),
    "cspnext_s": (0.33, 0.5, 128),
    "cspnext_m": (0.67, 0.75, 192),
    "cspnext_l": (1.0, 1.0, 256),
}

# Optional ImageNet CSPNeXt checkpoints (Apache-2.0 OpenMMLab releases).
_IMAGENET_URLS: Dict[str, str] = {
    "cspnext_tiny": (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth"
    ),
    "cspnext_s": (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"
    ),
}


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for CSPNeXt.")


def _make_norm(num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(num_features, eps=1e-3, momentum=0.03)


class ConvBNAct(nn.Module):
    """Conv2d + BN + SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        _require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = _make_norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        _require_torch()
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels, in_channels, kernel_size, stride=stride, groups=in_channels
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ChannelAttention(nn.Module):
    """Simple channel attention (global pool → 1×1 → sigmoid gate)."""

    def __init__(self, channels: int):
        _require_torch()
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AdaptiveAvgPool2d(1) then gate
        y = x.mean(dim=(2, 3), keepdim=True)
        y = torch.sigmoid(self.fc(y))
        return x * y


class CSPNeXtBlock(nn.Module):
    """3×3 conv + depthwise 5×5 residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        add_identity: bool = True,
        expansion: float = 0.5,
        kernel_size: int = 5,
    ):
        _require_torch()
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden, kernel_size=3, stride=1)
        self.conv2 = DepthwiseSeparableConvBNAct(
            hidden, out_channels, kernel_size=kernel_size, stride=1
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.add_identity:
            return out + x
        return out


class CSPLayer(nn.Module):
    """Cross-stage partial layer with CSPNeXt blocks + optional channel attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        expand_ratio: float = 0.5,
        add_identity: bool = True,
        channel_attention: bool = True,
    ):
        _require_torch()
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.main_conv = ConvBNAct(in_channels, mid, kernel_size=1)
        self.short_conv = ConvBNAct(in_channels, mid, kernel_size=1)
        self.final_conv = ConvBNAct(2 * mid, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(mid, mid, add_identity=add_identity, expansion=1.0)
                for _ in range(max(1, int(num_blocks)))
            ]
        )
        self.attention = ChannelAttention(2 * mid) if channel_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x_final = torch.cat([x_main, x_short], dim=1)
        if self.attention is not None:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling bottleneck (kernels 5, 9, 13)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
    ):
        _require_torch()
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, mid, kernel_size=1)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes]
        )
        self.conv2 = ConvBNAct(mid * (len(kernel_sizes) + 1), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        parts = [x] + [pool(x) for pool in self.pools]
        return self.conv2(torch.cat(parts, dim=1))


class CSPNeXt(nn.Module):
    """CSPNeXt-P5 backbone. Outputs stages selected by ``out_indices`` (stem=0)."""

    def __init__(
        self,
        *,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
        frozen_stages: int = -1,
    ):
        _require_torch()
        super().__init__()
        self.out_indices = tuple(int(i) for i in out_indices)
        self.frozen_stages = int(frozen_stages)
        self.widen_factor = float(widen_factor)
        self.deepen_factor = float(deepen_factor)

        stem_ch = int(64 * widen_factor)
        stem_mid = stem_ch // 2
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_mid, kernel_size=3, stride=2),
            ConvBNAct(stem_mid, stem_mid, kernel_size=3, stride=1),
            ConvBNAct(stem_mid, stem_ch, kernel_size=3, stride=1),
        )
        self.layers = ["stem"]
        self.out_channels_list: List[int] = []

        for i, (in_c, out_c, n_blocks, add_id, use_spp) in enumerate(_ARCH_P5):
            in_c = int(in_c * widen_factor)
            out_c = int(out_c * widen_factor)
            n_blocks = max(round(n_blocks * deepen_factor), 1)
            stage: List[nn.Module] = [
                ConvBNAct(in_c, out_c, kernel_size=3, stride=2),
            ]
            if use_spp:
                stage.append(SPPBottleneck(out_c, out_c))
            stage.append(
                CSPLayer(
                    out_c,
                    out_c,
                    num_blocks=n_blocks,
                    expand_ratio=expand_ratio,
                    add_identity=add_id,
                    channel_attention=channel_attention,
                )
            )
            name = f"stage{i + 1}"
            self.add_module(name, nn.Sequential(*stage))
            self.layers.append(name)

        # Channel count of each layer output (stem + 4 stages)
        self._layer_channels = [stem_ch] + [
            int(row[1] * widen_factor) for row in _ARCH_P5
        ]
        for idx in self.out_indices:
            self.out_channels_list.append(self._layer_channels[idx])

    def _freeze_stages(self) -> None:
        if self.frozen_stages < 0:
            return
        for i in range(self.frozen_stages + 1):
            if i >= len(self.layers):
                break
            m = getattr(self, self.layers[i])
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class CSPNeXtPAFPN(nn.Module):
    """Path-aggregation FPN with CSPNeXt blocks (P3–P5 when given 3 inputs)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
    ):
        _require_torch()
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.out_channels = int(out_channels)
        n = len(self.in_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct(self.in_channels[idx], self.in_channels[idx - 1], kernel_size=1)
            )
            self.top_down_blocks.append(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expand_ratio,
                    add_identity=False,
                    channel_attention=False,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n - 1):
            self.downsamples.append(
                ConvBNAct(self.in_channels[idx], self.in_channels[idx], kernel_size=3, stride=2)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    self.in_channels[idx] * 2,
                    self.in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expand_ratio,
                    add_identity=False,
                    channel_attention=False,
                )
            )

        self.out_convs = nn.ModuleList(
            [
                ConvBNAct(self.in_channels[i], out_channels, kernel_size=3)
                for i in range(n)
            ]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        n = len(self.in_channels)

        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            up = self.upsample(feat_high)
            # Match spatial size if pad makes odd dims
            low = inputs[idx - 1]
            if up.shape[-2:] != low.shape[-2:]:
                up = F.interpolate(up, size=low.shape[-2:], mode="nearest")
            inner = self.top_down_blocks[n - 1 - idx](torch.cat([up, low], dim=1))
            inner_outs.insert(0, inner)

        outs = [inner_outs[0]]
        for idx in range(n - 1):
            down = self.downsamples[idx](outs[-1])
            high = inner_outs[idx + 1]
            if down.shape[-2:] != high.shape[-2:]:
                down = F.interpolate(down, size=high.shape[-2:], mode="nearest")
            outs.append(
                self.bottom_up_blocks[idx](torch.cat([down, high], dim=1))
            )

        return tuple(conv(o) for conv, o in zip(self.out_convs, outs))


def _load_cspnext_imagenet(backbone: CSPNeXt, preset: str) -> bool:
    """Try to load OpenMMLab ImageNet CSPNeXt weights into ``backbone`` only.

    Returns True if any weights were loaded. Failures are non-fatal.
    """
    url = _IMAGENET_URLS.get(preset)
    if url is None:
        return False
    try:
        from torch.hub import load_state_dict_from_url

        ckpt = load_state_dict_from_url(url, map_location="cpu", progress=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Failed to download CSPNeXt ImageNet weights for {preset!r} ({exc}). "
            "Using random initialization.",
            RuntimeWarning,
        )
        return False

    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        return False

    # Strip common prefixes. Drop the ImageNet classifier (``head.fc``).
    # OpenMMLab names the separable conv ``depthwise_conv`` / ``pointwise_conv``;
    # this module uses ``depthwise`` / ``pointwise``.
    remapped: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("backbone.", "module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        if nk.startswith("head."):
            continue
        nk = nk.replace(".depthwise_conv.", ".depthwise.")
        nk = nk.replace(".pointwise_conv.", ".pointwise.")
        remapped[nk] = v

    try:
        backbone.load_state_dict(remapped, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"CSPNeXt ImageNet checkpoint for {preset!r} does not match this "
            "backbone after dropping the classifier head and aliasing "
            "depthwise_conv/pointwise_conv to depthwise/pointwise.\n" + str(exc)
        ) from exc
    return True


class CSPNeXtWithPAFPN(nn.Module):
    """CSPNeXt + PAFPN returning FPN-like ``OrderedDict`` keys ``\"0\"``, ``\"1\"``, …"""

    def __init__(
        self,
        preset: str = "cspnext_tiny",
        *,
        pretrained: bool = False,
        deepen_factor: Optional[float] = None,
        widen_factor: Optional[float] = None,
        out_channels: Optional[int] = None,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        expand_ratio: float = 0.5,
    ):
        _require_torch()
        super().__init__()
        key = preset.lower().replace("-", "_")
        if key not in CSPNEXT_PRESETS and deepen_factor is None:
            raise ValueError(
                f"Unknown CSPNeXt preset {preset!r}; "
                f"known: {sorted(CSPNEXT_PRESETS)} or pass deepen/widen explicitly."
            )
        d, w, oc = CSPNEXT_PRESETS.get(key, (1.0, 1.0, 256))
        deepen = float(deepen_factor if deepen_factor is not None else d)
        widen = float(widen_factor if widen_factor is not None else w)
        neck_out = int(out_channels if out_channels is not None else oc)

        self.preset = key
        self.body = CSPNeXt(
            deepen_factor=deepen,
            widen_factor=widen,
            out_indices=out_indices,
            expand_ratio=expand_ratio,
            frozen_stages=frozen_stages,
        )
        in_chs = list(self.body.out_channels_list)
        num_csp = max(round(3 * deepen), 1)
        self.neck = CSPNeXtPAFPN(
            in_chs,
            neck_out,
            num_csp_blocks=num_csp,
            expand_ratio=expand_ratio,
        )
        self.out_channels = neck_out
        self.fpn_strides = [8, 16, 32]  # P3–P5 for out_indices (2,3,4)

        if pretrained:
            _load_cspnext_imagenet(self.body, key)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feats = self.body(x)
        outs = self.neck(feats)
        return OrderedDict((str(i), t) for i, t in enumerate(outs))


def build_cspnext_pafpn_backbone(
    backbone_name: str = "cspnext_tiny",
    *,
    pretrained: bool = False,
    frozen_stages: int = -1,
    out_channels: Optional[int] = None,
) -> CSPNeXtWithPAFPN:
    """Factory matching ResNet-FPN builder style."""
    return CSPNeXtWithPAFPN(
        preset=backbone_name,
        pretrained=pretrained,
        frozen_stages=frozen_stages,
        out_channels=out_channels,
    )


__all__ = [
    "CSPNeXt",
    "CSPNeXtPAFPN",
    "CSPNeXtWithPAFPN",
    "CSPNEXT_PRESETS",
    "build_cspnext_pafpn_backbone",
    "ConvBNAct",
    "CSPLayer",
    "CSPNeXtBlock",
]
