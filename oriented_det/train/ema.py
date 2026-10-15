"""Exponential-moving-average weights for validation and best checkpoints.

Momentum at update ``t`` (0-based, before the increment) is

    (1 - m) * exp(-(1 + t) / gamma) + m

Early updates put almost all of the weight on the current parameters. Later
updates use ``m`` (default 0.0002). Floating parameters and floating buffers
are averaged. Integer buffers such as BatchNorm ``num_batches_tracked`` are left
alone.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def exp_momentum(step: int, momentum: float = 0.0002, gamma: float = 2000.0) -> float:
    """EMA mix coefficient on the live weights at update ``step``."""
    m = float(momentum)
    return (1.0 - m) * math.exp(-(1.0 + int(step)) / float(gamma)) + m


def _unwrap(model: "nn.Module") -> "nn.Module":
    return model.module if hasattr(model, "module") else model


def _floating_tensors(module: "nn.Module"):
    for name, tensor in module.named_parameters():
        if tensor.dtype.is_floating_point:
            yield name, tensor
    for name, tensor in module.named_buffers():
        if tensor.dtype.is_floating_point:
            yield name, tensor


class ExpMomentumEMA:
    """Shadow copy of floating parameters and buffers."""

    def __init__(
        self,
        model: "nn.Module",
        momentum: float = 0.0002,
        gamma: float = 2000.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for ExpMomentumEMA.")
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.step = 0
        module = _unwrap(model)
        self.shadow: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in _floating_tensors(module)
        }

    def update(self, model: "nn.Module") -> float:
        """Mix live weights into the shadow. Returns the coefficient used."""
        coefficient = exp_momentum(self.step, self.momentum, self.gamma)
        module = _unwrap(model)
        with torch.no_grad():
            for name, tensor in _floating_tensors(module):
                shadow = self.shadow.get(name)
                if shadow is None or shadow.shape != tensor.shape:
                    continue
                shadow.lerp_(tensor.detach(), coefficient)
        self.step += 1
        return coefficient

    def apply_to_model(self, model: "nn.Module") -> Dict[str, torch.Tensor]:
        """Copy shadow weights into ``model``. Returns a backup of the live tensors."""
        module = _unwrap(model)
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, tensor in _floating_tensors(module):
                shadow = self.shadow.get(name)
                if shadow is None or shadow.shape != tensor.shape:
                    continue
                backup[name] = tensor.detach().clone()
                tensor.copy_(shadow)
        return backup

    def restore_model(self, model: "nn.Module", backup: Optional[Dict[str, torch.Tensor]]) -> None:
        """Put training weights back after :meth:`apply_to_model`."""
        if not backup:
            return
        module = _unwrap(model)
        with torch.no_grad():
            for name, tensor in _floating_tensors(module):
                saved = backup.get(name)
                if saved is None or saved.shape != tensor.shape:
                    continue
                tensor.copy_(saved)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "momentum": float(self.momentum),
            "gamma": float(self.gamma),
            "shadow": {name: tensor.detach().clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step = int(state.get("step", 0))
        if "momentum" in state:
            self.momentum = float(state["momentum"])
        if "gamma" in state:
            self.gamma = float(state["gamma"])
        saved = state.get("shadow") or {}
        for name, tensor in self.shadow.items():
            src = saved.get(name)
            if src is None or tuple(src.shape) != tuple(tensor.shape):
                continue
            tensor.copy_(src.to(device=tensor.device, dtype=tensor.dtype))


__all__ = [
    "ExpMomentumEMA",
    "exp_momentum",
]
