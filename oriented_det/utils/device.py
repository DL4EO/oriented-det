"""Device selection helpers for CPU, CUDA, and MPS (Apple Silicon)."""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def get_device(prefer_cuda: bool = True) -> "torch.device":
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU.

    Use this when you want training or inference to use GPU when possible,
    including on macOS with M1/M2/M3 (MPS).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for device selection.")
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_gpu_device(device: "torch.device") -> bool:
    """Return True if the device is a GPU (CUDA or MPS)."""
    if device is None:
        return False
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if device_type == "cuda":
        return torch is not None and torch.cuda.is_available()
    if device_type == "mps":
        return torch is not None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return False
