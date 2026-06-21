"""ONNX Runtime device selection and session cache for TF export inference."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

_ORT_DEVICE_OVERRIDE: Optional[str] = None
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], object] = {}


def set_ort_device(device: Optional[str]) -> None:
    """Set runtime ORT device for this process (``cpu``, ``cuda``, ``auto``)."""
    global _ORT_DEVICE_OVERRIDE
    _ORT_DEVICE_OVERRIDE = device.lower().strip() if device else None


def get_ort_device() -> str:
    """Resolved ORT device string (override, env, or default ``cpu``)."""
    if _ORT_DEVICE_OVERRIDE:
        return _ORT_DEVICE_OVERRIDE
    return (os.environ.get("ORIENTED_DET_ORT_DEVICE") or "cpu").lower().strip()


def ort_providers_for_device(device: Optional[str] = None) -> List[str]:
    """Map device string to ONNX Runtime ``providers`` list."""
    import onnxruntime as ort

    d = (device if device is not None else get_ort_device()).lower().strip()
    if d in ("cuda", "gpu"):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ORT device=cuda requested but CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu matching your CUDA driver."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if d == "auto":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if d != "cpu":
        raise ValueError(f"Unknown ORT device {device!r}; use cpu, cuda, or auto.")
    return ["CPUExecutionProvider"]


def configure_ort_device(device: Optional[str]) -> List[str]:
    """Apply device override and return the provider list that will be used."""
    if device is not None:
        set_ort_device(device)
    providers = ort_providers_for_device()
    return providers


def clear_ort_session_cache() -> None:
    """Drop cached ORT sessions (tests)."""
    _SESSION_CACHE.clear()


def get_ort_session(onnx_path: str, device: Optional[str] = None):
    """Return a cached ``onnxruntime.InferenceSession`` for ``onnx_path``."""
    import onnxruntime as ort

    providers = ort_providers_for_device(device)
    key = (str(onnx_path), tuple(providers))
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = ort.InferenceSession(onnx_path, providers=providers)
    return _SESSION_CACHE[key]
