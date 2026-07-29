"""ONNX Runtime device selection and session cache for TF export inference."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_ORT_DEVICE_OVERRIDE: Optional[str] = None
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], object] = {}
_TF_GPUS_HIDDEN: bool = False


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
                "Install onnxruntime-gpu matching your CUDA driver "
                '(e.g. pip install "oriented-det[export]" / onnxruntime-gpu[cuda,cudnn]).'
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if d == "auto":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if d != "cpu":
        raise ValueError(f"Unknown ORT device {device!r}; use cpu, cuda, or auto.")
    return ["CPUExecutionProvider"]


def hide_tensorflow_gpus() -> bool:
    """Hide all GPUs from TensorFlow so ORT can own CUDA memory.

    Call **before** TF creates CUDA contexts (ideally before ``import tensorflow``
    finishes initializing devices). When ORT uses ``CUDAExecutionProvider``, TF
    otherwise claims every visible GPU and ORT then OOMs on ROI Align temps
    (e.g. ``/roi_align/Transpose_2`` ~287 MB that should fit on an A100).

    Returns:
        True if GPUs are hidden from TF (or none were visible).
    """
    global _TF_GPUS_HIDDEN
    if _TF_GPUS_HIDDEN:
        return True
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        _TF_GPUS_HIDDEN = True
        return True
    except Exception as exc:  # pragma: no cover - depends on TF init order
        print(
            f"Warning: could not hide GPUs from TensorFlow ({exc}). "
            "ORT CUDA may OOM if TF already initialized CUDA. "
            "Configure ORT device before importing tensorflow / tf_serving_model."
        )
        return False


def prepare_tensorflow_for_ort(providers: Sequence[str]) -> None:
    """If ORT will use CUDA, keep TensorFlow on CPU only."""
    if "CUDAExecutionProvider" in providers:
        hide_tensorflow_gpus()


def _cuda_provider_options() -> Dict[str, Any]:
    """ORT CUDA EP options (device id, arena, optional mem cap)."""
    opts: Dict[str, Any] = {
        "device_id": int(os.environ.get("ORIENTED_DET_ORT_CUDA_DEVICE_ID", "0")),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": os.environ.get(
            "ORIENTED_DET_ORT_CUDNN_CONV_ALGO", "HEURISTIC"
        ),
    }
    mem = os.environ.get("ORIENTED_DET_ORT_GPU_MEM_LIMIT")
    if mem:
        opts["gpu_mem_limit"] = int(mem)
    return opts


def _session_providers(
    providers: Sequence[str],
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Expand provider names into ORT session provider list (with CUDA options)."""
    out: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            out.append((p, _cuda_provider_options()))
        else:
            out.append(p)
    return out


def configure_ort_device(device: Optional[str] = None) -> List[str]:
    """Apply device override, hide TF GPUs when needed, return provider names."""
    if device is not None:
        set_ort_device(device)
    providers = ort_providers_for_device()
    prepare_tensorflow_for_ort(providers)
    return providers


def clear_ort_session_cache() -> None:
    """Drop cached ORT sessions (tests)."""
    _SESSION_CACHE.clear()


def get_ort_session(onnx_path: str, device: Optional[str] = None):
    """Return a cached ``onnxruntime.InferenceSession`` for ``onnx_path``."""
    import onnxruntime as ort

    providers = ort_providers_for_device(device)
    prepare_tensorflow_for_ort(providers)
    session_providers = _session_providers(providers)
    # Cache key uses provider *names* only (options are env-stable per process).
    key = (str(onnx_path), tuple(providers))
    if key not in _SESSION_CACHE:
        so = ort.SessionOptions()
        so.enable_mem_pattern = True
        _SESSION_CACHE[key] = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=session_providers,
        )
    return _SESSION_CACHE[key]
