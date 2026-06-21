"""Tests for ORT device / provider resolution (no GPU required)."""

from __future__ import annotations

import pytest

from export.ort_runtime import (
    clear_ort_session_cache,
    configure_ort_device,
    get_ort_device,
    ort_providers_for_device,
    set_ort_device,
)


def test_cpu_providers():
    set_ort_device("cpu")
    assert ort_providers_for_device() == ["CPUExecutionProvider"]


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown ORT device"):
        ort_providers_for_device("tpu")


def test_configure_returns_cpu_by_default():
    clear_ort_session_cache()
    set_ort_device(None)
    providers = configure_ort_device("cpu")
    assert providers == ["CPUExecutionProvider"]
    assert get_ort_device() == "cpu"


def test_cuda_without_gpu_raises():
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA EP available; skip missing-CUDA test")
    set_ort_device("cuda")
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        ort_providers_for_device()
