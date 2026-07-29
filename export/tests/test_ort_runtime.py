"""Tests for ORT device / provider resolution (no GPU required)."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

from export.ort_runtime import (
    clear_ort_session_cache,
    configure_ort_device,
    get_ort_device,
    hide_tensorflow_gpus,
    ort_providers_for_device,
    prepare_tensorflow_for_ort,
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


def test_prepare_tensorflow_for_ort_hides_gpus_when_cuda(monkeypatch):
    called = {"n": 0}

    def _fake_hide() -> bool:
        called["n"] += 1
        return True

    monkeypatch.setattr("export.ort_runtime.hide_tensorflow_gpus", _fake_hide)
    prepare_tensorflow_for_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert called["n"] == 1
    prepare_tensorflow_for_ort(["CPUExecutionProvider"])
    assert called["n"] == 1  # CPU path must not hide


def test_hide_tensorflow_gpus_idempotent():
    pytest.importorskip("tensorflow")
    assert hide_tensorflow_gpus() is True
    assert hide_tensorflow_gpus() is True


def test_ort_cuda_smoke_real_tile_1024():
    """One real 1024 tile must run on ORT CUDA without ROI Transpose OOM."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider not available")

    from pathlib import Path

    import numpy as np
    from PIL import Image

    clear_ort_session_cache()
    providers = configure_ort_device("cuda")
    assert "CUDAExecutionProvider" in providers

    onnx_path = Path("/tmp/odet_export_orcnn/detect/model.onnx")
    if not onnx_path.is_file():
        onnx_path = Path("/tmp/odet_export_orcnn/pre_nms.onnx")
    if not onnx_path.is_file():
        pytest.skip("No exported Oriented R-CNN ONNX at /tmp/odet_export_orcnn/")

    demo = Path("/home/jeffaudi/odet-planes/demo/planes_pleiades_neo.jpg")
    if not demo.is_file():
        pytest.skip("planes demo image missing")

    from export.ort_runtime import get_ort_session

    sess = get_ort_session(str(onnx_path), device="cuda")
    img = Image.open(demo).convert("RGB").resize((1024, 1024))
    x = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {"images": x})
    assert outs[0].shape[0] == 6000
    assert int(np.asarray(outs[3]).reshape(-1)[0]) >= 0
