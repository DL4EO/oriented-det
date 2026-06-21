"""Optional ONNX smoke (each test skips if deps missing)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

import wrappers as _wrappers  # noqa: E402
from oriented_det.models.rotated_retinanet import RotatedRetinaNet  # noqa: E402

BackboneExportWrapper = _wrappers.BackboneExportWrapper
RetinaNetBackboneHeadExportWrapper = _wrappers.RetinaNetBackboneHeadExportWrapper


def test_export_backbone_onnx_roundtrip_ort() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnx
    import onnxruntime as ort

    model = RotatedRetinaNet(num_classes=2, backbone_name="resnet18", pretrained_backbone=False)
    model.eval()
    wrap = BackboneExportWrapper(model.backbone)
    x = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        ref = wrap(x)
    buf = io.BytesIO()
    torch.onnx.export(
        wrap,
        x,
        buf,
        input_names=["images"],
        output_names=[f"fpn_{i}" for i in range(len(ref))],
        opset_version=17,
        do_constant_folding=True,
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x.numpy()})
    assert len(out) == len(ref)
    for a, b in zip(out, ref):
        assert a.shape == tuple(b.shape)


def test_export_retinanet_heads_onnx_checker() -> None:
    pytest.importorskip("onnx")
    import onnx

    model = RotatedRetinaNet(num_classes=2, backbone_name="resnet18", pretrained_backbone=False)
    model.eval()
    wrap = RetinaNetBackboneHeadExportWrapper(model)
    x = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        ref = wrap(x)
    names = []
    for i in range(len(ref) // 2):
        names.extend([f"level{i}_cls", f"level{i}_bbox"])
    buf = io.BytesIO()
    torch.onnx.export(
        wrap,
        x,
        buf,
        input_names=["images"],
        output_names=names,
        opset_version=17,
        do_constant_folding=True,
    )
    buf.seek(0)
    onnx.checker.check_model(onnx.load(buf))
