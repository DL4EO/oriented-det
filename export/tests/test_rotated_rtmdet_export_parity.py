"""Parity: PyTorch Rotated RTMDet detect vs padded pre-NMS export."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export import wrappers as _wrappers  # noqa: E402
from oriented_det.models.rotated_rtmdet import (  # noqa: E402
    RotatedRTMDet,
    _apply_rtmdet_nms,
)
from oriented_det.models.utils import rboxes_to_tensor  # noqa: E402

RotatedRTMDetPreNmsExportWrapper = _wrappers.RotatedRTMDetPreNmsExportWrapper


@pytest.fixture
def tiny_rtmdet() -> RotatedRTMDet:
    model = RotatedRTMDet(
        num_classes=3,
        backbone_name="cspnext_tiny",
        pretrained_backbone=False,
        nms_pre=32,
        score_threshold=0.05,
        final_nms_iou_threshold=0.3,
        final_nms_use_cpu=True,
        max_detections_per_image=64,
        nms_class_agnostic=False,
    )
    # Prior cls bias is ~0.01; lift it so random weights emit detections.
    with torch.no_grad():
        model.head.rtm_cls.bias.fill_(0.0)
    return model


def _sort_detections(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(scores, descending=True)
    return boxes[order], scores[order], labels[order]


def test_pre_nms_rtmdet_nms_matches_full_inference(tiny_rtmdet: RotatedRTMDet) -> None:
    """Export decode + RTMDet NMS should match RotatedRTMDet.eval."""
    model = tiny_rtmdet
    model.eval()
    h, w = 128, 128
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w, dtype=torch.float32)

    with torch.no_grad():
        full_out = model([x[0]])[0]
        wrap = RotatedRTMDetPreNmsExportWrapper(model, height=h, width=w)
        wrap.eval()
        boxes, scores, labels, count = wrap(x)
        n = int(count.item())
        live = (scores[:n] > 0) & (boxes[:n, 2] > 0) & (boxes[:n, 3] > 0)
        boxes_k, scores_k, labels_k = _apply_rtmdet_nms(
            boxes[:n][live],
            scores[:n][live],
            labels[:n][live],
            iou_threshold=model.final_nms_iou_threshold,
            max_detections_per_image=model.max_detections_per_image,
            final_nms_use_cpu=model.final_nms_use_cpu,
            class_agnostic=model.nms_class_agnostic,
        )

    assert full_out["scores"].numel() > 0
    full_boxes = rboxes_to_tensor(full_out["rboxes"])
    full_scores = full_out["scores"]
    full_labels = full_out["labels"]
    full_boxes, full_scores, full_labels = _sort_detections(
        full_boxes, full_scores, full_labels
    )
    boxes_k, scores_k, labels_k = _sort_detections(boxes_k, scores_k, labels_k)

    assert boxes_k.shape[0] == full_boxes.shape[0]
    assert torch.allclose(boxes_k, full_boxes, atol=1e-4, rtol=1e-4)
    assert torch.allclose(scores_k, full_scores, atol=1e-4, rtol=1e-4)
    assert torch.equal(labels_k, full_labels)


def test_rtmdet_pre_nms_onnx_checker(tiny_rtmdet: RotatedRTMDet) -> None:
    pytest.importorskip("onnx")
    import io

    import onnx

    model = tiny_rtmdet
    model.eval()
    h, w = 128, 128
    wrap = RotatedRTMDetPreNmsExportWrapper(model, height=h, width=w, max_candidates=64)
    wrap.eval()
    x = torch.randn(1, 3, h, w, dtype=torch.float32)
    buf = io.BytesIO()
    torch.onnx.export(
        wrap,
        x,
        buf,
        input_names=["images"],
        output_names=["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    buf.seek(0)
    onnx.checker.check_model(onnx.load(buf))


def test_rtmdet_pre_nms_onnx_ort_random_input(tiny_rtmdet: RotatedRTMDet) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import io

    import onnx
    import onnxruntime as ort

    model = tiny_rtmdet
    # Default eval BN running stats collapse an untrained CSPNeXt, so every
    # location ties in top-k. Kaiming convs keep the graph input-dependent.
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(mod.weight)
                torch.nn.init.zeros_(mod.bias)
    model.eval()
    h, w = 128, 128
    max_candidates = 64
    wrap = RotatedRTMDetPreNmsExportWrapper(
        model, height=h, width=w, max_candidates=max_candidates
    )
    wrap.eval()
    buf = io.BytesIO()
    trace_in = torch.rand(1, 3, h, w, dtype=torch.float32)
    torch.onnx.export(
        wrap,
        trace_in,
        buf,
        input_names=["images"],
        output_names=["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    raw = buf.getvalue()
    onnx.checker.check_model(onnx.load(io.BytesIO(raw)))
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    other = torch.rand(1, 3, h, w, dtype=torch.float32)
    outs = sess.run(None, {"images": other.numpy()})
    assert outs[0].shape == (max_candidates, 5)
    with torch.no_grad():
        ref_boxes, ref_scores, ref_labels, ref_count = wrap(other)
    assert int(np.asarray(outs[3]).reshape(-1)[0]) == int(ref_count.item())
    assert np.allclose(outs[0], ref_boxes.numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(outs[1], ref_scores.numpy(), atol=1e-4, rtol=1e-4)
    assert np.array_equal(outs[2], ref_labels.numpy())
