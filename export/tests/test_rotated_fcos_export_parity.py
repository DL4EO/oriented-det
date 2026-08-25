"""Parity: PyTorch Rotated FCOS detect vs pre-NMS export + postprocess."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

from export.postprocess import finalize_detections_numpy  # noqa: E402
import wrappers as _wrappers  # noqa: E402
from oriented_det import RotatedFCOS  # noqa: E402
from oriented_det.models.rotated_fcos import _apply_fcos_class_aware_nms  # noqa: E402
from oriented_det.models.utils import rboxes_to_tensor  # noqa: E402

RotatedFCOSPreNmsExportWrapper = _wrappers.RotatedFCOSPreNmsExportWrapper


@pytest.fixture
def tiny_fcos() -> RotatedFCOS:
    return RotatedFCOS(
        num_classes=3,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=5,
        returned_layers=[2, 3, 4],
        fpn_extra_level=True,
        fpn_strides=[8, 16, 32, 64, 128],
        nms_pre=32,
        score_threshold=0.01,
        final_nms_iou_threshold=0.3,
        final_nms_use_cpu=True,
        max_detections_per_image=256,
    )


def _finalize_kwargs_from_model(model: RotatedFCOS) -> dict:
    return {
        "nms_class_agnostic": False,
        "final_nms_iou_threshold": model.final_nms_iou_threshold,
        "max_detections_per_image": model.max_detections_per_image,
        "final_nms_use_cpu": model.final_nms_use_cpu,
        # Drop zeroed invalid top-k slots (same as production score floor).
        "score_threshold": max(model.score_threshold, 1e-6),
        "per_class_score_threshold": None,
        "class_id_to_name": {i + 1: f"class_{i}" for i in range(model.num_classes)},
        "max_output_slots": model.max_detections_per_image,
    }


def _sort_detections(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(scores, descending=True)
    return boxes[order], scores[order], labels[order]


def test_pre_nms_fcos_nms_matches_full_inference(tiny_fcos: RotatedFCOS) -> None:
    """Export decode + FCOS class-aware NMS should match RotatedFCOS.eval."""
    model = tiny_fcos
    model.eval()
    h, w = 128, 128
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w, dtype=torch.float32)

    with torch.no_grad():
        full_out = model([x[0]])[0]
        wrap = RotatedFCOSPreNmsExportWrapper(model, height=h, width=w, max_candidates=160)
        boxes, scores, labels, count = wrap(x)
        n = int(count.item())
        live = (scores[:n] > 0) & (boxes[:n, 2] > 0) & (boxes[:n, 3] > 0)
        boxes_k, scores_k, labels_k = _apply_fcos_class_aware_nms(
            boxes[:n][live],
            scores[:n][live],
            labels[:n][live],
            iou_threshold=model.final_nms_iou_threshold,
            max_detections_per_image=model.max_detections_per_image,
            final_nms_use_cpu=model.final_nms_use_cpu,
        )

    if full_out["scores"].numel() == 0:
        assert boxes_k.numel() == 0
        return

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


def test_pre_nms_finalize_matches_shared_nms(tiny_fcos: RotatedFCOS) -> None:
    """Keras-bundle finalize matches ``apply_final_rotated_nms`` (incl. w,h >= 1)."""
    from oriented_det.models.faster_rcnn_inference import (
        PreNmsDetections,
        apply_final_rotated_nms,
    )

    model = tiny_fcos
    model.eval()
    h, w = 128, 128
    torch.manual_seed(1)
    x = torch.rand(1, 3, h, w, dtype=torch.float32)

    with torch.no_grad():
        wrap = RotatedFCOSPreNmsExportWrapper(model, height=h, width=w, max_candidates=160)
        boxes, scores, labels, count = wrap(x)
        n = int(count.item())
        fk = _finalize_kwargs_from_model(model)
        det_np, num = finalize_detections_numpy(
            boxes.cpu().numpy(),
            scores.cpu().numpy(),
            labels.cpu().numpy(),
            n,
            **fk,
        )
        class _View:
            nms_class_agnostic = False
            final_nms_iou_threshold = model.final_nms_iou_threshold
            max_detections_per_image = model.max_detections_per_image
            final_nms_use_cpu = model.final_nms_use_cpu

        keep = scores[:n] >= float(fk["score_threshold"])
        ref = apply_final_rotated_nms(
            _View(),
            PreNmsDetections(boxes[:n][keep], scores[:n][keep], labels[:n][keep]),
        )

    assert num == int(ref.boxes.shape[0])
    if num == 0:
        return
    ref_boxes, ref_scores, ref_labels = _sort_detections(ref.boxes, ref.scores, ref.labels)
    exp_boxes = torch.from_numpy(det_np[:num, :5])
    exp_scores = torch.from_numpy(det_np[:num, 5])
    exp_labels = torch.from_numpy(det_np[:num, 6].astype(np.int64))
    exp_boxes, exp_scores, exp_labels = _sort_detections(exp_boxes, exp_scores, exp_labels)
    assert torch.allclose(exp_boxes, ref_boxes.cpu(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(exp_scores, ref_scores.cpu(), atol=1e-4, rtol=1e-4)
    assert torch.equal(exp_labels, ref_labels.cpu())


def test_fcos_pre_nms_onnx_checker(tiny_fcos: RotatedFCOS) -> None:
    pytest.importorskip("onnx")
    import io

    import onnx

    model = tiny_fcos
    model.eval()
    h, w = 128, 128
    wrap = RotatedFCOSPreNmsExportWrapper(model, height=h, width=w, max_candidates=64)
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
    )
    buf.seek(0)
    onnx.checker.check_model(onnx.load(buf))


def test_fcos_pre_nms_onnx_ort_random_input(tiny_fcos: RotatedFCOS) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import io

    import onnx
    import onnxruntime as ort

    model = tiny_fcos
    model.eval()
    h, w = 128, 128
    max_candidates = 64
    wrap = RotatedFCOSPreNmsExportWrapper(
        model, height=h, width=w, max_candidates=max_candidates
    )
    buf = io.BytesIO()
    torch.onnx.export(
        wrap,
        torch.zeros(1, 3, h, w, dtype=torch.float32),
        buf,
        input_names=["images"],
        output_names=["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
        opset_version=17,
        do_constant_folding=True,
    )
    raw = buf.getvalue()
    onnx.checker.check_model(onnx.load(io.BytesIO(raw)))
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])

    def _run(arr: np.ndarray) -> None:
        outs = sess.run(None, {"images": arr})
        assert outs[0].shape == (max_candidates, 5)
        count = int(np.asarray(outs[3]).reshape(-1)[0])
        assert 0 <= count <= max_candidates

    _run(np.zeros((1, 3, h, w), dtype=np.float32))
    rng = np.random.RandomState(0)
    _run(rng.rand(1, 3, h, w).astype(np.float32))
