"""Parity: PyTorch full detect vs pre-NMS export + postprocess."""

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

from export.postprocess import finalize_detections_numpy, meta_to_finalize_kwargs  # noqa: E402
import wrappers as _wrappers  # noqa: E402
from oriented_det import RotatedFasterRCNN  # noqa: E402
from oriented_det.models.faster_rcnn_inference import faster_rcnn_inference  # noqa: E402
from oriented_det.models.utils import rboxes_to_tensor  # noqa: E402

RotatedFasterRCNNPreNmsExportWrapper = _wrappers.RotatedFasterRCNNPreNmsExportWrapper

_DEPLOY_CONFIG = _REPO_ROOT / "deploy/app/config.json"
_DEPLOY_CKPT = _REPO_ROOT / "deploy/app/weights/model.pth"


@pytest.fixture
def tiny_faster_rcnn() -> RotatedFasterRCNN:
    return RotatedFasterRCNN(
        num_classes=3,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=5,
        rpn_post_nms_top_n=128,
        rpn_pre_nms_top_n=128,
        inference_pre_nms_score_threshold=0.01,
        final_nms_iou_threshold=0.3,
        nms_class_agnostic=True,
        final_nms_use_cpu=True,
        max_detections_per_image=64,
        roi_inference_top_class_only=True,
    )


def _finalize_kwargs_from_model(model: RotatedFasterRCNN) -> dict:
    return {
        "nms_class_agnostic": model.nms_class_agnostic,
        "final_nms_iou_threshold": model.final_nms_iou_threshold,
        "max_detections_per_image": model.max_detections_per_image,
        "final_nms_use_cpu": model.final_nms_use_cpu,
        "score_threshold": 0.05,
        "per_class_score_threshold": None,
        "class_id_to_name": {i + 1: f"class_{i}" for i in range(model.num_classes)},
        "max_output_slots": model.max_detections_per_image,
    }


def test_pre_nms_finalize_matches_full_inference(tiny_faster_rcnn: RotatedFasterRCNN) -> None:
    """Export path + postprocess should match faster_rcnn_inference (deterministic RPN)."""
    model = tiny_faster_rcnn
    model.eval()
    h, w = 128, 128
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w, dtype=torch.float32)

    with torch.no_grad():
        full_out = faster_rcnn_inference(model, [x[0]], deterministic_rpn=True)[0]
        wrap = RotatedFasterRCNNPreNmsExportWrapper(model, height=h, width=w, max_candidates=128)
        boxes, scores, labels, count = wrap(x)
        det_np, num = finalize_detections_numpy(
            boxes.cpu().numpy(),
            scores.cpu().numpy(),
            labels.cpu().numpy(),
            int(count.item()),
            **_finalize_kwargs_from_model(model),
        )

    if full_out["scores"].numel() == 0:
        assert num == 0
        return

    full_boxes = rboxes_to_tensor(full_out["rboxes"])
    full_scores = full_out["scores"]
    full_labels = full_out["labels"]

    assert num == int(full_boxes.shape[0])
    assert torch.allclose(
        torch.from_numpy(det_np[:num, :5]),
        full_boxes,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(
        torch.from_numpy(det_np[:num, 5]),
        full_scores,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(
        torch.from_numpy(det_np[:num, 6].astype(np.int64)),
        full_labels,
    )


def test_faster_rcnn_pre_nms_onnx_checker(tiny_faster_rcnn: RotatedFasterRCNN) -> None:
    pytest.importorskip("onnx")
    import io

    import onnx

    model = tiny_faster_rcnn
    model.eval()
    h, w = 128, 128
    wrap = RotatedFasterRCNNPreNmsExportWrapper(model, height=h, width=w, max_candidates=64)
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


def test_faster_rcnn_pre_nms_onnx_ort_random_input(tiny_faster_rcnn: RotatedFasterRCNN) -> None:
    """ORT must run on non-zero images (regression: pad_pre_nms used invalid Expand)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import io

    import numpy as np
    import onnxruntime as ort

    model = tiny_faster_rcnn
    model.eval()
    h, w = 128, 128
    wrap = RotatedFasterRCNNPreNmsExportWrapper(model, height=h, width=w, max_candidates=64)
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
    buf.seek(0)
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    for seed in range(8):
        torch.manual_seed(seed)
        x = torch.rand(1, 3, h, w, dtype=torch.float32).numpy()
        outs = sess.run(None, {"images": x})
        assert outs[0].shape == (64, 5)
        assert int(np.asarray(outs[3]).reshape(-1)[0]) >= 0


@pytest.mark.skipif(
    not _DEPLOY_CKPT.is_file() or not _DEPLOY_CONFIG.is_file(),
    reason="deploy/app weights or config missing",
)
def test_deploy_checkpoint_pre_nms_finalize() -> None:
    """Smoke parity on published deploy weights (slow; optional)."""
    import json

    from oriented_det.runtime.checkpoint import load_model_from_checkpoint

    model, config, class_names = load_model_from_checkpoint(
        str(_DEPLOY_CKPT), str(_DEPLOY_CONFIG), device="cpu"
    )
    assert isinstance(model, RotatedFasterRCNN)

    meta = {
        "production": json.loads(_DEPLOY_CONFIG.read_text()).get("production"),
        "class_names": class_names or [],
    }
    kwargs = meta_to_finalize_kwargs(meta)
    h, w = 256, 256
    wrap = RotatedFasterRCNNPreNmsExportWrapper(
        model, height=h, width=w, max_candidates=model.rpn_post_nms_top_n
    )
    x = torch.rand(1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        boxes, scores, labels, count = wrap(x)
        _, num = finalize_detections_numpy(
            boxes.numpy(),
            scores.numpy(),
            labels.numpy(),
            int(count.item()),
            **kwargs,
        )
    assert num >= 0
