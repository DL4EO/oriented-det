"""Tests for export wrappers (no ONNX/TF required)."""

from __future__ import annotations

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
from oriented_det import RotatedFasterRCNN  # noqa: E402
from oriented_det.models.rotated_retinanet import RotatedRetinaNet  # noqa: E402

BackboneExportWrapper = _wrappers.BackboneExportWrapper
RetinaNetBackboneHeadExportWrapper = _wrappers.RetinaNetBackboneHeadExportWrapper
RotatedFasterRCNNPreNmsExportWrapper = _wrappers.RotatedFasterRCNNPreNmsExportWrapper


@pytest.fixture
def tiny_retinanet() -> RotatedRetinaNet:
    return RotatedRetinaNet(
        num_classes=3,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=5,
    )


def test_backbone_wrapper_shapes(tiny_retinanet: RotatedRetinaNet) -> None:
    w = BackboneExportWrapper(tiny_retinanet.backbone)
    w.eval()
    x = torch.randn(2, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        outs = w(x)
    assert isinstance(outs, tuple)
    assert len(outs) >= 1
    for t in outs:
        assert t.dim() == 4
        assert t.shape[0] == 2


def test_retinanet_heads_wrapper_shapes(tiny_retinanet: RotatedRetinaNet) -> None:
    w = RetinaNetBackboneHeadExportWrapper(tiny_retinanet)
    w.eval()
    x = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        outs = w(x)
    assert len(outs) % 2 == 0
    for i in range(0, len(outs), 2):
        cls_t, box_t = outs[i], outs[i + 1]
        assert cls_t.shape[0] == 1 and box_t.shape[0] == 1
        assert cls_t.shape[1] > 0 and box_t.shape[1] % 5 == 0


def test_retinanet_heads_batch_unbind_matches_stack(tiny_retinanet: RotatedRetinaNet) -> None:
    """B>1 path uses torch.unbind inside extract_backbone_features input list."""
    w = RetinaNetBackboneHeadExportWrapper(tiny_retinanet)
    w.eval()
    x2 = torch.randn(2, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        a = w(x2)
    x1 = torch.stack([x2[0], x2[1]], dim=0)
    assert torch.allclose(x1, x2)
    with torch.no_grad():
        b0 = w(x2[0:1])
        b1 = w(x2[1:2])
    assert len(a) == len(b0) == len(b1)
    for i in range(len(a)):
        assert a[i].shape[0] == 2
        assert torch.allclose(a[i][0], b0[i][0], rtol=1e-4, atol=1e-5)
        assert torch.allclose(a[i][1], b1[i][0], rtol=1e-4, atol=1e-5)


@pytest.fixture
def tiny_faster_rcnn() -> RotatedFasterRCNN:
    return RotatedFasterRCNN(
        num_classes=3,
        backbone_name="resnet18",
        pretrained_backbone=False,
        trainable_layers=5,
        rpn_post_nms_top_n=64,
        rpn_pre_nms_top_n=64,
    )


def test_faster_rcnn_pre_nms_wrapper_shapes(tiny_faster_rcnn: RotatedFasterRCNN) -> None:
    h, w = 128, 128
    wwrap = RotatedFasterRCNNPreNmsExportWrapper(tiny_faster_rcnn, height=h, width=w, max_candidates=32)
    wwrap.eval()
    x = torch.randn(1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        boxes, scores, labels, count = wwrap(x)
    assert boxes.shape == (32, 5)
    assert scores.shape == (32,)
    assert labels.shape == (32,)
    assert count.ndim == 0
    assert int(count.item()) <= 32
