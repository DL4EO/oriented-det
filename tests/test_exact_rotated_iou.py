"""Exact polygon IoU backend (Shapely) and mAP path when final_nms_use_cpu is set."""

import pytest

from oriented_det.geometry import RBox
from oriented_det.ops.nms import oriented_nms
from oriented_det.ops.utils import SHAPELY_AVAILABLE, resolve_exact_polygon_iou_backend


def test_resolve_exact_polygon_iou_backend_prefers_shapely_when_installed():
    backend = resolve_exact_polygon_iou_backend(warn_if_fallback=False)
    if SHAPELY_AVAILABLE:
        assert backend == "shapely"
    else:
        assert backend == "python"


def test_oriented_nms_uses_exact_backend_for_overlapping_boxes():
    boxes = [
        RBox(0.0, 0.0, 40.0, 4.0, 0.1),
        RBox(2.0, 0.0, 40.0, 4.0, 0.1),
        RBox(100.0, 100.0, 10.0, 10.0, 0.0),
    ]
    scores = [0.9, 0.85, 0.7]
    keep = oriented_nms(boxes, scores, iou_threshold=0.3)
    assert 0 in keep
    assert 2 in keep
    assert len(keep) == 2


def test_compute_oriented_map_exact_skips_gpu(monkeypatch):
    """When use_exact_rotated_iou=True, oriented_box_iou_gpu must not be called."""
    from oriented_det.data import Detection, GroundTruth, compute_oriented_map

    called = {"gpu": False}

    def _gpu_iou(*args, **kwargs):
        called["gpu"] = True
        raise AssertionError("GPU IoU should not run in exact mAP mode")

    monkeypatch.setattr(
        "oriented_det.data.evaluation.oriented_box_iou_gpu",
        _gpu_iou,
        raising=False,
    )
    # Force import path used inside compute_ap
    import oriented_det.ops.gpu_ops as gpu_ops

    monkeypatch.setattr(gpu_ops, "oriented_box_iou_gpu", _gpu_iou)

    det = Detection(
        rbox=RBox(50.0, 50.0, 20.0, 8.0, 0.0),
        score=0.9,
        class_id=1,
        class_name="ship",
        image_id="img0",
    )
    gt = GroundTruth(
        rbox=RBox(52.0, 50.0, 20.0, 8.0, 0.0),
        class_id=1,
        class_name="ship",
        image_id="img0",
    )
    mAP, aps, _metrics = compute_oriented_map(
        {"img0": [det]},
        {"img0": [gt]},
        show_progress=False,
        device="cuda",
        use_exact_rotated_iou=True,
    )
    assert called["gpu"] is False
    assert "ship" in aps
    assert mAP == pytest.approx(1.0, abs=1e-6)
