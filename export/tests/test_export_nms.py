"""Numpy rotated IoU / NMS (no torch, no oriented-det)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export.nms import (  # noqa: E402
    any_pair_iou_at_least,
    max_pairwise_rbox_iou,
    rbox_iou,
    rboxes_to_corners,
    rotated_nms,
    shapely_available,
)
from export.postprocess import finalize_detections_numpy  # noqa: E402


def test_nms_lives_in_export_package() -> None:
    import importlib

    mod = importlib.import_module("export.nms")
    assert hasattr(mod, "rotated_nms")


def test_identical_axis_aligned_iou_is_one() -> None:
    box = np.array([10.0, 20.0, 8.0, 4.0, 0.0], dtype=np.float32)
    assert rbox_iou(box, box) > 0.99
    box = np.array([10.0, 20.0, 8.0, 4.0, 0.0], dtype=np.float32)
    assert rbox_iou(box, box) > 0.99


def test_disjoint_boxes_iou_is_zero() -> None:
    a = np.array([0.0, 0.0, 2.0, 2.0, 0.0])
    b = np.array([50.0, 50.0, 2.0, 2.0, 0.0])
    assert rbox_iou(a, b) < 1e-6


def test_axis_aligned_partial_overlap_iou() -> None:
    a = np.array([2.0, 2.0, 4.0, 4.0, 0.0])
    b = np.array([4.0, 2.0, 4.0, 4.0, 0.0])
    np.testing.assert_allclose(rbox_iou(a, b), 8.0 / 24.0, atol=1e-3)


def test_rbox_corners_match_expected_axis_aligned() -> None:
    corners = rboxes_to_corners(np.array([[10.0, 20.0, 8.0, 4.0, 0.0]]))[0]
    expected = np.array([[6.0, 18.0], [14.0, 18.0], [14.0, 22.0], [6.0, 22.0]], dtype=np.float64)
    np.testing.assert_allclose(corners, expected, atol=1e-5)


def test_rotated_nms_suppresses_duplicate() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [10.2, 10.1, 8.0, 4.0, 0.0],
            [40.0, 40.0, 6.0, 3.0, math.pi / 6],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.array([1, 1, 1], dtype=np.int64)
    keep = [int(i) for i in rotated_nms(
        boxes,
        scores,
        labels,
        iou_threshold=0.5,
        max_detections=10,
        class_agnostic=True,
    )]
    assert keep[0] == 0
    assert 2 in keep
    assert 1 not in keep


def test_class_aware_nms_keeps_overlapping_other_class() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [10.2, 10.1, 8.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = rotated_nms(
        boxes,
        scores,
        np.array([1, 2], dtype=np.int64),
        iou_threshold=0.5,
        max_detections=10,
        class_agnostic=False,
    )
    assert set(int(i) for i in keep) == {0, 1}


def test_finalize_score_floor_and_padding() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [40.0, 40.0, 6.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.05, 0.0], dtype=np.float32)
    labels = np.array([1, 1, 0], dtype=np.int32)
    det, num = finalize_detections_numpy(
        boxes,
        scores,
        labels,
        3,
        nms_class_agnostic=True,
        final_nms_iou_threshold=0.5,
        max_detections_per_image=8,
        final_nms_use_cpu=True,
        score_threshold=0.2,
        per_class_score_threshold=None,
        class_id_to_name={1: "a"},
        max_output_slots=8,
    )
    assert num == 1
    assert tuple(det.shape) == (8, 7)
    np.testing.assert_allclose(det[0, :5], boxes[0], atol=1e-5)
    np.testing.assert_allclose(det[0, 5], 0.9, atol=1e-5)
    np.testing.assert_allclose(det[1:], 0.0)


def test_max_pairwise_helpers() -> None:
    same = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert any_pair_iou_at_least(same, 0.5)
    assert max_pairwise_rbox_iou(same) > 0.99
    sep = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [80.0, 80.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert not any_pair_iou_at_least(sep, 0.5)
    assert max_pairwise_rbox_iou(sep) < 0.05


@pytest.mark.skipif(not shapely_available(), reason="shapely not installed")
def test_shapely_backend_matches_python_on_axis_aligned() -> None:
    a = np.array([2.0, 2.0, 4.0, 4.0, 0.0])
    b = np.array([4.0, 2.0, 4.0, 4.0, 0.0])
    np.testing.assert_allclose(
        rbox_iou(a, b, backend="python"),
        rbox_iou(a, b, backend="shapely"),
        atol=1e-4,
    )


@pytest.mark.skipif(not shapely_available(), reason="shapely not installed")
def test_shapely_nms_suppresses_duplicate() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [10.2, 10.1, 8.0, 4.0, 0.0],
            [40.0, 40.0, 6.0, 3.0, math.pi / 6],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.array([1, 1, 1], dtype=np.int64)
    keep = [int(i) for i in rotated_nms(
        boxes,
        scores,
        labels,
        iou_threshold=0.5,
        max_detections=10,
        class_agnostic=True,
        backend="shapely",
    )]
    assert keep[0] == 0
    assert 2 in keep
    assert 1 not in keep
