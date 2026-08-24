"""Pure-TF rotated IoU / NMS (requires TensorFlow)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

tf = pytest.importorskip("tensorflow")

from export.tf_nms import (  # noqa: E402
    rbox_iou_one_to_many,
    rboxes_to_corners,
    rotated_nms_tf,
    tf_finalize_detections,
)


def test_identical_axis_aligned_iou_is_one() -> None:
    box = tf.constant([10.0, 20.0, 8.0, 4.0, 0.0])
    ious = rbox_iou_one_to_many(box, box[None, :])
    np.testing.assert_allclose(ious.numpy(), [1.0], atol=1e-4)


def test_disjoint_boxes_iou_is_zero() -> None:
    a = tf.constant([0.0, 0.0, 2.0, 2.0, 0.0])
    b = tf.constant([[50.0, 50.0, 2.0, 2.0, 0.0]])
    ious = rbox_iou_one_to_many(a, b)
    np.testing.assert_allclose(ious.numpy(), [0.0], atol=1e-4)


def test_axis_aligned_partial_overlap_iou() -> None:
    # 4x4 box at origin vs 4x4 shifted by 2 on x → intersection 2x4 = 8, union 16+16-8=24
    a = tf.constant([2.0, 2.0, 4.0, 4.0, 0.0])
    b = tf.constant([[4.0, 2.0, 4.0, 4.0, 0.0]])
    ious = rbox_iou_one_to_many(a, b)
    np.testing.assert_allclose(ious.numpy(), [8.0 / 24.0], atol=1e-3)


def test_rbox_corners_match_expected_axis_aligned() -> None:
    corners = rboxes_to_corners(tf.constant([[10.0, 20.0, 8.0, 4.0, 0.0]])).numpy()[0]
    expected = np.array([[6.0, 18.0], [14.0, 18.0], [14.0, 22.0], [6.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(corners, expected, atol=1e-5)


def test_rotated_nms_suppresses_duplicate() -> None:
    boxes = tf.constant(
        [
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [10.2, 10.1, 8.0, 4.0, 0.0],
            [40.0, 40.0, 6.0, 3.0, math.pi / 6],
        ],
        dtype=tf.float32,
    )
    scores = tf.constant([0.9, 0.8, 0.7], dtype=tf.float32)
    labels = tf.constant([1, 1, 1], dtype=tf.int32)
    keep = rotated_nms_tf(
        boxes,
        scores,
        labels,
        iou_threshold=0.5,
        max_detections=10,
        class_agnostic=True,
    )
    keep_list = [int(i) for i in keep.numpy()]
    assert keep_list[0] == 0
    assert 2 in keep_list
    assert 1 not in keep_list


def test_tf_finalize_score_floor_and_padding() -> None:
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
    det, num = tf_finalize_detections(
        tf.constant(boxes),
        tf.constant(scores),
        tf.constant(labels),
        tf.constant(3, tf.int32),
        nms_class_agnostic=True,
        final_nms_iou_threshold=0.5,
        max_detections_per_image=8,
        score_threshold=0.2,
        per_class_score_threshold=None,
        class_id_to_name={1: "a"},
        max_output_slots=8,
    )
    assert int(num.numpy()) == 1
    assert tuple(det.shape) == (8, 7)
    np.testing.assert_allclose(det.numpy()[0, :5], boxes[0], atol=1e-5)
    np.testing.assert_allclose(det.numpy()[0, 5], 0.9, atol=1e-5)
    np.testing.assert_allclose(det.numpy()[1:], 0.0)
