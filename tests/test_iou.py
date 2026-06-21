import math
import pytest

from oriented_det.geometry import Polygon, RBox
from oriented_det.ops import iou, utils as ops_utils


def test_polygon_iou_partial_overlap():
    a = Polygon.rectangle(0, 0, 2, 2)
    b = Polygon.rectangle(0.5, 0.5, 2, 2)
    expected_inter = 1.5 * 1.5
    union = a.area + b.area - expected_inter
    expected = expected_inter / union
    assert math.isclose(iou.polygon_iou(a, b), expected, rel_tol=1e-6)


def test_rbox_iou_identical_gives_one():
    box = RBox(0, 0, 4, 2, math.radians(20))
    assert math.isclose(iou.rbox_iou(box, box), 1.0)


def test_rbox_iou_non_overlapping_zero():
    a = RBox(0, 0, 2, 1, 0)
    b = RBox(10, 0, 2, 1, 0)
    assert iou.rbox_iou(a, b) == 0.0


def test_batch_rbox_iou_returns_matrix():
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(1, 0, 2, 2, math.radians(45)),
    ]
    result = iou.batch_rbox_iou(boxes, boxes)
    assert len(result) == len(boxes)
    assert all(len(row) == len(boxes) for row in result)
    assert math.isclose(result[0][0], 1.0)


def test_rbox_iou_backend_validation():
    box = RBox(0, 0, 2, 1, 0)
    with pytest.raises(ValueError):
        iou.rbox_iou(box, box, backend="invalid")
    if ops_utils.TORCH_BOX_IOU_ROTATED is None:
        with pytest.raises(RuntimeError):
            iou.rbox_iou(box, box, backend="torch")


def test_rbox_iou_aabb_prefilter_consistency():
    """Verify that AABB pre-filtering doesn't change results."""
    # Test cases with various overlaps
    test_cases = [
        (RBox(0, 0, 2, 2, 0), RBox(0, 0, 2, 2, 0)),  # Identical
        (RBox(0, 0, 2, 2, 0), RBox(10, 10, 2, 2, 0)),  # Far apart
        (RBox(0, 0, 2, 2, 0), RBox(1, 1, 2, 2, 0)),  # Overlapping
        (RBox(0, 0, 2, 2, math.radians(45)), RBox(1, 1, 2, 2, math.radians(45))),  # Rotated overlapping
        (RBox(0, 0, 2, 2, math.radians(90)), RBox(10, 0, 2, 2, math.radians(90))),  # Rotated, far apart
    ]
    
    for box_a, box_b in test_cases:
        # Results should be identical with and without pre-filtering
        result_with = iou.rbox_iou(box_a, box_b, use_aabb_prefilter=True, backend="python")
        result_without = iou.rbox_iou(box_a, box_b, use_aabb_prefilter=False, backend="python")
        assert math.isclose(result_with, result_without, rel_tol=1e-9), \
            f"Mismatch for boxes {box_a} and {box_b}: {result_with} != {result_without}"


def test_batch_rbox_iou_aabb_prefilter_consistency():
    """Verify that AABB pre-filtering doesn't change batch IoU results."""
    boxes_a = [
        RBox(0, 0, 2, 2, 0),
        RBox(10, 10, 2, 2, 0),
        RBox(1, 1, 2, 2, math.radians(45)),
    ]
    boxes_b = [
        RBox(0, 0, 2, 2, 0),
        RBox(1, 1, 2, 2, 0),
        RBox(20, 20, 2, 2, 0),
    ]
    
    # Results should be identical with and without pre-filtering
    result_with = iou.batch_rbox_iou(boxes_a, boxes_b, use_aabb_prefilter=True, intersection_backend="python")
    result_without = iou.batch_rbox_iou(boxes_a, boxes_b, use_aabb_prefilter=False, intersection_backend="python")
    
    assert len(result_with) == len(result_without)
    for row_with, row_without in zip(result_with, result_without):
        assert len(row_with) == len(row_without)
        for val_with, val_without in zip(row_with, row_without):
            assert math.isclose(val_with, val_without, rel_tol=1e-9), \
                f"Mismatch in batch IoU: {val_with} != {val_without}"


def test_rbox_iou_is_symmetric():
    """IoU must be symmetric: IoU(a, b) == IoU(b, a)."""
    a = RBox(0, 0, 4, 2, math.radians(20))
    b = RBox(1, -0.5, 3, 1.5, math.radians(-10))
    iou_ab = iou.rbox_iou(a, b, backend="python")
    iou_ba = iou.rbox_iou(b, a, backend="python")
    assert math.isclose(iou_ab, iou_ba, rel_tol=1e-9, abs_tol=1e-12)


def test_batch_rbox_iou_matches_single_call_values():
    """Batch IoU entries should match pairwise single-call IoU."""
    boxes_a = [
        RBox(0, 0, 2, 2, 0),
        RBox(1, 0, 3, 1.5, math.radians(15)),
    ]
    boxes_b = [
        RBox(0.5, 0.0, 2, 2, 0),
        RBox(10, 10, 2, 2, 0),
    ]
    mat = iou.batch_rbox_iou(boxes_a, boxes_b, intersection_backend="python")
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            scalar = iou.rbox_iou(a, b, backend="python")
            assert math.isclose(mat[i][j], scalar, rel_tol=1e-9, abs_tol=1e-12)
