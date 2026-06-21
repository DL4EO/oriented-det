import math
import pytest

from oriented_det.geometry import RBox
from oriented_det.ops import nms, utils as ops_utils


def test_oriented_nms_suppresses_overlap():
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.2, 0.1, 2, 2, 0),
        RBox(5, 5, 2, 2, math.radians(30)),
    ]
    scores = [0.95, 0.9, 0.6]
    keep = nms.oriented_nms(boxes, scores, iou_threshold=0.3)
    assert keep == [0, 2]


def test_oriented_nms_accepts_tuples():
    boxes = [
        (0, 0, 2, 1, 0),
        (6, 0, 2, 1, 0),
    ]
    keep = nms.oriented_nms(boxes, [0.7, 0.8], iou_threshold=0.1)
    assert keep == [1, 0]


def test_oriented_nms_backend_validation():
    boxes = [RBox(0, 0, 2, 1, 0)]
    scores = [0.5]
    with pytest.raises(ValueError):
        nms.oriented_nms(boxes, scores, backend="invalid")
    if ops_utils.TORCH_NMS_ROTATED is None:
        with pytest.raises(RuntimeError):
            nms.oriented_nms(boxes, scores, backend="torch")


def test_oriented_nms_aabb_prefilter_correctness():
    """Verify that AABB pre-filtering in NMS produces correct results.
    
    This test ensures that the AABB optimization doesn't change NMS behavior.
    We test various scenarios including sparse and dense detections.
    """
    # Test 1: Sparse detections (AABB pre-filtering should help most here)
    sparse_boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(100, 100, 2, 2, 0),  # Far away
        RBox(200, 200, 2, 2, 0),  # Far away
    ]
    sparse_scores = [0.9, 0.8, 0.7]
    keep_sparse = nms.oriented_nms(sparse_boxes, sparse_scores, iou_threshold=0.3, backend="python")
    # All should be kept since they're far apart
    assert len(keep_sparse) == 3
    
    # Test 2: Dense overlapping detections
    # Boxes at (0,0) and (0.2, 0.2) should overlap significantly
    dense_boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.2, 0.2, 2, 2, 0),  # Overlapping with first
        RBox(10, 10, 2, 2, 0),  # Far away
    ]
    dense_scores = [0.95, 0.85, 0.75]
    keep_dense = nms.oriented_nms(dense_boxes, dense_scores, iou_threshold=0.3, backend="python")
    # First box should suppress second (they overlap), but keep the far one
    assert 0 in keep_dense  # Highest score kept
    assert 2 in keep_dense  # Far box kept
    assert len(keep_dense) == 2  # Only 2 should remain
    
    # Test 3: Rotated boxes
    rotated_boxes = [
        RBox(0, 0, 4, 2, math.radians(0)),
        RBox(1, 1, 4, 2, math.radians(45)),  # Rotated overlap
        RBox(20, 20, 4, 2, math.radians(90)),  # Far, rotated
    ]
    rotated_scores = [0.9, 0.8, 0.7]
    keep_rotated = nms.oriented_nms(rotated_boxes, rotated_scores, iou_threshold=0.3, backend="python")
    # Should keep first and third (first suppresses second due to overlap)
    assert 0 in keep_rotated
    assert 2 in keep_rotated


def test_class_aware_nms_different_classes():
    """Test that boxes of different classes don't suppress each other."""
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.2, 0.2, 2, 2, 0),  # Overlapping with first, but different class
        RBox(10, 10, 2, 2, 0),  # Far away, different class
    ]
    scores = [0.9, 0.8, 0.7]
    labels = [1, 2, 3]  # All different classes
    
    keep = nms.oriented_nms(boxes, scores, labels=labels, iou_threshold=0.3, backend="python")
    # All boxes should be kept since they're different classes
    assert len(keep) == 3
    assert set(keep) == {0, 1, 2}


def test_class_aware_nms_same_classes():
    """Test that boxes of the same class suppress each other normally."""
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.2, 0.2, 2, 2, 0),  # Overlapping with first, same class
        RBox(10, 10, 2, 2, 0),  # Far away, same class
    ]
    scores = [0.9, 0.8, 0.7]
    labels = [1, 1, 1]  # All same class
    
    keep = nms.oriented_nms(boxes, scores, labels=labels, iou_threshold=0.3, backend="python")
    # First box should suppress second (they overlap), but keep the far one
    assert 0 in keep  # Highest score kept
    assert 2 in keep  # Far box kept (same class but no overlap)
    assert len(keep) == 2
    assert 1 not in keep  # Second box should be suppressed


def test_class_aware_nms_mixed_classes():
    """Test class-aware NMS with mixed classes."""
    boxes = [
        RBox(0, 0, 2, 2, 0),           # Class 1, score 0.95
        RBox(0.2, 0.2, 2, 2, 0),       # Class 1, score 0.85 (overlaps with first)
        RBox(0.3, 0.3, 2, 2, 0),       # Class 2, score 0.75 (overlaps with first two)
        RBox(10, 10, 2, 2, 0),         # Class 1, score 0.65 (far away)
    ]
    scores = [0.95, 0.85, 0.75, 0.65]
    labels = [1, 1, 2, 1]
    
    keep = nms.oriented_nms(boxes, scores, labels=labels, iou_threshold=0.3, backend="python")
    # Box 0 (class 1, highest score) should suppress box 1 (same class, overlaps)
    # Box 2 (class 2) should be kept even though it overlaps with boxes 0 and 1 (different class)
    # Box 3 (class 1, far away) should be kept
    assert 0 in keep  # Highest score of class 1
    assert 2 in keep  # Different class, kept despite overlap
    assert 3 in keep  # Same class but far away
    assert len(keep) == 3
    assert 1 not in keep  # Suppressed by box 0 (same class, overlaps)


def test_class_aware_nms_labels_validation():
    """Test that labels parameter is properly validated."""
    boxes = [RBox(0, 0, 2, 1, 0), RBox(5, 5, 2, 1, 0)]
    scores = [0.9, 0.8]
    
    # Labels must match boxes length
    with pytest.raises(ValueError, match="labels must have the same length"):
        nms.oriented_nms(boxes, scores, labels=[1])


def test_class_aware_nms_backward_compatibility():
    """Test that class-aware NMS is backward compatible (labels optional)."""
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.2, 0.2, 2, 2, 0),  # Overlapping
        RBox(10, 10, 2, 2, 0),  # Far away
    ]
    scores = [0.9, 0.8, 0.7]
    
    # Without labels (original behavior)
    keep_no_labels = nms.oriented_nms(boxes, scores, iou_threshold=0.3, backend="python")
    
    # With labels=None (should behave the same)
    keep_labels_none = nms.oriented_nms(boxes, scores, labels=None, iou_threshold=0.3, backend="python")
    
    # Results should be identical
    assert keep_no_labels == keep_labels_none


def test_oriented_nms_is_idempotent_on_kept_subset():
    """Running NMS on already-kept detections should keep all of them."""
    boxes = [
        RBox(0, 0, 2, 2, 0),
        RBox(0.25, 0.1, 2, 2, 0),  # overlaps first
        RBox(4, 4, 2, 2, 0),
        RBox(8, 8, 2, 2, math.radians(15)),
    ]
    scores = [0.99, 0.88, 0.7, 0.6]
    keep1 = nms.oriented_nms(boxes, scores, iou_threshold=0.3, backend="python")
    kept_boxes = [boxes[i] for i in keep1]
    kept_scores = [scores[i] for i in keep1]

    keep2 = nms.oriented_nms(kept_boxes, kept_scores, iou_threshold=0.3, backend="python")
    assert keep2 == list(range(len(kept_boxes)))
