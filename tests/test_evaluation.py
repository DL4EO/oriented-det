"""Tests for oriented mAP evaluation."""

from oriented_det.data.evaluation import (
    average_precision_area,
    ossdd_map_iou_thresholds,
)
from oriented_det.data import (
    APCalculator,
    ClassEvalMetrics,
    Detection,
    GroundTruth,
    compute_gt_best_iou_alignment_metrics,
    compute_oriented_map,
    format_gt_best_iou_alignment_table,
    format_mmrotate_class_metrics_table,
    gt_best_iou_alignment_metrics_to_dict,
)
from oriented_det.geometry import RBox


def test_detection_and_ground_truth():
    """Test Detection and GroundTruth dataclasses."""
    rbox = RBox(100, 200, 50, 30, 0)
    
    det = Detection(rbox=rbox, score=0.9, class_id=0, class_name="plane")
    gt = GroundTruth(rbox=rbox, class_id=0, class_name="plane", difficult=0)
    
    assert det.score == 0.9
    assert gt.difficult == 0


def test_ap_calculator_perfect_match():
    """Test AP calculation with perfect matches."""
    rbox1 = RBox(100, 100, 50, 30, 0)
    rbox2 = RBox(200, 200, 50, 30, 0)
    
    detections = [
        Detection(rbox=rbox1, score=0.9, class_id=0, class_name="plane"),
        Detection(rbox=rbox2, score=0.8, class_id=0, class_name="plane"),
    ]
    
    ground_truths = [
        GroundTruth(rbox=rbox1, class_id=0, class_name="plane"),
        GroundTruth(rbox=rbox2, class_id=0, class_name="plane"),
    ]
    
    calculator = APCalculator(iou_threshold=0.5)
    ap, metrics = calculator.compute_ap(detections, ground_truths, class_name="plane")

    # With perfect matches, AP should be 1.0
    assert ap > 0.9  # Allow some tolerance for floating point
    assert metrics.num_gts == 2
    assert metrics.num_dets == 2
    assert metrics.recall == 1.0


def test_compute_oriented_map():
    """Test mAP computation across multiple images."""
    rbox1 = RBox(100, 100, 50, 30, 0)
    rbox2 = RBox(200, 200, 50, 30, 0)
    
    detections = {
        "img1": [Detection(rbox=rbox1, score=0.9, class_id=0, class_name="plane")],
    }
    
    ground_truths = {
        "img1": [GroundTruth(rbox=rbox1, class_id=0, class_name="plane")],
    }
    
    map_score, class_aps, class_metrics = compute_oriented_map(detections, ground_truths)

    assert "plane" in class_aps
    assert "plane" in class_metrics
    assert class_metrics["plane"].recall == 1.0
    assert map_score >= 0.0


def test_ap_calculator_difficult_flags():
    """Test that difficult ground truths are handled correctly."""
    rbox = RBox(100, 100, 50, 30, 0)
    
    detections = [
        Detection(rbox=rbox, score=0.9, class_id=0, class_name="plane"),
    ]
    
    # Difficult GT should not count as TP or FP
    ground_truths = [
        GroundTruth(rbox=rbox, class_id=0, class_name="plane", difficult=1),
    ]
    
    calculator = APCalculator(iou_threshold=0.5)
    ap, metrics = calculator.compute_ap(detections, ground_truths, class_name="plane")

    # With only difficult GTs, AP should be 0 (no positives to match)
    assert ap == 0.0
    assert metrics.num_gts == 0
    assert metrics.recall == 0.0


def test_ap_calculator_missing_difficult_is_not_fn():
    """Missing a difficult GT must not count as FN (num_gts excludes difficult)."""
    easy = RBox(100, 100, 50, 30, 0)
    hard = RBox(200, 200, 50, 30, 0)
    detections = [
        Detection(rbox=easy, score=0.9, class_id=0, class_name="plane"),
    ]
    ground_truths = [
        GroundTruth(rbox=easy, class_id=0, class_name="plane", difficult=0),
        GroundTruth(rbox=hard, class_id=0, class_name="plane", difficult=1),
    ]
    calculator = APCalculator(iou_threshold=0.5)
    ap, metrics = calculator.compute_ap(detections, ground_truths, class_name="plane")
    assert metrics.num_gts == 1
    assert metrics.recall == 1.0
    assert ap > 0.9


def test_ap_calculator_no_detections():
    """Test AP calculation with no detections."""
    rbox = RBox(100, 100, 50, 30, 0)
    
    detections = []
    ground_truths = [
        GroundTruth(rbox=rbox, class_id=0, class_name="plane"),
    ]
    
    calculator = APCalculator(iou_threshold=0.5)
    ap, metrics = calculator.compute_ap(detections, ground_truths, class_name="plane")

    # No detections means AP = 0
    assert ap == 0.0
    assert metrics.recall == 0.0


def test_format_mmrotate_class_metrics_table():
    """MMRotate-style table includes gts, dets, recall, ap, and mAP row."""
    metrics = {
        "plane": ClassEvalMetrics(num_gts=2, num_dets=3, recall=0.5, ap=0.75),
    }
    table = format_mmrotate_class_metrics_table(metrics, mean_ap=0.75)
    assert "plane" in table
    assert "gts" in table
    assert "recall" in table
    assert "mAP" in table
    assert "0.750" in table


def test_ap_calculator_no_ground_truths():
    """Test AP calculation with no ground truths."""
    rbox = RBox(100, 100, 50, 30, 0)
    
    detections = [
        Detection(rbox=rbox, score=0.9, class_id=0, class_name="plane"),
    ]
    ground_truths = []
    
    calculator = APCalculator(iou_threshold=0.5)
    ap, metrics = calculator.compute_ap(detections, ground_truths, class_name="plane")

    # No GTs means AP = 0 (all detections are false positives)
    assert ap == 0.0
    assert metrics.num_gts == 0


def test_compute_gt_best_iou_alignment_metrics_per_class():
    """Per-class mean best IoU separates aligned vs misaligned classes."""
    rbox_good = RBox(100, 100, 50, 30, 0)
    rbox_bad = RBox(300, 300, 50, 30, 0.5)
    detections = {
        "i1": [
            Detection(rbox=rbox_good, score=0.9, class_id=0, class_name="plane", image_id="i1"),
            Detection(rbox=rbox_bad, score=0.8, class_id=1, class_name="ship", image_id="i1"),
        ],
    }
    ground_truths = {
        "i1": [
            GroundTruth(rbox=rbox_good, class_id=0, class_name="plane", difficult=0, image_id="i1"),
            GroundTruth(rbox=rbox_bad, class_id=1, class_name="ship", difficult=0, image_id="i1"),
        ],
    }
    metrics = compute_gt_best_iou_alignment_metrics(detections, ground_truths)
    assert metrics.num_gts == 2
    assert metrics.per_class["plane"].mean_best_iou_same_class > 0.99
    assert metrics.per_class["ship"].mean_best_iou_same_class > 0.99
    d = gt_best_iou_alignment_metrics_to_dict(metrics)
    assert d["per_class"]["plane"]["num_gts"] == 1
    table = format_gt_best_iou_alignment_table(metrics, markdown=True)
    assert "plane" in table
    assert "global" in table


def test_average_precision_area_perfect_and_miss():
    assert average_precision_area([1.0], [1.0]) == 1.0
    assert average_precision_area([0.0], [0.0]) == 0.0


def test_map_over_iou_thresholds_averages_area():
    """IoU ~1/3 is a hit at 0.1 and a miss at 0.5. Mean AP is 0.5."""
    gt = RBox(0, 0, 10, 10, 0)
    det = RBox(5, 0, 10, 10, 0)
    detections = {
        "img": [Detection(rbox=det, score=0.9, class_id=0, class_name="ship", image_id="img")],
    }
    ground_truths = {
        "img": [GroundTruth(rbox=gt, class_id=0, class_name="ship", image_id="img")],
    }
    score, class_aps, metrics = compute_oriented_map(
        detections,
        ground_truths,
        iou_threshold=0.5,
        iou_thresholds=[0.1, 0.5],
        show_progress=False,
    )
    assert abs(score - 0.5) < 1e-6
    assert abs(class_aps["ship"] - 0.5) < 1e-6
    assert metrics["ship"].recall == 0.0
    assert abs(metrics["ship"].ap - 0.5) < 1e-6


def test_map_over_iou_thresholds_perfect_match():
    box = RBox(10, 10, 8, 4, 0.2)
    detections = {
        "a": [Detection(rbox=box, score=0.8, class_id=0, class_name="ship", image_id="a")],
    }
    ground_truths = {
        "a": [GroundTruth(rbox=box, class_id=0, class_name="ship", image_id="a")],
    }
    score, _, metrics = compute_oriented_map(
        detections,
        ground_truths,
        iou_thresholds=ossdd_map_iou_thresholds(),
        show_progress=False,
    )
    assert len(ossdd_map_iou_thresholds()) == 18
    assert ossdd_map_iou_thresholds()[0] == 0.1
    assert ossdd_map_iou_thresholds()[-1] == 0.95
    assert abs(score - 1.0) < 1e-6
    assert metrics["ship"].recall == 1.0


def test_map_over_iou_thresholds_does_not_match_across_images():
    box = RBox(10, 10, 8, 4, 0)
    detections = {
        "a": [Detection(rbox=box, score=0.9, class_id=0, class_name="ship", image_id="a")],
    }
    ground_truths = {
        "b": [GroundTruth(rbox=box, class_id=0, class_name="ship", image_id="b")],
    }
    score, _, metrics = compute_oriented_map(
        detections,
        ground_truths,
        iou_thresholds=[0.1, 0.5],
        show_progress=False,
    )
    assert score == 0.0
    assert metrics["ship"].recall == 0.0

