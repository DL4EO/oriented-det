"""Tests for oriented mAP evaluation."""

import math

import pytest

from oriented_det.data.evaluation import (
    average_precision_area,
    average_precision_coco101,
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


def test_average_precision_coco101_matches_hand_computation():
    """Two GTs, dets TP / FP / TP: recall 0.5, 0.5, 1.0; precision 1, 0.5, 2/3.

    Envelope: 1.0 up to recall 0.5, then 2/3 up to 1.0. Of the 101 recall points,
    51 (0.00 .. 0.50) read 1.0 and 50 (0.51 .. 1.00) read 2/3.
    The all-point area is 0.5 * 1.0 + 0.5 * 2/3.
    """
    recalls = [0.5, 0.5, 1.0]
    precisions = [1.0, 0.5, 2.0 / 3.0]
    expected_101 = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101.0
    assert average_precision_coco101(recalls, precisions) == pytest.approx(expected_101)
    assert average_precision_area(recalls, precisions) == pytest.approx(0.5 + 0.5 * 2.0 / 3.0)
    # Recall that stops short reads 0 past the last detection.
    assert average_precision_coco101([0.5], [1.0]) == pytest.approx(51.0 / 101.0)
    assert average_precision_coco101([], []) == 0.0


def test_coco101_caps_detections_per_image_and_class():
    """With 101 same-class dets on one image, pycocotools keeps the top 100 by score."""
    gt = RBox(1000, 1000, 10, 10, 0)
    far = [
        Detection(
            rbox=RBox(20 * (i + 1), 20, 10, 10, 0),
            score=1.0 - 0.001 * i,
            class_id=0,
            class_name="ship",
            image_id="img",
        )
        for i in range(100)
    ]
    hit = Detection(rbox=gt, score=0.01, class_id=0, class_name="ship", image_id="img")
    ground_truths = {"img": [GroundTruth(rbox=gt, class_id=0, class_name="ship", image_id="img")]}

    all_point, _, _ = compute_oriented_map(
        {"img": far + [hit]}, ground_truths, iou_thresholds=[0.5], show_progress=False
    )
    coco, _, _ = compute_oriented_map(
        {"img": far + [hit]},
        ground_truths,
        iou_thresholds=[0.5],
        show_progress=False,
        ap_interpolation="coco101",
    )
    # All-point keeps the 101st det, which is the only true positive (precision 1/101).
    assert all_point == pytest.approx(1.0 / 101.0)
    # coco101 drops it: zero recall.
    assert coco == 0.0


def test_coco101_leaves_classes_without_gt_out_of_the_mean():
    """A detection on a class with no GT is AP 0 for all_point; pycocotools skips the class."""
    box = RBox(100, 100, 20, 20, 0)
    dets = {
        "img": [
            Detection(rbox=box, score=0.9, class_id=0, class_name="ship", image_id="img"),
            Detection(rbox=RBox(300, 300, 20, 20, 0), score=0.8, class_id=1, class_name="ghost", image_id="img"),
        ]
    }
    gts = {"img": [GroundTruth(rbox=box, class_id=0, class_name="ship", image_id="img")]}
    all_point, aps_all, _ = compute_oriented_map(dets, gts, iou_thresholds=[0.5], show_progress=False)
    coco, aps_coco, metrics_coco = compute_oriented_map(
        dets, gts, iou_thresholds=[0.5], show_progress=False, ap_interpolation="coco101"
    )
    assert all_point == pytest.approx(0.5)
    assert aps_all["ghost"] == 0.0
    assert coco == pytest.approx(1.0)
    assert "ghost" not in aps_coco
    assert metrics_coco["ghost"].num_gts == 0


def test_ap_interpolation_rejects_unknown_value():
    with pytest.raises(ValueError):
        APCalculator(iou_threshold=0.5, ap_interpolation="voc07")


def test_eval_iou_is_exact_for_axis_aligned_boxes():
    """The eval IoU helper is closed-form when every angle is a multiple of 90°."""
    torch = pytest.importorskip("torch")
    from oriented_det.ops.gpu_ops import eval_oriented_box_iou_gpu, oriented_box_iou_gpu
    from oriented_det.ops.iou import rbox_iou

    rng = torch.Generator().manual_seed(0)
    n = 24
    xy = torch.rand((n, 2), generator=rng) * 200 + 50
    wh = torch.rand((n, 2), generator=rng) * 60 + 5
    a = torch.cat([xy, wh, torch.zeros((n, 1))], dim=1)
    # Same boxes nudged by a fraction of their size so IoU sits in the 0.6–0.95 band.
    b = a.clone()
    b[:, :2] += (torch.rand((n, 2), generator=rng) - 0.5) * 0.2 * wh
    # Half of them written as the 90° transpose, which is the same rectangle.
    b[::2, 2], b[::2, 3] = b[::2, 3].clone(), b[::2, 2].clone()
    b[::2, 4] = math.pi / 2

    got = eval_oriented_box_iou_gpu(a, b)
    expected = torch.tensor(
        [
            [
                rbox_iou(
                    RBox(*a[i].tolist()),
                    RBox(*b[j].tolist()),
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-5)
    # The sampling path is not exact on the same boxes.
    sampled = oriented_box_iou_gpu(a, b)
    assert (sampled - expected).abs().max() > 1e-3
    # A rotated box falls back to sampling.
    c = b.clone()
    c[0, 4] = 0.3
    assert torch.allclose(eval_oriented_box_iou_gpu(a, c), oriented_box_iou_gpu(a, c))


def test_coco101_matches_pycocotools_on_synthetic_set():
    """mAP 0.50:0.95 on 40 images × 3 classes equals COCOeval stats[0] to 1e-6."""
    pytest.importorskip("torch")
    pycocotools_coco = pytest.importorskip("pycocotools.coco")
    pycocotools_eval = pytest.importorskip("pycocotools.cocoeval")
    import contextlib
    import io
    import random

    rnd = random.Random(7)
    class_names = ["a", "b", "c"]
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    gts: dict = {}
    dets: dict = {}
    coco_images, coco_anns, coco_dets = [], [], []
    ann_id = 1
    for img_i in range(40):
        img_id = f"img{img_i}"
        coco_images.append({"id": img_i, "width": 640, "height": 640})
        gts[img_id] = []
        dets[img_id] = []
        for _ in range(rnd.randint(1, 6)):
            cls = rnd.randrange(3)
            w, h = rnd.uniform(10, 120), rnd.uniform(10, 120)
            cx, cy = rnd.uniform(70, 570), rnd.uniform(70, 570)
            gts[img_id].append(
                GroundTruth(rbox=RBox(cx, cy, w, h, 0), class_id=cls, class_name=class_names[cls], image_id=img_id)
            )
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_i,
                    "category_id": cls + 1,
                    "bbox": [cx - w / 2, cy - h / 2, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            # Zero to two detections per GT, jittered; some on the wrong class.
            for _ in range(rnd.randint(0, 2)):
                jw, jh = w * rnd.uniform(0.7, 1.3), h * rnd.uniform(0.7, 1.3)
                jx, jy = cx + rnd.uniform(-0.3, 0.3) * w, cy + rnd.uniform(-0.3, 0.3) * h
                dcls = cls if rnd.random() < 0.85 else rnd.randrange(3)
                score = rnd.random()
                dets[img_id].append(
                    Detection(rbox=RBox(jx, jy, jw, jh, 0), score=score, class_id=dcls, class_name=class_names[dcls], image_id=img_id)
                )
                coco_dets.append(
                    {"image_id": img_i, "category_id": dcls + 1, "bbox": [jx - jw / 2, jy - jh / 2, jw, jh], "score": score}
                )
        # A few background false positives.
        for _ in range(rnd.randint(0, 3)):
            dcls = rnd.randrange(3)
            w, h = rnd.uniform(10, 80), rnd.uniform(10, 80)
            cx, cy = rnd.uniform(50, 590), rnd.uniform(50, 590)
            score = rnd.random()
            dets[img_id].append(
                Detection(rbox=RBox(cx, cy, w, h, 0), score=score, class_id=dcls, class_name=class_names[dcls], image_id=img_id)
            )
            coco_dets.append({"image_id": img_i, "category_id": dcls + 1, "bbox": [cx - w / 2, cy - h / 2, w, h], "score": score})

    ours, _, _ = compute_oriented_map(
        dets, gts, class_names=class_names, iou_thresholds=thresholds,
        show_progress=False, ap_interpolation="coco101",
    )

    coco_gt = pycocotools_coco.COCO()
    coco_gt.dataset = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_dets)
        ev = pycocotools_eval.COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert ours == pytest.approx(float(ev.stats[0]), abs=1e-6)


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

