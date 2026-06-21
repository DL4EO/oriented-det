"""Tests for evaluation score threshold helpers and val stats."""

import pytest

torch = pytest.importorskip("torch")

from oriented_det.train.config import (
    EvaluationConfig,
    TrainingExperimentConfig,
    config_use_exact_rotated_iou_for_map,
    config_use_exact_rotated_iou_for_final_map,
)
from oriented_det.train.utils import (
    effective_score_threshold_for_class_name,
    filter_detections_by_score_threshold,
    scores_labels_pass_threshold,
)
from oriented_det.train.engine import _compute_val_stats_from_dicts
from oriented_det.data.evaluation import Detection, GroundTruth
from oriented_det.geometry import RBox


class _Det:
    def __init__(self, score: float, class_name: str, class_id: int):
        self.score = score
        self.class_name = class_name
        self.class_id = class_id


def test_effective_score_threshold_for_class_name():
    assert effective_score_threshold_for_class_name("car", 0.5, None) == 0.5
    assert effective_score_threshold_for_class_name("car", 0.5, {"car": 0.3}) == 0.3
    assert effective_score_threshold_for_class_name("bus", 0.5, {"car": 0.3}) == 0.5
    assert effective_score_threshold_for_class_name("Car", 0.5, {"car": 0.3}) == 0.3


def test_filter_detections_by_score_threshold():
    dets = [
        _Det(0.9, "car", 1),
        _Det(0.4, "car", 1),
        _Det(0.6, "bus", 2),
    ]
    out = filter_detections_by_score_threshold(dets, 0.5, {"car": 0.3})
    assert len(out) == 3
    out2 = filter_detections_by_score_threshold(dets, 0.5, None)
    assert len(out2) == 2


def test_scores_labels_pass_threshold():
    scores = torch.tensor([0.9, 0.4, 0.55])
    labels = torch.tensor([1, 1, 2])
    id_to_class = {1: "car", 2: "bus"}
    mask = scores_labels_pass_threshold(
        scores, labels, 0.5, {"car": 0.35}, id_to_class
    )
    assert mask.tolist() == [True, True, True]


def test_compute_val_stats_extended_gt_metrics_off():
    r = RBox(10.0, 10.0, 4.0, 4.0, 0.0)
    d = Detection(rbox=r, score=0.9, class_id=1, class_name="a", image_id="i1")
    g = GroundTruth(rbox=r, class_id=1, class_name="a", difficult=0, image_id="i1")
    stats = _compute_val_stats_from_dicts(
        {"i1": [d]},
        {"i1": [g]},
        score_threshold=0.5,
        iou_threshold=0.5,
        id_to_class={1: "a"},
        per_class_score_threshold=None,
        extended_gt_metrics=False,
    )
    assert "log_only_gt_mean_best_iou_any" not in stats
    assert stats["total_ground_truths"] == 1


def test_compute_val_stats_extended_gt_metrics_on():
    r = RBox(10.0, 10.0, 4.0, 4.0, 0.0)
    d = Detection(rbox=r, score=0.9, class_id=1, class_name="a", image_id="i1")
    g = GroundTruth(rbox=r, class_id=1, class_name="a", difficult=0, image_id="i1")
    stats = _compute_val_stats_from_dicts(
        {"i1": [d]},
        {"i1": [g]},
        score_threshold=0.5,
        iou_threshold=0.5,
        id_to_class={1: "a"},
        per_class_score_threshold=None,
        extended_gt_metrics=True,
    )
    assert "log_only_gt_mean_best_iou_any" in stats
    assert stats["log_only_gt_count_zero_best_iou_any"] == 0


def test_compute_val_stats_extended_gt_zero_iou_missed():
    r_gt = RBox(10.0, 10.0, 4.0, 4.0, 0.0)
    r_det = RBox(100.0, 100.0, 4.0, 4.0, 0.0)
    d = Detection(rbox=r_det, score=0.9, class_id=1, class_name="a", image_id="i1")
    g = GroundTruth(rbox=r_gt, class_id=1, class_name="a", difficult=0, image_id="i1")
    stats = _compute_val_stats_from_dicts(
        {"i1": [d]},
        {"i1": [g]},
        score_threshold=0.5,
        iou_threshold=0.5,
        id_to_class={1: "a"},
        per_class_score_threshold=None,
        extended_gt_metrics=True,
    )
    assert stats["log_only_gt_count_zero_best_iou_any"] == 1
    assert stats["log_only_gt_rate_zero_best_iou_any"] == 1.0


def test_evaluation_use_exact_rotated_iou_default():
    cfg = EvaluationConfig()
    assert cfg.use_exact_rotated_iou is True


def test_config_use_exact_rotated_iou_for_map(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(
        '{"model_type":"rotated_retinanet","dataset":{"data_root":"/tmp"},'
        '"evaluation":{"use_exact_rotated_iou":false}}',
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert cfg.evaluation.use_exact_rotated_iou is False
    assert config_use_exact_rotated_iou_for_map(cfg) is False


def test_config_use_exact_rotated_iou_for_final_map_override(tmp_path):
    p = tmp_path / "cfg_final.json"
    p.write_text(
        '{"model_type":"rotated_retinanet","dataset":{"data_root":"/tmp"},'
        '"evaluation":{"use_exact_rotated_iou":false,'
        '"use_exact_rotated_iou_for_final_map":true}}',
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert config_use_exact_rotated_iou_for_map(cfg) is False
    assert config_use_exact_rotated_iou_for_final_map(cfg) is True


def test_config_use_exact_rotated_iou_for_final_map_fallback(tmp_path):
    p = tmp_path / "cfg_fb.json"
    p.write_text(
        '{"model_type":"rotated_retinanet","dataset":{"data_root":"/tmp"},'
        '"evaluation":{"use_exact_rotated_iou":false}}',
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert config_use_exact_rotated_iou_for_final_map(cfg) is False


def test_compute_val_stats_sampled_iou_path():
    r = RBox(10.0, 10.0, 4.0, 4.0, 0.0)
    d = Detection(rbox=r, score=0.9, class_id=1, class_name="a", image_id="i1")
    g = GroundTruth(rbox=r, class_id=1, class_name="a", difficult=0, image_id="i1")
    stats = _compute_val_stats_from_dicts(
        {"i1": [d]},
        {"i1": [g]},
        score_threshold=0.5,
        iou_threshold=0.5,
        id_to_class={1: "a"},
        per_class_score_threshold=None,
        extended_gt_metrics=False,
        use_exact_rotated_iou=False,
        device=torch.device("cpu"),
    )
    assert stats["total_correct"] == 1
    assert stats["gt_covered_post_eval_threshold"] == 1
