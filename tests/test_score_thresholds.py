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
    align = stats.get("log_only_gt_alignment_metrics")
    assert isinstance(align, dict)
    assert "a" in align.get("per_class", {})
    assert align["per_class"]["a"]["mean_best_iou_same_class"] > 0.99


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
    assert cfg.val_exact_rotated_iou is True


def test_config_use_exact_rotated_iou_for_map(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(
        '{"model_type":"rotated_retinanet","dataset":{"data_root":"/tmp"},'
        '"evaluation":{"use_exact_rotated_iou":false}}',
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert cfg.evaluation.val_exact_rotated_iou is False
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


def test_compute_map_test_uses_preds_score_not_train_val(tmp_path, monkeypatch):
    """Test mAP uses preds_score_threshold; final val mAP keeps the train-val floor."""
    from oriented_det.train.engine import train

    calls = []

    def _fake_evaluate(*_args, **kwargs):
        calls.append(float(kwargs["score_threshold"]))
        return {"mAP": 0.0}

    monkeypatch.setattr("oriented_det.train.engine.evaluate", _fake_evaluate)

    model = torch.nn.Linear(1, 1)
    ckpt = tmp_path / "best.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)

    class _Loader:
        def __len__(self):
            return 1

    class _Ckpt:
        best_checkpoint_path = ckpt

    train(
        model=model,
        train_loader=_Loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        num_epochs=0,
        val_loader=_Loader(),
        test_loader=_Loader(),
        checkpoint_manager=_Ckpt(),
        eval_score_threshold=0.3,
        eval_test_score_threshold=0.05,
        eval_compute_map_final=True,
        eval_compute_map_test=True,
    )
    assert calls == [0.3, 0.05]


def test_ossdd_recipe_test_map_floor_is_preds_default():
    from pathlib import Path

    from oriented_det.train.config import (
        effective_eval_metric_thresholds,
        resolve_preds_score_threshold,
    )

    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(
        root / "configs" / "rotated_rtmdet" / "ossdd_le90_vh_1x_s.json"
    )
    train_sc, _, _ = effective_eval_metric_thresholds(cfg)
    test_sc, src = resolve_preds_score_threshold(cfg, data_split="test")
    assert cfg.evaluation.test_map is True
    assert cfg.evaluation.test_score_threshold is None
    assert train_sc == 0.3
    assert test_sc == 0.05
    assert "0.05" in src


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
