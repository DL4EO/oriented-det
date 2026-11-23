"""production.output_classes post-NMS allowlist (main / deploy / export)."""

from __future__ import annotations

from oriented_det.runtime.inference import (
    filter_detections_by_output_classes,
    normalize_output_classes,
)
from oriented_det.train.config import (
    ProductionConfig,
    TrainingExperimentConfig,
    resolve_output_classes,
)


def test_normalize_output_classes_strips_and_none() -> None:
    assert normalize_output_classes(None) is None
    assert normalize_output_classes(["drone", " ", "shahed"]) == ["drone", "shahed"]
    assert normalize_output_classes([]) == []


def test_filter_detections_by_output_classes_case_insensitive() -> None:
    dets = [
        {"label": 1, "score": 0.9},
        {"label": 2, "score": 0.8},
        {"label": 3, "score": 0.7},
    ]
    names = ["aircraft", "drone", "helicopter"]
    out = filter_detections_by_output_classes(dets, ["Drone", "AIRCRAFT"], names)
    assert [d["label"] for d in out] == [1, 2]


def test_filter_detections_by_output_classes_none_keeps_all() -> None:
    dets = [{"label": 1, "score": 0.9}, {"label": 2, "score": 0.8}]
    assert filter_detections_by_output_classes(dets, None, ["a", "b"]) == dets


def test_filter_detections_by_output_classes_empty_keeps_nothing() -> None:
    dets = [{"label": 1, "score": 0.9}]
    assert filter_detections_by_output_classes(dets, [], ["a"]) == []


def test_resolve_output_classes_from_production(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(
        '{"model_type":"rotated_fcos","dataset":{"data_root":"/tmp"},'
        '"production":{"output_classes":["drone","shahed"]}}',
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert resolve_output_classes(cfg) == ["drone", "shahed"]


def test_resolve_output_classes_null() -> None:
    cfg = TrainingExperimentConfig(
        model_type="rotated_fcos",
        production=ProductionConfig(output_classes=None),
    )
    # Minimal config may lack dataset; only exercise resolve helper.
    assert resolve_output_classes(cfg) is None
