"""Tests for strict TrainingExperimentConfig loading and config/schema alignment."""

import json
from pathlib import Path
from dataclasses import fields

import pytest

from oriented_det.train.config import (
    TrainingExperimentConfig,
    AugmentationConfig,
    CheckpointConfig,
    DataLoaderConfig,
    DatasetConfig,
    EvaluationConfig,
    ProductionConfig,
    LossConfig,
    ModelConfig,
    PreprocessingConfig,
    TensorboardConfig,
    TrainingConfig,
    resolve_inference_score_threshold,
    resolve_inference_sliding_window_overlap_pixels,
    effective_eval_metric_thresholds,
    apply_inference_config_to_model,
    _strict_section,
    _field_names,
)


def test_strict_section_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        _strict_section({"data_root": "/x", "typo": 1}, "dataset", DatasetConfig)


def test_strict_section_ignores_muted_keys():
    kwargs = _strict_section(
        {"data_root": "/x", "_muted_data_root": "/y", "_muted_typo": 1},
        "dataset",
        DatasetConfig,
    )

    assert kwargs == {"data_root": "/x"}


def test_load_rejects_model_anchor_angles_in_json(tmp_path: Path):
    """RPN reference angles are fixed in code (horizontal priors); not a config knob."""
    p = tmp_path / "bad_anchor_angles.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "model": {"anchor_angles": [0.0], "backbone": "resnet18"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown key\\(s\\) in config section 'model'"):
        TrainingExperimentConfig.load(p)


def test_production_section_load_and_strict(tmp_path: Path):
    p = tmp_path / "inf.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "evaluation": {"score_threshold": 0.1, "iou_threshold": 0.4},
                "production": {"overlap_pixels": 192, "score_threshold": 0.35},
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert cfg.production.overlap_pixels == 192
    assert cfg.production.score_threshold == 0.35
    assert resolve_inference_score_threshold(cfg) == 0.35
    assert resolve_inference_sliding_window_overlap_pixels(cfg) == 192
    sc, pc, iou = effective_eval_metric_thresholds(cfg)
    assert sc == 0.35
    assert pc is None
    assert iou == 0.4

    p_default = tmp_path / "inf_default_overlap.json"
    p_default.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "production": {},
            }
        ),
        encoding="utf-8",
    )
    cfg_default = TrainingExperimentConfig.load(p_default)
    assert resolve_inference_sliding_window_overlap_pixels(cfg_default) == 200

    bad = tmp_path / "bad_inf.json"
    bad.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "production": {"typo_key": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown key"):
        TrainingExperimentConfig.load(bad)


def test_effective_eval_per_class_merges_with_evaluation(tmp_path: Path):
    p = tmp_path / "merge_pc.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "evaluation": {
                    "per_class_score_threshold": {"plane": 0.2, "ship": 0.3},
                },
                "production": {"per_class_score_threshold": {"ship": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    sc, pc, iou = effective_eval_metric_thresholds(cfg)
    assert sc == cfg.evaluation.score_threshold
    assert pc is not None
    assert pc["plane"] == 0.2
    assert pc["ship"] == 0.5


def test_apply_inference_config_to_model_sets_existing_attrs():
    class _M:
        inference_pre_nms_score_threshold = 0.05
        rpn_pre_nms_top_n = 2000
        final_nms_iou_threshold = 0.1

    m = _M()
    apply_inference_config_to_model(
        m,
        ProductionConfig(
            inference_pre_nms_score_threshold=0.12,
            rpn_pre_nms_top_n=4000,
            final_nms_iou_threshold=0.35,
        ),
    )
    assert m.inference_pre_nms_score_threshold == 0.12
    assert m.rpn_pre_nms_top_n == 4000
    assert m.final_nms_iou_threshold == 0.35


def test_load_rejects_unknown_root_key(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {
                    "data_root": str(tmp_path),
                    "format": "dota",
                },
                "bogus_root": 1,
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown key\\(s\\) at config root"):
        TrainingExperimentConfig.load(p)


def test_load_ignores_muted_root_and_section_keys(tmp_path: Path):
    p = tmp_path / "muted.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "_muted_model_type": "rotated_retinanet",
                "_muted_bogus_root": 1,
                "dataset": {
                    "data_root": str(tmp_path),
                    "_muted_data_root": "/unused",
                    "_muted_bogus_dataset": 2,
                },
                "training": {
                    "learning_rate": 0.01,
                    "_muted_learning_rate": 0.2,
                    "_muted_unknown_training": True,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = TrainingExperimentConfig.load(p)

    assert cfg.model_type == "rotated_faster_rcnn"
    assert cfg.dataset is not None
    assert cfg.dataset.data_root == tmp_path
    assert cfg.training.learning_rate == 0.01


def test_legacy_loss_none_serializes_as_cross_entropy(tmp_path: Path):
    p = tmp_path / "legacy_loss.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "model": {"roi_loss_type": "cross_entropy"},
                "loss": {"loss_type": "none"},
            }
        ),
        encoding="utf-8",
    )

    cfg = TrainingExperimentConfig.load(p)
    assert cfg.loss.loss_type == "cross_entropy"

    saved = tmp_path / "saved.json"
    cfg.save(saved)
    saved_cfg = json.loads(saved.read_text(encoding="utf-8"))
    assert saved_cfg["loss"]["loss_type"] == "cross_entropy"


def test_legacy_gpu_oriented_iou_samples_stripped(tmp_path: Path):
    p = tmp_path / "legacy_gpu_iou.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "model": {"gpu_oriented_iou_samples": 225},
            }
        ),
        encoding="utf-8",
    )

    cfg = TrainingExperimentConfig.load(p)
    assert not hasattr(cfg.model, "gpu_oriented_iou_samples")


def test_config_schema_fieldnames_align_with_dataclasses():
    """config.schema.json properties should cover TrainingExperimentConfig + nested dataclasses."""
    schema_path = Path(__file__).resolve().parents[1] / "configs" / "config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    root_props = set(schema.get("properties", {}))

    def nested_keys(name: str) -> set[str]:
        return set(schema["properties"][name]["properties"])

    # Root keys that exist only in saved runs / optional metadata
    allowed_extra_schema = {
        "_base_",
    }
    dataclass_root = {f.name for f in fields(TrainingExperimentConfig)}
    missing_in_schema = dataclass_root - root_props - allowed_extra_schema
    extra_in_schema = (
        root_props
        - dataclass_root
        - {"_base_"}
    )
    assert not missing_in_schema, f"Schema missing top-level keys: {missing_in_schema}"
    # Schema may document _base_ only; that is not a dataclass field.
    assert not (extra_in_schema - {"_base_"}), f"Schema has unknown top-level keys vs dataclass: {extra_in_schema}"

    expected_sections = {
        "dataset": DatasetConfig,
        "data_loader": DataLoaderConfig,
        "augmentation": AugmentationConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "production": ProductionConfig,
        "checkpoint": CheckpointConfig,
        "loss": LossConfig,
        "tensorboard": TensorboardConfig,
        "preprocessing": PreprocessingConfig,
    }
    for section, dcls in expected_sections.items():
        assert nested_keys(section) == _field_names(dcls), section


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dataset", "exclude_difficult"),
        ("model", "nms_threshold"),
        ("model", "nms_iou_schedule_epochs"),
        ("model", "nms_iou_schedule_values"),
        ("training", "lr_scheduler_step_size"),
    ],
)
def test_deleted_compatibility_keys_fail(tmp_path: Path, section: str, key: str):
    p = tmp_path / "legacy.json"
    cfg = {
        "model_type": "rotated_faster_rcnn",
        "dataset": {"data_root": str(tmp_path)},
        section: {key: True},
    }
    if section == "model":
        cfg[section][key] = 0.5
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        TrainingExperimentConfig.load(p)


def test_deleted_root_compatibility_key_fails(tmp_path: Path):
    p = tmp_path / "legacy_root.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {"data_root": str(tmp_path)},
                "enable_augmentation": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown key"):
        TrainingExperimentConfig.load(p)


def test_all_repo_json_configs_load():
    root = Path(__file__).resolve().parents[1] / "configs"
    for p in sorted(root.rglob("*.json")):
        if p.name == "config.schema.json":
            continue
        cfg = TrainingExperimentConfig.load(p)
        assert isinstance(cfg, TrainingExperimentConfig)
        if cfg.model is not None:
            # Rotated RetinaNet uses the same pre-NMS score field as R-CNN;
            # ensure it is a real attribute (regression: must not be dropped or alias-only).
            assert isinstance(cfg.model.inference_pre_nms_score_threshold, float)
