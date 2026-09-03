"""Class-tile oversampling: GT class presence, lookalike skip, hard-tile compose."""

import json
from pathlib import Path

import pytest

from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig
from tools.train import (
    _class_tile_match_indices,
    _compose_tile_oversample_weights,
    _count_class_tile_gt_matches,
    _expand_indices_by_weights,
    _resolve_class_tile_oversample_classes,
)


class _Ann:
    def __init__(self, class_name: str):
        self.class_name = class_name


class _Sample:
    def __init__(self, class_names: list[str]):
        self.annotations = [_Ann(name) for name in class_names]


class _AnnDataset:
    """Tiny fake train set: each item is a list of remaining GT class_name values."""

    def __init__(self, tiles: list[list[str]]):
        self._tiles = tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> _Sample:
        return _Sample(self._tiles[idx])


def _mixed_dataset() -> _AnnDataset:
    return _AnnDataset(
        [
            ["class-a", "class-b"],
            ["class-b", "class-b"],
            [],
            ["lookalike"],
            ["lookalike", "class-a"],
            ["class-a", "class-a"],
        ]
    )


def test_match_indices_mixed_empty_and_lookalike_only():
    dataset = _mixed_dataset()
    matched = _class_tile_match_indices(dataset, ["class-a"], min_count=1)
    assert matched == [0, 4, 5]


def test_lookalike_boxes_never_count_as_semantic_matches():
    dataset = _AnnDataset([["lookalike"], ["lookalike", "lookalike"]])
    assert _class_tile_match_indices(dataset, ["class-a"], min_count=1) == []
    assert _count_class_tile_gt_matches(dataset[0].annotations, {"lookalike"}) == 0


def test_min_count_requires_enough_target_boxes():
    dataset = _mixed_dataset()
    assert _class_tile_match_indices(dataset, ["class-a"], min_count=2) == [5]
    assert _class_tile_match_indices(dataset, ["class-b"], min_count=2) == [1]


def test_matching_is_case_sensitive_exact_class_name():
    dataset = _AnnDataset([["Class-A"], ["class-a"]])
    assert _class_tile_match_indices(dataset, ["class-a"], min_count=1) == [1]


def test_resolve_ignores_unknown_and_lookalike_names_with_one_warning():
    with pytest.warns(UserWarning, match="ignored unknown or lookalike"):
        kept, ignored = _resolve_class_tile_oversample_classes(
            ["class-a", "lookalike", "not-a-class", "class-a"],
            known_class_names=["class-a", "class-b"],
        )
    assert kept == ["class-a"]
    assert ignored == ["lookalike", "not-a-class"]


def test_resolve_empty_or_null_class_list_is_disabled():
    assert _resolve_class_tile_oversample_classes(None) == ([], [])
    assert _resolve_class_tile_oversample_classes([]) == ([], [])


def test_compose_multiplies_hard_and_class_weights():
    weights = _compose_tile_oversample_weights(
        4,
        hard_indices=[0, 1],
        hard_factor=2.0,
        class_indices=[1, 2],
        class_factor=3.0,
    )
    assert weights == [2.0, 6.0, 3.0, 1.0]


def test_compose_disabled_or_factor_one_matches_hard_tile_only():
    hard_only = _compose_tile_oversample_weights(
        3, hard_indices=[0], hard_factor=2.0
    )
    assert hard_only == [2.0, 1.0, 1.0]
    assert _compose_tile_oversample_weights(3) == [1.0, 1.0, 1.0]
    assert (
        _compose_tile_oversample_weights(
            3,
            hard_indices=[0],
            hard_factor=2.0,
            class_indices=[1],
            class_factor=1.0,
        )
        == hard_only
    )
    assert (
        _compose_tile_oversample_weights(
            3,
            hard_indices=[0],
            hard_factor=2.0,
            class_indices=[],
            class_factor=4.0,
        )
        == hard_only
    )


def test_expand_uses_composed_weight_copies():
    weights = _compose_tile_oversample_weights(
        3,
        hard_indices=[0],
        hard_factor=2.0,
        class_indices=[0, 1],
        class_factor=3.0,
    )
    assert weights == [6.0, 3.0, 1.0]
    assert _expand_indices_by_weights(weights) == [0, 1, 2, 0, 0, 0, 0, 0, 1, 1]


def test_class_tile_config_defaults_and_load(tmp_path: Path):
    defaults = DatasetConfig(data_root=tmp_path)
    assert defaults.class_tile_oversample_classes is None
    assert defaults.class_tile_oversample_factor == 1.0
    assert defaults.class_tile_oversample_min_count == 1

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_type": "oriented_rcnn",
                "dataset": {
                    "data_root": str(tmp_path),
                    "class_tile_oversample_classes": ["class-a", "class-b"],
                    "class_tile_oversample_factor": 3.0,
                    "class_tile_oversample_min_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(cfg_path)
    assert cfg.dataset.class_tile_oversample_classes == ["class-a", "class-b"]
    assert cfg.dataset.class_tile_oversample_factor == 3.0
    assert cfg.dataset.class_tile_oversample_min_count == 2
