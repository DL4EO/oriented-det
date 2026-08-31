"""Hard-tile oversampling and drop_easy_empty_tiles helpers."""

import csv
import json
from pathlib import Path

import pytest
from torch.utils.data import Subset

from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig
from tools.train import (
    _dataset_index_to_image_stem,
    _drop_easy_empty_tiles,
    _load_tile_metrics_by_stem,
    _stems_vacuous_true_negatives_from_tile_csv,
)


class _StemDataset:
    """Minimal dataset with DOTA-style annotation paths for stem lookup."""

    def __init__(self, stems: list[str]):
        self._annotation_files = [Path(f"{stem}.txt") for stem in stems]

    def __len__(self) -> int:
        return len(self._annotation_files)

    def __getitem__(self, idx: int):
        return self._annotation_files[idx]


def _write_tile_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stems(dataset) -> list[str]:
    return [_dataset_index_to_image_stem(dataset, i) for i in range(len(dataset))]


@pytest.fixture
def tile_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tile_metrics.csv"
    _write_tile_csv(
        path,
        [
            {"image_id": "easy_empty", "tp": 0, "fp": 0, "fn": 0, "f1": 1.0},
            {"image_id": "legacy_easy", "tp": 0, "fp": 0, "fn": 0, "f1": 0.0},
            {"image_id": "hard_empty", "tp": 0, "fp": 5, "fn": 0, "f1": 0.0},
            {"image_id": "has_gt", "tp": 3, "fp": 1, "fn": 0, "f1": 0.75},
            {"image_id": "easy_gt", "tp": 2, "fp": 0, "fn": 0, "f1": 1.0},
        ],
    )
    return path


def test_vacuous_stems_include_legacy_f1_zero(tile_csv: Path):
    vacuous = _stems_vacuous_true_negatives_from_tile_csv(tile_csv)
    assert vacuous == {"easy_empty", "legacy_easy"}


def test_drop_easy_empty_keeps_hard_empty_and_missing_csv_row(tile_csv: Path):
    dataset = _StemDataset(
        ["easy_empty", "legacy_easy", "hard_empty", "has_gt", "easy_gt", "not_in_csv"]
    )
    reduced, n_dropped = _drop_easy_empty_tiles(dataset, tile_csv)
    assert n_dropped == 2
    assert isinstance(reduced, Subset)
    assert _stems(reduced) == ["hard_empty", "has_gt", "easy_gt", "not_in_csv"]


def test_hard_empty_is_oversample_candidate_after_drop(tile_csv: Path):
    dataset = _StemDataset(["easy_empty", "hard_empty", "has_gt", "easy_gt"])
    reduced, _ = _drop_easy_empty_tiles(dataset, tile_csv)
    stem_metrics = _load_tile_metrics_by_stem(tile_csv, "f1")
    vacuous = _stems_vacuous_true_negatives_from_tile_csv(tile_csv)
    hard = []
    for i in range(len(reduced)):
        stem = _dataset_index_to_image_stem(reduced, i)
        m = stem_metrics.get(stem)
        if m is not None and m < 0.8 and stem not in vacuous:
            hard.append(stem)
    assert hard == ["hard_empty", "has_gt"]


def test_drop_easy_empty_is_noop_when_no_vacuous_tiles(tmp_path: Path):
    path = tmp_path / "no_vacuous.csv"
    _write_tile_csv(
        path,
        [{"image_id": "hard_empty", "tp": 0, "fp": 2, "fn": 0, "f1": 0.0}],
    )
    dataset = _StemDataset(["hard_empty", "has_gt"])
    reduced, n_dropped = _drop_easy_empty_tiles(dataset, path)
    assert n_dropped == 0
    assert reduced is dataset


def test_drop_easy_empty_requires_count_columns(tmp_path: Path):
    path = tmp_path / "no_counts.csv"
    _write_tile_csv(path, [{"image_id": "a", "f1": 0.0}])
    with pytest.raises(ValueError, match="requires tp, fp, and fn columns"):
        _drop_easy_empty_tiles(_StemDataset(["a"]), path)


def test_drop_easy_empty_errors_if_every_tile_is_vacuous(tmp_path: Path):
    path = tmp_path / "all_vacuous.csv"
    _write_tile_csv(
        path,
        [{"image_id": "easy_empty", "tp": 0, "fp": 0, "fn": 0, "f1": 1.0}],
    )
    with pytest.raises(ValueError, match="removed every train tile"):
        _drop_easy_empty_tiles(_StemDataset(["easy_empty"]), path)


def test_drop_easy_empty_tiles_config_default_and_load(tmp_path: Path):
    assert DatasetConfig(data_root=tmp_path).drop_easy_empty_tiles is False
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_type": "oriented_rcnn",
                "dataset": {
                    "data_root": str(tmp_path),
                    "drop_easy_empty_tiles": True,
                    "tile_metrics_csv": "tile_metrics.csv",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(cfg_path)
    assert cfg.dataset.drop_easy_empty_tiles is True
    assert cfg.dataset.tile_metrics_csv == Path("tile_metrics.csv")
