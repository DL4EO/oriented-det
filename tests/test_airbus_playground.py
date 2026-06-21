"""Tests for Airbus Playground CSV generation and loader."""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from datetime import datetime, timezone

from oriented_det.data.airbus_playground import (
    AirbusPlaygroundCSVDataset,
    dated_playground_csv_filenames,
    detect_airbus_split_csv_format,
    generate_airbus_playground_csvs,
    resolve_playground_csv_filenames,
    timestamped_annotations_filename,
    timestamped_split_filename,
)


def test_dated_playground_csv_filenames_use_matching_utc_stamp():
    when = datetime(2026, 5, 16, 15, 30, 45, tzinfo=timezone.utc)
    assert dated_playground_csv_filenames(when) == (
        "annotations_20260516.csv",
        "split_20260516.csv",
    )
    assert timestamped_split_filename(when) == "split_20260516.csv"
    assert timestamped_annotations_filename(when) == "annotations_20260516.csv"


def test_resolve_playground_csv_filenames_defaults_to_dated_pair():
    when = datetime(2026, 5, 16, tzinfo=timezone.utc)
    assert resolve_playground_csv_filenames(None, None, when) == (
        "annotations_20260516.csv",
        "split_20260516.csv",
    )


def test_resolve_playground_csv_filenames_infers_pair_from_one_dated_name():
    when = datetime(2026, 5, 16, tzinfo=timezone.utc)
    assert resolve_playground_csv_filenames(None, "split_20260516.csv", when) == (
        "annotations_20260516.csv",
        "split_20260516.csv",
    )
    assert resolve_playground_csv_filenames("annotations_20260516.csv", None, when) == (
        "annotations_20260516.csv",
        "split_20260516.csv",
    )
    assert resolve_playground_csv_filenames("annotations.csv", None, when) == (
        "annotations.csv",
        "split.csv",
    )


def _write_tile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(path)


def _write_label(path: Path, *, tag: str = "car", with_object: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    if with_object:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [20, 10], [20, 18], [10, 18], [10, 10]]],
                },
                "properties": {"tags": [tag]},
            }
        )
    # Mask feature should be ignored by generator.
    features.append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [63, 0], [63, 63], [0, 63], [0, 0]]],
            },
            "properties": {"mask": True},
        }
    )
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_mock_export(root: Path) -> None:
    dataset_id = "dataset-A"
    zone_id = "zone-1"
    image_ids = ["image-1", "image-2"]
    tile_ids = ["tile-a", "tile-b"]

    for image_id in image_ids:
        for tile_id in tile_ids:
            tile_path = root / dataset_id / "samples" / zone_id / image_id / f"{tile_id}.jpg"
            _write_tile(tile_path)
            label_path = root / dataset_id / "labels" / zone_id / f"{tile_id}.json"
            _write_label(label_path, with_object=True)

    # Add one empty tile (label file has mask only).
    empty_tile = root / dataset_id / "samples" / zone_id / "image-empty" / "tile-empty.jpg"
    _write_tile(empty_tile)
    empty_label = root / dataset_id / "labels" / zone_id / "tile-empty.json"
    _write_label(empty_label, with_object=False)


def test_generate_csv_respects_image_group_split(tmp_path: Path):
    pytest.importorskip("shapely")
    _build_mock_export(tmp_path)

    annotations_path, split_path = generate_airbus_playground_csvs(
        tmp_path, num_splits=4, seed=7
    )

    assert annotations_path.exists()
    assert split_path.exists()

    split_by_image = {}
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "split.csv should contain rows"

    for row in rows:
        key = (row["dataset_id"], row["zone_id"], row["image_id"])
        split_by_image.setdefault(key, set()).add(row["split"])

    # All tiles from a single image group must map to one fold id only.
    for split_set in split_by_image.values():
        assert len(split_set) == 1
        (fold_str,) = tuple(split_set)
        int(fold_str)  # numeric fold column


def test_csv_loader_builds_samples_for_train_and_val(tmp_path: Path):
    pytest.importorskip("shapely")
    _build_mock_export(tmp_path)
    generate_airbus_playground_csvs(tmp_path, num_splits=2, seed=11)

    train_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="train",
        annotations_file="annotations.csv",
        split_file="split.csv",
        val_split_id=0,
    )
    val_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="val",
        annotations_file="annotations.csv",
        split_file="split.csv",
        val_split_id=0,
    )

    assert len(train_ds) + len(val_ds) == 5  # 4 labeled + 1 empty tile
    all_classes = set(train_ds.get_class_names()) | set(val_ds.get_class_names())
    assert all_classes == {"car"}

    # Ensure samples are valid DOTASample-compatible records.
    sample = train_ds[0]
    assert sample.image_path.exists()
    assert sample.width == 64
    assert sample.height == 64

    # Empty tiles should still be present via split.csv source.
    assert any(len(s.annotations) == 0 for s in list(train_ds) + list(val_ds))


def test_loader_val_split_id_rotates_validation_fold(tmp_path: Path):
    pytest.importorskip("shapely")
    _build_mock_export(tmp_path)
    generate_airbus_playground_csvs(tmp_path, num_splits=5, seed=0)

    with (tmp_path / "split.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert detect_airbus_split_csv_format([r["split"] for r in rows]) == "fold_ids"

    for vid in (0, 1, 2):
        n_csv = sum(int(r["split"]) == vid for r in rows)
        val_ds = AirbusPlaygroundCSVDataset(
            data_root=tmp_path,
            split="val",
            annotations_file="annotations.csv",
            split_file="split.csv",
            val_split_id=vid,
        )
        assert len(val_ds) == n_csv


def test_loader_legacy_train_val_split_csv_ignores_val_split_id(tmp_path: Path):
    pytest.importorskip("shapely")
    _build_mock_export(tmp_path)
    _, split_path = generate_airbus_playground_csvs(tmp_path, num_splits=2, seed=1)
    legacy_rows = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            fold = int(row["split"])
            row["split"] = "val" if fold == 0 else "train"
            legacy_rows.append(row)
    with split_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(legacy_rows)

    train_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="train",
        val_split_id=999,
        annotations_file="annotations.csv",
        split_file="split.csv",
    )
    val_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="val",
        val_split_id=999,
        annotations_file="annotations.csv",
        split_file="split.csv",
    )
    assert len(train_ds) + len(val_ds) == 5


def test_detect_split_format_errors_on_mixed_values() -> None:
    with pytest.raises(ValueError, match="expected integer fold id"):
        detect_airbus_split_csv_format(["0", "train"])


def test_generate_csv_with_ignore_map_and_stats(tmp_path: Path):
    pytest.importorskip("shapely")
    dataset_id = "dataset-B"
    zone_id = "zone-2"
    image_id = "image-1"
    for tile_id, label in [("tile-1", "car"), ("tile-2", "taxi"), ("tile-3", "Confuser")]:
        _write_tile(tmp_path / dataset_id / "samples" / zone_id / image_id / f"{tile_id}.jpg")
        _write_label(tmp_path / dataset_id / "labels" / zone_id / f"{tile_id}.json", tag=label, with_object=True)

    ann_path, split_path, stats = generate_airbus_playground_csvs(
        tmp_path,
        ignore_labels=["Confuser"],
        map_labels={"taxi": "car"},
        include_stats=True,
    )
    assert ann_path.exists()
    assert split_path.exists()

    with ann_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = [r["class_name"] for r in rows]
    assert "Confuser" not in labels
    assert labels.count("car") == 2

    assert stats["raw_objects"] == 3
    assert stats["ignored_objects"] == 1
    assert stats["mapped_objects"] == 1
    assert stats["kept_objects"] == 2
    assert stats["final_label_counts"]["car"] == 2
    assert stats["ignored_label_counts"]["Confuser"] == 1


def test_loader_applies_ignore_and_map_labels(tmp_path: Path):
    pytest.importorskip("shapely")
    dataset_id = "dataset-C"
    zone_id = "zone-3"
    image_id = "image-9"
    for tile_id, label in [("tile-1", "taxi"), ("tile-2", "Confuser")]:
        _write_tile(tmp_path / dataset_id / "samples" / zone_id / image_id / f"{tile_id}.jpg")
        _write_label(tmp_path / dataset_id / "labels" / zone_id / f"{tile_id}.json", tag=label, with_object=True)

    generate_airbus_playground_csvs(tmp_path, num_splits=2, seed=3)

    train_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="train",
        annotations_file="annotations.csv",
        split_file="split.csv",
        ignore_labels=["Confuser"],
        map_labels={"taxi": "car"},
    )
    val_ds = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="val",
        annotations_file="annotations.csv",
        split_file="split.csv",
        ignore_labels=["Confuser"],
        map_labels={"taxi": "car"},
    )

    classes = set(train_ds.get_class_names()) | set(val_ds.get_class_names())
    assert "Confuser" not in classes
    assert "car" in classes
