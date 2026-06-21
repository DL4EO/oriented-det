"""Tests for multi-root DOTA tile directory resolution."""

from pathlib import Path
import tempfile

import pytest
from torch.utils.data import ConcatDataset

from oriented_det.data.dota import (
    build_dota_split_dataset,
    collect_dota_image_paths,
    collect_dota_split_image_paths,
    dota_label_path_for_image,
    resolve_dota_tile_roots,
)


def _write_tile(root: Path, stem: str, class_name: str = "plane") -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "images" / f"{stem}.png").write_bytes(b"")
    (root / "labels" / f"{stem}.txt").write_text(
        f"0, 0, 10, 0, 10, 10, 0, 10, {class_name}, 0\n",
        encoding="utf-8",
    )


def test_resolve_dota_tile_roots_plural_and_singular():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "train", Path(tmp) / "val"
        a.mkdir()
        b.mkdir()

        assert resolve_dota_tile_roots(tiles_dir=a) == [a]
        assert resolve_dota_tile_roots(tiles_dirs=[a, b]) == [a, b]

        with pytest.raises(ValueError, match="non-empty"):
            resolve_dota_tile_roots(tiles_dirs=[])

        with pytest.raises(ValueError, match="requires"):
            resolve_dota_tile_roots()


def test_build_dota_split_dataset_concat_and_single():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_root = root / "train"
        val_root = root / "val"
        _write_tile(train_root, "t1")
        _write_tile(train_root, "t2")
        _write_tile(val_root, "v1")

        single = build_dota_split_dataset([train_root], split="train")
        assert len(single) == 2

        merged = build_dota_split_dataset([train_root, val_root], split="train")
        assert isinstance(merged, ConcatDataset)
        assert len(merged) == 3

        paths = collect_dota_image_paths([train_root, val_root])
        assert len(paths) == 3
        assert paths[0].parent.name == "images"


def test_collect_dota_split_image_paths_filter_empty_gt():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tiles"
        _write_tile(root, "empty")
        _write_tile(root, "has_obj")
        (root / "labels" / "empty.txt").write_text("", encoding="utf-8")

        all_paths = collect_dota_split_image_paths(
            [root], split="train", filter_empty_gt=False
        )
        filtered = collect_dota_split_image_paths(
            [root], split="train", filter_empty_gt=True
        )
        assert len(all_paths) == 2
        assert len(filtered) == 1
        assert filtered[0].stem == "has_obj"


def test_dota_label_path_for_image():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tiles"
        _write_tile(root, "img001")
        image_path = root / "images" / "img001.png"
        label_path = dota_label_path_for_image(image_path)
        assert label_path == root / "labels" / "img001.txt"
        assert label_path.exists()
