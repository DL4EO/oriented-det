"""Tests for HRSID COCO loader, train/test splits, and DOTA export."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from oriented_det.data import HRSIDDataset, build_split_dataset, export_hrsid_to_dota
from oriented_det.data.dota import DOTADataset
from oriented_det.data.hrsid import resolve_hrsid_imageset_split
from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig


def _write_mini_hrsid(root: Path) -> Path:
    images = root / "images"
    anns = root / "annotations"
    images.mkdir(parents=True)
    anns.mkdir(parents=True)
    Image.new("L", (80, 80), color=25).save(images / "train_a.png")
    Image.new("RGB", (80, 80), color=(5, 5, 5)).save(images / "test_a.png")
    train_payload = {
        "images": [{"id": 1, "file_name": "train_a.png", "width": 80, "height": 80}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]],
                "bbox": [10.0, 10.0, 40.0, 30.0],
                "iscrowd": 0,
                "area": 1200.0,
            }
        ],
        "categories": [{"id": 1, "name": "ship"}],
    }
    test_payload = {
        "images": [{"id": 2, "file_name": "test_a.png", "width": 80, "height": 80}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [
                    [8.0, 8.0, 24.0, 8.0, 40.0, 8.0, 40.0, 28.0, 24.0, 28.0, 8.0, 28.0]
                ],
                "bbox": [8.0, 8.0, 32.0, 20.0],
                "iscrowd": 0,
                "area": 640.0,
            }
        ],
        "categories": [{"id": 1, "name": "ship"}],
    }
    (anns / "train2017.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (anns / "test2017.json").write_text(json.dumps(test_payload), encoding="utf-8")
    return root


def test_hrsid_coco_splits(tmp_path: Path):
    root = _write_mini_hrsid(tmp_path / "HRSID")
    train = HRSIDDataset(root, split="train", difficult_strategy="keep")
    test = HRSIDDataset(root, split="test", difficult_strategy="keep")
    assert train.get_class_names() == ["ship"]
    assert len(train) == 1
    assert len(test) == 1
    assert train[0].image_path.stem == "train_a"
    assert test[0].image_path.stem == "test_a"
    assert len(train[0].annotations) == 1
    assert abs(train[0].annotations[0].rbox.area - 40.0 * 30.0) < 1e-2
    assert len(test[0].annotations) == 1


def test_hrsid_nested_root(tmp_path: Path):
    nested = tmp_path / "data"
    _write_mini_hrsid(nested / "HRSID")
    ds = HRSIDDataset(nested, split="train", difficult_strategy="keep")
    assert len(ds) == 1


def test_hrsid_filter_empty_gt(tmp_path: Path):
    root = _write_mini_hrsid(tmp_path / "HRSID")
    payload = json.loads((root / "annotations" / "train2017.json").read_text(encoding="utf-8"))
    payload["images"].append({"id": 9, "file_name": "empty.png", "width": 80, "height": 80})
    Image.new("RGB", (80, 80), color=(1, 1, 1)).save(root / "images" / "empty.png")
    (root / "annotations" / "train2017.json").write_text(json.dumps(payload), encoding="utf-8")
    full = HRSIDDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=False)
    filtered = HRSIDDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=True)
    assert len(full) == 2
    assert len(filtered) == 1
    assert filtered.empty_gt_filtered_count == 1


def test_resolve_hrsid_imageset_split_defaults():
    cfg = DatasetConfig(data_root=".", format="hrsid")
    assert resolve_hrsid_imageset_split(cfg, "train") == "train"
    assert resolve_hrsid_imageset_split(cfg, "val") == "test"
    cfg.val_split = "valsplit"
    assert resolve_hrsid_imageset_split(cfg, "val") == "valsplit"


def test_build_split_dataset_hrsid(tmp_path: Path):
    root = _write_mini_hrsid(tmp_path / "HRSID")
    cfg = DatasetConfig(
        data_root=root,
        format="hrsid",
        difficult_strategy="keep",
        filter_empty_gt=False,
    )
    train = build_split_dataset(cfg, "train")
    val = build_split_dataset(cfg, "val")
    assert len(train) == 1
    assert len(val) == 1


def test_export_hrsid_to_dota(tmp_path: Path):
    root = _write_mini_hrsid(tmp_path / "HRSID")
    out = tmp_path / "dota"
    counts = export_hrsid_to_dota(root, out, splits=("train", "test"), difficult_strategy="keep")
    assert counts["train"] == 1
    assert counts["test"] == 1
    with Image.open(out / "train" / "images" / "train_a.png") as img:
        assert img.mode == "RGB"
    dota = DOTADataset(
        root_dir=out / "test",
        split="test",
        label_dir=out / "test" / "labels",
        image_dir=out / "test" / "images",
        difficult_strategy="keep",
    )
    assert len(dota) == 1
    assert dota[0].annotations[0].class_name == "ship"


def test_hrsid_recipes_load():
    root = Path(__file__).resolve().parents[1]
    expected_hub = {
        "configs/oriented_rcnn/hrsid_le90_1x.json": "hf://oriented_rcnn_dota_le90_1x",
        "configs/rotated_faster_rcnn/hrsid_le90_1x.json": "hf://rotated_faster_rcnn_dota_le90_1x",
        "configs/rotated_fcos/hrsid_le90_1x.json": "hf://rotated_fcos_dota_le90_1x",
    }
    for rel, hub in expected_hub.items():
        cfg = TrainingExperimentConfig.load(root / rel)
        assert cfg.dataset.format == "hrsid"
        assert cfg.dataset.overlap == 0
        assert cfg.dataset.val_split == "test"
        assert cfg.dataset.test_split == "test"
        assert cfg.dataset.has_test_split()
        assert list(cfg.preprocessing.target_size) == [800, 800]
        assert cfg.preprocessing.resize_mode == "keep_ratio"
        assert cfg.checkpoint.load_from_checkpoint == hub
        assert cfg.training.num_epochs == 12
        assert cfg.production.overlap_pixels == 0
    fcos = TrainingExperimentConfig.load(root / "configs/rotated_fcos/hrsid_le90_1x.json")
    assert fcos.model.box_reg_loss_type == "riou"
    assert fcos.training.learning_rate == 0.0025
