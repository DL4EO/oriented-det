"""Tests for SSDD XML/COCO loaders, official 1/9 split, and DOTA export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from oriented_det.data import SSDDDataset, build_split_dataset, export_ssdd_to_dota
from oriented_det.data.dota import DOTADataset
from oriented_det.data.ssdd import (
    parse_ssdd_xml,
    resolve_ssdd_imageset_split,
    resolve_ssdd_root,
    ssdd_official_is_test,
)
from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig


def _ssdd_xml(*, name: str, cx: float, cy: float, w: float, h: float, angle_deg: float) -> str:
    return f"""<?xml version="1.0"?>
<annotation>
  <filename>{name}</filename>
  <object>
    <name>ship</name>
    <difficult>0</difficult>
    <robndbox>
      <cx>{cx}</cx>
      <cy>{cy}</cy>
      <w>{w}</w>
      <h>{h}</h>
      <angle>{angle_deg}</angle>
    </robndbox>
  </object>
</annotation>
"""


def _write_voc_ssdd(root: Path) -> Path:
    images = root / "JPEGImages"
    anns = root / "Annotations"
    images.mkdir(parents=True)
    anns.mkdir(parents=True)
    Image.new("RGB", (200, 120), color=(8, 8, 8)).save(images / "000001.jpg")
    Image.new("RGB", (200, 120), color=(12, 12, 12)).save(images / "000002.jpg")
    Image.new("RGB", (200, 120), color=(16, 16, 16)).save(images / "000003.jpg")
    (anns / "000001.xml").write_text(
        _ssdd_xml(name="000001.jpg", cx=100.0, cy=60.0, w=80.0, h=20.0, angle_deg=0.0),
        encoding="utf-8",
    )
    (anns / "000002.xml").write_text(
        _ssdd_xml(name="000002.jpg", cx=90.0, cy=50.0, w=40.0, h=12.0, angle_deg=45.0),
        encoding="utf-8",
    )
    (anns / "000003.xml").write_text(
        _ssdd_xml(name="000003.jpg", cx=80.0, cy=40.0, w=30.0, h=10.0, angle_deg=0.0),
        encoding="utf-8",
    )
    return root


def _write_coco_split_ssdd(root: Path) -> Path:
    for split, stem, color in (("train", "t2", (20, 20, 20)), ("test", "t1", (30, 30, 30))):
        img_dir = root / split / "images"
        img_dir.mkdir(parents=True)
        Image.new("L", (80, 60), color=color[0]).save(img_dir / f"{stem}.png")
        payload = {
            "images": [{"id": 1, "file_name": f"{stem}.png", "width": 80, "height": 60}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 30.0, 10.0, 30.0]],
                    "bbox": [10.0, 10.0, 40.0, 20.0],
                    "iscrowd": 0,
                    "area": 800.0,
                }
            ],
            "categories": [{"id": 1, "name": "ship"}],
        }
        (root / split / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_ssdd_official_is_test():
    assert ssdd_official_is_test("000001")
    assert ssdd_official_is_test("P0009")
    assert not ssdd_official_is_test("000002")
    assert not ssdd_official_is_test("000010")


def test_parse_ssdd_xml_rotated_bndbox_corners(tmp_path: Path):
    xml_path = tmp_path / "000002.xml"
    xml_path.write_text(
        """<annotation>
  <filename>000002.jpg</filename>
  <object>
    <name>ship</name>
    <difficult>0</difficult>
    <rotated_bndbox>
      <rotated_bbox_cx>235</rotated_bbox_cx>
      <rotated_bbox_cy>160</rotated_bbox_cy>
      <rotated_bbox_w>13</rotated_bbox_w>
      <rotated_bbox_h>49</rotated_bbox_h>
      <rotated_bbox_theta>85.03</rotated_bbox_theta>
      <x1>210</x1><y1>155</y1>
      <x2>260</x2><y2>151</y2>
      <x3>261</x3><y3>165</y3>
      <x4>211</x4><y4>169</y4>
    </rotated_bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )
    anns = parse_ssdd_xml(xml_path)
    assert len(anns) == 1
    assert abs(anns[0].rbox.cx - 235.5) < 1.0
    assert anns[0].rbox.area > 0


def test_parse_ssdd_xml_robndbox_degrees(tmp_path: Path):
    xml_path = tmp_path / "ship.xml"
    xml_path.write_text(
        _ssdd_xml(name="a.jpg", cx=100.0, cy=50.0, w=80.0, h=20.0, angle_deg=0.0),
        encoding="utf-8",
    )
    anns = parse_ssdd_xml(xml_path)
    assert len(anns) == 1
    assert anns[0].class_name == "ship"
    assert abs(anns[0].rbox.cx - 100.0) < 1e-3
    assert abs(anns[0].rbox.area - 80.0 * 20.0) < 1e-2


def test_ssdd_voc_official_split(tmp_path: Path):
    root = _write_voc_ssdd(tmp_path / "SSDD")
    train = SSDDDataset(root, split="train", difficult_strategy="keep")
    test = SSDDDataset(root, split="test", difficult_strategy="keep")
    assert train.get_class_names() == ["ship"]
    assert {s.image_path.stem for s in train} == {"000002", "000003"}
    assert {s.image_path.stem for s in test} == {"000001"}
    assert len(train[0].annotations) == 1


def test_ssdd_coco_split_folders(tmp_path: Path):
    root = _write_coco_split_ssdd(tmp_path / "SSDD_coco")
    train = SSDDDataset(root, split="train", difficult_strategy="keep")
    test = SSDDDataset(root, split="test", difficult_strategy="keep")
    assert len(train) == 1
    assert len(test) == 1
    assert train[0].image_path.stem == "t2"
    assert test[0].image_path.stem == "t1"
    assert len(train[0].annotations) == 1


def test_ssdd_filter_empty_gt(tmp_path: Path):
    root = _write_voc_ssdd(tmp_path / "SSDD")
    empty_xml = root / "Annotations" / "000002.xml"
    empty_xml.write_text(
        '<?xml version="1.0"?><annotation><filename>000002.jpg</filename></annotation>\n',
        encoding="utf-8",
    )
    full = SSDDDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=False)
    filtered = SSDDDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=True)
    assert len(full) == 2
    assert len(filtered) == 1
    assert filtered[0].image_path.stem == "000003"
    assert filtered.empty_gt_filtered_count == 1


def test_resolve_ssdd_imageset_split_defaults():
    cfg = DatasetConfig(data_root=".", format="ssdd")
    assert resolve_ssdd_imageset_split(cfg, "train") == "train"
    assert resolve_ssdd_imageset_split(cfg, "val") == "test"
    cfg.val_split = "inshore"
    assert resolve_ssdd_imageset_split(cfg, "val") == "inshore"


def test_build_split_dataset_ssdd(tmp_path: Path):
    root = _write_voc_ssdd(tmp_path / "SSDD")
    cfg = DatasetConfig(
        data_root=root,
        format="ssdd",
        difficult_strategy="keep",
        filter_empty_gt=False,
    )
    train = build_split_dataset(cfg, "train")
    val = build_split_dataset(cfg, "val")
    assert len(train) == 2
    assert len(val) == 1


def test_export_ssdd_to_dota(tmp_path: Path):
    root = _write_voc_ssdd(tmp_path / "SSDD")
    out = tmp_path / "dota"
    counts = export_ssdd_to_dota(root, out, splits=("train", "test"), difficult_strategy="keep")
    assert counts["train"] == 2
    assert counts["test"] == 1
    dota = DOTADataset(
        root_dir=out / "train",
        split="train",
        label_dir=out / "train" / "labels",
        image_dir=out / "train" / "images",
        difficult_strategy="keep",
    )
    assert len(dota) == 2


def test_resolve_ssdd_root_skips_hrsid_layout(tmp_path: Path):
    root = tmp_path / "HRSID_JPG"
    (root / "JPEGImages").mkdir(parents=True)
    anns = root / "annotations"
    anns.mkdir()
    (anns / "train2017.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_ssdd_root(root)


def test_python_m_oriented_det_cli_help():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "oriented_det.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ssdd-to-dota" in result.stdout
    assert "coco-to-dota" in result.stdout


def test_ssdd_recipe_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(root / "configs/rotated_faster_rcnn/ssdd_le90_1x.json")
    assert cfg.dataset.format == "ssdd"
    assert cfg.dataset.val_split == "test"
    assert cfg.dataset.test_split == "test"
    assert cfg.dataset.has_test_split()
    assert cfg.dataset.overlap == 0
    assert list(cfg.preprocessing.target_size) == [608, 608]
    assert cfg.preprocessing.resize_mode == "keep_ratio"
    assert cfg.checkpoint.load_from_checkpoint == "hf://rotated_faster_rcnn_dota_le90_1x"
    assert cfg.training.num_epochs == 12
    assert cfg.production.overlap_pixels == 0
