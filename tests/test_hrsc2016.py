"""Tests for HRSC2016 XML loader, ImageSets splits, and DOTA export."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from oriented_det.data import (
    DOTADataset,
    HRSC2016Dataset,
    build_split_dataset,
    export_hrsc2016_to_dota,
)
from oriented_det.data.hrsc2016 import (
    hrsc_mbox_to_annotation,
    parse_hrsc2016_xml,
    read_hrsc2016_imageset,
    resolve_hrsc2016_imageset_split,
)
from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig


def _hrsc_xml(
    *,
    image_id: str,
    width: int,
    height: int,
    objects: list[tuple[float, float, float, float, float, int]],
) -> str:
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<HRSC_Image>",
        f"  <Img_ID>{image_id}</Img_ID>",
        f"  <Img_FileName>{image_id}</Img_FileName>",
        "  <Img_FileFmt>bmp</Img_FileFmt>",
        f"  <Img_SizeWidth>{width}</Img_SizeWidth>",
        f"  <Img_SizeHeight>{height}</Img_SizeHeight>",
        "  <HRSC_Objects>",
    ]
    for i, (cx, cy, w, h, ang, difficult) in enumerate(objects, start=1):
        parts.extend(
            [
                "    <HRSC_Object>",
                f"      <Object_ID>{i}</Object_ID>",
                "      <Class_ID>100000001</Class_ID>",
                f"      <difficult>{difficult}</difficult>",
                f"      <mbox_cx>{cx}</mbox_cx>",
                f"      <mbox_cy>{cy}</mbox_cy>",
                f"      <mbox_w>{w}</mbox_w>",
                f"      <mbox_h>{h}</mbox_h>",
                f"      <mbox_ang>{ang}</mbox_ang>",
                "    </HRSC_Object>",
            ]
        )
    parts.extend(["  </HRSC_Objects>", "</HRSC_Image>", ""])
    return "\n".join(parts)


def _write_mini_hrsc(root: Path) -> Path:
    images = root / "FullDataSet" / "AllImages"
    anns = root / "FullDataSet" / "Annotations"
    splits = root / "ImageSets"
    images.mkdir(parents=True)
    anns.mkdir(parents=True)
    splits.mkdir(parents=True)

    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(images / "100000001.bmp")
    Image.new("RGB", (200, 100), color=(30, 50, 70)).save(images / "100000002.bmp")
    Image.new("RGB", (200, 100), color=(40, 60, 80)).save(images / "100000003.bmp")

    (anns / "100000001.xml").write_text(
        _hrsc_xml(
            image_id="100000001",
            width=200,
            height=100,
            objects=[(100.0, 50.0, 80.0, 20.0, 0.0, 0)],
        ),
        encoding="utf-8",
    )
    (anns / "100000002.xml").write_text(
        _hrsc_xml(
            image_id="100000002",
            width=200,
            height=100,
            objects=[(120.0, 40.0, 60.0, 16.0, math.pi / 4, 1)],
        ),
        encoding="utf-8",
    )
    (anns / "100000003.xml").write_text(
        _hrsc_xml(image_id="100000003", width=200, height=100, objects=[]),
        encoding="utf-8",
    )

    (splits / "train.txt").write_text("100000001\n100000003\n", encoding="utf-8")
    (splits / "val.txt").write_text("100000002\n", encoding="utf-8")
    (splits / "test.txt").write_text("100000002\n", encoding="utf-8")
    return root


def test_hrsc_mbox_to_annotation_horizontal():
    ann = hrsc_mbox_to_annotation(100.0, 50.0, 80.0, 20.0, 0.0, difficult=0)
    assert ann.class_name == "ship"
    assert ann.difficult == 0
    assert abs(ann.rbox.cx - 100.0) < 1e-3
    assert abs(ann.rbox.cy - 50.0) < 1e-3
    assert ann.rbox.area == 80.0 * 20.0


def test_parse_hrsc2016_xml(tmp_path: Path):
    xml_path = tmp_path / "ship.xml"
    xml_path.write_text(
        _hrsc_xml(
            image_id="1",
            width=200,
            height=100,
            objects=[(100.0, 50.0, 80.0, 20.0, 0.0, 0)],
        ),
        encoding="utf-8",
    )
    width, height, anns = parse_hrsc2016_xml(xml_path)
    assert width == 200
    assert height == 100
    assert len(anns) == 1
    assert anns[0].class_name == "ship"


def test_hrsc2016_dataset_splits_and_difficult(tmp_path: Path):
    root = _write_mini_hrsc(tmp_path / "HRSC2016")
    train = HRSC2016Dataset(root, split="train", difficult_strategy="keep")
    assert len(train) == 2
    assert train.get_class_names() == ["ship"]
    sample = train[0]
    assert sample.image_path.stem == "100000001"
    assert len(sample.annotations) == 1

    dropped = HRSC2016Dataset(root, split="val", difficult_strategy="drop")
    assert len(dropped) == 1
    assert len(dropped[0].annotations) == 0

    kept = HRSC2016Dataset(root, split="val", difficult_strategy="keep")
    assert len(kept[0].annotations) == 1
    assert kept[0].annotations[0].difficult == 1


def test_hrsc2016_filter_empty_gt(tmp_path: Path):
    root = _write_mini_hrsc(tmp_path / "HRSC2016")
    full = HRSC2016Dataset(root, split="train", difficult_strategy="keep", filter_empty_gt=False)
    filtered = HRSC2016Dataset(root, split="train", difficult_strategy="keep", filter_empty_gt=True)
    assert len(full) == 2
    assert len(filtered) == 1
    assert filtered[0].image_path.stem == "100000001"
    assert filtered.empty_gt_filtered_count == 1


def test_hrsc2016_trainval_fallback_without_file(tmp_path: Path):
    root = _write_mini_hrsc(tmp_path / "HRSC2016")
    ids = read_hrsc2016_imageset(root, "trainval")
    assert ids == ["100000001", "100000003", "100000002"]


def test_hrsc2016_nested_root(tmp_path: Path):
    nested = tmp_path / "data"
    _write_mini_hrsc(nested / "HRSC2016")
    ds = HRSC2016Dataset(nested, split="test", difficult_strategy="keep")
    assert len(ds) == 1


def test_resolve_hrsc2016_imageset_split_defaults():
    cfg = DatasetConfig(data_root=".", format="hrsc2016")
    assert resolve_hrsc2016_imageset_split(cfg, "train") == "trainval"
    assert resolve_hrsc2016_imageset_split(cfg, "val") == "test"
    cfg.train_split = "train"
    cfg.val_split = "val"
    assert resolve_hrsc2016_imageset_split(cfg, "train") == "train"
    assert resolve_hrsc2016_imageset_split(cfg, "val") == "val"


def test_build_split_dataset_hrsc2016(tmp_path: Path):
    root = _write_mini_hrsc(tmp_path / "HRSC2016")
    cfg = DatasetConfig(
        data_root=root,
        format="hrsc2016",
        train_split="train",
        val_split="test",
        difficult_strategy="keep",
        filter_empty_gt=True,
    )
    train = build_split_dataset(cfg, "train")
    val = build_split_dataset(cfg, "val")
    assert len(train) == 1
    assert len(val) == 1
    assert train.get_class_names() == ["ship"]


def test_export_hrsc2016_to_dota(tmp_path: Path):
    root = _write_mini_hrsc(tmp_path / "HRSC2016")
    out = tmp_path / "dota"
    counts = export_hrsc2016_to_dota(root, out, splits=("train",), difficult_strategy="keep")
    assert counts["train"] == 2
    dota = DOTADataset(
        root_dir=out / "train",
        split="train",
        label_dir=out / "train" / "labels",
        image_dir=out / "train" / "images",
        difficult_strategy="keep",
    )
    assert len(dota) == 2
    named = {s.image_path.stem: s for s in dota}
    assert named["100000001"].annotations[0].class_name == "ship"
    assert (out / "train" / "images" / "100000001.png").is_file()


def test_hrsc2016_recipes_load():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "configs/oriented_rcnn/hrsc2016_le90_1x.json",
        "configs/oriented_rcnn/hrsc2016_le90_3x.json",
        "configs/rotated_faster_rcnn/hrsc2016_le90_1x.json",
        "configs/rotated_faster_rcnn/hrsc2016_le90_3x.json",
        "configs/rotated_fcos/hrsc2016_le90_1x.json",
        "configs/rotated_fcos/hrsc2016_le90_3x.json",
    ):
        cfg = TrainingExperimentConfig.load(root / rel)
        assert cfg.dataset.format == "hrsc2016"
        assert cfg.dataset.train_split == "trainval"
        assert cfg.dataset.val_split == "test"
        assert list(cfg.preprocessing.target_size) == [800, 800]
    one_x = TrainingExperimentConfig.load(root / "configs/oriented_rcnn/hrsc2016_le90_1x.json")
    three_x = TrainingExperimentConfig.load(root / "configs/oriented_rcnn/hrsc2016_le90_3x.json")
    frcnn_1x = TrainingExperimentConfig.load(root / "configs/rotated_faster_rcnn/hrsc2016_le90_1x.json")
    frcnn_3x = TrainingExperimentConfig.load(root / "configs/rotated_faster_rcnn/hrsc2016_le90_3x.json")
    fcos_1x = TrainingExperimentConfig.load(root / "configs/rotated_fcos/hrsc2016_le90_1x.json")
    fcos_3x = TrainingExperimentConfig.load(root / "configs/rotated_fcos/hrsc2016_le90_3x.json")
    for cfg in (one_x, three_x, frcnn_1x, frcnn_3x, fcos_1x, fcos_3x):
        assert cfg.preprocessing.resize_mode == "keep_ratio"
        assert cfg.preprocessing.pad_size_divisor == 32
    assert one_x.preprocessing.enable_random_rotate is False
    assert frcnn_1x.preprocessing.enable_random_rotate is False
    for cfg in (three_x, frcnn_3x, fcos_1x, fcos_3x):
        assert cfg.preprocessing.enable_random_rotate is True
        assert cfg.preprocessing.random_rotate_prob == 0.5
        assert cfg.preprocessing.random_rotate_angle_range == 20
    for cfg in (one_x, three_x, frcnn_1x, frcnn_3x, fcos_1x, fcos_3x):
        assert cfg.model.final_nms_iou_threshold == 0.1
        assert cfg.production.final_nms_iou_threshold == 0.3
        assert cfg.evaluation.final_nms_iou_threshold == 0.1
    for cfg in (one_x, three_x):
        assert cfg.model.roi_box_reg_main_loss_type == "smooth_l1"
        assert cfg.model.roi_box_reg_aux_loss_type == "probiou"
        assert cfg.model.roi_box_reg_aux_weight == 0.1
        assert cfg.production.max_detections_per_image == 2000
    for cfg in (frcnn_1x, frcnn_3x):
        assert cfg.model_type == "rotated_faster_rcnn"
        assert cfg.model.roi_box_reg_main_loss_type == "probiou"
        assert cfg.model.roi_box_reg_aux_loss_type == "smooth_l1"
        assert cfg.model.roi_box_reg_aux_weight == 0.1
        assert cfg.production.max_detections_per_image == 2000
    assert one_x.training.num_epochs == 12
    assert list(one_x.training.lr_scheduler_milestones) == [8, 11]
    assert three_x.training.num_epochs == 36
    assert list(three_x.training.lr_scheduler_milestones) == [24, 33]
    assert three_x.training.learning_rate == 0.005
    assert three_x.training.lr_scheduler_gamma == 0.1
    assert frcnn_1x.training.num_epochs == 12
    assert frcnn_3x.training.num_epochs == 36
    assert list(frcnn_3x.training.lr_scheduler_milestones) == [24, 33]
    assert frcnn_3x.training.learning_rate == 0.005
    assert frcnn_3x.training.lr_scheduler_gamma == 0.1
    assert fcos_1x.training.num_epochs == 12
    assert list(fcos_1x.training.lr_scheduler_milestones) == [8, 11]
    assert fcos_3x.training.num_epochs == 36
    assert list(fcos_3x.training.lr_scheduler_milestones) == [24, 33]
    assert fcos_1x.model.box_reg_loss_type == "riou"
    assert fcos_3x.model.box_reg_loss_type == "riou"
    assert fcos_1x.training.learning_rate == 0.0025
    assert fcos_3x.training.learning_rate == 0.0025
    assert fcos_1x.production.max_detections_per_image == 2000
    assert fcos_3x.production.max_detections_per_image == 2000
