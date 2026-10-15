"""OSSDD amplitude scaling, rotated-box parsing, and DOTA export."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from oriented_det.data.evaluation import ossdd_map_iou_thresholds
from oriented_det.train.config import TrainingExperimentConfig
from tools.ossdd_fix_tiff import fix_linearraw_tiff
from tools.ossdd_to_dota import (
    amplitude_to_uint8,
    convert_ossdd,
    parse_rbb_text,
)


def test_parse_rbb_skips_count_and_keeps_corners():
    text = "2\n0 1 2 3 4 5 6 7 8 10 11\n1 || 8 || 9 || 10 || 11 || 12 || 13 || 14 || 15 || 4 || 5\n"
    boxes = parse_rbb_text(text)
    assert boxes == [
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
    ]


def test_amplitude_log_minmax_maps_dynamic_range():
    values = np.array([[0.0, 1.0], [10.0, 50000.0]], dtype=np.float32)
    out = amplitude_to_uint8(values)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2)
    assert int(out[0, 0]) == 0
    assert int(out[0, 1]) == 0
    assert int(out[1, 0]) == 255
    assert int(out[1, 1]) == 0


def _linearraw_chip(path: Path, values: np.ndarray) -> None:
    """700-style TIFF: pixel bytes first, IFD at the end, photometric 34892."""
    pixels = np.asarray(values, dtype="<f4").tobytes()
    height, width = values.shape
    ifd_offset = 8 + len(pixels)
    entries = [
        (256, 3, 1, width),
        (257, 3, 1, height),
        (258, 3, 1, 32),
        (259, 3, 1, 1),
        (262, 3, 1, 34892),
        (273, 4, 1, 8),
        (277, 3, 1, 1),
        (278, 3, 1, height),
        (279, 4, 1, len(pixels)),
        (284, 3, 1, 1),
        (339, 3, 1, 3),
    ]
    ifd = struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        if typ == 3:
            inline = struct.pack("<H", value) + b"\x00\x00"
        else:
            inline = struct.pack("<I", value)
        ifd += struct.pack("<HHI", tag, typ, count) + inline
    ifd += struct.pack("<I", 0)
    path.write_bytes(b"II*\x00" + struct.pack("<I", ifd_offset) + pixels + ifd)


def test_fix_linearraw_tiff_opens_in_pillow(tmp_path):
    path = tmp_path / "Chip_VH_scene.tif"
    values = np.zeros((700, 700), dtype=np.float32)
    values[0, 0] = np.float32(1.5)
    values[10, 20] = np.float32(-3.25)
    _linearraw_chip(path, values)
    with pytest.raises(UnidentifiedImageError):
        Image.open(path)
    assert fix_linearraw_tiff(path) == "fixed"
    with Image.open(path) as img:
        assert img.mode == "F"
        assert img.size == (700, 700)
        loaded = np.asarray(img)
    assert loaded[0, 0] == np.float32(1.5)
    assert loaded[10, 20] == np.float32(-3.25)
    assert fix_linearraw_tiff(path) == "skipped"


def test_convert_writes_dota_split(tmp_path):
    root = tmp_path / "ossdd"
    (root / "Chip_VH").mkdir(parents=True)
    chip = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32)
    Image.fromarray(chip, mode="F").save(root / "Chip_VH" / "Chip_VH_scene.tif")
    (root / "Metadata_rbb").mkdir()
    (root / "Metadata_rbb" / "Metadata_rbb_scene.txt").write_text(
        "1\n0 0 0 1 0 1 1 0 1 0 0\n",
        encoding="utf-8",
    )
    (root / "metadata.csv").write_text(
        "file_name,split\nChip_VH_scene.tif,train\n",
        encoding="utf-8",
    )
    out = tmp_path / "dota"
    counts = convert_ossdd(root, out, polarization="VH", splits=("train",))
    assert counts["train"] == 1
    label = (out / "train" / "labels" / "scene_VH.txt").read_text(encoding="utf-8")
    assert label.startswith("0, 0, 1, 0, 1, 1, 0, 1, ship, 0")
    with Image.open(out / "train" / "images" / "scene_VH.png") as img:
        assert img.mode == "RGB"
        assert img.size == (2, 2)


def test_ossdd_rtmdet_recipe_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(
        root / "configs" / "rotated_rtmdet" / "ossdd_le90_vh_1x_s.json"
    )
    assert cfg.model.backbone == "cspnext_s"
    assert cfg.training.num_epochs == 12
    assert cfg.training.lr_scheduler_cosine_start_epoch == 6
    assert cfg.training.lr_warmup_steps == 500
    assert cfg.checkpoint.load_from_checkpoint == "hf://rotated_rtmdet_dota_le90_3x"
    assert cfg.preprocessing.resize_mode == "crop"
    assert list(cfg.preprocessing.target_size) == [512, 512]
    assert cfg.dataset.train_tiles_dirs == [Path("/path/to/data/OSSDD-dota/train")]
    assert cfg.dataset.val_tiles_dirs == [Path("/path/to/data/OSSDD-dota/val")]
    assert cfg.dataset.test_tiles_dirs == [Path("/path/to/data/OSSDD-dota/test")]
    assert cfg.dataset.has_test_split()
    assert cfg.evaluation.test_map is True
    assert cfg.evaluation.map_iou_thresholds == ossdd_map_iou_thresholds()
    assert cfg.preprocessing.normalize_mean == [0.307941, 0.311986, 0.297761]
    assert cfg.preprocessing.normalize_std == [0.192315, 0.187769, 0.181551]


def test_ossdd_fcos_and_faster_rcnn_3x_recipes_load():
    root = Path(__file__).resolve().parents[1]
    fcos = TrainingExperimentConfig.load(
        root / "configs" / "rotated_fcos" / "ossdd_le90_vh_3x.json"
    )
    assert fcos.model_type == "rotated_fcos"
    assert fcos.training.num_epochs == 36
    assert list(fcos.training.lr_scheduler_milestones) == [24, 33]
    assert fcos.training.learning_rate == 0.0025
    assert fcos.data_loader.batch_size == 4
    assert fcos.checkpoint.load_from_checkpoint == "hf://rotated_fcos_dota_le90_1x"
    assert fcos.model.box_reg_loss_type == "riou"
    assert fcos.preprocessing.resize_mode == "crop"
    assert list(fcos.preprocessing.target_size) == [512, 512]
    assert fcos.dataset.test_tiles_dirs == [Path("/path/to/data/OSSDD-dota/test")]
    assert fcos.evaluation.test_map is True
    assert fcos.evaluation.map_iou_thresholds == ossdd_map_iou_thresholds()

    frcnn = TrainingExperimentConfig.load(
        root / "configs" / "rotated_faster_rcnn" / "ossdd_le90_vh_3x.json"
    )
    assert frcnn.model_type == "rotated_faster_rcnn"
    assert frcnn.training.num_epochs == 36
    assert list(frcnn.training.lr_scheduler_milestones) == [24, 33]
    assert frcnn.training.learning_rate == 0.005
    assert frcnn.data_loader.batch_size == 4
    assert frcnn.checkpoint.load_from_checkpoint == "hf://rotated_faster_rcnn_dota_le90_1x"
    assert frcnn.preprocessing.resize_mode == "crop"
    assert list(frcnn.preprocessing.target_size) == [512, 512]
    assert frcnn.dataset.train_tiles_dirs == fcos.dataset.train_tiles_dirs
    assert frcnn.dataset.val_tiles_dirs == fcos.dataset.val_tiles_dirs
    assert frcnn.evaluation.test_map is True
    assert frcnn.evaluation.map_iou_thresholds == ossdd_map_iou_thresholds()
