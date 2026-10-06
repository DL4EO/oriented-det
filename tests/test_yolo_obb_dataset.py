"""YOLO-OBB polygon label loader."""

from pathlib import Path

import pytest

from oriented_det.data.build import build_split_dataset
from oriented_det.data.yolo_obb import parse_yolo_obb_line
from oriented_det.train.config import DatasetConfig


def _write_dataset(root: Path) -> None:
    (root / "classes.txt").write_text("ship\nplane\n", encoding="utf-8")
    for split, color in (("train", (10, 20, 30)), ("val", (40, 50, 60))):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL required")
        Image.new("RGB", (100, 50), color).save(images / "a.png")
        # Axis-aligned quad covering the center half of the image.
        labels.joinpath("a.txt").write_text(
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
            encoding="utf-8",
        )
    (root / "images" / "train" / "empty.png")
    from PIL import Image

    Image.new("RGB", (100, 50), (0, 0, 0)).save(root / "images" / "train" / "empty.png")
    (root / "labels" / "train" / "empty.txt").write_text("", encoding="utf-8")


def test_parse_normalized_quad_to_le90():
    ann = parse_yolo_obb_line(
        "1 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0",
        width=100,
        height=50,
        class_names=("ship", "plane"),
    )
    assert ann.class_name == "plane"
    rbox = ann.rbox
    assert rbox.cx == pytest.approx(50.0, abs=1.0)
    assert rbox.cy == pytest.approx(25.0, abs=1.0)
    assert rbox.width > 0 and rbox.height > 0


def test_out_of_range_coords_rejected():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        parse_yolo_obb_line(
            "0 0.0 0.0 1.2 0.0 1.0 1.0 0.0 1.0",
            width=10,
            height=10,
            class_names=("ship",),
        )


def test_build_split_dataset_yolo_obb(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetConfig(
        data_root=tmp_path,
        format="yolo_obb",
        filter_empty_gt=True,
    )
    train = build_split_dataset(cfg, "train")
    assert train.get_class_names() == ["ship", "plane"]
    assert len(train) == 1
    sample = train[0]
    assert sample.annotations[0].class_name == "ship"
    assert sample.width == 100 and sample.height == 50
    val = build_split_dataset(cfg, "val", filter_empty_gt=False)
    assert len(val) == 1
