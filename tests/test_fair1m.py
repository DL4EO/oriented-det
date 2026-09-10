"""Tests for FAIR1M XML loader, holdout split, and DOTA export."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from oriented_det.data import (
    DOTADataset,
    FAIR1M_CLASSES,
    FAIR1MDataset,
    build_split_dataset,
    export_fair1m_to_dota,
)
from oriented_det.data.fair1m import (
    fair1m_holdout_stems,
    fair1m_points_to_annotation,
    normalize_fair1m_class_name,
    parse_fair1m_xml,
    resolve_fair1m_imageset_split,
)
from oriented_det.data.fair1m_classes import FAIR1M_FINE_TO_COARSE, FAIR1M_GROUPS
from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig


def _fair1m_xml(
    *,
    filename: str,
    objects: list[tuple[str, list[tuple[float, float]], int]],
) -> str:
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotation>",
        "  <source>",
        f"    <filename>{filename}</filename>",
        "  </source>",
        "  <objects>",
    ]
    for class_name, points, difficult in objects:
        parts.append("    <object>")
        parts.append("      <points>")
        for x, y in points:
            parts.append(f"        <point>{x},{y}</point>")
        # FAIR1M XMLs often repeat the first point as a 5th entry.
        x0, y0 = points[0]
        parts.append(f"        <point>{x0},{y0}</point>")
        parts.append("      </points>")
        parts.append("      <possibleresult>")
        parts.append(f"        <name>{class_name}</name>")
        parts.append("      </possibleresult>")
        parts.append(f"      <difficult>{difficult}</difficult>")
        parts.append("    </object>")
    parts.extend(["  </objects>", "</annotation>", ""])
    return "\n".join(parts)


def _write_mini_fair1m_kaggle(root: Path) -> Path:
    """Kaggle-like Dataset/Images/Train + Dataset/Labels/Train layout."""
    images = root / "Dataset" / "Images" / "Train"
    labels = root / "Dataset" / "Labels" / "Train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(images / "t_0.jpg")
    Image.new("RGB", (200, 100), color=(30, 50, 70)).save(images / "t_1.jpg")
    Image.new("RGB", (200, 100), color=(40, 60, 80)).save(images / "t_2.jpg")

    (labels / "t_0.xml").write_text(
        _fair1m_xml(
            filename="t_0.jpg",
            objects=[
                (
                    "Boeing737",
                    [(40.0, 20.0), (120.0, 20.0), (120.0, 50.0), (40.0, 50.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    (labels / "t_1.xml").write_text(
        _fair1m_xml(
            filename="t_1.jpg",
            objects=[
                (
                    "Small_Car",
                    [(10.0, 10.0), (40.0, 10.0), (40.0, 25.0), (10.0, 25.0)],
                    1,
                )
            ],
        ),
        encoding="utf-8",
    )
    (labels / "t_2.xml").write_text(
        _fair1m_xml(filename="t_2.jpg", objects=[]),
        encoding="utf-8",
    )
    return root


def _write_mini_fair1m_ollypowell(root: Path) -> Path:
    """Published ollypowell dump: JPG ``t_N``/``v_N`` + Notebook_Working ``N.xml``."""
    train_img = root / "Dataset" / "Images" / "Train"
    val_img = root / "Dataset" / "Images" / "Val"
    train_xml = root / "Notebook_Working" / "train_labels"
    val_xml = root / "Notebook_Working" / "val_labels"
    for d in (train_img, val_img, train_xml, val_xml):
        d.mkdir(parents=True)

    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(train_img / "t_0.jpg")
    Image.new("RGB", (200, 100), color=(30, 50, 70)).save(train_img / "t_1.jpg")
    Image.new("RGB", (180, 90), color=(50, 60, 70)).save(val_img / "v_0.jpg")

    (train_xml / "0.xml").write_text(
        _fair1m_xml(
            filename="0.tif",
            objects=[
                (
                    "Boeing737",
                    [(40.0, 20.0), (120.0, 20.0), (120.0, 50.0), (40.0, 50.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    (train_xml / "1.xml").write_text(
        _fair1m_xml(
            filename="1.tif",
            objects=[
                (
                    "Bridge",
                    [(10.0, 10.0), (40.0, 10.0), (40.0, 25.0), (10.0, 25.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    (val_xml / "0.xml").write_text(
        _fair1m_xml(
            filename="0.tif",
            objects=[
                (
                    "Tennis Court",
                    [(20.0, 20.0), (80.0, 20.0), (80.0, 60.0), (20.0, 60.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    return root


def _write_mini_fair1m_official(root: Path) -> Path:
    """Official train/part1 + validation layout."""
    train_img = root / "train" / "part1" / "images"
    train_xml = root / "train" / "part1" / "labelXml"
    val_img = root / "validation" / "images"
    val_xml = root / "validation" / "labelXml"
    for d in (train_img, train_xml, val_img, val_xml):
        d.mkdir(parents=True)

    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(train_img / "0001.tif")
    Image.new("RGB", (128, 128), color=(40, 50, 60)).save(val_img / "0002.tif")
    (train_xml / "0001.xml").write_text(
        _fair1m_xml(
            filename="0001.tif",
            objects=[
                (
                    "Bridge",
                    [(20.0, 20.0), (100.0, 20.0), (100.0, 40.0), (20.0, 40.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    (val_xml / "0002.xml").write_text(
        _fair1m_xml(
            filename="0002.tif",
            objects=[
                (
                    "Tennis Court",
                    [(30.0, 30.0), (90.0, 30.0), (90.0, 80.0), (30.0, 80.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    return root


def test_fair1m_classes_canonical():
    assert len(FAIR1M_CLASSES) == 37
    assert len(FAIR1M_GROUPS) == 5
    assert FAIR1M_FINE_TO_COARSE["Boeing737"] == "airplane"
    assert normalize_fair1m_class_name("Small_Car") == "Small Car"


def test_fair1m_points_to_annotation_le90():
    ann = fair1m_points_to_annotation(
        [(40.0, 20.0), (120.0, 20.0), (120.0, 50.0), (40.0, 50.0)],
        "Boeing737",
    )
    assert ann.class_name == "Boeing737"
    assert abs(ann.rbox.cx - 80.0) < 1e-3
    assert abs(ann.rbox.cy - 35.0) < 1e-3
    assert abs(ann.rbox.width - 80.0) < 1e-3 or abs(ann.rbox.height - 80.0) < 1e-3


def test_parse_fair1m_xml(tmp_path: Path):
    xml_path = tmp_path / "t_0.xml"
    xml_path.write_text(
        _fair1m_xml(
            filename="t_0.jpg",
            objects=[
                (
                    "Passenger Ship",
                    [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)],
                    0,
                )
            ],
        ),
        encoding="utf-8",
    )
    filename, anns = parse_fair1m_xml(xml_path)
    assert filename == "t_0.jpg"
    assert len(anns) == 1
    assert anns[0].class_name == "Passenger Ship"


def test_fair1m_kaggle_dataset(tmp_path: Path):
    root = _write_mini_fair1m_kaggle(tmp_path / "kaggle")
    ds = FAIR1MDataset(root, split="train", difficult_strategy="keep")
    assert len(ds) == 3
    assert ds.get_class_names() == list(FAIR1M_CLASSES)
    sample = ds[0]
    assert sample.image_path.stem == "t_0"
    assert sample.annotations[0].class_name == "Boeing737"

    dropped = FAIR1MDataset(root, split="train", difficult_strategy="drop")
    hard = [s for s in dropped if s.image_path.stem == "t_1"][0]
    assert len(hard.annotations) == 0


def test_fair1m_ollypowell_notebook_working_layout(tmp_path: Path):
    """JPG stems t_N/v_N pair with Notebook_Working N.xml (published Kaggle dump)."""
    from oriented_det.data.fair1m import discover_fair1m_pairs

    root = _write_mini_fair1m_ollypowell(tmp_path / "ollypowell")
    train_pairs = discover_fair1m_pairs(root, "train")
    val_pairs = discover_fair1m_pairs(root, "val")
    assert len(train_pairs) == 2
    assert all(xml is not None and xml.is_file() for _, xml in train_pairs)
    assert train_pairs[0][0].name == "t_0.jpg"
    assert train_pairs[0][1].name == "0.xml"
    assert len(val_pairs) == 1
    assert val_pairs[0][0].name == "v_0.jpg"
    assert val_pairs[0][1].name == "0.xml"

    train = FAIR1MDataset(root, split="train", difficult_strategy="keep")
    val = FAIR1MDataset(root, split="val", difficult_strategy="keep")
    assert len(train) == 2
    assert train[0].annotations[0].class_name == "Boeing737"
    assert len(val) == 1
    assert val[0].annotations[0].class_name == "Tennis Court"

    # Pointing at Dataset/ still finds sibling Notebook_Working labels.
    dataset_only = root / "Dataset"
    assert all(
        xml is not None for _, xml in discover_fair1m_pairs(dataset_only, "train")
    )


def test_fair1m_filter_empty_gt(tmp_path: Path):
    root = _write_mini_fair1m_kaggle(tmp_path / "kaggle")
    full = FAIR1MDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=False)
    filtered = FAIR1MDataset(root, split="train", difficult_strategy="keep", filter_empty_gt=True)
    assert len(full) == 3
    assert len(filtered) == 2
    assert filtered.empty_gt_filtered_count == 1


def test_fair1m_official_splits(tmp_path: Path):
    root = _write_mini_fair1m_official(tmp_path / "FAIR1M")
    train = FAIR1MDataset(root, split="train", difficult_strategy="keep")
    val = FAIR1MDataset(root, split="val", difficult_strategy="keep")
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].annotations[0].class_name == "Bridge"
    assert val[0].annotations[0].class_name == "Tennis Court"


def test_resolve_fair1m_imageset_split_defaults():
    cfg = DatasetConfig(data_root=".", format="fair1m")
    assert resolve_fair1m_imageset_split(cfg, "train") == "train"
    assert resolve_fair1m_imageset_split(cfg, "val") == "val"
    cfg.train_split = "trainval"
    cfg.val_split = "test"
    assert resolve_fair1m_imageset_split(cfg, "train") == "trainval"
    assert resolve_fair1m_imageset_split(cfg, "val") == "test"


def test_build_split_dataset_fair1m(tmp_path: Path):
    root = _write_mini_fair1m_official(tmp_path / "FAIR1M")
    cfg = DatasetConfig(
        data_root=root,
        format="fair1m",
        difficult_strategy="keep",
        filter_empty_gt=True,
    )
    train = build_split_dataset(cfg, "train")
    val = build_split_dataset(cfg, "val")
    assert len(train) == 1
    assert len(val) == 1


def test_fair1m_holdout_stems_deterministic():
    stems = [f"t_{i}" for i in range(100)]
    a_train, a_val = fair1m_holdout_stems(stems, val_fraction=0.1, seed=0)
    b_train, b_val = fair1m_holdout_stems(stems, val_fraction=0.1, seed=0)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_train) + len(a_val) == 100
    assert 5 <= len(a_val) <= 20
    assert not set(a_train) & set(a_val)


def test_export_fair1m_to_dota_with_holdout(tmp_path: Path):
    root = _write_mini_fair1m_kaggle(tmp_path / "kaggle")
    out = tmp_path / "dota"
    # Expand mini set so holdout is non-empty both sides.
    images = root / "Dataset" / "Images" / "Train"
    labels = root / "Dataset" / "Labels" / "Train"
    for i in range(3, 40):
        Image.new("RGB", (64, 64), color=(i, i, i)).save(images / f"t_{i}.jpg")
        (labels / f"t_{i}.xml").write_text(
            _fair1m_xml(
                filename=f"t_{i}.jpg",
                objects=[
                    (
                        "Van",
                        [(5.0, 5.0), (40.0, 5.0), (40.0, 20.0), (5.0, 20.0)],
                        0,
                    )
                ],
            ),
            encoding="utf-8",
        )
    counts = export_fair1m_to_dota(
        root,
        out,
        splits=("train", "val"),
        difficult_strategy="keep",
        val_fraction=0.2,
        split_seed=0,
    )
    assert counts["train"] > 0
    assert counts["val"] > 0
    assert (out / "ImageSets" / "train.txt").is_file()
    exported = list((out / "train" / "images").glob("*.jpg"))
    assert exported
    assert not list((out / "train" / "images").glob("*.png"))
    dota = DOTADataset(
        root_dir=out / "train",
        split="train",
        label_dir=out / "train" / "labels",
        image_dir=out / "train" / "images",
        difficult_strategy="keep",
    )
    assert len(dota) == counts["train"]


def test_export_fair1m_to_dota_force_png(tmp_path: Path):
    root = _write_mini_fair1m_kaggle(tmp_path / "kaggle")
    out = tmp_path / "dota"
    export_fair1m_to_dota(
        root,
        out,
        splits=("train",),
        difficult_strategy="keep",
        image_format="png",
    )
    assert list((out / "train" / "images").glob("*.png"))
    assert not list((out / "train" / "images").glob("*.jpg"))


def test_require_pillow_allows_large_rasters():
    from PIL import Image as PILImage

    from oriented_det.data.fair1m import _require_pillow

    _require_pillow()
    assert PILImage.MAX_IMAGE_PIXELS is None


def test_fair1m_recipes_load():
    root = Path(__file__).resolve().parents[1]
    expected_hub = {
        "configs/oriented_rcnn/fair1m_le90_1x.json": "hf://oriented_rcnn_dota_le90_1x",
        "configs/rotated_faster_rcnn/fair1m_le90_1x.json": "hf://rotated_faster_rcnn_dota_le90_1x",
        "configs/rotated_fcos/fair1m_le90_1x.json": "hf://rotated_fcos_dota_le90_1x",
    }
    for rel, hub in expected_hub.items():
        cfg = TrainingExperimentConfig.load(root / rel)
        assert cfg.dataset.format == "dota"
        assert cfg.dataset.overlap == 200
        assert cfg.dataset.filter_empty_gt is True
        assert list(cfg.preprocessing.target_size) == [1024, 1024]
        assert cfg.model.final_nms_iou_threshold == 0.1
        assert cfg.production.final_nms_iou_threshold == 0.1
        assert cfg.evaluation.final_nms_iou_threshold == 0.1
        assert cfg.training.num_epochs == 12
        assert cfg.checkpoint.load_from_checkpoint == hub
        assert cfg.checkpoint.load_optimizer_state is False
        assert cfg.checkpoint.resume_from_checkpoint_epoch is False
    fcos = TrainingExperimentConfig.load(root / "configs/rotated_fcos/fair1m_le90_1x.json")
    assert fcos.model.box_reg_loss_type == "riou"
    assert fcos.training.learning_rate == 0.0025
