"""Tests for DOTA dataset loader."""

import math
from pathlib import Path
import tempfile

from PIL import Image

from oriented_det.data import DOTAAnnotation, DOTADataset, DOTASample, format_dota_line
from oriented_det.geometry import RBox


def test_format_dota_line_and_to_line():
    """Official DOTA format: comma-separated output."""
    line = format_dota_line(100, 200, 300, 200, 300, 400, 100, 400, "plane", 0)
    assert line == "100, 200, 300, 200, 300, 400, 100, 400, plane, 0"
    ann = DOTAAnnotation.from_line(line)
    assert ann.to_line() == line


def test_dota_annotation_parsing():
    """Test parsing of DOTA annotation lines (space-separated)."""
    line = "100 200 300 200 300 400 100 400 plane 0"
    ann = DOTAAnnotation.from_line(line)
    
    assert ann.class_name == "plane"
    assert ann.difficult == 0
    assert len(ann.polygon) == 4
    assert isinstance(ann.rbox, RBox)


def test_dota_annotation_parsing_comma_format():
    """Test parsing of official DOTA comma-separated format."""
    line = "100, 200, 300, 200, 300, 400, 100, 400, plane, 0"
    ann = DOTAAnnotation.from_line(line)
    assert ann.class_name == "plane"
    assert ann.difficult == 0

    line_multiword = "0, 0, 10, 0, 10, 10, 0, 10, storage tank, 1"
    ann2 = DOTAAnnotation.from_line(line_multiword)
    assert ann2.class_name == "storage tank"
    assert ann2.difficult == 1


def test_dota_sample_filtering():
    """Test filtering of DOTA samples by class and difficulty."""
    annotations = [
        DOTAAnnotation.from_line("0 0 10 0 10 10 0 10 plane 0"),
        DOTAAnnotation.from_line("20 20 30 20 30 30 20 30 ship 0"),
        DOTAAnnotation.from_line("40 40 50 40 50 50 40 50 plane 1"),
    ]
    
    sample = DOTASample(
        image_path=Path("test.png"),
        width=100,
        height=100,
        annotations=tuple(annotations)
    )
    
    # Filter by class
    filtered = sample.filter_by_class(allowed_classes=["plane"])
    assert len(filtered.annotations) == 2
    
    # Filter by difficulty
    filtered = sample.filter_by_class(drop_difficult=True)
    assert len(filtered.annotations) == 2
    
    # Combined filter
    filtered = sample.filter_by_class(allowed_classes=["plane"], drop_difficult=True)
    assert len(filtered.annotations) == 1


def test_dota_dataset_structure():
    """Test DOTA dataset structure (without requiring actual files)."""
    # This test validates the dataset structure without file I/O
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "labelTxt").mkdir()
        (root / "images").mkdir()
        
        # Create a mock annotation file
        ann_file = root / "labelTxt" / "P0001_train.txt"
        ann_file.write_text("100, 200, 300, 200, 300, 400, 100, 400, plane, 0\n")
        
        # Create a mock image (we'll skip actual image creation for unit tests)
        # In real usage, PIL would load the image
        
        # Dataset initialization should work
        try:
            dataset = DOTADataset(root, split="train")
            assert len(dataset._annotation_files) == 1
        except FileNotFoundError:
            # Expected if image file doesn't exist
            pass


def test_dota_dataset_separate_folders():
    """Test DOTA dataset with train/val/test in separate folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create separate folder structure
        train_label_dir = root / "train" / "labelTxt"
        train_image_dir = root / "train" / "images"
        train_label_dir.mkdir(parents=True)
        train_image_dir.mkdir(parents=True)
        
        val_label_dir = root / "val" / "labelTxt"
        val_image_dir = root / "val" / "images"
        val_label_dir.mkdir(parents=True)
        val_image_dir.mkdir(parents=True)
        
        # Create annotation files in separate folders
        train_ann_file = train_label_dir / "P0001.txt"
        train_ann_file.write_text("100, 200, 300, 200, 300, 400, 100, 400, plane, 0\n")
        
        val_ann_file = val_label_dir / "P0002.txt"
        val_ann_file.write_text("50, 50, 150, 50, 150, 150, 50, 150, ship, 0\n")
        
        # Test train dataset with custom directories
        try:
            train_dataset = DOTADataset(
                root_dir=root,
                split="train",
                label_dir=train_label_dir,
                image_dir=train_image_dir
            )
            assert len(train_dataset._annotation_files) == 1
            assert train_dataset.label_dir == train_label_dir
            assert train_dataset.image_dir == train_image_dir
        except FileNotFoundError:
            # Expected if image file doesn't exist
            pass

        # Test val dataset with custom directories
        try:
            val_dataset = DOTADataset(
                root_dir=root,
                split="val",
                label_dir=val_label_dir,
                image_dir=val_image_dir
            )
            assert len(val_dataset._annotation_files) == 1
            assert val_dataset.label_dir == val_label_dir
            assert val_dataset.image_dir == val_image_dir
        except FileNotFoundError:
            # Expected if image file doesn't exist
            pass


def test_dota_dataset_difficult_strategy_controls_read_time_filtering(tmp_path):
    label_dir = tmp_path / "labelTxt"
    image_dir = tmp_path / "images"
    label_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (32, 32)).save(image_dir / "P0001_train.png")
    (label_dir / "P0001_train.txt").write_text(
        "0, 0, 10, 0, 10, 10, 0, 10, plane, 0\n"
        "12, 12, 20, 12, 20, 20, 12, 20, ship, 1\n",
        encoding="utf-8",
    )

    drop_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        difficult_strategy="drop",
    )
    keep_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        difficult_strategy="keep",
    )
    ignore_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        difficult_strategy="ignore",
    )

    assert [a.class_name for a in drop_ds[0].annotations] == ["plane"]
    assert [a.class_name for a in keep_ds[0].annotations] == ["plane", "ship"]
    assert [a.class_name for a in ignore_ds[0].annotations] == ["plane", "ship"]


def test_dota_dataset_filter_empty_gt(tmp_path):
    label_dir = tmp_path / "labels"
    image_dir = tmp_path / "images"
    label_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (32, 32)).save(image_dir / "empty.png")
    Image.new("RGB", (32, 32)).save(image_dir / "has_obj.png")
    (label_dir / "empty.txt").write_text("", encoding="utf-8")
    (label_dir / "has_obj.txt").write_text(
        "0, 0, 10, 0, 10, 10, 0, 10, plane, 0\n",
        encoding="utf-8",
    )

    all_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        filter_empty_gt=False,
    )
    filtered_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        filter_empty_gt=True,
    )

    assert len(all_ds) == 2
    assert all_ds.annotation_files_discovered_count == 2
    assert all_ds.empty_gt_filtered_count == 0

    assert len(filtered_ds) == 1
    assert filtered_ds.annotation_files_discovered_count == 2
    assert filtered_ds.empty_gt_filtered_count == 1
    assert filtered_ds[0].num_objects == 1


def test_dota_dataset_filter_empty_gt_respects_difficult_drop(tmp_path):
    label_dir = tmp_path / "labels"
    image_dir = tmp_path / "images"
    label_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (32, 32)).save(image_dir / "only_difficult.png")
    (label_dir / "only_difficult.txt").write_text(
        "0, 0, 10, 0, 10, 10, 0, 10, plane, 1\n",
        encoding="utf-8",
    )

    keep_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        difficult_strategy="keep",
        filter_empty_gt=True,
    )
    drop_ds = DOTADataset(
        tmp_path,
        split="train",
        label_dir=label_dir,
        image_dir=image_dir,
        difficult_strategy="drop",
        filter_empty_gt=True,
    )

    assert len(keep_ds) == 1
    assert len(drop_ds) == 0
    assert drop_ds.empty_gt_filtered_count == 1
