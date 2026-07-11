"""Tests for CSV annotation load/save used by the annotation viewer."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from annotation_csv import (  # noqa: E402
    AnnotationRecord,
    load_annotation_csv,
    next_ann_id,
    save_annotation_csv,
)
from oriented_det import RBox  # noqa: E402


def test_load_save_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "ann.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_id", "class", "geometry"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "image_id": "a.jpg",
                "class": "Plane",
                "geometry": "[(10, 10), (50, 10), (50, 30), (10, 30), (10, 10)]",
            }
        )
    by_image = load_annotation_csv(csv_path)
    assert len(by_image["a.jpg"]) == 1
    rec = by_image["a.jpg"][0]
    assert rec.class_name == "Plane"
    assert rec.rbox.width > 0

    rec.rbox = RBox(30, 20, 40, 20, 0.5)
    rec.source = "manual"
    out_path = tmp_path / "out.csv"
    save_annotation_csv(out_path, by_image)

    reloaded = load_annotation_csv(out_path)
    rb = reloaded["a.jpg"][0].rbox
    assert abs(rb.cx - 30) < 0.01
    assert abs(rb.angle - 0.5) < 0.01
    assert reloaded["a.jpg"][0].source == "manual"


def test_next_ann_id_numeric() -> None:
    by_image = {
        "a.jpg": [
            AnnotationRecord("1", "a.jpg", "x", RBox(0, 0, 1, 1, 0)),
            AnnotationRecord("42", "a.jpg", "x", RBox(0, 0, 1, 1, 0)),
        ]
    }
    assert next_ann_id(by_image) == "43"


def test_load_from_rbox_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "obb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image_id", "class", "cx", "cy", "width", "height", "angle_deg"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "7",
                "image_id": "b.png",
                "class": "ship",
                "cx": "100",
                "cy": "200",
                "width": "80",
                "height": "20",
                "angle_deg": "15",
            }
        )
    rec = load_annotation_csv(csv_path)["b.png"][0]
    assert rec.class_name == "ship"
    assert abs(rec.rbox.cx - 100) < 0.01
