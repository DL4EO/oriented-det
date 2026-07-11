"""Load and save polygon / OBB annotation CSV files for the annotation viewer."""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from oriented_det import RBox

from hbb_to_obb import OrientedAnn, polygon_string, write_oriented_csv

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class AnnotationRecord:
    """One editable oriented annotation row."""

    ann_id: str
    image_id: str
    class_name: str
    rbox: RBox
    source: str = "manual"
    difficult: int = 0
    score: Optional[float] = None
    match_iou: Optional[float] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def to_oriented_ann(self) -> OrientedAnn:
        return OrientedAnn(
            ann_id=self.ann_id,
            image_id=self.image_id,
            class_name=self.class_name,
            rbox=self.rbox,
            source=self.source,
            difficult=self.difficult,
            score=self.score,
            match_iou=self.match_iou,
        )


def _rbox_from_row(row: dict) -> RBox:
    geometry = (row.get("geometry") or "").strip()
    if geometry:
        pts = ast.literal_eval(geometry)
        if pts and pts[0] == pts[-1]:
            pts = pts[:-1]
        return RBox.from_points(pts)
    cx = float(row["cx"])
    cy = float(row["cy"])
    width = float(row["width"])
    height = float(row["height"])
    angle_deg = row.get("angle_deg", "0")
    angle = math.radians(float(angle_deg)) if angle_deg != "" else 0.0
    return RBox(cx, cy, width, height, angle)


def _parse_optional_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_annotation_csv(path: Path) -> Dict[str, List[AnnotationRecord]]:
    """Load annotations grouped by ``image_id``."""
    by_image: Dict[str, List[AnnotationRecord]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        known = {
            "id",
            "image_id",
            "class",
            "geometry",
            "cx",
            "cy",
            "width",
            "height",
            "angle_deg",
            "source",
            "score",
            "match_iou",
            "difficult",
        }
        for row in reader:
            image_id = row.get("image_id", "").strip()
            if not image_id:
                continue
            ann_id = str(row.get("id", "")).strip() or f"{image_id}:{len(by_image.get(image_id, []))}"
            extra = {k: v for k, v in row.items() if k not in known and v}
            rec = AnnotationRecord(
                ann_id=ann_id,
                image_id=image_id,
                class_name=str(row.get("class", "object")).strip() or "object",
                rbox=_rbox_from_row(row),
                source=str(row.get("source", "") or "manual").strip() or "manual",
                difficult=int(float(row.get("difficult") or 0)),
                score=_parse_optional_float(row.get("score", "")),
                match_iou=_parse_optional_float(row.get("match_iou", "")),
                extra=extra,
            )
            by_image.setdefault(image_id, []).append(rec)
    return by_image


def save_annotation_csv(path: Path, by_image: Dict[str, List[AnnotationRecord]]) -> None:
    """Write all annotations to a CSV file (sorted by image_id, stable row order)."""
    rows: List[OrientedAnn] = []
    for image_id in sorted(by_image):
        for rec in by_image[image_id]:
            rows.append(rec.to_oriented_ann())
    write_oriented_csv(path, rows)


def list_images_from_annotations(
    by_image: Dict[str, Sequence[AnnotationRecord]],
    data_root: Path,
    images_dir: str = "images",
) -> List[str]:
    """Build ordered image list: annotated images first, then unannotated scans."""
    seen: set[str] = set()
    ordered: List[str] = []
    for image_id in sorted(by_image):
        if image_id not in seen:
            ordered.append(image_id)
            seen.add(image_id)

    images_path = data_root / images_dir
    if images_path.is_dir():
        for path in sorted(images_path.iterdir()):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                if path.name not in seen:
                    ordered.append(path.name)
                    seen.add(path.name)
    return ordered


def next_ann_id(by_image: Dict[str, List[AnnotationRecord]]) -> str:
    """Return a new numeric string id one above the max existing id."""
    max_id = 0
    for recs in by_image.values():
        for rec in recs:
            try:
                max_id = max(max_id, int(rec.ann_id))
            except ValueError:
                continue
    return str(max_id + 1)


def image_path_for_id(
    data_root: Path,
    image_id: str,
    images_dir: str = "images",
) -> Path:
    direct = data_root / image_id
    if direct.is_file():
        return direct
    under_images = data_root / images_dir / image_id
    if under_images.is_file():
        return under_images
    return under_images
