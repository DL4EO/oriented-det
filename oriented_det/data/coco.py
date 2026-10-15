"""Official COCO 2017 detection boxes as horizontal le90 labels.

Layout under ``data_root``:

- ``train2017/*.jpg`` and ``val2017/*.jpg``
- ``annotations/instances_{train,val}2017.json``

The detection ``bbox`` (``x, y, w, h``) becomes le90 ``(cx, cy, w, h, 0)``.
Segmentation polygons are ignored. ``iscrowd`` annotations are dropped.
Category ids are not contiguous; they map onto labels ``1..N`` in id order.
Images with no objects are kept unless ``filter_empty_gt`` is set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from ..geometry import RBox
from .dota import DOTAAnnotation, DOTASample

_SPLIT_TO_FOLDER = {
    "train": "train2017",
    "train2017": "train2017",
    "val": "val2017",
    "val2017": "val2017",
}


def annotation_from_coco_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    class_name: str,
) -> DOTAAnnotation:
    """COCO ``bbox`` ``[x, y, w, h]`` → horizontal le90 ``(cx, cy, w, h, 0)``."""
    width_f = float(width)
    height_f = float(height)
    if width_f <= 0 or height_f <= 0:
        raise ValueError("bbox width/height must be positive")
    rbox = RBox(
        float(x) + width_f / 2.0,
        float(y) + height_f / 2.0,
        width_f,
        height_f,
        0.0,
    )
    return DOTAAnnotation(
        class_name=str(class_name).strip(),
        difficult=0,
        polygon=rbox.to_polygon(),
        rbox=rbox,
    )


def coco_category_names(categories: Sequence[dict]) -> Tuple[Dict[int, str], List[str]]:
    """Map COCO category ids onto class names in ascending id order.

    Returns ``(category_id → name, names)``. Label ``k`` is ``names[k - 1]``.
    Gaps in the id list do not leave holes in ``names``.
    """
    ordered: List[Tuple[int, str]] = []
    seen: set[int] = set()
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        try:
            cid = int(cat["id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(cat.get("name") or "").strip()
        if not name or cid in seen:
            continue
        seen.add(cid)
        ordered.append((cid, name))
    ordered.sort(key=lambda item: item[0])
    id_to_name = {cid: name for cid, name in ordered}
    return id_to_name, [name for _, name in ordered]


def resolve_coco_split(dataset_cfg, role: str) -> str:
    """Map a train-loop role to a COCO folder name (``train2017`` / ``val2017``)."""
    name = (role or "").strip().lower()
    if name == "train":
        override = getattr(dataset_cfg, "train_split", None)
        return str(override).strip() if override else "train2017"
    if name == "val":
        override = getattr(dataset_cfg, "val_split", None)
        return str(override).strip() if override else "val2017"
    if name in _SPLIT_TO_FOLDER:
        return name
    raise ValueError(
        f"Unsupported COCO split {role!r}. Expected train, val, train2017, or val2017."
    )


def _folder_name(split: str) -> str:
    key = split.strip().lower()
    if key not in _SPLIT_TO_FOLDER:
        raise ValueError(
            f"Unsupported COCO split {split!r}. Expected train, val, train2017, or val2017."
        )
    return _SPLIT_TO_FOLDER[key]


@dataclass(frozen=True)
class _CocoRecord:
    image_path: Path
    width: int
    height: int
    annotations: Tuple[DOTAAnnotation, ...]


def _load_instances(ann_file: Path, image_dir: Path) -> Tuple[List[str], List[_CocoRecord]]:
    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO JSON must be an object: {ann_file}")
    id_to_name, class_names = coco_category_names(payload.get("categories") or [])
    if not class_names:
        raise ValueError(f"COCO JSON has no categories: {ann_file}")

    images_by_id: Dict[int, dict] = {}
    for img in payload.get("images") or []:
        if not isinstance(img, dict):
            continue
        try:
            iid = int(img["id"])
        except (KeyError, TypeError, ValueError):
            continue
        images_by_id[iid] = img

    anns_by_image: Dict[int, List[DOTAAnnotation]] = {iid: [] for iid in images_by_id}
    for raw in payload.get("annotations") or []:
        if not isinstance(raw, dict):
            continue
        if int(raw.get("iscrowd") or 0) != 0:
            continue
        try:
            image_id = int(raw["image_id"])
            category_id = int(raw["category_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if image_id not in anns_by_image:
            continue
        class_name = id_to_name.get(category_id)
        if class_name is None:
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            ann = annotation_from_coco_bbox(
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                class_name,
            )
        except (TypeError, ValueError):
            continue
        anns_by_image[image_id].append(ann)

    records: List[_CocoRecord] = []
    for image_id, img in images_by_id.items():
        file_name = str(img.get("file_name") or "").strip()
        if not file_name:
            continue
        image_path = image_dir / Path(file_name).name
        if not image_path.is_file():
            continue
        try:
            width = int(float(img.get("width") or 0))
            height = int(float(img.get("height") or 0))
        except (TypeError, ValueError):
            width, height = 0, 0
        records.append(
            _CocoRecord(
                image_path=image_path,
                width=width,
                height=height,
                annotations=tuple(anns_by_image.get(image_id, ())),
            )
        )
    return class_names, records


class CocoDetectionDataset:
    """COCO 2017 detection split yielding ``DOTASample`` with θ = 0."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        difficult_strategy: str = "keep",
        filter_empty_gt: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
    ):
        from .lookalike import resolve_lookalike_label_set

        ds = (difficult_strategy or "keep").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(
                f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'."
            )
        folder = _folder_name(split)
        root = Path(data_root)
        image_dir = root / folder
        ann_file = root / "annotations" / f"instances_{folder}.json"
        if not ann_file.is_file():
            raise FileNotFoundError(
                f"COCO annotations not found: {ann_file}. "
                f"Expected annotations/instances_{folder}.json under {root}."
            )
        if not image_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {image_dir}")

        self.root = root
        self.split = folder
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        self.class_names, discovered = _load_instances(ann_file, image_dir)
        self._records_discovered_count = len(discovered)
        kept: List[_CocoRecord] = []
        for record in discovered:
            if self.filter_empty_gt and self._effective_gt_count(record) == 0:
                continue
            kept.append(record)
        self._records = kept
        self._empty_gt_filtered_count = self._records_discovered_count - len(self._records)

    @property
    def tiles_discovered_count(self) -> int:
        return self._records_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        return self._empty_gt_filtered_count

    def _load_size(self, record: _CocoRecord) -> Tuple[int, int]:
        if record.width > 0 and record.height > 0:
            return record.width, record.height
        if Image is None:
            raise RuntimeError("PIL/Pillow is required to read COCO images.")
        with Image.open(record.image_path) as img:
            return int(img.size[0]), int(img.size[1])

    def _effective_gt_count(self, record: _CocoRecord) -> int:
        if not record.annotations:
            return 0
        sample = DOTASample(
            image_path=record.image_path,
            width=record.width or 1,
            height=record.height or 1,
            annotations=record.annotations,
        )
        sample = sample.filter_by_class(
            allowed_classes=self.allowed_classes,
            ignore_labels=self.ignore_labels,
            drop_difficult=self._drop_difficult,
            lookalike_labels=self.lookalike_labels,
        )
        return len(sample.annotations)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        record = self._records[idx]
        width, height = self._load_size(record)
        sample = DOTASample(
            image_path=record.image_path,
            width=width,
            height=height,
            annotations=record.annotations,
        )
        if self.allowed_classes is not None or self.ignore_labels or self._drop_difficult:
            sample = sample.filter_by_class(
                allowed_classes=self.allowed_classes,
                ignore_labels=self.ignore_labels,
                drop_difficult=self._drop_difficult,
                lookalike_labels=self.lookalike_labels,
            )
        return sample

    def get_class_names(self) -> List[str]:
        return list(self.class_names)


def format_coco_empty_gt_filter_log(dataset: CocoDetectionDataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


__all__ = [
    "CocoDetectionDataset",
    "annotation_from_coco_bbox",
    "coco_category_names",
    "format_coco_empty_gt_filter_log",
    "resolve_coco_split",
]
