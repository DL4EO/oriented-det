"""YOLO-OBB polygon labels → le90 ``DOTASample``.

On disk::

    data_root/images/{split}/*.{jpg,png,tif}
    data_root/labels/{split}/{stem}.txt
    data_root/classes.txt

Each label line is ``class_id x1 y1 x2 y2 x3 y3 x4 y4`` with coordinates in
``[0, 1]``. Class names come from ``dataset.class_names`` or ``classes.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .coco_obb import annotation_from_polygon_coords
from .dota import DOTAAnnotation, DOTASample

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_yolo_obb_class_names(
    data_root: Path,
    class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Class id order: explicit list, otherwise ``data_root/classes.txt``."""
    if class_names:
        names = [str(name) for name in class_names]
        if any(not name for name in names):
            raise ValueError("dataset.class_names entries must be non-empty.")
        return names
    path = Path(data_root) / "classes.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO-OBB class names missing: set dataset.class_names or write {path}."
        )
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"{path} has no class names.")
    return names


def parse_yolo_obb_line(
    line: str,
    *,
    width: int,
    height: int,
    class_names: Sequence[str],
) -> DOTAAnnotation:
    """One normalized quad → le90 annotation. Coordinates outside ``[0, 1]`` raise."""
    parts = line.split()
    if len(parts) != 9:
        raise ValueError(
            f"YOLO-OBB label must be 'class x1 y1 x2 y2 x3 y3 x4 y4', got {len(parts)} fields."
        )
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"YOLO-OBB class id must be an integer, got {parts[0]!r}.") from exc
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(
            f"YOLO-OBB class id {class_id} is outside 0..{len(class_names) - 1}."
        )
    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"YOLO-OBB coordinates must be floats: {line!r}") from exc
    if any(value < 0.0 or value > 1.0 for value in coords):
        raise ValueError(f"YOLO-OBB coordinates must be in [0, 1]: {line!r}")
    pixels: List[float] = []
    for i in range(0, 8, 2):
        pixels.append(coords[i] * float(width))
        pixels.append(coords[i + 1] * float(height))
    return annotation_from_polygon_coords(pixels, class_names[class_id])


def _image_size(path: Path) -> Tuple[int, int]:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required to read YOLO-OBB images.")
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def _read_annotations(
    label_path: Path,
    *,
    width: int,
    height: int,
    class_names: Sequence[str],
) -> Tuple[DOTAAnnotation, ...]:
    if not label_path.is_file():
        return ()
    text = label_path.read_text(encoding="utf-8")
    anns: List[DOTAAnnotation] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        anns.append(
            parse_yolo_obb_line(line, width=width, height=height, class_names=class_names)
        )
    return tuple(anns)


class YoloOBBDataset:
    """Folder dataset of YOLO-OBB quads yielding ``DOTASample``."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
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
        self.root = Path(data_root)
        self.split = split.strip()
        self.class_names = load_yolo_obb_class_names(self.root, class_names)
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        image_dir = self.root / "images" / self.split
        self.label_dir = self.root / "labels" / self.split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"YOLO-OBB images directory not found: {image_dir}")
        discovered = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS
        )
        self._records_discovered_count = len(discovered)
        kept: List[Path] = []
        for path in discovered:
            if self.filter_empty_gt and self._effective_gt_count(path) == 0:
                continue
            kept.append(path)
        self._images = kept
        self._empty_gt_filtered_count = self._records_discovered_count - len(self._images)

    @property
    def tiles_discovered_count(self) -> int:
        return self._records_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        return self._empty_gt_filtered_count

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def _annotations_for(self, image_path: Path) -> Tuple[DOTAAnnotation, ...]:
        width, height = _image_size(image_path)
        return _read_annotations(
            self._label_path(image_path),
            width=width,
            height=height,
            class_names=self.class_names,
        )

    def _effective_gt_count(self, image_path: Path) -> int:
        anns = self._annotations_for(image_path)
        if not anns:
            return 0
        width, height = _image_size(image_path)
        sample = DOTASample(image_path=image_path, width=width, height=height, annotations=anns)
        sample = sample.filter_by_class(
            allowed_classes=self.allowed_classes,
            ignore_labels=self.ignore_labels,
            drop_difficult=self._drop_difficult,
            lookalike_labels=self.lookalike_labels,
        )
        return len(sample.annotations)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        image_path = self._images[idx]
        width, height = _image_size(image_path)
        anns = _read_annotations(
            self._label_path(image_path),
            width=width,
            height=height,
            class_names=self.class_names,
        )
        sample = DOTASample(image_path=image_path, width=width, height=height, annotations=anns)
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


def format_yolo_obb_empty_gt_filter_log(dataset: YoloOBBDataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


__all__ = [
    "YoloOBBDataset",
    "format_yolo_obb_empty_gt_filter_log",
    "load_yolo_obb_class_names",
    "parse_yolo_obb_line",
]
