"""HRSC2016 native XML loader (single-class ship) yielding DOTASample objects.

Official layout (NWPU HRSC2016)::

    HRSC2016/
      FullDataSet/AllImages/*.bmp
      FullDataSet/Annotations/*.xml
      ImageSets/{train,val,test,trainval}.txt

``mbox_cx/cy/w/h/ang`` in each XML object is a rotated box with **angle in radians**.
Boxes are converted through the same polygon → RBox path as DOTA so training uses
**le90** (long-edge, ``[-π/2, π/2)``). Fine-grained ``Class_ID`` values are ignored;
every object is ``ship``.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..geometry import RBox, transforms
from .dota import DOTAAnnotation, DOTASample

HRSC2016_CLASSES: list[str] = ["ship"]
HRSC2016_CLASS_SET = frozenset(HRSC2016_CLASSES)
HRSC2016_IMAGESET_NAMES = frozenset({"train", "val", "test", "trainval"})
_IMAGE_SUFFIXES = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")


def _xml_text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return str(node.text).strip()


def parse_hrsc2016_xml_bytes(raw: bytes) -> ET.Element:
    """Parse HRSC XML bytes, trying UTF-8 then GB2312 (common on the original release)."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "gb2312", "gbk", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            return ET.fromstring(text)
        except ET.ParseError as exc:
            last_error = exc
            continue
    raise ValueError(f"Could not parse HRSC2016 XML ({last_error})")


def hrsc_mbox_to_annotation(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_rad: float,
    *,
    difficult: int = 0,
    class_name: str = "ship",
) -> DOTAAnnotation:
    """Convert an HRSC rotated box (radians) to a DOTAAnnotation (polygon + RBox)."""
    raw = RBox(cx, cy, width, height, angle_rad)
    polygon = raw.to_polygon()
    rbox = transforms.polygon_to_rbox(polygon)
    return DOTAAnnotation(
        class_name=class_name,
        difficult=int(difficult),
        polygon=polygon,
        rbox=rbox,
    )


def parse_hrsc2016_xml(path: str | Path) -> Tuple[int, int, Tuple[DOTAAnnotation, ...]]:
    """Parse one HRSC annotation XML.

    Returns:
        ``(img_width, img_height, annotations)``. Width/height come from the XML
        header when present; otherwise ``(0, 0)``.
    """
    xml_path = Path(path)
    root = parse_hrsc2016_xml_bytes(xml_path.read_bytes())

    width = int(float(_xml_text(root.find("Img_SizeWidth")) or "0"))
    height = int(float(_xml_text(root.find("Img_SizeHeight")) or "0"))

    objects_parent = root.find("HRSC_Objects")
    object_nodes = list(objects_parent.findall("HRSC_Object")) if objects_parent is not None else []
    if not object_nodes:
        object_nodes = list(root.findall("HRSC_Object"))

    annotations: List[DOTAAnnotation] = []
    for obj in object_nodes:
        try:
            cx = float(_xml_text(obj.find("mbox_cx")))
            cy = float(_xml_text(obj.find("mbox_cy")))
            w = float(_xml_text(obj.find("mbox_w")))
            h = float(_xml_text(obj.find("mbox_h")))
            ang = float(_xml_text(obj.find("mbox_ang")))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        difficult_raw = _xml_text(obj.find("difficult")) or "0"
        try:
            difficult = int(float(difficult_raw))
        except ValueError:
            difficult = 0
        try:
            annotations.append(
                hrsc_mbox_to_annotation(cx, cy, w, h, ang, difficult=difficult)
            )
        except ValueError:
            continue
    return width, height, tuple(annotations)


def resolve_hrsc2016_root(data_root: str | Path) -> Path:
    """Return the directory that contains ``FullDataSet/`` and ``ImageSets/``."""
    root = Path(data_root)
    if (root / "FullDataSet").is_dir() and (root / "ImageSets").is_dir():
        return root
    nested = root / "HRSC2016"
    if (nested / "FullDataSet").is_dir() and (nested / "ImageSets").is_dir():
        return nested
    raise FileNotFoundError(
        f"HRSC2016 root not found under {root}. Expected FullDataSet/ and ImageSets/ "
        f"(or HRSC2016/FullDataSet and HRSC2016/ImageSets)."
    )


def resolve_hrsc2016_imageset_split(dataset_cfg, role: str) -> str:
    """Map train-loop role (``train`` / ``val``) to an ImageSets name.

    Defaults match MMRotate: train on **trainval**, evaluate on **test**.
    """
    name = (role or "").strip().lower()
    if name == "train":
        override = getattr(dataset_cfg, "train_split", None)
        return str(override).strip() if override else "trainval"
    if name == "val":
        override = getattr(dataset_cfg, "val_split", None)
        return str(override).strip() if override else "test"
    if name in HRSC2016_IMAGESET_NAMES:
        return name
    raise ValueError(
        f"Unsupported HRSC2016 split {role!r}. Expected train, val, test, or trainval."
    )


def _imageset_file(root: Path, split: str) -> Path:
    direct = root / "ImageSets" / f"{split}.txt"
    if direct.is_file():
        return direct
    nested = root / "ImageSets" / "Main" / f"{split}.txt"
    if nested.is_file():
        return nested
    raise FileNotFoundError(
        f"HRSC2016 ImageSets file not found for split {split!r}: tried {direct} and {nested}"
    )


def read_hrsc2016_imageset(root: str | Path, split: str) -> List[str]:
    """Return image ids listed in ``ImageSets/{split}.txt`` (no extension)."""
    root = Path(root)
    split = split.strip().lower()
    if split == "trainval":
        trainval_direct = root / "ImageSets" / "trainval.txt"
        trainval_main = root / "ImageSets" / "Main" / "trainval.txt"
        if not trainval_direct.is_file() and not trainval_main.is_file():
            train_ids = read_hrsc2016_imageset(root, "train")
            val_ids = read_hrsc2016_imageset(root, "val")
            seen: set[str] = set()
            ordered: List[str] = []
            for image_id in train_ids + val_ids:
                if image_id not in seen:
                    seen.add(image_id)
                    ordered.append(image_id)
            return ordered

    path = _imageset_file(root, split)
    ids: List[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        image_id = Path(token).stem
        if image_id not in seen:
            seen.add(image_id)
            ids.append(image_id)
    return ids


def find_hrsc2016_image(image_dir: Path, image_id: str) -> Optional[Path]:
    for suffix in _IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


class HRSC2016Dataset:
    """HRSC2016 dataset yielding ``DOTASample`` (single-class ``ship``)."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "trainval",
        difficult_strategy: str = "drop",
        filter_empty_gt: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
    ):
        from .lookalike import resolve_lookalike_label_set

        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(
                f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'."
            )
        split_name = split.strip().lower()
        if split_name not in HRSC2016_IMAGESET_NAMES:
            raise ValueError(
                f"Unsupported HRSC2016 split {split!r}. Expected train, val, test, or trainval."
            )

        self.root = resolve_hrsc2016_root(data_root)
        self.split = split_name
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        self.image_dir = self.root / "FullDataSet" / "AllImages"
        self.ann_dir = self.root / "FullDataSet" / "Annotations"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"HRSC2016 image directory not found: {self.image_dir}")
        if not self.ann_dir.is_dir():
            raise FileNotFoundError(f"HRSC2016 annotation directory not found: {self.ann_dir}")

        discovered_ids = read_hrsc2016_imageset(self.root, self.split)
        self._image_ids_discovered_count = len(discovered_ids)

        kept: List[str] = []
        for image_id in discovered_ids:
            if self.filter_empty_gt and self._effective_gt_count(image_id) == 0:
                continue
            kept.append(image_id)
        self._image_ids = kept
        # Reuse DOTA stem helper in train.py (looks for ``_annotation_files``).
        self._annotation_files = [self.ann_dir / f"{image_id}.xml" for image_id in self._image_ids]
        self._empty_gt_filtered_count = self._image_ids_discovered_count - len(self._image_ids)

    @property
    def tiles_discovered_count(self) -> int:
        return self._image_ids_discovered_count

    @property
    def annotation_files_discovered_count(self) -> int:
        return self._image_ids_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        return self._empty_gt_filtered_count

    def _xml_path(self, image_id: str) -> Path:
        return self.ann_dir / f"{image_id}.xml"

    def _parse_annotations(self, image_id: str) -> Tuple[DOTAAnnotation, ...]:
        xml_path = self._xml_path(image_id)
        if not xml_path.is_file():
            warnings.warn(f"HRSC2016 annotation missing for {image_id}: {xml_path}", UserWarning)
            return tuple()
        _, _, annotations = parse_hrsc2016_xml(xml_path)
        return annotations

    def _effective_gt_count(self, image_id: str) -> int:
        annotations = self._parse_annotations(image_id)
        if not annotations:
            return 0
        sample = DOTASample(
            image_path=self._xml_path(image_id),
            width=0,
            height=0,
            annotations=annotations,
        )
        sample = sample.filter_by_class(
            allowed_classes=self.allowed_classes,
            ignore_labels=self.ignore_labels,
            drop_difficult=self._drop_difficult,
            lookalike_labels=self.lookalike_labels,
        )
        return len(sample.annotations)

    def _load_image_size(self, image_path: Path) -> Tuple[int, int]:
        _require_pillow()
        with Image.open(image_path) as img:
            return img.size

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._image_ids):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        image_id = self._image_ids[idx]
        image_path = find_hrsc2016_image(self.image_dir, image_id)
        if image_path is None:
            raise FileNotFoundError(
                f"HRSC2016 image not found for id {image_id} under {self.image_dir}"
            )
        _, _, annotations = parse_hrsc2016_xml(self._xml_path(image_id))
        width, height = self._load_image_size(image_path)
        sample = DOTASample(
            image_path=image_path,
            width=width,
            height=height,
            annotations=annotations,
        )
        if self.allowed_classes is not None or self.ignore_labels is not None or self._drop_difficult:
            sample = sample.filter_by_class(
                allowed_classes=self.allowed_classes,
                ignore_labels=self.ignore_labels,
                drop_difficult=self._drop_difficult,
                lookalike_labels=self.lookalike_labels,
            )
        return sample

    def __iter__(self) -> Iterator[DOTASample]:
        for idx in range(len(self)):
            yield self[idx]

    def get_class_names(self) -> List[str]:
        """Always ``['ship']`` — HRSC2016 is trained as a single-class dataset."""
        return list(HRSC2016_CLASSES)


def format_hrsc_empty_gt_filter_log(dataset: HRSC2016Dataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


def export_hrsc2016_to_dota(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = ("trainval", "test"),
    difficult_strategy: str = "keep",
    same_folder: bool = False,
) -> Dict[str, int]:
    """Write HRSC2016 splits as DOTA-format PNG + ``.txt`` folders.

    Layout (``same_folder=False``)::

        output_dir/{split}/images/*.png
        output_dir/{split}/labels/*.txt

    After this, ``tools/tile_dota.py`` can tile a split if images are larger than
    the training canvas. Native ``dataset.format: hrsc2016`` training does not
    require this conversion.
    """
    _require_pillow()
    out_root = Path(output_dir)
    counts: Dict[str, int] = {}
    for split in splits:
        dataset = HRSC2016Dataset(
            data_root,
            split=split,
            difficult_strategy=difficult_strategy,
            filter_empty_gt=False,
        )
        split_dir = out_root / split
        if same_folder:
            image_dir = split_dir
            label_dir = split_dir
        else:
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for sample in dataset:
            stem = sample.image_path.stem
            dest_image = image_dir / f"{stem}.png"
            with Image.open(sample.image_path) as img:
                img.convert("RGB").save(dest_image)
            lines = [ann.to_line() for ann in sample.annotations]
            (label_dir / f"{stem}.txt").write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            n += 1
        counts[split] = n
    return counts


__all__ = [
    "HRSC2016_CLASSES",
    "HRSC2016_CLASS_SET",
    "HRSC2016_IMAGESET_NAMES",
    "HRSC2016Dataset",
    "export_hrsc2016_to_dota",
    "find_hrsc2016_image",
    "format_hrsc_empty_gt_filter_log",
    "hrsc_mbox_to_annotation",
    "parse_hrsc2016_xml",
    "parse_hrsc2016_xml_bytes",
    "read_hrsc2016_imageset",
    "resolve_hrsc2016_imageset_split",
    "resolve_hrsc2016_root",
]
