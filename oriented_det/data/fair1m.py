"""FAIR1M native XML loader yielding DOTASample objects (37 fine-grained classes).

Supported layouts under ``data_root`` (first match wins for discovery)::

Official / TorchGeo::

    train/part1/images/*.tif + train/part1/labelXml/*.xml
    train/part2/images/*.tif + train/part2/labelXml/*.xml
    validation/images/*.tif + validation/labelXml/*.xml

Kaggle (ollypowell FAIR1M JPG dump)::

    Dataset/Images/Train/*.jpg + Dataset/Labels/Train/*.xml
    (also accepts Dataset/labelXml/Train, or XML next to JPG)

    Published dump layout (JPG stems ``t_N`` / ``v_N``, XML stems ``N``)::

        Dataset/Images/{Train,Val}/*.jpg
        Notebook_Working/train_labels/N.xml
        Notebook_Working/val_labels/N.xml

    Optional parquet (``Dataset/labels_geodata.parquet``) documents the same
    ``jpg_file_name`` ↔ ``tiff_file_name`` map; the loader does not require it.

Flat::

    images/* + labelXml/*   (or labels/)

XML objects use four ``point`` corners (``x,y``) under ``objects/object`` with
class name in ``possibleresult/name``. Boxes go through ``polygon_to_rbox``
(**le90**), same as DOTA / HRSC.
"""

from __future__ import annotations

import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from .dota import DOTAAnnotation, DOTASample
from .fair1m_classes import FAIR1M_CLASSES, FAIR1M_CLASS_SET

FAIR1M_SPLIT_NAMES = frozenset({"train", "val", "test", "trainval"})
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
_LABEL_DIR_NAMES = ("labelXml", "labelXmls", "Labels", "labels", "Annotations", "annotations")


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")
    # Satellite rasters routinely exceed Pillow's ~89 MP decompression-bomb limit.
    Image.MAX_IMAGE_PIXELS = None


def _xml_text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return str(node.text).strip()


def parse_fair1m_xml_bytes(raw: bytes) -> ET.Element:
    """Parse FAIR1M XML bytes (UTF-8, then common Chinese encodings)."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb2312", "gbk", "latin-1"):
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
    raise ValueError(f"Could not parse FAIR1M XML ({last_error})")


def normalize_fair1m_class_name(name: str) -> str:
    """Normalize XML class strings to canonical ``FAIR1M_CLASSES`` spellings."""
    raw = str(name).strip()
    if not raw:
        return raw
    if raw in FAIR1M_CLASS_SET:
        return raw
    spaced = raw.replace("_", " ")
    if spaced in FAIR1M_CLASS_SET:
        return spaced
    lower_map = {c.lower(): c for c in FAIR1M_CLASSES}
    hit = lower_map.get(raw.lower()) or lower_map.get(spaced.lower())
    return hit if hit is not None else raw


def fair1m_points_to_annotation(
    points: Sequence[Tuple[float, float]],
    class_name: str,
    *,
    difficult: int = 0,
) -> DOTAAnnotation:
    """Build a DOTAAnnotation from four FAIR1M polygon corners."""
    if len(points) < 4:
        raise ValueError(f"Expected at least 4 points, got {len(points)}")
    coords: List[float] = []
    for x, y in points[:4]:
        coords.extend((float(x), float(y)))
    return DOTAAnnotation.from_corners(
        coords,
        normalize_fair1m_class_name(class_name),
        difficult=int(difficult),
    )


def _parse_point_text(text: str) -> Optional[Tuple[float, float]]:
    parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip()]
    if len(parts) != 2:
        parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_fair1m_xml(path: str | Path) -> Tuple[str, Tuple[DOTAAnnotation, ...]]:
    """Parse one FAIR1M annotation XML.

    Returns:
        ``(image_filename, annotations)``. Filename comes from ``source/filename``
        when present; otherwise the XML stem.
    """
    xml_path = Path(path)
    root = parse_fair1m_xml_bytes(xml_path.read_bytes())

    filename = ""
    source = root.find("source")
    if source is not None:
        filename = _xml_text(source.find("filename"))
    if not filename:
        filename = xml_path.name.replace(".xml", "")

    objects_parent = root.find("objects")
    object_nodes = list(objects_parent.findall("object")) if objects_parent is not None else []
    if not object_nodes:
        object_nodes = list(root.findall("object"))

    annotations: List[DOTAAnnotation] = []
    for obj in object_nodes:
        points_parent = obj.find("points")
        if points_parent is None:
            continue
        pts: List[Tuple[float, float]] = []
        for point in points_parent.findall("point"):
            parsed = _parse_point_text(_xml_text(point))
            if parsed is not None:
                pts.append(parsed)
        if len(pts) < 4:
            continue
        poss = obj.find("possibleresult")
        name = _xml_text(poss.find("name")) if poss is not None else ""
        if not name:
            name = _xml_text(obj.find("name"))
        if not name:
            continue
        difficult_raw = _xml_text(obj.find("difficult")) or "0"
        try:
            difficult = int(float(difficult_raw))
        except ValueError:
            difficult = 0
        try:
            annotations.append(
                fair1m_points_to_annotation(pts, name, difficult=difficult)
            )
        except ValueError:
            continue
    return filename, tuple(annotations)


def resolve_fair1m_root(data_root: str | Path) -> Path:
    """Return the FAIR1M root (accepts a wrapping ``FAIR1M`` / ``Dataset`` folder)."""
    root = Path(data_root)
    if _looks_like_fair1m_root(root):
        return root
    for child_name in ("FAIR1M", "FAIR1M1.0", "FAIR1M2.0", "Dataset", "fair1m"):
        nested = root / child_name
        if _looks_like_fair1m_root(nested):
            return nested
    if (root / "Images").is_dir():
        return root
    raise FileNotFoundError(
        f"FAIR1M root not found under {root}. Expected official train/validation "
        f"layout, Kaggle Dataset/Images/, or images/+labelXml/."
    )


def _looks_like_fair1m_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    if (root / "train").is_dir() or (root / "validation").is_dir():
        return True
    if (root / "Images").is_dir() or (root / "images").is_dir():
        return True
    if (root / "Dataset" / "Images").is_dir():
        return True
    return False


def resolve_fair1m_imageset_split(dataset_cfg, role: str) -> str:
    """Map train-loop role (``train`` / ``val``) to a FAIR1M split name.

    Defaults: train → **train**, val → **val**.
    """
    name = (role or "").strip().lower()
    if name == "train":
        override = getattr(dataset_cfg, "train_split", None)
        return str(override).strip() if override else "train"
    if name == "val":
        override = getattr(dataset_cfg, "val_split", None)
        return str(override).strip() if override else "val"
    if name in FAIR1M_SPLIT_NAMES:
        return name
    raise ValueError(
        f"Unsupported FAIR1M split {role!r}. Expected train, val, test, or trainval."
    )


def _iter_image_files(image_dir: Path) -> List[Path]:
    files: List[Path] = []
    for suffix in _IMAGE_SUFFIXES:
        files.extend(sorted(image_dir.glob(f"*{suffix}")))
        files.extend(sorted(image_dir.glob(f"*{suffix.upper()}")))
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in files:
        if path not in seen and path.is_file():
            seen.add(path)
            ordered.append(path)
    return ordered


def _find_label_dir(image_dir: Path) -> Optional[Path]:
    """Locate a label directory paired with ``image_dir``."""
    parent = image_dir.parent
    for name in _LABEL_DIR_NAMES:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
        if image_dir.name and (parent.parent / name / image_dir.name).is_dir():
            return parent.parent / name / image_dir.name
    if any(image_dir.glob("*.xml")):
        return image_dir
    return None


def _fair1m_layout_search_roots(root: Path) -> List[Path]:
    """Roots that may hold ``Dataset/`` or ``Notebook_Working/`` (Kaggle dump)."""
    roots: List[Path] = [root]
    if root.name.lower() == "dataset" and root.parent.is_dir():
        roots.append(root.parent)
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in roots:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def _ollypowell_xml_stem(image_stem: str) -> Optional[str]:
    """Map Kaggle JPG stem ``t_12`` / ``v_12`` to XML stem ``12``."""
    stem = str(image_stem).strip()
    for prefix in ("t_", "v_", "T_", "V_"):
        if stem.startswith(prefix) and stem[len(prefix) :]:
            return stem[len(prefix) :]
    return None


def _discover_ollypowell_label_dirs(root: Path, split: str) -> List[Path]:
    """Label folders for the ollypowell Kaggle dump (numeric XML stems)."""
    split = split.strip().lower()
    rels: List[str] = []
    if split in {"train", "trainval"}:
        rels.extend(
            (
                "Notebook_Working/train_labels",
                "Dataset/Labels/Train",
                "Dataset/Labels/train",
                "Labels/Train",
                "Labels/train",
            )
        )
    if split in {"val", "trainval"}:
        rels.extend(
            (
                "Notebook_Working/val_labels",
                "Dataset/Labels/Val",
                "Dataset/Labels/val",
                "Labels/Val",
                "Labels/val",
            )
        )
    dirs: List[Path] = []
    seen: set[Path] = set()
    for search_root in _fair1m_layout_search_roots(root):
        for rel in rels:
            candidate = search_root / rel
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            # Prefer flat label dirs (skip nested official mirrors under val_labels/).
            if any(candidate.glob("*.xml")):
                seen.add(resolved)
                dirs.append(candidate)
    return dirs


def _resolve_fair1m_xml_path(
    image_path: Path,
    *,
    label_dir: Optional[Path],
    extra_label_dirs: Sequence[Path] = (),
) -> Optional[Path]:
    """Resolve XML for one image (same stem, then ollypowell ``t_N``→``N``)."""
    stem = image_path.stem
    numeric = _ollypowell_xml_stem(stem)
    search_dirs: List[Path] = []
    if label_dir is not None:
        search_dirs.append(label_dir)
    for directory in extra_label_dirs:
        if directory not in search_dirs:
            search_dirs.append(directory)

    candidates: List[Path] = []
    for directory in search_dirs:
        candidates.append(directory / f"{stem}.xml")
        candidates.append(directory / f"{image_path.name}.xml")
        if numeric is not None:
            candidates.append(directory / f"{numeric}.xml")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _discover_split_image_dirs(root: Path, split: str) -> List[Path]:
    """Return image directories that belong to ``split``."""
    split = split.strip().lower()
    dirs: List[Path] = []

    if split in {"train", "trainval"}:
        for part in ("part1", "part2"):
            for img_name in ("images", "Images", "images-1", "images-2"):
                candidate = root / "train" / part / img_name
                if candidate.is_dir():
                    dirs.append(candidate)
        for img_name in ("images", "Images"):
            candidate = root / "train" / img_name
            if candidate.is_dir():
                dirs.append(candidate)
        for base in (root, root / "Dataset"):
            for name in ("Train", "train"):
                candidate = base / "Images" / name
                if candidate.is_dir():
                    dirs.append(candidate)
                candidate = base / "images" / name
                if candidate.is_dir():
                    dirs.append(candidate)

    if split in {"val", "trainval"}:
        for base in (root, root / "Dataset"):
            for folder in ("validation", "val", "Val", "Validation"):
                for img_name in ("images", "Images", ""):
                    candidate = base / folder / img_name if img_name else base / folder
                    if candidate.is_dir() and _iter_image_files(candidate):
                        dirs.append(candidate)
            for name in ("Val", "val", "Validation", "validation"):
                candidate = base / "Images" / name
                if candidate.is_dir():
                    dirs.append(candidate)

    if split == "test":
        for base in (root, root / "Dataset"):
            for folder in ("test", "Test"):
                for img_name in ("images", "Images", ""):
                    candidate = base / folder / img_name if img_name else base / folder
                    if candidate.is_dir():
                        nested = [
                            p
                            for p in candidate.iterdir()
                            if p.is_dir() and _iter_image_files(p)
                        ]
                        if nested:
                            dirs.extend(sorted(nested))
                        elif _iter_image_files(candidate):
                            dirs.append(candidate)
            for name in ("Test", "test"):
                candidate = base / "Images" / name
                if candidate.is_dir():
                    dirs.append(candidate)

    if split in {"train", "trainval"} and not dirs:
        for name in ("images", "Images"):
            candidate = root / name
            if candidate.is_dir() and _iter_image_files(candidate):
                dirs.append(candidate)

    seen: set[Path] = set()
    unique: List[Path] = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(d)
    return unique


def discover_fair1m_pairs(root: str | Path, split: str) -> List[Tuple[Path, Optional[Path]]]:
    """Return ``(image_path, xml_path_or_None)`` pairs for a split."""
    root = resolve_fair1m_root(root)
    split = split.strip().lower()
    if split not in FAIR1M_SPLIT_NAMES:
        raise ValueError(f"Unsupported FAIR1M split {split!r}")

    pairs: List[Tuple[Path, Optional[Path]]] = []
    image_dirs = _discover_split_image_dirs(root, split)
    if not image_dirs:
        raise FileNotFoundError(
            f"No FAIR1M image directories found for split {split!r} under {root}"
        )

    extra_label_dirs = _discover_ollypowell_label_dirs(root, split)
    for image_dir in image_dirs:
        label_dir = _find_label_dir(image_dir)
        for image_path in _iter_image_files(image_dir):
            xml_path = _resolve_fair1m_xml_path(
                image_path,
                label_dir=label_dir,
                extra_label_dirs=extra_label_dirs,
            )
            pairs.append((image_path, xml_path))
    return pairs


def fair1m_holdout_stems(
    stems: Sequence[str],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Deterministic image-level train/val split (split **before** tiling)."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    ordered = sorted({str(s).strip() for s in stems if str(s).strip()})
    train: List[str] = []
    val: List[str] = []
    for stem in ordered:
        digest = hashlib.md5(f"{int(seed)}:{stem}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
        if bucket < float(val_fraction):
            val.append(stem)
        else:
            train.append(stem)
    if not train or not val:
        raise ValueError(
            f"Holdout produced empty split (train={len(train)}, val={len(val)}); "
            "adjust val_fraction or provide more stems."
        )
    return train, val


def write_fair1m_holdout_split_files(
    stems: Sequence[str],
    output_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[Path, Path]:
    """Write ``train.txt`` / ``val.txt`` stem lists under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_stems, val_stems = fair1m_holdout_stems(
        stems, val_fraction=val_fraction, seed=seed
    )
    train_path = out / "train.txt"
    val_path = out / "val.txt"
    train_path.write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_stems) + "\n", encoding="utf-8")
    return train_path, val_path


def read_fair1m_stem_list(path: str | Path) -> List[str]:
    """Read one stem per line from a split file."""
    stems: List[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        stem = Path(token).stem
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


class FAIR1MDataset:
    """FAIR1M dataset yielding ``DOTASample`` (37 fine-grained classes)."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        difficult_strategy: str = "drop",
        filter_empty_gt: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
        map_labels: Optional[Dict[str, str]] = None,
        stem_list: Optional[Sequence[str]] = None,
        stem_list_file: Optional[str | Path] = None,
    ):
        from .lookalike import resolve_lookalike_label_set

        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(
                f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'."
            )
        split_name = split.strip().lower()
        if split_name not in FAIR1M_SPLIT_NAMES:
            raise ValueError(
                f"Unsupported FAIR1M split {split!r}. Expected train, val, test, or trainval."
            )

        self.root = resolve_fair1m_root(data_root)
        self.split = split_name
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self.map_labels = dict(map_labels or {})
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        pairs = discover_fair1m_pairs(self.root, self.split)
        if stem_list_file is not None:
            allowed_stems = set(read_fair1m_stem_list(stem_list_file))
            pairs = [(img, xml) for img, xml in pairs if img.stem in allowed_stems]
        elif stem_list is not None:
            allowed_stems = {str(s).strip() for s in stem_list if str(s).strip()}
            pairs = [(img, xml) for img, xml in pairs if img.stem in allowed_stems]

        self._pairs_discovered_count = len(pairs)
        kept: List[Tuple[Path, Optional[Path]]] = []
        for image_path, xml_path in pairs:
            if self.filter_empty_gt and self._effective_gt_count(image_path, xml_path) == 0:
                continue
            kept.append((image_path, xml_path))
        self._pairs = kept
        self._annotation_files = [xml for _, xml in self._pairs if xml is not None]
        self._empty_gt_filtered_count = self._pairs_discovered_count - len(self._pairs)

    @property
    def tiles_discovered_count(self) -> int:
        return self._pairs_discovered_count

    @property
    def annotation_files_discovered_count(self) -> int:
        return self._pairs_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        return self._empty_gt_filtered_count

    def _load_annotations(self, xml_path: Optional[Path]) -> Tuple[DOTAAnnotation, ...]:
        if xml_path is None or not xml_path.is_file():
            return tuple()
        _, annotations = parse_fair1m_xml(xml_path)
        if not self.map_labels:
            return annotations
        mapped: List[DOTAAnnotation] = []
        for ann in annotations:
            new_name = self.map_labels.get(ann.class_name, ann.class_name)
            if new_name == ann.class_name:
                mapped.append(ann)
            else:
                mapped.append(
                    DOTAAnnotation(
                        class_name=new_name,
                        difficult=ann.difficult,
                        polygon=ann.polygon,
                        rbox=ann.rbox,
                    )
                )
        return tuple(mapped)

    def _effective_gt_count(self, image_path: Path, xml_path: Optional[Path]) -> int:
        annotations = self._load_annotations(xml_path)
        if not annotations:
            return 0
        sample = DOTASample(
            image_path=image_path,
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
        return len(self._pairs)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        image_path, xml_path = self._pairs[idx]
        annotations = self._load_annotations(xml_path)
        width, height = self._load_image_size(image_path)
        sample = DOTASample(
            image_path=image_path,
            width=width,
            height=height,
            annotations=annotations,
        )
        if (
            self.allowed_classes is not None
            or self.ignore_labels is not None
            or self._drop_difficult
        ):
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
        """Canonical 37-class list (or mapped coarse names when ``map_labels`` is set)."""
        if self.map_labels:
            from .lookalike import filter_semantic_class_names

            names = {self.map_labels.get(c, c) for c in FAIR1M_CLASSES}
            return filter_semantic_class_names(names, self._lookalike_set)
        return list(FAIR1M_CLASSES)


def format_fair1m_empty_gt_filter_log(dataset: FAIR1MDataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


def _normalize_export_image_format(image_format: str) -> str:
    fmt = (image_format or "original").strip().lower()
    if fmt == "jpeg":
        return "jpg"
    if fmt not in {"original", "png", "jpg"}:
        raise ValueError(
            f"Unsupported image_format={image_format!r}; expected original, png, or jpg"
        )
    return fmt


def _export_fair1m_image(
    source: Path,
    dest_dir: Path,
    stem: str,
    *,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> Path:
    """Copy or convert one FAIR1M raster into the DOTA export folder."""
    _require_pillow()
    fmt = _normalize_export_image_format(image_format)
    src_suf = source.suffix.lower()
    if fmt == "original":
        if src_suf in {".jpg", ".jpeg"}:
            dest = dest_dir / f"{stem}.jpg"
            if not dest.exists():
                shutil.copy2(source, dest)
            return dest
        if src_suf == ".png":
            dest = dest_dir / f"{stem}.png"
            if not dest.exists():
                shutil.copy2(source, dest)
            return dest
        fmt = "png"
    dest = dest_dir / (f"{stem}.jpg" if fmt == "jpg" else f"{stem}.png")
    if dest.exists():
        return dest
    with Image.open(source) as img:
        rgb = img.convert("RGB")
        if fmt == "jpg":
            rgb.save(dest, format="JPEG", quality=int(jpeg_quality), subsampling=0)
        else:
            rgb.save(dest)
    return dest


def export_fair1m_to_dota(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "val"),
    difficult_strategy: str = "keep",
    same_folder: bool = False,
    val_fraction: Optional[float] = None,
    split_seed: int = 0,
    map_labels: Optional[Dict[str, str]] = None,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> Dict[str, int]:
    """Write FAIR1M splits as DOTA-format images + ``.txt`` folders.

    When ``val_fraction`` is set, build a deterministic image-level holdout from
    ``train`` images before writing (and before any later tiling).

    ``image_format``: ``original`` copies JPEG/PNG as-is (TIFF/BMP → PNG);
    ``png`` / ``jpg`` force a convert. ``odet tile-dota`` reads JPEG and PNG.
    """
    _require_pillow()
    out_root = Path(output_dir)
    counts: Dict[str, int] = {}

    split_list = [s.strip().lower() for s in splits if str(s).strip()]
    stem_filters: Dict[str, Optional[List[str]]] = {s: None for s in split_list}

    if val_fraction is not None:
        if "train" not in split_list or "val" not in split_list:
            raise ValueError(
                "val_fraction requires splits to include both 'train' and 'val'"
            )
        train_pairs = discover_fair1m_pairs(data_root, "train")
        all_stems = [img.stem for img, _ in train_pairs]
        train_stems, val_stems = fair1m_holdout_stems(
            all_stems, val_fraction=float(val_fraction), seed=int(split_seed)
        )
        write_fair1m_holdout_split_files(
            all_stems,
            out_root / "ImageSets",
            val_fraction=float(val_fraction),
            seed=int(split_seed),
        )
        stem_filters["train"] = train_stems
        stem_filters["val"] = val_stems

    for split in split_list:
        source_split = (
            "train"
            if (val_fraction is not None and split in {"train", "val"})
            else split
        )
        dataset = FAIR1MDataset(
            data_root,
            split=source_split,
            difficult_strategy=difficult_strategy,
            filter_empty_gt=False,
            map_labels=map_labels,
            stem_list=stem_filters.get(split),
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
            _export_fair1m_image(
                sample.image_path,
                image_dir,
                stem,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            lines = [ann.to_line() for ann in sample.annotations]
            (label_dir / f"{stem}.txt").write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            n += 1
        counts[split] = n
    return counts


__all__ = [
    "FAIR1M_CLASSES",
    "FAIR1M_CLASS_SET",
    "FAIR1M_SPLIT_NAMES",
    "FAIR1MDataset",
    "discover_fair1m_pairs",
    "export_fair1m_to_dota",
    "fair1m_holdout_stems",
    "fair1m_points_to_annotation",
    "format_fair1m_empty_gt_filter_log",
    "normalize_fair1m_class_name",
    "parse_fair1m_xml",
    "parse_fair1m_xml_bytes",
    "read_fair1m_stem_list",
    "resolve_fair1m_imageset_split",
    "resolve_fair1m_root",
    "write_fair1m_holdout_split_files",
]
