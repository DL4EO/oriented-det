"""Airbus Playground CSV generation and runtime dataset loader."""

from __future__ import annotations

import csv
import json
import random
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from .dota import DOTAAnnotation, DOTASample


@dataclass(frozen=True)
class TileRecord:
    """A single tile discovered in an Airbus Playground export."""

    dataset_id: str
    zone_id: str
    image_id: str
    tile_id: str
    tile_relpath: str
    label_relpath: str


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")


def _resolve_csv_path(data_root: Path, path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    if path.is_absolute():
        return path
    return data_root / path


_DATED_PLAYGROUND_CSV_RE = re.compile(r"^(annotations|split)_(\d{8})\.csv$")


def _playground_csv_date_stamp(
    when: datetime | None = None,
    *,
    utc: bool = True,
) -> str:
    dt = when if when is not None else datetime.now(timezone.utc if utc else None)
    if utc and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d")


def dated_playground_csv_filenames(
    when: datetime | None = None,
    *,
    utc: bool = True,
) -> Tuple[str, str]:
    """Return paired dated CSV basenames from one run, e.g. annotations/split ``_20260516.csv``."""
    stamp = _playground_csv_date_stamp(when, utc=utc)
    return f"annotations_{stamp}.csv", f"split_{stamp}.csv"


def timestamped_split_filename(
    when: datetime | None = None,
    *,
    utc: bool = True,
) -> str:
    """Return the dated split CSV basename from :func:`dated_playground_csv_filenames`."""
    return dated_playground_csv_filenames(when, utc=utc)[1]


def timestamped_annotations_filename(
    when: datetime | None = None,
    *,
    utc: bool = True,
) -> str:
    """Return the dated annotations CSV basename from :func:`dated_playground_csv_filenames`."""
    return dated_playground_csv_filenames(when, utc=utc)[0]


def resolve_playground_csv_filenames(
    annotations_file: str | None,
    split_file: str | None,
    when: datetime | None = None,
    *,
    utc: bool = True,
) -> Tuple[str, str]:
    """Resolve CLI/config CSV names; default is a same-day annotations + split pair."""
    if annotations_file is None and split_file is None:
        return dated_playground_csv_filenames(when, utc=utc)

    ann = annotations_file or "annotations.csv"
    split = split_file or "split.csv"
    if annotations_file is None and split_file is not None:
        paired = _paired_dated_playground_csv(split_file, want="annotations")
        if paired is not None:
            ann = paired
    if split_file is None and annotations_file is not None:
        paired = _paired_dated_playground_csv(annotations_file, want="split")
        if paired is not None:
            split = paired
    return ann, split


def _paired_dated_playground_csv(name: str, *, want: Literal["annotations", "split"]) -> str | None:
    m = _DATED_PLAYGROUND_CSV_RE.match(name)
    if not m or m.group(1) == want:
        return None
    return f"{want}_{m.group(2)}.csv"


def _normalize_dota_coords(coords: Sequence[float]) -> str:
    # Keep compact but stable formatting for portability/readability.
    return " ".join(f"{float(value):.3f}" for value in coords)


def _split_playground_tags(class_name: str) -> List[str]:
    """Split a Playground concatenated class_name on commas into stripped tags."""
    return [p.strip() for p in str(class_name).split(",") if p.strip()]


def apply_airbus_difficult_tags(
    class_name: str,
    *,
    difficult_tags: Optional[Sequence[str]] = None,
    base_difficult: int = 0,
) -> Tuple[str, int]:
    """Strip configured difficult tags and set difficult=1 when any match.

    Playground stores multi-tag objects as ``", ".join(sorted(tags))``. Exact tag
    match (not substring): if any part is in ``difficult_tags``, set difficult=1
    (OR with ``base_difficult``) and rejoin remaining parts with ``", "``.

    Raises:
        ValueError: if stripping leaves an empty semantic name.
    """
    tags = _split_playground_tags(class_name)
    if not tags:
        raise ValueError(f"Empty class_name after tag split: {class_name!r}")

    difficult_set = {str(t).strip() for t in (difficult_tags or []) if str(t).strip()}
    if not difficult_set:
        return str(class_name).strip(), int(base_difficult)

    has_difficult = any(t in difficult_set for t in tags)
    remaining = [t for t in tags if t not in difficult_set]
    if not remaining:
        raise ValueError(
            f"class_name {class_name!r} has only difficult_tags {sorted(difficult_set)}; "
            "no semantic class left after stripping. Map or filter this label explicitly."
        )
    semantic = ", ".join(remaining)
    difficult = 1 if has_difficult else 0
    return semantic, int(base_difficult) | difficult


def discover_airbus_tiles(data_root: str | Path) -> List[TileRecord]:
    """Discover all tiles in Airbus Playground export folders.

    Expected layout:
      data_root/dataset_id/samples/zone_id/image_id/tile_id.jpg
      data_root/dataset_id/labels/zone_id/tile_id.json
    """
    root = Path(data_root)
    records: List[TileRecord] = []

    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        samples_dir = dataset_dir / "samples"
        labels_dir = dataset_dir / "labels"
        if not samples_dir.exists() or not labels_dir.exists():
            continue

        dataset_id = dataset_dir.name
        for zone_dir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
            zone_id = zone_dir.name
            for image_dir in sorted(p for p in zone_dir.iterdir() if p.is_dir()):
                image_id = image_dir.name
                for tile_path in sorted(image_dir.glob("*.jpg")):
                    tile_id = tile_path.stem
                    label_path = labels_dir / zone_id / f"{tile_id}.json"
                    records.append(
                        TileRecord(
                            dataset_id=dataset_id,
                            zone_id=zone_id,
                            image_id=image_id,
                            tile_id=tile_id,
                            tile_relpath=str(tile_path.relative_to(root)),
                            label_relpath=str(label_path.relative_to(root)),
                        )
                    )
    return records


def _load_airbus_label_rows(
    label_path: Path,
    *,
    difficulty: int = 0,
    ignore_labels: Optional[set[str]] = None,
    map_labels: Optional[Dict[str, str]] = None,
    difficult_tags: Optional[Sequence[str]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, int]]:
    """Parse one Airbus label JSON into DOTA coordinate rows.

    Returns list of tuples: (dota_coords, class_name, difficult).

    When ``difficult_tags`` is set (e.g. ``["Partially Hidden"]``), matching tags
    are stripped from the semantic class name and ``difficult`` is set to 1.
    ``map_labels`` is applied to the remaining concatenated name (exact match).
    """
    if not label_path.exists():
        return []

    try:
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("shapely is required to convert polygons to OBB.") from exc

    content = json.loads(label_path.read_text(encoding="utf-8"))
    features = content.get("features", [])
    rows: List[Tuple[str, str, int]] = []
    difficult_set = {str(t).strip() for t in (difficult_tags or []) if str(t).strip()}

    for feature in features:
        properties = feature.get("properties", {}) or {}
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        if properties.get("mask", False):
            continue

        tags = properties.get("tags", [])
        if not tags:
            continue
        raw_tags = sorted(str(t).strip() for t in tags if str(t).strip())
        if not raw_tags:
            continue
        # Airbus uses many tags per object; sort then concatenate with ", " for stable labels
        raw_class_name = ", ".join(raw_tags)
        if stats is not None:
            stats["raw_label_counts"][raw_class_name] += 1

        has_difficult = any(t in difficult_set for t in raw_tags)
        semantic_tags = [t for t in raw_tags if t not in difficult_set]
        if has_difficult and not semantic_tags:
            raise ValueError(
                f"{label_path}: tags {raw_tags} have only difficult_tags "
                f"{sorted(difficult_set)}; no semantic class left after stripping."
            )
        class_name = ", ".join(semantic_tags) if semantic_tags else raw_class_name
        object_difficult = 1 if has_difficult else int(difficulty)

        if ignore_labels is not None and class_name in ignore_labels:
            if stats is not None:
                stats["ignored_label_counts"][class_name] += 1
                stats["ignored_objects"] += 1
            continue

        mapped_name = map_labels.get(class_name, class_name) if map_labels is not None else class_name
        if mapped_name != class_name and stats is not None:
            stats["mapped_objects"] += 1
            stats["mapping_pairs"][f"{class_name}->{mapped_name}"] += 1

        try:
            polygon = shape(geometry)
            obb = polygon.minimum_rotated_rectangle
            coords = list(obb.exterior.coords)
        except Exception:
            continue

        if len(coords) >= 5:
            coords = coords[:-1]
        if len(coords) != 4:
            continue

        flat = [float(v) for xy in coords for v in xy]
        rows.append((_normalize_dota_coords(flat), mapped_name, int(object_difficult)))
        if stats is not None:
            stats["final_label_counts"][mapped_name] += 1
            stats["kept_objects"] += 1

    return rows


def _image_group_key(tile: TileRecord) -> Tuple[str, str, str]:
    return (tile.dataset_id, tile.zone_id, tile.image_id)


def detect_airbus_split_csv_format(split_values: Sequence[str]) -> Literal["train_val", "fold_ids"]:
    """Infer whether split.csv uses legacy train/val strings or integer fold ids.

    Fold mode: every non-empty value must parse as a non-negative int. Training code treats
    fold ``val_split_id`` (default 0) as validation and all other folds as training, unless
    ``train_includes_val`` is enabled (train on all folds; val fold for monitoring only).
    """
    non_empty = [str(s).strip() for s in split_values if s is not None and str(s).strip() != ""]
    if not non_empty:
        raise ValueError("split.csv contains no non-empty split values.")
    lowered = {s.lower() for s in non_empty}
    if lowered <= {"train", "val"}:
        return "train_val"
    for s in non_empty:
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(
                f"split.csv: expected integer fold id or 'train'/'val'; got {s!r}"
            ) from exc
        if v < 0:
            raise ValueError(f"split.csv: fold id must be non-negative; got {v}")
    return "fold_ids"


def generate_airbus_playground_csvs(
    data_root: str | Path,
    *,
    annotations_file: str = "annotations.csv",
    split_file: str = "split.csv",
    num_splits: int = 10,
    seed: int = 42,
    ignore_labels: Optional[Sequence[str]] = None,
    map_labels: Optional[Dict[str, str]] = None,
    difficult_tags: Optional[Sequence[str]] = None,
    include_stats: bool = False,
) -> Tuple[Path, Path] | Tuple[Path, Path, Dict[str, Any]]:
    """Generate annotations.csv and split.csv from Airbus export folders.

    Each image group ``(dataset_id, zone_id, image_id)`` is assigned an integer fold id in
    ``0 .. num_splits-1`` (balanced round-robin on a shuffled group list). The ``split`` column
    stores that integer as text. By convention fold ``0`` is the default validation fold; set
    ``dataset.val_split_id`` in the training config to use another fold as validation.

    When ``difficult_tags`` is set (e.g. ``["Partially Hidden"]``), matching tags are stripped
    from ``class_name`` and the CSV ``difficult`` column is set to 1.
    """
    if num_splits < 2:
        raise ValueError("num_splits must be at least 2.")

    root = Path(data_root)
    tiles = discover_airbus_tiles(root)
    if not tiles:
        raise FileNotFoundError(f"No Airbus Playground tiles found under {root}")

    groups: Dict[Tuple[str, str, str], List[TileRecord]] = defaultdict(list)
    for tile in tiles:
        groups[_image_group_key(tile)].append(tile)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    group_to_fold = {gk: str(i % num_splits) for i, gk in enumerate(group_keys)}

    ignore_set = set(ignore_labels or [])
    map_dict = dict(map_labels or {})
    stats: Dict[str, Any] = {
        "raw_label_counts": Counter(),
        "ignored_label_counts": Counter(),
        "final_label_counts": Counter(),
        "mapping_pairs": Counter(),
        "ignored_objects": 0,
        "mapped_objects": 0,
        "kept_objects": 0,
    }

    split_rows: List[Dict[str, str]] = []
    annotation_rows: List[Dict[str, str]] = []
    for tile in sorted(tiles, key=lambda x: x.tile_relpath):
        split_value = group_to_fold[_image_group_key(tile)]
        split_rows.append(
            {
                "tile_relpath": tile.tile_relpath,
                "dataset_id": tile.dataset_id,
                "zone_id": tile.zone_id,
                "image_id": tile.image_id,
                "tile_id": tile.tile_id,
                "split": split_value,
            }
        )

        label_rows = _load_airbus_label_rows(
            root / tile.label_relpath,
            ignore_labels=ignore_set,
            map_labels=map_dict,
            difficult_tags=difficult_tags,
            stats=stats,
        )
        for dota_coords, class_name, difficult in label_rows:
            annotation_rows.append(
                {
                    "tile_relpath": tile.tile_relpath,
                    "dataset_id": tile.dataset_id,
                    "zone_id": tile.zone_id,
                    "image_id": tile.image_id,
                    "tile_id": tile.tile_id,
                    "dota_coords": dota_coords,
                    "class_name": class_name,
                    "difficult": str(difficult),
                }
            )

    annotations_path = _resolve_csv_path(root, annotations_file)
    split_path = _resolve_csv_path(root, split_file)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_relpath",
                "dataset_id",
                "zone_id",
                "image_id",
                "tile_id",
                "dota_coords",
                "class_name",
                "difficult",
            ],
        )
        writer.writeheader()
        writer.writerows(annotation_rows)

    with split_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tile_relpath", "dataset_id", "zone_id", "image_id", "tile_id", "split"],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    if include_stats:
        split_counts = Counter(row["split"] for row in split_rows)
        tiles_per_split = {k: int(split_counts[k]) for k in sorted(split_counts.keys(), key=int)}
        summary = {
            "total_tiles": len(split_rows),
            "total_groups": len(group_keys),
            "num_splits": num_splits,
            "tiles_per_split": tiles_per_split,
            "default_val_fold_id": 0,
            "seed": seed,
            "ignore_labels": sorted(ignore_set),
            "map_labels": map_dict,
            "raw_objects": int(sum(stats["raw_label_counts"].values())),
            "ignored_objects": int(stats["ignored_objects"]),
            "mapped_objects": int(stats["mapped_objects"]),
            "kept_objects": int(stats["kept_objects"]),
            "raw_label_counts": dict(sorted(stats["raw_label_counts"].items())),
            "ignored_label_counts": dict(sorted(stats["ignored_label_counts"].items())),
            "final_label_counts": dict(sorted(stats["final_label_counts"].items())),
            "mapping_pairs": dict(sorted(stats["mapping_pairs"].items())),
        }
        return annotations_path, split_path, summary

    return annotations_path, split_path


class AirbusPlaygroundCSVDataset:
    """CSV-backed Airbus Playground dataset yielding DOTASample objects."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str,
        annotations_file: str | Path = "annotations.csv",
        split_file: str | Path = "split.csv",
        val_split_id: int = 0,
        train_includes_val: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        difficult_strategy: str = "drop",
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
        map_labels: Optional[Dict[str, str]] = None,
        difficult_tags: Optional[Sequence[str]] = None,
        filter_empty_gt: bool = False,
    ):
        from .lookalike import resolve_lookalike_label_set

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")

        if val_split_id < 0:
            raise ValueError(f"val_split_id must be non-negative; got {val_split_id}")

        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(
                f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'."
            )

        self.data_root = Path(data_root)
        self.split = split
        self.val_split_id = int(val_split_id)
        self.train_includes_val = bool(train_includes_val)
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None
        self.difficult_strategy = ds
        self.ignore_labels = set(ignore_labels or [])
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)
        # Lookalike wins over ignore_labels.
        self.ignore_labels -= self._lookalike_set
        self.map_labels = dict(map_labels or {})
        self.difficult_tags = [str(t).strip() for t in (difficult_tags or []) if str(t).strip()]
        self.filter_empty_gt = bool(filter_empty_gt)

        self.annotations_path = _resolve_csv_path(self.data_root, annotations_file)
        self.split_path = _resolve_csv_path(self.data_root, split_file)
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations CSV not found: {self.annotations_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_path}")

        self._samples_meta = self._load_split_rows()
        self._tiles_before_empty_filter = len(self._samples_meta)
        self._annotations_by_tile = self._load_annotations_rows()
        if self.filter_empty_gt:
            self._samples_meta = [
                row
                for row in self._samples_meta
                if len(self._annotations_by_tile.get(row.get("tile_relpath", ""), [])) > 0
            ]
        self._empty_gt_filtered_count = self._tiles_before_empty_filter - len(self._samples_meta)

    @property
    def tiles_discovered_count(self) -> int:
        """Tiles in split CSV before ``filter_empty_gt`` (if enabled)."""
        return self._tiles_before_empty_filter

    @property
    def empty_gt_filtered_count(self) -> int:
        """Tiles removed by ``filter_empty_gt`` at init."""
        return self._empty_gt_filtered_count

    def _load_split_rows(self) -> List[Dict[str, str]]:
        with self.split_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        if not all_rows:
            return []

        mode = detect_airbus_split_csv_format([r.get("split", "") for r in all_rows])
        want_val = self.split == "val"
        include_all_for_train = self.split == "train" and self.train_includes_val
        rows: List[Dict[str, str]] = []
        for row in all_rows:
            raw = row.get("split", "")
            if include_all_for_train:
                rows.append(row)
                continue
            if mode == "train_val":
                if raw.strip().lower() == self.split:
                    rows.append(row)
            else:
                try:
                    sid = int(str(raw).strip())
                except ValueError:
                    continue
                is_val = sid == self.val_split_id
                if is_val == want_val:
                    rows.append(row)

        # Stable order for reproducibility.
        rows.sort(key=lambda row: row.get("tile_relpath", ""))
        return rows

    def _load_annotations_rows(self) -> Dict[str, List[DOTAAnnotation]]:
        selected_tiles = {row.get("tile_relpath", "") for row in self._samples_meta}
        annotations_by_tile: Dict[str, List[DOTAAnnotation]] = defaultdict(list)

        with self.annotations_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tile_relpath = row.get("tile_relpath", "")
                if tile_relpath not in selected_tiles:
                    continue

                raw_class_name = str(row.get("class_name", "")).strip()
                if not raw_class_name:
                    continue

                try:
                    base_difficult = int(str(row.get("difficult", "0")).strip() or "0")
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid difficult value in {self.annotations_path} "
                        f"for tile {tile_relpath!r}: {row.get('difficult')!r}"
                    ) from exc

                try:
                    class_name, difficult = apply_airbus_difficult_tags(
                        raw_class_name,
                        difficult_tags=self.difficult_tags,
                        base_difficult=base_difficult,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to apply difficult_tags for tile {tile_relpath!r} "
                        f"class_name={raw_class_name!r}: {exc}"
                    ) from exc

                class_name = self.map_labels.get(class_name, class_name)
                is_lookalike = class_name in self._lookalike_set
                # Lookalike wins over ignore_labels (checked after map_labels).
                if class_name in self.ignore_labels and not is_lookalike:
                    continue
                if (
                    self.allowed_classes is not None
                    and class_name not in self.allowed_classes
                    and not is_lookalike
                ):
                    raise ValueError(
                        f"Unknown class_name {class_name!r} (from CSV {raw_class_name!r}) "
                        f"on tile {tile_relpath!r} is not in allowed_classes={sorted(self.allowed_classes)}. "
                        "Add a dataset.map_labels entry for the full concatenated string, "
                        "or include it in allowed_classes. Do not silently drop."
                    )

                if self.difficult_strategy == "drop" and difficult == 1:
                    continue

                coords = str(row.get("dota_coords", "")).strip()
                if not coords:
                    continue
                try:
                    coord_vals = [float(x) for x in coords.split()]
                    if len(coord_vals) != 8:
                        raise ValueError(f"expected 8 floats, got {len(coord_vals)}")
                    ann = DOTAAnnotation.from_corners(coord_vals, class_name, difficult)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse annotation for tile {tile_relpath!r} "
                        f"class_name={class_name!r} dota_coords={coords!r}: {exc}"
                    ) from exc
                annotations_by_tile[tile_relpath].append(ann)

        return annotations_by_tile

    def get_raw_class_names_from_csv(self) -> List[str]:
        """Return unique class_name values in the CSV for the current split (before map_labels/ignore_labels).
        Useful to verify CSV content and that map_labels keys match."""
        selected_tiles = {row.get("tile_relpath", "") for row in self._samples_meta}
        raw: Set[str] = set()
        with self.annotations_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("tile_relpath", "") not in selected_tiles:
                    continue
                cn = str(row.get("class_name", "")).strip()
                if cn:
                    raw.add(cn)
        return sorted(raw)

    def _load_image_size(self, image_path: Path) -> Tuple[int, int]:
        _require_pillow()
        with Image.open(image_path) as img:
            return img.size

    def __len__(self) -> int:
        return len(self._samples_meta)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        row = self._samples_meta[idx]
        tile_relpath = row.get("tile_relpath", "")
        if not tile_relpath:
            raise ValueError("split.csv row is missing tile_relpath")

        image_path = self.data_root / tile_relpath
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        width, height = self._load_image_size(image_path)
        annotations = tuple(self._annotations_by_tile.get(tile_relpath, []))
        return DOTASample(
            image_path=image_path,
            width=width,
            height=height,
            annotations=annotations,
        )

    def __iter__(self) -> Iterable[DOTASample]:
        for idx in range(len(self)):
            yield self[idx]

    def get_class_names(self) -> List[str]:
        from .lookalike import filter_semantic_class_names

        classes = set()
        for annotations in self._annotations_by_tile.values():
            for ann in annotations:
                classes.add(ann.class_name)
        return filter_semantic_class_names(classes, self._lookalike_set)


def format_airbus_empty_gt_filter_log(dataset: AirbusPlaygroundCSVDataset, *, split: str) -> str:
    """One-line summary for train logs when ``filter_empty_gt`` is enabled."""
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} tiles "
        f"({kept} kept)"
    )


__all__ = [
    "TileRecord",
    "discover_airbus_tiles",
    "detect_airbus_split_csv_format",
    "generate_airbus_playground_csvs",
    "dated_playground_csv_filenames",
    "resolve_playground_csv_filenames",
    "timestamped_annotations_filename",
    "timestamped_split_filename",
    "apply_airbus_difficult_tags",
    "AirbusPlaygroundCSVDataset",
    "format_airbus_empty_gt_filter_log",
]
