"""DOTA dataset loader with efficient polygon parsing and rbox conversion."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import warnings

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..geometry import Polygon, QBox, RBox, transforms


def format_dota_line(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
    category: str,
    difficult: int,
) -> str:
    """Format a single annotation line in official DOTA format (comma-separated).

    See https://captain-whu.github.io/DOTA/dataset.html
    """
    def _fmt(v: float) -> str:
        # DOTA commonly stores integer pixel coords, but floats are also valid.
        # Keep floats as-is, but drop the trailing ".0" when the value is integral.
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)

    return (
        f"{_fmt(x1)}, {_fmt(y1)}, {_fmt(x2)}, {_fmt(y2)}, "
        f"{_fmt(x3)}, {_fmt(y3)}, {_fmt(x4)}, {_fmt(y4)}, {category}, {int(difficult)}"
    )


@dataclass(frozen=True)
class DOTAAnnotation:
    """Single DOTA annotation entry."""

    class_name: str
    difficult: int  # 0 or 1
    polygon: Polygon
    rbox: RBox

    @classmethod
    def from_line(cls, line: str, *, class_map: Optional[Dict[str, int]] = None) -> "DOTAAnnotation":
        """Parse a single DOTA annotation line.
        
        DOTA Polygon Format (official default):
        ---------------------------------------
        Official DOTA (captain-whu.github.io) uses comma-separated:
          "x1, y1, x2, y2, x3, y3, x4, y4, category, difficult"
        We produce this format by default (see format_dota_line, to_line).
        For reading, we also accept space-separated:
          "x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult"
        (class = parts[8:-1] for multi-word names).
        
        Corner Order Convention:
        - The 4 corners are ordered sequentially around the polygon perimeter
        - Typically, corners follow a consistent winding order (clockwise or counter-clockwise)
        - The first corner (x1, y1) is usually the top-left or top-most point
        - Subsequent corners follow the polygon boundary
        
        Conversion to QBox/RBox:
        - The polygon is first converted to a QBox, which normalizes the point order
        - QBox ensures counter-clockwise orientation and orders points starting from top-most
        - RBox is then derived from QBox, computing center, dimensions, and angle
        
        Args:
            line: DOTA annotation line
            class_map: Optional mapping from class names to IDs
        
        Returns:
            DOTAAnnotation with parsed polygon and RBox
        
        Example:
            >>> line = "100 200 300 200 300 400 100 400 plane 0"
            >>> ann = DOTAAnnotation.from_line(line)
            >>> # Creates a rectangle with corners at (100,200), (300,200), (300,400), (100,400)
        """
        raw = line.strip()
        if "," in raw:
            # Official DOTA: x1, y1, x2, y2, x3, y3, x4, y4, category, difficult (10 fields)
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 10:
                raise ValueError(f"Invalid DOTA annotation line (comma): {line}")
            coords = [float(parts[i]) for i in range(8)]
            class_name = parts[8].strip()
            difficult = int(parts[9])
        else:
            # Space-separated: allows multi-word class (e.g. "General Cargo")
            parts = raw.split()
            if len(parts) < 9:
                raise ValueError(f"Invalid DOTA annotation line: {line}")
            coords = [float(parts[i]) for i in range(8)]
            class_name = " ".join(parts[8:-1]) if len(parts) > 9 else parts[8]
            difficult = int(parts[-1])
        
        # Extract 4 corner points: (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        
        # Convert to Polygon (validates and normalizes orientation)
        polygon = Polygon(points)
        
        # Convert to RBox via QBox (QBox normalizes point order)
        # QBox ensures counter-clockwise order and starts from top-most point
        rbox = transforms.polygon_to_rbox(polygon)
        
        return cls(class_name=class_name, difficult=difficult, polygon=polygon, rbox=rbox)

    def to_line(self) -> str:
        """Serialize to official DOTA format (comma-separated)."""
        coords = []
        for p in self.polygon:
            coords.extend(p)
        return format_dota_line(*coords, self.class_name, self.difficult)


@dataclass(frozen=True)
class DOTASample:
    """Single DOTA image sample with annotations."""

    image_path: Path
    width: int
    height: int
    annotations: Tuple[DOTAAnnotation, ...]
    
    @property
    def num_objects(self) -> int:
        return len(self.annotations)
    
    def filter_by_class(
        self,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        drop_difficult: bool = False,
        lookalike_labels: Optional[Sequence[str]] = None,
    ) -> "DOTASample":
        """Return a filtered copy of this sample.

        Lookalike routing names (reserved ``lookalike`` plus optional aliases) are
        kept even when absent from ``allowed_classes`` and even when listed in
        ``ignore_labels`` (lookalike wins over ignore).
        """
        from .lookalike import resolve_lookalike_label_set

        filtered = list(self.annotations)
        look_set = resolve_lookalike_label_set(lookalike_labels)

        if drop_difficult:
            filtered = [ann for ann in filtered if ann.difficult == 0]

        if allowed_classes is not None:
            allowed_set = set(allowed_classes)
            filtered = [
                ann
                for ann in filtered
                if ann.class_name in allowed_set or ann.class_name in look_set
            ]

        if ignore_labels is not None:
            ignore_set = set(ignore_labels) - look_set
            filtered = [ann for ann in filtered if ann.class_name not in ignore_set]

        return DOTASample(
            image_path=self.image_path,
            width=self.width,
            height=self.height,
            annotations=tuple(filtered)
        )


class DOTADataset:
    """Efficient DOTA dataset loader with lazy parsing."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        label_dir: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        class_map: Optional[Dict[str, int]] = None,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
        # "drop" -> remove at read-time, "ignore"/"keep" -> keep in sample.annotations.
        difficult_strategy: str = "drop",
        filter_empty_gt: bool = False,
    ):
        """Initialize DOTA dataset.

        Args:
            root_dir: Root directory of DOTA dataset (used as base if label_dir/image_dir not specified)
            split: Split name ("train", "val", "test") - used for pattern matching
            split_file: Optional path to a file listing image names (one per line).
                       If provided, only images listed in this file will be used.
                       This follows the official DOTA dataset convention where
                       train.txt, val.txt, test.txt list the image names.
            label_dir: Optional custom path to label directory. If None, uses root_dir/labelTxt.
                      This allows having train/val/test in separate folders.
                      Can be the same as image_dir to read images and annotations from one folder.
            image_dir: Optional custom path to image directory. If None, uses root_dir/images.
                      This allows having train/val/test in separate folders.
                      Can be the same as label_dir to read images and annotations from one folder.
            class_map: Optional mapping from class names to numeric IDs
            allowed_classes: Optional list of allowed class names (whitelist)
            ignore_labels: Optional list of class names to exclude (blacklist)
            lookalike_labels: Optional extra aliases treated as hard-negative lookalikes
                (reserved ``lookalike`` is always included).
            difficult_strategy: How to handle difficult annotations: drop | ignore | keep.
            filter_empty_gt: If True, drop tiles whose effective GT count is zero after
                difficult_strategy, allowed_classes, and ignore_labels (MMRotate DOTADataset parity).
                Lookalike boxes count as non-empty GT.
        """
        from .lookalike import resolve_lookalike_label_set

        self.root_dir = Path(root_dir)
        self.filter_empty_gt = bool(filter_empty_gt)
        self.split = split
        self.split_file = Path(split_file) if split_file else None
        self.class_map = class_map or {}
        self.allowed_classes = allowed_classes
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)
        # Normalize difficult strategy
        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'.")
        self.difficult_strategy = ds
        # Read-time drop of difficult=1 (only when strategy is "drop")
        self._drop_difficult = self.difficult_strategy == "drop"
        
        # Support custom label and image directories for split-specific folders
        if label_dir is not None:
            self.label_dir = Path(label_dir)
        else:
            self.label_dir = self.root_dir / "labelTxt"
        
        if image_dir is not None:
            self.image_dir = Path(image_dir)
        else:
            self.image_dir = self.root_dir / "images"
        
        if not self.label_dir.exists():
            raise FileNotFoundError(f"DOTA label directory not found: {self.label_dir}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"DOTA image directory not found: {self.image_dir}")
        
        discovered = self._discover_annotation_files()
        self._annotation_files_discovered_count = len(discovered)
        if self.filter_empty_gt:
            self._annotation_files = [
                ann_path
                for ann_path in discovered
                if self._effective_gt_count(ann_path) > 0
            ]
            self._empty_gt_filtered_count = (
                self._annotation_files_discovered_count - len(self._annotation_files)
            )
        else:
            self._annotation_files = discovered
            self._empty_gt_filtered_count = 0

    @property
    def annotation_files_discovered_count(self) -> int:
        """Label files found before ``filter_empty_gt`` (if enabled)."""
        return self._annotation_files_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        """Tiles removed by ``filter_empty_gt`` at init."""
        return self._empty_gt_filtered_count

    def _discover_annotation_files(self) -> List[Path]:
        """Discover annotation files for the split.
        
        Two modes:
        1. If split_file is provided: Read image names from file and find corresponding
           annotation files (e.g., train.txt lists "P0001", finds "P0001_train.txt")
        2. Otherwise: Pattern match on annotation filenames (e.g., "*_train.txt")
        
        Returns:
            Sorted list of annotation file paths
        """
        if self.split_file is not None:
            # Mode 1: Use split file (official DOTA convention)
            split_path = Path(self.split_file)
            if not split_path.is_absolute():
                # Try relative to root_dir
                split_path = self.root_dir / split_path
            
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {self.split_file} "
                    f"(tried {split_path})"
                )
            
            # Read image names from split file
            image_names = set()
            with split_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Remove extension if present
                        name = Path(line).stem
                        image_names.add(name)
            
            # Find corresponding annotation files
            annotation_files = []
            for img_name in image_names:
                # Try different annotation file naming conventions
                for suffix in [f"_{self.split}.txt", ".txt"]:
                    ann_file = self.label_dir / f"{img_name}{suffix}"
                    if ann_file.exists():
                        annotation_files.append(ann_file)
                        break
            
            return sorted(annotation_files)
        else:
            # Mode 2: Pattern matching (backward compatible)
            # First try pattern with split suffix (e.g., "*_train.txt")
            pattern = re.compile(rf".*_{self.split}\.txt$", re.IGNORECASE)
            files = [f for f in self.label_dir.iterdir() if f.is_file() and pattern.match(f.name)]
            
            # If no files found with split suffix, fall back to all .txt files
            # This supports cases where annotation files have the same name as images
            if not files:
                files = [f for f in self.label_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
            
            return sorted(files)
    
    def _load_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Load image dimensions efficiently."""
        if Image is None:
            raise RuntimeError("PIL/Pillow is required to load image dimensions.")
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    
    def _parse_annotation_lines(self, ann_path: Path) -> Tuple[DOTAAnnotation, ...]:
        """Parse object lines from a DOTA label file (no image I/O)."""
        annotations: List[DOTAAnnotation] = []
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                try:
                    annotations.append(
                        DOTAAnnotation.from_line(line, class_map=self.class_map)
                    )
                except (ValueError, Exception):
                    continue
        return tuple(annotations)

    def _effective_gt_count(self, ann_path: Path) -> int:
        """GT count after the same filters applied when loading a sample."""
        annotations = self._parse_annotation_lines(ann_path)
        if not annotations:
            return 0
        if (
            self.allowed_classes is not None
            or self.ignore_labels is not None
            or self._drop_difficult
        ):
            sample = DOTASample(
                image_path=ann_path,
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
        return len(annotations)

    def _parse_annotation_file(self, ann_path: Path) -> Optional[DOTASample]:
        """Parse a single DOTA annotation file.
        
        Returns:
            DOTASample if image exists, None if image is missing (with warning printed)
        """
        image_name = ann_path.stem + ".png"
        image_path = self.image_dir / image_name
        
        if not image_path.exists():
            image_name = ann_path.stem + ".jpg"
            image_path = self.image_dir / image_name
        
        if not image_path.exists():
            warnings.warn(f"Image not found for annotation: {ann_path}. Skipping this sample.", UserWarning)
            return None
        
        width, height = self._load_image_size(image_path)
        annotations = self._parse_annotation_lines(ann_path)

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
    
    def __len__(self) -> int:
        return len(self._annotation_files)
    
    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._annotation_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        ann_path = self._annotation_files[idx]
        sample = self._parse_annotation_file(ann_path)
        if sample is None:
            # Image was missing (warning already printed)
            # Raise error for indexed access since we can't return the requested item
            raise FileNotFoundError(
                f"Image not found for annotation at index {idx}: {ann_path}. "
                f"Use iteration (for sample in dataset) to automatically skip missing images."
            )
        return sample
    
    def __iter__(self) -> Iterator[DOTASample]:
        for ann_path in self._annotation_files:
            sample = self._parse_annotation_file(ann_path)
            if sample is not None:
                yield sample
    
    def get_class_names(self) -> List[str]:
        """Extract unique semantic class names (excludes lookalike routing labels)."""
        from .lookalike import filter_semantic_class_names

        classes = set()
        for sample in self:
            for ann in sample.annotations:
                classes.add(ann.class_name)
        return filter_semantic_class_names(classes, self._lookalike_set)


def resolve_dota_tile_roots(
    *,
    tiles_dirs: Optional[Sequence[str | Path]] = None,
    tiles_dir: Optional[str | Path] = None,
    split_label: str = "split",
) -> List[Path]:
    """Resolve one or more DOTA tile roots from singular/plural config fields."""
    if tiles_dirs is not None:
        roots = [Path(p) for p in tiles_dirs]
        if not roots:
            raise ValueError(f"dataset.*_tiles_dirs for {split_label} must be non-empty when set")
        return roots
    if tiles_dir is not None:
        return [Path(tiles_dir)]
    raise ValueError(
        f"DOTA {split_label} requires dataset.*_tiles_dir or dataset.*_tiles_dirs"
    )


def _dota_dirs_for_root(root: Path, *, same_folder: bool) -> tuple[Path, Path, Path]:
    root = Path(root)
    if same_folder:
        return root, root, root
    return root, root / "labels", root / "images"


def iter_dota_datasets(dataset) -> Iterator["DOTADataset"]:
    """Yield ``DOTADataset`` instances from a dataset or ``ConcatDataset``."""
    from torch.utils.data import ConcatDataset

    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            yield from iter_dota_datasets(sub)
    elif isinstance(dataset, DOTADataset):
        yield dataset


def dota_empty_gt_filter_summary(dataset) -> Tuple[int, int, int]:
    """Return (discovered, kept, filtered) tile counts for logging."""
    discovered = kept = filtered = 0
    for ds in iter_dota_datasets(dataset):
        discovered += ds.annotation_files_discovered_count
        kept += len(ds)
        filtered += ds.empty_gt_filtered_count
    return discovered, kept, filtered


def format_dota_empty_gt_filter_log(dataset, *, split: str) -> str:
    """One-line summary for train logs when ``filter_empty_gt`` is enabled."""
    discovered, kept, filtered = dota_empty_gt_filter_summary(dataset)
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} tiles "
        f"({kept} kept)"
    )


def build_dota_split_dataset(
    tile_roots: Sequence[str | Path],
    *,
    split: str,
    same_folder: bool = False,
    difficult_strategy: str = "drop",
    allowed_classes: Optional[Sequence[str]] = None,
    ignore_labels: Optional[Sequence[str]] = None,
    lookalike_labels: Optional[Sequence[str]] = None,
    filter_empty_gt: bool = False,
):
    """Build one ``DOTADataset`` or ``ConcatDataset`` over multiple tile roots."""
    from torch.utils.data import ConcatDataset

    roots = [Path(r) for r in tile_roots]
    if not roots:
        raise ValueError("tile_roots must be non-empty")

    datasets = [
        DOTADataset(
            root_dir=root,
            split=split,
            label_dir=label_dir,
            image_dir=image_dir,
            difficult_strategy=difficult_strategy,
            allowed_classes=allowed_classes,
            ignore_labels=ignore_labels,
            lookalike_labels=lookalike_labels,
            filter_empty_gt=filter_empty_gt,
        )
        for root, label_dir, image_dir in (
            _dota_dirs_for_root(r, same_folder=same_folder) for r in roots
        )
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def collect_dota_image_paths(
    tile_roots: Sequence[str | Path],
    *,
    same_folder: bool = False,
) -> List[Path]:
    """Collect image paths from one or more DOTA tile directories."""
    paths: List[Path] = []
    for root in tile_roots:
        _, _, image_dir = _dota_dirs_for_root(Path(root), same_folder=same_folder)
        if not image_dir.exists():
            raise FileNotFoundError(f"DOTA image directory not found: {image_dir}")
        paths.extend(sorted(image_dir.glob("*.jpg")))
        paths.extend(sorted(image_dir.glob("*.png")))
    return paths


def _image_path_for_annotation(ann_path: Path, image_dir: Path) -> Optional[Path]:
    """Resolve the image file for a DOTA label path (png preferred, then jpg)."""
    for ext in (".png", ".jpg"):
        candidate = image_dir / f"{ann_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_dota_split_image_paths(
    tile_roots: Sequence[str | Path],
    *,
    split: str,
    same_folder: bool = False,
    difficult_strategy: str = "drop",
    allowed_classes: Optional[Sequence[str]] = None,
    ignore_labels: Optional[Sequence[str]] = None,
    lookalike_labels: Optional[Sequence[str]] = None,
    filter_empty_gt: bool = False,
    log_filter_empty_gt: bool = False,
) -> List[Path]:
    """Collect image paths using the same tile set as :func:`build_dota_split_dataset`.

    When ``filter_empty_gt`` is True, skips tiles with no effective GT after
    ``difficult_strategy``, ``allowed_classes``, and ``ignore_labels`` (training parity).
    Set ``log_filter_empty_gt`` to print the same summary as training startup.
    """
    dataset = build_dota_split_dataset(
        tile_roots,
        split=split,
        same_folder=same_folder,
        difficult_strategy=difficult_strategy,
        allowed_classes=allowed_classes,
        ignore_labels=ignore_labels,
        lookalike_labels=lookalike_labels,
        filter_empty_gt=filter_empty_gt,
    )
    if log_filter_empty_gt and filter_empty_gt:
        print("DOTA filter_empty_gt (preds/metrics, same as training):")
        print(format_dota_empty_gt_filter_log(dataset, split=split))
    paths: List[Path] = []
    for ds in iter_dota_datasets(dataset):
        for ann_path in ds._annotation_files:
            image_path = _image_path_for_annotation(ann_path, ds.image_dir)
            if image_path is not None:
                paths.append(image_path)
    return sorted(paths)


def dota_label_path_for_image(image_path: Path, *, same_folder: bool = False) -> Path:
    """Resolve the DOTA annotation ``.txt`` path for an image file."""
    image_path = Path(image_path)
    if same_folder:
        return image_path.with_suffix(".txt")
    parent = image_path.parent
    if parent.name == "images":
        root = parent.parent
        label_dir = root / "labels"
        if not label_dir.exists():
            label_dir = root / "labelTxt"
        return label_dir / f"{image_path.stem}.txt"
    label_dir = parent / "labels"
    if label_dir.exists():
        return label_dir / f"{image_path.stem}.txt"
    label_txt = parent / "labelTxt"
    if label_txt.exists():
        return label_txt / f"{image_path.stem}.txt"
    return parent / f"{image_path.stem}.txt"


def dota_dataset_class_names(dataset) -> List[str]:
    """Union class names from ``DOTADataset``, ``ConcatDataset``, or ``Subset``."""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        return dota_dataset_class_names(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        seen: set[str] = set()
        names: List[str] = []
        for sub in dataset.datasets:
            for name in dota_dataset_class_names(sub):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return sorted(names)
    if isinstance(dataset, DOTADataset):
        return dataset.get_class_names()
    raise TypeError(f"Unsupported dataset type for class discovery: {type(dataset)!r}")


def build_dota_loader(
    root_dir: str | Path,
    *,
    split: str = "train",
    split_file: Optional[str | Path] = None,
    label_dir: Optional[str | Path] = None,
    image_dir: Optional[str | Path] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    class_map: Optional[Dict[str, int]] = None,
    allowed_classes: Optional[Sequence[str]] = None,
    ignore_labels: Optional[Sequence[str]] = None,
    lookalike_labels: Optional[Sequence[str]] = None,
    difficult_strategy: str = "drop",
    filter_empty_gt: bool = False,
    collate_fn: Optional[Callable] = None,
) -> "DataLoader":
    """Build a PyTorch DataLoader for DOTA dataset.

    Args:
        root_dir: Root directory of DOTA dataset (used as base if label_dir/image_dir not specified)
        split: Split name ("train", "val", "test")
        split_file: Optional path to a file listing image names (one per line).
                   If provided, only images listed in this file will be used.
                   This follows the official DOTA dataset convention.
        label_dir: Optional custom path to label directory. If None, uses root_dir/labelTxt.
                  This allows having train/val/test in separate folders.
        image_dir: Optional custom path to image directory. If None, uses root_dir/images.
                  This allows having train/val/test in separate folders.
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        class_map: Optional mapping from class names to numeric IDs
        allowed_classes: Optional list of allowed class names (whitelist)
        ignore_labels: Optional list of class names to exclude (blacklist)
        lookalike_labels: Optional extra lookalike aliases (reserved ``lookalike`` always included)
        difficult_strategy: drop | ignore | keep (same as DOTADataset)
        filter_empty_gt: Drop tiles with no effective GT (same as DOTADataset)
        collate_fn: Optional custom collate function. If None, uses collate_dota_samples
                   from oriented_det.train.utils which converts DOTASample objects
                   to (images, targets) format expected by training engines.
    
    Returns:
        DataLoader that yields batches in format determined by collate_fn.
        Default collate_fn returns (images, targets) tuple for training.
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise RuntimeError("PyTorch is required to build DataLoader.")
    
    dataset = DOTADataset(
        root_dir=root_dir,
        split=split,
        split_file=split_file,
        label_dir=label_dir,
        image_dir=image_dir,
        class_map=class_map,
        allowed_classes=allowed_classes,
        ignore_labels=ignore_labels,
        lookalike_labels=lookalike_labels,
        difficult_strategy=difficult_strategy,
        filter_empty_gt=filter_empty_gt,
    )
    
    # Use provided collate_fn or default to collate_dota_samples
    if collate_fn is None:
        # Lazy import to avoid circular dependencies
        try:
            from ..train.utils import collate_dota_samples
            collate_fn = collate_dota_samples
        except ImportError:
            # Fallback if train module not available
            collate_fn = _default_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _default_collate_fn(batch):
    """Default collate function that returns the batch as-is.
    
    This is a module-level function (not a lambda) so it can be pickled
    when using multiprocessing with num_workers > 0.
    Note: This returns a list, not the (images, targets) tuple expected by training.
    Use collate_dota_samples for training.
    """
    return batch


__all__ = [
    "DOTAAnnotation",
    "DOTASample",
    "DOTADataset",
    "build_dota_loader",
    "build_dota_split_dataset",
    "collect_dota_image_paths",
    "collect_dota_split_image_paths",
    "dota_dataset_class_names",
    "dota_label_path_for_image",
    "resolve_dota_tile_roots",
]
