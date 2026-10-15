"""COCO instance polygons → le90 oriented boxes (DOTAAnnotation).

MS COCO ``segmentation`` polygons are converted to a 4-corner rbox:

- 4 vertices → ``DOTAAnnotation.from_corners`` (same path as DOTA XML loaders)
- n-gons → convex hull + min-area rectangle (stdlib; no Shapely), then the same
  8-corner path so training never sees instance-seg masks

RLE / crowd annotations are skipped. Used by SSDD and HRSID native loaders and
``odet coco-to-dota``.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from tqdm import tqdm

from oriented_det.utils.progress import tqdm_progress_stream

from .dota import DOTAAnnotation, DOTASample

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew monotone-chain convex hull, CCW, no collinear extras."""
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return unique
    lower: List[Tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect_corners(points: Sequence[Tuple[float, float]]) -> Tuple[float, ...]:
    """Minimum-area enclosing rectangle as 8 corner coords (x1,y1,...,x4,y4).

    Rotating calipers on the convex hull: for each hull edge, project all hull
    vertices into the edge frame and take the axis-aligned box.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        raise ValueError(f"Need at least 3 unique points for a min-area rectangle, got {len(hull)}")
    best_area = float("inf")
    best: Optional[Tuple[float, ...]] = None
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % n]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        ux, uy = dx / length, dy / length
        vx, vy = -uy, ux
        proj_u = [p[0] * ux + p[1] * uy for p in hull]
        proj_v = [p[0] * vx + p[1] * vy for p in hull]
        min_u, max_u = min(proj_u), max(proj_u)
        min_v, max_v = min(proj_v), max(proj_v)
        area = (max_u - min_u) * (max_v - min_v)
        if area < best_area:
            best_area = area
            best = (
                min_u * ux + min_v * vx,
                min_u * uy + min_v * vy,
                max_u * ux + min_v * vx,
                max_u * uy + min_v * vy,
                max_u * ux + max_v * vx,
                max_u * uy + max_v * vy,
                min_u * ux + max_v * vx,
                min_u * uy + max_v * vy,
            )
    if best is None:
        raise ValueError("Could not compute a min-area rectangle")
    return best


def flatten_polygon_coords(raw: Sequence[float]) -> List[Tuple[float, float]]:
    """COCO segmentation list ``[x1, y1, x2, y2, ...]`` → point tuples."""
    if len(raw) < 6 or len(raw) % 2 != 0:
        raise ValueError(f"Polygon coords must be an even-length list of ≥6 numbers, got {len(raw)}")
    return [(float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw), 2)]


def annotation_from_polygon_coords(
    coords: Sequence[float],
    class_name: str,
    *,
    difficult: int = 0,
) -> DOTAAnnotation:
    """Build a le90 ``DOTAAnnotation`` from a COCO-style flat polygon.

    Quadrilaterals keep their four vertices. n-gons become the min-area
    rectangle so ``RBox.from_polygon`` (4 points only) is never called.
    """
    points = flatten_polygon_coords(coords)
    if len(points) == 4:
        flat = [c for p in points for c in p]
        return DOTAAnnotation.from_corners(flat, class_name, difficult=difficult)
    rect = min_area_rect_corners(points)
    return DOTAAnnotation.from_corners(rect, class_name, difficult=difficult)


def annotation_from_hbb(
    x: float,
    y: float,
    width: float,
    height: float,
    class_name: str,
    *,
    difficult: int = 0,
) -> DOTAAnnotation:
    """Axis-aligned COCO ``bbox`` ``[x, y, w, h]`` → le90 rbox (θ ≈ 0)."""
    if width <= 0 or height <= 0:
        raise ValueError("bbox width/height must be positive")
    x2 = x + width
    y2 = y + height
    return DOTAAnnotation.from_corners(
        (x, y, x2, y, x2, y2, x, y2),
        class_name,
        difficult=difficult,
    )


def _largest_polygon_coords(segmentation: object) -> Optional[List[float]]:
    """Return the longest polygon ring; skip RLE dicts and empty lists."""
    if isinstance(segmentation, dict):
        return None
    if not isinstance(segmentation, list) or not segmentation:
        return None
    if segmentation and isinstance(segmentation[0], (int, float)):
        coords = [float(v) for v in segmentation]
        return coords if len(coords) >= 6 else None
    rings: List[List[float]] = []
    for ring in segmentation:
        if not isinstance(ring, list) or len(ring) < 6:
            continue
        rings.append([float(v) for v in ring])
    if not rings:
        return None
    return max(rings, key=len)


@dataclass(frozen=True)
class CocoImageRecord:
    """One COCO image plus its oriented annotations."""

    image_id: int
    file_name: str
    width: int
    height: int
    annotations: Tuple[DOTAAnnotation, ...]


def parse_coco_instances(
    ann_file: str | Path,
    *,
    default_class: str = "ship",
) -> Tuple[CocoImageRecord, ...]:
    """Parse a COCO instances JSON into per-image ``DOTAAnnotation`` tuples."""
    path = Path(ann_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO JSON must be an object: {path}")

    cat_names: Dict[int, str] = {}
    for cat in payload.get("categories") or []:
        if not isinstance(cat, dict):
            continue
        try:
            cid = int(cat["id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(cat.get("name") or default_class).strip() or default_class
        cat_names[cid] = name

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
        except (KeyError, TypeError, ValueError):
            continue
        if image_id not in anns_by_image:
            continue
        cat_id = raw.get("category_id")
        try:
            class_name = cat_names.get(int(cat_id), default_class) if cat_id is not None else default_class
        except (TypeError, ValueError):
            class_name = default_class
        difficult_raw = raw.get("difficult", 0)
        try:
            difficult = int(float(difficult_raw))
        except (TypeError, ValueError):
            difficult = 0
        coords = _largest_polygon_coords(raw.get("segmentation"))
        try:
            if coords is not None:
                ann = annotation_from_polygon_coords(coords, class_name, difficult=difficult)
            else:
                bbox = raw.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                ann = annotation_from_hbb(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    class_name,
                    difficult=difficult,
                )
        except (TypeError, ValueError):
            continue
        anns_by_image[image_id].append(ann)

    records: List[CocoImageRecord] = []
    for image_id, img in images_by_id.items():
        file_name = str(img.get("file_name") or "").strip()
        if not file_name:
            continue
        try:
            width = int(float(img.get("width") or 0))
            height = int(float(img.get("height") or 0))
        except (TypeError, ValueError):
            width, height = 0, 0
        records.append(
            CocoImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                annotations=tuple(anns_by_image.get(image_id, ())),
            )
        )
    return tuple(records)


def iter_image_files(directory: Path) -> List[Path]:
    """Sorted image files directly under ``directory`` (not recursive)."""
    if not directory.is_dir():
        return []
    found: List[Path] = []
    seen: set[Path] = set()
    for suffix in _IMAGE_SUFFIXES:
        for path in directory.glob(f"*{suffix}"):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
        for path in directory.glob(f"*{suffix.upper()}"):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
    found.sort(key=lambda p: p.name.lower())
    return found


def find_image_file(image_dirs: Sequence[Path], file_name: str) -> Optional[Path]:
    """Resolve a COCO ``file_name`` against one or more image directories."""
    name = file_name.strip().replace("\\", "/")
    if not name:
        return None
    basename = Path(name).name
    for directory in image_dirs:
        if not directory.is_dir():
            continue
        direct = directory / name
        if direct.is_file():
            return direct
        nested = directory / Path(name)
        if nested.is_file():
            return nested
        base = directory / basename
        if base.is_file():
            return base
        stem = Path(basename).stem
        for suffix in _IMAGE_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def parse_dota_label_txt(path: str | Path) -> Tuple[DOTAAnnotation, ...]:
    """Read a DOTA ``.txt`` label file (comma or space separated)."""
    label_path = Path(path)
    if not label_path.is_file():
        return tuple()
    anns: List[DOTAAnnotation] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip()
        if not token or token.startswith("gsd:") or token.startswith("#"):
            continue
        try:
            anns.append(DOTAAnnotation.from_line(token))
        except (TypeError, ValueError):
            continue
    return tuple(anns)


def dota_label_dir_for_images(image_dir: Path) -> Optional[Path]:
    """Sibling ``labelTxt`` / ``labels`` next to an ``images`` folder, else same folder."""
    parent = image_dir.parent
    if image_dir.name.lower() == "images":
        for name in ("labelTxt", "labels", "labeltxt"):
            candidate = parent / name
            if candidate.is_dir():
                return candidate
        return parent
    for name in ("labelTxt", "labels", "labeltxt"):
        candidate = image_dir / name
        if candidate.is_dir():
            return candidate
        sibling = parent / name
        if sibling.is_dir():
            return sibling
    same_txt = list(image_dir.glob("*.txt"))
    if same_txt:
        return image_dir
    return None


def normalize_export_image_format(image_format: str) -> str:
    fmt = (image_format or "original").strip().lower()
    if fmt in {"jpeg"}:
        fmt = "jpg"
    if fmt not in {"original", "jpg", "png"}:
        raise ValueError(f"Unsupported image_format={image_format!r}; expected original, jpg, or png")
    return fmt


def export_rgb_image(
    source: Path,
    dest_dir: Path,
    stem: str,
    *,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> Path:
    """Copy or convert one raster to RGB JPEG/PNG (grayscale SAR → 3-channel)."""
    _require_pillow()
    fmt = normalize_export_image_format(image_format)
    src_suf = source.suffix.lower()
    if fmt == "original":
        if src_suf in {".jpg", ".jpeg", ".png"}:
            dest = dest_dir / f"{stem}{'.jpg' if src_suf in {'.jpg', '.jpeg'} else '.png'}"
            with Image.open(source) as img:
                if img.mode == "RGB" and not dest.exists():
                    shutil.copy2(source, dest)
                    return dest
                if not dest.exists():
                    img.convert("RGB").save(
                        dest,
                        format="JPEG" if dest.suffix.lower() in {".jpg", ".jpeg"} else "PNG",
                        **({"quality": int(jpeg_quality), "subsampling": 0} if dest.suffix.lower() in {".jpg", ".jpeg"} else {}),
                    )
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


def write_dota_split(
    samples: Iterable[DOTASample],
    output_dir: str | Path,
    split: str,
    *,
    same_folder: bool = False,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> int:
    """Write one DOTA split (images + ``.txt`` labels) from ``DOTASample`` objects."""
    split_dir = Path(output_dir) / split
    if same_folder:
        image_dir = split_dir
        label_dir = split_dir
    else:
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    total = len(samples) if isinstance(samples, Sized) else None
    for sample in tqdm(
        samples,
        total=total,
        desc=f"Writing {split}",
        file=tqdm_progress_stream(),
    ):
        stem = Path(sample.image_path).stem
        export_rgb_image(
            Path(sample.image_path),
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
    return n


def export_coco_json_to_dota(
    ann_file: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    same_folder: bool = False,
    image_format: str = "original",
    jpeg_quality: int = 95,
    default_class: str = "ship",
) -> int:
    """Export one COCO instances JSON + image folder to a DOTA split directory."""
    records = parse_coco_instances(ann_file, default_class=default_class)
    image_dirs = [Path(image_dir)]
    samples: List[DOTASample] = []
    for record in records:
        image_path = find_image_file(image_dirs, record.file_name)
        if image_path is None:
            continue
        width, height = record.width, record.height
        if width <= 0 or height <= 0:
            _require_pillow()
            with Image.open(image_path) as img:
                width, height = img.size
        samples.append(
            DOTASample(
                image_path=image_path,
                width=width,
                height=height,
                annotations=record.annotations,
            )
        )
    return write_dota_split(
        samples,
        output_dir,
        split,
        same_folder=same_folder,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


__all__ = [
    "CocoImageRecord",
    "annotation_from_hbb",
    "annotation_from_polygon_coords",
    "convex_hull",
    "dota_label_dir_for_images",
    "export_coco_json_to_dota",
    "export_rgb_image",
    "find_image_file",
    "flatten_polygon_coords",
    "iter_image_files",
    "min_area_rect_corners",
    "normalize_export_image_format",
    "parse_coco_instances",
    "parse_dota_label_txt",
    "write_dota_split",
]
