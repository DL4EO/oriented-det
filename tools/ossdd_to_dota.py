#!/usr/bin/env python3
"""Export OSSDD Sentinel-1 chips to DOTA folders for Rotated RTMDet.

Reads the extracted OpenSARShip-Ship Detection Dataset (Hammer et al., 2026):
32-bit VV/VH amplitude GeoTIFFs, ``metadata.csv`` splits, and rotated-box
text files. Writes PNG chips plus ``labels/*.txt`` in official DOTA line format.

Amplitude follows the paper: clip non-positive values and values above 20000,
``log10``, then per-chip min-max to uint8. The gray chip is repeated to three
channels so the CSPNeXt stem can load it. Train chips stay 700×700; the recipe
random-crops them to 512. Val and test chips are already 512×512.

The 700×700 train chips use photometric 34892, which Pillow will not open.
Run ``tools/ossdd_fix_tiff.py`` on the extracted tree before this export.

Usage:
    python tools/ossdd_fix_tiff.py --data-root /path/to/data/OSSDD
    python tools/ossdd_to_dota.py \\
        --data-root /path/to/data/OSSDD \\
        --output-dir /path/to/data/OSSDD-dota \\
        --polarization VH
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.dota import format_dota_line
from oriented_det.utils import tqdm_progress_stream

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "validate": "val",
    "test": "test",
}
_AMPLITUDE_CLIP = 20000.0


def find_ossdd_root(data_root: Path) -> Path:
    """Return the directory that contains ``metadata.csv``."""
    root = Path(data_root)
    if (root / "metadata.csv").is_file():
        return root
    matches = sorted(root.rglob("metadata.csv"))
    if not matches:
        raise FileNotFoundError(f"No metadata.csv under {root}")
    return matches[0].parent


def normalize_split(value: str) -> str | None:
    return _SPLIT_ALIASES.get(value.strip().lower())


def bare_stem(file_name: str) -> str:
    """Chip id without extension and without a leading ``Chip_VV_`` / ``Chip_VH_``."""
    stem = Path(file_name).stem
    lower = stem.lower()
    for prefix in ("chip_vv_", "chip_vh_"):
        if lower.startswith(prefix):
            return stem[len(prefix):]
    return stem


def parse_rbb_text(text: str) -> list[tuple[float, ...]]:
    """Return 8-corner tuples from an OSSDD rotated-box file."""
    boxes: list[tuple[float, ...]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.replace("||", " ").replace("|", " ").replace(",", " ")
        parts = line.split()
        nums: list[float] = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                nums = []
                break
        if not nums:
            continue
        # First line is often a lone object count.
        if len(nums) == 1:
            continue
        # object_id, x1, y1, ..., x4, y4, optional center.
        if len(nums) >= 9:
            corners = tuple(nums[1:9])
        elif len(nums) >= 8:
            corners = tuple(nums[:8])
        else:
            continue
        boxes.append(corners)
    return boxes


def amplitude_to_uint8(values: np.ndarray) -> np.ndarray:
    """Paper scaling: clip, log10, per-chip min-max → uint8."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    clipped = np.where(finite, arr, np.nan)
    clipped = np.where(clipped > _AMPLITUDE_CLIP, np.nan, clipped)
    positive = clipped > 0
    db = np.full(clipped.shape, np.nan, dtype=np.float32)
    db[positive] = np.log10(clipped[positive])
    if not np.any(np.isfinite(db)):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanmin(db))
    hi = float(np.nanmax(db))
    if hi <= lo:
        scaled = np.zeros(arr.shape, dtype=np.float32)
    else:
        scaled = (np.nan_to_num(db, nan=lo) - lo) / (hi - lo)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def save_rgb_png(gray_u8: np.ndarray, dest: Path) -> None:
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(dest)


def _index_files(root: Path, suffix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        found.setdefault(path.name.lower(), path)
    return found


def _pick(index: dict[str, Path], names: list[str]) -> Path | None:
    for name in names:
        hit = index.get(name.lower())
        if hit is not None:
            return hit
    return None


def chip_path(index: dict[str, Path], file_name: str, polarization: str) -> Path | None:
    pol = polarization.upper()
    bare = bare_stem(file_name)
    return _pick(
        index,
        [
            f"Chip_{pol}_{bare}.tif",
            f"Chip_{pol}_{bare}.tiff",
            f"{bare}.tif",
            f"{bare}.tiff",
            Path(file_name).name,
        ],
    )


def rbb_path(index: dict[str, Path], file_name: str) -> Path | None:
    bare = bare_stem(file_name)
    stem = Path(file_name).stem
    return _pick(
        index,
        [
            f"Metadata_rbb_{bare}.txt",
            f"metadata_rbb_{bare}.txt",
            f"{bare}.txt",
            f"Metadata_rbb_{stem}.txt",
            f"{stem}.txt",
        ],
    )


def _read_rows(metadata_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{metadata_csv} has no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        file_key = fields.get("file_name") or fields.get("filename")
        split_key = fields.get("split")
        if file_key is None or split_key is None:
            raise ValueError(
                f"{metadata_csv} needs file_name and split columns, got {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            rows.append({"file_name": row[file_key], "split": row[split_key]})
        return rows


def convert_ossdd(
    data_root: Path,
    output_dir: Path,
    *,
    polarization: str = "VH",
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, int]:
    """Write ``{split}/images`` and ``{split}/labels`` under ``output_dir``."""
    root = find_ossdd_root(data_root)
    pol = polarization.upper()
    if pol not in {"VV", "VH"}:
        raise ValueError(f"polarization must be VV or VH, got {polarization!r}")
    wanted = {normalize_split(name) for name in splits}
    if None in wanted:
        raise ValueError(f"Unknown split in {splits}")
    rows = _read_rows(root / "metadata.csv")
    tifs = _index_files(root, ".tif")
    tifs.update(_index_files(root, ".tiff"))
    rbbs = _index_files(root, ".txt")
    counts = {name: 0 for name in ("train", "val", "test")}
    missing_chip = 0
    missing_rbb = 0
    selected = []
    for row in rows:
        split = normalize_split(row["split"] or "")
        if split in wanted:
            selected.append((split, row))
    for split, row in tqdm(
        selected,
        desc=f"Exporting {pol} chips",
        file=tqdm_progress_stream(),
    ):
        chip = chip_path(tifs, row["file_name"], pol)
        if chip is None:
            missing_chip += 1
            continue
        label = rbb_path(rbbs, row["file_name"])
        boxes: list[tuple[float, ...]] = []
        if label is None:
            missing_rbb += 1
        else:
            boxes = parse_rbb_text(label.read_text(encoding="utf-8"))
        stem = f"{bare_stem(row['file_name'])}_{pol}"
        image_dest = output_dir / split / "images" / f"{stem}.png"
        label_dest = output_dir / split / "labels" / f"{stem}.txt"
        with Image.open(chip) as img:
            gray = amplitude_to_uint8(np.asarray(img))
        save_rgb_png(gray, image_dest)
        lines = [
            format_dota_line(*corners, "ship", 0)
            for corners in boxes
        ]
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
    if missing_chip:
        print(f"Skipped {missing_chip} rows with no {pol} chip")
    if missing_rbb:
        print(f"Wrote {missing_rbb} chips with no rotated-box file (empty labels)")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polarization", default="VH", choices=["VH", "VV", "vh", "vv"])
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    counts = convert_ossdd(
        args.data_root,
        args.output_dir,
        polarization=args.polarization,
        splits=splits,
    )
    written = ", ".join(f"{name}={counts[name]}" for name in ("train", "val", "test"))
    print(f"Wrote {args.output_dir} ({written})")


if __name__ == "__main__":
    main()
