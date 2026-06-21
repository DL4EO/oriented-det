#!/usr/bin/env python3
"""Export Airbus Playground folder layout to DOTA-format directories (images + labels).

Reads from the Playground export structure (dataset_id/samples/..., dataset_id/labels/...),
converts polygon annotations to DOTA OBB format (8 coords + class + difficulty), and writes
images and .txt label files. Optionally splits output into train/val by group ratio, or via an
existing split CSV from ``generate_airbus_playground_csv.py`` (integer fold ids or legacy train/val).

Usage:
    # Single output dir (no split)
    python tools/playground_to_dota.py --data-root /path/to/playground_export --output-dir /path/to/dota_out

    # Train/val split by image group (ratio; independent of generate_airbus_playground_csv folds)
    python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --val-ratio 0.2 --seed 42

    # Use existing split CSV (integer fold ids; val fold 0 by default, or pass --val-split-id)
    python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --split-file /path/to/export/split.csv

    # Label filtering and mapping
    python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --ignore-label Confuser --map-label "taxi=car"
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure oriented_det is importable when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data import format_dota_line
from oriented_det.data.airbus_playground import (
    TileRecord,
    discover_airbus_tiles,
    detect_airbus_split_csv_format,
    _load_airbus_label_rows,
    _image_group_key,
)


def _unique_tile_basename(tile: TileRecord) -> str:
    """Return a unique basename for the tile (no extension) to avoid collisions across zones."""
    return f"{tile.dataset_id}_{tile.zone_id}_{tile.image_id}_{tile.tile_id}"


def _compute_split_by_ratio(
    tiles: List[TileRecord],
    val_ratio: float,
    seed: int,
) -> Set[Tuple[str, str, str]]:
    """Compute which (dataset_id, zone_id, image_id) groups are validation from a ratio."""
    groups: Dict[Tuple[str, str, str], List[TileRecord]] = defaultdict(list)
    for tile in tiles:
        groups[_image_group_key(tile)].append(tile)
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    n_val = max(1, int(round(len(group_keys) * val_ratio)))
    return set(group_keys[:n_val])


def _load_split_file(split_path: Path, *, val_split_id: int = 0) -> Dict[str, str]:
    """Load split CSV: tile_relpath -> 'train' | 'val'.

    Supports legacy ``train`` / ``val`` strings or integer fold ids (same convention as
    ``generate_airbus_playground_csvs``); for fold ids, tiles whose fold equals ``val_split_id``
    are mapped to ``val``.
    """
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
    if not raw_rows:
        return {}
    mode = detect_airbus_split_csv_format([r.get("split", "") for r in raw_rows])
    out: Dict[str, str] = {}
    for row in raw_rows:
        rel = (row.get("tile_relpath") or "").strip()
        if not rel:
            continue
        split_raw = row.get("split") or ""
        if mode == "train_val":
            sv = split_raw.strip().lower()
            if sv in ("train", "val"):
                out[rel] = sv
        else:
            try:
                sid = int(str(split_raw).strip())
            except ValueError:
                continue
            out[rel] = "val" if sid == val_split_id else "train"
    return out


def run(
    data_root: Path,
    output_dir: Path,
    *,
    val_ratio: Optional[float] = None,
    seed: int = 42,
    split_file: Optional[Path] = None,
    val_split_id: int = 0,
    ignore_labels: Optional[List[str]] = None,
    map_labels: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Export Playground export to DOTA dirs. Returns stats dict (tiles_written, train_count, val_count, objects_total, objects_skipped)."""
    root = Path(data_root).resolve()
    out = Path(output_dir).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {root}")

    tiles = discover_airbus_tiles(root)
    if not tiles:
        raise FileNotFoundError(f"No Airbus Playground tiles found under {root}")

    # Resolve train/val assignment
    use_split_dirs = False
    val_keys: Optional[Set[Tuple[str, str, str]]] = None
    split_by_tile: Optional[Dict[str, str]] = None

    if split_file is not None:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        split_by_tile = _load_split_file(split_file, val_split_id=val_split_id)
        if not split_by_tile:
            raise ValueError("Split file has no valid tile_relpath -> split rows.")
        use_split_dirs = True
    elif val_ratio is not None:
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0, 1).")
        val_keys = _compute_split_by_ratio(tiles, val_ratio, seed)
        use_split_dirs = True

    ignore_set: Optional[Set[str]] = None
    if ignore_labels:
        ignore_set = set(ignore_labels)
    map_dict = dict(map_labels or {})

    # Output layout: [output_dir/] [train|val/] images/, labels/
    def out_paths(tile: TileRecord, split: str) -> Tuple[Path, Path]:
        base = _unique_tile_basename(tile)
        if use_split_dirs:
            img_dir = out / split / "images"
            lbl_dir = out / split / "labels"
        else:
            img_dir = out / "images"
            lbl_dir = out / "labels"
        return img_dir / f"{base}.jpg", lbl_dir / f"{base}.txt"

    def get_split(tile: TileRecord) -> str:
        if split_by_tile is not None:
            return split_by_tile.get(tile.tile_relpath, "train")
        if val_keys is not None:
            return "val" if _image_group_key(tile) in val_keys else "train"
        return "train"

    stats = {
        "tiles_total": len(tiles),
        "tiles_skipped": 0,
        "tiles_written": 0,
        "train_tiles": 0,
        "val_tiles": 0,
        "objects_total": 0,
    }

    for tile in sorted(tiles, key=lambda t: t.tile_relpath):
        img_src = root / tile.tile_relpath
        if not img_src.exists():
            stats["tiles_skipped"] += 1
            if not dry_run:
                print(f"Warning: image not found {img_src}", file=sys.stderr)
            continue

        split = get_split(tile)
        img_dst, lbl_dst = out_paths(tile, split)

        rows = _load_airbus_label_rows(
            root / tile.label_relpath,
            ignore_labels=ignore_set,
            map_labels=map_dict if map_dict else None,
        )

        if not dry_run:
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, img_dst)
            # Official DOTA format: comma-separated (x1, y1, x2, y2, x3, y3, x4, y4, category, difficult)
            lines = []
            for dota_coords, class_name, difficult in rows:
                coords = [float(x) for x in dota_coords.split()]
                lines.append(format_dota_line(*coords, class_name, difficult))
            lbl_dst.write_text("\n".join(lines), encoding="utf-8")

        stats["tiles_written"] += 1
        stats["objects_total"] += len(rows)
        if split == "train":
            stats["train_tiles"] += 1
        else:
            stats["val_tiles"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Airbus Playground export to DOTA-format directories (images + .txt labels).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root of Playground export (dataset_id/samples/..., labels/...).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for DOTA dirs (images/ and labels/, or train/val subdirs).")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation ratio by image group (0–1). If set, creates output_dir/train and output_dir/val.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed when using --val-ratio.")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to split CSV (tile_relpath, split). Overrides --val-ratio. Integer split column uses --val-split-id as the val fold.",
    )
    parser.add_argument(
        "--val-split-id",
        type=int,
        default=0,
        help="When --split-file has integer fold ids, treat this fold as validation (default 0).",
    )
    parser.add_argument("--ignore-label", action="append", default=[], dest="ignore_labels", help="Label to ignore (repeatable).")
    parser.add_argument(
        "--map-label",
        action="append",
        default=[],
        dest="map_label_list",
        help="Mapping as src=dst (repeatable). Example: taxi=car",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; only report stats.")
    args = parser.parse_args()

    map_labels: Dict[str, str] = {}
    for entry in args.map_label_list:
        if "=" not in entry:
            print(f"Error: invalid --map-label '{entry}'. Use src=dst.", file=sys.stderr)
            return 1
        src, dst = entry.split("=", 1)
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            print(f"Error: empty src or dst in --map-label '{entry}'.", file=sys.stderr)
            return 1
        map_labels[src] = dst

    try:
        stats = run(
            args.data_root,
            args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_file=args.split_file,
            val_split_id=args.val_split_id,
            ignore_labels=args.ignore_labels or None,
            map_labels=map_labels if map_labels else None,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Playground → DOTA export " + ("(dry run)" if args.dry_run else "complete."))
    print(f"  Tiles total:   {stats['tiles_total']}")
    print(f"  Tiles skipped:  {stats['tiles_skipped']}")
    print(f"  Tiles written:  {stats['tiles_written']}")
    if stats["train_tiles"] or stats["val_tiles"]:
        print(f"  Train tiles:    {stats['train_tiles']}")
        print(f"  Val tiles:      {stats['val_tiles']}")
    print(f"  Objects total:  {stats['objects_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
