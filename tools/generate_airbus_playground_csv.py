#!/usr/bin/env python3
"""Generate annotations.csv and split.csv for Airbus Playground exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from oriented_det.data.airbus_playground import (
    generate_airbus_playground_csvs,
    resolve_playground_csv_filenames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate annotations.csv and split.csv from Airbus Playground export folders. "
            "Split assignment is grouped by (dataset_id, zone_id, image_id) to avoid leakage. "
            "The split column holds integer fold ids 0..num_splits-1 (not train/val strings); "
            "fold 0 is the conventional validation fold unless you set dataset.val_split_id in training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root export folder containing dataset_id subfolders.",
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default=None,
        help=(
            "Output annotations CSV filename (relative to --data-root). "
            "Default: paired dated name annotations_YYYYMMDD.csv (UTC), matching --split-file."
        ),
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help=(
            "Output split CSV filename (relative to --data-root). "
            "Default: paired dated name split_YYYYMMDD.csv (UTC), matching --annotations-file."
        ),
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=10,
        metavar="K",
        help="Number of disjoint folds (K>=2). Groups are shuffled then round-robin assigned ids 0..K-1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--ignore-label",
        action="append",
        default=[],
        help="Label to drop (repeatable). Example: --ignore-label Confuser",
    )
    parser.add_argument(
        "--map-label",
        action="append",
        default=[],
        help="Label mapping in src=dst form (repeatable). Example: --map-label taxi=car",
    )
    parser.add_argument(
        "--difficult-tag",
        action="append",
        default=[],
        help=(
            "Exact Playground tag that sets difficult=1 and is stripped from class_name "
            "(repeatable). Example: --difficult-tag 'Partially Hidden'"
        ),
    )
    return parser.parse_args()


def _parse_map_args(entries: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --map-label value '{entry}'. Expected src=dst.")
        src, dst = entry.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid --map-label value '{entry}'. Empty src/dst is not allowed.")
        mapping[src] = dst
    return mapping


def _print_stats(stats: Dict[str, object]) -> None:
    print("\nDataset label statistics")
    print("-" * 40)
    print(f"Total groups: {stats['total_groups']}")
    tiles_ps = stats.get("tiles_per_split", {})
    if isinstance(tiles_ps, dict) and tiles_ps:
        per = ", ".join(f"{k}:{tiles_ps[k]}" for k in sorted(tiles_ps.keys(), key=lambda x: int(str(x))))
        print(f"Total tiles: {stats['total_tiles']} (per fold: {per})")
    else:
        print(f"Total tiles: {stats['total_tiles']}")
    print(
        f"Folds: num_splits={stats.get('num_splits', '?')} — default val fold id is "
        f"{stats.get('default_val_fold_id', 0)} (override with dataset.val_split_id in config)."
    )
    print(
        f"Objects: raw={stats['raw_objects']} kept={stats['kept_objects']} "
        f"ignored={stats['ignored_objects']} mapped={stats['mapped_objects']}"
    )

    final_counts = stats.get("final_label_counts", {})
    if isinstance(final_counts, dict) and final_counts:
        print("Final label counts:")
        for label, count in sorted(final_counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            print(f"  - {label}: {count}")

    ignored_counts = stats.get("ignored_label_counts", {})
    if isinstance(ignored_counts, dict) and ignored_counts:
        print("Ignored label counts:")
        for label, count in sorted(ignored_counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            print(f"  - {label}: {count}")


def main() -> None:
    args = parse_args()
    if args.num_splits < 2:
        raise SystemExit("Error: --num-splits must be at least 2.")
    map_labels = _parse_map_args(args.map_label)
    annotations_file, split_file = resolve_playground_csv_filenames(
        args.annotations_file,
        args.split_file,
    )
    annotations_path, split_path, stats = generate_airbus_playground_csvs(
        args.data_root,
        annotations_file=annotations_file,
        split_file=split_file,
        num_splits=args.num_splits,
        seed=args.seed,
        ignore_labels=args.ignore_label,
        map_labels=map_labels,
        difficult_tags=args.difficult_tag or None,
        include_stats=True,
    )
    print(f"Wrote annotations CSV: {annotations_path}")
    print(f"Wrote split CSV: {split_path}")
    print(
        "Set in your training config: "
        f"dataset.annotations_file={annotations_path.name!r}, "
        f"dataset.split_file={split_path.name!r}"
    )
    _print_stats(stats)


if __name__ == "__main__":
    main()
