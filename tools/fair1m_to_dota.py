#!/usr/bin/env python3
"""Export FAIR1M (native XML) to DOTA-format image + label folders.

Native training can use ``dataset.format: fair1m`` on whole images, but FAIR1M
rasters are typically 1k–10k px, so the recommended path is:

1. ``odet fair1m-to-dota`` (this tool) — optional ``--val-fraction`` holdout;
   default ``--image-format original`` copies JPEG/PNG (no PNG rewrite)
2. ``odet tile-dota`` at 1024 / overlap 200 (JPEG in → JPEG tiles)
3. Train with ``dataset.format: dota`` recipes under ``configs/*/fair1m_le90_1x.json``

Usage:
    odet fair1m-to-dota --data-root /path/to/FAIR1M --output-dir /path/to/FAIR1M-dota \\
        --splits train,val
    python tools/fair1m_to_dota.py --data-root /path/to/Dataset --output-dir /tmp/fair1m_dota
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.fair1m import FAIR1M_SPLIT_NAMES, export_fair1m_to_dota


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export FAIR1M XML/images to DOTA image + .txt folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="FAIR1M root (official train/validation or Kaggle Dataset/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory; one subdirectory per split.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated split names (default: train,val).",
    )
    parser.add_argument(
        "--difficult-strategy",
        type=str,
        default="keep",
        choices=("drop", "ignore", "keep"),
        help="How to handle XML difficult=1 objects in exported labels.",
    )
    parser.add_argument(
        "--same-folder",
        action="store_true",
        help="Write images and .txt labels in the same split directory.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=(
            "When set, build a deterministic image-level holdout from the train "
            "images (Kaggle has no labeled val). Writes ImageSets/train.txt and "
            "val.txt. Requires splits to include train and val."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for --val-fraction holdout (default: 0).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="original",
        choices=("original", "png", "jpg", "jpeg"),
        help=(
            "Export image format (default: original). original copies JPEG/PNG "
            "and converts TIFF/BMP to PNG. png/jpg force a convert."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 when writing JPEG (default: 95).",
    )
    args = parser.parse_args(argv)

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in FAIR1M_SPLIT_NAMES]
    if unknown:
        parser.error(f"Unknown split(s) {unknown}; expected {sorted(FAIR1M_SPLIT_NAMES)}")

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("JPEG quality must be between 1 and 100")

    counts = export_fair1m_to_dota(
        args.data_root,
        args.output_dir,
        splits=splits,
        difficult_strategy=args.difficult_strategy,
        same_folder=args.same_folder,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Wrote DOTA export under {args.output_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} image(s)")
    imagesets = args.output_dir / "ImageSets"
    if imagesets.is_dir():
        print(f"  ImageSets: {imagesets}")
    print(
        "Next: odet tile-dota <split_dir> --tile-size 1024 --overlap 200 "
        "--min-overlap 0.7  (run once per split)"
    )


if __name__ == "__main__":
    main()
