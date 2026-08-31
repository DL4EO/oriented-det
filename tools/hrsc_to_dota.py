#!/usr/bin/env python3
"""Export HRSC2016 (native XML) to DOTA-format PNG + label folders.

Native training uses ``dataset.format: hrsc2016`` and does not require this
conversion. Use this tool when you want DOTA loaders or ``odet tile-dota``
(images larger than the training canvas).

Usage:
    odet hrsc-to-dota --data-root /path/to/HRSC2016 --output-dir /path/to/HRSC2016-dota
    python tools/hrsc_to_dota.py --data-root /path/to/HRSC2016 --output-dir /tmp/hrsc_dota --splits trainval,test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.hrsc2016 import HRSC2016_IMAGESET_NAMES, export_hrsc2016_to_dota


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export HRSC2016 XML/BMP splits to DOTA PNG + .txt folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="HRSC2016 root (contains FullDataSet/ and ImageSets/).",
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
        default="trainval,test",
        help="Comma-separated ImageSets names (default: trainval,test).",
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
    args = parser.parse_args(argv)

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in HRSC2016_IMAGESET_NAMES]
    if unknown:
        parser.error(f"Unknown split(s) {unknown}; expected {sorted(HRSC2016_IMAGESET_NAMES)}")

    counts = export_hrsc2016_to_dota(
        args.data_root,
        args.output_dir,
        splits=splits,
        difficult_strategy=args.difficult_strategy,
        same_folder=args.same_folder,
    )
    print(f"Wrote DOTA export under {args.output_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} image(s)")


if __name__ == "__main__":
    main()
