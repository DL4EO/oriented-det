#!/usr/bin/env python
"""Filter model oriented boxes by horizontal GT overlap (removes false positives).

Reads combined ``detections.json`` from ``odet image-demo --json-batch``, matches
predictions to horizontal GT per image by oriented IoU, and keeps only detections
assigned to a GT box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbb_to_obb import add_gt_cli_args, filter_predictions, resolve_gt_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gt_cli_args(parser)
    parser.add_argument("--detections-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.3,
        help="Minimum oriented IoU to keep a detection as GT-matched (default: 0.3).",
    )
    args = parser.parse_args()

    gt_by_image = resolve_gt_from_args(args)
    filtered = filter_predictions(args.detections_json, gt_by_image, args.match_iou)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("GT-FILTERED ORIENTED BOXES")
    print("=" * 60)
    print(f"Raw model predictions:     {filtered['total_predictions_raw']}")
    print(f"GT boxes:                  {filtered['total_gt']}")
    print(f"Kept (GT-matched):         {filtered['total_predictions']}")
    print(f"False positives removed:   {filtered['total_false_positives_removed']}")
    print(f"Missed GT (no match):      {filtered['total_missed_gt']}")
    print(f"Wrote: {args.output_json}")

    per_image_equal = sum(
        1 for r in filtered["results"] if r["num_pred"] == r["num_gt"]
    )
    print(f"Per-image exact count match: {per_image_equal}/{filtered['num_images']}")


if __name__ == "__main__":
    main()
