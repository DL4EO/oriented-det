#!/usr/bin/env python
"""Compare horizontal GT box counts vs model oriented predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbb_to_obb import add_gt_cli_args, resolve_gt_from_args


def load_prediction_counts(detections_json: Path) -> Counter:
    payload = json.loads(detections_json.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for row in payload.get("results", []):
        name = row.get("image_name") or Path(row.get("image_path", "")).name
        counts[name] = int(row.get("num_pred", len(row.get("predictions", []))))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gt_cli_args(parser)
    parser.add_argument("--detections-json", type=Path, required=True)
    args = parser.parse_args()

    gt_by_image = resolve_gt_from_args(args)
    straight = Counter({k: len(v) for k, v in gt_by_image.items()})
    oriented = load_prediction_counts(args.detections_json)

    total_straight = sum(straight.values())
    total_oriented = sum(oriented.values())

    print("=" * 60)
    print("HBB vs OBB COUNT COMPARISON")
    print("=" * 60)
    print(f"Horizontal GT boxes:       {total_straight}")
    print(f"Model oriented predictions:{total_oriented}")

    matched_images = sorted(set(straight) & set(oriented))
    same = sum(1 for img in matched_images if straight[img] == oriented[img])
    diffs = [
        (img, straight[img], oriented[img])
        for img in matched_images
        if straight[img] != oriented[img]
    ]

    print()
    print(f"Per-image equal counts: {same}/{len(matched_images)}")
    if diffs:
        print(f"Images with different counts: {len(diffs)}")
        print("Top 10 mismatches (image, hbb, obb):")
        for image_id, s, o in sorted(diffs, key=lambda x: abs(x[1] - x[2]), reverse=True)[:10]:
            print(f"  {image_id}: hbb={s}, obb={o}, delta={o - s:+d}")


if __name__ == "__main__":
    main()
