#!/usr/bin/env python
"""Convert horizontal GT boxes to oriented boxes using model predictions.

For each GT box, uses the matched oriented model prediction when available;
otherwise falls back to the horizontal GT box (0° rotation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hbb_to_obb import (
    add_gt_cli_args,
    build_oriented_annotations,
    filter_predictions,
    resolve_gt_from_args,
    write_oriented_csv,
    write_oriented_dota_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gt_cli_args(parser)
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="Combined detections.json from odet image-demo --json-batch.",
    )
    parser.add_argument(
        "--filtered-json",
        type=Path,
        default=None,
        help="Optional pre-filtered JSON; if omitted, GT filter runs on the fly.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "dota"),
        default="csv",
        help="Output format: csv (single annotations_oriented.csv) or dota (per-image .txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path or DOTA labels directory.",
    )
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument(
        "--model-source-label",
        default="model",
        help="Value for the CSV 'source' column on model-matched rows.",
    )
    args = parser.parse_args()

    gt_by_image = resolve_gt_from_args(args)

    if args.filtered_json is not None:
        filtered = json.loads(args.filtered_json.read_text(encoding="utf-8"))
    else:
        filtered = filter_predictions(args.detections_json, gt_by_image, args.match_iou)

    rows = build_oriented_annotations(
        gt_by_image,
        filtered,
        model_source_label=args.model_source_label,
    )

    if args.output_format == "dota":
        if args.output is not None:
            out_dir = args.output
        elif args.dataset_root is not None:
            out_dir = args.dataset_root / "labels_oriented"
        else:
            parser.error("--output or --dataset-root required for --output-format dota")
        write_oriented_dota_labels(out_dir, rows)
    else:
        if args.output is not None:
            out_path = args.output
        elif args.dataset_root is not None:
            out_path = args.dataset_root / "annotations_oriented.csv"
        elif args.annotations is not None:
            out_path = args.annotations.parent / "annotations_oriented.csv"
        else:
            parser.error("--output required when --dataset-root is not set")
        write_oriented_csv(out_path, rows)


if __name__ == "__main__":
    main()
