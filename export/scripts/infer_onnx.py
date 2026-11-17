#!/usr/bin/env python3
"""Run a pre-NMS ONNX export on a folder of images (ORT + Python NMS).

Works from the repo (``odet export infer`` / ``python -m export infer``) and from a copied
``onnx_export/`` bundle (``python infer_onnx.py ...``). The copied bundle does not need oriented-det.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = (_HERE / "runtime.py").is_file() and (_HERE / "nms.py").is_file()
if _BUNDLE:
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from preprocess import canvas_size_from_meta  # noqa: E402
    from runtime import (  # noqa: E402
        detect_array,
        detect_image,
        list_images,
        load_export_meta,
        merge_per_class_score_threshold,
    )
else:
    _REPO_ROOT = _HERE.parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from export.preprocess import canvas_size_from_meta  # noqa: E402
    from export.runtime import (  # noqa: E402
        detect_array,
        detect_image,
        list_images,
        load_export_meta,
        merge_per_class_score_threshold,
    )


def parse_per_class_score_threshold(value: str) -> tuple[str, float]:
    """Parse one ``CLASS=FLOAT`` override.

    Raises ``argparse.ArgumentTypeError`` when ``value`` is not ``CLASS=FLOAT``.
    """
    text = str(value).strip()
    name, sep, raw = text.partition("=")
    name = name.strip()
    raw = raw.strip()
    if not sep or not name or not raw:
        raise argparse.ArgumentTypeError(f"Expected CLASS=FLOAT, got {value!r}")
    try:
        score = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected CLASS=FLOAT, got {value!r} (invalid float {raw!r})"
        ) from None
    return name, score


def _per_class_override(
    pairs: list[tuple[str, float]] | None,
) -> dict[str, float] | None:
    """Collapse repeatable CLI pairs. ``None`` or empty keeps the sidecar map."""
    if not pairs:
        return None
    return {name: score for name, score in pairs}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="ONNX Runtime inference for an export bundle.")
    p.add_argument("--onnx", type=Path, required=True, help="Path to model.onnx.")
    p.add_argument("--meta", type=Path, default=None, help="Sidecar JSON (default: next to --onnx).")
    p.add_argument("--images", type=Path, default=None, help="Folder of images.")
    p.add_argument("--output", type=Path, default=None, help="Write per-image JSON + predictions.json.")
    p.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Override the global production score floor (default: meta). "
            "Does not change per-class floors."
        ),
    )
    p.add_argument(
        "--per-class-score-threshold",
        action="append",
        type=parse_per_class_score_threshold,
        default=None,
        metavar="CLASS=FLOAT",
        help=(
            "Override one production.per_class_score_threshold entry (repeatable). "
            "Omitted classes keep the sidecar floor. Names match case-insensitively "
            "and keep the sidecar spelling."
        ),
    )
    p.add_argument(
        "--nms-backend",
        choices=("python", "shapely", "auto"),
        default=None,
        help="CPU NMS IoU backend (default: meta / python).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a zeros tensor through ONNX + NMS (no images).",
    )
    args = p.parse_args(argv)
    per_class_override = _per_class_override(args.per_class_score_threshold)

    if not args.onnx.is_file():
        raise SystemExit(f"Missing ONNX: {args.onnx}")
    meta = load_export_meta(args.onnx, args.meta)
    prod = meta.get("production") or {}
    effective_per_class = merge_per_class_score_threshold(
        prod.get("per_class_score_threshold"),
        per_class_override,
    )
    score_shown = (
        args.score_threshold
        if args.score_threshold is not None
        else (meta.get("postprocess") or meta).get("score_threshold")
    )
    print(f"ONNX  {args.onnx}")
    print(f"mode  {meta.get('mode')}")
    print(f"thr   score={score_shown} per_class={effective_per_class}")

    if args.smoke:
        import numpy as np

        h, w = canvas_size_from_meta(meta)
        blob = np.zeros((1, 3, h, w), dtype=np.float32)
        dets = detect_array(
            blob,
            args.onnx,
            meta,
            score_threshold=args.score_threshold,
            per_class_score_threshold=per_class_override,
            nms_backend=args.nms_backend,
        )
        print(f"smoke detections: {len(dets)}")
        return

    if args.images is None:
        raise SystemExit("Pass --images DIR or --smoke")
    files = list_images(args.images)
    all_preds = []
    out_dir = args.output
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        dets = detect_image(
            path,
            args.onnx,
            meta,
            score_threshold=args.score_threshold,
            per_class_score_threshold=per_class_override,
            nms_backend=args.nms_backend,
        )
        rec = {"image": path.name, "count": len(dets), "detections": dets}
        all_preds.append(rec)
        print(f"{path.name}: {len(dets)}")
        if out_dir is not None:
            (out_dir / f"{path.stem}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if out_dir is not None:
        (out_dir / "predictions.json").write_text(json.dumps(all_preds, indent=2), encoding="utf-8")
        print(f"Wrote {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
