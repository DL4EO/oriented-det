#!/usr/bin/env python3
"""Build a TensorFlow SavedModel (onnx2tf core + TF rotated NMS).

Reload without oriented-det::

    import tensorflow as tf
    from export.tf_savedmodel import call_saved_model
    sm = tf.saved_model.load("./odet_export/saved_model")
    detections, num = call_saved_model(sm, images)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_onnx_path(meta: dict, meta_path: Path, onnx_path: Path | None) -> Path:
    if onnx_path is not None and onnx_path.is_file():
        return onnx_path
    if meta.get("onnx_path"):
        p = Path(meta["onnx_path"])
        if p.is_file():
            return p
        sibling = meta_path.parent / p.name
        if sibling.is_file():
            return sibling
    stem = meta_path.name
    if stem.endswith(".export_meta.json"):
        candidate = meta_path.with_name(stem[: -len(".export_meta.json")] + ".onnx")
    else:
        candidate = meta_path.with_suffix(".onnx")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "ONNX file not found; pass --onnx or ensure meta onnx_path / sibling .onnx exists."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="ONNX + meta → TensorFlow SavedModel (pure TF graph, no ORT)."
    )
    p.add_argument("--meta", type=Path, required=True, help="*.export_meta.json from export_onnx.py.")
    p.add_argument("--onnx", type=Path, default=None, help="ONNX file (default: from meta / sibling).")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output SavedModel directory.",
    )
    args = p.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        print(
            'Install TensorFlow extras: pip install "oriented-det[export]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    from export.postprocess import meta_to_finalize_kwargs
    from export.tf_savedmodel import build_savedmodel_from_onnx

    meta = _load_meta(args.meta)
    onnx_file = _resolve_onnx_path(meta, args.meta, args.onnx)
    h = int(meta["input"]["shape"][2])
    w = int(meta["input"]["shape"][3])
    finalize_kwargs = meta_to_finalize_kwargs(meta)
    try:
        out = build_savedmodel_from_onnx(
            onnx_file,
            args.output,
            finalize_kwargs=finalize_kwargs,
            height=h,
            width=w,
            meta=meta,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as e:
        print(
            f"SavedModel export failed: {e}\n"
            "A portable SavedModel needs onnx2tf to lower the pre-NMS ONNX graph. "
            "Some Faster R-CNN graphs fail on ScatterND; Oriented R-CNN is more likely "
            "to convert. The Keras detect bundle (ORT + Python NMS) remains the "
            "supported full-detect path.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    print(f"Wrote SavedModel: {out}")


if __name__ == "__main__":
    main()
