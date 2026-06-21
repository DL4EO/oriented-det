#!/usr/bin/env python3
"""Build e2e Faster R-CNN SavedModel (Keras + ORT core + exact rotated NMS)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export.postprocess import meta_to_finalize_kwargs  # noqa: E402
from export.tf_serving_model import FasterRCNNDetectLayer, save_keras_detect_bundle  # noqa: E402


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_onnx_path(meta: dict, meta_path: Path, onnx_path: Optional[Path]) -> Path:
    if onnx_path is not None and onnx_path.is_file():
        return onnx_path
    if meta.get("onnx_path"):
        p = Path(meta["onnx_path"])
        if p.is_file():
            return p
    candidate = meta_path.with_suffix("").with_suffix(".onnx")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "ONNX file not found; pass --onnx or ensure meta onnx_path / sibling .onnx exists."
    )


def build_savedmodel(
    output_path: Path,
    meta_path: Path,
    *,
    onnx_path: Optional[Path] = None,
) -> None:
    import tensorflow as tf

    meta = _load_meta(meta_path)
    finalize_kwargs = meta_to_finalize_kwargs(meta)
    onnx_file = _resolve_onnx_path(meta, meta_path, onnx_path)

    h = int(meta["input"]["shape"][2])
    w = int(meta["input"]["shape"][3])
    inputs = tf.keras.Input(shape=(3, h, w), batch_size=1, name="images")
    layer = FasterRCNNDetectLayer(
        onnx_path=str(onnx_file.resolve()),
        ort_output_names=list(meta.get("output_names") or []),
        finalize_kwargs=finalize_kwargs,
        max_output_slots=int(finalize_kwargs["max_output_slots"]),
    )
    layer_out = layer(inputs)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[layer_out["detections"], layer_out["num_detections"]],
    )

    full_meta = dict(meta)
    full_meta["core_backend"] = "onnxruntime_keras"
    full_meta["onnx_path"] = str(onnx_file)
    full_meta["savedmodel_outputs"] = {
        "detections": {
            "shape": [finalize_kwargs["max_output_slots"], 7],
            "layout": "cx,cy,w,h,angle,score,label",
        },
        "num_detections": {"dtype": "int32"},
    }

    keras_path = save_keras_detect_bundle(model, output_path, full_meta)
    print(f"Wrote detect bundle: {output_path} (keras: {keras_path.name}, core: onnxruntime)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build e2e Faster R-CNN SavedModel.")
    p.add_argument(
        "--tf-core",
        type=Path,
        default=None,
        help="Ignored (kept for CLI compatibility with README pipeline).",
    )
    p.add_argument("--meta", type=Path, required=True, help="*.export_meta.json from export_onnx.py.")
    p.add_argument("--onnx", type=Path, default=None, help="ONNX file (default: from meta / sibling).")
    p.add_argument("--output", type=Path, required=True, help="Output SavedModel directory.")
    args = p.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        print("Install TensorFlow: pip install -r export/requirements-export.txt", file=sys.stderr)
        raise SystemExit(1) from e

    build_savedmodel(args.output, args.meta, onnx_path=args.onnx)


if __name__ == "__main__":
    main()
