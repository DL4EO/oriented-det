#!/usr/bin/env python3
"""Smoke-test an export detect bundle (keras_model.keras) or legacy TF SavedModel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_keras_bundle(bundle_dir: Path, height: int, width: int, ort_device: str | None) -> None:
    # ORT device + hide TF GPUs *before* importing tensorflow / keras.
    from export.ort_runtime import configure_ort_device, get_ort_device

    providers = configure_ort_device(ort_device)
    print(f"  ort_device: {get_ort_device()}  providers: {providers}")

    import tensorflow as tf

    from export.tf_serving_model import load_keras_detect_model

    keras_path = bundle_dir / "keras_model.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing {keras_path}")
    model = load_keras_detect_model(keras_path)
    x = tf.zeros([1, 3, height, width], dtype=tf.float32)
    detections, num_detections = model(x, training=False)
    print(f"  detections: shape={detections.shape} dtype={detections.dtype}")
    print(f"  num_detections: {int(num_detections.numpy())}")
    meta_path = bundle_dir / "export_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  core_backend: {meta.get('core_backend', 'unknown')}")


def _run_saved_model(sm_dir: Path, height: int, width: int) -> None:
    import tensorflow as tf

    sm = tf.saved_model.load(str(sm_dir))
    sigs = list(getattr(sm, "signatures", {}).keys())
    print("Signatures:", sigs)
    name = "serving_default" if "serving_default" in sigs else sigs[0]
    fn = sm.signatures[name]

    kwargs = {}
    _pos, kw = fn.structured_input_signature
    if kw:
        for key, spec in kw.items():
            if spec.dtype.is_floating and spec.shape.rank == 4:
                shape = [1, 3, height, width]
                kwargs[key] = tf.zeros(shape, dtype=spec.dtype)
    if not kwargs:
        raise SystemExit("Could not infer image input from SavedModel signature.")
    out = fn(**kwargs)
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test export detect bundle or SavedModel.")
    p.add_argument(
        "--saved-model",
        type=Path,
        required=True,
        help="Directory with keras_model.keras (+ export_meta.json) or TF SavedModel.",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--ort-device",
        default=None,
        choices=("cpu", "cuda", "auto"),
        help="ONNX Runtime EP for keras bundle (default: cpu or ORIENTED_DET_ORT_DEVICE).",
    )
    args = p.parse_args()

    bundle = args.saved_model
    try:
        if (bundle / "keras_model.keras").is_file():
            _run_keras_bundle(bundle, args.height, args.width, args.ort_device)
        else:
            _run_saved_model(bundle, args.height, args.width)
    except ImportError as e:
        if "tensorflow" in str(e).lower() or getattr(e, "name", "") == "tensorflow":
            print(
                "Install TensorFlow: pip install -r export/requirements-export.txt",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        raise


if __name__ == "__main__":
    main()
