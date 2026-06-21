#!/usr/bin/env python3
"""Optional: convert SavedModel to TFLite (float32)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="SavedModel → TFLite float32.")
    p.add_argument("--saved-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output .tflite path")
    args = p.parse_args()

    try:
        import tensorflow as tf
    except ImportError as e:
        print("Install TensorFlow: pip install -r export/requirements-export.txt", file=sys.stderr)
        raise SystemExit(1) from e

    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.saved_model))
    converter.optimizations = []
    tflite_model = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_model)
    print(f"Wrote {args.output} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()
