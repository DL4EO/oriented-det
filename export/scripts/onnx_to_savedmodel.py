#!/usr/bin/env python3
"""Convert ONNX to TensorFlow SavedModel using onnx2tf."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _find_onnx2tf() -> list[str]:
    exe = shutil.which("onnx2tf")
    if exe:
        return [exe]
    return [sys.executable, "-m", "onnx2tf"]


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX → SavedModel via onnx2tf.")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output directory for SavedModel.")
    args = p.parse_args()

    # tf_converter can fail on some ops (e.g. ScatterND); flatbuffer_direct still useful for TFLite.
    # build_faster_rcnn_savedmodel.py falls back to ONNX Runtime when no saved_model.pb is found.
    cmd = _find_onnx2tf() + ["-i", str(args.onnx), "-o", str(args.output), "-osd"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"SavedModel directory: {args.output}")


if __name__ == "__main__":
    main()
