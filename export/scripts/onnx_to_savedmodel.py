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


def convert_onnx_to_savedmodel(
    onnx_path: Path,
    output_dir: Path,
    *,
    keep_ncw: bool = False,
) -> Path:
    """Run onnx2tf and return the SavedModel directory.

    Raises:
        FileNotFoundError: onnx2tf is not installed.
        subprocess.CalledProcessError: conversion failed.
        FileNotFoundError: no ``saved_model.pb`` was produced.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _find_onnx2tf() + ["-i", str(onnx_path), "-o", str(output_dir), "-osd"]
    if keep_ncw:
        cmd.append("-k")
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "onnx2tf not found. Install with: pip install \"oriented-det[export]\""
        ) from e
    pb = output_dir / "saved_model.pb"
    if not pb.is_file():
        # onnx2tf sometimes nests the SavedModel one level down
        nested = list(output_dir.glob("*/saved_model.pb"))
        if nested:
            return nested[0].parent
        raise FileNotFoundError(
            f"onnx2tf did not write saved_model.pb under {output_dir} "
            "(this graph may fail to convert, e.g. ScatterND)."
        )
    return output_dir


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX → SavedModel via onnx2tf.")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output directory for SavedModel.")
    p.add_argument(
        "--keep-ncw",
        action="store_true",
        help="Keep NCHW layout (onnx2tf -k). Default converts to NHWC.",
    )
    args = p.parse_args()

    # tf_converter can fail on some ops (e.g. ScatterND); flatbuffer_direct still useful for TFLite.
    # Detect SavedModel export falls back to ONNX Runtime when conversion fails.
    sm_dir = convert_onnx_to_savedmodel(args.onnx, args.output, keep_ncw=args.keep_ncw)
    print(f"SavedModel directory: {sm_dir}")


if __name__ == "__main__":
    main()
