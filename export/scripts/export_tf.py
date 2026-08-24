#!/usr/bin/env python3
"""Orchestrate ONNX pre-NMS export + Keras detect bundle (no Makefile)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_PRE_NMS_MODES = (
    "faster_rcnn_pre_nms",
    "oriented_rcnn_pre_nms",
    "rotated_fcos_pre_nms",
)


def _require_export_extras(*, require_ort: bool = True) -> None:
    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    if require_ort:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            missing.append("onnxruntime")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")
    if missing:
        raise SystemExit(
            "Missing export dependencies: "
            + ", ".join(missing)
            + '. Install with: pip install "oriented-det[export]"'
        )


def _call_main(module_path: str, argv: list[str]) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    if not hasattr(mod, "main"):
        raise SystemExit(f"{module_path} has no main()")
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old


def run_export_tf(
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    mode: str = "faster_rcnn_pre_nms",
    height: int = 1024,
    width: int = 1024,
    device: str = "cpu",
    opset: int = 17,
    skip_ort: bool = False,
    saved_model: bool = False,
) -> Path:
    """Export ONNX then build detect bundle under ``output_dir``. Returns detect dir."""
    if mode not in _PRE_NMS_MODES:
        raise SystemExit(
            f"export-tf requires a pre-NMS detect mode ({', '.join(_PRE_NMS_MODES)}), got {mode!r}."
        )

    config = Path(config)
    checkpoint = Path(checkpoint)
    if not config.is_file():
        raise SystemExit(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    # ORT is only needed for post-ONNX smoke unless --skip-ort; inference still needs it later.
    _require_export_extras(require_ort=not skip_ort)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "pre_nms.onnx"
    detect_dir = output_dir / "detect"
    # export_onnx writes <stem>.export_meta.json next to the .onnx
    meta_path = onnx_path.with_suffix(".export_meta.json")

    onnx_argv = [
        "odet-export-onnx",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(onnx_path),
        "--height",
        str(height),
        "--width",
        str(width),
        "--mode",
        mode,
        "--device",
        device,
        "--opset",
        str(opset),
    ]
    if skip_ort:
        onnx_argv.append("--skip-ort")
    _call_main("export.scripts.export_onnx", onnx_argv)

    if not meta_path.is_file():
        raise SystemExit(f"Expected export meta at {meta_path}")

    detect_argv = [
        "odet-export-detect",
        "--meta",
        str(meta_path),
        "--onnx",
        str(onnx_path),
        "--output",
        str(detect_dir),
    ]
    _call_main("export.scripts.build_faster_rcnn_savedmodel", detect_argv)

    keras_path = detect_dir / "keras_model.keras"
    bundled_onnx = detect_dir / "model.onnx"
    if not keras_path.is_file() or not bundled_onnx.is_file():
        raise SystemExit(
            f"Detect bundle incomplete under {detect_dir} "
            "(expected keras_model.keras and model.onnx)."
        )
    print(f"Detect bundle: {keras_path}")
    if saved_model:
        sm_dir = output_dir / "saved_model"
        sm_argv = [
            "odet-export-savedmodel",
            "--meta",
            str(meta_path),
            "--onnx",
            str(onnx_path),
            "--output",
            str(sm_dir),
        ]
        _call_main("export.scripts.build_tf_savedmodel", sm_argv)
        if not (sm_dir / "saved_model.pb").is_file():
            raise SystemExit(f"SavedModel incomplete under {sm_dir} (expected saved_model.pb).")
        print(f"SavedModel: {sm_dir}")
    return detect_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export ONNX pre-NMS + Keras detect bundle (installable; no Makefile). "
            'Requires: pip install "oriented-det[export]".'
        )
    )
    p.add_argument("--config", type=Path, required=True, help="Training JSON config path.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Weights .pth path.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("odet_export"),
        help="Output directory (default: ./odet_export). Writes pre_nms.onnx + detect/.",
    )
    p.add_argument(
        "--saved-model",
        action="store_true",
        help=(
            "Also write a TensorFlow SavedModel under <output-dir>/saved_model "
            "(onnx2tf + TF rotated NMS). Load with tf.saved_model.load; no oriented-det "
            "at inference. May fail on graphs onnx2tf cannot convert (e.g. ScatterND)."
        ),
    )
    p.add_argument(
        "--mode",
        choices=_PRE_NMS_MODES,
        default="faster_rcnn_pre_nms",
        help="Pre-NMS detect export mode (default: faster_rcnn_pre_nms).",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--skip-ort",
        action="store_true",
        help=(
            "Skip onnxruntime validation after ONNX export "
            "(also skips requiring onnxruntime at export time; install it before inference)."
        ),
    )
    args = p.parse_args()

    run_export_tf(
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        height=args.height,
        width=args.width,
        device=args.device,
        opset=args.opset,
        skip_ort=args.skip_ort,
        saved_model=args.saved_model,
    )


if __name__ == "__main__":
    main()
