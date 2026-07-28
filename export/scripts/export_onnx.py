#!/usr/bin/env python3
"""Export a tensor-only subgraph from oriented-det to ONNX.

See export/README.md and export/contract.json for supported modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from export import wrappers as _wrappers  # noqa: E402
except ImportError:  # pragma: no cover - legacy path layout
    _EXPORT_DIR = _REPO_ROOT / "export"
    if str(_EXPORT_DIR) not in sys.path:
        sys.path.insert(0, str(_EXPORT_DIR))
    import wrappers as _wrappers  # noqa: E402

from oriented_det.models.oriented_rcnn import OrientedRCNN, RotatedFasterRCNN  # noqa: E402
from oriented_det.models.rotated_retinanet import RotatedRetinaNet  # noqa: E402
from oriented_det.runtime.checkpoint import load_model_from_checkpoint  # noqa: E402

BackboneExportWrapper = _wrappers.BackboneExportWrapper
RetinaNetBackboneHeadExportWrapper = _wrappers.RetinaNetBackboneHeadExportWrapper
RotatedFasterRCNNPreNmsExportWrapper = _wrappers.RotatedFasterRCNNPreNmsExportWrapper
OrientedRCNNPreNmsExportWrapper = _wrappers.OrientedRCNNPreNmsExportWrapper

PRE_NMS_OUTPUTS = (
    "pre_nms_boxes",
    "pre_nms_scores",
    "pre_nms_labels",
    "pre_nms_count",
)
FASTER_RCNN_PRE_NMS_OUTPUTS = PRE_NMS_OUTPUTS  # backwards-compatible alias
_PRE_NMS_MODES = frozenset({"faster_rcnn_pre_nms", "oriented_rcnn_pre_nms"})


def _build_wrapper(
    model: torch.nn.Module,
    mode: str,
    height: int,
    width: int,
) -> torch.nn.Module:
    mt = type(model).__name__
    if mode == "backbone":
        if not hasattr(model, "backbone"):
            raise ValueError(f"Model {mt} has no 'backbone' attribute.")
        return BackboneExportWrapper(model.backbone)
    if mode == "retinanet_heads":
        if not isinstance(model, RotatedRetinaNet):
            raise ValueError(
                f"retinanet_heads mode requires RotatedRetinaNet, got {mt}. "
                "Use --mode backbone for two-stage models."
            )
        return RetinaNetBackboneHeadExportWrapper(model)
    if mode == "faster_rcnn_pre_nms":
        if not isinstance(model, RotatedFasterRCNN):
            raise ValueError(
                f"faster_rcnn_pre_nms requires RotatedFasterRCNN, got {mt}."
            )
        return RotatedFasterRCNNPreNmsExportWrapper(model, height=height, width=width)
    if mode == "oriented_rcnn_pre_nms":
        if not isinstance(model, OrientedRCNN):
            raise ValueError(
                f"oriented_rcnn_pre_nms requires OrientedRCNN, got {mt}."
            )
        return OrientedRCNNPreNmsExportWrapper(model, height=height, width=width)
    raise ValueError(f"Unknown mode: {mode}")


def _output_names_for_wrapper(
    wrapper: torch.nn.Module,
    mode: str,
    device: torch.device,
    height: int,
    width: int,
) -> list[str]:
    if mode in _PRE_NMS_MODES:
        return list(PRE_NMS_OUTPUTS)

    dummy = torch.zeros(1, 3, 128, 128, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = wrapper(dummy)
    names: list[str] = []
    if mode == "backbone":
        for i in range(len(out)):
            names.append(f"fpn_level_{i}")
        return names
    if mode == "retinanet_heads":
        for i in range(len(out) // 2):
            names.append(f"level{i}_cls_logits")
            names.append(f"level{i}_bbox_pred")
        return names
    return [f"out_{i}" for i in range(len(out))]


def _production_dict(config: Any) -> dict | None:
    prod = getattr(config, "production", None)
    if prod is None:
        return None
    if hasattr(prod, "__dataclass_fields__"):
        return asdict(prod)
    if isinstance(prod, dict):
        return prod
    return None


def _validate_onnx_ort(onnx_path: Path, dummy: torch.Tensor, output_names: list[str]) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Skipping ORT validation (install onnx and onnxruntime via oriented-det[export]).")
        return

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {"images": dummy.detach().cpu().numpy()}
    outs = sess.run(output_names, feeds)
    print(f"ORT smoke OK: {len(outs)} outputs")
    for name, arr in zip(output_names, outs):
        print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export oriented-det subgraph to ONNX.")
    p.add_argument("--config", type=Path, required=True, help="Training JSON config path.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Weights .pth path.")
    p.add_argument("--output", type=Path, required=True, help="Output .onnx file path.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--mode",
        choices=(
            "backbone",
            "retinanet_heads",
            "faster_rcnn_pre_nms",
            "oriented_rcnn_pre_nms",
        ),
        default="backbone",
        help=(
            "backbone | retinanet_heads | faster_rcnn_pre_nms | "
            "oriented_rcnn_pre_nms (two-stage detect pre-NMS)."
        ),
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow dynamic batch size (N) on input images; H and W stay fixed.",
    )
    p.add_argument(
        "--skip-ort",
        action="store_true",
        help="Skip onnxruntime validation after export.",
    )
    p.add_argument("--device", default="cpu", help="Device to trace on (cpu recommended for reproducibility).")
    args = p.parse_args()

    if args.dynamic_batch and args.mode in _PRE_NMS_MODES:
        raise SystemExit(
            f"--dynamic-batch is not supported with --mode {args.mode} "
            "(pre-NMS detect export requires batch size 1)."
        )

    model, config, class_names = load_model_from_checkpoint(
        str(args.checkpoint),
        str(args.config),
        device=args.device,
    )
    h, w = int(args.height), int(args.width)
    wrapper = _build_wrapper(model, args.mode, h, w)
    wrapper.eval()
    wrapper.to(args.device)

    dev = torch.device(args.device)
    out_names = _output_names_for_wrapper(wrapper, args.mode, dev, h, w)
    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32, device=dev)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}
        for name in out_names:
            dynamic_axes[name] = {0: "batch"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )

    meta: dict[str, Any] = {
        "mode": args.mode,
        "input": {"name": "images", "shape": [1, 3, h, w], "dtype": "float32"},
        "output_names": out_names,
        "opset": args.opset,
        "dynamic_batch": args.dynamic_batch,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "class_names": class_names or [],
        "num_classes": int(getattr(config, "num_classes", 0) or 0),
        "production": _production_dict(config),
    }
    if args.mode in _PRE_NMS_MODES and isinstance(
        wrapper, (RotatedFasterRCNNPreNmsExportWrapper, OrientedRCNNPreNmsExportWrapper)
    ):
        meta["max_pre_nms_candidates"] = wrapper.max_candidates
        prod = meta.get("production") or {}
        meta["max_detections_per_image"] = prod.get("max_detections_per_image")

    meta["onnx_path"] = str(args.output.resolve())
    meta_path = args.output.with_suffix(".export_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote ONNX: {args.output}")
    print(f"Wrote meta: {meta_path}")

    if not args.skip_ort:
        _validate_onnx_ort(args.output, dummy, out_names)


if __name__ == "__main__":
    main()
