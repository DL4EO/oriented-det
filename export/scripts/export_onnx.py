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

from export import wrappers as _wrappers  # noqa: E402
from export.runtime import copy_runtime_sidecars  # noqa: E402
from oriented_det.models.oriented_rcnn import OrientedRCNN, RotatedFasterRCNN  # noqa: E402
from oriented_det.models.rotated_fcos import RotatedFCOS  # noqa: E402
from oriented_det.models.rotated_retinanet import RotatedRetinaNet  # noqa: E402
from oriented_det.models.rotated_rtmdet import RotatedRTMDet  # noqa: E402
from oriented_det.runtime.checkpoint import load_model_from_checkpoint  # noqa: E402

BackboneExportWrapper = _wrappers.BackboneExportWrapper
RetinaNetBackboneHeadExportWrapper = _wrappers.RetinaNetBackboneHeadExportWrapper
RotatedFasterRCNNPreNmsExportWrapper = _wrappers.RotatedFasterRCNNPreNmsExportWrapper
OrientedRCNNPreNmsExportWrapper = _wrappers.OrientedRCNNPreNmsExportWrapper
RotatedFCOSHeadsExportWrapper = _wrappers.RotatedFCOSHeadsExportWrapper
RotatedFCOSPreNmsExportWrapper = _wrappers.RotatedFCOSPreNmsExportWrapper
RotatedRTMDetPreNmsExportWrapper = _wrappers.RotatedRTMDetPreNmsExportWrapper

PRE_NMS_OUTPUTS = (
    "pre_nms_boxes",
    "pre_nms_scores",
    "pre_nms_labels",
    "pre_nms_count",
)
FASTER_RCNN_PRE_NMS_OUTPUTS = PRE_NMS_OUTPUTS  # backwards-compatible alias
_PRE_NMS_MODES = frozenset(
    {
        "faster_rcnn_pre_nms",
        "oriented_rcnn_pre_nms",
        "rotated_fcos_pre_nms",
        "rotated_rtmdet_pre_nms",
    }
)


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
    if mode == "rotated_fcos_heads":
        if not isinstance(model, RotatedFCOS):
            raise ValueError(
                f"rotated_fcos_heads requires RotatedFCOS, got {mt}."
            )
        return RotatedFCOSHeadsExportWrapper(model, height=height, width=width)
    if mode == "rotated_fcos_pre_nms":
        if not isinstance(model, RotatedFCOS):
            raise ValueError(
                f"rotated_fcos_pre_nms requires RotatedFCOS, got {mt}."
            )
        return RotatedFCOSPreNmsExportWrapper(model, height=height, width=width)
    if mode == "rotated_rtmdet_pre_nms":
        if not isinstance(model, RotatedRTMDet):
            raise ValueError(
                f"rotated_rtmdet_pre_nms requires RotatedRTMDet, got {mt}."
            )
        return RotatedRTMDetPreNmsExportWrapper(model, height=height, width=width)
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
    if mode == "rotated_fcos_heads":
        n_levels = max(1, len(wrapper.model.fpn_strides))
        names: list[str] = []
        for i in range(n_levels):
            names.extend(
                [
                    f"level{i}_cls",
                    f"level{i}_bbox",
                    f"level{i}_angle",
                    f"level{i}_centerness",
                ]
            )
        return names

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


def _config_section(config: Any, name: str) -> dict:
    obj = getattr(config, name, None)
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _production_dict(config: Any) -> dict | None:
    prod = _config_section(config, "production")
    return prod or None


def build_export_meta(
    *,
    mode: str,
    config: Any,
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    height: int,
    width: int,
    opset: int,
    dynamic_batch: bool,
    config_path: str,
    checkpoint_path: str,
    class_names: list[str],
    output_names: list[str],
    onnx_path: Path,
) -> dict[str, Any]:
    """Sidecar JSON: ONNX I/O plus preprocess / postprocess parameters."""
    prep = _config_section(config, "preprocessing")
    prod = _production_dict(config) or {}
    ev = _config_section(config, "evaluation")
    mean = list(prep.get("normalize_mean") or [0.485, 0.456, 0.406])
    std = list(prep.get("normalize_std") or [0.229, 0.224, 0.225])
    score_thr = float(prod.get("score_threshold") or ev.get("score_threshold") or 0.2)
    nms_iou = float(
        prod.get("final_nms_iou_threshold")
        or getattr(model, "final_nms_iou_threshold", 0.5)
        or 0.5
    )
    nms_ag = bool(prod.get("nms_class_agnostic", True))
    max_det = int(
        prod.get("max_detections_per_image")
        or getattr(model, "max_detections_per_image", 500)
        or 500
    )
    pre_nms_score = float(
        prod.get("inference_pre_nms_score_threshold")
        or getattr(model, "score_threshold", 0.05)
        or 0.05
    )
    meta: dict[str, Any] = {
        "mode": mode,
        "input": {
            "name": "images",
            "shape": [1, 3, int(height), int(width)],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "normalized": True,
        },
        "output_names": list(output_names),
        "opset": int(opset),
        "dynamic_batch": bool(dynamic_batch),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "class_names": list(class_names or []),
        "num_classes": int(getattr(config, "num_classes", 0) or 0),
        "production": prod or None,
        "preprocess": {
            "resize_mode": prep.get("resize_mode") or "fixed",
            "target_size": [int(height), int(width)],
            "normalize_mean": mean,
            "normalize_std": std,
            "pad_size_divisor": int(prep.get("pad_size_divisor") or 1),
        },
        "postprocess": {
            "score_threshold": score_thr,
            "nms_iou_threshold": nms_iou,
            "nms_class_agnostic": nms_ag,
            "max_detections": max_det,
            "final_nms_use_cpu": bool(prod.get("final_nms_use_cpu", True)),
            "pre_nms_score_threshold": pre_nms_score,
            "nms_backend": str(prod.get("nms_backend") or "python"),
        },
        "normalize_mean": mean,
        "normalize_std": std,
        "image_size": int(height) if int(height) == int(width) else None,
        "score_threshold": score_thr,
        "nms_iou_threshold": nms_iou,
        "nms_class_agnostic": nms_ag,
        "max_detections": max_det,
        "pre_nms_score_threshold": pre_nms_score,
        "onnx_path": str(Path(onnx_path).resolve()),
    }
    if mode in _PRE_NMS_MODES and hasattr(wrapper, "max_candidates"):
        meta["max_pre_nms_candidates"] = int(wrapper.max_candidates)
        meta["max_detections_per_image"] = max_det
    if mode in {"rotated_fcos_heads", "rotated_fcos_pre_nms"} and isinstance(model, RotatedFCOS):
        meta["fpn_strides"] = [int(s) for s in model.fpn_strides]
        meta["nms_pre"] = int(model.nms_pre)
        meta["min_bbox_size"] = float(model.min_bbox_size)
    if mode == "rotated_rtmdet_pre_nms" and isinstance(model, RotatedRTMDet):
        meta["fpn_strides"] = [int(s) for s in model.fpn_strides]
        meta["nms_pre"] = int(model.nms_pre)
        meta["min_bbox_size"] = float(model.min_bbox_size)
    return meta


def _validate_onnx_ort(onnx_path: Path, dummy: torch.Tensor, output_names: list[str]) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Skipping ORT validation (install onnx and onnxruntime: pip install -r export/requirements-export.txt).")
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
            "rotated_fcos_heads",
            "rotated_fcos_pre_nms",
            "rotated_rtmdet_pre_nms",
        ),
        default="rotated_fcos_pre_nms",
        help=(
            "Default rotated_fcos_pre_nms (decode in ONNX, NMS in Python). "
            "Also: backbone | retinanet_heads | faster_rcnn_pre_nms | "
            "oriented_rcnn_pre_nms | rotated_fcos_heads | rotated_rtmdet_pre_nms."
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

    meta = build_export_meta(
        mode=args.mode,
        config=config,
        model=model,
        wrapper=wrapper,
        height=h,
        width=w,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        class_names=list(class_names or []),
        output_names=out_names,
        onnx_path=args.output,
    )
    meta_path = args.output.with_suffix(".export_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote ONNX: {args.output}")
    print(f"Wrote meta: {meta_path}")
    copy_runtime_sidecars(args.output)

    if not args.skip_ort:
        _validate_onnx_ort(args.output, dummy, out_names)


if __name__ == "__main__":
    main()
