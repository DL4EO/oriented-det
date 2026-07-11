#!/usr/bin/env python
"""Image demo for oriented-det: run inference with config + checkpoint.

Takes config, checkpoint, and image(s), runs inference, and saves visualizations.
Uses oriented-det's config (JSON) and checkpoints. Supports a single image or a directory of images.

Inference path (matches ``oriented_det.runtime.inference`` semantics):

- If the image size **equals** the model input from config (``preprocessing.target_size`` as
  height×width, same as training), runs **one forward pass** on a tensor built with
  ToTensor+normalize only (no resize).
- Otherwise uses **padded-canvas / sliding windows** to that size (zero-pad when smaller, tile
  when larger), merges boxes in original image coordinates, then NMS.
- With ``--zoom 2`` or ``--zoom 4``, resizes only the inference image, runs the same logic above,
  then maps detections back to the original image size before visualization.

Example:
  # Single image
  python tools/image_demo.py demo/demo.jpg configs/oriented_rcnn/dota_le90_1x.json runs/.../checkpoints/best.pth --out-file result.jpg

  # All images in demo/
  python tools/image_demo.py demo configs/.../config.json runs/.../checkpoints/best.pth --out-dir demo/out
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add project root so "from oriented_det" works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from oriented_det import RBox
from oriented_det.train.config import (
    get_preprocessing_params,
    resolve_inference_score_threshold,
    resolve_inference_sliding_window_overlap_pixels,
)
from oriented_det.runtime.checkpoint import load_model_from_checkpoint, resolve_inference_config_path
from oriented_det.runtime.inference import (
    count_sliding_window_positions,
    get_model_size,
    preprocess_crop,
    run_inference,
    run_inference_sliding_window,
    apply_nms_to_detections,
    visualize_results,
)


def _resolve_post_nms_threshold(config, model, cli_value):
    if cli_value is not None:
        return cli_value
    if hasattr(model, "final_nms_iou_threshold"):
        return float(model.final_nms_iou_threshold)
    model_cfg = getattr(config, "model", None)
    if model_cfg is not None:
        return float(model_cfg.final_nms_iou_threshold)
    return 0.5


def _resolve_window_margin_pixels(config, cli_value):
    if cli_value is not None:
        return float(cli_value)
    production = getattr(config, "production", None)
    if production is not None and getattr(production, "ignore_margin_pixels", None) is not None:
        return float(production.ignore_margin_pixels)
    dataset = getattr(config, "dataset", None)
    overlap = getattr(dataset, "overlap", None) if dataset is not None else None
    if overlap is not None:
        return float(overlap) / 2.0
    return None


def _zoomed_image(image: Image.Image, zoom: float) -> Image.Image:
    if zoom == 1.0:
        return image
    width, height = image.size
    return image.resize(
        (max(1, int(round(width * zoom))), max(1, int(round(height * zoom)))),
        Image.Resampling.LANCZOS,
    )


def _scale_detections(detections, scale: float):
    if scale == 1.0:
        return detections
    scaled = []
    for det in detections:
        rbox = det["rbox"]
        scaled.append(
            {
                **det,
                "rbox": RBox(
                    rbox.cx * scale,
                    rbox.cy * scale,
                    rbox.width * scale,
                    rbox.height * scale,
                    rbox.angle,
                ),
            }
        )
    return scaled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run oriented-det inference on image(s) with config and checkpoint (image_demo-style)."
    )
    parser.add_argument("img", type=Path, help="Image file or directory of images")
    parser.add_argument(
        "config_or_checkpoint",
        help=(
            "Path to config JSON when CHECKPOINT is also provided, otherwise checkpoint "
            "(.pth or hf://<slug>) whose pretrained sidecar config will be used"
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Optional checkpoint (.pth or hf://<slug>) when CONFIG is provided",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Path to output file (single image only; default: <img_stem>_detections.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory when img is a directory (default: <img>/out)",
    )
    parser.add_argument(
        "--json-per-image",
        action="store_true",
        help=(
            "Write rich per-image JSON next to each visualization (<output_stem>.json): "
            "rbox, polygon corners, and full run metadata."
        ),
    )
    parser.add_argument(
        "--json-batch",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Write compact batch JSON for downstream tools (e.g. filter_predictions_by_gt). "
            "One image: PATH is the output file. Directory: PATH is the output directory "
            "(default: same as --out-dir), with one JSON per image plus combined detections.json."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for inference (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Score threshold for detections (default: production/evaluation from config)",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=None,
        help="IoU threshold for merge NMS (default: production/model final_nms from config)",
    )
    parser.add_argument(
        "--overlap-pixels",
        type=int,
        default=None,
        help="Window overlap in pixels per axis (default: production.overlap_pixels from config, else 200)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="If set, overlap as fraction of tile size [0,1); overrides --overlap-pixels",
    )
    parser.add_argument(
        "--ignore-margin-pixels",
        type=float,
        default=None,
        help=(
            "Drop sliding-window detections whose centroid falls in this interior margin "
            "(default: production.ignore_margin_pixels, else dataset overlap/2)."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="Keep only these class names (e.g. ship plane); labels are 1-based foreground ids from config",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help=(
            "Resize the image by this factor for inference, then map detections back to the "
            "original image for output (e.g. 2 or 4; default: 1)."
        ),
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=None,
        help=(
            "Sliding-window micro-batch (windows per forward). Skips auto GPU probing when set "
            "(e.g. 8). Default: ORIENTED_DET_WINDOW_BATCH_SIZE env or auto on CUDA/MPS."
        ),
    )
    args = parser.parse_args()
    if args.zoom <= 0:
        parser.error("--zoom must be > 0")
    return args


def _filter_detections_by_classes(detections, class_names, allowed_classes):
    if not allowed_classes:
        return detections
    allowed_labels = set()
    if class_names:
        name_to_label = {name: i + 1 for i, name in enumerate(class_names)}
        for name in allowed_classes:
            if name in name_to_label:
                allowed_labels.add(name_to_label[name])
            elif name.isdigit():
                allowed_labels.add(int(name))
    else:
        allowed_labels = {int(name) for name in allowed_classes if name.isdigit()}
    if not allowed_labels:
        print(f"  Warning: no matching labels for classes={allowed_classes!r}; keeping no detections")
        return []
    return [d for d in detections if int(d["label"]) in allowed_labels]


def _class_name_for_label(class_names, label):
    if class_names and 1 <= int(label) <= len(class_names):
        return class_names[int(label) - 1]
    return None


def _detection_to_json(det, class_names):
    rbox = det["rbox"]
    polygon = [[float(x), float(y)] for x, y in rbox.to_polygon().points]
    label = int(det["label"])
    return {
        "label": label,
        "class_name": _class_name_for_label(class_names, label),
        "score": float(det["score"]),
        "rbox": {
            "cx": float(rbox.cx),
            "cy": float(rbox.cy),
            "width": float(rbox.width),
            "height": float(rbox.height),
            "angle": float(rbox.angle),
        },
        "polygon": polygon,
    }


def save_detections_json(
    json_path,
    *,
    img_path,
    out_path,
    detections,
    class_names,
    original_size,
    inference_size,
    model_canvas,
    score_thr,
    nms_thr,
    overlap_pixels,
    overlap_ratio,
    window_margin_pixels,
    zoom,
    classes_filter,
):
    payload = {
        "source_image": str(img_path),
        "output_image": str(out_path),
        "image_size": {"width": int(original_size[0]), "height": int(original_size[1])},
        "inference_size": {"width": int(inference_size[0]), "height": int(inference_size[1])},
        "model_canvas": {"width": int(model_canvas[0]), "height": int(model_canvas[1])},
        "zoom": float(zoom),
        "score_threshold": float(score_thr),
        "nms_threshold": float(nms_thr),
        "overlap_pixels": int(overlap_pixels) if overlap_pixels is not None else None,
        "overlap_ratio": float(overlap_ratio) if overlap_ratio is not None else None,
        "ignore_margin_pixels": (
            float(window_margin_pixels) if window_margin_pixels is not None else None
        ),
        "classes_filter": list(classes_filter) if classes_filter else None,
        "class_names": list(class_names) if class_names else None,
        "num_detections": len(detections),
        "detections": [_detection_to_json(det, class_names) for det in detections],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved detections JSON to {json_path}")


def _rbox_to_list(rbox: RBox) -> List[float]:
    return [float(rbox.cx), float(rbox.cy), float(rbox.width), float(rbox.height), float(rbox.angle)]


def _detections_to_batch_result(
    img_path: Path,
    image_width: int,
    image_height: int,
    detections: Sequence[dict],
    class_names: Optional[Sequence[str]],
    *,
    score_threshold: float,
    nms_threshold: float,
    preprocessing: dict,
) -> Dict[str, Any]:
    """Build one image entry for combined detections.json (save_predictions-compatible)."""
    predictions = []
    for det in detections:
        label = int(det["label"])
        predictions.append(
            {
                "bbox": _rbox_to_list(det["rbox"]),
                "score": float(det["score"]),
                "label": label,
                "class_name": _class_name_for_label(class_names, label) or f"class_{label}",
            }
        )
    return {
        "image_name": img_path.name,
        "image_path": str(img_path.resolve()),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "resize_mode": preprocessing.get("resize_mode", "fixed"),
        "target_size": preprocessing.get("target_size", [1024, 1024]),
        "num_pred": len(predictions),
        "score_threshold": float(score_threshold),
        "nms_threshold": float(nms_threshold),
        "predictions": predictions,
    }


def _resolve_batch_json_output_path(
    *,
    json_arg: Optional[str],
    img_path: Path,
    out_path: Optional[Path],
    json_dir: Optional[Path],
    multi_image: bool,
) -> Optional[Path]:
    if json_arg is None:
        return None
    if json_dir is not None:
        return json_dir / f"{img_path.stem}.json"
    if json_arg:
        json_path = Path(json_arg)
        if multi_image:
            json_path.mkdir(parents=True, exist_ok=True)
            return json_path / f"{img_path.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        return json_path
    if out_path is not None:
        return out_path.with_suffix(".json")
    return img_path.parent / f"{img_path.stem}_detections.json"


def collect_images(path: Path):
    """Return list of image paths: single file or all images in directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")) + sorted(path.glob("*.png"))
        return [p for p in images if not p.stem.endswith("_detections")]
    return []


def resolve_config_and_checkpoint(args):
    """Resolve optional config + checkpoint positional arguments."""
    from oriented_det.pretrained import ensure_checkpoint, resolve_checkpoint_sidecar_config

    if args.checkpoint is None:
        checkpoint_ref = args.config_or_checkpoint
        checkpoint_path = ensure_checkpoint(checkpoint_ref)
        config_path = resolve_checkpoint_sidecar_config(checkpoint_path)
        if config_path is None:
            print(
                "Config argument was omitted, but no pretrained sidecar config was found "
                f"for checkpoint: {checkpoint_ref}"
            )
            print("Pass an explicit config JSON: image_demo.py IMG CONFIG CHECKPOINT")
            sys.exit(1)
        return config_path, checkpoint_path

    config_path = Path(args.config_or_checkpoint)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    checkpoint_path = ensure_checkpoint(args.checkpoint)
    return resolve_inference_config_path(checkpoint_path, config_path), checkpoint_path


def main():
    args = parse_args()

    config_path, checkpoint_path = resolve_config_and_checkpoint(args)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    images = collect_images(args.img)
    if not images:
        print(f"No images found at {args.img}")
        sys.exit(1)

    # Load model from config + checkpoint (model type, num_classes, preprocessing, class_names from config)
    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model, config, class_names = load_model_from_checkpoint(
        str(checkpoint_path), str(config_path), device=args.device
    )
    preprocessing = get_preprocessing_params(config)
    slice_h, slice_w = get_model_size(preprocessing)
    score_thr = (
        args.score_thr
        if args.score_thr is not None
        else resolve_inference_score_threshold(config)
    )
    nms_thr = _resolve_post_nms_threshold(config, model, args.nms_thr)
    overlap_pixels = (
        args.overlap_pixels
        if args.overlap_pixels is not None
        else resolve_inference_sliding_window_overlap_pixels(config)
    )
    window_margin_pixels = _resolve_window_margin_pixels(config, args.ignore_margin_pixels)
    print(
        f"Preprocessing: resize_mode={preprocessing['resize_mode']}, "
        f"target_size={preprocessing['target_size']} (model canvas {slice_w}×{slice_h})"
    )
    print(
        f"Inference thresholds: score>={score_thr}, merge NMS IoU<={nms_thr}, "
        f"overlap_pixels={overlap_pixels}, ignore_margin_pixels={window_margin_pixels!r}"
    )
    if args.zoom != 1.0:
        print(f"Zoomed inference: {args.zoom:g}× (detections will be mapped back to original size)")

    out_dir = None
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = args.out_dir
    elif len(images) > 1 and args.out_file is None:
        # Default out dir when multiple images and no --out-file
        out_dir = args.img / "out" if args.img.is_dir() else args.img.parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

    json_dir: Optional[Path] = None
    if args.json_batch is not None:
        if args.json_batch == "" and out_dir is not None:
            json_dir = out_dir
        elif args.json_batch and len(images) > 1:
            json_dir = Path(args.json_batch)
            json_dir.mkdir(parents=True, exist_ok=True)

    all_batch_results: List[Dict[str, Any]] = []
    combined_json_path: Optional[Path] = None
    if args.json_batch is not None and len(images) > 1:
        if args.json_batch and args.json_batch != "":
            combined_json_path = Path(args.json_batch) / "detections.json"
        elif out_dir is not None:
            combined_json_path = out_dir / "detections.json"
        elif args.img.is_dir():
            combined_json_path = args.img / "detections.json"
        else:
            combined_json_path = args.img.parent / "detections.json"

    for img_path in images:
        print(f"Inference: {img_path}")
        original_image = Image.open(img_path).convert("RGB")
        ow, oh = original_image.size  # width, height
        inference_image = _zoomed_image(original_image, args.zoom)
        iw, ih = inference_image.size

        if args.zoom != 1.0:
            print(f"  -> zoom {args.zoom:g}×: original {ow}×{oh}, inference {iw}×{ih}")

        if iw == slice_w and ih == slice_h:
            print(f"  -> single forward (image {iw}×{ih} matches model canvas)")
            tensor = preprocess_crop(inference_image, preprocessing, slice_h, slice_w)
            detections = run_inference(
                model, tensor, args.device, score_threshold=score_thr
            )
            detections = apply_nms_to_detections(
                detections, iou_threshold=nms_thr
            )
        else:
            if args.overlap_ratio is not None:
                omsg = f"overlap_ratio={args.overlap_ratio}"
                n_windows = count_sliding_window_positions(
                    ih, iw, preprocessing, overlap_ratio=args.overlap_ratio
                )
            else:
                omsg = f"overlap_pixels={overlap_pixels}"
                n_windows = count_sliding_window_positions(
                    ih, iw, preprocessing, overlap_pixels=overlap_pixels
                )
            print(
                f"  -> pad/tile (image {iw}×{ih} vs canvas {slice_w}×{slice_h}, "
                f"{omsg}, {n_windows} windows)"
            )
            detections = run_inference_sliding_window(
                model,
                inference_image,
                args.device,
                preprocessing,
                score_threshold=score_thr,
                nms_threshold=nms_thr,
                overlap_ratio=args.overlap_ratio,
                overlap_pixels=overlap_pixels,
                slice_h=slice_h,
                slice_w=slice_w,
                window_margin_pixels=window_margin_pixels,
                window_batch_size=args.window_batch_size,
            )
        print(f"  -> {len(detections)} detections (score >= {score_thr}, NMS <= {nms_thr})")
        if args.classes:
            before = len(detections)
            detections = _filter_detections_by_classes(detections, class_names, args.classes)
            print(f"  -> {len(detections)} after class filter {args.classes!r} (removed {before - len(detections)})")
        if args.zoom != 1.0:
            detections = _scale_detections(detections, 1.0 / args.zoom)
            print(f"  -> mapped detections back to original {ow}×{oh} image")

        if out_dir is not None:
            out_path = out_dir / f"{img_path.stem}_detections.png"
        elif args.out_file is not None:
            out_path = args.out_file
        else:
            out_path = img_path.parent / f"{img_path.stem}_detections.png"

        if detections:
            visualize_results(original_image, detections, class_names=class_names, output_path=out_path)
        else:
            if out_path:
                original_image.save(out_path)
                print(f"  Saved image (no detections) to {out_path}")
            else:
                print("  No detections to visualize.")
        if args.json_per_image and out_path:
            save_detections_json(
                out_path.with_suffix(".json"),
                img_path=img_path,
                out_path=out_path,
                detections=detections,
                class_names=class_names,
                original_size=(ow, oh),
                inference_size=(iw, ih),
                model_canvas=(slice_w, slice_h),
                score_thr=score_thr,
                nms_thr=nms_thr,
                overlap_pixels=overlap_pixels,
                overlap_ratio=args.overlap_ratio,
                window_margin_pixels=window_margin_pixels,
                zoom=args.zoom,
                classes_filter=args.classes,
            )

        if args.json_batch is not None:
            batch_result = _detections_to_batch_result(
                img_path,
                ow,
                oh,
                detections,
                class_names,
                score_threshold=score_thr,
                nms_threshold=nms_thr,
                preprocessing=preprocessing,
            )
            all_batch_results.append(batch_result)
            batch_json_path = _resolve_batch_json_output_path(
                json_arg=args.json_batch,
                img_path=img_path,
                out_path=out_path,
                json_dir=json_dir,
                multi_image=len(images) > 1,
            )
            if batch_json_path is not None:
                batch_json_path.parent.mkdir(parents=True, exist_ok=True)
                batch_json_path.write_text(json.dumps(batch_result, indent=2) + "\n", encoding="utf-8")
                print(f"  Saved batch JSON to {batch_json_path}")

    if combined_json_path is not None and all_batch_results:
        combined_json_path.parent.mkdir(parents=True, exist_ok=True)
        combined_payload = {
            "num_images": len(all_batch_results),
            "total_predictions": sum(r["num_pred"] for r in all_batch_results),
            "results": all_batch_results,
        }
        combined_json_path.write_text(json.dumps(combined_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote combined JSON ({len(all_batch_results)} images) to {combined_json_path}")

    print("Done.")


if __name__ == "__main__":
    main()
