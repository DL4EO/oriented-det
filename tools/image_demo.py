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

Example:
  # Single image
  python tools/image_demo.py demo/demo.jpg configs/oriented_rcnn/dota_le90_1x.json runs/.../checkpoints/best.pth --out-file result.jpg

  # All images in demo/
  python tools/image_demo.py demo configs/.../config.json runs/.../checkpoints/best.pth --out-dir demo/out
"""

import argparse
import sys
from pathlib import Path

# Add project root so "from oriented_det" works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from oriented_det.train.config import get_preprocessing_params
from oriented_det.runtime.checkpoint import load_model_from_checkpoint, resolve_inference_config_path
from oriented_det.runtime.inference import (
    get_model_size,
    preprocess_crop,
    run_inference,
    run_inference_sliding_window,
    apply_nms_to_detections,
    visualize_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run oriented-det inference on image(s) with config and checkpoint (image_demo-style)."
    )
    parser.add_argument("img", type=Path, help="Image file or directory of images")
    parser.add_argument("config", type=Path, help="Path to experiment config (e.g. .json)")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint (.pth) or hf://<slug>")
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
        "--device",
        default="cuda:0",
        help="Device for inference (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--overlap-pixels",
        type=int,
        default=200,
        help="Window overlap in pixels per axis when not using --overlap-ratio (default: 200)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="If set, overlap as fraction of tile size [0,1); overrides --overlap-pixels",
    )
    args = parser.parse_args()
    return args


def collect_images(path: Path):
    """Return list of image paths: single file or all images in directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")) + sorted(path.glob("*.png"))
        return images
    return []


def main():
    args = parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    from oriented_det.pretrained import ensure_checkpoint

    checkpoint_path = ensure_checkpoint(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    config_path = resolve_inference_config_path(checkpoint_path, args.config)

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
    print(
        f"Preprocessing: resize_mode={preprocessing['resize_mode']}, "
        f"target_size={preprocessing['target_size']} (model canvas {slice_w}×{slice_h})"
    )

    out_dir = None
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = args.out_dir
    elif len(images) > 1 and args.out_file is None:
        # Default out dir when multiple images and no --out-file
        out_dir = args.img / "out" if args.img.is_dir() else args.img.parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"Inference: {img_path}")
        original_image = Image.open(img_path).convert("RGB")
        ow, oh = original_image.size  # width, height

        if ow == slice_w and oh == slice_h:
            print(f"  -> single forward (image {ow}×{oh} matches model canvas)")
            tensor = preprocess_crop(original_image, preprocessing, slice_h, slice_w)
            detections = run_inference(
                model, tensor, args.device, score_threshold=args.score_thr
            )
            detections = apply_nms_to_detections(
                detections, iou_threshold=args.nms_thr
            )
        else:
            if args.overlap_ratio is not None:
                omsg = f"overlap_ratio={args.overlap_ratio}"
            else:
                omsg = f"overlap_pixels={args.overlap_pixels}"
            print(f"  -> pad/tile (image {ow}×{oh} vs canvas {slice_w}×{slice_h}, {omsg})")
            detections = run_inference_sliding_window(
                model,
                original_image,
                args.device,
                preprocessing,
                score_threshold=args.score_thr,
                nms_threshold=args.nms_thr,
                overlap_ratio=args.overlap_ratio,
                overlap_pixels=args.overlap_pixels,
                slice_h=slice_h,
                slice_w=slice_w,
            )
        print(f"  -> {len(detections)} detections (score >= {args.score_thr}, NMS <= {args.nms_thr})")

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

    print("Done.")


if __name__ == "__main__":
    main()
