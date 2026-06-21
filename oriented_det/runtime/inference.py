"""Example: Running inference with oriented detection models.

This script demonstrates how to:
1. Load a trained model
2. Preprocess images
3. Run inference
4. Apply NMS to filter detections
5. Visualize results

With --verbose / --stats, prints detailed diagnostics to understand where the model
underperforms: raw outputs, score distribution, effect of score threshold and NMS,
per-class counts, and optional ground-truth matching (recall/precision).
"""

import argparse
import os
from pathlib import Path
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torchvision import transforms as T
    from PIL import Image
except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Please install: pip install torch torchvision Pillow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

from oriented_det import OrientedRCNN, RotatedRetinaNet, RBox
from oriented_det.ops import nms
from oriented_det.utils import viz
from oriented_det.data.preprocessing import (
    apply_spatial_preprocess,
    build_spatial_meta_from_dims,
    get_model_canvas_size,
    remap_detections_to_original,
)
from oriented_det.train.config import TrainingExperimentConfig, get_preprocessing_params
from oriented_det.train.utils import scores_labels_pass_threshold

# ImageNet normalization (fallback; prefer config.preprocessing to match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB
MMDET_MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
MMDET_STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]


def load_model(checkpoint_path: Path, model_type: str = "oriented_rcnn", num_classes: int = 15, device: str = "cpu"):
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: "oriented_rcnn" or "rotated_retinanet"
        num_classes: Number of detection classes
        device: Device to load model on
    """
    device = torch.device(device)
    
    # Create model
    if model_type == "oriented_rcnn":
        model = OrientedRCNN(
            num_classes=num_classes,
            backbone_name="resnet50",
            pretrained_backbone=False,
        )
    elif model_type == "rotated_retinanet":
        model = RotatedRetinaNet(
            num_classes=num_classes,
            backbone_name="resnet50",
            pretrained_backbone=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
    
    model.to(device)
    model.eval()
    return model


def get_model_size(preprocessing: dict = None):
    """Return (slice_h, slice_w) from preprocessing config for sliding-window slice size."""
    if preprocessing is None:
        return (1024, 1024)
    mode = preprocessing.get("resize_mode", "fixed")
    ts = preprocessing.get("target_size", (1024, 1024))
    return get_model_canvas_size(mode, ts)


def _sliding_window_grid(
    image_height: int,
    image_width: int,
    slice_h: int,
    slice_w: int,
    overlap_ratio: Optional[float] = None,
    overlap_pixels: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Generate (x0, y0) top-left positions for sliding windows.

    Stride along each axis is ``size - overlap``. Overlap is either:
    - **Ratio mode:** ``overlap_ratio in [0, 1)`` → overlap_h = ratio * slice_h, etc.
    - **Pixel mode:** ``overlap_pixels`` (used for both axes, clamped to ``[0, min(slice)-1]``).
      If ``overlap_ratio is None`` and ``overlap_pixels is None``, **200** px is used (DOTA tile default).
    If ``overlap_ratio is not None``, it takes precedence over ``overlap_pixels``.
    """
    if image_height <= slice_h and image_width <= slice_w:
        return [(0, 0)]
    if overlap_ratio is not None:
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1) for sliding windows")
        overlap_h = float(overlap_ratio) * slice_h
        overlap_w = float(overlap_ratio) * slice_w
    else:
        op = 200 if overlap_pixels is None else int(overlap_pixels)
        if op < 0:
            raise ValueError("overlap_pixels must be >= 0")
        max_oh = max(0, slice_h - 1)
        max_ow = max(0, slice_w - 1)
        oh = min(op, max_oh)
        ow = min(op, max_ow)
        overlap_h, overlap_w = float(oh), float(ow)
    stride_y = max(1, int(round(slice_h - overlap_h)))
    stride_x = max(1, int(round(slice_w - overlap_w)))
    positions = []
    y0 = 0
    while y0 < image_height:
        x0 = 0
        while x0 < image_width:
            positions.append((x0, y0))
            if image_width <= slice_w:
                break
            x0 += stride_x
            if x0 >= image_width:
                break
        if image_height <= slice_h:
            break
        y0 += stride_y
        if y0 >= image_height:
            break
    return positions


def resolve_sliding_window_margin_pixels(
    window_margin_pixels: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    overlap_pixels: Optional[int] = None,
    slice_h: int = 1024,
    slice_w: int = 1024,
) -> Tuple[float, float]:
    """Per-axis interior margin for sliding-window merges (default: half of tile overlap).

    When ``window_margin_pixels`` is set, the same margin applies on both axes.
    Otherwise derives from ``overlap_ratio`` or ``overlap_pixels`` (same convention as
    :func:`_sliding_window_grid`). Pass ``0`` or set margin to ``0`` to disable.
    """
    if window_margin_pixels is not None:
        m = max(0.0, float(window_margin_pixels))
        return m, m
    if overlap_ratio is not None:
        oh = float(overlap_ratio) * slice_h
        ow = float(overlap_ratio) * slice_w
        return max(0.0, oh / 2.0), max(0.0, ow / 2.0)
    op = 200 if overlap_pixels is None else int(overlap_pixels)
    if op < 0:
        raise ValueError("overlap_pixels must be >= 0")
    max_oh = max(0, slice_h - 1)
    max_ow = max(0, slice_w - 1)
    oh = min(op, max_oh)
    ow = min(op, max_ow)
    return oh / 2.0, ow / 2.0


def _centroid_in_sliding_window_interior(
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
    margin_x: float,
    margin_y: float,
    src_x0: int,
    src_y0: int,
    image_width: int,
    image_height: int,
) -> bool:
    """True when centroid lies in the window interior, excluding overlap bands at interior edges.

    On sides that touch the full-image border, the margin band is not applied so objects
    near the scene edge are kept.
    """
    if margin_x <= 0 and margin_y <= 0:
        return True
    touches_left = src_x0 <= 0
    touches_top = src_y0 <= 0
    touches_right = src_x0 + crop_w >= image_width
    touches_bottom = src_y0 + crop_h >= image_height
    lo_x = 0.0 if touches_left else float(margin_x)
    hi_x = float(crop_w) if touches_right else float(crop_w - margin_x)
    lo_y = 0.0 if touches_top else float(margin_y)
    hi_y = float(crop_h) if touches_bottom else float(crop_h - margin_y)
    if hi_x <= lo_x or hi_y <= lo_y:
        return False
    return lo_x <= cx <= hi_x and lo_y <= cy <= hi_y


def count_sliding_window_positions(
    image_height: int,
    image_width: int,
    preprocessing: Optional[dict] = None,
    overlap_ratio: Optional[float] = None,
    overlap_pixels: Optional[int] = 200,
) -> int:
    """Number of 1024 (or model-size) windows ``run_inference_sliding_window`` runs for an image.

    For images that fit in one model canvas, returns 1. Use to reason about total work when
    large images are evaluated via a sliding window (overlap and tile size set how many).
    """
    slice_h, slice_w = get_model_size(preprocessing)
    if image_height <= slice_h and image_width <= slice_w:
        return 1
    return len(
        _sliding_window_grid(
            image_height, image_width, slice_h, slice_w,
            overlap_ratio=overlap_ratio, overlap_pixels=overlap_pixels,
        )
    )


# Cached result of OOM-sweep for sliding-window micro-batch size (per model + canvas + device).
_WINDOW_BATCH_SIZE_CACHE: Dict[Tuple, int] = {}


def _probe_max_window_batch_size(
    model: torch.nn.Module,
    device: str,
    preprocessing: dict,
    slice_h: int,
    slice_w: int,
) -> int:
    """Binary search for the largest micro-batch the model can run in one forward (same as inference)."""
    dev = torch.device(device)
    if dev.type == "cpu":
        return 4
    if dev.type not in ("cuda", "mps"):
        return 8
    high = 64 if dev.type == "cuda" else 32

    canvas = Image.new("RGB", (slice_w, slice_h), 0)
    t0 = preprocess_crop(canvas, preprocessing, slice_h, slice_w)
    param = next(model.parameters(), None)
    p_dtype = param.dtype if param is not None else torch.float32
    t0 = t0.to(device=dev, dtype=p_dtype)

    def _synchronize() -> None:
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        if dev.type == "mps" and hasattr(torch, "mps"):
            try:
                if torch.mps.is_available():
                    torch.mps.synchronize()
            except Exception:
                pass

    def _oom(e: Exception) -> bool:
        m = str(e).lower()
        return (
            "out of memory" in m
            or "mps backend out of memory" in m
            or m.endswith(" oom")
            or ("cuda error" in m and "out of memory" in m)
        )

    def try_bs(n: int) -> bool:
        if n < 1:
            return True
        try:
            with torch.inference_mode():
                batch = [t0.clone() for _ in range(n)]
                _ = model(batch)
            _synchronize()
            return True
        except RuntimeError as e:
            if not _oom(e):
                raise
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            return False

    if not try_bs(1):
        return 1
    lo, hi, best = 1, high, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_bs(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    # Binary search can spike VRAM; a single-tile forward + cache trim stabilizes the allocator
    # before long inference loops (esp. when the first real batch is smaller than the probe peak).
    try:
        with torch.inference_mode():
            _ = model([t0.clone()])
        _synchronize()
    except RuntimeError:
        pass
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return max(1, best)


def resolve_window_batch_size(
    model: torch.nn.Module,
    device: str,
    preprocessing: dict,
    override: Optional[int] = None,
) -> int:
    """Resolve sliding-window micro-batch: how many windows per model forward.

    * ``override`` (e.g. CLI ``--window-batch-size``) if a positive int wins.
    * **Env** ``ORIENTED_DET_WINDOW_BATCH_SIZE``: a **positive int** = fixed; ``auto`` / **unset** =
      binary-search on **CUDA** / **MPS** (cached; one probe per process); **CPU** uses 4.
    * Old default 8: set ``ORIENTED_DET_WINDOW_BATCH_SIZE=8`` explicitly if needed.
    """
    if override is not None and int(override) > 0:
        return int(override)
    raw = (os.environ.get("ORIENTED_DET_WINDOW_BATCH_SIZE") or "").strip()
    t = raw.lower()
    if raw and t not in ("auto", "probe", "0"):
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    slice_h, slice_w = get_model_size(preprocessing)
    key = (
        id(model),
        str(device),
        int(slice_h),
        int(slice_w),
        preprocessing.get("resize_mode") if isinstance(preprocessing, dict) else None,
        str(preprocessing.get("target_size")) if isinstance(preprocessing, dict) else "",
    )
    if key in _WINDOW_BATCH_SIZE_CACHE:
        return _WINDOW_BATCH_SIZE_CACHE[key]
    if torch.device(device).type in ("cuda", "mps"):
        print(
            "[inference] Probing max ORIENTED_DET window batch (GPU; binary search, can take a while)…",
            flush=True,
        )
    n = _probe_max_window_batch_size(model, device, preprocessing, slice_h, slice_w)
    _WINDOW_BATCH_SIZE_CACHE[key] = n
    if torch.device(device).type in ("cuda", "mps") and n >= 1:
        print(
            f"[inference] ORIENTED_DET window batch (auto) = {n} "
            f"(override: env ORIENTED_DET_WINDOW_BATCH_SIZE, CLI --window-batch-size; use e.g. 8 to fix a value)",
            flush=True,
        )
    return n


def _crop_or_pad(image: Image.Image, x0: int, y0: int, slice_w: int, slice_h: int, image_width: int, image_height: int):
    """Extract a slice_w x slice_h region; pad with zeros if the window extends beyond the image."""
    if np is None:
        raise RuntimeError("numpy required for sliding-window inference")
    # Clamp to image bounds for the source region
    x1 = min(x0 + slice_w, image_width)
    y1 = min(y0 + slice_h, image_height)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    crop_w = x1 - src_x0
    crop_h = y1 - src_y0
    crop = image.crop((src_x0, src_y0, x1, y1)) if crop_w > 0 and crop_h > 0 else None
    # Build slice_w x slice_h canvas (zeros)
    canvas = np.zeros((slice_h, slice_w, 3), dtype=np.uint8)
    if crop_w > 0 and crop_h > 0 and crop is not None:
        crop_np = np.array(crop)
        # Place crop into canvas at offset (src_x0 - x0, src_y0 - y0)
        dx = src_x0 - x0
        dy = src_y0 - y0
        canvas[dy : dy + crop_h, dx : dx + crop_w, :] = crop_np
    return Image.fromarray(canvas, "RGB")


def preprocess_crop(crop_pil: Image.Image, preprocessing: dict, slice_h: int, slice_w: int):
    """Preprocess a slice_w x slice_h crop (PIL) to tensor with same normalization as training."""
    mean = preprocessing.get("normalize_mean", MMDET_MEAN)
    std = preprocessing.get("normalize_std", MMDET_STD)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return transform(crop_pil)


def _preprocess_image_tensor_training_style(
    pil_image: Image.Image,
    preprocessing: dict = None,
    *,
    random_crop: bool = False,
):
    """Spatial preprocess + ToTensor + Normalize, matching training dataloader."""
    if preprocessing is None:
        preprocessing = {
            "resize_mode": "fixed",
            "target_size": (1024, 1024),
            "normalize_mean": MMDET_MEAN,
            "normalize_std": MMDET_STD,
        }
    mode = preprocessing.get("resize_mode", "fixed")
    ts = preprocessing.get("target_size", (1024, 1024))
    mean = preprocessing.get("normalize_mean", MMDET_MEAN)
    std = preprocessing.get("normalize_std", MMDET_STD)
    spatial = apply_spatial_preprocess(
        pil_image, [], mode, ts, random_crop=random_crop and mode == "crop"
    )
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return transform(spatial.image)


def preprocess_image(
    image_path: Path,
    target_size: int = 1024,
    normalize: bool = True,
    preprocessing: dict = None,
):
    """Load and preprocess image for inference.
    Uses preprocessing dict from config when provided (same as training); else target_size + ImageNet norm.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    if preprocessing is not None:
        mode = preprocessing.get("resize_mode", "fixed")
        ts = preprocessing.get("target_size", (1024, 1024))
        mean = preprocessing.get("normalize_mean", MMDET_MEAN)
        std = preprocessing.get("normalize_std", MMDET_STD)
        spatial = apply_spatial_preprocess(image, [], mode, ts, random_crop=False)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        tensor = transform(spatial.image)
        return tensor, image, original_size
    else:
        transforms = [T.Resize(target_size), T.ToTensor()]
        if normalize:
            transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transform = T.Compose(transforms)
    tensor = transform(image)
    return tensor, image, original_size


def run_inference(
    model,
    image_tensor,
    device,
    score_threshold: float = 0.5,
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    class_names: Optional[Sequence[str]] = None,
):
    """Run inference on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        score_threshold: Minimum confidence score (default for classes not in per_class map)
        per_class_score_threshold: Optional class_name -> min score (post-NMS)
        class_names: Foreground class names in order (label i corresponds to class_names[i-1])
    
    Returns:
        List of detections with rboxes, scores, and labels
    """
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    
    detections = []
    output = outputs[0]
    
    # Extract detections
    rboxes = output.get("rboxes", [])
    scores = output.get("scores", torch.tensor([]))
    labels = output.get("labels", torch.tensor([], dtype=torch.int64))
    
    id_to_class = {i + 1: n for i, n in enumerate(class_names)} if class_names else {}
    # Filter by score threshold
    if len(scores) > 0:
        if per_class_score_threshold and id_to_class:
            mask = scores_labels_pass_threshold(
                scores, labels, score_threshold, per_class_score_threshold, id_to_class
            )
        else:
            mask = scores >= score_threshold
        rboxes = [rbox for rbox, keep in zip(rboxes, mask) if keep]
        scores = scores[mask].cpu().tolist()
        labels = labels[mask].cpu().tolist()
        
        detections = [
            {"rbox": rbox, "score": score, "label": label}
            for rbox, score, label in zip(rboxes, scores, labels)
        ]
    
    return detections


def _detections_from_raw_output(
    raw_output,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    class_names: Optional[Sequence[str]] = None,
):
    """Convert one model output dict to filtered detections list."""
    detections = []
    output = raw_output

    rboxes = output.get("rboxes", [])
    scores = output.get("scores", torch.tensor([]))
    labels = output.get("labels", torch.tensor([], dtype=torch.int64))

    id_to_class = {i + 1: n for i, n in enumerate(class_names)} if class_names else {}
    if len(scores) > 0:
        if per_class_score_threshold and id_to_class:
            mask = scores_labels_pass_threshold(
                scores, labels, score_threshold, per_class_score_threshold, id_to_class
            )
        else:
            mask = scores >= score_threshold
        rboxes = [rbox for rbox, keep in zip(rboxes, mask) if keep]
        scores = scores[mask].cpu().tolist()
        labels = labels[mask].cpu().tolist()
        detections = [
            {"rbox": rbox, "score": score, "label": label}
            for rbox, score, label in zip(rboxes, scores, labels)
        ]
    return detections


def run_inference_sliding_window(
    model,
    image,
    device,
    preprocessing: dict,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    overlap_ratio: Optional[float] = None,
    overlap_pixels: Optional[int] = 200,
    slice_h: int = None,
    slice_w: int = None,
    window_batch_size: Optional[int] = None,
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    class_names: Optional[Sequence[str]] = None,
    window_margin_pixels: Optional[float] = None,
):
    """Run inference via sliding windows (or one padded canvas if the image fits in one window).

    Pads partial windows at image edges with zeros; images smaller than ``(slice_h, slice_w)``
    use a single padded canvas (no resize stretch). Returns detections in original image coords.

    Overlap: ``overlap_ratio`` in ``[0,1)`` if set; else ``overlap_pixels`` per axis (default 200),
    same convention as :func:`_sliding_window_grid`.

    Before merging windows, detections whose centroid falls in the overlap band of a window
    are dropped on interior sides (margin defaults to half of overlap per axis; see
    :func:`resolve_sliding_window_margin_pixels`). Sides that touch the full-image border
    keep the margin band so edge objects are not removed.

    ``window_batch_size``: if a positive int, use it; if ``None``, use
    :func:`resolve_window_batch_size` (env / auto on GPU).

    ``image``: PIL Image (RGB) or numpy array (H, W, 3).
    """
    if np is None:
        raise RuntimeError("numpy required for sliding-window inference")
    # NumPy arrays also have .size (scalar = numel); use explicit type checks
    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    else:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("image must be HWC")
        image_height, image_width = arr.shape[:2]
        image = Image.fromarray(arr)

    if slice_h is None or slice_w is None:
        slice_h, slice_w = get_model_size(preprocessing)

    bs = resolve_window_batch_size(model, device, preprocessing, override=window_batch_size)

    positions = _sliding_window_grid(
        image_height, image_width, slice_h, slice_w,
        overlap_ratio=overlap_ratio, overlap_pixels=overlap_pixels,
    )
    margin_x, margin_y = resolve_sliding_window_margin_pixels(
        window_margin_pixels=window_margin_pixels,
        overlap_ratio=overlap_ratio,
        overlap_pixels=overlap_pixels,
        slice_h=slice_h,
        slice_w=slice_w,
    )
    all_detections = []
    eps = 1e-3

    def _flush_batch(batch_tensors, batch_meta):
        if not batch_tensors:
            return
        with torch.no_grad():
            outputs = model([t.to(device) for t in batch_tensors])
        for out, meta in zip(outputs, batch_meta):
            x0, y0, crop_w, crop_h, src_x0, src_y0 = meta
            window_detections = _detections_from_raw_output(
                out,
                score_threshold=score_threshold,
                per_class_score_threshold=per_class_score_threshold,
                class_names=class_names,
            )
            for d in window_detections:
                r = d["rbox"]
                if r.cx < -eps or r.cy < -eps or r.cx >= crop_w - eps or r.cy >= crop_h - eps:
                    continue
                if not _centroid_in_sliding_window_interior(
                    float(r.cx),
                    float(r.cy),
                    crop_w,
                    crop_h,
                    margin_x,
                    margin_y,
                    src_x0,
                    src_y0,
                    image_width,
                    image_height,
                ):
                    continue
                shifted_rbox = RBox(r.cx + x0, r.cy + y0, r.width, r.height, r.angle)
                all_detections.append(
                    {"rbox": shifted_rbox, "score": d["score"], "label": d["label"]}
                )

    batch_tensors = []
    batch_meta = []

    for x0, y0 in positions:
        # Valid image content in the slice canvas occupies [0, crop_w) x [0, crop_h); the rest is zero padding.
        # Mapping (cx, cy) -> full image uses (x0 + cx, y0 + cy) only for pixels that came from the image;
        # detections centered in padding would map to bogus coordinates and hurt mAP.
        x1 = min(x0 + slice_w, image_width)
        y1 = min(y0 + slice_h, image_height)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        crop_w = x1 - src_x0
        crop_h = y1 - src_y0
        if crop_w <= 0 or crop_h <= 0:
            continue
        crop_pil = _crop_or_pad(image, x0, y0, slice_w, slice_h, image_width, image_height)
        tensor = preprocess_crop(crop_pil, preprocessing, slice_h, slice_w)
        batch_tensors.append(tensor)
        batch_meta.append((x0, y0, crop_w, crop_h, src_x0, src_y0))

        if len(batch_tensors) >= bs:
            _flush_batch(batch_tensors, batch_meta)
            batch_tensors = []
            batch_meta = []

    _flush_batch(batch_tensors, batch_meta)
    return apply_nms_to_detections(all_detections, nms_threshold)


def run_inference_auto(
    image_path: Path = None,
    image=None,
    model=None,
    device: str = "cpu",
    preprocessing: dict = None,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    overlap_ratio: Optional[float] = None,
    overlap_pixels: Optional[int] = 200,
    window_batch_size: Optional[int] = None,
    return_raw_output: bool = False,
    per_class_score_threshold: Optional[Dict[str, float]] = None,
    class_names: Optional[Sequence[str]] = None,
    window_margin_pixels: Optional[float] = None,
):
    """Run inference: single-image training-style resize for small/equal inputs, sliding windows for larger inputs.

    When image height/width are <= model canvas from ``preprocessing`` (e.g. 1024×1024),
    uses the same **Resize + ToTensor + Normalize** path as training / ``preprocess_image``
    (effectively zooming smaller tiles to target size). Only larger images use
    ``run_inference_sliding_window`` (zero-padded windows, ToTensor+normalize per crop).
    Detections are in original image pixel coordinates.

    Provide either ``image_path`` (to load) or ``image`` (PIL or numpy HWC RGB).
    Returns detections as list of dicts: rbox, score, label.
    If ``return_raw_output=True``, returns ``(detections, raw_output_or_none)``.
    Raw output is only available for single-image path (non-sliding).

    ``window_batch_size``: if a positive int, use for sliding micro-batch; if ``None``,
    :func:`resolve_window_batch_size` (env or auto on GPU) applies.
    """
    if image is None and image_path is None:
        raise ValueError("Provide image_path or image")
    if image is None:
        image = Image.open(image_path).convert("RGB")
    elif not isinstance(image, Image.Image):
        if np is None:
            raise RuntimeError("numpy required when image is an array")
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("image array must be HWC RGB")
        image = Image.fromarray(arr)

    slice_h, slice_w = get_model_size(preprocessing)
    image_width, image_height = image.size

    if image_height <= slice_h and image_width <= slice_w:
        tensor = _preprocess_image_tensor_training_style(image, preprocessing)
        raw_output = run_inference_raw(model, tensor, device)
        dets = []
        rboxes = raw_output.get("rboxes", [])
        scores = raw_output.get("scores", torch.tensor([]))
        labels = raw_output.get("labels", torch.tensor([], dtype=torch.int64))
        id_to_class = {i + 1: n for i, n in enumerate(class_names)} if class_names else {}
        if len(scores) > 0:
            if per_class_score_threshold and id_to_class:
                mask = scores_labels_pass_threshold(
                    scores, labels, score_threshold, per_class_score_threshold, id_to_class
                )
            else:
                mask = scores >= score_threshold
            rboxes = [rbox for rbox, keep in zip(rboxes, mask) if keep]
            scores = scores[mask].cpu().tolist()
            labels = labels[mask].cpu().tolist()
            dets = [{"rbox": rbox, "score": score, "label": label} for rbox, score, label in zip(rboxes, scores, labels)]
        mode = (preprocessing or {}).get("resize_mode", "fixed")
        ts = (preprocessing or {}).get("target_size", (slice_h, slice_w))
        meta = build_spatial_meta_from_dims(mode, image_width, image_height, ts)
        remapped = remap_detections_to_original(dets, meta)
        if return_raw_output:
            return remapped, raw_output
        return remapped

    dets = run_inference_sliding_window(
        model, image, device, preprocessing,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        overlap_ratio=overlap_ratio,
        overlap_pixels=overlap_pixels,
        slice_h=slice_h,
        slice_w=slice_w,
        window_batch_size=window_batch_size,
        per_class_score_threshold=per_class_score_threshold,
        class_names=class_names,
        window_margin_pixels=window_margin_pixels,
    )
    if return_raw_output:
        return dets, None
    return dets


def run_inference_raw(model, image_tensor, device):
    """Run inference and return all detections without score filtering (for diagnostics).

    Returns:
        output dict with "rboxes", "scores", "labels" (tensors/lists as from model).
    """
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    return outputs[0]


def _scores_to_list(scores):
    """Convert scores tensor or list to list of floats."""
    if scores is None or (isinstance(scores, torch.Tensor) and scores.numel() == 0):
        return []
    if torch.is_tensor(scores):
        return scores.cpu().tolist()
    return [float(s) for s in scores]


def compute_inference_stats(
    raw_output,
    score_threshold: float,
    nms_threshold: float,
    class_names: list = None,
):
    """Compute detailed stats along the inference pipeline (threshold, NMS).

    Args:
        raw_output: Single-image output dict from model (rboxes, scores, labels).
        score_threshold: Score threshold used for filtering.
        nms_threshold: IoU threshold used for NMS.
        class_names: Optional list of class names for per-class breakdown.

    Returns:
        Dict with keys: num_raw, score_min, score_max, score_mean, score_std,
        score_percentiles, counts_at_thresholds, per_class_raw, num_after_threshold,
        num_before_nms, num_after_nms, num_suppressed_by_nms, per_class_after_nms, etc.
    """
    rboxes = raw_output.get("rboxes", [])
    scores = raw_output.get("scores", torch.tensor([]))
    labels = raw_output.get("labels", torch.tensor([], dtype=torch.int64))

    scores_list = _scores_to_list(scores)
    if isinstance(labels, torch.Tensor):
        labels_list = labels.cpu().tolist() if labels.numel() else []
    else:
        labels_list = list(labels)

    num_raw = len(scores_list)
    stats = {
        "num_raw": num_raw,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
    }

    if num_raw == 0:
        stats["score_min"] = stats["score_max"] = stats["score_mean"] = stats["score_std"] = None
        stats["score_percentiles"] = {}
        stats["counts_at_thresholds"] = {}
        stats["per_class_raw"] = {}
        stats["num_after_threshold"] = 0
        stats["num_before_nms"] = 0
        stats["num_after_nms"] = 0
        stats["num_suppressed_by_nms"] = 0
        stats["per_class_after_nms"] = {}
        return stats

    arr = np.array(scores_list) if np is not None else scores_list
    if np is not None:
        stats["score_min"] = float(np.min(arr))
        stats["score_max"] = float(np.max(arr))
        stats["score_mean"] = float(np.mean(arr))
        stats["score_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
        stats["score_percentiles"] = {
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }
    else:
        stats["score_min"] = min(scores_list)
        stats["score_max"] = max(scores_list)
        stats["score_mean"] = sum(scores_list) / len(scores_list)
        stats["score_std"] = 0.0
        stats["score_percentiles"] = {}

    # Counts at common thresholds
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    stats["counts_at_thresholds"] = {}
    for t in thresholds:
        if np is not None:
            stats["counts_at_thresholds"][t] = int((arr >= t).sum())
        else:
            stats["counts_at_thresholds"][t] = sum(1 for s in scores_list if s >= t)

    # Per-class counts (raw)
    per_class = defaultdict(int)
    for lab in labels_list:
        per_class[lab] += 1
    stats["per_class_raw"] = dict(per_class)

    # After score threshold
    mask = [s >= score_threshold for s in scores_list]
    rboxes_filtered = [r for r, m in zip(rboxes, mask) if m]
    scores_filtered = [s for s, m in zip(scores_list, mask) if m]
    labels_filtered = [l for l, m in zip(labels_list, mask) if m]
    num_after_threshold = len(scores_filtered)
    stats["num_after_threshold"] = num_after_threshold
    stats["num_filtered_by_threshold"] = num_raw - num_after_threshold

    if num_after_threshold == 0:
        stats["num_before_nms"] = 0
        stats["num_after_nms"] = 0
        stats["num_suppressed_by_nms"] = 0
        stats["per_class_after_nms"] = {}
        return stats

    # NMS (class-aware when labels are present)
    keep_indices = nms.oriented_nms(
        rboxes_filtered,
        scores_filtered,
        iou_threshold=nms_threshold,
        labels=labels_filtered if labels_filtered else None,
    )
    num_after_nms = len(keep_indices)
    stats["num_before_nms"] = num_after_threshold
    stats["num_after_nms"] = num_after_nms
    stats["num_suppressed_by_nms"] = num_after_threshold - num_after_nms

    per_class_nms = defaultdict(int)
    for i in keep_indices:
        per_class_nms[labels_filtered[i]] += 1
    stats["per_class_after_nms"] = dict(per_class_nms)

    return stats


def print_inference_stats(stats, class_names: list = None):
    """Print a detailed inference stats report to stdout."""
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE STATS (diagnostics)")
    print("=" * 60)

    print("\n--- Raw model outputs ---")
    print(f"  Total detections (before any filter): {stats['num_raw']}")

    if stats["num_raw"] == 0:
        print("  No detections; score distribution and NMS stats skipped.")
        print("=" * 60)
        return

    print("\n--- Score distribution ---")
    print(f"  Min: {stats['score_min']:.4f}  Max: {stats['score_max']:.4f}")
    print(f"  Mean: {stats['score_mean']:.4f}  Std: {stats['score_std']:.4f}")
    if stats.get("score_percentiles"):
        p = stats["score_percentiles"]
        print(f"  Percentiles: p25={p['p25']:.4f}  p50={p['p50']:.4f}  p75={p['p75']:.4f}  p90={p['p90']:.4f}")

    print("\n--- Counts at score thresholds ---")
    for t, count in sorted(stats["counts_at_thresholds"].items()):
        print(f"  score >= {t:.2f}: {count} detections")

    print("\n--- Per-class counts (raw) ---")
    for cid, count in sorted(stats["per_class_raw"].items()):
        i = int(cid)
        name = (
            class_names[i - 1]
            if class_names and 1 <= i <= len(class_names)
            else f"class_{i}"
        )
        print(f"  {name}: {count}")

    print("\n--- Score threshold ---")
    print(f"  Threshold: {stats['score_threshold']:.4f}")
    print(f"  After threshold: {stats['num_after_threshold']} kept, {stats['num_filtered_by_threshold']} removed")

    print("\n--- NMS ---")
    print(f"  IoU threshold: {stats['nms_threshold']:.4f}")
    print(f"  Before NMS: {stats['num_before_nms']}  After NMS: {stats['num_after_nms']}  Suppressed: {stats['num_suppressed_by_nms']}")

    print("\n--- Per-class counts (after threshold + NMS) ---")
    for cid, count in sorted(stats["per_class_after_nms"].items()):
        i = int(cid)
        name = (
            class_names[i - 1]
            if class_names and 1 <= i <= len(class_names)
            else f"class_{i}"
        )
        print(f"  {name}: {count}")

    print("=" * 60)


def load_gt_for_image(image_path: Path, labels_path: Path = None, label_dir: Path = None):
    """Load ground-truth DOTA annotations for an image.

    Tries: labels_path if given; else image_path.with_suffix('.txt');
    else label_dir / (image_path.name with .txt) if label_dir given.

    Returns:
        List of (rbox, class_name) or empty list.
    """
    from oriented_det.data import DOTAAnnotation

    paths_to_try = []
    if labels_path is not None and labels_path.exists():
        paths_to_try.append(labels_path)
    if not paths_to_try:
        p = image_path.with_suffix(".txt")
        if p.exists():
            paths_to_try.append(p)
    if not paths_to_try and label_dir is not None:
        p = label_dir / (image_path.stem + ".txt")
        if p.exists():
            paths_to_try.append(p)
    if not paths_to_try:
        return []

    gt_list = []
    with open(paths_to_try[0], "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ann = DOTAAnnotation.from_line(line)
                gt_list.append((ann.rbox, ann.class_name))
            except Exception:
                continue
    return gt_list


def compute_matching_stats(
    detections,
    gt_list,
    iou_threshold: float = 0.5,
    class_map: dict = None,
):
    """Compute recall/precision vs ground truth (class-aware IoU matching).

    detections: list of dicts with 'rbox', 'score', 'label' (int).
    gt_list: list of (rbox, class_name).
    class_map: optional dict class_name -> class_id (for GT). If None, no class matching.
    Returns dict with total_gt, total_det, matched_gt, matched_det, recall, precision.
    """
    from oriented_det.ops import iou as iou_ops

    total_gt = len(gt_list)
    total_det = len(detections)
    if total_gt == 0 and total_det == 0:
        return {"total_gt": 0, "total_det": 0, "matched_gt": 0, "matched_det": 0, "recall": 0.0, "precision": 0.0}

    # Map GT class_name -> class_id if we have class_map
    gt_with_id = []
    for rbox, cname in gt_list:
        cid = class_map.get(cname, -1) if class_map else -1
        gt_with_id.append((rbox, cname, cid))

    matched_gt = set()
    matched_det = set()
    for i, det in enumerate(detections):
        det_rbox = det["rbox"]
        det_cid = det["label"]
        for j, (gt_rbox, gt_cname, gt_cid) in enumerate(gt_with_id):
            if class_map is not None and gt_cid >= 0 and det_cid != gt_cid:
                continue
            iou = iou_ops.rbox_iou(det_rbox, gt_rbox)
            if iou >= iou_threshold:
                matched_gt.add(j)
                matched_det.add(i)
                break

    recall = len(matched_gt) / total_gt if total_gt else 0.0
    precision = len(matched_det) / total_det if total_det else 0.0
    return {
        "total_gt": total_gt,
        "total_det": total_det,
        "matched_gt": len(matched_gt),
        "matched_det": len(matched_det),
        "recall": recall,
        "precision": precision,
    }


def apply_nms_to_detections(detections, iou_threshold: float = 0.5):
    """Apply NMS to filter overlapping detections (class-aware when labels present).
    
    When detections include a "label" key, NMS is class-aware: boxes of different
    classes do not suppress each other (e.g. a boat inside a dock both remain).
    
    Args:
        detections: List of detection dicts with 'rbox', 'score', and optionally 'label'
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    rboxes = [d["rbox"] for d in detections]
    scores = [d["score"] for d in detections]
    labels = None
    if all("label" in d for d in detections):
        labels = [d["label"] for d in detections]
    
    keep_indices = nms.oriented_nms(
        rboxes,
        scores,
        iou_threshold=iou_threshold,
        labels=labels,
    )
    
    return [detections[i] for i in keep_indices]


def visualize_results(image, detections, class_names: list = None, output_path: Path = None):
    """Visualize detection results on image.
    
    Args:
        image: PIL Image
        detections: List of detection dicts
        class_names: Optional list of class names
        output_path: Optional path to save visualization
    """
    if not detections:
        print("No detections to visualize")
        return image
    
    # Convert rboxes to polygons for visualization
    polygons = [det["rbox"].to_polygon() for det in detections]
    
    # Create labels (model uses 1-based foreground ids; class_names[i] is id i+1, see run_inference docstring)
    labels = []
    if class_names:
        for det in detections:
            lab = int(det["label"])
            cname = (
                class_names[lab - 1]
                if 1 <= lab <= len(class_names)
                else f"class_{lab}"
            )
            label = f"{cname}: {det['score']:.2f}"
            labels.append(label)
    else:
        for det in detections:
            labels.append(f"Class {det['label']}: {det['score']:.2f}")
    
    # Draw boxes
    result = viz.draw_polygons(
        image,
        polygons,
        labels=labels,
    )
    
    if output_path:
        result.save(output_path)
        print(f"Saved visualization to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference with oriented detection model")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--model-type", choices=["oriented_rcnn", "rotated_retinanet"], default="oriented_rcnn",
                       help="Model type to use")
    parser.add_argument("--num-classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--device", default=None,
                       help="Device to run inference on (default: auto-detect: cuda > mps > cpu)")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                       help="Minimum confidence score")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                       help="IoU threshold for NMS")
    parser.add_argument("--output", type=Path, help="Path to save visualization")
    parser.add_argument("--class-names", nargs="+", help="List of class names")
    parser.add_argument("--verbose", "--stats", dest="verbose", action="store_true",
                       help="Print detailed inference stats (raw outputs, score distribution, threshold, NMS, optional GT matching)")
    parser.add_argument("--labels", type=Path, default=None,
                       help="Path to ground-truth DOTA .txt file (for matching stats when --verbose)")
    parser.add_argument("--label-dir", type=Path, default=None,
                       help="Directory containing label files (e.g. labelTxt); used to find image_name.txt when --verbose")
    parser.add_argument("--iou-matching", type=float, default=0.5,
                       help="IoU threshold for GT matching stats (default: 0.5)")
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to experiment config.json; use same preprocessing as training")
    parser.add_argument(
        "--overlap-pixels",
        type=int,
        default=200,
        help="Window overlap in pixels (per axis) when not using --overlap-ratio (default: 200, like DOTA tile_dota).",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="If set, window overlap as fraction of tile size [0,1); overrides --overlap-pixels.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=None,
        help=(
            "Sliding-window micro-batch (windows per forward). If omitted, uses "
            "ORIENTED_DET_WINDOW_BATCH_SIZE (int) or auto on CUDA/MPS (binary search; cached per run)."
        ),
    )

    args = parser.parse_args()

    if args.device is None:
        from oriented_det.utils import get_device
        args.device = str(get_device())

    preprocessing = None
    if args.config is not None and args.config.exists():
        config = TrainingExperimentConfig.load(args.config)
        preprocessing = get_preprocessing_params(config)
        print(f"Preprocessing from config: resize_mode={preprocessing['resize_mode']}, target_size={preprocessing['target_size']}")
    else:
        preprocessing = {"resize_mode": "fixed", "target_size": (1024, 1024), "normalize_mean": MMDET_MEAN, "normalize_std": MMDET_STD}

    # Load model
    if args.checkpoint:
        model = load_model(args.checkpoint, args.model_type, args.num_classes, args.device)
    else:
        print("Warning: No checkpoint provided, using untrained model")
        if args.model_type == "oriented_rcnn":
            model = OrientedRCNN(num_classes=args.num_classes, backbone_name="resnet50")
        else:
            model = RotatedRetinaNet(num_classes=args.num_classes, backbone_name="resnet50")
        model.to(args.device)
        model.eval()
    
    print(f"Loading image from {args.image}")
    original_image = Image.open(args.image).convert("RGB")
    original_size = original_image.size  # (width, height)
    slice_h, slice_w = get_model_size(preprocessing)
    ow, oh = original_size[0], original_size[1]
    if args.overlap_ratio is not None:
        print(
            f"Using pad/tile windows (image {ow}x{oh}, model canvas {slice_w}x{slice_h}, "
            f"overlap_ratio={args.overlap_ratio})"
        )
    else:
        print(
            f"Using pad/tile windows (image {ow}x{oh}, model canvas {slice_w}x{slice_h}, "
            f"overlap_pixels={args.overlap_pixels})"
        )

    if args.verbose:
        crop_pil = _crop_or_pad(original_image, 0, 0, slice_w, slice_h, ow, oh)
        tensor = preprocess_crop(crop_pil, preprocessing, slice_h, slice_w)
        raw_output = run_inference_raw(model, tensor, args.device)
        pipeline_stats = compute_inference_stats(
            raw_output,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            class_names=args.class_names,
        )
        print_inference_stats(pipeline_stats, class_names=args.class_names)
        print("(Stats above are for the first window (0,0) only; full inference merges all windows.)\n")

    print("Running inference...")
    detections = run_inference_auto(
        image_path=args.image,
        image=original_image,
        model=model,
        device=args.device,
        preprocessing=preprocessing,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        overlap_ratio=args.overlap_ratio,
        overlap_pixels=args.overlap_pixels,
        window_batch_size=args.window_batch_size,
    )
    print(f"Found {len(detections)} detections (pad/tile + NMS)")
    
    # Optional: ground-truth matching stats
    if args.verbose:
        gt_list = load_gt_for_image(args.image, labels_path=args.labels, label_dir=args.label_dir)
        if gt_list:
            class_map = None
            if args.class_names:
                class_map = {name: i for i, name in enumerate(args.class_names)}
            match_stats = compute_matching_stats(
                detections,
                gt_list,
                iou_threshold=args.iou_matching,
                class_map=class_map,
            )
            print("\n--- Ground-truth matching (after threshold + NMS) ---")
            print(f"  IoU threshold: {args.iou_matching}")
            print(f"  GT objects: {match_stats['total_gt']}  Detections: {match_stats['total_det']}")
            print(f"  Matched GT: {match_stats['matched_gt']}  Matched det: {match_stats['matched_det']}")
            print(f"  Recall:    {match_stats['recall']:.2%} (matched GT / total GT)")
            print(f"  Precision: {match_stats['precision']:.2%} (matched det / total det)")
        else:
            print("\n--- Ground-truth matching ---")
            print("  No GT labels found (use --labels or --label-dir to enable matching stats).")
    
    # Visualize
    output_path = args.output or args.image.parent / f"{args.image.stem}_detections.png"
    visualize_results(original_image, detections, args.class_names, output_path)
    
    # Print summary
    if detections:
        print("\nDetection summary:")
        for i, det in enumerate(detections, 1):
            rbox = det["rbox"]
            print(f"  {i}. Class {det['label']}: score={det['score']:.3f}, "
                  f"center=({rbox.cx:.1f}, {rbox.cy:.1f}), "
                  f"size=({rbox.width:.1f}x{rbox.height:.1f}), "
                  f"angle={rbox.angle:.3f} rad")


if __name__ == "__main__":
    main()
