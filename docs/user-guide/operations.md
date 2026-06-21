# Operations Module

The operations module provides efficient implementations of Intersection-over-Union (IoU) and Non-Maximum Suppression (NMS) for rotated bounding boxes.

## Overview

The `oriented_det.ops` module provides:

- **IoU calculations** - Compute overlap between rotated boxes
- **NMS** - Filter overlapping detections
- **GPU acceleration** - Automatic fallback to CPU when needed

## IoU (Intersection over Union)

### Basic Usage

```python
from oriented_det.geometry import RBox
from oriented_det.ops import iou

# Create two boxes
box1 = RBox(0, 0, 10, 10, 0)
box2 = RBox(5, 5, 10, 10, 0)

# Compute IoU
overlap = iou.rbox_iou(box1, box2)
print(f"IoU: {overlap}")  # 0.25
```

### Polygon IoU

For arbitrary polygons:

```python
from oriented_det.geometry import Polygon
from oriented_det.ops import iou

poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])

overlap = iou.polygon_iou(poly1, poly2)
```

### Batch IoU

Compute IoU for multiple boxes at once:

```python
from oriented_det.ops import iou

boxes_a = [RBox(0, 0, 10, 10, 0), RBox(20, 20, 10, 10, 0)]
boxes_b = [RBox(5, 5, 10, 10, 0), RBox(25, 25, 10, 10, 0)]

# Returns matrix of IoU values
iou_matrix = iou.batch_rbox_iou(boxes_a, boxes_b)
```

### Backend Selection

Control which backend to use:

```python
# Auto (default) - uses GPU if available, falls back to CPU
overlap = iou.rbox_iou(box1, box2, backend="auto")

# Force Python implementation
overlap = iou.rbox_iou(box1, box2, backend="python")

# Force PyTorch implementation (requires torchvision)
overlap = iou.rbox_iou(box1, box2, backend="torch")
```

## NMS (Non-Maximum Suppression)

### Basic Usage

```python
from oriented_det.geometry import RBox
from oriented_det.ops import nms

# Create boxes and scores
boxes = [
    RBox(0, 0, 10, 10, 0),
    RBox(5, 5, 10, 10, 0),  # Overlaps with first
    RBox(100, 100, 10, 10, 0),  # No overlap
]
scores = [0.9, 0.8, 0.6]

# Apply NMS
keep = nms.oriented_nms(boxes, scores, iou_threshold=0.5)
print(f"Kept indices: {keep}")  # [0, 2]
```

### Limiting Detections

Limit the number of detections:

```python
keep = nms.oriented_nms(
    boxes,
    scores,
    iou_threshold=0.5,
    max_detections=10
)
```

### Backend Selection

```python
# Auto (default)
keep = nms.oriented_nms(boxes, scores, backend="auto")

# Python implementation
keep = nms.oriented_nms(boxes, scores, backend="python")

# PyTorch implementation (GPU-accelerated)
keep = nms.oriented_nms(boxes, scores, backend="torch")
```

## GPU Acceleration

When PyTorch and TorchVision are installed with CUDA support, operations automatically use GPU:

```python
import torch

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Operations will automatically use GPU if available
# No code changes needed!
```

## GPU Operations

For maximum performance, OrientedDet provides dedicated GPU-accelerated operations that are fully vectorized with no Python loops. These operations work directly with PyTorch tensors and provide significant speedup over CPU implementations.

### Overview

The GPU operations module (`oriented_det.ops.gpu_ops`) provides:
- **`oriented_box_iou_gpu()`** - GPU-accelerated IoU computation using sampling-based approximation
- **`generate_oriented_anchors_gpu()`** - Vectorized anchor generation for FPN levels
- **`match_anchors_to_gt_gpu()`** - GPU-accelerated anchor-to-ground-truth matching
- **`oriented_nms_gpu()`** - GPU-accelerated Non-Maximum Suppression

**Key Features:**
- Fully vectorized operations (no Python loops)
- Automatic memory management with chunking for large inputs
- Sampling-based IoU approximation (configurable precision)
- AABB pre-filtering for efficiency

### GPU IoU Computation

`oriented_box_iou_gpu()` computes IoU between sets of oriented boxes using a sampling-based approximation:

```python
import torch
from oriented_det.ops.gpu_ops import oriented_box_iou_gpu

# Create boxes as tensors: [N, 5] format [cx, cy, w, h, angle]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
boxes1 = torch.tensor([
    [100, 100, 50, 30, 0.0],      # Box 1: center (100, 100), 50x30, no rotation
    [200, 150, 40, 60, 0.5],      # Box 2: center (200, 150), 40x60, rotated
], device=device)

boxes2 = torch.tensor([
    [105, 105, 50, 30, 0.0],      # Overlaps with box1
    [300, 300, 50, 50, 0.0],      # No overlap
], device=device)

# Compute IoU matrix: [N, M]
iou_matrix = oriented_box_iou_gpu(boxes1, boxes2, num_samples=100)
# Returns: [[0.85, 0.0], [0.0, 0.0]]  # Approximate IoU values
```

**Parameters:**
- `boxes1`: `[N, 5]` tensor with format `[cx, cy, w, h, angle]`
- `boxes2`: `[M, 5]` tensor with format `[cx, cy, w, h, angle]`
- `num_samples`: Number of samples per box for approximation (`None`: geometry-based count from box size and aspect ratio — typically **25** (5×5) for 10×10 px boxes up to **1024** for extreme elongated OBBs. Target spacing **2 px** via `ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX`. Disable geometry with `ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE=0` for flat **100**-sample grid, debug only)
  - Higher values = more accurate but slower
  - Typical values: 25 (5×5), 49 (7×7), 100 (10×10)

**Performance Notes:**
- Uses AABB pre-filtering to eliminate ~80-90% of expensive computations
- Automatically chunks large inputs to manage GPU memory
- Sampling approximation is typically accurate to within 1-2% of exact IoU

### GPU Anchor Generation

`generate_oriented_anchors_gpu()` generates oriented anchors for multiple FPN levels:

```python
import torch
from oriented_det.ops.gpu_ops import generate_oriented_anchors_gpu
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FPN configuration
image_size = (800, 800)  # (height, width)
feature_map_sizes = [
    (200, 200),  # P3: stride 4
    (100, 100),  # P4: stride 8
    (50, 50),    # P5: stride 16
    (25, 25),    # P6: stride 32
    (13, 13),    # P7: stride 64
]

anchor_scales = [8]  # Single scale (applied per level)
anchor_ratios = [0.5, 1.0, 2.0]  # Aspect ratios
anchor_angles = [-math.pi/2, 0, math.pi/2]  # Rotation angles
stride_per_level = [4, 8, 16, 32, 64]

# Generate anchors for all FPN levels
anchors_per_level = generate_oriented_anchors_gpu(
    image_size=image_size,
    feature_map_sizes=feature_map_sizes,
    anchor_scales=anchor_scales,
    anchor_ratios=anchor_ratios,
    anchor_angles=anchor_angles,
    stride_per_level=stride_per_level,
    device=device
)

# Each element is [H*W*A, 5] where A = num_ratios * num_angles
print(f"P3 anchors: {anchors_per_level[0].shape}")  # [200*200*9, 5] = [360000, 5]
print(f"P4 anchors: {anchors_per_level[1].shape}")  # [100*100*9, 5] = [90000, 5]
```

**Use Cases:**
- Generating anchors for training RPN or Rotated RetinaNet
- Creating anchor grids for inference
- Batch anchor generation for multiple images

### GPU Anchor Matching

`match_anchors_to_gt_gpu()` matches anchors to ground truth boxes using GPU-accelerated IoU:

```python
import torch
from oriented_det.ops.gpu_ops import match_anchors_to_gt_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Anchors: [N, 5] format [cx, cy, w, h, angle]
anchors = torch.tensor([
    [100, 100, 50, 30, 0.0],
    [200, 200, 40, 40, 0.0],
    [300, 300, 60, 20, 0.5],
], device=device)

# Ground truth boxes: [M, 5]
gt_boxes = torch.tensor([
    [105, 105, 50, 30, 0.0],  # Matches anchor 0
    [250, 250, 50, 50, 0.0],  # No match (low IoU)
], device=device)

# Match anchors to GT
labels, matched_gt_indices = match_anchors_to_gt_gpu(
    anchors=anchors,
    gt_boxes=gt_boxes,
    positive_iou_threshold=0.7,  # IoU >= 0.7 is positive
    negative_iou_threshold=0.3,  # IoU < 0.3 is negative
)

# labels: [N] with -1=ignore, 0=background, 1=foreground
# matched_gt_indices: [N] index of matched GT (-1 if none)
print(labels)  # [1, 0, -1]  # Positive, negative, ignore
print(matched_gt_indices)  # [0, -1, -1]  # Matched to GT 0
```

**Matching Strategy:**
- **Positive**: IoU >= `positive_iou_threshold` OR best match for a GT box
- **Negative**: IoU < `negative_iou_threshold` AND not best match for any GT
- **Ignore**: IoU between thresholds (not used for training)

**Performance:**
- Automatically chunks large anchor sets to manage GPU memory
- Uses GPU-accelerated IoU computation for fast matching

### GPU NMS

`oriented_nms_gpu()` performs Non-Maximum Suppression on GPU:

```python
import torch
from oriented_det.ops.gpu_ops import oriented_nms_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detections: [N, 5] format [cx, cy, w, h, angle]
boxes = torch.tensor([
    [100, 100, 50, 30, 0.0],
    [105, 105, 50, 30, 0.0],  # Overlaps with box 0
    [200, 200, 40, 40, 0.0],  # No overlap
], device=device)

# Confidence scores: [N]
scores = torch.tensor([0.9, 0.8, 0.7], device=device)

# Apply NMS
keep_indices = oriented_nms_gpu(
    boxes=boxes,
    scores=scores,
    iou_threshold=0.5,
    max_detections=100,  # Optional: limit output
)

print(keep_indices)  # [0, 2]  # Kept boxes 0 and 2 (box 1 suppressed)
```

**Parameters:**
- `boxes`: `[N, 5]` tensor with format `[cx, cy, w, h, angle]`
- `scores`: `[N]` tensor with confidence scores
- `iou_threshold`: Suppression threshold (default: 0.5)
- `max_detections`: Maximum detections to keep (optional)

**Performance:**
- Uses sampling-based IoU (25 samples by default for speed)
- Automatically limits input size for efficiency
- Fully vectorized greedy NMS algorithm

### When to Use GPU Operations

**Use GPU operations when:**
- Processing large batches of boxes (>1000 boxes)
- Training models (anchor generation, matching)
- Real-time inference with many detections
- GPU memory is available

**Use CPU operations when:**
- Processing small numbers of boxes (<100 boxes)
- Debugging or development
- GPU memory is limited
- Deterministic results are required (GPU uses approximation)

**Performance Comparison:**
- GPU IoU: ~10-50x faster than CPU for large batches (>1000 boxes)
- GPU NMS: ~5-20x faster than CPU for large detection sets
- GPU anchor generation: ~100x faster than CPU loops

### Device Management

All GPU operations automatically handle device placement:

```python
import torch
from oriented_det.ops.gpu_ops import oriented_box_iou_gpu

# Automatically uses device of input tensors
device = torch.device('cuda:0')
boxes1 = torch.randn(100, 5, device=device)
boxes2 = torch.randn(50, 5, device=device)

iou_matrix = oriented_box_iou_gpu(boxes1, boxes2)  # Runs on GPU

# CPU fallback
device = torch.device('cpu')
boxes1_cpu = torch.randn(100, 5, device=device)
boxes2_cpu = torch.randn(50, 5, device=device)

iou_matrix_cpu = oriented_box_iou_gpu(boxes1_cpu, boxes2_cpu)  # Runs on CPU
```

**Note:** GPU operations will work on CPU if CUDA is unavailable, but performance will be slower than dedicated CPU implementations.

## Performance Tips

1. **Use batch operations** - `batch_rbox_iou` is more efficient than loops
2. **AABB pre-filtering** - Automatically enabled, eliminates ~80-90% of expensive rotated IoU computations
3. **Backend selection** - Use `"auto"` for best performance (automatically uses GPU when available)
4. **Limit detections** - Use `max_detections` to cap output size

## Performance Notes

### NMS Implementation

The current NMS implementation uses several optimizations:

- **AABB Pre-filtering**: Before computing expensive rotated IoU, boxes are filtered using axis-aligned bounding boxes. This eliminates ~80-90% of rotated IoU computations.
- **Batch Processing**: IoU is computed in batches rather than individual comparisons, reducing Python overhead.
- **Pre-computed AABBs**: AABBs are computed once per box, not per comparison.

**Performance Comparison:**
- Naive implementation: ~31.5s for 2000 boxes
- Optimized implementation: ~11.9s for 2000 boxes (~2.6x faster)

### GPU Acceleration

When PyTorch and TorchVision are installed with CUDA support, operations automatically use GPU:

```python
import torch

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Operations will automatically use GPU if available
# No code changes needed!
```

**Note:** `torchvision.ops.nms_rotated` does not exist in current torchvision versions. The framework uses an optimized Python implementation with AABB pre-filtering.

## See Also

- [API Reference](../api/ops.md) - Complete API documentation
- [Geometry Guide](geometry.md) - Working with geometric primitives

