# Operations API Reference

::: oriented_det.ops
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true

## Important Notes

### NMS Performance

**Important:** `torchvision.ops.nms_rotated` **does not exist** in current torchvision versions (tested with 0.22.0+).

**Current Implementation:**
- The framework uses a **Python-based NMS** implementation with optimizations:
  - AABB (axis-aligned bounding box) pre-filtering before expensive rotated IoU computation
  - Batch IoU computation for better performance
  - Class-aware NMS support (boxes of different classes don't suppress each other)
- Performance: ~2.6x faster than naive implementation (11.9s for 2000 boxes vs 31.5s)
- **Future:** GPU-accelerated kernels via custom CUDA implementation planned

**Optimization Details:**
- AABB pre-filtering eliminates ~80-90% of rotated IoU computations
- Pre-computed AABBs avoid redundant calculations
- Batch processing reduces Python overhead

### Backend Selection

For CPU-based IoU, use `iou.rbox_iou()` and `nms.oriented_nms()` with `intersection_backend` (`"auto"`, `"python"`, or `"shapely"`). For GPU-accelerated operations, use `oriented_det.ops.gpu_ops` (e.g., `oriented_box_iou_gpu`, `oriented_nms_gpu`). Models automatically use GPU kernels when available.

## Examples

### Basic IoU Computation

```python
from oriented_det.geometry import RBox
from oriented_det.ops import iou

boxes = [
    RBox(0, 0, 2, 1, 0),
    RBox(0.5, 0.1, 2, 1, 0),
    RBox(4, 0, 2, 1, 0),
]

# Compute IoU between two boxes
overlap = iou.rbox_iou(boxes[0], boxes[1])

# Batch IoU computation (more efficient)
iou_matrix = iou.batch_rbox_iou(boxes, boxes)
```

### Basic NMS

```python
from oriented_det.geometry import RBox
from oriented_det.ops import nms

boxes = [
    RBox(0, 0, 2, 1, 0),
    RBox(0.5, 0.1, 2, 1, 0),  # Overlaps with first
    RBox(4, 0, 2, 1, 0),  # No overlap
]
scores = [0.9, 0.8, 0.6]

# Apply NMS
keep = nms.oriented_nms(boxes, scores, iou_threshold=0.3)
```

### Class-Aware NMS

```python
# Boxes of different classes don't suppress each other
labels = [0, 0, 1]  # First two boxes are class 0, third is class 1
keep = nms.oriented_nms(boxes, scores, iou_threshold=0.3, labels=labels)
```

## GPU Operations

For maximum performance with large batches, use the GPU-accelerated operations:

### GPU IoU Computation

```python
import torch
from oriented_det.ops.gpu_ops import oriented_box_iou_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input format: [N, 5] tensors with [cx, cy, w, h, angle]
boxes1 = torch.tensor([
    [100, 100, 50, 30, 0.0],
    [200, 150, 40, 60, 0.5],
], device=device)

boxes2 = torch.tensor([
    [105, 105, 50, 30, 0.0],
    [300, 300, 50, 50, 0.0],
], device=device)

# Compute IoU matrix: [N, M]
iou_matrix = oriented_box_iou_gpu(boxes1, boxes2, num_samples=100)
```

### GPU Anchor Generation

```python
import torch
from oriented_det.ops.gpu_ops import generate_oriented_anchors_gpu
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate anchors for FPN levels
anchors_per_level = generate_oriented_anchors_gpu(
    image_size=(800, 800),
    feature_map_sizes=[(200, 200), (100, 100), (50, 50)],
    anchor_scales=[8],
    anchor_ratios=[0.5, 1.0, 2.0],
    anchor_angles=[-math.pi/2, 0, math.pi/2],
    stride_per_level=[4, 8, 16],
    device=device
)
# Returns list of [H*W*A, 5] anchor tensors
```

### GPU Anchor Matching

```python
import torch
from oriented_det.ops.gpu_ops import match_anchors_to_gt_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

anchors = torch.randn(1000, 5, device=device)  # [N, 5]
gt_boxes = torch.randn(10, 5, device=device)  # [M, 5]

labels, matched_gt_indices = match_anchors_to_gt_gpu(
    anchors=anchors,
    gt_boxes=gt_boxes,
    positive_iou_threshold=0.7,
    negative_iou_threshold=0.3,
)
# labels: [N] with -1=ignore, 0=background, 1=foreground
# matched_gt_indices: [N] index of matched GT (-1 if none)
```

### GPU NMS

```python
import torch
from oriented_det.ops.gpu_ops import oriented_nms_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

boxes = torch.randn(1000, 5, device=device)  # [N, 5]
scores = torch.rand(1000, device=device)  # [N]

keep_indices = oriented_nms_gpu(
    boxes=boxes,
    scores=scores,
    iou_threshold=0.5,
    max_detections=100,
)
# Returns tensor of kept box indices
```

**See Also:**
- [GPU Operations Guide](../user-guide/operations.md#gpu-operations) - Detailed usage guide

