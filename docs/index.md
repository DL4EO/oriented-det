# OrientedDet

**OrientedDet** is a lightweight, modern PyTorch library for **rotated object detection** in aerial and satellite imagery.  
It focuses on clean geometry, reliable operators, simple datasets, and practical baseline models — without the complexity of large detection frameworks.

OrientedDet is designed for researchers, practitioners, and geospatial developers who need **accurate rotation-aware detectors** with a clean and minimal API.

For a quick overview and installation on GitHub, see the [repository README](https://github.com/DL4EO/oriented-det/blob/main/README.md).

## 🚀 Features

### 🔸 Core Geometry
- Rotated bounding boxes (`rbox`: cx, cy, w, h, angle)
- Quadrilateral boxes (`qbox`) and conversions
- Polygon ↔ rbox ↔ hbox conversion utilities
- Angle normalization (`le90`, 0–180°, etc.) - **Fixed**: Corrected le90 normalization to properly handle width/height swaps and angle normalization
- Flip, rotate, and scale transformations
- Visualization helpers for debugging

### 🔸 Fast Rotated IoU + Rotated NMS
- Python-based NMS using polygon intersection (CPU)
- **Note:** `torchvision.ops.nms_rotated` does not exist in current torchvision versions
- Optimized with AABB pre-filtering (~2.6x faster than naive implementation)
- Clean wrappers with fully validated interfaces
- Future: GPU-accelerated kernels via custom CUDA implementation

### 🔸 Remote-Sensing Datasets
- DOTA polygon loader → rbox/qbox conversion
- **Three loading modes**: Pattern matching, split file, separate folders
- Image tiling / patch generation with configurable overlap
- Label filtering, edge handling, ignore masks
- Oriented mAP evaluation compatible with DOTA protocol

### 🔸 Baseline Models
- **Oriented R-CNN** — horizontal RPN with midpoint-offset proposals and oriented ROI head
- **Rotated Faster R-CNN** — oriented RPN and oriented ROI two-stage detector
- **Rotated RetinaNet** (1-stage baseline) — oriented anchors and focal loss head
- **True oriented detection**: Predicts rotation angles, not just axis-aligned boxes
- Standard backbones (ResNet + FPN)
- **OrientedDet pretrained weights** via Hugging Face Hub (`odet pretrained download`)

### 🔸 Simple Training Pipeline
A clean, readable PyTorch training loop with efficient features:
- Mixed precision training (AMP)
- Gradient accumulation
- Checkpointing with best model tracking
- Metric tracking and TensorBoard logging
- Performance profiling support
- Robust error handling

## Quick Start

Hands-on installation and a minimal walkthrough are in [Getting Started](getting-started/installation.md). For JSON-driven training and every config option, see [Configuration](user-guide/configuration.md) and [Training](user-guide/training.md).

## Installation

From PyPI:

```bash
pip install oriented-det
```

For development (clone + [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/DL4EO/oriented-det
cd oriented-det
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ".[dev]"
```

See [Installation](getting-started/installation.md) for CUDA, macOS, and CPU setups.

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation and quick start guides
- **[User Guide](user-guide/geometry.md)** - Detailed usage for each module
- **[API Reference](api/geometry.md)** - Complete API documentation
- **[Tools](examples/visualization.md)** - Ready-to-use scripts and practical examples

## Important Notes

### True Oriented Detection

✅ **OrientedRCNN**, **RotatedFasterRCNN**, and **RotatedRetinaNet** perform **true oriented object detection**:
- Predict oriented bounding boxes with 5 parameters (cx, cy, w, h, angle)
- Preserve angle information throughout training and inference
- Use oriented IoU for matching and oriented NMS for post-processing
- Output RBoxes include predicted angles (not `angle=0`)

### Angle Normalization

- Default: Full circle representation `(-π, π]`
- For DOTA compatibility: Use `normalize_le90()` for `[-π/2, π/2)` convention
- **Fixed**: Corrected le90 normalization to properly handle width/height swaps

### NMS Performance

- **Note:** `torchvision.ops.nms_rotated` does not exist in current torchvision versions
- Uses optimized Python implementation with AABB pre-filtering (~2.6x faster)
- Future: GPU-accelerated kernels planned

### Memory Optimization

- `OrientedRCNN` uses memory-efficient ROI align (chunked processing)
- Recommended: `roi_chunk_size=16-32` for typical GPU memory (8-24GB)
- Enable `roi_use_checkpoint=True` for additional memory savings (~2x less memory)

## License

Apache-2.0 — Copyright © Jeff Faudi and [DL4EO](https://dl4eo.com). See [LICENSE](https://github.com/DL4EO/oriented-det/blob/main/LICENSE) for details.
