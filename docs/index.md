# OrientedDet

**OrientedDet** (**oriented-det**) is an open-source PyTorch framework for **oriented and rotated object detection** in aerial and satellite images.  
It focuses on clean geometry, reliable operators, simple datasets, and practical baseline models — without the complexity of large detection frameworks.

OrientedDet is designed for researchers, practitioners, and geospatial developers who need **accurate rotation-aware detectors** with a clean and minimal API.

For a quick overview and installation on GitHub, see the [repository README](https://github.com/DL4EO/oriented-det/blob/main/README.md). Short overview: [oriented detection in aerial and satellite images](https://deeplearning.earth/oriented-detection-in-satellite-images/). Write-ups with code, metrics, and inference knobs: [DeepLearning.Earth](https://deeplearning.earth). For consulting, training workshops, or custom development, see [DL4EO](https://dl4eo.com).

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
- **HRSC2016**, **FAIR1M**, **SSDD**, and **HRSID** native loaders (SSDD / HRSID finetune DOTA 1× Hub locally; no SAR zoo)
- **Three loading modes**: Pattern matching, split file, separate folders
- Image tiling / patch generation with configurable overlap
- Label filtering, edge handling, ignore masks
- Oriented mAP evaluation compatible with DOTA protocol

### 🔸 Baseline Models
- **Oriented R-CNN** — horizontal RPN + MidpointOffset (6-param) proposals + oriented RoIAlign ([Xie et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.html))
- **Rotated Faster R-CNN** — horizontal RPN + horizontal RoIAlign + rotated ROI head (MMRotate DOTA baseline)
- **Rotated RetinaNet** (1-stage baseline) — oriented anchors and focal loss head
- **Rotated FCOS** (anchor-free 1-stage) — distance-angle coder, centerness, L1 / KFIoU / decoded rIoU
- **Rotated RTMDet** (speed tier) — CSPNeXt + PAFPN, SepBN head, dynamic soft-label assigner, decoded rIoU; Hub SS Task 1 **77.60%**, MS/MS **80.13%** (`rotated_rtmdet_dota_le90_3x_ms`)
- **YOLO-OBB** (speed tier) — C2f CSP + PAFPN, DFL box head, task-aligned assigner; Hub Task 1 **77.80%** (top of the DOTA zoo)
- **True oriented detection**: Predicts rotation angles, not just axis-aligned boxes
- Standard backbones (ResNet + FPN, CSPNeXt + PAFPN, C2f CSP + PAFPN)
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
- **[Examples](examples/inference.md)** - Training, inference, [Docker deploy](examples/deploy.md), and [ONNX export](examples/export.md)
- **[Roadmap](roadmap.md)** - Planned releases (next v0.5–v0.7, then v1.0 / v1.1; v0.4.3 RTMDet MS Hub shipped)

## Roadmap

**v0.4.3** is shipped — RTMDet-S MS Hub **`rotated_rtmdet_dota_le90_3x_ms`** at official Task 1 **80.13%** (MS/MS; MS/SS **78.92%**) plus live `odet dota-submit --resize-rates`. **v0.4.2** shipped native YOLO-OBB Hub **`yolo_obb_dota_le90_12x`** at SS Task 1 **77.80%**; **v0.4.0** shipped RTMDet SS **77.60%**. Next: Swin-T + FPN (**v0.5–v0.5.2**), then **RoI Transformer** (**v0.6** R50, **v0.6.1** Swin-T) and **ReDet** (**v0.7**). Swin stays on the FPN accuracy line; RTMDet and YOLO-OBB stay the CNN speed models. YOLO-OBB MS Hub and recipe work on the six current models remain optional. Oriented DINO is **v1.1**, after the stable API. Details: **[Roadmap](roadmap.md)**.

## Important Notes

### True Oriented Detection

✅ **OrientedRCNN**, **RotatedFasterRCNN**, **RotatedRetinaNet**, **RotatedFCOS**, **RotatedRTMDet**, and **YoloOBB** perform **true oriented object detection**:
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

The framework license does not grant rights to the **DOTA** or **HRSC** datasets. See [Apache 2.0 vs DOTA / HRSC dataset terms](https://deeplearning.earth/posts/2026-09-10_oriented_det_apache_license_versus_dota/). For consulting, training workshops, or custom development, see [DL4EO](https://dl4eo.com).
