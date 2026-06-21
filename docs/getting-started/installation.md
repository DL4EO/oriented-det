# Installation

## Requirements

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) (recommended for Python versions, virtualenv, and installs)
- PyTorch and TorchVision (see below)
- Pillow >= 9.0

## Install from repository (Linux, CUDA 12.1)

From the project directory:

```bash
cd oriented-det

# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv venv --python 3.12
source .venv/bin/activate

# PyTorch with CUDA 12.1 (2.3.0 or newer)
uv pip install "torch>=2.3.0" "torchvision>=0.18.0" --index-url https://download.pytorch.org/whl/cu121

# Project dependencies and editable install (includes huggingface_hub for Hub weights)
uv pip install -r requirements.txt
uv pip install -e .
```

## PyTorch and TorchVision (other setups)

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your OS and CUDA/CPU setup.

For CPU-only:

```bash
uv pip install torch torchvision
```

### macOS Apple Silicon (M1/M2/M3)

On Mac with Apple Silicon, install the default PyTorch build (no CUDA). PyTorch will use the **MPS (Metal Performance Shaders)** backend for GPU acceleration:

```bash
uv pip install torch torchvision
```

Then install oriented-det (see below). The library auto-detects MPS and uses the Apple GPU when available. If you hit mixed-precision (AMP) errors on older PyTorch, train with `--no-amp`. For best MPS and AMP support, use PyTorch 2.5 or newer.

## Install OrientedDet (after PyTorch)

If you already cloned the repo and installed PyTorch (e.g. via the steps above or for macOS/CPU):

```bash
cd oriented-det
uv pip install -r requirements.txt
uv pip install -e .
```

From PyPI:

```bash
pip install oriented-det
```

For development and testing:

```bash
uv pip install -e ".[dev]"
```

This adds pytest, pytest-cov, black, and ruff.

## Verify Installation

```bash
pytest tests/test_geometry.py tests/test_iou.py tests/test_nms.py
```

Or in Python:

```python
import oriented_det
from oriented_det.geometry import RBox, Polygon
from oriented_det.ops import iou, nms

rbox = RBox(100, 200, 50, 30, 0.5)
print(f"RBox area: {rbox.area}")

box1 = RBox(0, 0, 10, 10, 0)
box2 = RBox(5, 5, 10, 10, 0)
overlap = iou.rbox_iou(box1, box2)
print(f"IoU: {overlap}")
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure:
1. PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
2. TorchVision is installed: `python -c "import torchvision; print(torchvision.__version__)"`
3. The package is installed: `uv pip list | grep oriented-det` (or `pip list` if you used PyPI only)

### GPU Support

GPU acceleration is used when available: **CUDA** (Linux/Windows) or **MPS** (macOS Apple Silicon). To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Apple Silicon) available: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
```

Or use the helper:

```python
from oriented_det.utils import get_device
print(get_device())  # e.g. device(type='mps') or device(type='cuda') or device(type='cpu')
```
