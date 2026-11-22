# Contributing

Thank you for your interest in contributing to OrientedDet!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/DL4EO/oriented-det.git
   cd oriented-det
   ```
3. Create a venv and install in development mode ([uv](https://docs.astral.sh/uv/)):
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install -e ".[dev]"
   ```

## Development Setup

### Running Tests

CI runs `pytest tests/` on GitHub Actions (CPU PyTorch). Locally:

```bash
# Run all tests
pytest

# Library tests only
pytest tests/

# ONNX export tests (optional onnx/ort tests skip unless oriented-det[export] is installed)
pytest export/tests/

# Run specific test file
pytest tests/test_geometry.py

# With coverage
pytest --cov=oriented_det tests/
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black oriented_det/ tests/ export/

# Lint
ruff check oriented_det/ tests/ export/
```

## Contribution Guidelines

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings (Google style)
- Keep functions focused and small

### Testing

- Add tests for new features
- Ensure all tests pass
- Aim for good test coverage

### Documentation

- Update docstrings for new functions/classes
- Add tool scripts or examples to user guide if applicable
- Update API reference

### Pull Requests

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Ensure all tests pass
6. Submit a pull request

## Project Structure

```
oriented_det/
├── geometry/      # Core geometry primitives
├── ops/           # IoU and NMS operations
├── data/          # Dataset loading and utilities
├── models/        # Model architectures
├── train/         # Training utilities
└── utils/         # Configuration and visualization
```

## Areas for Contribution

Aligned with the [Roadmap](roadmap.md):

- **v0.3:** HRSC2016 Hub zoo; FAIR1M / SSDD / HRSID loaders + 1× recipes (finetune DOTA 1× Hub; no FAIR1M / SAR zoo)
- **v0.4.0:** Rotated RTMDet (shipped)
- **v0.4.2:** shipped — native YOLO-OBB Hub `yolo_obb_dota_le90_12x` (**77.80%** Task 1; COCO trunk `yolo_obb_coco_300e`)
- **v0.5–v0.5.2:** Swin-T + FPN (Oriented R-CNN 1× and 3× Hub in v0.5, Faster R-CNN in v0.5.1, FCOS in v0.5.2 if latency allows). Swin stays off RTMDet and YOLO-OBB
- **Optional:** DOTA multi-scale training (patch 1024, overlap 500, resize rates 0.5 / 1.0 / 1.5; same tiles for RTMDet-S and YOLO-OBB)
- **Optional:** recipe improvements on the six current models, kept when AP75 or a weak class moves on a held-out split at the operating resolution
- **v1.1:** oriented DINO in this repo, after the v1.0 API (DOTA; Hub only if it beats Swin Oriented R-CNN)
- **Docs:** Tutorials, hosted MkDocs, eval reports
- **Export:** ONNX head wrappers + postprocess parity for new detectors
- **Ops:** Fused CUDA rotated IoU/NMS kernels
- Bug fixes and tool scripts

## Questions?

Feel free to open an issue for questions or discussions!

