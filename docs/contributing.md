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

# Run specific test file
pytest tests/test_geometry.py

# With coverage
pytest --cov=oriented_det tests/
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black oriented_det/ tests/

# Lint
ruff check oriented_det/ tests/
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

- **v0.3:** HRSC2016 and FAIR1M dataset loaders
- **v0.4:** RTMDet-R and native YOLO-OBB (clean-room, no Ultralytics)
- **v0.5+:** Swin-FPN backbone
- **Docs:** Tutorials, hosted MkDocs, eval reports
- **Ops:** Fused CUDA rotated IoU/NMS kernels
- Bug fixes and tool scripts

## Questions?

Feel free to open an issue for questions or discussions!

