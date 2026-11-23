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

## Releasing / Publishing to PyPI

Maintainers only. Workflow detail and Trusted Publishing setup: [`.github/workflows/README.md`](../.github/workflows/README.md).

Version **0.4.2** — tag releases as **`v0.4.2`** (git) matching `version` in `pyproject.toml`.

Configs: edit **`configs/`** at the repo root, then **`make sync-configs`** so **`oriented_det/configs/`** stays in sync (see **`oriented_det/configs/vendored_manifest.txt`**). CI runs **`make check-configs`**.

Install publishing tools (included in `.[dev]`):

```bash
uv pip install -e ".[dev]"   # or: make publish-deps
```

### Step-by-step release

#### 1. One-time PyPI setup

1. Create accounts on [test.pypi.org](https://test.pypi.org) and [pypi.org](https://pypi.org) (can use the same email).
2. **Register the project name** `oriented-det` on PyPI (first upload creates it; TestPyPI is separate).
3. **Trusted Publishing (recommended)** — on each index, add a publisher for `DL4EO/oriented-det`, workflow `publish.yml`, environments `testpypi` / `pypi`. See [.github/workflows/README.md](../.github/workflows/README.md).
4. **Or API tokens** — create `pypi-…` tokens and export locally (or add GitHub secrets `TESTPYPI_API_TOKEN`, `PYPI_API_TOKEN`):

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEI…   # TestPyPI or PyPI token
```

#### 2. Pre-release checklist

```bash
make check-configs
pytest tests/ -q
```

Bump version in `pyproject.toml` and `docs/changelog.md` if needed (currently **0.4.2**).

#### 3. Build locally

```bash
make sync-configs   # after changing configs/ or the manifest
make build          # check-configs + sdist + wheel into dist/
make twine-check
```

Smoke-test the wheel:

```bash
python -m venv /tmp/odet-smoke && source /tmp/odet-smoke/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install dist/*.whl
odet --help
python -c "from oriented_det.geometry import RBox; print(RBox(0,0,1,1,0).area)"
```

#### 4. Upload to TestPyPI

```bash
make publish-testpypi
```

Install from TestPyPI (PyPI index still needed for dependencies like `torch`):

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oriented-det==0.4.2
odet --help
```

Or trigger **Actions → Publish → Run workflow** (target: `testpypi`) after configuring Trusted Publishing or secrets.

#### 5. Tag and upload to production PyPI

When TestPyPI looks good:

```bash
git tag -a v0.4.2 -m "Release 0.4.2"
git push origin v0.4.2          # triggers publish.yml → PyPI (if CI is configured)
```

Local upload (fallback):

```bash
make publish-pypi
```

Create a [GitHub Release](https://github.com/DL4EO/oriented-det/releases) from tag `v0.4.2` with notes from `docs/changelog.md`.

#### 6. After publish

- Verify `pip install oriented-det` in a clean environment.
- Users still install **PyTorch** separately for their CUDA/CPU platform.

### Makefile targets

| Target | Action |
|--------|--------|
| `make publish-deps` | Install `build` and `twine` |
| `make build` | `check-configs` + `python -m build` |
| `make twine-check` | Validate `dist/*` metadata |
| `make publish-testpypi` | Build, check, upload to TestPyPI |
| `make publish-pypi` | Build, check, upload to PyPI |

**Trusted Publishing** avoids long-lived tokens; keep `make publish-*` for manual releases. Details: [.github/workflows/README.md](../.github/workflows/README.md).

## Questions?

Feel free to open an issue for questions or discussions!

