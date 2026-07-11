.PHONY: help install check-install docs-deps train wizard train-wizard train-multi-gpu train-help lr-finder stats tensorboard free-gpu clean \
	preds metrics eval-val train-preds viewer demo test docs docs-serve sync-configs check-configs build twine-check publish-deps publish-testpypi publish-pypi upload-pretrained

.DEFAULT_GOAL := help

# Default pytest command
PYTEST ?= pytest

# Test directory
TESTS ?= tests/

# Python path
PYTHONPATH ?= $(shell pwd)

# =============================================================================
# User-tunable defaults (override: make <target> VAR=value or export VAR)
# =============================================================================

# Training / wizard / stats / lr-finder
CONFIG ?= configs/oriented_rcnn/dota_le90_1x.json

# Sliding-window micro-batch for large-image inference (env consumed by save_predictions / oriented_det.runtime.inference).
ORIENTED_DET_WINDOW_BATCH_SIZE ?= 64

# Prefer pip-installed cuDNN over system /usr/local/cuda (avoids libcudnn_cnn_train.so.8 / GET engine errors)
CUDNN_LIB := $(shell python -c "import os, site; p=[x for x in site.getsitepackages() if 'site-packages' in x][0]; print(os.path.join(p,'nvidia/cudnn/lib'))" 2>/dev/null)
TRAIN_ENV = LD_LIBRARY_PATH="$(CUDNN_LIB):$$LD_LIBRARY_PATH"

# VIEWER_PRED_DIR: predictions dir for make viewer (default: latest under predictions/)
VIEWER_PRED_DIR ?=
SAVE_TRAIN_PRED_OUT ?=
# make metrics: explicit predictions dir (default: newest predictions/*/)
METRICS_PRED_DIR ?=
# make demo: inference on all top-level images in DEMO_DIR (*.jpg, *.jpeg, *.png; see tools/image_demo.py)
IMAGE_DEMO_DEVICE ?= cuda:0
IMAGE_DEMO_OUT_DIR ?= demo/out
DEMO_DIR ?= demo

# Learning rate finder default batch size (lr-finder target also accepts BATCH_SIZE=).
LR_FINDER_BATCH_SIZE ?= 8

# Hugging Face Hub pretrained weights (see oriented_det/pretrained/manifest.json)
HF_REPO_ID ?= dl4eo/oriented-det-pretrained
HF_REVISION ?= main
PRETRAINED_DIR ?= pretrained
HF_COMMIT_MESSAGE ?= Update OrientedDet pretrained checkpoints

help:
	@echo "OrientedDet — Makefile targets (defaults: top of this Makefile; bare \`make\` shows this help)"
	@echo ""
	@echo "=== Setup & checks ==="
	@echo "  make install                - Install the oriented-det package in editable mode"
	@echo "  make check-install          - Check if the package is installed"
	@echo ""
	@echo "=== Configs (PyPI vendoring) ==="
	@echo "  make sync-configs             - Copy manifest-listed configs/ → oriented_det/configs/"
	@echo "  make check-configs            - Fail if vendored configs differ from configs/ (CI)"
	@echo ""
	@echo "=== Packaging & publishing ==="
	@echo "  make build                  - check-configs, then build sdist+wheel into dist/"
	@echo "  make publish-deps           - Install publishing tools (twine)"
	@echo "  make twine-check            - Run twine check dist/*"
	@echo "  make publish-testpypi       - Upload dist/* to TestPyPI (TWINE_USERNAME/TWINE_PASSWORD or token)"
	@echo "  make publish-pypi           - Upload dist/* to PyPI (TWINE_USERNAME/TWINE_PASSWORD or token)"
	@echo ""
	@echo "=== Pretrained (Hugging Face Hub) ==="
	@echo "  make upload-pretrained        - Upload manifest-listed .pth (+ sidecar .json/.log) from $(PRETRAINED_DIR)/ to $(HF_REPO_ID)"
	@echo "    HF_REPO_ID=  HF_REVISION=  HF_COMMIT_MESSAGE=  PRETRAINED_DIR="
	@echo "    Requires: hf auth login (huggingface_hub CLI)"
	@echo ""
	@echo "=== Training & data prep (all use CONFIG unless noted) ==="
	@echo "  make train                  - Train (checkpoint behavior: JSON checkpoint.* only)"
	@echo "  make train CONFIG=<path>    - Same with a custom config file"
	@echo "  make train DEBUG=1          - Train with --debug (extra logs for diagnosing)"
	@echo "  make train-multi-gpu        - Multi-GPU: same CONFIG as make train (torchrun wrapper; see Multi-GPU below)"
	@echo "  make train-multi-gpu NPROC=4  - Use 4 GPUs (omit NPROC for all available)"
	@echo "  make wizard                 - Dataset/config diagnostics (no training, no run directory)"
	@echo "  make stats                  - Dataset stats + normalization for CONFIG [MAX_SAMPLES=] [SPLIT=train|val]"
	@echo "  make lr-finder              - LR sweep / suggested base LR for CONFIG [OUTPUT=] [BATCH_SIZE=]"
	@echo "  make train-help             - Raw python training command examples (fine-tuning, overrides)"
	@echo ""
	@echo "=== Tiled val: predictions & offline metrics ==="
	@echo "  make eval-val               - preds then metrics on newest predictions/<ts>/ (same vars as below)"
	@echo "  make preds                  - Val inference → predictions/<ts>/predictions.json (production.*; no GPU mAP)"
	@echo "  make metrics                - Offline mAP/PR on METRICS_PRED_DIR or latest predictions/"
	@echo "  make train-preds            - Train split + tile_metrics.csv (latest exp; SAVE_TRAIN_PRED_OUT= optional)"
	@echo ""
	@echo "=== Viewers, demos, TensorBoard ==="
	@echo "  make tensorboard            - TensorBoard for all experiments under runs/"
	@echo "  make viewer                 - Gradio viewer (VIEWER_PRED_DIR= or latest predictions/; see docs/eval-reports/)"
	@echo "  make demo                   - image_demo on all top-level images in DEMO_DIR ($(DEMO_DIR)) → $(IMAGE_DEMO_OUT_DIR)/"
	@echo ""
	@echo "=== Export (TensorFlow) ==="
	@echo "  cd export && make help    - ONNX + Keras detect bundle (see export/README.md)"
	@echo ""
	@echo "=== Docs & tests ==="
	@echo "  make docs-deps              - pip install -e \".[docs]\" (MkDocs + mkdocstrings)"
	@echo "  make docs                   - Build documentation (installs docs deps if needed → site/)"
	@echo "  make docs-serve             - Serve docs at http://127.0.0.1:8000 (installs deps if needed)"
	@echo "  make test                   - Run all tests (TESTS=$(TESTS))"
	@echo "  make test TESTS=<path>      - Run specific test file(s) or directory"
	@echo "  make test TESTS=tests/test_geometry.py tests/test_rpn.py"
	@echo ""
	@echo "=== Utilities ==="
	@echo "  make free-gpu               - Kill processes using the GPU (free memory before re-training)"
	@echo "  make clean                  - Remove generated outputs (see recipe for details)"
	@echo ""
	@echo "=== Defaults, overrides, and auto-discovery ==="
	@echo "  Default CONFIG: $(CONFIG)"
	@echo "  Pin experiment when newest run is wrong or has no checkpoint:"
	@echo "    EXPERIMENT=runs/<model>/<timestamp>     (preds, train-preds)"
	@echo "  Pin directories: METRICS_PRED_DIR=  SAVE_TRAIN_PRED_OUT="
	@echo "  Newest experiment dir: latest timestamp under runs/<model>/<timestamp>/ (sort by timestamp, not model name)"
	@echo "  Checkpoint pick order: checkpoint_best.pth → best_*.pth → newest checkpoint_epoch_*.pth"
	@echo "  DOTA_DATA_ROOT: if unset for preds / train-preds, data paths come from experiment config.json"
	@echo "  Checkpoint loading (training): config-only — checkpoint.load_from_checkpoint, load_from_experiment,"
	@echo "    discover_previous_run, resume_from_checkpoint_epoch, etc."
	@echo "  preds: thresholds, overlap, NMS, margins from config production.*"
	@echo "    (evaluation.* is training-time val only). Advanced overrides: call tools/save_predictions.py manually."
	@echo "  stats / training paths: edit tools/train.py for DATA_ROOT, fine-tuning, pretrained weights as needed."
	@echo "  make demo: DEMO_DIR=$(DEMO_DIR) — only *.jpg, *.jpeg, *.png directly under that folder (not subdirs e.g. out/)."
	@echo ""
	@echo "=== Multi-GPU / environment ==="
	@echo "  Prefer make train-multi-gpu over raw python: tools/train_multi_gpu.py / torchrun, DDP env, and"
	@echo "  TRAIN_ENV (pip cuDNN first on LD_LIBRARY_PATH — avoids GET engine / libcudnn mismatch)."
	@echo "  Full process log: TRAIN_MULTI_GPU_LOG=$(TRAIN_MULTI_GPU_LOG) (stdout+stderr tee)."

# --- Config vendoring (repo configs/ → oriented_det/configs/) ---
sync-configs:
	@python tools/sync_vendored_configs.py

check-configs:
	@python tools/sync_vendored_configs.py --check

# --- Packaging / publishing ---
build: check-configs
	@echo "Building sdist+wheel into dist/ ..."
	@python -m build

twine-check:
	@python -m twine check dist/*

publish-testpypi: build twine-check
	@python -m twine upload --repository testpypi dist/*

publish-pypi: build twine-check
	@python -m twine upload dist/*

publish-deps:
	@pip install -U build twine

# Upload published checkpoints to Hugging Face Hub (filenames must match manifest).
# Prerequisite: python tools/publish_checkpoint.py … && update oriented_det/pretrained/manifest.json
upload-pretrained: check-install
	@command -v hf >/dev/null 2>&1 || { \
		echo "Error: 'hf' CLI not found. Install oriented-det (huggingface_hub) and run: hf auth login"; \
		exit 1; \
	}
	@echo "Checking manifest-listed checkpoints under $(PRETRAINED_DIR)/ ..."
	@python -c "import json, sys; from pathlib import Path; \
		m=json.loads(Path('oriented_det/pretrained/manifest.json').read_text()); \
		root=Path('$(PRETRAINED_DIR)'); missing=[]; \
		[missing.append(f) for f in (e['filename'] for e in m['assets'].values()) if not (root/f).is_file()]; \
		sys.exit('Missing: ' + ', '.join(missing) if missing else 0)"
	@echo "Uploading to $(HF_REPO_ID) (revision=$(HF_REVISION)) ..."
	@hf upload $(HF_REPO_ID) $(PRETRAINED_DIR)/ \
		--include "*.pth" \
		--include "*.json" \
		--include "*.log" \
		--repo-type model \
		--revision $(HF_REVISION) \
		--commit-message "$(HF_COMMIT_MESSAGE)"
	@echo "Done. Verify: odet pretrained list && odet pretrained download <slug>"

# Install the package in editable mode
install:
	@echo "Installing oriented-det package in editable mode (uv pip)..."
	@uv pip install -e .
	@echo "Installation complete!"

# Check if the package is installed
check-install:
	@python -c "import oriented_det; print('✓ oriented_det is installed')" 2>/dev/null || \
		(echo "✗ oriented_det is not installed. Run 'make install' first." && exit 1)

# --- Training & data prep ---
# Train using CONFIG (checkpoint behavior from JSON checkpoint.* only)
# Usage: make train
#        make train CONFIG=configs/oriented_rcnn/dota_le90_1x.json
#        make train DEBUG=1   # extra debug logs (loss breakdown, per-class mAP, RPN/ROI stats)
train: check-install
	@echo "Starting training with config: $(CONFIG)"
	@$(TRAIN_ENV) odet train --config $(CONFIG) $(if $(DEBUG),--debug,)

# Run wizard diagnostics (no training, no run directory)
# Usage: make wizard [CONFIG=<path>]
wizard: check-install
	@echo "Running wizard diagnostics with config: $(CONFIG)"
	@$(TRAIN_ENV) odet train --config $(CONFIG) --wizard

# Learning rate finder: sweep LR and suggest base learning rate for config (default LR_FINDER_BATCH_SIZE=8; override BATCH_SIZE=)
# Usage: make lr-finder [CONFIG=<path>] [OUTPUT=lr_finder.png] [BATCH_SIZE=8]
lr-finder: check-install
	@echo "Running LR finder with config: $(CONFIG)"
	@$(TRAIN_ENV) odet lr-finder --config $(CONFIG) $(if $(OUTPUT),--output $(OUTPUT),) --batch-size $(or $(BATCH_SIZE),$(LR_FINDER_BATCH_SIZE))

# Backward compatibility
train-wizard: wizard
	@:

# Dataset stats (sanity checks, class/annotation stats) and normalization mean/std for the config.
# Usage: make stats [CONFIG=<path>] [MAX_SAMPLES=500] [SPLIT=train]
# Use STATS_ONLY=1 for sanity + stats only (no normalization). Use NORM_ONLY=1 for normalization only.
stats: check-install
	@echo "Running dataset stats for config: $(CONFIG)"
	@RUN="odet stats --config $(CONFIG)"; \
	if [ -n "$(MAX_SAMPLES)" ]; then RUN="$$RUN --max-samples $(MAX_SAMPLES)"; fi; \
	if [ -n "$(SPLIT)" ]; then RUN="$$RUN --split $(SPLIT)"; fi; \
	if [ "$(STATS_ONLY)" = "1" ]; then RUN="$$RUN --stats-only"; fi; \
	if [ "$(NORM_ONLY)" = "1" ]; then RUN="$$RUN --normalization-only"; fi; \
	$$RUN

# Multi-GPU: CONFIG as-is (same as make train)
# Usage: make train-multi-gpu [NPROC=4] [CONFIG=<path>] [BATCH_SIZE=8] [TRAIN_MULTI_GPU_LOG=path]
train-multi-gpu: check-install
	@echo "Starting multi-GPU training with config: $(CONFIG)"
	@echo "Full stdout/stderr (launcher + all ranks): $(TRAIN_MULTI_GPU_LOG)"
	@$(TRAIN_ENV); \
	mkdir -p "$(dir $(TRAIN_MULTI_GPU_LOG))"; \
	EXTRA=; \
	if [ -n "$(BATCH_SIZE)" ]; then EXTRA="--batch-size $(BATCH_SIZE)"; fi; \
	if [ -n "$(DEBUG)" ]; then EXTRA="$$EXTRA --debug"; fi; \
	if [ -z "$(NPROC)" ]; then \
		echo "Using all available GPUs..."; \
		/bin/bash -o pipefail -c "odet train-multi-gpu --config $(CONFIG) $$EXTRA 2>&1 | tee \"$(TRAIN_MULTI_GPU_LOG)\""; \
	else \
		echo "Using $(NPROC) GPUs..."; \
		/bin/bash -o pipefail -c "odet train-multi-gpu --nproc-per-node $(NPROC) --config $(CONFIG) $$EXTRA 2>&1 | tee \"$(TRAIN_MULTI_GPU_LOG)\""; \
	fi

# Training example (shows usage for fine-tuning)
train-help:
	@echo "For Makefile defaults, CONFIG=, and auto-discovery of runs/ / checkpoints, see: make help"
	@echo ""
	@echo "Config-Based Training Usage:"
	@echo ""
	@echo "Basic training:"
	@echo "  odet train --config configs/oriented_rcnn/dota_le90_1x.json"
	@echo ""
	@echo "With overrides:"
	@echo "  odet train --config configs/oriented_rcnn/dota_le90_1x.json \\"
	@echo "      --batch-size 4 --use-amp"
	@echo ""
	@echo "Fine-tuning example (simpler script):"
	@echo "  python tools/train_example.py /path/to/dota/dataset \\"
	@echo "      --model-type oriented_rcnn \\"
	@echo "      --num-classes 15 \\"
	@echo "      --batch-size 4"
	@echo ""
	@echo "Note: See configs/ directory for reference configuration files"
	@echo "      tools/train_example.py is a simpler example script"

# --- TensorBoard ---
# Open TensorBoard for all experiments in runs/
tensorboard:
	@if [ -d "runs" ]; then \
		echo "Serving TensorBoard at http://localhost:6006 (logdir: runs/)"; \
		echo ""; \
		PYTHONWARNINGS=ignore::DeprecationWarning python -m tensorboard.main --logdir runs --bind_all; \
	else \
		echo "No runs directory found. Run training first."; \
		exit 1; \
	fi

# --- Tiled val: preds / metrics / train-preds ---
# Run inference on val (no mAP). Same checkpoint discovery as before; then run make metrics on the output dir.
# Optionally pin an experiment directory:
#   make preds EXPERIMENT=runs/oriented_rcnn/20260616-030231
preds: check-install
	@echo "Finding latest experiment and checkpoint..."; \
	if [ ! -d "runs" ]; then \
		echo "Error: No runs directory found. Run training first."; \
		exit 1; \
	fi; \
	if [ -n "$(EXPERIMENT)" ]; then \
		LATEST_EXP="$(EXPERIMENT)"; \
	else \
		LATEST_EXP=$$(find runs -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort -t/ -k3 -r | head -1); \
	fi; \
	if [ -z "$$LATEST_EXP" ]; then \
		echo "Error: No experiment directories found in runs/"; \
		echo "Run training first to create experiments."; \
		exit 1; \
	fi; \
	if [ ! -d "$$LATEST_EXP" ]; then \
		echo "Error: EXPERIMENT directory not found: $$LATEST_EXP"; \
		exit 1; \
	fi; \
	MODEL_TYPE=$$(basename $$(dirname "$$LATEST_EXP")); \
	echo "Experiment: $$LATEST_EXP"; \
	echo "Model type: $$MODEL_TYPE"; \
	echo "preds: val inference from experiment config (production.*; see tools/save_predictions.py). GPU mAP: run make metrics."; \
	if [ -n "$$DOTA_DATA_ROOT" ]; then \
		$(TRAIN_ENV) odet preds --model-type "$$MODEL_TYPE" --experiment-dir "$$LATEST_EXP" --data-root "$$DOTA_DATA_ROOT" --data-split val --no-diagnostics; \
	else \
		echo "DOTA_DATA_ROOT not set; using data path from experiment config (dataset.data_root / val_tiles_dir)."; \
		$(TRAIN_ENV) odet preds --model-type "$$MODEL_TYPE" --experiment-dir "$$LATEST_EXP" --data-split val --no-diagnostics; \
	fi

# Tiled val: inference then offline mAP/PR (metrics uses newest predictions/<ts>/ unless METRICS_PRED_DIR= is set).
eval-val: preds metrics

# Offline mAP / PR / analysis from predictions.json (no GPU inference).
# Uses METRICS_PRED_DIR or the newest directory under predictions/.
metrics: check-install
	@PRED="$(METRICS_PRED_DIR)"; \
	if [ -z "$$PRED" ]; then \
		if [ ! -d "predictions" ]; then \
			echo "Error: predictions/ missing. Run make preds first."; \
			exit 1; \
		fi; \
		PRED=$$(find predictions -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1); \
	fi; \
	if [ -z "$$PRED" ] || [ ! -f "$$PRED/predictions.json" ]; then \
		echo "Error: predictions.json not found under $$PRED"; \
		exit 1; \
	fi; \
	echo "metrics: $$PRED (recompute mAP/PR from JSON; pass CLI flags to save_predictions.py to override thresholds)"; \
	$(TRAIN_ENV) odet preds --metrics-from-json "$$PRED"

# Train split: run save_predictions with --data-split train and write tile_metrics.csv (for dataset.tile_metrics_csv / hard-tile oversampling).
# Default output: SAVE_TRAIN_PRED_OUT or <latest_exp>/train_tile_eval (predictions.json, analysis_*.json, tile_metrics.csv).
# Optionally pin an experiment directory:
#   make train-preds EXPERIMENT=runs/oriented_rcnn/20260616-030231
# Usage: make train-preds
#        make train-preds DOTA_DATA_ROOT=/path/to/dota
#        make train-preds SAVE_TRAIN_PRED_OUT=/path/to/out
train-preds: check-install
	@echo "Finding latest experiment..."; \
	if [ ! -d "runs" ]; then \
		echo "Error: No runs directory found. Run training first."; \
		exit 1; \
	fi; \
	if [ -n "$(EXPERIMENT)" ]; then \
		LATEST_EXP="$(EXPERIMENT)"; \
		LATEST_EXP="$${LATEST_EXP%/}"; \
	else \
		LATEST_EXP=$$(find runs -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort -t/ -k3 -r | head -1); \
	fi; \
	if [ -z "$$LATEST_EXP" ]; then \
		echo "Error: No experiment directories found in runs/"; \
		exit 1; \
	fi; \
	if [ ! -d "$$LATEST_EXP" ]; then \
		echo "Error: EXPERIMENT directory not found: $$LATEST_EXP"; \
		exit 1; \
	fi; \
	MODEL_TYPE=$$(basename $$(dirname "$$LATEST_EXP")); \
	OUT="$(SAVE_TRAIN_PRED_OUT)"; \
	if [ -z "$$OUT" ]; then OUT="$$LATEST_EXP/train_tile_eval"; fi; \
	mkdir -p "$$OUT"; \
	echo "Experiment: $$LATEST_EXP"; \
	echo "Model type: $$MODEL_TYPE"; \
	echo "Output dir: $$OUT"; \
	echo "Tile metrics: $$OUT/tile_metrics.csv"; \
	if [ -n "$$DOTA_DATA_ROOT" ]; then \
		odet preds --model-type "$$MODEL_TYPE" --experiment-dir "$$LATEST_EXP" --data-root "$$DOTA_DATA_ROOT" --data-split train --tile-metrics-csv tile_metrics.csv --output-dir "$$OUT"; \
	else \
		echo "DOTA_DATA_ROOT not set; using data path from experiment config (dataset.data_root / train_tiles_dir)."; \
		odet preds --model-type "$$MODEL_TYPE" --experiment-dir "$$LATEST_EXP" --data-split train --tile-metrics-csv tile_metrics.csv --output-dir "$$OUT"; \
	fi; \
	echo ""; \
	echo "For training: set dataset.tile_metrics_csv to: $$OUT/tile_metrics.csv"

# --- Demo inference (tools/image_demo.py) ---
# All top-level images in DEMO_DIR with latest runs/* checkpoint (see demo/README.md).
# Override: DEMO_DIR=... IMAGE_DEMO_OUT_DIR=... IMAGE_DEMO_DEVICE=cpu
demo: check-install
	@set -e; \
	if [ ! -d "$(DEMO_DIR)" ]; then \
		echo "Error: DEMO_DIR does not exist: $(DEMO_DIR)"; \
		exit 1; \
	fi; \
	if [ ! -d "runs" ]; then \
		echo "Error: No runs directory found. Run training first."; \
		exit 1; \
	fi; \
	LATEST_EXP=$$(find runs -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort -t/ -k3 -r | head -1); \
	if [ -z "$$LATEST_EXP" ]; then \
		echo "Error: No experiment directories found in runs/"; \
		exit 1; \
	fi; \
	CKPT="$$LATEST_EXP/checkpoints/checkpoint_best.pth"; \
	if [ ! -f "$$CKPT" ]; then \
		CKPT=$$(ls -1 "$$LATEST_EXP/checkpoints/best_"*.pth 2>/dev/null | sort | head -1); \
	fi; \
	if [ -z "$$CKPT" ] || [ ! -f "$$CKPT" ]; then \
		CKPT=$$(ls -1t "$$LATEST_EXP/checkpoints/checkpoint_epoch_"*.pth 2>/dev/null | head -1); \
	fi; \
	if [ -z "$$CKPT" ] || [ ! -f "$$CKPT" ]; then \
		echo "Error: No checkpoint found under $$LATEST_EXP/checkpoints/"; \
		exit 1; \
	fi; \
	mkdir -p "$(IMAGE_DEMO_OUT_DIR)"; \
	echo "image_demo: $(DEMO_DIR)/ -> $(IMAGE_DEMO_OUT_DIR) (exp $$LATEST_EXP)"; \
	$(TRAIN_ENV) odet image-demo "$(DEMO_DIR)" "$$LATEST_EXP/config.json" "$$CKPT" \
		--out-dir "$(IMAGE_DEMO_OUT_DIR)" --device "$(IMAGE_DEMO_DEVICE)"

# --- Gradio viewers ---
# Launch Gradio app to view predictions (auto-detects latest predictions folder)
viewer: check-install
	@if [ -n "$(VIEWER_PRED_DIR)" ]; then \
		PRED="$(VIEWER_PRED_DIR)"; \
		echo "Using VIEWER_PRED_DIR: $$PRED"; \
	elif [ -d "predictions" ] && [ -n "$$(find predictions -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)" ]; then \
		PRED=$$(find predictions -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1); \
		echo "Latest local predictions: $$PRED"; \
	else \
		echo "Error: no predictions dir. Set VIEWER_PRED_DIR=predictions/<timestamp> or run make preds."; \
		exit 1; \
	fi; \
	if [ ! -f "$$PRED/predictions.json" ]; then \
		echo "Error: predictions.json not found under $$PRED"; \
		exit 1; \
	fi; \
	if [ -z "$$DOTA_DATA_ROOT" ]; then \
		echo "Warning: DOTA_DATA_ROOT not set; viewer uses paths from predictions metadata."; \
		odet viewer --mode predictions --predictions-dir "$$PRED"; \
	else \
		odet viewer --mode predictions --predictions-dir "$$PRED" --data-root "$$DOTA_DATA_ROOT"; \
	fi

# --- Utilities ---
# Free GPU memory by killing processes using the GPU (use after a crash or before re-launching training)
free-gpu:
	@odet free-gpu --force

# Clean generated output (checkpoints under runs/, site/, tool PNGs)
clean:
	@echo "Cleaning generated output files..."
	@rm -f tools/visualization.png tools/demo.png tools/*.png tools/*_detections.png
	@if [ -d "runs" ]; then \
		echo "Cleaning checkpoint files (keeping experiment directories and configs)..."; \
		find runs -name "*.pth" -type f -delete 2>/dev/null || true; \
	fi
	@if [ -d "site" ]; then \
		echo "Cleaning documentation site directory..."; \
		rm -rf site; \
	fi
	@echo "Done!"

# --- Docs & tests ---
# Run tests
test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) $(TESTS)

# MkDocs + mkdocstrings (optional extra; not required for training)
docs-deps: check-install
	@echo "Installing documentation dependencies (uv pip install -e \".[docs]\")..."
	@uv pip install -e ".[docs]"

# Build the documentation (uses active Python: python -m mkdocs)
docs: check-install
	@python -c "import mkdocs" 2>/dev/null || $(MAKE) docs-deps
	@echo "Building documentation..."
	@python -m mkdocs build
	@echo "Documentation built successfully in site/"

# Serve the documentation locally (with auto-reload)
docs-serve: check-install
	@python -c "import mkdocs" 2>/dev/null || $(MAKE) docs-deps
	@echo "Starting documentation server..."
	@echo "Documentation will be available at http://127.0.0.1:8000"
	@echo "Press Ctrl+C to stop the server"
	@python -m mkdocs serve
