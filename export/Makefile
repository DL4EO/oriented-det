.DEFAULT_GOAL := help

# Repository root (parent of export/)
REPO_ROOT := $(abspath $(CURDIR)/..)

# Defaults: published deploy weights (override: make export-tf CONFIG=... CKPT=...)
# Override for a trained experiment, e.g. CONFIG=runs/rotated_faster_rcnn/<timestamp>/config.json
CONFIG ?= $(REPO_ROOT)/configs/rotated_faster_rcnn/dota_le90.json
CKPT ?= $(REPO_ROOT)/pretrained/rotated_faster_rcnn_r50_fpn_082.pth
HEIGHT ?= 1024
WIDTH ?= 1024
ARTIFACTS ?= $(CURDIR)/artifacts
ONNX_OUT ?= $(ARTIFACTS)/faster_rcnn_pre_nms.onnx
DETECT_DIR ?= $(ARTIFACTS)/faster_rcnn_detect
DEVICE ?= cpu
# ONNX Runtime EP for TF inference (preds / predict): cpu | cuda | auto
ORT_DEVICE ?= auto
# Val preds / metrics (TF export); compare with repo-root make preds / make metrics (PyTorch)
DATA_ROOT ?=
DATA_SPLIT ?= val
VAL_DIR ?=
PREDICTIONS_DIR ?= $(ARTIFACTS)/predictions
METRICS_PRED_DIR ?=

.PHONY: help export-tf export-onnx export-detect predict test preds metrics eval-tf

## help: Show export Makefile targets
help:
	@echo "oriented-det export — Makefile targets"
	@echo "===================================="
	@echo ""
	@echo "  make export-tf     - ONNX (faster_rcnn_pre_nms) + Keras detect bundle"
	@echo "  make export-onnx   - ONNX only"
	@echo "  make export-detect - Keras bundle from existing ONNX + meta"
	@echo "  make predict       - Smoke-test detect bundle (DETECT_DIR, HEIGHT, WIDTH)"
	@echo "  make preds         - Val inference via Keras bundle → $(PREDICTIONS_DIR)/<ts>/"
	@echo "  make metrics       - Offline mAP/PR on METRICS_PRED_DIR or newest TF predictions/"
	@echo "  make eval-tf       - preds then metrics (compare with repo-root make eval-val)"
	@echo "  make test          - pytest export/tests/"
	@echo ""
	@echo "Defaults (paths relative to repo root):"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CKPT=$(CKPT)"
	@echo "  HEIGHT=$(HEIGHT)  WIDTH=$(WIDTH)"
	@echo "  DEVICE=$(DEVICE) (PyTorch trace for export-onnx)"
	@echo "  ORT_DEVICE=$(ORT_DEVICE) (ONNX Runtime for predict / preds)"
	@echo "  ARTIFACTS=$(ARTIFACTS)"
	@echo "  REPO_ROOT=$(REPO_ROOT)"

## export-onnx: PyTorch → ONNX (Rotated Faster R-CNN pre-NMS)
export-onnx:
	@mkdir -p "$(ARTIFACTS)"
	cd "$(REPO_ROOT)" && odet export-onnx \
	  --config "$(CONFIG)" \
	  --checkpoint "$(CKPT)" \
	  --output "$(ONNX_OUT)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --mode faster_rcnn_pre_nms \
	  --device "$(DEVICE)"

## export-detect: Build Keras detect bundle from ONNX + sidecar meta
export-detect:
	cd "$(REPO_ROOT)" && python export/scripts/build_faster_rcnn_savedmodel.py \
	  --meta "$(ONNX_OUT:.onnx=.export_meta.json)" \
	  --onnx "$(ONNX_OUT)" \
	  --output "$(DETECT_DIR)"

## export-tf: export-onnx then export-detect
export-tf: export-onnx export-detect
	@echo "Detect bundle: $(DETECT_DIR)/keras_model.keras"

## predict: Smoke-test keras detect bundle
predict:
	cd "$(REPO_ROOT)" && python export/scripts/predict_savedmodel.py \
	  --saved-model "$(DETECT_DIR)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --ort-device "$(ORT_DEVICE)"

## preds: Val inference with TF/Keras detect bundle (requires make export-tf)
preds:
	@if [ ! -f "$(DETECT_DIR)/keras_model.keras" ]; then \
		echo "Error: $(DETECT_DIR)/keras_model.keras missing. Run: make export-tf"; \
		exit 1; \
	fi
	@echo "TF export preds: CONFIG=$(CONFIG) DETECT_DIR=$(DETECT_DIR) ORT_DEVICE=$(ORT_DEVICE)"; \
	ROOT_ARGS=""; \
	if [ -n "$(DATA_ROOT)" ]; then ROOT_ARGS="--data-root $(DATA_ROOT)"; fi; \
	if [ -n "$(VAL_DIR)" ]; then ROOT_ARGS="$$ROOT_ARGS --val-dir $(VAL_DIR)"; fi; \
	cd "$(REPO_ROOT)" && python export/scripts/save_predictions_tf.py \
	  --config "$(CONFIG)" \
	  --detect-dir "$(DETECT_DIR)" \
	  --reference-checkpoint "$(CKPT)" \
	  --data-split "$(DATA_SPLIT)" \
	  $$ROOT_ARGS \
	  --ort-device "$(ORT_DEVICE)" \
	  --no-diagnostics

## metrics: Offline mAP/PR from predictions.json (no TF inference)
metrics:
	@PRED="$(METRICS_PRED_DIR)"; \
	if [ -z "$$PRED" ]; then \
		if [ ! -d "$(PREDICTIONS_DIR)" ]; then \
			echo "Error: $(PREDICTIONS_DIR) missing. Run: make preds"; \
			exit 1; \
		fi; \
		PRED=$$(find "$(PREDICTIONS_DIR)" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1); \
	fi; \
	if [ -z "$$PRED" ] || [ ! -f "$$PRED/predictions.json" ]; then \
		echo "Error: predictions.json not found under $$PRED"; \
		exit 1; \
	fi; \
	echo "TF export metrics: $$PRED"; \
	cd "$(REPO_ROOT)" && odet preds --metrics-from-json "$$PRED"

## eval-tf: preds then metrics on newest export/artifacts/predictions/<ts>/
eval-tf: preds metrics

## test: Run export unit tests (no TF/ONNX required for most tests)
test:
	cd "$(REPO_ROOT)" && PYTHONPATH="$(REPO_ROOT):$(REPO_ROOT)/export" pytest export/tests/
