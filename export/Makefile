.DEFAULT_GOAL := help

# Repository root (parent of export/)
REPO_ROOT := $(abspath $(CURDIR)/..)

# Defaults: recipe + Hub pretrained weights (override: make export-tf CONFIG=... CKPT=...)
# Download CKPT if missing: odet pretrained download rotated_faster_rcnn_dota_le90_1x
# Override for a trained experiment, e.g. CONFIG=runs/rotated_faster_rcnn/<timestamp>/config.json
CONFIG ?= $(REPO_ROOT)/configs/rotated_faster_rcnn/dota_le90_1x.json
CKPT ?= $(REPO_ROOT)/pretrained/rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.pth
HEIGHT ?= 1024
WIDTH ?= 1024
ARTIFACTS ?= $(CURDIR)/artifacts
# export-tf / export-detect: faster_rcnn_pre_nms | oriented_rcnn_pre_nms | rotated_fcos_pre_nms
# export-onnx alone also accepts backbone | retinanet_heads
MODE ?= faster_rcnn_pre_nms
ONNX_OUT ?= $(ARTIFACTS)/pre_nms.onnx
DETECT_DIR ?= $(ARTIFACTS)/detect
DEVICE ?= cpu
OPSET ?= 17
# Set SKIP_ORT=1 to skip ORT validation after ONNX export
SKIP_ORT ?= 0
# ONNX Runtime EP for TF inference (preds / predict): cpu | cuda | auto
ORT_DEVICE ?= auto
# Val preds / metrics (TF export); compare with repo-root make preds / make metrics (PyTorch)
DATA_ROOT ?=
DATA_SPLIT ?= val
VAL_DIR ?=
PREDICTIONS_DIR ?= $(ARTIFACTS)/predictions
METRICS_PRED_DIR ?=

.PHONY: help export-tf export-onnx export-detect predict test preds metrics eval-tf

## help: Show export Makefile targets (thin wrappers around odet)
help:
	@echo "oriented-det export — Makefile targets (call odet; no Make required for pip installs)"
	@echo "===================================="
	@echo ""
	@echo "  make export-tf     - odet export-tf (ONNX + Keras detect bundle)"
	@echo "  make export-onnx   - odet export-onnx"
	@echo "  make export-detect - odet export-detect"
	@echo "  make predict       - Smoke-test detect bundle (DETECT_DIR, HEIGHT, WIDTH)"
	@echo "  make preds         - odet export-preds → $(PREDICTIONS_DIR)/<ts>/"
	@echo "  make metrics       - Offline mAP/PR on METRICS_PRED_DIR or newest TF predictions/"
	@echo "  make eval-tf       - preds then metrics"
	@echo "  make test          - pytest export/tests/"
	@echo ""
	@echo "Defaults (checkout paths):"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CKPT=$(CKPT)"
	@echo "  MODE=$(MODE)  (export-tf: pre-NMS modes only)"
	@echo "  HEIGHT=$(HEIGHT)  WIDTH=$(WIDTH)"
	@echo "  DEVICE=$(DEVICE)  OPSET=$(OPSET)  SKIP_ORT=$(SKIP_ORT)"
	@echo "  ORT_DEVICE=$(ORT_DEVICE)"
	@echo "  ARTIFACTS=$(ARTIFACTS)"
	@echo "  REPO_ROOT=$(REPO_ROOT)"
	@echo ""
	@echo "Pip / no-checkout:"
	@echo "  odet export-tf --config ... --checkpoint ... --output-dir ./odet_export"
	@echo ""
	@echo "Oriented R-CNN example:"
	@echo "  make export-tf MODE=oriented_rcnn_pre_nms \\"
	@echo "    CONFIG=$(REPO_ROOT)/configs/oriented_rcnn/dota_le90_1x.json \\"
	@echo "    CKPT=$(REPO_ROOT)/pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.pth"
	@echo ""
	@echo "Rotated FCOS example:"
	@echo "  make export-tf MODE=rotated_fcos_pre_nms \\"
	@echo "    CONFIG=$(REPO_ROOT)/configs/rotated_fcos/dota_le90_3x.json \\"
	@echo "    CKPT=$(REPO_ROOT)/pretrained/rotated_fcos_r50_fpn_dota_le90_3x_riou-a39c80c1.pth"

## export-tf: ONNX + Keras detect bundle via odet
export-tf:
	@mkdir -p "$(ARTIFACTS)"
	cd "$(REPO_ROOT)" && odet export-tf \
	  --config "$(CONFIG)" \
	  --checkpoint "$(CKPT)" \
	  --mode "$(MODE)" \
	  --output-dir "$(ARTIFACTS)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --device "$(DEVICE)" \
	  --opset $(OPSET) \
	  $(if $(filter 1 true TRUE yes YES,$(SKIP_ORT)),--skip-ort,)

## export-onnx: PyTorch → ONNX only
export-onnx:
	@mkdir -p "$(ARTIFACTS)"
	cd "$(REPO_ROOT)" && odet export-onnx \
	  --config "$(CONFIG)" \
	  --checkpoint "$(CKPT)" \
	  --output "$(ONNX_OUT)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --mode "$(MODE)" \
	  --device "$(DEVICE)" \
	  --opset $(OPSET) \
	  $(if $(filter 1 true TRUE yes YES,$(SKIP_ORT)),--skip-ort,)

## export-detect: Keras bundle from existing ONNX + meta
export-detect:
	cd "$(REPO_ROOT)" && odet export-detect \
	  --meta "$(ONNX_OUT:.onnx=.export_meta.json)" \
	  --onnx "$(ONNX_OUT)" \
	  --output "$(DETECT_DIR)"

## predict: Smoke-test keras detect bundle
predict:
	cd "$(REPO_ROOT)" && python -m export.scripts.predict_savedmodel \
	  --saved-model "$(DETECT_DIR)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --ort-device "$(ORT_DEVICE)"

## preds: Val inference with TF/Keras detect bundle
preds:
	@if [ ! -f "$(DETECT_DIR)/keras_model.keras" ]; then \
		echo "Error: $(DETECT_DIR)/keras_model.keras missing. Run: make export-tf"; \
		exit 1; \
	fi
	@echo "TF export preds: CONFIG=$(CONFIG) DETECT_DIR=$(DETECT_DIR) ORT_DEVICE=$(ORT_DEVICE)"; \
	ROOT_ARGS=""; \
	if [ -n "$(DATA_ROOT)" ]; then ROOT_ARGS="--data-root $(DATA_ROOT)"; fi; \
	if [ -n "$(VAL_DIR)" ]; then ROOT_ARGS="$$ROOT_ARGS --val-dir $(VAL_DIR)"; fi; \
	TS=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p "$(PREDICTIONS_DIR)"; \
	cd "$(REPO_ROOT)" && odet export-preds \
	  --config "$(CONFIG)" \
	  --detect-dir "$(DETECT_DIR)" \
	  --reference-checkpoint "$(CKPT)" \
	  --data-split "$(DATA_SPLIT)" \
	  --output-dir "$(PREDICTIONS_DIR)/$$TS" \
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

## eval-tf: preds then metrics
eval-tf: preds metrics

## test: Run export unit tests
test:
	cd "$(REPO_ROOT)" && PYTHONPATH="$(REPO_ROOT)" pytest export/tests/
