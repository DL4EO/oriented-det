.DEFAULT_GOAL := help

# Repository root (parent of export/)
REPO_ROOT := $(abspath $(CURDIR)/..)

# Defaults: Rotated FCOS DOTA 1× Hub checkpoint. Artifacts → ../onnx_export (override ARTIFACTS=).
# Override for a trained experiment, e.g. EXPERIMENT=runs/rotated_fcos/<timestamp>
CONFIG ?= $(REPO_ROOT)/configs/rotated_fcos/dota_le90_1x.json
CKPT ?= $(REPO_ROOT)/pretrained/rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.pth
HEIGHT ?= 1024
WIDTH ?= 1024
ARTIFACTS ?= $(REPO_ROOT)/onnx_export
MODE ?= rotated_fcos_pre_nms
ONNX_OUT ?= $(ARTIFACTS)/model.onnx
DEVICE ?= cpu
OPSET ?= 18
# Set SKIP_ORT=1 to skip ORT validation after ONNX export
SKIP_ORT ?= 0
# ONNX Runtime EP for preds: cpu | cuda | auto
ORT_DEVICE ?= auto
# Val preds / metrics; compare with repo-root make preds / make metrics (PyTorch)
DATA_ROOT ?=
DATA_SPLIT ?= val
VAL_DIR ?=
PREDICTIONS_DIR ?= $(ARTIFACTS)/predictions
METRICS_PRED_DIR ?=
EXPERIMENT ?=

.PHONY: help export-onnx export-verify export-demo export-zip export-preds export-metrics test

# checkpoint_best.pth → best_*.pth → newest checkpoint_epoch_*.pth
define FIND_CKPT
$(shell d="$(1)/checkpoints"; \
  if [ -f "$$d/checkpoint_best.pth" ]; then printf '%s\n' "$$d/checkpoint_best.pth"; \
  else ls -1t "$$d"/best_*.pth "$$d"/checkpoint_epoch_*.pth 2>/dev/null | head -1; fi)
endef

ifeq ($(strip $(EXPERIMENT)),)
  EXPORT_CONFIG := $(CONFIG)
  EXPORT_CKPT := $(CKPT)
else
  EXPORT_CONFIG := $(EXPERIMENT)/config.json
  EXPORT_CKPT := $(call FIND_CKPT,$(EXPERIMENT))
endif

## help: Show export Makefile targets (thin wrappers around odet export)
help:
	@echo "OrientedDet ONNX export — Makefile targets (odet export → onnx_export/)"
	@echo "===================================="
	@echo ""
	@echo "  make export-onnx     - odet export onnx → $(ONNX_OUT)"
	@echo "  make export-demo     - bundled plane image + NMS assertions (no oriented-det)"
	@echo "  make export-zip      - zip $(ARTIFACTS)/ (no __pycache__) → $(REPO_ROOT)/onnx_export.zip"
	@echo "  make export-verify   - ORT zeros smoke on that ONNX"
	@echo "  make export-preds    - odet export preds → $(PREDICTIONS_DIR)/<ts>/"
	@echo "  make export-metrics  - Offline mAP/PR on METRICS_PRED_DIR or newest ONNX predictions/"
	@echo "  make test            - pytest export/tests/"
	@echo ""
	@echo "Defaults:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CKPT=$(CKPT)"
	@echo "  MODE=$(MODE)"
	@echo "  HEIGHT=$(HEIGHT)  WIDTH=$(WIDTH)"
	@echo "  DEVICE=$(DEVICE)  ARTIFACTS=$(ARTIFACTS)"
	@echo ""
	@echo "From repo root:"
	@echo "  make export-onnx EXPERIMENT=runs/rotated_fcos/<timestamp>"
	@echo "  make export-onnx EXPERIMENT=runs/oriented_rcnn/<timestamp> MODE=oriented_rcnn_pre_nms"
	@echo "  make export-onnx EXPERIMENT=runs/rotated_faster_rcnn/<timestamp> MODE=faster_rcnn_pre_nms"

## export-onnx: PyTorch → ONNX + consumer sidecar copies
export-onnx:
	@if [ -z "$(EXPORT_CKPT)" ] || [ ! -f "$(EXPORT_CKPT)" ]; then \
		echo "Error: checkpoint not found: $(EXPORT_CKPT)"; \
		echo "Download Hub weights (odet pretrained download rotated_fcos_dota_le90_1x) or set EXPERIMENT=/CKPT="; \
		exit 1; \
	fi
	@if [ ! -f "$(EXPORT_CONFIG)" ]; then \
		echo "Error: config not found: $(EXPORT_CONFIG)"; \
		exit 1; \
	fi
	@mkdir -p "$(ARTIFACTS)"
	cd "$(REPO_ROOT)" && odet export onnx \
	  --config "$(EXPORT_CONFIG)" \
	  --checkpoint "$(EXPORT_CKPT)" \
	  --output "$(ONNX_OUT)" \
	  --height $(HEIGHT) --width $(WIDTH) \
	  --mode "$(MODE)" \
	  --device "$(DEVICE)" \
	  --opset $(OPSET) \
	  $(if $(filter 1 true TRUE yes YES,$(SKIP_ORT)),--skip-ort,)

## export-verify: ORT zeros smoke on ONNX_OUT
export-verify:
	cd "$(REPO_ROOT)" && odet export infer --onnx "$(ONNX_OUT)" --smoke

## export-demo: Bundled plane image + NMS assertions
export-demo:
	cd "$(REPO_ROOT)" && odet export demo --onnx "$(ONNX_OUT)"

## export-zip: Zip ARTIFACTS without Python cache
EXPORT_ZIP ?= $(REPO_ROOT)/onnx_export.zip
export-zip:
	@if [ ! -f "$(ONNX_OUT)" ]; then \
		echo "Error: $(ONNX_OUT) missing. Run: make export-onnx"; \
		exit 1; \
	fi
	cd "$(REPO_ROOT)" && python export/scripts/zip_bundle.py \
	  --dir "$(ARTIFACTS)" \
	  --output "$(EXPORT_ZIP)"

## export-preds: Val inference with ONNX + Python NMS
export-preds:
	@if [ ! -f "$(ONNX_OUT)" ]; then \
		echo "Error: $(ONNX_OUT) missing. Run: make export-onnx"; \
		exit 1; \
	fi
	@echo "ONNX export preds: CONFIG=$(EXPORT_CONFIG) ONNX=$(ONNX_OUT) ORT_DEVICE=$(ORT_DEVICE)"; \
	ROOT_ARGS=""; \
	if [ -n "$(DATA_ROOT)" ]; then ROOT_ARGS="--data-root $(DATA_ROOT)"; fi; \
	if [ -n "$(VAL_DIR)" ]; then ROOT_ARGS="$$ROOT_ARGS --val-dir $(VAL_DIR)"; fi; \
	TS=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p "$(PREDICTIONS_DIR)"; \
	cd "$(REPO_ROOT)" && odet export preds \
	  --config "$(EXPORT_CONFIG)" \
	  --onnx "$(ONNX_OUT)" \
	  --reference-checkpoint "$(EXPORT_CKPT)" \
	  --data-split "$(DATA_SPLIT)" \
	  --output-dir "$(PREDICTIONS_DIR)/$$TS" \
	  $$ROOT_ARGS \
	  --ort-device "$(ORT_DEVICE)"

## export-metrics: Offline mAP/PR from predictions.json (no inference)
export-metrics:
	@PRED="$(METRICS_PRED_DIR)"; \
	if [ -z "$$PRED" ]; then \
		if [ ! -d "$(PREDICTIONS_DIR)" ]; then \
			echo "Error: $(PREDICTIONS_DIR) missing. Run: make export-preds"; \
			exit 1; \
		fi; \
		PRED=$$(find "$(PREDICTIONS_DIR)" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1); \
	fi; \
	if [ -z "$$PRED" ] || [ ! -f "$$PRED/predictions.json" ]; then \
		echo "Error: predictions.json not found under $$PRED"; \
		exit 1; \
	fi; \
	echo "ONNX export metrics: $$PRED"; \
	cd "$(REPO_ROOT)" && odet preds --metrics-from-json "$$PRED"

## test: Run export unit tests
test:
	cd "$(REPO_ROOT)" && PYTHONPATH="$(REPO_ROOT)" pytest export/tests/
