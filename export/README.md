# TensorFlow export (Phase 1)

This folder contains **PyTorch → ONNX → TensorFlow SavedModel** tooling. It does **not** reimplement training or oriented NMS in TensorFlow.

## Layout

| Path | Purpose |
|------|---------|
| [contract.json](contract.json) | Machine-readable export contract (modes, inputs, what is out-of-graph). |
| [PARITY.md](PARITY.md) | What matches PyTorch vs what is split across graphs. |
| [requirements-export.txt](requirements-export.txt) | Pins for ONNX / onnx2tf / TensorFlow (or `uv pip install -e ".[export]"`). |
| [fixtures/](fixtures/README.md) | Optional golden tensors / tiny images for stricter regression (not used by default CI). |
| [wrappers.py](wrappers.py) | `nn.Module` wrappers that expose tensor-only forwards for ONNX. |
| [postprocess.py](postprocess.py) | Exact rotated NMS + production score filter for the detect bundle. |
| [tf_serving_model.py](tf_serving_model.py) | Keras `FasterRCNNDetectLayer` (ORT core + NMS). |
| [Makefile](Makefile) | `make export-tf`, `preds`, `metrics`, `eval-tf`, `predict`, `test`. |
| [scripts/save_predictions_tf.py](scripts/save_predictions_tf.py) | Val inference via Keras bundle → `artifacts/predictions/<ts>/predictions.json`. |
| [scripts/](scripts/README.md) | CLI entrypoints. |
| [tests/](tests/README.md) | Pytest smoke tests (wrappers; optional ONNX/TF if deps installed). |

## Install

```bash
cd /path/to/oriented-det
uv pip install -e .
uv pip install -r export/requirements-export.txt
# or: uv pip install -e ".[export]"
```

TensorFlow wheels are large; keep this environment separate from minimal CPU-only dev envs if you prefer.

## 1) Export ONNX

**Backbone only** (any model with a `backbone`):

```bash
python export/scripts/export_onnx.py \
  --config deploy/app/config.json \
  --checkpoint path/to/model.pth \
  --output export/artifacts/model_backbone.onnx \
  --height 1024 --width 1024 \
  --mode backbone
```

**Rotated RetinaNet — backbone + cls/reg heads** (no decode/NMS in graph):

```bash
python export/scripts/export_onnx.py \
  --config path/to/config.json \
  --checkpoint path/to/model.pth \
  --output export/artifacts/retinanet_heads.onnx \
  --height 1024 --width 1024 \
  --mode retinanet_heads
```

Requires `model_type` in config to match a `RotatedRetinaNet` checkpoint.

**Rotated Faster R-CNN — full detect (pre-NMS ONNX + Keras bundle with exact NMS):**

```bash
python export/scripts/export_onnx.py \
  --config deploy/app/config.json \
  --checkpoint deploy/app/weights/model.pth \
  --output export/artifacts/faster_rcnn_pre_nms.onnx \
  --height 1024 --width 1024 \
  --mode faster_rcnn_pre_nms

python export/scripts/build_faster_rcnn_savedmodel.py \
  --meta export/artifacts/faster_rcnn_pre_nms.export_meta.json \
  --onnx export/artifacts/faster_rcnn_pre_nms.onnx \
  --output export/artifacts/faster_rcnn_detect

python export/scripts/predict_savedmodel.py \
  --saved-model export/artifacts/faster_rcnn_detect \
  --height 1024 --width 1024
```

Produces `keras_model.keras` + `export_meta.json` under the output directory. Load in Python:

```python
from export.tf_serving_model import load_keras_detect_model
model = load_keras_detect_model("export/artifacts/faster_rcnn_detect/keras_model.keras")
```

Or from this directory:

```bash
cd export && make export-tf
```

Uses `deploy/app/config.json` and `deploy/app/weights/model.pth` by default (`make help` for overrides).

### Compare PyTorch vs TF metrics

Run the same val split with both backends, then compare `predictions.json` metadata / `analysis_*.json`:

```bash
# PyTorch (repo root)
make preds && make metrics

# TensorFlow export (this folder)
cd export && make export-tf && make eval-tf
# or: make preds && make metrics
```

TF predictions land under `export/artifacts/predictions/<timestamp>/` (not repo-root `predictions/`).  
`make metrics` there reuses `tools/save_predictions.py --metrics-from-json` so mAP/PR definitions match.  
Inference uses the same resize + normalize as PyTorch `make preds`; NMS/score thresholds come from `production.*` in config.  
Images larger than the model canvas are not tiled in the TF preds path (use pre-tiled val tiles).

**ONNX Runtime device** (exported ONNX core only; post-NMS stays CPU/PyTorch):

```bash
cd export
make predict ORT_DEVICE=cuda
make preds ORT_DEVICE=auto    # CUDA if onnxruntime-gpu is installed, else CPU
```

CLI: `--ort-device cpu|cuda|auto` on `save_predictions_tf.py` / `predict_savedmodel.py`.  
Env fallback: `ORIENTED_DET_ORT_DEVICE=cuda`. Requires `onnxruntime-gpu` for CUDA.

Optional: `onnx_to_savedmodel.py` for TFLite experiments (may not produce a TF SavedModel for this graph).

## 2) Convert to SavedModel

```bash
python export/scripts/onnx_to_savedmodel.py \
  --onnx export/artifacts/retinanet_heads.onnx \
  --output export/artifacts/saved_model
```

This shells to **`onnx2tf`** (must be on `PATH` from the pip install above).

## 3) Run SavedModel (smoke)

```bash
python export/scripts/predict_savedmodel.py --saved-model export/artifacts/saved_model
```

## 4) Optional TFLite

```bash
python export/scripts/to_tflite.py \
  --saved-model export/artifacts/saved_model \
  --output export/artifacts/model.tflite
```

## Contract summary

- **Input:** `float32` NCHW RGB in **[0, 1]** (same convention as PyTorch training/inference in this repo).
- **Fixed H, W** at export unless you pass `--dynamic-batch` (batch axis only).
- **NMS:** not exported by default; see [PARITY.md](PARITY.md).

## Troubleshooting

**ONNX Runtime `Expand` / `invalid expand shape` on val images:** Older exports padded pre-NMS tensors with slice assignment, which ONNX lowers to `Expand` and fails when the dynamic ROI candidate count is not exactly `0`, `1`, or `max_pre_nms_candidates`. Re-export with current `oriented_det` (`pad_pre_nms_detections` uses `torch.cat` zero-padding). Confirm `max_pre_nms_candidates` in `.export_meta.json` matches `production.rpn_post_nms_top_n` (typically 8000 for deploy).

## Artifacts directory

Write ONNX and SavedModel outputs under `export/artifacts/` (gitignored) or any path you pass to `--output`.
