# TensorFlow export (Phase 1)

This folder contains **PyTorch → ONNX → Keras detect bundle** tooling. The Faster R-CNN path wraps ONNX Runtime + Python rotated NMS inside a Keras model (`keras_model.keras`). It does **not** reimplement training or oriented NMS as a pure TensorFlow graph.

Optional **onnx2tf** conversion (experimental SavedModel / TFLite) is documented below for backbone / RetinaNet heads only.

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
| [Makefile](Makefile) | Thin in-repo wrappers around `odet` (`make export-tf`, …). Not required for pip installs. |
| [scripts/export_tf.py](scripts/export_tf.py) | `odet export-tf` orchestrator (ONNX + detect bundle → `./odet_export`). |
| [scripts/save_predictions_tf.py](scripts/save_predictions_tf.py) | Val inference via Keras bundle → `./odet_export/predictions/<ts>/`. |
| [scripts/](scripts/README.md) | CLI entrypoints. |
| [tests/](tests/README.md) | Pytest smoke tests (wrappers; optional ONNX/TF if deps installed). |

## Install

**Pip (recommended for consumers):**

```bash
pip install "oriented-det[export]"
odet pretrained download rotated_faster_rcnn_dota_le90_1x
odet export-tf \
  --config path/to/config.json \
  --checkpoint path/to/model.pth \
  --output-dir ./odet_export
# Oriented R-CNN: add --mode oriented_rcnn_pre_nms
```

No source checkout or `make` required. Artifacts land under `./odet_export/` (`pre_nms.onnx`, `detect/keras_model.keras`, `detect/model.onnx`).

**Editable checkout:**

```bash
cd /path/to/oriented-det
uv pip install -e ".[export]"
# or: uv pip install -e . && uv pip install -r export/requirements-export.txt
```

TensorFlow and `onnxruntime-gpu[cuda,cudnn]` wheels are large; keep this environment separate from minimal CPU-only dev envs if you prefer. ORT still supports CPU (`CPUExecutionProvider`); use `ORIENTED_DET_ORT_DEVICE=cuda` / `--ort-device cuda` for GPU.

Download Hub weights if needed:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_1x
```

## 1) Export ONNX

**Backbone only** (any model with a `backbone`):

```bash
odet export-onnx \
  --config configs/rotated_faster_rcnn/dota_le90_1x.json \
  --checkpoint pretrained/rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.pth \
  --output ./odet_export/model_backbone.onnx \
  --height 1024 --width 1024 \
  --mode backbone
```

**Rotated RetinaNet — backbone + cls/reg heads** (no decode/NMS in graph):

```bash
odet export-onnx \
  --config configs/rotated_retinanet/dota_le90_1x.json \
  --checkpoint path/to/model.pth \
  --output ./odet_export/retinanet_heads.onnx \
  --height 1024 --width 1024 \
  --mode retinanet_heads
```

Requires `model_type` in config to match a `RotatedRetinaNet` checkpoint.

**Rotated Faster R-CNN — full detect (pre-NMS ONNX + Keras bundle with exact NMS):**

```bash
odet export-tf \
  --config configs/rotated_faster_rcnn/dota_le90_1x.json \
  --checkpoint pretrained/rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.pth \
  --output-dir ./odet_export \
  --mode faster_rcnn_pre_nms

odet export-detect \
  --meta ./odet_export/pre_nms.export_meta.json \
  --onnx ./odet_export/pre_nms.onnx \
  --output ./odet_export/detect

# Smoke (optional):
python -m export.scripts.predict_savedmodel \
  --saved-model ./odet_export/detect \
  --height 1024 --width 1024
```

Produces `keras_model.keras` + `model.onnx` + `export_meta.json` under the detect directory.
The ONNX file is copied into the detect bundle as `model.onnx` (relocatable; load with
`load_keras_detect_model`). Load in Python:

```python
from export.tf_serving_model import load_keras_detect_model
model = load_keras_detect_model("./odet_export/detect")
```

In a git checkout, Makefile targets wrap the same `odet` commands:

```bash
cd export && make export-tf   # uses repo configs/ + pretrained/ defaults
odet export-preds --config ... --detect-dir ./odet_export/detect
```

Defaults for `make` (`make help` for overrides):

- `CONFIG=configs/rotated_faster_rcnn/dota_le90_1x.json`
- `CKPT=pretrained/rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.pth`
- `MODE=faster_rcnn_pre_nms`
- `ARTIFACTS=export/artifacts` → `pre_nms.onnx` + `detect/`

For a trained experiment: `make export-tf CONFIG=runs/rotated_faster_rcnn/<ts>/config.json CKPT=...`.  
For a local deploy smoke app, copy config + weights into `deploy/example/app/` (see [`deploy/example/README.md`](../deploy/example/README.md)) and pass those paths explicitly.

**Oriented R-CNN — full detect (same Keras bundle contract):**

```bash
# Download weights if needed:
# odet pretrained download oriented_rcnn_dota_le90_1x

odet export-tf \
  --config configs/oriented_rcnn/dota_le90_1x.json \
  --checkpoint pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.pth \
  --mode oriented_rcnn_pre_nms \
  --output-dir ./odet_export_oriented

# Or in-repo Makefile:
cd export && make export-tf \
  MODE=oriented_rcnn_pre_nms \
  CONFIG=configs/oriented_rcnn/dota_le90_1x.json \
  CKPT=pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.pth \
  ARTIFACTS=$(pwd)/artifacts/oriented_rcnn
```

(`make` runs `odet` from the repo root; prefer paths relative to the repo root.)
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
# Recommended CUDA eval: ORT owns the GPU(s); TensorFlow is forced onto CPU.
odet export-preds --ort-device cuda --config … --detect-dir ./odet_export/detect …
# or:
cd export && make preds ORT_DEVICE=cuda
make predict ORT_DEVICE=cuda
make preds ORT_DEVICE=auto    # CUDA EP if available, else CPU
```

CLI: `--ort-device cpu|cuda|auto` on `save_predictions_tf.py` / `predict_savedmodel.py`.  
Env: `ORIENTED_DET_ORT_DEVICE=cuda` (fallback). The `[export]` extra installs `onnxruntime-gpu[cuda,cudnn]` (CPU EP still works).

When `ort-device` resolves to CUDA, oriented-det calls `tf.config.set_visible_devices([], "GPU")` **before** loading the Keras bundle so TF does not claim A100s and starve ORT (symptoms: `/roi_align/Transpose_*` BFC OOM ~287 MB that should fit). Do not import `tensorflow` / `export.tf_serving_model` before `configure_ort_device(...)`.

**Realistic CUDA vs CPU expectations (Oriented / Faster R-CNN detect bundle):**

| Stage | Typical cost (1024², P≈6000) | Device |
|-------|------------------------------|--------|
| ORT pre-NMS graph (backbone→ROI→decode) | ~40–80 ms on A100; ~1–1.5 s on CPU | ORT CUDA EP / CPU EP |
| Final rotated NMS + score filter | Dominates wall clock if `production.final_nms_use_cpu: true` and thousands of pre-NMS boxes | PyTorch CPU (exact polygon) or CUDA if `final_nms_use_cpu: false` |

So **`nvidia-smi` at ~0% util with ~5 GB resident is expected** when exact CPU NMS runs for seconds after a short ORT CUDA burst — the session lives on GPU, but compute is idle during NMS. The “18 Memcpy nodes” warning is from a few `Atan` angle-wrap nodes (no CUDA `Atan` in opset 17); **GridSample / Conv / ScatterND still run on CUDA**. Those copies are negligible vs NMS.

Wall-clock tips:

- Keep `ORT_DEVICE=cuda` for the ORT core (large win vs CPU ORT alone).
- Production score floor is applied **before** NMS (parity-safe for greedy NMS) so exact CPU NMS is not O(n²) on every sub-threshold candidate.
- For max throughput (approximate IoU NMS): set `production.final_nms_use_cpu: false` and re-export / use a bundle with that meta — NMS can run on CUDA via `oriented_nms_gpu`.
- Use `ORT_DEVICE=cpu` only when debugging or when CUDA EP is unavailable; it will not fix NMS cost.

Optional CUDA knobs:

| Env | Meaning |
|-----|---------|
| `ORIENTED_DET_ORT_CUDA_DEVICE_ID` | ORT GPU index (default `0`) |
| `ORIENTED_DET_ORT_GPU_MEM_LIMIT` | Soft GPU mem cap in bytes for the CUDA EP |
| `ORIENTED_DET_ORT_CUDNN_CONV_ALGO` | cuDNN algo search (`HEURISTIC` default; `DEFAULT` / `EXHAUSTIVE`) |

Alternative two-GPU layout (manual): run TF with empty visible devices via the above, or set `CUDA_VISIBLE_DEVICES` only if you understand both TF and ORT see the same mask.
## 2) Convert to SavedModel (experimental onnx2tf)

For backbone / RetinaNet heads graphs (full Faster R-CNN detect often fails on ScatterND):

```bash
python export/scripts/onnx_to_savedmodel.py \
  --onnx export/artifacts/retinanet_heads.onnx \
  --output export/artifacts/saved_model
```

This shells to **`onnx2tf`** (must be on `PATH` from the pip install above).

## 3) Run SavedModel / detect bundle (smoke)

```bash
# Keras detect bundle (primary Faster R-CNN path)
python export/scripts/predict_savedmodel.py \
  --saved-model export/artifacts/faster_rcnn_detect

# or experimental onnx2tf SavedModel
python export/scripts/predict_savedmodel.py --saved-model export/artifacts/saved_model
```

## 4) Optional TFLite

```bash
python export/scripts/to_tflite.py \
  --saved-model export/artifacts/saved_model \
  --output export/artifacts/model.tflite
```

Full detect TFLite is not supported while NMS remains in Python (see [PARITY.md](PARITY.md)).

## Contract summary

- **Input:** `float32` NCHW RGB in **[0, 1]** (same convention as PyTorch training/inference in this repo).
- **Fixed H, W** at export unless you pass `--dynamic-batch` (batch axis only; **not** allowed for `faster_rcnn_pre_nms` / `oriented_rcnn_pre_nms`).
- **NMS:** not in ONNX; applied in the Keras detect bundle — see [PARITY.md](PARITY.md).
- **Detect bundle files:** `keras_model.keras`, `model.onnx`, `export_meta.json`.

## Troubleshooting

**ONNX Runtime `Expand` / `invalid expand shape` on val images:** Older exports padded pre-NMS tensors with slice assignment, which ONNX lowers to `Expand` and fails when the dynamic ROI candidate count is not exactly `0`, `1`, or `max_pre_nms_candidates`. Re-export with current `oriented_det` (`pad_pre_nms_detections` uses `torch.cat` zero-padding). Confirm `max_pre_nms_candidates` in `.export_meta.json` matches `production.rpn_post_nms_top_n` (often 6000 on Hub DOTA configs).

**ONNX Runtime OOM on `/roi_align/Expand` (~hundreds of GB):** Oriented R-CNN’s old ONNX ROI path did `feature.expand(N, …)` before `grid_sample`, which ORT materializes as `N×C×H×W` (e.g. 6000×256×256×256 at 1024²). Current `oriented_roi_align` packs ROI grids and keeps feature batch=1. Re-export; `SKIP_ORT=1` is not a fix.

**ONNX Runtime `/roi_align/Reshape_*` size mismatch on real tiles:** Older exports traced proposal padding with `int(shape[0])` control flow. When the zeros dummy already had exactly `max_pre_nms_candidates` RPN keeps, padding was dropped from the graph; real images with fewer keeps then reshaped a size-`P` constant with dynamic `N<P`. Re-export with current `oriented_det` (always `cat([proposals[:k], zeros(k)])[:k]`). Dummy ORT smoke alone is not enough — validate on a non-zero tile.

**ONNX Runtime CUDA OOM on `/roi_align/Transpose_*` (~287 MB) while CPU works:** TensorFlow eagerly claimed all GPUs in the same process before ORT. Use `--ort-device cuda` / `ORT_DEVICE=cuda` with current oriented-det (hides TF GPUs). Confirm logs no longer show TF creating `GPU:0..N` before inference. Fallback: `--ort-device cpu`.

**ORT_DEVICE=cuda but ~0% GPU util / ~same wall time as CPU:** Usually **not** a full-graph CPU fallback. ORT CUDA runs the detect graph in tens of ms; `production.final_nms_use_cpu: true` then spends seconds on exact polygon NMS. Memcpy logs around `Atan` are small host copies — GridSample stays on CUDA. Compare ORT-only vs full bundle timing; current builds pre-filter by `production.score_threshold` before NMS. For approximate fast NMS set `final_nms_use_cpu: false`.

## Artifacts directory

Write ONNX and detect-bundle outputs under `export/artifacts/` (gitignored) or any path you pass to `--output`.
