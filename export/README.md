# ONNX export

The export tool writes an **ONNX** graph plus sidecar metadata so a consumer can run the same **preprocess** and **postprocess** as training/eval.

**Consumers do not need oriented-det or PyTorch.** Infer and demo need numpy, Pillow, and ONNX Runtime:

```bash
pip install -r export/requirements-runtime.txt
python -m export demo
python -m export infer --onnx ./onnx_export/model.onnx --images ./tiles --output ./onnx_export/predictions
```

Exporting a new graph from a checkpoint still needs oriented-det. From the repo root:

```bash
uv pip install -e ".[export]"
odet export onnx \
  --config runs/rotated_fcos/<ts>/config.json \
  --checkpoint runs/rotated_fcos/<ts>/checkpoints/checkpoint_best.pth \
  --output ./onnx_export/model.onnx
```

`python -m export` is the same CLI. Default mode is `rotated_fcos_pre_nms`. Artifacts land in [`../onnx_export/`](../onnx_export/README.md). TensorFlow / SavedModel are not supported.

Makefile wrappers (same defaults: FCOS DOTA 1× Hub weights):

```bash
make export-onnx            # pretrained/rotated_fcos_*_dota_le90_1x → onnx_export/model.onnx
make export-onnx EXPERIMENT=runs/rotated_fcos/<timestamp>
make export-onnx EXPERIMENT=runs/oriented_rcnn/<timestamp> EXPORT_MODE=oriented_rcnn_pre_nms
make export-onnx EXPERIMENT=runs/rotated_faster_rcnn/<timestamp> EXPORT_MODE=faster_rcnn_pre_nms
make export-onnx EXPERIMENT=runs/rotated_rtmdet/<timestamp> EXPORT_MODE=rotated_rtmdet_pre_nms
make export-demo            # plane image + NMS assertions → onnx_export/demo/
make export-zip             # zip onnx_export/ (no __pycache__) → onnx_export.zip
```

Download Hub weights first if you use the default checkpoint: `odet pretrained download rotated_fcos_dota_le90_1x`.

## Deliverable

| File | Role |
|------|------|
| `model.onnx` | Backbone + head + box decode. Outputs padded pre-NMS tensors. |
| `model.export_meta.json` | Input spec, `normalize_mean` / `std`, class names, NMS / score thresholds. |

Python (this package; **no oriented-det**). The same files are copied into `onnx_export/` at export:

| Module | Role |
|--------|------|
| [preprocess.py](preprocess.py) | RGB image → NCHW float32 `(x/255 - mean) / std` |
| [nms.py](nms.py) | Rotated IoU + greedy NMS |
| [postprocess.py](postprocess.py) | Score floor + NMS via `nms` |
| [runtime.py](runtime.py) | `detect_image(...)` glue (ORT) |
| [ort_runtime.py](ort_runtime.py) | ONNX Runtime session / device |

ONNX input: NCHW **RGB**, already mean/std-normalized, shape `[1, 3, 1024, 1024]`.  
ONNX outputs: `pre_nms_boxes`, `pre_nms_scores`, `pre_nms_labels`, `pre_nms_count`.

```python
from PIL import Image
from export.runtime import detect_image

dets = detect_image(Image.open("tile.jpg"), "onnx_export/model.onnx")
# each det: cx, cy, w, h, angle_rad, score, label, class_name  (original image pixels)
```

From the copied bundle only (no repo, no oriented-det):

```bash
cd onnx_export
pip install -r requirements-runtime.txt
python demo.py
python infer_onnx.py --onnx model.onnx --smoke
```

```python
from runtime import detect_image
dets = detect_image("tile.jpg", "model.onnx")
```

## Modes

| `--mode` | Models | Notes |
|----------|--------|--------|
| `rotated_fcos_pre_nms` | Rotated FCOS | Default. Decode in ONNX; NMS in Python. |
| `rotated_rtmdet_pre_nms` | Rotated RTMDet | Decode in ONNX; NMS in Python. |
| `oriented_rcnn_pre_nms` | Oriented R-CNN | ROI decode in ONNX; NMS in Python. |
| `faster_rcnn_pre_nms` | Rotated Faster R-CNN | ROI decode in ONNX; NMS in Python. |
| `rotated_fcos_heads` / `retinanet_heads` / `backbone` | Tensor-only subgraphs | Full detections need decode + NMS elsewhere. |

User-facing walkthrough: [docs/examples/export.md](../docs/examples/export.md). Parity: [PARITY.md](PARITY.md).

## Layout

| Path | Purpose |
|------|---------|
| [contract.json](contract.json) | ONNX I/O + preprocess / postprocess contract |
| [wrappers.py](wrappers.py) | PyTorch modules traced to ONNX (producer) |
| [preprocess.py](preprocess.py) | Image → ONNX tensor |
| [nms.py](nms.py) | Rotated IoU / NMS (source; copied to `onnx_export/` on export) |
| [postprocess.py](postprocess.py) | Pre-NMS tensors → detections |
| [runtime.py](runtime.py) | ORT inference helper |
| [ort_runtime.py](ort_runtime.py) | ORT session cache / device |
| [cli.py](cli.py) | `odet export` / `python -m export` (`onnx`, `infer`, `demo`, `preds`) |
| [scripts/](scripts/README.md) | CLI implementations |
| [demo/](demo/README.md) | Bundled plane image for `python -m export demo` |
| [`../onnx_export/`](../onnx_export/README.md) | ONNX artifacts + copies of the consumer Python stack |
| [requirements-runtime.txt](requirements-runtime.txt) | Infer/demo deps (no oriented-det) |
| [requirements-export.txt](requirements-export.txt) | Checkpoint → ONNX deps |
| [tests/](tests/README.md) | Pytest |
| [PARITY.md](PARITY.md) | What matches PyTorch inference |
