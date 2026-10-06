# ONNX export

Export a trained OrientedDet checkpoint to a **pre-NMS ONNX** graph plus Python preprocess / rotated NMS. Consumers run ONNX Runtime only — no PyTorch and no oriented-det.

The producer CLI is **`odet export`** (ONNX only; same as `python -m export`). Folder README: [`export/README.md`](https://github.com/DL4EO/oriented-det/blob/main/export/README.md). Sliding-window tiling and the Tile Geo Process HTTP service stay on the [Docker deploy](deploy.md) path in this release.

## Install

Producer (checkpoint → ONNX):

```bash
uv pip install -e ".[export]"
```

Consumer (infer / demo without this repo):

```bash
pip install -r export/requirements-runtime.txt
```

GPU ORT: install `onnxruntime-gpu[cuda,cudnn]` instead of `onnxruntime` (do not install both).

## Export a graph

Default is Rotated FCOS (`rotated_fcos_pre_nms`). Download Hub weights or point at a training run:

```bash
odet pretrained download rotated_fcos_dota_le90_1x
make export-onnx
```

From a run directory:

```bash
make export-onnx EXPERIMENT=runs/rotated_fcos/<timestamp>
make export-onnx EXPERIMENT=runs/oriented_rcnn/<timestamp> EXPORT_MODE=oriented_rcnn_pre_nms
make export-onnx EXPERIMENT=runs/rotated_faster_rcnn/<timestamp> EXPORT_MODE=faster_rcnn_pre_nms
make export-onnx EXPERIMENT=runs/rotated_rtmdet/<timestamp> EXPORT_MODE=rotated_rtmdet_pre_nms
```

Equivalent CLI (placeholder paths):

```bash
odet export onnx \
  --config /path/to/oriented-det/runs/rotated_fcos/<timestamp>/config.json \
  --checkpoint /path/to/oriented-det/runs/rotated_fcos/<timestamp>/checkpoints/checkpoint_best.pth \
  --output /path/to/oriented-det/onnx_export/model.onnx
```

Artifacts (gitignored except the folder README) land in `onnx_export/`:

| File | Role |
|------|------|
| `model.onnx` | Backbone + head + box decode; padded pre-NMS tensors |
| `model.export_meta.json` | Mean/std, class names, score floor, NMS IoU |
| `preprocess.py`, `nms.py`, `postprocess.py`, `runtime.py`, `ort_runtime.py` | Copies of the consumer stack |
| `demo.py`, `infer_onnx.py` | Copies of the infer scripts |

ONNX input is NCHW RGB, already mean/std-normalized, typically `[1, 3, 1024, 1024]`. Do not feed raw JPEGs into the graph.

## Infer

```bash
odet export infer --onnx ./onnx_export/model.onnx --smoke
odet export infer --onnx ./onnx_export/model.onnx --images /path/to/data/tiles --output ./onnx_export/predictions
make export-demo
```

Python:

```python
from PIL import Image
from export.runtime import detect_image

dets = detect_image(Image.open("tile.jpg"), "onnx_export/model.onnx")
# each det: cx, cy, w, h, angle_rad, score, label, class_name  (original image pixels)
```

From a copied `onnx_export/` folder only:

```bash
cd onnx_export
pip install -r requirements-runtime.txt
python demo.py
```

```python
from runtime import detect_image
dets = detect_image("tile.jpg", "model.onnx")
```

Val-split predictions in the same JSON shape as `odet preds` (then `make export-metrics` / `odet preds --metrics-from-json`):

```bash
odet export preds \
  --config /path/to/oriented-det/runs/rotated_fcos/<timestamp>/config.json \
  --onnx /path/to/oriented-det/onnx_export/model.onnx
```

Sliding-window tiling is out of scope for the ONNX path — use pre-tiled images (same as PyTorch `odet preds` on DOTA tiles). Preprocess is bilinear resize to a **fixed** canvas (default 1024×1024). That matches DOTA tiles, not `keep_ratio` + pad (HRSC / SSDD / HRSID).

## Modes

| `--mode` / `EXPORT_MODE` | Detector |
|--------------------------|----------|
| `rotated_fcos_pre_nms` (default) | Rotated FCOS |
| `rotated_rtmdet_pre_nms` | Rotated RTMDet |
| `oriented_rcnn_pre_nms` | Oriented R-CNN |
| `faster_rcnn_pre_nms` | Rotated Faster R-CNN |

Decode runs inside ONNX. Final rotated NMS is numpy (`export.nms`; optional `--nms-backend shapely`). See [export/PARITY.md](https://github.com/DL4EO/oriented-det/blob/main/export/PARITY.md).
