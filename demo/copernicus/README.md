# Copernicus / Sentinel-2 demo

Sample tiles for zero-shot **ship** detection with the DOTA-pretrained Oriented R-CNN (`oriented_rcnn_dota_le90_1x`).

## Files

| File | Description |
|------|-------------|
| `T30NZM_*_2976_7936.png` | Example 1024×1024 Copernicus tile |
| `*_detections.png` / `*_detections.json` | Visualization and prediction outputs |

## Single tile (4× zoom, ships only)

```bash
python tools/image_demo.py \
  demo/copernicus/T30NZM_20260616T101021_TCI_10m_2976_7936.png \
  hf://oriented_rcnn_dota_le90_1x \
  --out-file demo/copernicus/T30NZM_20260616T101021_TCI_10m_2976_7936_detections.png \
  --score-thr 0.15 \
  --nms-thr 0.2 \
  --classes ship \
  --zoom 4 \
  --overlap-pixels 512 \
  --ignore-margin-pixels 256 \
  --json-per-image \
  --device mps
```

Writes `*_detections.png` and `*_detections.json` (boxes in original 1024×1024 coordinates).

## Batch: folder of tiles

Process every PNG/JPG in a directory with `image_demo.py` (one model load, outputs to `--out-dir`):

```bash
python tools/image_demo.py \
  "/path/to/S2C_..._SAFE_dataset 1/images" \
  hf://oriented_rcnn_dota_le90_1x \
  --out-dir "/path/to/S2C_..._SAFE_dataset 1/detections" \
  --score-thr 0.15 \
  --nms-thr 0.2 \
  --classes ship \
  --zoom 4 \
  --overlap-pixels 512 \
  --ignore-margin-pixels 256 \
  --json-per-image \
  --device mps
```

### Inference settings

| Setting | Value | Notes |
|---------|-------|-------|
| `--zoom` | 4 | 4× in-memory upscale; 64 windows per tile |
| `--overlap-pixels` | 512 | Larger than biggest ship footprint |
| `--ignore-margin-pixels` | 256 | Half overlap; dedup at tile seams |
| `--score-thr` | 0.15 | Post-decode filter |
| `--nms-thr` | 0.2 | Merge NMS after tiling |
| `--classes` | ship | Drop non-ship categories |

**Runtime:** ~1–2 min per tile on CPU at 4× (64 windows). On Apple Silicon use `--device mps`.

See also [tools/README.md](../../tools/README.md) (`image_demo.py` options).
