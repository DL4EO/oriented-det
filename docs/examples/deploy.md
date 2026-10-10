# Docker deploy (Tile Geo Process)

Publish an oriented-det checkpoint as a **Sanic** service: base64 JPEG/PNG tiles in, **GeoJSON** polygons out. The example lives in [`deploy/example/`](https://github.com/DL4EO/oriented-det/tree/main/deploy/example/). Folder README: [`deploy/example/README.md`](https://github.com/DL4EO/oriented-det/blob/main/deploy/example/README.md).

This path still runs **PyTorch** inside an NVIDIA CUDA 12.1 image. For inference without PyTorch, see [ONNX export](export.md).

## Bake a checkpoint

Weights are gitignored. From the repo root, copy a Hub sidecar + `.pth` (or a `runs/` best checkpoint) into `deploy/example/app/`, then write `description.json`:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x
mkdir -p deploy/example/app/weights
cp pretrained/oriented_rcnn_*_dota_le90_1x*.json deploy/example/app/config.json
cp pretrained/oriented_rcnn_*_dota_le90_1x*.pth  deploy/example/app/weights/model.pth
python deploy/scripts/generate_description.py \
  --config deploy/example/app/config.json \
  --out deploy/example/app/description.json \
  --deploy-version 0.4.0
```

## Build and run

```bash
docker build -f deploy/example/Dockerfile -t odet-example:latest .
docker run --rm -p 8080:8080 --gpus all odet-example:latest
```

| Verb | Path | Role |
|------|------|------|
| GET | `/api/v1/health` | liveness |
| GET | `/api/v1/describe` | class list + metadata |
| GET | `/api/v1/openapi` | Tile Geo Process YAML |
| POST | `/api/v1/process` | `{resolution, tiles[]}` → FeatureCollection |
| GET | `/swagger/` | OpenAPI UI |

`resolution` is meters per pixel (scales length / width on each feature). A second POST while inference is running returns 429.

`config.production.*` sets the score floor, final NMS, sliding-window overlap, and canvas flags (`stick_to_model_canvas` default true → 1024 tiles; larger images tile). Walkthrough with overlays: [Deploy oriented-det in Docker](https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/).
