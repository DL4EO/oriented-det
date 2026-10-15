# Deploy example (DOTA smoke)

Minimal Sanic Tile Geo Process app for **oriented-det** inference. Bake a DOTA checkpoint (Hub slug or a training run), build the NVIDIA CUDA image, `POST /api/v1/process` with a base64 tile, get GeoJSON polygons.

Walkthrough: [Deploy oriented-det in Docker](https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/). Docs: [examples/deploy](https://dl4eo.github.io/oriented-det/examples/deploy/).

## Publish

From the repository root. Weights and `config.json` are gitignored; do not commit customer or Hub artifacts here.

### From a Hub slug

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

### From a training run

```bash
cp runs/oriented_rcnn/<run_id>/config.json deploy/example/app/config.json
cp runs/oriented_rcnn/<run_id>/checkpoints/checkpoint_best.pth deploy/example/app/weights/model.pth
```

Then the same `generate_description.py` command as above.

## Build and run

Build context is the **repo root** (the Dockerfile copies `oriented_det/` and `pyproject.toml`):

```bash
docker build -f deploy/example/Dockerfile -t odet-example:latest .
docker run --rm -p 8080:8080 --gpus all odet-example:latest
```

`--gpus all` is the intended NVIDIA path (`nvidia/cuda:12.1.0-runtime` + torch cu121). Without a GPU the process falls back to CPU.

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health   # OK
# OpenAPI UI: http://127.0.0.1:8080/swagger/
```

`POST /api/v1/process` body: `{"resolution": <meters per pixel>, "tiles": [<base64 JPEG or PNG>]}`. Response: GeoJSON FeatureCollection (category, confidence, length/width in meters). One request at a time (HTTP 429 if busy).

Inference knobs come from `config.production.*` (score floor, NMS, sliding-window overlap, canvas flags). DOTA Hub Oriented R-CNN 1× uses score 0.55 and NMS 0.1.
