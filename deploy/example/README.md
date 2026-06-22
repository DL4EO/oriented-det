# Deploy example (DOTA smoke)

Minimal Sanic Tile Geo Process app for trying **oriented-det** inference after a DOTA training run. ## Publish

From the framework repo root, after `odet train` on a DOTA config:

```bash
# Copy experiment config + best checkpoint into deploy/example/app/
cp runs/oriented_rcnn/<run_id>/config.json deploy/example/app/config.json
cp runs/oriented_rcnn/<run_id>/checkpoints/best_*.pth deploy/example/app/weights/model.pth

docker build -f deploy/example/Dockerfile -t odet-example:latest .
docker run --rm -p 8080:8080 --gpus all odet-example:latest
```

Weights and `config.json` are gitignored; do not commit customer or large artifacts here.
