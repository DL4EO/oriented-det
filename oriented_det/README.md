# oriented_det

PyTorch package for rotated (oriented) object detection in aerial and satellite imagery.

## Subpackages

- **geometry** — Polygons, `RBox`, `QBox`, transforms (rbox ↔ qbox ↔ polygon).
- **ops** — Rotated IoU, oriented NMS; CPU and GPU implementations.
- **data** — DOTA loader, tiling, augmentation, oriented mAP evaluation; Airbus Playground support.
- **models** — Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet; backbones (ResNet + FPN), bbox coders, MMRotate weight loading.
- **train** — Training engine, checkpointing, config loading, profiler.
- **utils** — Config utilities, logging, visualization.

Docs: [User guide](../docs/user-guide/) · [Configuration](../docs/user-guide/configuration.md) (JSON recipes) · [API](../docs/api/) (MkDocs + docstrings).
