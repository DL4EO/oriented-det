# Roadmap

High-level plan for Oriented-Det after **v0.1** (geometry, IoU/NMS, DOTA, three ResNet-FPN detectors, config training, Hub weights).

> **Maintainers:** a longer planning doc lives at `docs/roadmap-detailed.md` (gitignored, local only).

## Releases

| Version | Focus |
|---------|--------|
| **v0.1** | Shipped — Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet; DOTA; `odet train`; Hub |
| **v0.1.1** | Shipped — ProbIoU Faster R-CNN 1×/3× on Hub (77.57% / 83.42% eval-val mAP50) |
| **v0.2** | **Rotated FCOS** — anchor-free single-stage; DOTA le90 1×/3× configs |
| **v0.3** | **HRSC2016** and **FAIR1M** dataset loaders; cross-dataset benchmarks |
| **v0.4** | Production speed tier: **RTMDet-R**, then **native YOLO-OBB** |
| **v0.5–v0.8** | **Swin-FPN** backbone; Oriented R-CNN + Swin-T on Hub; extend to FCOS / speed models |
| **v1.0** | Stable API, hosted docs, complete model zoo (accuracy / balanced / speed tiers) |

## Model tiers (target v1.0)

| Tier | Models |
|------|--------|
| **Accuracy** | Oriented R-CNN, Rotated Faster R-CNN (probiou) — ResNet50 and Swin-T |
| **Balanced** | Rotated FCOS |
| **Speed** | RTMDet-R, native Rotated YOLO-OBB |
| **Datasets** | DOTA, HRSC2016, FAIR1M |

## Ongoing

- Hosted MkDocs site and dataset tutorials
- Optional **fused CUDA** rotated IoU/NMS kernels (after single-stage models land)
- Export parity for new detectors (ONNX head wrappers + postprocess)

## Out of scope (for now)

- Ultralytics wrapper or AGPL dependencies
- End-to-end DETR-style detectors in core `odet train`
- MMCV / MMDet as runtime dependencies

See [Contributing](contributing.md) for areas where help is welcome.
