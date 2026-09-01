# Roadmap

High-level plan for Oriented-Det after **v0.2** (geometry, IoU/NMS, DOTA, four ResNet-FPN detectors including Rotated FCOS, config training, Hub weights).

> **Maintainers:** a longer planning doc lives at `docs/roadmap-detailed.md` (gitignored, local only).

## Releases

| Version | Focus |
|---------|--------|
| **v0.1** | Shipped — Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet; DOTA; `odet train`; Hub |
| **v0.1.1** | Shipped — ProbIoU Faster R-CNN 1×/3× on Hub (77.57% / 83.42% eval-val mAP50) |
| **v0.2** | Shipped — **Rotated FCOS** (anchor-free single-stage); DOTA le90 1×/3×; Hub `rotated_fcos_dota_le90_3x` (82.32% eval-val) |
| **v0.3** | **HRSC2016** loader + Hub 3× zoo (shipped); **FAIR1M** remaining |
| **v0.4** | Production speed tier: **RTMDet-R**, then **native YOLO-OBB** |
| **v0.5–v0.8** | **Swin-FPN** backbone; Oriented R-CNN + Swin-T on Hub; extend to FCOS / speed models |
| **v1.0** | Stable API, hosted docs, complete model zoo (accuracy / balanced / speed tiers) |

## Model tiers (target v1.0)

| Tier | Models |
|------|--------|
| **Accuracy** | Oriented R-CNN, Rotated Faster R-CNN (probiou) — ResNet50 and Swin-T |
| **Balanced** | Rotated FCOS |
| **Speed** | RTMDet-R, native Rotated YOLO-OBB |
| **Legacy** | Rotated RetinaNet (L1; MMRotate parity) |
| **Datasets** | DOTA, HRSC2016, FAIR1M |

## Closed ablations (not Hub)

- **RetinaNet ProbIoU 1×** — no zoo-worthy gain vs L1 (~63% vs 64% eval-val); keep L1 RetinaNet only
- Extra FRCNN angle / rIoU-aux recipes — did not beat the published ProbIoU 1.0 / 0.1 recipe
- **FCOS 1× ProbIoU aux** — 66.8% train-time mAP50 vs 76.5% for 1× KFIoU aux; recipe removed

## Ongoing

- Hosted MkDocs site and dataset tutorials
- Optional GPU / fused CUDA rotated IoU/NMS (profiling-driven; unblocked now that FCOS shipped)
- Export parity for new detectors (ONNX head wrappers + postprocess)

## Out of scope (for now)

- Ultralytics wrapper or AGPL dependencies
- End-to-end DETR-style detectors in core `odet train`
- MMCV / MMDet as runtime dependencies
- RetinaNet ProbIoU Hub weights; FRCNN angle-fine-tune Hub twin without a clear eval-val win

See [Contributing](contributing.md) for areas where help is welcome.
