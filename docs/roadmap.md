# Roadmap

High-level plan for Oriented-Det after **v0.3** (four ResNet-FPN detectors, DOTA Task 1 zoo, HRSC Hub 3×, FAIR1M / SSDD / HRSID loaders, ONNX pre-NMS for FCOS / Oriented R-CNN / Faster R-CNN).

> **Maintainers:** a longer planning doc lives at `docs/roadmap-detailed.md` (gitignored, local only).

## Releases

| Version | Focus |
|---------|--------|
| **v0.1** | Shipped — Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet; DOTA; `odet train`; Hub |
| **v0.1.1** | Shipped — ProbIoU Faster R-CNN on Hub (DOTA zoo is now 1×: **74.42%** official Task 1) |
| **v0.2** | Shipped — **Rotated FCOS** (anchor-free single-stage); DOTA le90 Hub `rotated_fcos_dota_le90_1x` (73.07% official Task 1) |
| **v0.3** | Shipped — **HRSC2016** loader + Hub 3× zoo; **FAIR1M** loader + 1× recipes + tutorial; **SSDD** + **HRSID** native loaders + 1× recipes (**finetune DOTA 1× Hub; no SAR zoo**); **ONNX** pre-NMS restore (`python -m export`; DOTA 1024 tiles) |
| **v0.4.0** | Shipped — **Rotated RTMDet** (CSPNeXt-S + PAFPN); Hub `rotated_rtmdet_dota_le90_3x`, **77.60%** official Task 1 |
| **v0.4.1** | **Native YOLO-OBB** (already in the tree; COCO pretrain and DOTA training next) |
| **v0.5–v0.8** | **Swin-FPN** backbone; Oriented R-CNN + Swin-T on Hub; extend to FCOS / speed models |
| **v1.0** | Stable API, hosted docs, complete model zoo (accuracy / balanced / speed tiers) |

## Model tiers (target v1.0)

| Tier | Models |
|------|--------|
| **Accuracy** | Oriented R-CNN, Rotated Faster R-CNN (probiou) — ResNet50 and Swin-T |
| **Balanced** | Rotated FCOS |
| **Speed** | RTMDet-R, native Rotated YOLO-OBB |
| **Legacy** | Rotated RetinaNet (L1; MMRotate parity) |
| **Datasets** | DOTA, HRSC2016, FAIR1M, SSDD, HRSID |

## Closed ablations (not Hub)

- **RetinaNet ProbIoU 1×** — no zoo-worthy gain vs L1 (~63% vs 64% eval-val); keep L1 RetinaNet only
- Extra FRCNN angle / rIoU-aux recipes — did not beat the published ProbIoU 1.0 / 0.1 recipe
- **FCOS 1× ProbIoU aux** — 66.8% train-time mAP50 vs 76.5% for 1× KFIoU aux; recipe removed

## Ongoing

- **Ongoing:** Hosted MkDocs site and dataset tutorials
- **Ongoing:** Optional GPU / fused CUDA rotated IoU/NMS (profiling-driven; unblocked now that FCOS shipped)
- **Ongoing:** ONNX remaining scope — RetinaNet pre-NMS detect graph; `keep_ratio` + pad canvas (HRSC / SSDD / HRSID); sliding-window tiling. Three-detector DOTA-tile export shipped in v0.3 (`python -m export`; see [ONNX export](examples/export.md)).

## Out of scope (for now)

- Ultralytics wrapper or AGPL dependencies
- End-to-end DETR-style detectors in core `odet train`
- MMCV / MMDet as runtime dependencies
- RetinaNet ProbIoU Hub weights; FRCNN angle-fine-tune Hub twin without a clear eval-val win

See [Contributing](contributing.md) for areas where help is welcome.
