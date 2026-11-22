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
| **v0.4.1** | Shipped — **Citation and discoverability** (`CITATION.cff`, shared oriented/rotated + aerial/satellite copy) |
| **v0.4.2** | Shipped — **Native YOLO-OBB**; Hub `yolo_obb_dota_le90_12x` **77.80%** official Task 1 (COCO trunk `yolo_obb_coco_300e`) |
| **v0.5** | **Swin-T + FPN** on Oriented R-CNN: DOTA le90 1×, then 3× on Hub |
| **v0.5.1** | **Swin-T Rotated Faster R-CNN** (ProbIoU), same FPN |
| **v0.5.2** | **Swin-T Rotated FCOS**, only if 1024-tile latency stays on the accuracy tier |
| **v1.0** | Stable API, hosted docs, complete model zoo (accuracy / balanced / speed tiers) |
| **v1.1** | **Oriented DINO** in this repo. DOTA only. Hub only if it beats Swin Oriented R-CNN on official Task 1 |

**v0.5–v0.5.2 detail.** Swin-T is a backbone swap onto the shared FPN. Stages at strides 4, 8, 16, and 32 feed that FPN. New work is ImageNet initialization and FPN lateral convolutions for channels **96 / 192 / 384 / 768**. Heads, assigners, rotated IoU, NMS, and ONNX pre-NMS stay. Window attention (size **7**) pads each stage to a multiple of the window. v0.5 is one release: the 1× run freezes that contract, and the 3× is the Hub line (a 1× number is not the zoo entry next to YOLO-OBB at **77.80%** / RTMDet at **77.60%**). These releases stay on the single-scale 1024 tiles. v0.5.1 reuses the FPN tensors on Rotated Faster R-CNN. v0.5.2 adds Rotated FCOS only when single-tile 1024 latency is acceptable. RTMDet and YOLO-OBB stay the CNN speed models. Swin does not move onto them: RTMDet’s neck is CSPNeXt-PAFPN with a SepBN head built for those CSP widths, and YOLO-OBB is the same speed tier. A Swin RTMDet would be a new neck plus a slower backbone, which is neither this FPN accuracy line nor the CSPNeXt speed model. The published RTMDet DOTA lift (MMRotate RTMDet-s MS+RR, **79.98%** mAP50) is the multi-scale tile set on CSPNeXt-S, listed under possible work.

**v1.1 detail.** Oriented DINO is a new `model_type` in `odet train`, not a second repository. It reuses DOTA tiling, the le90 convention, rotated IoU, the loaders, and official Task 1 eval. The detector is its own path: multi-scale deformable decoder, a 5-parameter le90 box, and rotated IoU in the bipartite match. No NMS. No Hub weights until official Task 1 beats the Swin Oriented R-CNN from v0.5. Vanilla DETR (single-scale, no deformable attention) is not a release.

## Model tiers (target v1.0)

| Tier | Models |
|------|--------|
| **Accuracy** | Oriented R-CNN, Rotated Faster R-CNN (probiou) — ResNet50 and Swin-T. Swin-T FCOS only if v0.5.2 ships |
| **Balanced** | Rotated FCOS (ResNet50) |
| **Speed** | RTMDet-R, native Rotated YOLO-OBB |
| **Legacy** | Rotated RetinaNet (L1; MMRotate parity) |
| **Datasets** | DOTA, HRSC2016, FAIR1M, SSDD, HRSID |

## Closed ablations (not Hub)

- **RetinaNet ProbIoU 1×** — no zoo-worthy gain vs L1 (~63% vs 64% eval-val); keep L1 RetinaNet only
- Extra FRCNN angle / rIoU-aux recipes — did not beat the published ProbIoU 1.0 / 0.1 recipe
- **FCOS 1× ProbIoU aux** — 66.8% train-time mAP50 vs 76.5% for 1× KFIoU aux; recipe removed

## Possible work (not a release)

**DOTA multi-scale training (optional).** The multi-scale part is the tile set, not a new head: patch **1024**, overlap **500**, resize rates **0.5 / 1.0 / 1.5**. Three `train_tiles_dirs` are enough; `data_loader.shuffle` already mixes them. Validation stays on the single-scale tiles. A number next to MMRotate’s RTMDet-s MS+RR **79.98%** also needs the multi-scale test split and patch merge, not only single-scale `odet dota-submit`. `tools/tile_dota.py` can set the overlap today. It has no resize-rate flag; that flag is shared work, not a second tiler. The same tiles cover Rotated RTMDet-S (the Hub 3× recipe, `rotated_rtmdet_dota_le90_3x`) and YOLO-OBB (the Hub 12× recipe, `yolo_obb_dota_le90_12x` / `dota_le90_12x_s_coco`, mosaic left on). No new head on either.

**Recipe improvements on the six current models (optional).** Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet, Rotated FCOS, Rotated RTMDet, and YOLO-OBB. No new architecture. Keep a change when **AP75** or a **weak class** moves on a **held-out** split scored at the **operating resolution** (the recipe canvas: DOTA and FAIR1M single-scale 1024 tiles, HRSC / SSDD / HRSID keep-ratio). DOTA `make eval-val` is leaky train+val and does not count. On DOTA the held-out number is official Task 1 at that same 1024 window. HRSC, FAIR1M, SSDD, and HRSID eval-val are already held-out.

## Ongoing

- **Ongoing:** Hosted MkDocs site and dataset tutorials
- **Ongoing:** Optional GPU / fused CUDA rotated IoU/NMS (profiling-driven; unblocked now that FCOS shipped)
- **Ongoing:** ONNX remaining scope — RetinaNet pre-NMS detect graph; `keep_ratio` + pad canvas (HRSC / SSDD / HRSID); sliding-window tiling. Three-detector DOTA-tile export shipped in v0.3 (`python -m export`; see [ONNX export](examples/export.md)).

## Out of scope (for now)

- Swin-T on RTMDet or YOLO-OBB (those stay the CNN speed models; Swin attaches to the shared FPN)
- Ultralytics wrapper or AGPL dependencies
- Vanilla DETR and any set-prediction detector other than oriented DINO (v1.1)
- MMCV / MMDet as runtime dependencies
- RetinaNet ProbIoU Hub weights; FRCNN angle-fine-tune Hub twin without a clear eval-val win

See [Contributing](contributing.md) for areas where help is welcome.
