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
| **v0.4.3** | Shipped — **RTMDet-S MS Hub** `rotated_rtmdet_dota_le90_3x_ms` **80.13%** MS/MS Task 1 (MS/SS **78.92%**); live `odet dota-submit --resize-rates` |
| **v0.5** | **Swin-T + FPN** on Oriented R-CNN: DOTA le90 1×, then 3× on Hub |
| **v0.5.1** | **Swin-T Rotated Faster R-CNN** (ProbIoU), same FPN |
| **v0.5.2** | **Swin-T Rotated FCOS**, only if 1024-tile latency stays on the accuracy tier |
| **v0.6** | **RoI Transformer** (R50 + FPN): DOTA le90; MMRotate `roi_trans_*` pair |
| **v0.6.1** | **RoI Transformer + Swin-T**, same head on the shared Swin-FPN from v0.5 |
| **v0.7** | **ReDet** (rotation-equivariant ReResNet + ReFPN): DOTA le90 Hub only if Task 1 beats the leading accuracy-tier Hub line |
| **v1.0** | Stable API, hosted docs, complete model zoo (accuracy / balanced / speed tiers) |
| **v1.1** | **Oriented DINO** in this repo. DOTA only. Hub only if it beats the leading accuracy-tier Hub checkpoint (Swin ORCNN / RoI Trans / ReDet as shipped) |

**v0.5–v0.5.2 detail.** Swin-T is a backbone swap onto the shared FPN. Stages at strides 4, 8, 16, and 32 feed that FPN. New work is ImageNet initialization and FPN lateral convolutions for channels **96 / 192 / 384 / 768**. Heads, assigners, rotated IoU, NMS, and ONNX pre-NMS stay. Window attention (size **7**) pads each stage to a multiple of the window. v0.5 is one release: the 1× run freezes that contract, and the 3× is the Hub line (a 1× number is not the zoo entry next to YOLO-OBB SS **77.80%** / RTMDet SS **77.60%** / RTMDet MS/MS **80.13%**). These releases stay on the single-scale 1024 tiles. v0.5.1 reuses the FPN tensors on Rotated Faster R-CNN. v0.5.2 adds Rotated FCOS only when single-tile 1024 latency is acceptable. RTMDet and YOLO-OBB stay the CNN speed models. Swin does not move onto them: RTMDet’s neck is CSPNeXt-PAFPN with a SepBN head built for those CSP widths, and YOLO-OBB is the same speed tier. A Swin RTMDet would be a new neck plus a slower backbone, which is neither this FPN accuracy line nor the CSPNeXt speed model. RTMDet MS Hub (**80.13%** MS/MS) shipped in **v0.4.3**; MMRotate RTMDet-s MS+RR (**79.98%**) is the external reference.

**v0.6–v0.6.1 detail.** **RoI Transformer** is a new two-stage accuracy detector: horizontal RPN proposals, then a learned **RoI transform** (HBB → OBB) before the rotated second stage (Ding et al.; MMRotate `roi_trans_*`). v0.6 ships **ResNet50 + FPN** first to freeze the head contract against MMRotate. v0.6.1 reuses the **Swin-T + FPN** from v0.5 (same **96 / 192 / 384 / 768** laterals). Not a backbone-only swap on Oriented R-CNN — new `model_type` / head path. SS 1024 tiles unless a later optional MS recipe says otherwise. Pair configs: MMRotate `roi_trans_r50_fpn_*` and `roi_trans_swin_tiny_fpn_1x_dota_le90` (zoo ~**77.51%** Task 1 class).

**v0.7 detail.** **ReDet** adds rotation-equivariant **ReResNet** and **ReFPN** (Han et al.; MMRotate `redet_*`). That is new backbone **and** neck infrastructure, not a drop-in on the shared FPN — hence after RoI Transformer. DOTA le90 Hub only when official Task 1 beats the best accuracy-tier Hub checkpoint then shipping (Swin Oriented R-CNN and/or RoI Transformer ± Swin).

**v1.1 detail.** Oriented DINO is a new `model_type` in `odet train`, not a second repository. It reuses DOTA tiling, the le90 convention, rotated IoU, the loaders, and official Task 1 eval. The detector is its own path: multi-scale deformable decoder, a 5-parameter le90 box, and rotated IoU in the bipartite match. No NMS. No Hub weights until official Task 1 beats the leading accuracy-tier Hub line from v0.5–v0.7. Vanilla DETR (single-scale, no deformable attention) is not a release.

## Model tiers (target v1.0)

| Tier | Models |
|------|--------|
| **Accuracy** | Oriented R-CNN, Rotated Faster R-CNN (probiou) — ResNet50 and Swin-T; **RoI Transformer** (R50, then Swin-T); **ReDet** (ReResNet + ReFPN). Swin-T FCOS only if v0.5.2 ships |
| **Balanced** | Rotated FCOS (ResNet50) |
| **Speed** | RTMDet-R, native Rotated YOLO-OBB |
| **Legacy** | Rotated RetinaNet (L1; MMRotate parity) |
| **Datasets** | DOTA, HRSC2016, FAIR1M, SSDD, HRSID |

Speed-tier designs (RTMDet / YOLO-OBB) currently lead the published SS Hub Task 1 table; the accuracy tier is the FPN / equivariant line meant to reclaim that lead (Swin, RoI Transformer, ReDet). Tier names are architecture roles, not a claim that speed models are weaker today.

## Closed ablations (not Hub)

- **RetinaNet ProbIoU 1×** — no zoo-worthy gain vs L1 (~63% vs 64% eval-val); keep L1 RetinaNet only
- Extra FRCNN angle / rIoU-aux recipes — did not beat the published ProbIoU 1.0 / 0.1 recipe
- **FCOS 1× ProbIoU aux** — 66.8% train-time mAP50 vs 76.5% for 1× KFIoU aux; recipe removed

## Possible work (not a release)

**YOLO-OBB multi-scale Hub (optional).** RTMDet-S MS Hub and live MS Task 1 shipped in **v0.4.3**. The same tile set (patch **1024**, overlap **500**, rates **0.5 / 1.0 / 1.5**; `odet tile-dota --resize-rate`) and `odet dota-submit --resize-rates 0.5,1.0,1.5` already cover YOLO-OBB. Recipe [`dota_le90_12x_s_coco_ms.json`](../configs/yolo_obb/dota_le90_12x_s_coco_ms.json) is in-tree (MS train / SS val); a Hub slug waits on a finished MS/MS Task 1 number. No new head.

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
