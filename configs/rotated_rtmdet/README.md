# Rotated RTMDet configs

Native **Rotated RTMDet** (`model_type: rotated_rtmdet`) — CSPNeXt-S + PAFPN.

| Recipe | Config | Notes |
|--------|--------|-------|
| DOTA le90 3× **S** | [`dota_le90_3x_s.json`](dota_le90_3x_s.json) | **Hub.** Official Task 1 **77.60%** (AP75 52.01, COCO mAP 49.02). Leaky eval-val **82.75%** (gap **5.15**, metrics margin 0). Deploy `production.score_threshold` **0.45** and `production.ignore_margin_pixels` **0**. |
| OSSDD le90 1× **S** (VH) | [`ossdd_le90_vh_1x_s.json`](ossdd_le90_vh_1x_s.json) | Finetune the DOTA Hub checkpoint on Sentinel-1 GRD ships. Train on official **train**, validate on official **val**. mAP is the mean of area-under-PR over rotated IoU 0.10–0.95 step 0.05 (`evaluation.map_iou_thresholds`). `evaluation.test_map` scores the held-out **test** split once on the best checkpoint at `evaluation.test_score_threshold` (null → 0.05). `make eval-test` uses that same floor. No Hub slug. |

Hub slug: `rotated_rtmdet_dota_le90_3x` (`hf://rotated_rtmdet_dota_le90_3x`). Report: [`docs/eval-reports/rotated_rtmdet_dota_le90_3x/`](../../docs/eval-reports/rotated_rtmdet_dota_le90_3x/model_analysis.md). Run: `runs/rotated_rtmdet/20261002-102116`.

```bash
odet train --config configs/rotated_rtmdet/dota_le90_3x_s.json
odet pretrained download rotated_rtmdet_dota_le90_3x
odet dota-submit --checkpoint hf://rotated_rtmdet_dota_le90_3x \
  --test-dir /path/to/data/DOTA-v1.0/test --output-dir work_dirs/Task1_rtmdet_s_3x
```

The DOTA recipe always sees **1024×1024**. Mosaic is off. Random rotate is on (p=0.5, ±180°). A tile that contains `storage-tank` or `roundabout` rotates by 90°, 180°, −90°, or −180°. Val and inference do not rotate.

[`dota_le90_3x_s.json`](dota_le90_3x_s.json) is 36 epochs, batch 8, AdamW `2.5e-4`, weight decay `0.05` on conv weights only, no gradient clip, warmup 1000 steps, then a flat LR until epoch 18 and cosine to `1.25e-5` by epoch 36. EMA (momentum 0.0002, gamma 2000) is what validation and `best_mAP_*.pth` store. `model.rtmdet_angle_weight` is 0, so the angle L1 term is dropped; rotated IoU stays at weight 2.0. `rtmdet_nms_pre` is 30000. Normalization is the DOTA tile mean and standard deviation.

## Official Task 1

| | AP50 | AP75 | COCO mAP | leaky eval-val | eval-val − Task 1 |
|--|-----:|-----:|---------:|---------------:|------------------:|
| RTMDet-S 3× (`runs/rotated_rtmdet/20261002-102116`) | **77.60** | 52.01 | 49.02 | 82.75 | **5.15** |

Leaky eval-val is `predictions/20261003_021342` (all 7669 val tiles, score ≥ 0.05, metrics margin **0**, 57,768 objects). Best F1 on that sweep is **0.50**, so deploy uses **0.45**. `production.ignore_margin_pixels` is **0**. Helicopter is 61.6 on Task 1 and 90.0 on leaky eval-val.

## OSSDD (Sentinel-1, 2026)

[OSSDD](https://arxiv.org/abs/2608.01963) is Sentinel-1 GRD ship chips (VV and VH, rotated boxes). This repo loads the DOTA export, not the zip. VH is the paper's stronger polarization. The network sees **512×512**: train chips are 700×700 and are random-cropped; val uses a center crop. `ossdd_to_dota.py` applies the paper's clip / log10 / per-chip min-max, and the dataset config then uses the DOTA checkpoint mean/std so the finetune sees the same input scale as `rotated_rtmdet_dota_le90_3x`. The 15-way DOTA classification conv is skipped on load and re-initialized for `ship`. `production.score_threshold` **0.3** is untuned.

Download `ossdd.zip` only (**14.4 GB**). Extracted chips are about **58 GB**. Do not also pull the WebDataset shards (the Hub total is 72.3 GB because both copies are stored).

```bash
hf download sylviaHoch/OpenSARShip-Ship-Detection-Dataset \
  ossdd.zip --repo-type dataset --local-dir /path/to/data/OSSDD
unzip /path/to/data/OSSDD/ossdd.zip -d /path/to/data/OSSDD
python tools/ossdd_fix_tiff.py --data-root /path/to/data/OSSDD
python tools/ossdd_to_dota.py \
  --data-root /path/to/data/OSSDD \
  --output-dir /path/to/data/OSSDD-dota \
  --polarization VH
odet train --config configs/rotated_rtmdet/ossdd_le90_vh_1x_s.json
```

[`ossdd_le90_vh_1x_s.json`](ossdd_le90_vh_1x_s.json) is 12 epochs, batch 8, AdamW `2.5e-4`, warmup 500 steps, flat LR until epoch 6, then cosine to `1.25e-5`. Dataset paths live in [`ossdd.json`](../_base_/datasets/ossdd.json), including `test_tiles_dirs`. Reported mAP averages the area under the precision-recall curve at rotated IoU 0.10, 0.15, …, 0.95. `best_metric` uses that average. GT-cover stays at IoU 0.5. After training, `evaluation.test_map` loads the best val checkpoint and reports test mAP at `evaluation.test_score_threshold` (null → 0.05). Periodic val mAP stays at `val_score_threshold` (0.3). `make eval-test EXPERIMENT=runs/rotated_rtmdet/<timestamp>` uses that same test floor. Pass `--polarization VV` and point the config paths at that export to train the other band. Annotations are CC-BY-NC-SA-4.0.

### OSSDD 1x VH results

Three finetunes of the DOTA hubs, VH only, 512 crop. Validation is the 923 non-empty val chips (2,040 rotated ships). Test is all 3,036 chips, empty chips kept (14,145 rotated ships). The train logs report **AP50**: one rotated IoU of 0.5, boxes below score 0.3 dropped before the precision-recall curve. The saved configs also list `map_iou_thresholds` 0.10–0.95; that average is not the number printed in these logs.

| | Val AP50 | Test AP50 | Test precision | Test recall | Wall time | Run |
|---|---:|---:|---:|---:|---:|---|
| RTMDet-S 1× | 77.84% | **88.49%** | 87.0% (13,354 / 15,345) | 94.4% (13,354 / 14,145) | 45 min | `runs/rotated_rtmdet/20261005-143859` |
| Faster R-CNN 1× | 40.28% | 60.10% | 85.5% (12,185 / 14,245) | 86.1% (12,185 / 14,145) | 2 h 55 min | `runs/rotated_faster_rcnn/20261006-025419` |
| FCOS 1× | 27.58% | 58.21% | 82.5% (12,976 / 15,731) | 91.7% (12,976 / 14,145) | 1 h 29 min | `runs/rotated_fcos/20261006-012500` |

On validation, FCOS emits about 1.1 boxes per image against 2.2 ships (recall 43.0%, 878 / 2,040). Faster R-CNN recall there is 59.1% (1,205 / 2,040). RTMDet recall is 81.2% (1,656 / 2,040). FCOS gained 5.5 AP50 points from epoch 8 to 12, and Faster R-CNN gained 3.5. Both were still rising. That is a few points of headroom, not the gap to RTMDet.

[Hammer et al.](https://arxiv.org/abs/2608.01963) Table 6, VH **test**, is a different measurement. Their detectors are axis-aligned ResNet-50 models trained from scratch for 25 epochs. Precision and recall use IoU **0.2** and confidence 0.3. Their mAP is the mean of the precision-recall area from IoU **0.10 to 0.95** in steps of 0.05.

| Paper, VH test | Precision (IoU 0.2) | Recall (IoU 0.2) | mAP (IoU 0.10–0.95) |
|---|---:|---:|---:|
| Faster R-CNN | 0.7442 | 0.8942 | 0.7018 |
| FCOS | 0.8424 | 0.8091 | 0.6961 |
| DETR | 0.7444 | 0.8891 | 0.5999 |

At score 0.3, our Faster R-CNN test precision is higher than theirs and our recall is a little lower, at IoU 0.5 rather than 0.2. Our FCOS test recall is higher than theirs at that stricter IoU. Do not rank our AP50 against their mAP. A 36-epoch RTMDet run with the same normalization scored 78.11% val AP50 and 88.23% test AP50; the 1× was kept.

Base fragment: [`rotated_rtmdet_cspnext_s.json`](../_base_/models/rotated_rtmdet_cspnext_s.json).

Parent: [configs/README.md](../README.md).
