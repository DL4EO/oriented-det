# Model Analysis Report

Hub slug **`rotated_rtmdet_dota_le90_3x_ms`**. Official DOTA v1.0 Task 1 **80.13%** under **MS/MS** (`--resize-rates 0.5,1.0,1.5`, overlap 500 in resized space, NMS 0.1). Same weights with **MS/SS** (default SS sliding windows) scored **78.92%**. SS Hub `rotated_rtmdet_dota_le90_3x` is **77.60%**. In-train leaky val (score ≥ 0.3, every 4 epochs) ended at **90.41%**. Deploy `production.score_threshold` **0.45** (unchanged from SS Hub; no new F1 sweep).

## Multiscale protocols

| Shorthand | Train tiles | Task 1 test | This run |
|-----------|-------------|-------------|----------|
| **SS** | Single-scale 1024 / overlap 200 | SS windows (overlap 200) | Hub `rotated_rtmdet_dota_le90_3x` **77.60%** |
| **MS/SS** | MS rates 0.5 / 1.0 / 1.5, overlap 500 | SS windows | **78.92%** |
| **MS/MS** | same MS train | Live multi-rate windows (`--resize-rates`) | **80.13%** (advertised) |

## Official Task 1 — MS/MS (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8868 |
| baseball-diamond | 0.8615 |
| bridge | 0.5701 |
| ground-track-field | 0.7988 |
| small-vehicle | 0.8052 |
| large-vehicle | 0.8490 |
| ship | 0.8877 |
| tennis-court | 0.9090 |
| basketball-court | 0.8826 |
| storage-tank | 0.8808 |
| soccer-ball-field | 0.6455 |
| roundabout | 0.7346 |
| harbor | 0.7817 |
| swimming-pool | 0.8179 |
| helicopter | 0.7076 |
| **mAP50** | **0.8013** |

- Description submitted: oriented-det pretrained weights RTMDet 3x MS (MS/MS rates)
- Experiment: `runs/rotated_rtmdet/20261123-224134` (`best_mAP_0.90.pth`)
- Recipe: [`configs/rotated_rtmdet/dota_le90_3x_s_ms.json`](../../../configs/rotated_rtmdet/dota_le90_3x_s_ms.json)
- Submit: `odet dota-submit --checkpoint hf://rotated_rtmdet_dota_le90_3x_ms --test-dir /path/to/data/DOTA-v1.0/test --output-dir work_dirs/Task1_rtmdet_s_3x_ms_rates --resize-rates 0.5,1.0,1.5`

## Official Task 1 — MS/SS (same weights)

| | AP50 | AP75 | COCO mAP |
|--|-----:|-----:|---------:|
| MS/SS | **0.7892** | 0.5683 | 0.5190 |

Per-class AP50 (MS/SS): plane 0.8843, baseball-diamond 0.8584, bridge 0.5460, ground-track-field 0.7348, small-vehicle 0.8085, large-vehicle 0.8271, ship 0.8865, tennis-court 0.9077, basketball-court 0.8649, storage-tank 0.8780, soccer-ball-field 0.6373, roundabout 0.6967, harbor 0.7786, swimming-pool 0.8295, helicopter 0.6994.

## Compare — SS Hub

Report: [`../rotated_rtmdet_dota_le90_3x/model_analysis.md`](../rotated_rtmdet_dota_le90_3x/model_analysis.md). Task 1 **77.60%** (AP75 52.01, COCO 49.02); leaky eval-val 82.75%.

## In-train leaky val (score ≥ 0.3)

Final periodic mAP at epoch 36: **90.41%**. Not comparable to Task 1 or to SS Hub’s `make eval-val` at score 0.05.
