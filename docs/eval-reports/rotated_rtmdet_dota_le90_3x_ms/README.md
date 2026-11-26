# `rotated_rtmdet_dota_le90_3x_ms`

Rotated RTMDet-S 3× trained on DOTA multi-scale tiles. Official Task 1 numbers:

| Protocol | Task 1 AP50 | Notes |
|----------|------------:|-------|
| **MS/MS** (advertised) | **80.13%** | `odet dota-submit --resize-rates 0.5,1.0,1.5` |
| MS/SS | 78.92% | same weights, single-scale sliding windows |
| SS Hub (`rotated_rtmdet_dota_le90_3x`) | 77.60% | SS train + SS test |

See [`model_analysis.md`](model_analysis.md) and [`configs/rotated_rtmdet/README.md`](../../../configs/rotated_rtmdet/README.md#multiscale-ss--msss--msms).
