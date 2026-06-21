# Base configs

These files are **base configs** only. They are not used as top-level training configs; they are included via `_base_` in configs under [oriented_rcnn/](../oriented_rcnn/), [rotated_faster_rcnn/](../rotated_faster_rcnn/), and [rotated_retinanet/](../rotated_retinanet/).

## Contents

- **datasets/** — Dataset defaults: `dota_le90.json` (`data_root`, tile dirs, format, classes, etc.).
- **models/** — Model backbones and heads: `oriented_rcnn_r50.json`, `rotated_faster_rcnn_r50.json`, `rotated_retinanet_r50.json`.
- **schedules/** — Training schedules: `1x.json`, `3x.json`, `6x.json` (epochs, LR, warmup, etc.).
- **fp16.json** — Mixed-precision (AMP) settings.
- **augmentation.json** — Albumentations / data augmentation.
- **preprocessing.json** — Preprocessing options.

See [configs/README.md](../README.md) for how configs are composed and used.
