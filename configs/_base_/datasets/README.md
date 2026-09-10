# Dataset base configs

JSON fragments included via `"_base_"` in training recipes under the framework repo root.

- **`dota_le90.json`** — Tiled DOTA v1.0 train/val folders and dataset-specific normalization (from `odet stats`). All DOTA 1× recipes inherit this (Oriented R-CNN, Faster R-CNN, FCOS, RetinaNet) so satellite tiles are centered with DOTA mean/std rather than ImageNet defaults.
- **`hrsc2016.json`** — Official HRSC2016 layout (`dataset.format: hrsc2016`), ImageSets trainval/test, square pad-800 default. Oriented R-CNN, Faster R-CNN, and FCOS 1×/3× override to `keep_ratio` + pad-32. Eval uses the same whole-image path as training; DOTA recipes stay on `fixed` sliding windows.
- **`fair1m.json`** — Tiled FAIR1M after `odet fair1m-to-dota` + `odet tile-dota` (1024 / overlap 200). Raw dump is `/path/to/data/FAIR1M` (official train 16,488 + val 8,287). Convert with `--splits train,val` (no `--val-fraction` unless the dump has no val; default `--image-format original` copies JPEG/PNG). Point `train_tiles_dir` / `val_tiles_dir` at `FAIR1M-dota/{train,val}/tiles_1024` — not the parent split folder (that still holds full images). Recipes use `dataset.format: dota` and finetune the matching DOTA 1× Hub checkpoint (no FAIR1M zoo).

Default Oriented R-CNN DOTA recipes: [`dota_le90_1x.json`](../../oriented_rcnn/dota_le90_1x.json) (full 1× baseline and default `make train`), [`dota_le90_3x.json`](../../oriented_rcnn/dota_le90_3x.json) (inherits 1×). HRSC2016: [`hrsc2016_le90_1x.json`](../../oriented_rcnn/hrsc2016_le90_1x.json), [`hrsc2016_le90_3x.json`](../../oriented_rcnn/hrsc2016_le90_3x.json). FAIR1M: [`fair1m_le90_1x.json`](../../oriented_rcnn/fair1m_le90_1x.json).

For **Airbus Playground CSV** datasets, keep dataset JSON in your own config tree and inherit `@odet:configs/_base_/...` fragments. See [Data loading](../../../docs/user-guide/data.md#airbus-playground).
