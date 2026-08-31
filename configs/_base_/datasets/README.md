# Dataset base configs

JSON fragments included via `"_base_"` in training recipes under the framework repo root.

- **`dota_le90.json`** — Tiled DOTA v1.0 train/val folders and dataset-specific normalization (from `odet stats`).
- **`hrsc2016.json`** — Official HRSC2016 layout (`dataset.format: hrsc2016`), ImageSets trainval/test, square pad-800 default. Oriented R-CNN, Faster R-CNN, and FCOS 1×/3× override to `keep_ratio` + pad-32. Eval uses the same whole-image path as training; DOTA recipes stay on `fixed` sliding windows.

Default Oriented R-CNN DOTA recipes: [`dota_le90_1x.json`](../../oriented_rcnn/dota_le90_1x.json) (full 1× baseline and default `make train`), [`dota_le90_3x.json`](../../oriented_rcnn/dota_le90_3x.json) (inherits 1×). HRSC2016: [`hrsc2016_le90_1x.json`](../../oriented_rcnn/hrsc2016_le90_1x.json), [`hrsc2016_le90_3x.json`](../../oriented_rcnn/hrsc2016_le90_3x.json).

For **Airbus Playground CSV** datasets, keep dataset JSON in your own config tree and inherit `@odet:configs/_base_/...` fragments. See [Data loading](../../../docs/user-guide/data.md#airbus-playground).
