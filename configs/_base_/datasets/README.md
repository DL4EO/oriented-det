# Dataset base configs (DOTA)

JSON fragments included via `"_base_"` in training recipes under the framework repo root.

- **`dota_le90.json`** — Tiled DOTA v1.0 train/val folders and dataset-specific normalization (from `odet stats`).

Default Oriented R-CNN DOTA recipes: [`dota_le90_1x.json`](../../oriented_rcnn/dota_le90_1x.json) (full 1× baseline and default `make train`), [`dota_le90_3x.json`](../../oriented_rcnn/dota_le90_3x.json) (inherits 1×).

For **Airbus Playground CSV** datasets, keep dataset JSON in your own config tree and inherit `@odet:configs/_base_/...` fragments. See [Data loading](../../../docs/user-guide/data.md#airbus-playground).
