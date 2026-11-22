# `oriented_det.utils`

- **`progress.tqdm_progress_stream`**: file object for `tqdm(..., file=...)` so progress shows on the real TTY when stdout/stderr are piped to a log (same idea as `progress_stream` in `tools/train.py`).
- **`viz`**: `draw_boxes` / `draw_polygons`. Prediction overlays in `odet image-demo` and eval visualizations use outline width **`DEFAULT_LINE_WIDTH`** (4). The Gradio predictions viewer (`tools/app.py`) uses `--line-width` instead (screen pixels, default 1) and draws the stroke after zoom.

See the package `__init__.py` for the public exports.
