# Documentation (MkDocs)

This folder is the source for the **MkDocs** documentation site. It contains the full user guide, API reference (generated from docstrings), and examples.

For project overview, installation, and quick start, see the [repository README](https://github.com/DL4EO/oriented-det/blob/main/README.md). Hands-on write-ups: [DeepLearning.Earth](https://deeplearning.earth).

## Build and serve

From the repository root (installs MkDocs into the active environment if missing):

```bash
make docs          # build → site/
make docs-serve    # live preview at http://127.0.0.1:8000
```

Or install dependencies once and run MkDocs directly:

```bash
make docs-deps     # uv pip install -e ".[docs]"
python -m mkdocs build
python -m mkdocs serve
```

Output is in `site/`.

Then open http://127.0.0.1:8000 in your browser.

## Deploy (GitHub Pages)

The public site is deployed by `.github/workflows/pages.yml`.

One-time repository setup: in **Settings → Pages**, set **Source** to **GitHub Actions**.

After that, pushes to `main` that touch docs/source files publish to:

```text
https://dl4eo.github.io/oriented-det/
```

## Structure

- **docs/** — Source Markdown (this folder)
- **roadmap.md** — Public release plan (v0.4.2 YOLO-OBB Hub shipped at **77.80%** Task 1; next v0.5–v0.5.2 Swin-T on the shared FPN; v1.1 oriented DINO). DOTA multi-scale training and recipe work on the six current models are optional, not releases
- **user-guide/configuration.md** — Canonical JSON training config reference
- **eval-reports/** — Published reports per Hub slug (markdown; most DOTA/HRSC slugs also ship `analysis_iou0.50.json`); see [eval-reports/README.md](eval-reports/README.md). Raw `predictions.json` lives in gitignored `predictions/`. YOLO-OBB 12× is Task 1 markdown only.
- **examples/deploy.md** — Docker Tile Geo Process example
- **examples/export.md** — ONNX export (`odet export`)
- **code-analysis-report.md** — Deep code/doc analysis (2026-06-15): inventory, validation, findings
- **mkdocs.yml** — MkDocs configuration and nav
- **site/** — Generated HTML (created by `mkdocs build`)

API pages are generated from Python docstrings via `mkdocstrings`. See [documentation.md](documentation.md) for more on writing and building the docs.

## Further reading

Hands-on write-ups (code, metrics, inference knobs, and project rationale) live on **[DeepLearning.Earth](https://deeplearning.earth)**:

| Topic | Post |
|-------|------|
| Oriented detection in aerial and satellite images | [https://deeplearning.earth/oriented-detection-in-satellite-images/](https://deeplearning.earth/oriented-detection-in-satellite-images/) |
| Why OrientedDet / Apache 2.0 stack | [https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) |
| Visual detections (15 DOTA classes) | [https://deeplearning.earth/posts/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes/](https://deeplearning.earth/posts/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes/) |
| `odet image-demo` on Apple Silicon | [https://deeplearning.earth/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/](https://deeplearning.earth/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) |
| Sliding-window inference (`demo/large.jpg`) | [https://deeplearning.earth/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/](https://deeplearning.earth/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/) |
| Sentinel-2 ships (`--zoom`, class filter) | [https://deeplearning.earth/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/](https://deeplearning.earth/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) |
| Sampled rIoU, ProbIoU, Faster R-CNN Task 1 | [https://deeplearning.earth/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/](https://deeplearning.earth/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) |
| Zoo / MMRotate parity | [https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/](https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) |
| Rotated FCOS, decoded rIoU, four-family zoo | [https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) |
| FCOS vs Oriented R-CNN (MPS latency, thresholds) | [https://deeplearning.earth/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/](https://deeplearning.earth/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) |
| Side-by-side browser demo (three 1× checkpoints) | [https://deeplearning.earth/posts/2026-09-06_oriented_det_optical_satellite_demo/](https://deeplearning.earth/posts/2026-09-06_oriented_det_optical_satellite_demo/) |
| Apache 2.0 vs DOTA / HRSC dataset terms | [https://deeplearning.earth/posts/2026-09-10_oriented_det_apache_license_versus_dota/](https://deeplearning.earth/posts/2026-09-10_oriented_det_apache_license_versus_dota/) |
| Docker deploy (Tile Geo Process) | [https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/](https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/) |
| ONNX export (no PyTorch on infer) | [https://deeplearning.earth/posts/2026-10-08_onnx_export_without_pytorch/](https://deeplearning.earth/posts/2026-10-08_onnx_export_without_pytorch/) |
| Rotated RTMDet, 77.60% Task 1 | [https://deeplearning.earth/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/](https://deeplearning.earth/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/) |
| Which oriented detector to train | [https://deeplearning.earth/posts/2026-10-29_which_oriented_detector_to_train/](https://deeplearning.earth/posts/2026-10-29_which_oriented_detector_to_train/) |
| YOLO-OBB, 77.80% Task 1 (v0.4.2) | [https://deeplearning.earth/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/](https://deeplearning.earth/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/) |
