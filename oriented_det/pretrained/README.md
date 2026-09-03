# Pretrained Hub (library)

Manifest-driven downloads from [Hugging Face Hub](https://huggingface.co/docs/hub/models-downloading).

- **Manifest:** `manifest.json` — `repo_id`, `revision`, and registered assets (slug → hashed filename + metadata).
- **Default repo:** `dl4eo/oriented-det-pretrained` (override with `ORIENTED_DET_HF_REPO_ID`).
- **Local cache directory:** `<oriented-det-install>/pretrained/` (framework root, **not** the product repo). Override with `ORIENTED_DET_PRETRAINED_DIR`.

## Asset ids (slugs)

Use manifest **slugs** (not mAP numbers) with `hf://` and the CLI:

- `oriented_rcnn_dota_le90_3x`
- `oriented_rcnn_hrsc2016_le90_3x`
- `rotated_faster_rcnn_dota_le90_3x`
- `rotated_faster_rcnn_hrsc2016_le90_3x`
- `rotated_retinanet_dota_le90_3x`
- `rotated_fcos_dota_le90_3x`
- `rotated_fcos_hrsc2016_le90_3x`

On-disk / Hub filenames include a **SHA-256[:8]** suffix (see `tools/publish_checkpoint.py`).
Each published weight can have sidecar artifacts beside it in `pretrained/`: `<weight-stem>.json` for the exact final run config and `<weight-stem>.log` for the training log. DOTA 3× sidecars set `production.score_threshold` to eval-val F1 − **0.05** (Oriented R-CNN **0.7**, Faster R-CNN **0.6**, RetinaNet **0.45**, FCOS **0.2**). HRSC 3× sidecars use the same rule (Oriented R-CNN / Faster R-CNN **0.85**, FCOS **0.2**).

## Usage

```python
from oriented_det.pretrained import ensure_checkpoint

path = ensure_checkpoint("hf://oriented_rcnn_dota_le90_3x")
```

```bash
odet pretrained download oriented_rcnn_dota_le90_3x
odet pretrained list
```

Training (`tools/train.py`) and inference call `ensure_checkpoint` when `checkpoint.load_from_checkpoint` points at a registered asset that is missing locally.
Simple inference (`odet image-demo`, `odet preds`, deploy paths through `oriented_det.runtime.checkpoint`) prefers the sidecar JSON only when the provided config is that checkpoint's manifest `source_recipe`. A different user-provided config is kept as-is.

## Publishing weights

1. `python tools/publish_checkpoint.py <run_ckpt> pretrained/<basename_without_hash>`
2. Copy `<experiment>/config.json` and `train.log` to `pretrained/<weight-stem>.json` and `.log`
3. Update `manifest.json` (`filename`, `sha256`, `source_recipe`, `eval_map50`, …). **`eval_map50`** is mAP50 from `odet preds` on val tiles (same protocol as `make eval-val`), not `compute_map_final` from training — see [pretrained/README.md](../../pretrained/README.md).
4. Upload: `make upload-pretrained` (`.pth` plus sidecar `.json` / `.log` when present)

See also [pretrained/README.md](../../pretrained/README.md) at the repository root.
