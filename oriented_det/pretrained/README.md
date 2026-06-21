# Pretrained Hub (library)

Manifest-driven downloads from [Hugging Face Hub](https://huggingface.co/docs/hub/models-downloading).

- **Manifest:** `manifest.json` — `repo_id`, `revision`, and registered assets (slug → hashed filename + metadata).
- **Default repo:** `dl4eo/oriented-det-pretrained` (override with `ORIENTED_DET_HF_REPO_ID`).
- **Local cache directory:** `<oriented-det-install>/pretrained/` (framework root, **not** the product repo). Override with `ORIENTED_DET_PRETRAINED_DIR`.

## Asset ids (slugs)

Use manifest **slugs** (not mAP numbers) with `hf://` and the CLI:

- `rotated_retinanet_dota_le90_1x`
- `rotated_retinanet_dota_le90_3x`
- `rotated_faster_rcnn_dota_le90_3x`
- `oriented_rcnn_dota_le90_1x`

On-disk / Hub filenames include a **SHA-256[:8]** suffix (see `tools/publish_checkpoint.py`).

## Usage

```python
from oriented_det.pretrained import ensure_checkpoint

path = ensure_checkpoint("hf://rotated_faster_rcnn_dota_le90_3x")
```

```bash
odet pretrained download rotated_retinanet_dota_le90_3x
odet pretrained download oriented_rcnn_dota_le90_1x
odet pretrained list
```

Training (`tools/train.py`) and inference call `ensure_checkpoint` when `checkpoint.load_from_checkpoint` points at a registered asset that is missing locally.

## Publishing weights

1. `python tools/publish_checkpoint.py <run_ckpt> pretrained/<basename_without_hash>`
2. Update `manifest.json` (`filename`, `sha256`, `source_recipe`, `eval_map50`, …). **`eval_map50`** is mAP50 from `odet preds` on val tiles (same protocol as `make eval-val`), not `compute_map_final` from training — see [pretrained/README.md](../../pretrained/README.md).
3. Upload: `make upload-pretrained` (or `hf upload …` — see [pretrained/README.md](../../pretrained/README.md)).

See also [pretrained/README.md](../../pretrained/README.md) at the repository root.
