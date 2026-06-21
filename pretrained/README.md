# Pretrained weights

Large checkpoint files live here (typically **gitignored**). Registered assets are listed in [`oriented_det/pretrained/manifest.json`](../oriented_det/pretrained/manifest.json) and downloaded from Hugging Face Hub (`dl4eo/oriented-det-pretrained`).

## Naming

| Piece | Role | Example |
|-------|------|---------|
| **Manifest slug** | Stable id for `hf://` and `odet pretrained download` | `rotated_retinanet_dota_le90_1x` |
| **`.pth` filename** | Content-addressed blob on disk / Hub | `rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.pth` |
| **mAP** | Metadata only (eval-val protocol via `odet preds`; see below) | See tables below |

**Do not compare manifest `eval_map50` to training `compute_map_final` mAP.** They use different pipelines:

| Metric | Source | Typical use |
|--------|--------|-------------|
| **`eval_map50` in manifest / zoo tables** | `odet preds` on val tiles (`make eval-val` / `make metrics`) — all val tiles, `filter_empty_gt=false`, `production.*` decode | Published Hub metadata |
| **Periodic mAP during training** | `evaluation.compute_map_every_n_epochs` on non-empty val tiles; often GPU-sampled IoU (`use_exact_rotated_iou: false`) | Monitor convergence |
| **Final mAP after training** | `evaluation.compute_map_final` on best checkpoint; often exact CPU polygon IoU (`use_exact_rotated_iou_for_final_map: true`) | Training log headline number |

Example (Rotated RetinaNet 3×): Hub `eval_map50` is **71.52%** (preds on val); training periodic mAP at epoch 24 was **71.13%** (GPU sampling); training **final** mAP was **75.94%** (exact CPU on best checkpoint).

Publish or refresh a checkpoint:

```bash
python tools/publish_checkpoint.py runs/<model>/<run>/checkpoints/best_mAP_*.pth \
  pretrained/<basename_without_hash>
```

Then update `oriented_det/pretrained/manifest.json` with the new `filename` and `sha256`.

Upload to Hugging Face Hub:

```bash
hf auth login   # once
make upload-pretrained
```

Overrides: `HF_REPO_ID=`, `HF_REVISION=`, `HF_COMMIT_MESSAGE=`, `PRETRAINED_DIR=`.

## Download

```bash
odet pretrained list
odet pretrained download rotated_retinanet_dota_le90_3x
odet pretrained download rotated_faster_rcnn_dota_le90_3x
odet pretrained download oriented_rcnn_dota_le90_1x
```

```json
"load_from_checkpoint": "hf://oriented_rcnn_dota_le90_1x"
```

Environment overrides: see [oriented_det/pretrained/README.md](../oriented_det/pretrained/README.md).

## DOTA le90 pretrain zoo

**Training split: train+val** (`train` + `val` tile roots). **Eval split: val** (mAP on val tiles only). This is DOTA **pretrain** convention, not a fine-tune train/val holdout.

**mAP** below is **`make eval-val`** mAP50 (all 7,669 val tiles, `filter_empty_gt=false`, rotated IoU ≥ 0.50). Training-time periodic mAP uses non-empty tiles only and may be higher.

### Rotated RetinaNet R50-FPN

| Slug | Recipe | eval-val mAP50 | Config |
|------|--------|----------------|--------|
| `rotated_retinanet_dota_le90_1x` | 1× (12 ep) | 64.14% | [`dota_le90_1x.json`](../configs/rotated_retinanet/dota_le90_1x.json) |
| `rotated_retinanet_dota_le90_3x` | 3× (36 ep) | 71.52% | [`dota_le90_3x.json`](../configs/rotated_retinanet/dota_le90_3x.json) |

### Rotated Faster R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config |
|------|--------|----------------|--------|
| `rotated_faster_rcnn_dota_le90_3x` | 3× ProbIoU aux (default `make train`) | 76.41% | [`dota_le90_3x.json`](../configs/rotated_faster_rcnn/dota_le90_3x.json) |

### Oriented R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config |
|------|--------|----------------|--------|
| `oriented_rcnn_dota_le90_1x` | 1× (12 ep) | 74.79% | [`dota_le90_1x.json`](../configs/oriented_rcnn/dota_le90_1x.json) |

Per-class AP and eval reports: `predictions/20260615_*` and `predictions/20260618_140030` directories linked from [configs/rotated_retinanet/README.md](../configs/rotated_retinanet/README.md), [configs/rotated_faster_rcnn/README.md](../configs/rotated_faster_rcnn/README.md), and [configs/oriented_rcnn/README.md](../configs/oriented_rcnn/README.md).
