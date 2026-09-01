# Pretrained weights

Large checkpoint files live here (typically **gitignored**). Registered assets are listed in [`oriented_det/pretrained/manifest.json`](../oriented_det/pretrained/manifest.json) and downloaded from Hugging Face Hub (`dl4eo/oriented-det-pretrained`).

## Naming

| Piece | Role | Example |
|-------|------|---------|
| **Manifest slug** | Stable id for `hf://` and `odet pretrained download` | `oriented_rcnn_dota_le90_1x` |
| **`.pth` filename** | Content-addressed blob on disk / Hub | `oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.pth` |
| **mAP** | Metadata only (eval-val protocol via `odet preds`; see below) | See tables below |

**Do not compare manifest `eval_map50` to training `compute_map_final` mAP.** They use different pipelines:

| Metric | Source | Typical use |
|--------|--------|-------------|
| **`eval_map50` in manifest / zoo tables** | `odet preds` on val tiles (`make eval-val` / `make metrics`) — all val tiles, `filter_empty_gt=false`, decode with `evaluation.final_nms_iou_threshold` when set (recipes: **0.1**); deploy still uses `production` NMS **0.3** | Published Hub metadata |
| **Periodic mAP during training** | `evaluation.compute_map_every_n_epochs` on non-empty val tiles; often GPU-sampled IoU (`use_exact_rotated_iou: false`) | Monitor convergence |
| **Final mAP after training** | `evaluation.compute_map_final` on best checkpoint; often exact CPU polygon IoU (`use_exact_rotated_iou_for_final_map: true`) | Training log headline number |

Example (Oriented R-CNN 1×): Hub `eval_map50` is **74.79%** from `odet preds` on val tiles using the published `oriented_rcnn_dota_le90_1x` checkpoint.

Publish or refresh a checkpoint:

```bash
python tools/publish_checkpoint.py runs/<model>/<run>/checkpoints/best_mAP_*.pth \
  pretrained/<basename_without_hash>
cp runs/<model>/<run>/config.json pretrained/<weight-stem>.json
cp runs/<model>/<run>/train.log pretrained/<weight-stem>.log
```

Then update `oriented_det/pretrained/manifest.json` with the new `filename` and `sha256`.

Upload to Hugging Face Hub (weights plus sidecar `.json` / `.log` when present):

```bash
hf auth login   # once
make upload-pretrained
```

**RetinaNet checkpoint compatibility:** weights trained before the MMRotate parity release (separate cls/reg subnets, 3×3 heads, `LastLevelP6P7` FPN) will not load. Use newly trained or re-published Hub slugs after that release.

Overrides: `HF_REPO_ID=`, `HF_REVISION=`, `HF_COMMIT_MESSAGE=`, `PRETRAINED_DIR=`.

## Download

```bash
odet pretrained list
odet pretrained download oriented_rcnn_dota_le90_1x
```

```json
"load_from_checkpoint": "hf://oriented_rcnn_dota_le90_1x"
```

Environment overrides: see [oriented_det/pretrained/README.md](../oriented_det/pretrained/README.md).

## DOTA le90 pretrain zoo

**Training split: train+val** (`train` + `val` tile roots). **Eval split: val** (mAP on val tiles only). This is DOTA **pretrain** convention, not a fine-tune train/val holdout.

**mAP** below is **`make eval-val`** mAP50 (all 7,669 val tiles, `filter_empty_gt=false`, rotated IoU ≥ 0.50). Training-time periodic mAP uses non-empty tiles only and may be higher.

### Oriented R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `oriented_rcnn_dota_le90_1x` | 1× (12 ep) | 74.79% | [`dota_le90_1x.json`](../configs/oriented_rcnn/dota_le90_1x.json) | [`oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.json`](./oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.json) | [`oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.log`](./oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.log) |
| `oriented_rcnn_dota_le90_3x` | 3× (36 ep) | 79.40% | [`dota_le90_3x.json`](../configs/oriented_rcnn/dota_le90_3x.json) | [`oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.json`](./oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.json) | [`oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.log`](./oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.log) |

### Rotated Faster R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `rotated_faster_rcnn_dota_le90_1x` | 1× ProbIoU main | 77.57% | [`dota_le90_1x.json`](../configs/rotated_faster_rcnn/dota_le90_1x.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.json`](./rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.log`](./rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.log) |
| `rotated_faster_rcnn_dota_le90_3x` | 3× ProbIoU main | 83.42% | [`dota_le90_3x.json`](../configs/rotated_faster_rcnn/dota_le90_3x.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.json`](./rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.log`](./rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.log) |
| `rotated_faster_rcnn_dota_le90_3x_ce` | 3× CE baseline | 75.58% | [`dota_le90_3x.json`](../configs/rotated_faster_rcnn/dota_le90_3x.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_3x_ce-c077eeee.json`](./rotated_faster_rcnn_r50_fpn_dota_le90_3x_ce-c077eeee.json) | [`rotated_faster_rcnn_r50_fpn_dota_le90_3x_ce-c077eeee.log`](./rotated_faster_rcnn_r50_fpn_dota_le90_3x_ce-c077eeee.log) |

### Rotated RetinaNet R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `rotated_retinanet_dota_le90_1x` | 1× (12 ep) | 64.14% | [`dota_le90_1x.json`](../configs/rotated_retinanet/dota_le90_1x.json) | [`rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json`](./rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.json) | [`rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.log`](./rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.log) |
| `rotated_retinanet_dota_le90_3x` | 3× (36 ep) | 71.52% | [`dota_le90_3x.json`](../configs/rotated_retinanet/dota_le90_3x.json) | [`rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.json`](./rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.json) | [`rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.log`](./rotated_retinanet_r50_fpn_dota_le90_3x-8decc6f1.log) |

### Rotated FCOS R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `rotated_fcos_dota_le90_3x` | 3× decoded rIoU primary | 82.32% | [`dota_le90_3x.json`](../configs/rotated_fcos/dota_le90_3x.json) | [`rotated_fcos_r50_fpn_dota_le90_3x-6e383331.json`](./rotated_fcos_r50_fpn_dota_le90_3x-6e383331.json) | [`rotated_fcos_r50_fpn_dota_le90_3x-6e383331.log`](./rotated_fcos_r50_fpn_dota_le90_3x-6e383331.log) |
| `rotated_fcos_dota_le90_3x_kfiou_aux` | 3× L1 + KFIoU aux 0.1 | 77.18% | [`dota_le90_1x_l1_kfiou_aux.json`](../configs/rotated_fcos/dota_le90_1x_l1_kfiou_aux.json) (1× lineage; 3× recipe retired) | [`rotated_fcos_r50_fpn_dota_le90_3x_kfiou_aux-83c78863.json`](./rotated_fcos_r50_fpn_dota_le90_3x_kfiou_aux-83c78863.json) | [`rotated_fcos_r50_fpn_dota_le90_3x_kfiou_aux-83c78863.log`](./rotated_fcos_r50_fpn_dota_le90_3x_kfiou_aux-83c78863.log) |

## HRSC2016 le90 zoo

**Training split: ImageSets trainval.** **Eval split: ImageSets test** (453 images; 15 empty). Whole-image `keep_ratio` + pad-32 (no native sliding windows).

**mAP** below is **`make eval-val`** mAP50 (rotated IoU ≥ 0.50, `evaluation.final_nms_iou_threshold` **0.1**; deploy production NMS **0.3**).

### Oriented R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `oriented_rcnn_hrsc2016_le90_3x` | 3× keep-ratio + pad-32, rotate ±20° | 90.41% | [`hrsc2016_le90_3x.json`](../configs/oriented_rcnn/hrsc2016_le90_3x.json) | [`oriented_rcnn_r50_fpn_hrsc2016_le90_3x-dd8a195b.json`](./oriented_rcnn_r50_fpn_hrsc2016_le90_3x-dd8a195b.json) | [`oriented_rcnn_r50_fpn_hrsc2016_le90_3x-dd8a195b.log`](./oriented_rcnn_r50_fpn_hrsc2016_le90_3x-dd8a195b.log) |

### Rotated Faster R-CNN R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `rotated_faster_rcnn_hrsc2016_le90_3x` | 3× keep-ratio + pad-32, rotate ±20° | 88.77% | [`hrsc2016_le90_3x.json`](../configs/rotated_faster_rcnn/hrsc2016_le90_3x.json) | [`rotated_faster_rcnn_r50_fpn_hrsc2016_le90_3x-a755ae37.json`](./rotated_faster_rcnn_r50_fpn_hrsc2016_le90_3x-a755ae37.json) | [`rotated_faster_rcnn_r50_fpn_hrsc2016_le90_3x-a755ae37.log`](./rotated_faster_rcnn_r50_fpn_hrsc2016_le90_3x-a755ae37.log) |

### Rotated FCOS R50-FPN

| Slug | Recipe | eval-val mAP50 | Config | Final config | Final log |
|------|--------|----------------|--------|--------------|-----------|
| `rotated_fcos_hrsc2016_le90_3x` | 3× decoded rIoU, rotate ±20° | 88.34% | [`hrsc2016_le90_3x.json`](../configs/rotated_fcos/hrsc2016_le90_3x.json) | [`rotated_fcos_r50_fpn_hrsc2016_le90_3x-ad7b8f44.json`](./rotated_fcos_r50_fpn_hrsc2016_le90_3x-ad7b8f44.json) | [`rotated_fcos_r50_fpn_hrsc2016_le90_3x-ad7b8f44.log`](./rotated_fcos_r50_fpn_hrsc2016_le90_3x-ad7b8f44.log) |

Per-class AP and eval reports: [`docs/eval-reports/`](../docs/eval-reports/) (tracked reports; raw `predictions.json` under gitignored [`predictions/`](../predictions/) for `odet viewer`).
