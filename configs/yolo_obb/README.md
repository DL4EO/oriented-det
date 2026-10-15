# YOLO-OBB configs

Native **YOLO-OBB** (`model_type: yolo_obb`) — C2f CSP + PAFPN, distribution focal box head, task-aligned assigner. Clean-room. No pretrained backbone.

| Recipe | Config | Notes |
|--------|--------|-------|
| COCO 2017 **300e S** | [`coco_300e_s.json`](coco_300e_s.json) | Horizontal pretrain. 80 classes, 640, batch 16, SGD `0.0025` cosine to `2.5e-5`, warmup 3 epochs (22179 steps). Mosaic off for the last 10 epochs. Angle head held at 0. |
| DOTA le90 1× **S** | [`dota_le90_1x_s.json`](dota_le90_1x_s.json) | Debug only. 12 epochs, warmup 500, mosaic off, random rotate off. |
| DOTA le90 **6× S** | [`dota_le90_6x_s.json`](dota_le90_6x_s.json) | First real run. 72 epochs, warmup 10000, CachedMosaic, per-tile random rotate p=0.5 ±180°. Mosaic off for the last 10 epochs. Not trained yet. |
| DOTA le90 **6× S from COCO** | [`dota_le90_6x_s_coco.json`](dota_le90_6x_s_coco.json) | Loads the COCO trunk. Restarts the 15-way classifier and the angle head. Mosaic stays on for all 72 epochs. Warmup 1000 steps. Angle loss weight 0.2. |

One Scaleway L4-1-24G (€0.79/h), the same GPU as the measured DOTA run (72 epochs, 1024, batch 4, 16 hours). Idle GPU time is accepted. Scaled from that run: COCO 300 epochs at 640 is about 180–240 hours, the DOTA finetune is 16 hours, and the GPU cost is about €155–200 (under €300).

```bash
mkdir -p /path/to/data/coco
cd /path/to/data/coco
curl -L -O http://images.cocodataset.org/zips/train2017.zip
curl -L -O http://images.cocodataset.org/zips/val2017.zip
curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```

That layout is `train2017/`, `val2017/`, and `annotations/instances_{train,val}2017.json`. The loader reads the detection `bbox` as a horizontal le90 box and ignores the segmentation polygon.

```bash
odet train --config configs/yolo_obb/coco_300e_s.json
```

Point `checkpoint.load_from_checkpoint` in [`dota_le90_6x_s_coco.json`](dota_le90_6x_s_coco.json) at the COCO run's best checkpoint (the placeholder is not a committed weight), then:

```bash
odet train --config configs/yolo_obb/dota_le90_6x_s_coco.json
odet train --config configs/yolo_obb/dota_le90_6x_s.json
```

DOTA recipes always see **1024×1024**. The COCO pretrain sees **640×640**, with mosaic off for the last 10 epochs, no random rotate, and no MixUp. Mosaic builds a 2× canvas, random-resizes it (`mosaic_ratio_range` 0.5–1.5), then crops back to the recipe canvas. Each of the four tiles is rotated before paste when random rotate is on. Val / inference never mosaic or rotate. MixUp is not used. Image normalization matches [`dota_le90_1x_s.json`](dota_le90_1x_s.json).

Loss weights in the model base: ProbIoU **7.5**, DFL **1.5**, le90 angle L1 **0.2**. Nano is `configs/_base_/models/yolo_obb_n.json` (swap `backbone` to `yolo_obb_n`). There is no separate nano DOTA recipe.

Base fragments: [`yolo_obb_n.json`](../_base_/models/yolo_obb_n.json), [`yolo_obb_s.json`](../_base_/models/yolo_obb_s.json).

Parent: [configs/README.md](../README.md).
