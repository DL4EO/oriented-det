# Runtime (inference, checkpoints, collate)

Shared logic used by `odet` CLIs and deploy — not tied to a top-level `tools/` package path.

| Module | Role |
|--------|------|
| `inference.py` | Sliding-window inference (DOTA `fixed`/`crop` when the raster exceeds the canvas), whole-image **pad** / **keep_ratio** (HRSC2016, same as training), NMS, `run_inference_auto` |
| `checkpoint.py` | `load_model_from_checkpoint`, `infer_num_classes_from_checkpoint` — infers foreground class counts from R-CNN heads or RetinaNet `head.conv_bbox`/`head.conv_cls` shapes; Rotated RetinaNet construction mirrors `tools/train.py:create_model_from_config` (FPN layers, octave anchors, stacked head convs) so weights load without shape mismatches. `roi_use_hbb_for_matching` is passed only to Oriented R-CNN (not Rotated Faster R-CNN). |
| `collate.py` | Dataset collate and normalization constants. Train order: optional Albumentations → spatial resize → random flips → optional random rotate (`enable_random_rotate`) → `pad_size_divisor` → batch pad → normalize. After per-image `pad_size_divisor`, tensors are padded to the batch max H×W so `keep_ratio` can stack (MMRotate DataContainer). `content_size` stays per-image; production inference is still one image + divisor pad. |

Import as `from oriented_det.runtime.inference import run_inference_auto`, etc.

When inference loads a registered checkpoint from `pretrained/` with that checkpoint's manifest `source_recipe`, `checkpoint.py` prefers the checkpoint sidecar JSON (`<weight-stem>.json`) if present. A different user-provided config is kept as-is. This preserves the exact shipped class names and production settings for `odet image-demo`, `odet preds`, and deploy.

For image demos, use **`odet image-demo`** (config + checkpoint). The module CLI `python -m oriented_det.runtime.inference` is a legacy helper with limited `--model-type` choices (`oriented_rcnn`, `rotated_retinanet` only).
