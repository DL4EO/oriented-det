# Runtime (inference, checkpoints, collate)

Shared logic used by `odet` CLIs, deploy, and export — not tied to a top-level `tools/` package path.

| Module | Role |
|--------|------|
| `inference.py` | Sliding-window inference, NMS, `run_inference_auto` |
| `checkpoint.py` | `load_model_from_checkpoint`, `infer_num_classes_from_checkpoint` — Rotated RetinaNet construction mirrors `tools/train.py:create_model_from_config` (FPN layers, octave anchors, stacked head convs) so weights load without shape mismatches. |
| `collate.py` | Dataset collate and normalization constants |

Import as `from oriented_det.runtime.inference import run_inference_auto`, etc.

For image demos, use **`odet image-demo`** (config + checkpoint). The module CLI `python -m oriented_det.runtime.inference` is a legacy helper with limited `--model-type` choices (`oriented_rcnn`, `rotated_retinanet` only).
