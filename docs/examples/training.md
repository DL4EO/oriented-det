# Training Tool

Config-based training for oriented object detection.

## Running

```bash
uv pip install -e .
odet train --config configs/rotated_faster_rcnn/dota_le90_3x.json --batch-size 4
```

From the repo root you can also use `make train` (same default `CONFIG`).

## What happens

1. **Dataset** — paths and augmentations from JSON (`dataset`, `augmentation` sections)
2. **Model** — `model_type` selects Oriented R-CNN, Rotated Faster R-CNN, or Rotated RetinaNet
3. **Training loop** — AMP, gradient accumulation, checkpointing, TensorBoard (see [Training user guide](../user-guide/training.md))
4. **Validation** — metrics during training per `evaluation.*` in config

## Key concepts

- **Config inheritance** — `_base_` merges fragments from `configs/_base_/`
- **Checkpoints** — written under `runs/<model_type>/<timestamp>/`
- **Overrides** — CLI flags override JSON fields where supported (`odet train --help`)

See [Configuration](../user-guide/configuration.md) for the full schema.
