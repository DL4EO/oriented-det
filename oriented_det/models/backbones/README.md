# backbones

Feature extractors for detectors:

| Module | Role |
|--------|------|
| `resnet_fpn.py` | ResNet + FPN (`setup_backbone`, MMDet-style returned layers) |
| `utils.py` | Backbone helpers shared by model constructors |

Configured via `model.backbone`, `model.fpn_returned_layers`, `model.frozen_stages` / `model.trainable_layers` in [Configuration](../../docs/user-guide/configuration.md).

`build_resnet_fpn_backbone` keeps torchvision's `FrozenBatchNorm2d` default (frozen BN statistics), matching MMRotate's `norm_eval=True` detection recipe. Pass `norm_layer=torch.nn.BatchNorm2d` explicitly to train with live batch statistics.

Parent package: [models/README.md](../README.md).
