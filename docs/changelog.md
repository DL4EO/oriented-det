# Changelog

All notable changes to OrientedDet will be documented in this file.

## [0.1.0] - 2026-05-27

### Added

- Core geometry (Polygon, QBox, RBox) and transforms
- Rotated IoU, NMS, and optional GPU kernels
- DOTA loader, tiling, augmentations, oriented mAP
- Airbus Playground CSV dataset support
- Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet
- JSON config training via `odet train`
- Pretrained weights on Hugging Face Hub (`dl4eo/oriented-det-pretrained`), including Oriented R-CNN 1× DOTA le90 (`74.79%` eval-val mAP50)
- MkDocs user guide and API reference

### Notes

- Public home: https://github.com/DL4EO/oriented-det
