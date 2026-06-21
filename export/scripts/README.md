# export/scripts

CLI utilities (run from repository root so `oriented_det` and `export` resolve on `PYTHONPATH`):

| Script | Purpose |
|--------|---------|
| [export_onnx.py](export_onnx.py) | PyTorch → ONNX (`backbone`, `retinanet_heads`, `faster_rcnn_pre_nms`). |
| [build_faster_rcnn_savedmodel.py](build_faster_rcnn_savedmodel.py) | ONNX + meta → Keras detect bundle (`keras_model.keras`, exact rotated NMS). |
| [onnx_to_savedmodel.py](onnx_to_savedmodel.py) | ONNX → TFLite / optional TF via `onnx2tf` (experimental for Faster R-CNN). |
| [predict_savedmodel.py](predict_savedmodel.py) | Smoke-test a detect bundle or legacy SavedModel. |
| [save_predictions_tf.py](save_predictions_tf.py) | Val split inference via Keras bundle → `predictions.json` (`--ort-device`, `make preds ORT_DEVICE=cuda`). |
| [to_tflite.py](to_tflite.py) | SavedModel → float32 TFLite. |

See [../README.md](../README.md) for full usage.
