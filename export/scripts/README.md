# export/scripts

CLI utilities. Prefer **`odet`** after `pip install "oriented-det[export]"` (no Makefile). From a checkout, `PYTHONPATH` includes the repo root so `oriented_det` and `export` resolve.

| Script | `odet` command | Purpose |
|--------|----------------|---------|
| [export_tf.py](export_tf.py) | `export-tf` | ONNX pre-NMS + Keras detect bundle → `./odet_export/`. Optional `--saved-model`. |
| [export_onnx.py](export_onnx.py) | `export-onnx` | PyTorch → ONNX (`backbone`, `retinanet_heads`, `faster_rcnn_pre_nms`, `oriented_rcnn_pre_nms`, `rotated_fcos_pre_nms`). |
| [build_faster_rcnn_savedmodel.py](build_faster_rcnn_savedmodel.py) | `export-detect` | ONNX + meta → Keras detect bundle (`keras_model.keras` + `model.onnx`). |
| [build_tf_savedmodel.py](build_tf_savedmodel.py) | `export-savedmodel` | ONNX + meta → TF SavedModel (onnx2tf + TF NMS). |
| [save_predictions_tf.py](save_predictions_tf.py) | `export-preds` | Val split inference via Keras bundle → `./odet_export/predictions/<ts>/`. |
| [onnx_to_savedmodel.py](onnx_to_savedmodel.py) | — | ONNX → TFLite / optional TF via `onnx2tf` (experimental). |
| [predict_savedmodel.py](predict_savedmodel.py) | — | Smoke-test a detect bundle or SavedModel (`python -m export.scripts.predict_savedmodel`). |
| [to_tflite.py](to_tflite.py) | — | SavedModel → float32 TFLite. |

See [../README.md](../README.md) for full usage.
