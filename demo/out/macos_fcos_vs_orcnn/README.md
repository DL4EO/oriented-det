# macOS FCOS vs Oriented R-CNN

Local MPS smoke outputs from comparing Hub 3× DOTA checkpoints on Apple Silicon.

| File | Contents |
|------|----------|
| `summary.json` | Timed runs, detection counts, class histograms |
| `*_oriented_rcnn.png` / `*_rotated_fcos.png` | Side-by-side visualizations |
| `cli_demo_fcos.png` | `odet image-demo` smoke on `demo/demo.jpg` |

Protocol: `--device mps`, `score>=0.3`, `nms<=0.1`, slugs `oriented_rcnn_dota_le90_3x` and `rotated_fcos_dota_le90_3x`.
