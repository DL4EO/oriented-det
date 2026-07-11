# Demo images

Place test images here (e.g. `.jpg`, `.jpeg`, `.png`) to run inference with `odet image-demo` or `make demo`.

## Getting demo images

- **Sample image:** Download at least one example into `demo/` (top-level files only are picked up by directory mode):

  ```bash
  curl -sL -o demo/demo.jpg https://raw.githubusercontent.com/open-mmlab/mmrotate/main/demo/demo.jpg
  ```

  Add more `.jpg` / `.jpeg` / `.png` files in `demo/` as needed; `make demo` processes **all** of them.

- **Your own images:** Use any aerial/satellite or oriented-object image. DOTA-style tiled crops (e.g. 1024×1024) work well if your config was trained on similar data. Use another folder: `make demo DEMO_DIR=/path/to/images`.

## Usage

From the **oriented-det** project root:

```bash
# Single image
odet image-demo demo/demo.jpg runs/<model>/<timestamp>/config.json runs/.../checkpoints/checkpoint_best.pth --out-file result.jpg

# All top-level images in demo/ (outputs to demo/out by default)
odet image-demo demo configs/.../config.json runs/.../checkpoints/best.pth --out-dir demo/out
```

**Makefile (repo root):** uses the **latest** experiment under `runs/` (latest checkpoint under `runs/`):

```bash
make demo              # all *.jpg / *.jpeg / *.png directly under demo/ → demo/out/<stem>_detections.png
# Optional: DEMO_DIR=other/dir IMAGE_DEMO_DEVICE=cpu IMAGE_DEMO_OUT_DIR=/tmp/demo-out
```

Only images **directly inside** `DEMO_DIR` are used (not subfolders such as `demo/out/`). See `tools/README.md` for more options (`--score-thr`, `--nms-thr`, `--classes`, `--zoom`, `--device`, `--overlap-pixels` / `--overlap-ratio`). If the image size matches the model input from your config (e.g. 1024×1024), inference uses a single forward pass; otherwise images are padded or tiled to that size. With `--zoom 2` or `--zoom 4`, inference runs on the zoomed image and the final visualization remains at the original size.
