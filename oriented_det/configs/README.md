# Built-in configs (PyPI)

This directory is a **vendored copy** of selected files from the repository [`configs/`](../../configs/) tree. It is included in the `oriented-det` wheel so `pip install oriented-det` can resolve paths like `configs/oriented_rcnn/dota_le90_1x.json` without a git checkout.

## Source of truth

Edit configs only under **`configs/`** at the repo root, then sync:

```bash
make sync-configs
```

The list of files to ship is **`vendored_manifest.txt`** in this folder (one path per line, relative to `configs/`).

## Drift check

Before commit or release:

```bash
make check-configs
```

CI runs the same check. `make build` also depends on `check-configs`.

## Developing new recipes

- Add or change JSON under repo **`configs/`**.
- To ship a recipe on PyPI, add its path (and any new `_base_` fragments it needs) to **`vendored_manifest.txt`**, then run **`make sync-configs`**.
