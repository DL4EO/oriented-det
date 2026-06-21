# Building documentation

This site is built from the `docs/` folder with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). API pages are generated from Python docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

Project overview and install steps: [repository README on GitHub](https://github.com/DL4EO/oriented-det/blob/main/README.md).

## Build and serve

From the repository root (the `docs` target installs MkDocs into the active environment if needed):

```bash
make docs          # build static site → site/
make docs-serve    # http://127.0.0.1:8000 with auto-reload
```

Manual equivalent:

```bash
make docs-deps     # uv pip install -e ".[docs]"
python -m mkdocs build
python -m mkdocs serve
```

## Deploy (e.g. GitHub Pages)

```bash
mkdocs gh-deploy
```

Requires `gh-pages` branch permissions on the remote.

## Writing

- Add Markdown under `docs/` and register the page in `mkdocs.yml` → `nav`.
- Links to files **outside** `docs/` (e.g. `tools/README.md`, `configs/*.json`) should use full GitHub URLs so the static site does not warn about missing targets.
- API reference pages use `::: oriented_det.module` blocks; keep docstrings in sync with the code.

Folder layout and maintainer checklist: [docs/README.md](https://github.com/DL4EO/oriented-det/blob/main/docs/README.md) (source on GitHub; not duplicated in this site because `index.md` is the home page).
