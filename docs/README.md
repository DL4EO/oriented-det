# Documentation (MkDocs)

This folder is the source for the **MkDocs** documentation site. It contains the full user guide, API reference (generated from docstrings), and examples.

For project overview, installation, and quick start, see the [repository README](https://github.com/DL4EO/oriented-det/blob/main/README.md).

## Build and serve

From the repository root (installs MkDocs into the active environment if missing):

```bash
make docs          # build → site/
make docs-serve    # live preview at http://127.0.0.1:8000
```

Or install dependencies once and run MkDocs directly:

```bash
make docs-deps     # uv pip install -e ".[docs]"
python -m mkdocs build
python -m mkdocs serve
```

Output is in `site/`.

Then open http://127.0.0.1:8000 in your browser.

## Deploy (GitHub Pages)

The public site is deployed by `.github/workflows/pages.yml`.

One-time repository setup: in **Settings → Pages**, set **Source** to **GitHub Actions**.

After that, pushes to `main` that touch docs/source files publish to:

```text
https://dl4eo.github.io/oriented-det/
```

## Structure

- **docs/** — Source Markdown (this folder)
- **roadmap.md** — Public release plan (v0.2–v1.0)
- **user-guide/configuration.md** — Canonical JSON training config reference
- **eval-reports/** — Published eval-val reports per Hub slug (markdown + analysis JSON); see [eval-reports/README.md](eval-reports/README.md). Raw `predictions.json` lives in gitignored `predictions/`.
- **code-analysis-report.md** — Deep code/doc analysis (2026-06-15): inventory, validation, findings
- **mkdocs.yml** — MkDocs configuration and nav
- **site/** — Generated HTML (created by `mkdocs build`)

API pages are generated from Python docstrings via `mkdocstrings`. See [documentation.md](documentation.md) for more on writing and building the docs.
