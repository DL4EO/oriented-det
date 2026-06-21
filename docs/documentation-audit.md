# Documentation audit log

Tracking alignment of human docs with `oriented_det/`, `tools/`, and `configs/`.

| Surface | Status | Notes |
|---------|--------|-------|
| `docs/user-guide/configuration.md` | **done** | Canonical config + recipe catalog (public DOTA recipes only) |
| `docs/user-guide/training.md` | **done** | Config section → link |
| `docs/user-guide/geometry.md` | **done** | Unchanged core |
| `docs/user-guide/operations.md` | **done** | ops README links to user guide |
| `docs/user-guide/data.md` | **done** | DOTA + Airbus Playground |
| `docs/user-guide/models.md` | **done** | Three detectors; Hub weights; trimmed MMRotate marketing |
| `docs/user-guide/utils.md` | **done** | Points to Configuration |
| `docs/index.md` | **done** | DL4EO URLs; Hub pretrained |
| `docs/getting-started/*` | **done** | installation, quickstart (`odet train`), tools |
| `docs/examples/*` | **done** | `odet train` / `odet preds` |
| `demo/README.md` | **done** | `odet image-demo`, `make demo` |
| `docs/api/*.md` | **done** | mkdocstrings |
| `configs/README.md` | **done** | No private product repos |
| `README.md` | **done** | DL4EO publish; tiled eval; no full-val Make targets |
| `Makefile` | **done** | Removed preds-dota-val, metrics-dota-val, viewer-dota-val, view-dataset |
| `tools/README.md` | **done** | Tiled preds/metrics/viewer only |
| `pretrained/README.md` | **done** | Hub + make preds/metrics |
| `oriented_det/**/README.md` | **done** | External project root wording |
| `export/**`, `deploy/**` | **done** | In-repo example paths |
| `tests/README.md` | **done** | Full module list + CI note |
| `.github/workflows/test.yml` | **done** | CPU pytest on push/PR |
| Install / dev workflow | **done** (2026-06-02) | **uv** only in human docs; `make install` / `docs-deps` use `uv pip`; PyPI consumers may still use `pip install oriented-det` |
| `docs/getting-started/tools.md` | **done** (2026-06-02) | `odet preds`, `predictions/` at repo root, model types, tile overlap default 200 |

**Validation:** `mkdocs build`; `pytest tests/test_training_config_strict.py tests/test_utils_config.py tests/test_pretrained_hub.py`

---

## Deep analysis (2026-06-15)

Full report: [`code-analysis-report.md`](code-analysis-report.md).

| Surface | Status | Notes |
|---------|--------|-------|
| `docs/code-analysis-report.md` | **new** | Inventory matrix, automated checks, flow traces, training run cross-ref, findings F1–F10 |
| `docs/getting-started/tools.md` | **done** (2026-06-15) | Full `odet` command table (F3) |
| `docs/user-guide/models.md` | **done** (2026-06-15) | Training vs eval/export paths; GitHub link for pretrained README (F2, F5) |
| `docs/examples/inference.md` | **done** (2026-06-15) | Analysis artifacts + `--no-per-class-threshold-analysis` (F4) |
| `docs/user-guide/data.md` | **done** (2026-06-15) | Out-of-tree links → GitHub URLs (F5) |
| `docs/user-guide/configuration.md` | **done** (2026-06-15) | RetinaNet `rpn_*`/`roi_*` reuse documented (F6) |
| `pretrained/README.md`, config Hub tables | **done** (2026-06-15) | `eval_map50` vs training final mAP clarified (F1) |
| Config schema ↔ `config.py` | **verified** | All sections field-aligned (F8) |
| `export/PARITY.md` ↔ contract/tests | **verified** | (F9) |
| Run `rotated_retinanet/20260612-121232` | **verified** | Matches `dota_le90_3x` recipe; final mAP 75.94% (F10) |

**Automated validation (2026-06-15):** `make docs` pass (4 link warnings before fixes); 47 targeted pytest pass; 383 full pytest pass with editable install; Hub manifest 4/4 runs present.

**Follow-up fixes (2026-06-15):** Applied doc-only actions F1–F6 from `code-analysis-report.md` §7.
