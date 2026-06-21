# GitHub Actions workflows

## `test.yml`

Runs on every push/PR to `main` / `master`:

- `make check-configs`
- `pytest tests/`

## `publish.yml`

Builds `dist/*` and uploads to PyPI or TestPyPI.

| Trigger | Destination |
|---------|-------------|
| Push tag `v*.*.*` (e.g. `v0.1.0`) | **PyPI** (`environment: pypi`) |
| Manual **workflow_dispatch** | **TestPyPI** (default) or **PyPI** |

### One-time setup (Trusted Publishing — recommended)

Configure on [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org) for project **`oriented-det`**:

1. **Account settings → Publishing** → Add a new pending publisher (or per-project **Publishing** tab).
2. **PyPI publisher**
   - Owner: `DL4EO`
   - Repository: `oriented-det`
   - Workflow: `publish.yml`
   - Environment: `pypi` (production) and/or `testpypi` (TestPyPI)
3. Create matching **GitHub environments** (Settings → Environments): `pypi`, `testpypi`. Optional: require reviewers on `pypi`.
4. Remove the `password:` lines from `publish.yml` once OIDC is verified — `pypa/gh-action-pypi-publish` uses Trusted Publishing automatically when no token is passed and `id-token: write` is set.

### Fallback: API tokens

If Trusted Publishing is not configured yet, set repository secrets:

| Secret | Used for |
|--------|----------|
| `TESTPYPI_API_TOKEN` | TestPyPI uploads (`workflow_dispatch` → testpypi) |
| `PYPI_API_TOKEN` | Production PyPI (tag push or manual → pypi) |

Create tokens at **Account settings → API tokens** on each index (scope: entire account or project `oriented-det`).

### Manual TestPyPI run from GitHub

1. **Actions** → **Publish** → **Run workflow**
2. Branch: `main`
3. Target: `testpypi`
4. Run

### Production release from GitHub

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

The tag push starts `publish.yml` and uploads to PyPI.
