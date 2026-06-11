# 3LC integration — development

Developer notes for the 3LC integration in this fork. User-facing
documentation lives in [README.md](README.md).

## Branches

| Branch | Purpose |
| --- | --- |
| `master` | Mirror of `ultralytics/master`. No fork-specific changes. |
| `develop` | The live integration branch. Feature branches PR into this. |
| `feature/<owner>/<topic>` | Short-lived feature branches off `develop`. |
| `tlc_<version>` | Historical, frozen branches. Read-only. |

## Local development

Install the dev dependencies on top of upstream's requirements. `3lc` 3.0+ is
published on 3LC's public package index, not PyPI — see the header of
[`requirements-dev.txt`](../../../requirements-dev.txt) for details.

```bash
# pip
pip install -r requirements.txt -r requirements-dev.txt

# uv
uv pip install -r requirements.txt -r requirements-dev.txt --index-strategy unsafe-best-match
```

Lint, type-check and test (scope is defined in `ruff.toml` / `ty.toml`):

```bash
ruff check
ty check
pytest -o "addopts=" tests/  # -o sidesteps upstream's broken [tool.pytest] block in pyproject.toml
```

## CI

`.github/workflows/3lc-ci.yml` runs `ruff check` and `pytest` on PRs and
pushes to `develop`. `ty check` and `ruff format --check` are configured and
runnable locally, but not yet gated in CI — they will be turned on when the
3.0 migration lands. Upstream's `ci-testing.yml` continues to fire on
`master` and is not touched here.
