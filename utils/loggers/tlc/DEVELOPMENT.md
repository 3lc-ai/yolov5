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

## Releases

There are none. The fork is not packaged, versioned or tagged — the HEAD of
`develop` is what users clone. This means every commit on `develop` must be
self-consistent: documentation and install instructions may only promise what
the code on that same commit delivers.

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

Lint, format, type-check and test (scope is defined in `ruff.toml` / `ty.toml`):

```bash
ruff check
ruff format --check utils/loggers/tlc/ tests/
ty check
pytest -o "addopts=" tests/  # -o sidesteps upstream's broken [tool.pytest] block in pyproject.toml
```

The end-to-end tests in `tests/` are skipped unless `TLC_API_KEY` is set to a
valid 3LC API key.

## CI

`.github/workflows/3lc-ci.yml` runs on PRs and pushes to `develop`:

- **lint**: `ruff check` and `ruff format --check` (no dependencies needed).
- **tests**: installs the full requirements (including `3lc` and
  `3lc-ultralytics`), then runs `ty check` and `pytest`. The end-to-end tests
  need the `TLC_API_KEY` repository secret.

Upstream's `ci-testing.yml` continues to fire on `master` and is not touched
here.

## Upstream sync (quarterly)

Merge upstream into `develop` roughly once a quarter:

```bash
git remote add ultralytics https://github.com/ultralytics/yolov5.git  # once
git fetch ultralytics
git checkout develop
git checkout -b chore/<owner>/upstream-sync
git merge ultralytics/master
```

Notes:

- Fork-owned files (`utils/loggers/tlc/**`, `tests/**`, `ruff.toml`,
  `ty.toml`, `requirements-dev.txt`, `.github/workflows/3lc-ci.yml`) are never
  touched by upstream, so conflicts are confined to the few upstream files the
  integration hooks into: `train.py`, `val.py`, `utils/loggers/__init__.py`,
  `models/yolo.py`, `utils/general.py` and `README.md`.
- On conflicts, take upstream's version and re-apply the integration hook
  points.
- PR the sync branch into `develop` so CI validates the merge, then
  fast-forward the `master` mirror to `ultralytics/master`.
