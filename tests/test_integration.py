"""End-to-end tests for the 3LC YOLOv5 integration.

These tests run the real train.py / val.py entry points on coco128 with the 3LC
integration active. They require a valid 3LC API key — the `TLC_API_KEY`
repository secret in CI, or the local 3LC configuration when run locally — and
fail without one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tlc_project_root(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An isolated 3LC project root, so test tables and runs don't pollute the local one."""
    return str(tmp_path_factory.mktemp("3lc_project_root"))


def run_script(args: list[str], tlc_project_root: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"TLC_PROJECT_ROOT_URL": tlc_project_root}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)


def assert_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"{' '.join(result.args[1:3])} failed\nstdout: {result.stdout[-5000:]}\nstderr: {result.stderr[-5000:]}"
    )


def test_train_one_epoch(tlc_project_root: str, tmp_path: Path) -> None:
    """Train for one epoch on coco128, creating 3LC tables and collecting metrics after training."""
    result = run_script(
        [
            "train.py",
            "--data",
            "coco128.yaml",
            "--weights",
            "yolov5n.pt",
            "--img",
            "320",
            "--batch-size",
            "16",
            "--epochs",
            "1",
            "--project",
            str(tmp_path / "runs"),
        ],
        tlc_project_root,
    )
    assert_succeeded(result)


def test_collect_metrics(tlc_project_root: str) -> None:
    """Collect metrics on the train and val splits of coco128 without training."""
    result = run_script(
        [
            "val.py",
            "--task",
            "collect",
            "--data",
            "coco128.yaml",
            "--weights",
            "yolov5n.pt",
            "--img",
            "320",
        ],
        tlc_project_root,
    )
    assert_succeeded(result)
