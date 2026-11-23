"""Consumer stack must import and run without oriented-det."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ONNX = _REPO / "onnx_export" / "model.onnx"

_BLOCK_PRELUDE = f"""
import sys
sys.path.insert(0, {_REPO.as_posix()!r})
class _Block:
    def find_spec(self, fullname, path, target=None):
        if fullname and (fullname == "oriented_det" or fullname.startswith("oriented_det.")):
            raise ImportError("oriented_det blocked for standalone export runtime")
        return None
sys.meta_path.insert(0, _Block())
for _name in list(sys.modules):
    if _name == "oriented_det" or _name.startswith("oriented_det."):
        del sys.modules[_name]
"""

_BLOCK_ORIENTED = """
import sys
class _Block:
    def find_spec(self, fullname, path, target=None):
        if fullname and (fullname == "oriented_det" or fullname.startswith("oriented_det.")):
            raise ImportError("oriented_det blocked for standalone export runtime")
        if fullname and (fullname == "export" or fullname.startswith("export.")):
            raise ImportError("repo export package blocked for copied bundle")
        return None
sys.meta_path.insert(0, _Block())
"""


def _isolated_env(pythonpath: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(pythonpath)
    return env


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_PRELUDE + "\n" + code],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def test_consumer_modules_import_without_oriented_det() -> None:
    proc = _run(
        "from export.preprocess import preprocess_rgb\n"
        "from export.nms import rotated_nms\n"
        "from export.postprocess import finalize_detections_numpy\n"
        "from export.runtime import detect_image, draw_detections, load_export_meta\n"
        "from export.scripts import demo, infer_onnx\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_demo_help_without_oriented_det() -> None:
    proc = _run(
        "from export.cli import main\n"
        "try:\n"
        "    main(['demo', '--help'])\n"
        "except SystemExit as e:\n"
        "    raise SystemExit(e.code or 0)\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower() or "Usage:" in proc.stdout


@pytest.mark.skipif(not _ONNX.is_file(), reason="onnx_export/model.onnx not present")
def test_demo_runs_without_oriented_det() -> None:
    proc = _run(
        "import sys\n"
        "from export.scripts.demo import main\n"
        f"sys.argv = ['export-demo', '--onnx', {_ONNX.as_posix()!r}]\n"
        "main()\n"
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "NMS OK" in proc.stdout


def test_copied_bundle_imports_without_repo_export(tmp_path: Path) -> None:
    from export.runtime import copy_runtime_sidecars

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    copy_runtime_sidecars(onnx)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BLOCK_ORIENTED
            + "from runtime import detect_image, load_export_meta\n"
            "from nms import rotated_nms\n"
            "from preprocess import preprocess_rgb\n"
            "from postprocess import finalize_detections_numpy\n"
            "print('ok')\n",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_env(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.skipif(not _ONNX.is_file(), reason="onnx_export/model.onnx not present")
def test_copied_demo_runs_without_repo_export(tmp_path: Path) -> None:
    from export.runtime import copy_runtime_sidecars

    shutil.copy2(_ONNX, tmp_path / "model.onnx")
    onnx_data = Path(str(_ONNX) + ".data")
    if onnx_data.is_file():
        shutil.copy2(onnx_data, tmp_path / "model.onnx.data")
    meta = _ONNX.with_suffix(".export_meta.json")
    if meta.is_file():
        shutil.copy2(meta, tmp_path / "model.export_meta.json")
    copy_runtime_sidecars(tmp_path / "model.onnx")
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "demo.py"), "--onnx", str(tmp_path / "model.onnx")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_env(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "NMS OK" in proc.stdout
