"""CLI wiring smoke (no Makefile)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_cli_export_commands_map_to_python_modules() -> None:
    """ONNX export is ``odet export``; former TF / hyphenated top-level names stay gone."""
    from export import cli
    from oriented_det import cli as odet_cli

    assert odet_cli._COMMANDS["export"][0] == "export.cli"
    for name in ("export-tf", "export-onnx", "export-detect", "export-savedmodel", "export-preds"):
        assert name not in odet_cli._COMMANDS
    for name, module in (
        ("onnx", "export.scripts.export_onnx"),
        ("infer", "export.scripts.infer_onnx"),
        ("demo", "export.scripts.demo"),
        ("preds", "export.scripts.save_predictions_onnx"),
    ):
        assert cli._COMMANDS[name][0] == module
    assert "tf" not in cli._COMMANDS
    assert "detect" not in cli._COMMANDS
    assert "savedmodel" not in cli._COMMANDS


def test_odet_export_help(capsys) -> None:
    from oriented_det.cli import main as odet_main

    old_argv = sys.argv
    try:
        odet_main(["export"])
    finally:
        sys.argv = old_argv
    out = capsys.readouterr().out
    assert "odet export" in out
    assert "onnx" in out
    assert "TensorFlow" in out


def test_odet_rejects_legacy_hyphenated_export_commands() -> None:
    from oriented_det.cli import main as odet_main

    old_argv = sys.argv
    try:
        with pytest.raises(SystemExit) as ei:
            odet_main(["export-onnx"])
        assert ei.value.code == 2
    finally:
        sys.argv = old_argv


def test_odet_export_rejects_tf(capsys) -> None:
    from oriented_det.cli import main as odet_main

    old_argv = sys.argv
    try:
        with pytest.raises(SystemExit) as ei:
            odet_main(["export", "tf"])
        assert ei.value.code == 2
    finally:
        sys.argv = old_argv
    err = capsys.readouterr()
    assert "Unknown command" in err.err


@pytest.mark.parametrize(
    "argv",
    [
        ["onnx", "--help"],
        ["infer", "--help"],
        ["demo", "--help"],
        ["preds", "--help"],
        ["export-onnx", "--help"],
    ],
)
def test_export_subcommand_help(argv: list[str], capsys) -> None:
    from export.cli import main

    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code in (0, None)
    out = capsys.readouterr().out
    assert "usage:" in out.lower() or "Usage:" in out


def test_unknown_tf_command_exits(capsys) -> None:
    from export.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["tf"])
    assert ei.value.code == 2
    err = capsys.readouterr()
    assert "Unknown command" in err.err


def test_copy_runtime_sidecars_writes_consumer_stack(tmp_path: Path) -> None:
    from export.runtime import copy_runtime_sidecars

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    written = copy_runtime_sidecars(onnx)
    rel = {str(p.relative_to(tmp_path)) for p in written}
    assert rel == {
        "nms.py",
        "preprocess.py",
        "postprocess.py",
        "runtime.py",
        "ort_runtime.py",
        "requirements-runtime.txt",
        "demo.py",
        "infer_onnx.py",
        "demo/planes_pleiades_neo.jpg",
    }
    src = Path(__file__).resolve().parents[1]
    assert (tmp_path / "nms.py").read_bytes() == (src / "nms.py").read_bytes()
    assert (tmp_path / "runtime.py").read_bytes() == (src / "runtime.py").read_bytes()
    assert (tmp_path / "demo.py").read_bytes() == (src / "scripts" / "demo.py").read_bytes()
    assert (tmp_path / "requirements-runtime.txt").read_text(encoding="utf-8") == (
        src / "requirements-runtime.txt"
    ).read_text(encoding="utf-8")


def test_copy_runtime_sidecars_fallback_requirements(tmp_path: Path, monkeypatch) -> None:
    import export.runtime as runtime

    missing = tmp_path / "export_pkg"
    missing.mkdir()
    for name in ("nms.py", "preprocess.py", "postprocess.py", "runtime.py", "ort_runtime.py"):
        (missing / name).write_text("# stub\n", encoding="utf-8")
    (missing / "scripts").mkdir()
    (missing / "scripts" / "demo.py").write_text("# demo\n", encoding="utf-8")
    (missing / "scripts" / "infer_onnx.py").write_text("# infer\n", encoding="utf-8")
    # Intentionally omit requirements-runtime.txt and demo/*.jpg
    monkeypatch.setattr(runtime, "_EXPORT_DIR", missing)

    onnx = tmp_path / "out" / "model.onnx"
    onnx.parent.mkdir()
    onnx.write_bytes(b"stub")
    written = runtime.copy_runtime_sidecars(onnx)
    rel = {str(p.relative_to(onnx.parent)) for p in written}
    assert "requirements-runtime.txt" in rel
    assert "demo/planes_pleiades_neo.jpg" not in rel
    text = (onnx.parent / "requirements-runtime.txt").read_text(encoding="utf-8")
    assert "onnxruntime" in text


def test_zip_bundle_skips_pycache(tmp_path: Path) -> None:
    import zipfile

    from export.scripts.zip_bundle import zip_bundle

    bundle = tmp_path / "onnx_export"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"stub")
    (bundle / "runtime.py").write_text("ok\n", encoding="utf-8")
    cache = bundle / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"cache")
    (bundle / "nms.pyc").write_bytes(b"cache")
    out = tmp_path / "onnx_export.zip"
    zip_bundle(bundle, out)
    names = zipfile.ZipFile(out).namelist()
    assert "onnx_export/model.onnx" in names
    assert "onnx_export/runtime.py" in names
    assert all("__pycache__" not in n and not n.endswith(".pyc") for n in names)
