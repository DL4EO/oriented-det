"""CLI wiring + export-tf orchestration smoke (no Makefile)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_cli_export_commands_map_to_python_modules() -> None:
    from oriented_det import cli

    assert not hasattr(cli, "_run_export_make")
    assert not hasattr(cli, "_EXPORT_MAKE_TARGETS")
    for name, module in (
        ("export-tf", "export.scripts.export_tf"),
        ("export-detect", "export.scripts.build_faster_rcnn_savedmodel"),
        ("export-preds", "export.scripts.save_predictions_tf"),
        ("export-onnx", "export.scripts.export_onnx"),
    ):
        assert cli._COMMANDS[name][0] == module


@pytest.mark.parametrize(
    "argv",
    [
        ["export-tf", "--help"],
        ["export-detect", "--help"],
        ["export-preds", "--help"],
    ],
)
def test_odet_export_subcommand_help(argv: list[str], capsys) -> None:
    from oriented_det.cli import main

    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code in (0, None)
    out = capsys.readouterr().out
    assert "usage:" in out.lower() or "Usage:" in out


def test_export_tf_orchestrates_onnx_then_detect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from export.scripts import export_tf

    calls: list[tuple[str, list[str]]] = []

    def fake_call(module_path: str, argv: list[str]) -> None:
        calls.append((module_path, list(argv)))
        if module_path == "export.scripts.export_onnx":
            out = Path(argv[argv.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx-stub")
            out.with_suffix(".export_meta.json").write_text("{}", encoding="utf-8")
        elif module_path == "export.scripts.build_faster_rcnn_savedmodel":
            detect = Path(argv[argv.index("--output") + 1])
            detect.mkdir(parents=True, exist_ok=True)
            (detect / "keras_model.keras").write_text("stub", encoding="utf-8")
            (detect / "model.onnx").write_bytes(b"onnx-stub")

    monkeypatch.setattr(export_tf, "_require_export_extras", lambda **kwargs: None)
    monkeypatch.setattr(export_tf, "_call_main", fake_call)

    cfg = tmp_path / "cfg.json"
    ckpt = tmp_path / "model.pth"
    cfg.write_text("{}", encoding="utf-8")
    ckpt.write_bytes(b"x")
    out = tmp_path / "odet_export"

    detect_dir = export_tf.run_export_tf(
        config=cfg,
        checkpoint=ckpt,
        output_dir=out,
        mode="faster_rcnn_pre_nms",
        height=64,
        width=64,
        skip_ort=True,
    )

    assert detect_dir == out / "detect"
    assert (out / "pre_nms.onnx").is_file()
    assert (out / "detect" / "keras_model.keras").is_file()
    assert [c[0] for c in calls] == [
        "export.scripts.export_onnx",
        "export.scripts.build_faster_rcnn_savedmodel",
    ]
    onnx_argv = calls[0][1]
    assert "--mode" in onnx_argv and "faster_rcnn_pre_nms" in onnx_argv
    assert str(out / "pre_nms.onnx") in onnx_argv
    detect_argv = calls[1][1]
    assert str(out / "detect") in detect_argv
    assert "make" not in " ".join(sum((c[1] for c in calls), [])).lower()


def test_export_tf_accepts_fcos_pre_nms_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from export.scripts import export_tf

    calls: list[tuple[str, list[str]]] = []

    def fake_call(module_path: str, argv: list[str]) -> None:
        calls.append((module_path, list(argv)))
        if module_path == "export.scripts.export_onnx":
            out = Path(argv[argv.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx-stub")
            out.with_suffix(".export_meta.json").write_text("{}", encoding="utf-8")
        elif module_path == "export.scripts.build_faster_rcnn_savedmodel":
            detect = Path(argv[argv.index("--output") + 1])
            detect.mkdir(parents=True, exist_ok=True)
            (detect / "keras_model.keras").write_text("stub", encoding="utf-8")
            (detect / "model.onnx").write_bytes(b"onnx-stub")

    monkeypatch.setattr(export_tf, "_require_export_extras", lambda **kwargs: None)
    monkeypatch.setattr(export_tf, "_call_main", fake_call)

    cfg = tmp_path / "cfg.json"
    ckpt = tmp_path / "model.pth"
    cfg.write_text("{}", encoding="utf-8")
    ckpt.write_bytes(b"x")

    export_tf.run_export_tf(
        config=cfg,
        checkpoint=ckpt,
        output_dir=tmp_path / "out",
        mode="rotated_fcos_pre_nms",
        skip_ort=True,
    )
    assert "rotated_fcos_pre_nms" in calls[0][1]


def test_export_tf_rejects_non_pre_nms_mode() -> None:
    from export.scripts import export_tf

    with pytest.raises(SystemExit, match="pre-NMS"):
        export_tf.run_export_tf(
            config=Path("c.json"),
            checkpoint=Path("m.pth"),
            output_dir=Path("out"),
            mode="backbone",  # type: ignore[arg-type]
        )


def test_export_tf_rejects_missing_paths(tmp_path: Path) -> None:
    from export.scripts import export_tf

    missing_cfg = tmp_path / "missing.json"
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"x")
    with pytest.raises(SystemExit, match="Config not found"):
        export_tf.run_export_tf(
            config=missing_cfg,
            checkpoint=ckpt,
            output_dir=tmp_path / "out",
        )

    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="Checkpoint not found"):
        export_tf.run_export_tf(
            config=cfg,
            checkpoint=tmp_path / "missing.pth",
            output_dir=tmp_path / "out",
        )


def test_export_tf_rejects_incomplete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from export.scripts import export_tf

    def fake_call(module_path: str, argv: list[str]) -> None:
        if module_path == "export.scripts.export_onnx":
            out = Path(argv[argv.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx-stub")
            out.with_suffix(".export_meta.json").write_text("{}", encoding="utf-8")
        elif module_path == "export.scripts.build_faster_rcnn_savedmodel":
            detect = Path(argv[argv.index("--output") + 1])
            detect.mkdir(parents=True, exist_ok=True)
            (detect / "keras_model.keras").write_text("stub", encoding="utf-8")
            # intentionally omit model.onnx

    monkeypatch.setattr(export_tf, "_require_export_extras", lambda **kwargs: None)
    monkeypatch.setattr(export_tf, "_call_main", fake_call)

    cfg = tmp_path / "cfg.json"
    ckpt = tmp_path / "model.pth"
    cfg.write_text("{}", encoding="utf-8")
    ckpt.write_bytes(b"x")
    with pytest.raises(SystemExit, match="incomplete"):
        export_tf.run_export_tf(
            config=cfg,
            checkpoint=ckpt,
            output_dir=tmp_path / "out",
            skip_ort=True,
        )
