"""DOTA v1.0 Task 1 submission helpers and unlabeled test image discovery."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from oriented_det.data.build import collect_split_images
from oriented_det.data.dota import collect_dota_unlabeled_image_paths
from oriented_det.data.dota_classes import DOTA_V1_CLASSES
from oriented_det.data.dota_task1 import (
    dummy_task1_line,
    format_task1_line,
    predictions_json_to_task1,
    rbox_to_task1_poly,
    task1_filename,
    write_task1_submission,
)


def _dota_config() -> SimpleNamespace:
    ds = SimpleNamespace(
        format="dota",
        same_folder=False,
        filter_empty_gt=False,
        difficult_strategy="drop",
        allowed_classes=None,
        ignore_labels=None,
    )
    return SimpleNamespace(dataset=ds)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_format_task1_line_and_poly():
    line = format_task1_line("P0003", 0.85, (1, 2, 3, 4, 5, 6, 7, 8))
    assert line == "P0003 0.8500 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00"

    poly = rbox_to_task1_poly([10.0, 10.0, 20.0, 10.0, 0.0])
    assert len(poly) == 8
    # Axis-aligned box centered at (10, 10), 20×10 → corners at x∈{0,20}, y∈{5,15}.
    xs, ys = poly[0::2], poly[1::2]
    assert set(xs) == {0.0, 20.0}
    assert set(ys) == {5.0, 15.0}


def test_predictions_json_dummy_and_image_id(tmp_path: Path):
    payload = {
        "metadata": {"class_names": list(DOTA_V1_CLASSES)},
        "results": [
            {
                "image_name": "P0003.png",
                "predictions": [
                    {
                        "bbox": [10.0, 10.0, 20.0, 10.0, 0.0],
                        "score": 0.9,
                        "label": 1,
                        "class_name": "plane",
                    }
                ],
            }
        ],
    }
    lines = predictions_json_to_task1(payload, dummy_if_empty=True)
    assert set(lines) == set(DOTA_V1_CLASSES)
    assert len(lines["plane"]) == 1
    assert lines["plane"][0].startswith("P0003 0.9000 ")
    dummy = dummy_task1_line()
    for name in DOTA_V1_CLASSES:
        if name != "plane":
            assert lines[name] == [dummy]

    json_path = tmp_path / "predictions.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "Task1"
    zip_path = write_task1_submission(json_path, out_dir)
    txts = sorted(p.name for p in out_dir.glob("Task1_*.txt"))
    expected = sorted(task1_filename(c) for c in DOTA_V1_CLASSES)
    assert txts == expected
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert sorted(names) == expected
    assert all("/" not in n for n in names)
    plane_text = (out_dir / "Task1_plane.txt").read_text(encoding="utf-8").strip()
    assert plane_text.startswith("P0003 ")
    empty_text = (out_dir / "Task1_helicopter.txt").read_text(encoding="utf-8").strip()
    assert empty_text == dummy


def test_collect_dota_unlabeled_nested_and_flat(tmp_path: Path):
    nested = tmp_path / "test"
    _write_png(nested / "images" / "P0001.png")
    _write_png(nested / "images" / "P0002.jpg")
    nested_paths = collect_dota_unlabeled_image_paths([nested])
    assert {p.name for p in nested_paths} == {"P0001.png", "P0002.jpg"}

    flat = tmp_path / "test_images"
    _write_png(flat / "P0003.png")
    flat_paths = collect_dota_unlabeled_image_paths([flat])
    assert [p.name for p in flat_paths] == ["P0003.png"]

    via_images = collect_dota_unlabeled_image_paths([nested / "images"])
    assert {p.name for p in via_images} == {"P0001.png", "P0002.jpg"}


def test_collect_split_images_test_no_labels(tmp_path: Path):
    test_root = tmp_path / "DOTA" / "test"
    _write_png(test_root / "images" / "P0099.png")
    images, label_dir, fmt = collect_split_images(
        _dota_config(),
        tmp_path / "DOTA",
        data_split="test",
    )
    assert fmt == "dota"
    assert label_dir is None
    assert [p.name for p in images] == ["P0099.png"]

    alt = tmp_path / "other_test"
    _write_png(alt / "images" / "P0100.png")
    images2, _, _ = collect_split_images(
        _dota_config(),
        tmp_path / "DOTA",
        data_split="test",
        test_dir=alt,
    )
    assert [p.name for p in images2] == ["P0100.png"]


def test_resolve_preds_model_paths_hub_sidecar():
    from tools.save_predictions import resolve_preds_model_paths

    exp, ckpt, cfg = resolve_preds_model_paths(
        experiment_dir=None,
        checkpoint="hf://oriented_rcnn_dota_le90_3x",
        config_path=None,
        auto_detect=False,
    )
    assert exp == ""
    assert ckpt == "hf://oriented_rcnn_dota_le90_3x"
    assert Path(cfg).name == "oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.json"
    assert Path(cfg).is_file()


def test_resolve_preds_model_paths_checkpoint_needs_config(tmp_path: Path):
    from tools.save_predictions import resolve_preds_model_paths

    ckpt = tmp_path / "orphan.pth"
    ckpt.write_bytes(b"x")
    try:
        resolve_preds_model_paths(
            experiment_dir=None,
            checkpoint=str(ckpt),
            config_path=None,
            auto_detect=False,
        )
    except ValueError as exc:
        assert "sidecar" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for checkpoint without sidecar")


def test_dota_submit_inference_keeps_overlap_copies(monkeypatch, tmp_path):
    captured = {}

    def fake_resolve(**_kwargs):
        return "exp", "ckpt.pth", "cfg.json"

    def fake_run(**kwargs):
        captured.update(kwargs)
        pred_dir = tmp_path / "pred_out"
        pred_dir.mkdir()
        json_path = pred_dir / "predictions.json"
        json_path.write_text(json.dumps({"metadata": {}, "results": []}), encoding="utf-8")
        return {"predictions_json": str(json_path), "output_dir": str(pred_dir)}

    monkeypatch.setattr("tools.save_predictions.resolve_preds_model_paths", fake_resolve)
    monkeypatch.setattr("tools.save_predictions.run_inference_and_save", fake_run)

    from tools.dota_task1_submit import main

    out = tmp_path / "Task1"
    main(
        [
            "--checkpoint",
            "hf://oriented_rcnn_dota_le90_3x",
            "--test-dir",
            str(tmp_path / "test"),
            "--output-dir",
            str(out),
        ]
    )
    assert captured["window_margin_pixels"] == 0.0
    assert captured["nms_threshold"] == 0.1
    assert captured["run_diagnostics"] is False


def test_dota_submit_checkpoint_requires_test_dir(tmp_path: Path):
    from tools.dota_task1_submit import main

    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint",
                "hf://oriented_rcnn_dota_le90_3x",
                "--output-dir",
                str(tmp_path / "Task1"),
            ]
        )


def test_predictions_json_from_run_without_output_dir_key(tmp_path: Path):
    from tools.dota_task1_submit import _predictions_json_from_run

    pred = tmp_path / "predictions.json"
    pred.write_text("{}", encoding="utf-8")
    found = _predictions_json_from_run(
        {"total_images": 1},
        output_dir=str(tmp_path),
    )
    assert found == pred

    found_key = _predictions_json_from_run({"predictions_json": str(pred)})
    assert found_key == pred
