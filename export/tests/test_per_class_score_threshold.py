"""Inference-time override of production.per_class_score_threshold. No ONNX export."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from export.runtime import detect_array, detect_image, merge_per_class_score_threshold
from export.scripts.infer_onnx import main, parse_per_class_score_threshold

_SIDECAR = {
    "drone": 0.25,
    "shahed": 0.2,
    "aircraft": 1.1,
    "helicopter": 1.1,
}


def test_merge_partial_override_replaces_named_classes_only() -> None:
    sidecar = dict(_SIDECAR)
    override = {"Drone": 0.45, "bird": 0.5}
    merged = merge_per_class_score_threshold(sidecar, override)

    assert merged == {
        "drone": 0.45,
        "shahed": 0.2,
        "aircraft": 1.1,
        "helicopter": 1.1,
        "bird": 0.5,
    }
    assert sidecar == _SIDECAR
    assert merge_per_class_score_threshold(sidecar, None) == _SIDECAR
    assert merge_per_class_score_threshold(sidecar, None) is not sidecar
    assert merge_per_class_score_threshold(sidecar, {}) == _SIDECAR
    assert merge_per_class_score_threshold(None, None) is None
    assert merge_per_class_score_threshold(None, {"Drone": 0.45}) == {"Drone": 0.45}
    assert merge_per_class_score_threshold(
        {"drone": 0.2},
        {"drone": 0.4, "Drone": 0.45},
    ) == {"drone": 0.45}


def test_detect_array_partial_override_keeps_sidecar_classes(monkeypatch) -> None:
    """Drone uses the override; other classes keep the sidecar floors.

    A global score_threshold must not clear the per-class map. Aircraft at 0.9
    stays dropped by the sidecar floor 1.1.
    """
    from export import postprocess

    sidecar = dict(_SIDECAR)
    meta = {
        "class_names": ["drone", "shahed", "aircraft", "helicopter"],
        "production": {
            "score_threshold": 0.3,
            "per_class_score_threshold": sidecar,
            "final_nms_iou_threshold": 0.99,
            "max_detections_per_image": 20,
            "nms_class_agnostic": True,
        },
        "output_names": ["pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"],
    }
    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 8.0, 0.0],
            [40.0, 10.0, 20.0, 8.0, 0.0],
            [10.0, 80.0, 20.0, 8.0, 0.0],
            [80.0, 80.0, 20.0, 8.0, 0.0],
            [120.0, 10.0, 20.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.50, 0.40, 0.90, 0.30, 0.50], dtype=np.float32)
    labels = np.array([1, 1, 3, 2, 4], dtype=np.int64)

    def fake_pre_nms(images, onnx_path, names):
        return boxes, scores, labels, 5

    recorded = []
    real = postprocess.ort_pre_nms_to_detections

    def spy(images, onnx_path, names, fk):
        recorded.append(dict(fk))
        return real(images, onnx_path, names, fk)

    monkeypatch.setattr(postprocess, "ort_run_pre_nms", fake_pre_nms)
    monkeypatch.setattr("export.runtime.ort_pre_nms_to_detections", spy)
    blob = np.zeros((1, 3, 8, 8), dtype=np.float32)

    kept_global = detect_array(blob, "model.onnx", meta, score_threshold=0.05)
    assert Counter(d["class_name"] for d in kept_global) == Counter({"drone": 2, "shahed": 1})
    assert recorded[0]["score_threshold"] == 0.05
    assert recorded[0]["per_class_score_threshold"] is sidecar

    kept = detect_array(
        blob,
        "model.onnx",
        meta,
        score_threshold=0.05,
        per_class_score_threshold={"Drone": 0.45},
    )
    assert Counter(d["class_name"] for d in kept) == Counter({"drone": 1, "shahed": 1})
    assert kept[0]["score"] == pytest.approx(0.5)
    assert recorded[1]["score_threshold"] == 0.05
    assert recorded[1]["per_class_score_threshold"] == {
        "drone": 0.45,
        "shahed": 0.2,
        "aircraft": 1.1,
        "helicopter": 1.1,
    }
    assert sidecar["drone"] == 0.25


def test_detect_image_forwards_per_class_override(monkeypatch, tmp_path: Path) -> None:
    from export import runtime

    seen = {}

    def fake_detect_array(*args, **kwargs):
        seen["kwargs"] = kwargs
        return [{"class_name": "drone"}]

    monkeypatch.setattr(
        runtime,
        "preprocess_path",
        lambda *args, **kwargs: (np.zeros((1, 3, 4, 4), dtype=np.float32), 4, 4),
    )
    monkeypatch.setattr(runtime, "detect_array", fake_detect_array)
    image = tmp_path / "tile.jpg"
    image.write_bytes(b"unused")
    out = detect_image(
        image,
        tmp_path / "model.onnx",
        {"production": {"per_class_score_threshold": dict(_SIDECAR)}},
        score_threshold=0.05,
        per_class_score_threshold={"drone": 0.45},
    )
    assert out == [{"class_name": "drone"}]
    assert seen["kwargs"]["score_threshold"] == 0.05
    assert seen["kwargs"]["per_class_score_threshold"] == {"drone": 0.45}


def _write_bundle(tmp_path: Path) -> Path:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    meta = {
        "mode": "rotated_rtmdet_pre_nms",
        "postprocess": {"score_threshold": 0.3},
        "production": {
            "score_threshold": 0.3,
            "per_class_score_threshold": dict(_SIDECAR),
        },
    }
    (tmp_path / "model.export_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return onnx


def test_infer_flag_reaches_detect_and_prints_effective_map(tmp_path: Path, monkeypatch, capsys) -> None:
    onnx = _write_bundle(tmp_path)
    seen = {}

    def fake_detect_array(*args, **kwargs):
        seen["array"] = kwargs
        return []

    def fake_detect_image(*args, **kwargs):
        seen["image"] = kwargs
        return []

    monkeypatch.setattr("export.scripts.infer_onnx.detect_array", fake_detect_array)
    monkeypatch.setattr("export.scripts.infer_onnx.detect_image", fake_detect_image)

    main(
        [
            "--onnx",
            str(onnx),
            "--smoke",
            "--score-threshold",
            "0.5",
            "--per-class-score-threshold",
            "Drone=0.45",
            "--per-class-score-threshold",
            "shahed=0.35",
        ]
    )
    smoke_out = capsys.readouterr().out
    assert "thr   score=0.5 per_class=" in smoke_out
    assert "'drone': 0.45" in smoke_out
    assert "'shahed': 0.35" in smoke_out
    assert "'aircraft': 1.1" in smoke_out
    assert "'helicopter': 1.1" in smoke_out
    assert "'Drone'" not in smoke_out
    assert seen["array"]["score_threshold"] == 0.5
    assert seen["array"]["per_class_score_threshold"] == {"Drone": 0.45, "shahed": 0.35}

    seen.clear()
    main(["--onnx", str(onnx), "--smoke", "--score-threshold", "0.5"])
    score_only = capsys.readouterr().out
    assert "thr   score=0.5 per_class=" in score_only
    assert "'drone': 0.25" in score_only
    assert "'aircraft': 1.1" in score_only
    assert seen["array"]["score_threshold"] == 0.5
    assert seen["array"]["per_class_score_threshold"] is None

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "a.jpg").write_bytes(b"unused")
    main(
        [
            "--onnx",
            str(onnx),
            "--images",
            str(tiles),
            "--per-class-score-threshold",
            "aircraft=0.4",
        ]
    )
    assert seen["image"]["score_threshold"] is None
    assert seen["image"]["per_class_score_threshold"] == {"aircraft": 0.4}
    image_out = capsys.readouterr().out
    assert "thr   score=0.3 per_class=" in image_out
    assert "'aircraft': 0.4" in image_out
    assert "'drone': 0.25" in image_out


@pytest.mark.parametrize("bad", ["drone", "drone=", "=0.4", "drone=abc", " =0.2"])
def test_infer_rejects_malformed_per_class_score(bad: str, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--onnx", "model.onnx", "--smoke", "--per-class-score-threshold", bad])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "CLASS=FLOAT" in err
    assert bad in err


def test_parse_per_class_score_threshold_accepts_class_equals_float() -> None:
    assert parse_per_class_score_threshold(" drone = 0.45 ") == ("drone", 0.45)
    with pytest.raises(Exception, match="CLASS=FLOAT"):
        parse_per_class_score_threshold("shahed")
