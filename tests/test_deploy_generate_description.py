"""Tests for deploy/scripts/generate_description.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GEN_PATH = _ROOT / "deploy/scripts/generate_description.py"
_spec = importlib.util.spec_from_file_location("deploy_generate_description", _GEN_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_class_names = _mod.extract_class_names
detect_tier = _mod.detect_tier
build_description_document = _mod.build_description_document
generate = _mod.generate

from oriented_det.data.dota_classes import DOTA_V1_CLASSES


def test_extract_class_names_from_class_names():
    cfg = {"class_names": ["a", "b"]}
    assert extract_class_names(cfg) == ["a", "b"]


def test_extract_class_names_from_class_map():
    cfg = {"class_map": {"b": 2, "a": 1}}
    assert extract_class_names(cfg) == ["a", "b"]


def test_detect_tier_dota_full():
    assert detect_tier(list(DOTA_V1_CLASSES), "dota") == "dota_v1_full"


def test_detect_tier_dota_subset():
    assert detect_tier(["plane", "ship"], "dota") == "dota_subset"


def test_detect_tier_airbus():
    assert detect_tier(["A", "B"], "airbus_playground") == "airbus_playground"


def test_detect_tier_hrsc2016():
    assert detect_tier(["ship"], "hrsc2016") == "hrsc2016"


def test_detect_tier_generic():
    assert detect_tier(["x"], "custom") == "generic"


def test_build_dota_full_title():
    cfg = {
        "model_type": "rotated_faster_rcnn",
        "experiment_timestamp": "20260101-000000",
        "dataset": {"format": "dota"},
        "class_names": list(DOTA_V1_CLASSES),
    }
    doc = build_description_document(cfg, tier="dota_v1_full", class_names=list(DOTA_V1_CLASSES))
    assert "DOTA v1.0" in doc["title"]
    assert doc["capabilities"]["tags"] == list(DOTA_V1_CLASSES)
    assert "input" in doc and "definitions" in doc


def test_generate_writes_file(tmp_path: Path):
    cfg = {
        "model_type": "rotated_faster_rcnn",
        "dataset": {"format": "dota"},
        "class_names": ["plane"],
    }
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "description.json"
    meta = generate(conf, out, overrides_path=None)
    assert meta["tier"] == "dota_subset"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["capabilities"]["tags"] == ["plane"]
    assert "_publish" not in json.dumps(data)


def test_generate_merge_overrides(tmp_path: Path):
    cfg = {"model_type": "m", "class_names": ["plane"], "dataset": {"format": "dota"}}
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(cfg), encoding="utf-8")
    ovr = tmp_path / "overrides.json"
    ovr.write_text(json.dumps({"organization": "OrgX", "email": "e@x.org"}), encoding="utf-8")
    out = tmp_path / "description.json"
    generate(conf, out, overrides_path=ovr)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["organization"] == "OrgX"
    assert data["email"] == "e@x.org"


def test_image_ref_wins_over_overrides_name(tmp_path: Path):
    cfg = {"model_type": "m", "class_names": ["plane"], "dataset": {"format": "dota"}}
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(cfg), encoding="utf-8")
    ovr = tmp_path / "overrides.json"
    ovr.write_text(json.dumps({"name": "should-not-win"}), encoding="utf-8")
    out = tmp_path / "description.json"
    generate(conf, out, overrides_path=ovr, image_ref="eu.gcr.io/proj/oriented-det")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["name"] == "eu.gcr.io/proj/oriented-det"


def test_deploy_version_wins_over_overrides_and_config(tmp_path: Path):
    cfg = {
        "model_type": "m",
        "class_names": ["plane"],
        "dataset": {"format": "dota"},
        "experiment_timestamp": "20990101-000000",
    }
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(cfg), encoding="utf-8")
    ovr = tmp_path / "overrides.json"
    ovr.write_text(json.dumps({"version": "9.9.9", "organization": "Org"}), encoding="utf-8")
    out = tmp_path / "description.json"
    generate(
        conf,
        out,
        overrides_path=ovr,
        deploy_version="0.7.1",
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "0.7.1"
    assert data["organization"] == "Org"
