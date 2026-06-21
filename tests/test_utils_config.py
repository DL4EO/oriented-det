import json
import sys
from pathlib import Path

import pytest

from oriented_det.utils import config


def test_load_config_from_dict_and_overrides():
    cfg = config.load_config({"model": {"lr": 0.1, "layers": 2}})
    assert cfg["model"]["lr"] == 0.1
    overridden = config.load_config(cfg.to_dict(), overrides={"model.lr": 0.01})
    assert overridden["model"]["lr"] == 0.01


def test_apply_overrides_with_strings(tmp_path: Path):
    data = {"trainer": {"epochs": 12, "device": "cpu"}}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))
    cfg = config.load_config(path, overrides=["trainer.device='cuda'", "trainer.epochs=24"])
    assert cfg["trainer"]["device"] == "cuda"
    assert cfg.get("trainer.epochs") == 24


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "missing.json")


def test_invalid_override_string():
    with pytest.raises(ValueError):
        config.apply_overrides({}, ["invalid"])


def test_merge_dicts():
    """Test merge_dicts utility."""
    a = {"a": 1, "b": {"x": 1}}
    b = {"b": {"y": 2}, "c": 3}
    merged = config.merge_dicts(a, b)
    assert merged["a"] == 1
    assert merged["b"]["x"] == 1
    assert merged["b"]["y"] == 2
    assert merged["c"] == 3


def test_frozen_config_immutability():
    """Test that FrozenConfig is truly immutable."""
    cfg = config.load_config({"key": "value"})
    with pytest.raises(TypeError):
        cfg["new_key"] = "new_value"
    with pytest.raises(TypeError):
        cfg["key"] = "modified"


def test_config_inheritance_single_base(tmp_path: Path):
    """Test config inheritance with a single base config."""
    # Create base config
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"a": 1, "b": {"x": 10, "y": 20}}))
    
    # Create child config
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({
        "_base_": "base.json",
        "b": {"y": 30, "z": 40},
        "c": 3
    }))
    
    cfg = config.load_config(child_path)
    assert cfg["a"] == 1  # From base
    assert cfg["b"]["x"] == 10  # From base
    assert cfg["b"]["y"] == 30  # Overridden
    assert cfg["b"]["z"] == 40  # Added
    assert cfg["c"] == 3  # Added


def test_config_inheritance_multiple_bases(tmp_path: Path):
    """Test config inheritance with multiple base configs."""
    # Create base configs
    base1_path = tmp_path / "base1.json"
    base1_path.write_text(json.dumps({"a": 1, "b": 2}))
    
    base2_path = tmp_path / "base2.json"
    base2_path.write_text(json.dumps({"b": 3, "c": 4}))
    
    # Create child config
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({
        "_base_": ["base1.json", "base2.json"],
        "c": 5
    }))
    
    cfg = config.load_config(child_path)
    assert cfg["a"] == 1  # From base1
    assert cfg["b"] == 3  # From base2 (overrides base1)
    assert cfg["c"] == 5  # Overridden in child


def test_config_inheritance_nested(tmp_path: Path):
    """Test nested config inheritance (base configs with their own bases)."""
    # Create grandparent config
    grandparent_path = tmp_path / "grandparent.json"
    grandparent_path.write_text(json.dumps({"a": 1}))
    
    # Create parent config with base
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps({
        "_base_": "grandparent.json",
        "b": 2
    }))
    
    # Create child config
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({
        "_base_": "parent.json",
        "c": 3
    }))
    
    cfg = config.load_config(child_path)
    assert cfg["a"] == 1  # From grandparent
    assert cfg["b"] == 2  # From parent
    assert cfg["c"] == 3  # From child


def test_config_inheritance_circular_dependency(tmp_path: Path):
    """Test that circular dependencies are detected."""
    # Create config A that depends on B
    config_a = tmp_path / "a.json"
    config_a.write_text(json.dumps({"_base_": "b.json", "value": "a"}))
    
    # Create config B that depends on A (circular!)
    config_b = tmp_path / "b.json"
    config_b.write_text(json.dumps({"_base_": "a.json", "value": "b"}))
    
    with pytest.raises(ValueError, match="Circular dependency"):
        config.load_config(config_a)


def test_config_inheritance_missing_base(tmp_path: Path):
    """Test that missing base config raises FileNotFoundError."""
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({"_base_": "nonexistent.json", "a": 1}))
    
    with pytest.raises(FileNotFoundError):
        config.load_config(child_path)


def test_config_inheritance_relative_paths(tmp_path: Path):
    """Test that relative paths are resolved correctly."""
    # Create subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    
    # Create base config in parent directory
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"a": 1}))
    
    # Create child config in subdirectory
    child_path = subdir / "child.json"
    child_path.write_text(json.dumps({"_base_": "../base.json", "b": 2}))
    
    cfg = config.load_config(child_path)
    assert cfg["a"] == 1
    assert cfg["b"] == 2


def test_config_inheritance_with_overrides(tmp_path: Path):
    """Test that overrides work with inherited configs."""
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"a": 1, "b": {"x": 10}}))
    
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({"_base_": "base.json", "c": 3}))
    
    cfg = config.load_config(child_path, overrides={"a": 2, "b.x": 20})
    assert cfg["a"] == 2  # Overridden
    assert cfg["b"]["x"] == 20  # Overridden
    assert cfg["c"] == 3  # From child


def test_muted_keys_ignored():
    """Test that _muted_ prefixed keys are ignored."""
    cfg = config.load_config({
        "learning_rate": 0.01,
        "_muted_learning_rate": 0.005,
        "training": {
            "epochs": 10,
            "_muted_epochs": 5,
        },
    })
    assert cfg["learning_rate"] == 0.01
    assert "_muted_learning_rate" not in cfg
    assert cfg["training"]["epochs"] == 10
    assert "_muted_epochs" not in cfg["training"]


def test_base_framework_fallback_prefers_repo_over_vendored_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Editable installs: full repo configs win over vendored package subset."""
    odet_root = tmp_path / "oriented-det"
    pkg = odet_root / "oriented_det"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    vendored = pkg / "configs" / "_base_" / "schedules"
    vendored.mkdir(parents=True)
    (vendored / "1x.json").write_text(json.dumps({"training": {"num_epochs": 12}}))
    repo_sched = odet_root / "configs" / "_base_" / "schedules"
    repo_sched.mkdir(parents=True)
    (repo_sched / "6x.json").write_text(json.dumps({"training": {"num_epochs": 72}}))

    product = tmp_path / "product" / "configs" / "rotated_faster_rcnn"
    product.mkdir(parents=True)
    recipe = product / "recipe.json"
    recipe.write_text(json.dumps({"_base_": ["../_base_/schedules/6x.json"], "data_loader": {"batch_size": 4}}))

    monkeypatch.setenv("ORIENTED_DET_ROOT", str(odet_root))
    monkeypatch.setitem(sys.modules, "oriented_det", type(sys)("oriented_det"))
    sys.modules["oriented_det"].__file__ = str(pkg / "__init__.py")

    cfg = config.load_config(recipe)
    assert cfg["training"]["num_epochs"] == 72
    assert cfg["data_loader"]["batch_size"] == 4


def test_config_inheritance_odet_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@odet: resolves paths under ORIENTED_DET_ROOT."""
    odet_root = tmp_path / "oriented-det"
    (odet_root / "configs/_base_/models").mkdir(parents=True)
    (odet_root / "configs/_base_/models/m.json").write_text(
        json.dumps({"model_type": "rotated_faster_rcnn"})
    )
    monkeypatch.setenv("ORIENTED_DET_ROOT", str(odet_root))

    product_configs = tmp_path / "external-project" / "configs"
    product_configs.mkdir(parents=True)
    (product_configs / "dataset.json").write_text(
        json.dumps({"dataset": {"format": "airbus_playground", "data_root": "/data"}})
    )
    recipe = product_configs / "recipe.json"
    recipe.write_text(
        json.dumps({
            "_base_": [
                "dataset.json",
                "@odet:configs/_base_/models/m.json",
            ],
            "data_loader": {"batch_size": 8},
        })
    )

    cfg = config.load_config(recipe)
    assert cfg["dataset"]["format"] == "airbus_playground"
    assert cfg["model_type"] == "rotated_faster_rcnn"
    assert cfg["data_loader"]["batch_size"] == 8


def test_muted_keys_from_file(tmp_path: Path):
    """Test _muted_ keys are stripped when loading from file."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "a": 1,
        "_muted_a": 99,
        "nested": {"x": 2, "_muted_x": 88},
    }))
    cfg = config.load_config(path)
    assert cfg["a"] == 1
    assert "_muted_a" not in cfg
    assert cfg["nested"]["x"] == 2
    assert "_muted_x" not in cfg["nested"]

