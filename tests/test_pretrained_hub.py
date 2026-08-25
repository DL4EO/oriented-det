"""Tests for Hugging Face pretrained checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oriented_det.pretrained import hub
from oriented_det.pretrained.hub import (
    download_asset,
    ensure_checkpoint,
    get_pretrained_dir,
    list_assets,
    resolve_checkpoint_sidecar_config,
    resolve_checkpoint_source_recipe,
    resolve_pretrained_path,
)


def test_list_assets():
    assets = list_assets()
    assert assets["oriented_rcnn_dota_le90_1x"] == (
        "oriented_rcnn_r50_fpn_dota_le90_1x-5b128e72.pth"
    )
    assert assets["oriented_rcnn_dota_le90_3x"] == (
        "oriented_rcnn_r50_fpn_dota_le90_3x-68957f98.pth"
    )
    assert assets["rotated_retinanet_dota_le90_1x"] == (
        "rotated_retinanet_r50_fpn_dota_le90_1x-bb9a0bd2.pth"
    )
    assert assets["rotated_faster_rcnn_dota_le90_1x"] == (
        "rotated_faster_rcnn_r50_fpn_dota_le90_1x-0733c506.pth"
    )
    assert assets["rotated_faster_rcnn_dota_le90_3x"] == (
        "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    )
    assert assets["rotated_fcos_dota_le90_3x_riou"] == (
        "rotated_fcos_r50_fpn_dota_le90_3x_riou-a39c80c1.pth"
    )
    assert assets["rotated_fcos_dota_le90_3x_kfiou_aux"] == (
        "rotated_fcos_r50_fpn_dota_le90_3x_kfiou_aux-83c78863.pth"
    )


def test_resolve_hf_uri_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(tmp_path / "pretrained"))
    monkeypatch.delenv("ORIENTED_DET_PROJECT_ROOT", raising=False)
    path = resolve_pretrained_path("hf://rotated_faster_rcnn_dota_le90_3x_ce")
    assert path == (
        tmp_path / "pretrained" / "rotated_faster_rcnn_r50_fpn_dota_le90_3x_ce-c077eeee.pth"
    ).resolve()


def test_resolve_pretrained_relative_ignores_project_root(tmp_path, monkeypatch):
    product = tmp_path / "product"
    cache = tmp_path / "pretrained-cache"
    product.mkdir()
    cache.mkdir()
    monkeypatch.setenv("ORIENTED_DET_PROJECT_ROOT", str(product))
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(cache))
    hashed = "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    (cache / hashed).write_bytes(b"x")
    path = resolve_pretrained_path(f"pretrained/{hashed}")
    assert path == (cache / hashed).resolve()
    assert not (product / "pretrained").exists()


def test_resolve_pretrained_relative_slug_maps_to_hashed_filename(tmp_path, monkeypatch):
    cache = tmp_path / "pretrained-cache"
    cache.mkdir()
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(cache))
    hashed = "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    (cache / hashed).write_bytes(b"x")
    path = resolve_pretrained_path("pretrained/rotated_faster_rcnn_dota_le90_3x")
    assert path == (cache / hashed).resolve()


def test_resolve_checkpoint_sidecar_config_uses_weight_stem(tmp_path, monkeypatch):
    cache = tmp_path / "pretrained-cache"
    cache.mkdir()
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(cache))
    hashed = "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    sidecar = cache / "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.json"
    (cache / hashed).write_bytes(b"x")
    sidecar.write_text("{}", encoding="utf-8")

    path = resolve_checkpoint_sidecar_config("hf://rotated_faster_rcnn_dota_le90_3x")
    assert path == sidecar.resolve()


def test_resolve_checkpoint_source_recipe_from_weight_filename():
    recipe = resolve_checkpoint_source_recipe("rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth")
    assert recipe == "configs/rotated_faster_rcnn/dota_le90_3x.json"


def test_resolve_checkpoint_sidecar_config_ignores_run_checkpoints(tmp_path, monkeypatch):
    product = tmp_path / "product"
    cache = tmp_path / "pretrained-cache"
    run_dir = product / "runs" / "rotated_faster_rcnn" / "exp" / "checkpoints"
    run_dir.mkdir(parents=True)
    cache.mkdir()
    monkeypatch.setenv("ORIENTED_DET_PROJECT_ROOT", str(product))
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(cache))
    ckpt = run_dir / "best.pth"
    ckpt.write_bytes(b"x")
    (run_dir / "best.json").write_text("{}", encoding="utf-8")

    assert resolve_checkpoint_sidecar_config("runs/rotated_faster_rcnn/exp/checkpoints/best.pth") is None


def test_resolve_runs_relative_uses_project_root(tmp_path, monkeypatch):
    product = tmp_path / "product"
    cache = tmp_path / "pretrained-cache"
    product.mkdir()
    cache.mkdir()
    run_ckpt = product / "runs" / "rotated_faster_rcnn" / "exp" / "checkpoints" / "best.pth"
    run_ckpt.parent.mkdir(parents=True)
    run_ckpt.write_bytes(b"x")
    monkeypatch.setenv("ORIENTED_DET_PROJECT_ROOT", str(product))
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(cache))
    path = resolve_pretrained_path("runs/rotated_faster_rcnn/exp/checkpoints/best.pth")
    assert path == run_ckpt.resolve()


def test_get_pretrained_dir_defaults_to_framework_pretrained(monkeypatch):
    monkeypatch.delenv("ORIENTED_DET_PRETRAINED_DIR", raising=False)
    monkeypatch.delenv("ORIENTED_DET_PROJECT_ROOT", raising=False)
    expected = hub.get_framework_root() / "pretrained"
    assert get_pretrained_dir() == expected.resolve()


def test_ensure_checkpoint_skips_download_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(tmp_path / "pretrained"))
    ckpt = tmp_path / "pretrained" / "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake")

    with patch.object(hub, "download_asset") as mock_dl:
        result = ensure_checkpoint("hf://rotated_faster_rcnn_dota_le90_3x", quiet=True)
    mock_dl.assert_not_called()
    assert result == ckpt.resolve()


def test_ensure_checkpoint_downloads_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(tmp_path / "pretrained"))
    ckpt = tmp_path / "pretrained" / "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"

    with patch.object(hub, "download_asset", return_value=ckpt) as mock_dl:
        result = ensure_checkpoint("hf://rotated_faster_rcnn_dota_le90_3x", quiet=True)
    mock_dl.assert_called_once()
    assert result == ckpt


def test_ensure_checkpoint_unknown_file_no_download(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(tmp_path / "pretrained"))
    with patch.object(hub, "download_asset") as mock_dl:
        result = ensure_checkpoint("pretrained/not_in_manifest.pth", quiet=True)
    mock_dl.assert_not_called()
    assert not result.exists()


def test_download_asset_calls_hf_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", str(tmp_path / "pretrained"))
    filename = "rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth"
    dest = tmp_path / "pretrained" / filename
    dest.parent.mkdir(parents=True)

    def fake_download(**kwargs):
        out = Path(kwargs["local_dir"]) / kwargs["filename"]
        out.write_bytes(b"weights")
        return str(out)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        path = download_asset("rotated_faster_rcnn_dota_le90_3x")

    assert path.exists()
    assert path.name == filename


def test_download_asset_requires_hub(monkeypatch):
    monkeypatch.setenv("ORIENTED_DET_PRETRAINED_DIR", "/tmp/oriented-det-test-pretrained")
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked_import):
        with pytest.raises(ImportError, match="huggingface_hub"):
            download_asset("rotated_faster_rcnn_dota_le90_3x")


def test_manifest_is_valid_json():
    manifest_path = Path(hub.__file__).parent / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["repo_id"]
    assets = data["assets"]
    assert "rotated_retinanet_dota_le90_1x" in assets
    assert "rotated_faster_rcnn_dota_le90_3x" in assets
    for entry in assets.values():
        assert entry["filename"].endswith(".pth")
        assert len(entry["sha256"]) == 64
