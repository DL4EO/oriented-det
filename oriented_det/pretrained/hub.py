"""Download OrientedDet pretrained checkpoints from the Hugging Face Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

_MANIFEST_NAME = "manifest.json"
_HF_URI_PREFIX = "hf://"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def get_framework_root() -> Path:
    """Oriented-det install / source tree root (where ``pretrained/`` lives by default).

    Not affected by ``ORIENTED_DET_PROJECT_ROOT`` (product repos use that only for
    ``runs/``, ``predictions/``, and non-pretrained relative checkpoint paths).
    """
    # /.../<repo>/oriented_det/pretrained/hub.py -> parents[2] is <repo>
    return Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Product repo root for ``runs/`` and other project-relative paths.

    Mirrors ``oriented_det.train.utils.get_project_root`` without importing the
    training package (which may import PyTorch).
    """
    env = os.environ.get("ORIENTED_DET_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return get_framework_root()


def get_pretrained_dir() -> Path:
    """Directory for shared Hub checkpoints (decoupled from ``ORIENTED_DET_PROJECT_ROOT``)."""
    env = os.environ.get("ORIENTED_DET_PRETRAINED_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_framework_root() / "pretrained"


def _default_pretrained_dir() -> Path:
    return get_pretrained_dir()


def load_manifest() -> Dict[str, Any]:
    """Load the bundled pretrained asset manifest."""
    manifest_path = _package_dir() / _MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def list_assets() -> Dict[str, str]:
    """Map manifest slug -> published ``.pth`` filename on the Hub."""
    manifest = load_manifest()
    assets = manifest.get("assets", {})
    return {
        key: entry["filename"]
        for key, entry in assets.items()
        if isinstance(entry, dict) and entry.get("filename")
    }


def _repo_settings(manifest: Dict[str, Any]) -> tuple[str, str]:
    repo_id = os.environ.get("ORIENTED_DET_HF_REPO_ID", manifest.get("repo_id", "")).strip()
    if not repo_id:
        raise ValueError("manifest.json is missing repo_id and ORIENTED_DET_HF_REPO_ID is unset")
    revision = os.environ.get("ORIENTED_DET_HF_REVISION", manifest.get("revision", "main")).strip()
    return repo_id, revision


def _asset_entry_for_name(name: str, manifest: Dict[str, Any]) -> Optional[tuple[str, Dict[str, Any]]]:
    assets: Dict[str, Any] = manifest.get("assets", {})
    if name in assets:
        return name, assets[name]

    stem = Path(name).stem
    if stem in assets:
        return stem, assets[stem]

    if name.endswith(".pth"):
        for key, entry in assets.items():
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename", "")
            if filename == name or Path(filename).name == name:
                return key, entry
            if Path(filename).stem == stem:
                return key, entry
    return None


def _filename_for_manifest_ref(ref: str, manifest: Dict[str, Any]) -> str:
    found = _asset_entry_for_name(ref, manifest)
    if found is not None:
        return found[1]["filename"]
    if ref.endswith(".pth"):
        return ref
    return f"{ref}.pth"


def _parse_hf_uri(value: str) -> Optional[str]:
    if not value.startswith(_HF_URI_PREFIX):
        return None
    ref = value[len(_HF_URI_PREFIX) :].strip()
    if not ref:
        raise ValueError(f"Invalid Hugging Face checkpoint URI: {value!r}")
    if "/" in ref and not ref.endswith(".pth"):
        raise ValueError(
            f"Invalid Hugging Face checkpoint URI: {value!r} "
            "(use hf://<slug> or hf://<filename.pth>)"
        )
    return ref


def _is_pretrained_relative(raw: str) -> bool:
    parts = Path(raw).parts
    return bool(parts) and parts[0] == "pretrained"


def resolve_pretrained_path(
    checkpoint: PathLike,
    *,
    pretrained_dir: Optional[Path] = None,
) -> Path:
    """Resolve a config checkpoint path to an absolute local path (no download)."""
    raw = str(checkpoint).strip()
    hf_ref = _parse_hf_uri(raw)
    if hf_ref is not None:
        base = pretrained_dir or _default_pretrained_dir()
        manifest = load_manifest()
        filename = _filename_for_manifest_ref(hf_ref, manifest)
        return (base / filename).resolve()

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    if _is_pretrained_relative(raw):
        base = pretrained_dir or _default_pretrained_dir()
        rel = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
        resolved = (base / rel).resolve()
        if resolved.exists():
            return resolved
        manifest = load_manifest()
        found = _asset_entry_for_name(rel.name, manifest)
        if found is not None:
            return (base / found[1]["filename"]).resolve()
        return resolved

    return (get_project_root() / path).resolve()


def resolve_checkpoint_sidecar_config(
    checkpoint: PathLike,
    *,
    pretrained_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return the ``.json`` sidecar next to a local pretrained checkpoint, when present.

    This is intentionally scoped to files under the configured pretrained directory so
    regular run checkpoints keep using their experiment ``config.json``.
    """
    path = resolve_pretrained_path(checkpoint, pretrained_dir=pretrained_dir)
    base = (pretrained_dir or _default_pretrained_dir()).resolve()
    try:
        path.parent.resolve().relative_to(base)
    except ValueError:
        return None

    sidecar = path.with_suffix(".json")
    return sidecar.resolve() if sidecar.is_file() else None


def resolve_checkpoint_source_recipe(checkpoint: PathLike) -> Optional[str]:
    """Return the manifest ``source_recipe`` for a registered checkpoint reference."""
    manifest = load_manifest()
    raw = str(checkpoint).strip()
    lookup = raw
    hf_ref = _parse_hf_uri(raw)
    if hf_ref is not None:
        lookup = hf_ref
    else:
        lookup = Path(raw).name

    found = _asset_entry_for_name(lookup, manifest)
    if found is None:
        return None
    source_recipe = found[1].get("source_recipe")
    return str(source_recipe) if source_recipe else None


def download_asset(
    asset_name: str,
    *,
    dest: Optional[Path] = None,
    pretrained_dir: Optional[Path] = None,
) -> Path:
    """Download a manifest asset from Hugging Face Hub.

    Args:
        asset_name: Manifest slug (e.g. ``rotated_retinanet_dota_le90_1x``) or ``.pth`` filename.
        dest: Optional explicit destination file. Defaults to ``<pretrained_dir>/<filename>``.

    Returns:
        Path to the downloaded checkpoint file.
    """
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "HF_HUB_OFFLINE is set; cannot download pretrained weights from Hugging Face Hub."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download pretrained checkpoints. "
            "Reinstall oriented-det (huggingface_hub is a core dependency)."
        ) from e

    manifest = load_manifest()
    lookup_name = asset_name
    hf_ref = _parse_hf_uri(asset_name)
    if hf_ref is not None:
        lookup_name = hf_ref

    found = _asset_entry_for_name(lookup_name, manifest)
    if found is None:
        known = ", ".join(sorted(manifest.get("assets", {})))
        raise KeyError(f"Unknown pretrained asset {asset_name!r}. Known assets: {known}")

    _key, entry = found
    filename = entry["filename"]
    repo_id, revision = _repo_settings(manifest)

    out_dir = pretrained_dir or _default_pretrained_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = (dest or out_dir / filename).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=target.parent,
    )
    downloaded_path = Path(downloaded).resolve()
    if downloaded_path != target and downloaded_path.name == target.name:
        return downloaded_path
    return downloaded_path if downloaded_path.exists() else target


def ensure_checkpoint(
    checkpoint: PathLike,
    *,
    pretrained_dir: Optional[Path] = None,
    quiet: bool = False,
) -> Path:
    """Return a local checkpoint path, downloading from Hugging Face Hub when registered and missing."""
    path = resolve_pretrained_path(checkpoint, pretrained_dir=pretrained_dir)
    if path.exists():
        return path

    manifest = load_manifest()
    raw = str(checkpoint).strip()
    lookup = path.name
    hf_ref = _parse_hf_uri(raw)
    if hf_ref is not None:
        lookup = hf_ref
    elif _is_pretrained_relative(raw):
        lookup = Path(raw).name

    if _asset_entry_for_name(lookup, manifest) is None:
        return path

    if not quiet:
        repo_id, _ = _repo_settings(manifest)
        print(f"Pretrained checkpoint not found at {path}; downloading from {repo_id} ...")

    return download_asset(lookup, dest=path, pretrained_dir=pretrained_dir)
