"""Lightweight configuration helpers with strong validation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, MutableSequence, Sequence
import ast
import json
import os

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None  # type: ignore

# Keys starting with this prefix are ignored when loading configs.
# Use for keeping alternative values visible without affecting behavior, e.g.:
#   "learning_rate": 0.01,
#   "_muted_learning_rate": 0.005  # ignored
MUTED_KEY_PREFIX = "_muted_"


def _ensure_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Configuration root must be a mapping.")


def _deep_freeze(node: Any) -> Any:
    if isinstance(node, Mapping):
        return FrozenConfig({k: _deep_freeze(v) for k, v in node.items()})
    if isinstance(node, list):
        return tuple(_deep_freeze(v) for v in node)
    return node


def _deep_clone(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {k: _deep_clone(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deep_clone(v) for v in node]
    return node


def _set_by_path(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = target
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[parts[-1]] = value


def _parse_scalar(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


@dataclass(frozen=True)
class FrozenConfig(Mapping[str, Any]):
    """Immutable configuration wrapper with dotted-key access."""

    _data: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cursor: Any = self
        for part in dotted_key.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                return default
        return cursor

    def to_dict(self) -> Dict[str, Any]:
        def thaw(value: Any) -> Any:
            if isinstance(value, FrozenConfig):
                return {k: thaw(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [thaw(v) for v in value]
            return value

        return {k: thaw(v) for k, v in self.items()}


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = _deep_clone(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = _deep_clone(value)
    return result


def apply_overrides(cfg: Mapping[str, Any], overrides: Sequence[str] | Mapping[str, Any]) -> Dict[str, Any]:
    result = _deep_clone(cfg)
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            _set_by_path(result, key, value)
    else:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override '{override}' must contain '='.")
            key, value = override.split("=", 1)
            _set_by_path(result, key, _parse_scalar(value))
    return result


def _load_from_path(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configs.")
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    return _ensure_mapping(data)


# _base_ path prefix: paths are relative to oriented-det repository root (ORIENTED_DET_ROOT).
ODET_BASE_PREFIX = "@odet:"


def _oriented_det_root() -> Path:
    """oriented-det repository root (``ORIENTED_DET_ROOT``)."""
    raw = os.environ.get("ORIENTED_DET_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    try:
        import oriented_det  # noqa: WPS433 — runtime package location
    except ImportError as exc:
        raise FileNotFoundError(
            "Cannot resolve @odet: base — set ORIENTED_DET_ROOT or install oriented_det."
        ) from exc
    return Path(oriented_det.__file__).resolve().parent.parent


def _framework_config_roots() -> list[Path]:
    """Framework config directories to try for ``_base_`` fallbacks (in order).

    Resolution order:
    - ``ORIENTED_DET_CONFIG_ROOT`` (explicit override, sole entry when set)
    - Editable checkout: ``<repo>/configs`` (full tree; preferred over vendored subset)
    - Vendored wheel subset: ``oriented_det/configs``
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(root: Path | None) -> None:
        if root is None:
            return
        resolved = root.expanduser().resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)

    env = os.environ.get("ORIENTED_DET_CONFIG_ROOT")
    if env:
        _add(Path(env))
        return roots

    try:
        import oriented_det

        pkg = Path(oriented_det.__file__).resolve().parent
        repo_configs = pkg.parent / "configs"
        vendored = pkg / "configs"
        # Repo tree is authoritative in editable installs; vendored may omit schedules like 6x.
        if repo_configs.is_dir() and repo_configs != vendored:
            _add(repo_configs)
        _add(vendored)
    except Exception:
        pass
    return roots


def _framework_config_root() -> Path | None:
    """Primary framework config tree (first entry from :func:`_framework_config_roots`)."""
    roots = _framework_config_roots()
    return roots[0] if roots else None


def _normalize_base_ref_for_framework(base_path: str) -> str:
    """Strip leading ``../`` so ``../_base_/models/x.json`` -> ``_base_/models/x.json``."""
    ref = base_path.replace("\\", "/").lstrip("/")
    while ref.startswith("../"):
        ref = ref[3:]
    if ref.startswith("./"):
        ref = ref[2:]
    return ref


def _resolve_base_path(base_path: str, current_file: Path) -> Path:
    """Resolve a _base_ reference: @odet:, absolute, file-relative, or framework fallback.

    External projects can keep dataset bases locally and inherit
    model/schedule bases from the framework ``configs/_base_/`` tree when absent.
    """
    if base_path.startswith(ODET_BASE_PREFIX):
        rel = base_path[len(ODET_BASE_PREFIX) :].lstrip("/")
        return (_oriented_det_root() / rel).resolve()

    base_path_obj = Path(base_path)
    if base_path_obj.is_absolute():
        if not base_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {base_path_obj}")
        return base_path_obj

    local = (current_file.parent / base_path_obj).resolve()
    if local.exists():
        return local

    rel = _normalize_base_ref_for_framework(base_path)
    checked: list[Path] = []
    for fw_root in _framework_config_roots():
        fallback = (fw_root / rel).resolve()
        checked.append(fallback)
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Config file not found: {local}"
        + (f" (also checked framework: {', '.join(str(p) for p in checked)})" if checked else "")
    )


def _strip_muted_keys(cfg: Dict[str, Any], prefix: str = MUTED_KEY_PREFIX) -> None:
    """Remove keys starting with prefix recursively. Modifies cfg in place."""
    keys_to_remove = [k for k in cfg if isinstance(k, str) and k.startswith(prefix)]
    for k in keys_to_remove:
        del cfg[k]
    for v in cfg.values():
        if isinstance(v, dict):
            _strip_muted_keys(v, prefix)


def _delete_keys(cfg: Dict[str, Any], keys_to_delete: Dict[str, Any]) -> None:
    """Delete keys specified in keys_to_delete from cfg.
    
    MMRotate-style deletion: {"key": {"_delete_": True}} removes "key" from parent.
    """
    for key, value in keys_to_delete.items():
        if isinstance(value, dict) and value.get("_delete_") is True:
            if key in cfg:
                del cfg[key]
        elif isinstance(value, dict) and isinstance(cfg.get(key), dict):
            _delete_keys(cfg[key], value)


def _load_config_with_base(
    path: Path,
    *,
    visited: set[Path] | None = None,
) -> Dict[str, Any]:
    """Load config file and recursively resolve _base_ inheritance.
    
    Args:
        path: Path to config file
        visited: Set of already visited paths to detect circular dependencies
        
    Returns:
        Merged configuration dictionary with _base_ resolved
    """
    if visited is None:
        visited = set()
    
    path = path.resolve()
    
    # Detect circular dependencies
    if path in visited:
        raise ValueError(f"Circular dependency detected: {path} is already being loaded")
    visited.add(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    cfg_dict = _load_from_path(path)
    
    # Handle _base_ inheritance
    if "_base_" in cfg_dict:
        base_refs = cfg_dict.pop("_base_")
        
        # Normalize to list
        if isinstance(base_refs, str):
            base_refs = [base_refs]
        elif not isinstance(base_refs, list):
            raise TypeError(f"_base_ must be a string or list of strings, got {type(base_refs)}")
        
        # Load and merge base configs in order
        merged = {}
        for base_ref in base_refs:
            if not isinstance(base_ref, str):
                raise TypeError(f"Base reference must be a string, got {type(base_ref)}")
            base_path = _resolve_base_path(base_ref, path)
            base_cfg = _load_config_with_base(base_path, visited=visited)
            merged = merge_dicts(merged, base_cfg)
        
        # Merge current config over base configs
        merged = merge_dicts(merged, cfg_dict)
        cfg_dict = merged
    
    visited.remove(path)
    return cfg_dict


def load_config(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    overrides: Sequence[str] | Mapping[str, Any] | None = None,
) -> FrozenConfig:
    """Load a configuration file or dict and apply optional overrides.
    
    Supports MMRotate-style nested config inheritance via _base_ field.
    
    Args:
        source: Path to config file or dict
        overrides: Optional overrides to apply after loading
        
    Returns:
        FrozenConfig instance with merged configuration
        
    Examples:
        >>> # Simple config
        >>> cfg = load_config("config.json")
        
        >>> # Config with base inheritance
        >>> # config.json: {"_base_": "base.json", "lr": 0.01}
        >>> cfg = load_config("config.json")
        
        >>> # With overrides
        >>> cfg = load_config("config.json", overrides=["training.lr=0.001"])
    """
    if isinstance(source, Mapping):
        cfg_dict = _deep_clone(_ensure_mapping(source))
        _strip_muted_keys(cfg_dict)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        # Load with base inheritance support
        cfg_dict = _load_config_with_base(path)
    
    # Strip _muted_* keys (keeps alternative values visible without affecting behavior)
    _strip_muted_keys(cfg_dict)
    
    # Handle deletion syntax (must be done before overrides)
    _delete_keys(cfg_dict, cfg_dict)
    
    if overrides:
        cfg_dict = apply_overrides(cfg_dict, overrides)

    return _deep_freeze(cfg_dict)


__all__ = [
    "ODET_BASE_PREFIX",
    "FrozenConfig",
    "MUTED_KEY_PREFIX",
    "load_config",
    "merge_dicts",
    "apply_overrides",
]
