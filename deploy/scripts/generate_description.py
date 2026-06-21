"""Generate a lightweight deploy description document from a training config.

This is intentionally dependency-light and shared across local deploy tooling and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from oriented_det.data.dota_classes import DOTA_V1_CLASSES
from oriented_det.utils.config import merge_dicts


def extract_class_names(cfg: Dict[str, Any]) -> List[str]:
    """Extract class names in a stable, user-facing order."""
    if isinstance(cfg.get("class_names"), list):
        return [str(x) for x in cfg["class_names"]]
    class_map = cfg.get("class_map")
    if isinstance(class_map, dict):
        return sorted([str(k) for k in class_map.keys()])
    return []


def detect_tier(class_names: List[str], dataset_format: str) -> str:
    fmt = str(dataset_format or "").lower()
    if fmt == "airbus_playground":
        return "airbus_playground"
    if fmt == "dota":
        if list(class_names) == list(DOTA_V1_CLASSES):
            return "dota_v1_full"
        return "dota_subset"
    return "generic"


def build_description_document(
    cfg: Dict[str, Any],
    *,
    tier: str,
    class_names: List[str],
    image_ref: Optional[str] = None,
    deploy_version: Optional[str] = None,
) -> Dict[str, Any]:
    model_type = str(cfg.get("model_type") or cfg.get("model", {}).get("model_type") or "model")
    ts = cfg.get("experiment_timestamp")
    title = f"{model_type}"
    if tier == "dota_v1_full":
        title = f"{model_type} (DOTA v1.0 full)"
    elif tier == "dota_subset":
        title = f"{model_type} (DOTA subset)"
    elif tier == "airbus_playground":
        title = f"{model_type} (Airbus Playground)"
    if ts:
        title = f"{title} — {ts}"

    doc: Dict[str, Any] = {
        "title": title,
        "name": image_ref or str(cfg.get("name") or "oriented-det"),
        "version": str(deploy_version or cfg.get("version") or "0.0.0"),
        # Optional metadata that may be supplied via deploy overrides.
        "organization": str(cfg.get("organization") or ""),
        "email": str(cfg.get("email") or ""),
        "capabilities": {
            "tier": tier,
            "tags": list(class_names),
        },
        "input": {"type": "image"},
        "definitions": {"classes": list(class_names)},
        "metadata": {
            "model_type": model_type,
            "dataset_format": str((cfg.get("dataset") or {}).get("format") or ""),
        },
    }
    # Keep output compact: drop empty optional metadata keys.
    if not doc["organization"]:
        doc.pop("organization", None)
    if not doc["email"]:
        doc.pop("email", None)
    return doc


def generate(
    config_path: Path,
    output_path: Path,
    *,
    overrides_path: Optional[Path] = None,
    image_ref: Optional[str] = None,
    deploy_version: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a JSON object.")

    overrides: Dict[str, Any] = {}
    if overrides_path is not None:
        overrides_raw = json.loads(Path(overrides_path).read_text(encoding="utf-8"))
        if isinstance(overrides_raw, dict):
            overrides = overrides_raw

    merged = merge_dicts(cfg, overrides) if overrides else dict(cfg)
    class_names = extract_class_names(merged)
    dataset_format = str((merged.get("dataset") or {}).get("format") or "")
    tier = detect_tier(class_names, dataset_format)

    doc = build_description_document(
        merged,
        tier=tier,
        class_names=class_names,
        image_ref=image_ref,
        deploy_version=deploy_version,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return {"tier": tier, "output": str(output_path)}


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--overrides", default=None, type=Path)
    p.add_argument("--image-ref", default=None, type=str)
    p.add_argument("--deploy-version", default=None, type=str)
    args = p.parse_args(list(argv) if argv is not None else None)
    generate(
        args.config,
        args.out,
        overrides_path=args.overrides,
        image_ref=args.image_ref,
        deploy_version=args.deploy_version,
    )


if __name__ == "__main__":
    main()

