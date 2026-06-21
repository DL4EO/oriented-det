"""ROI grouped cross-entropy curriculum (coarse-to-fine classification in one run)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def grouped_ce_alpha_for_epoch(
    epoch: int,
    *,
    enabled: bool,
    schedule_type: Optional[str],
    start_epoch: int,
    end_epoch: int,
    power: float = 1.0,
) -> float:
    """Return grouped-loss mix factor in [0, 1] (1 = fully grouped, 0 = fine CE only)."""
    if not enabled:
        return 0.0
    sched = (schedule_type or "step").strip().lower()
    if sched in ("step", "steps", "hard"):
        return 1.0 if epoch < end_epoch else 0.0
    if sched in ("linear_ramp", "ramp", "linear"):
        if end_epoch <= start_epoch:
            return 0.0
        if epoch <= start_epoch:
            return 1.0
        if epoch >= end_epoch:
            return 0.0
        t = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return float((1.0 - t) ** power)
    return 0.0


@dataclass(frozen=True)
class GroupedCeSpec:
    """Resolved group membership for ROI classifier curriculum."""

    group_index_lists: Tuple[Tuple[int, ...], ...]  # 1-indexed foreground class ids per group
    class_in_group_id: Tuple[int, ...]  # length num_classes+1; -1 bg/unmapped, else group index


def build_grouped_ce_spec(
    groups: Dict[str, Sequence[str]],
    class_map: Dict[str, int],
    num_foreground_classes: int,
) -> GroupedCeSpec:
    """Map config group names to 1-indexed class id lists.

    Each class name may appear in at most one group. Classes not listed in any group
    use fine-grained CE only (even when grouped_alpha > 0).
    """
    if num_foreground_classes < 1:
        raise ValueError("num_foreground_classes must be >= 1 for grouped CE")

    name_to_group: Dict[str, int] = {}
    group_lists: List[List[int]] = []
    for g_idx, class_names in enumerate(groups.values()):
        ids: List[int] = []
        for name in class_names:
            if name not in class_map:
                continue
            cid = int(class_map[name])
            if cid < 1 or cid > num_foreground_classes:
                continue
            if name in name_to_group:
                raise ValueError(
                    f"Class {name!r} appears in multiple roi_grouped_ce groups "
                    f"(groups {name_to_group[name]} and {g_idx})"
                )
            name_to_group[name] = g_idx
            ids.append(cid)
        group_lists.append(sorted(set(ids)))

    class_in_group = [-1] * (num_foreground_classes + 1)
    for g_idx, ids in enumerate(group_lists):
        for cid in ids:
            class_in_group[cid] = g_idx

    frozen_lists = tuple(tuple(ids) for ids in group_lists)
    return GroupedCeSpec(group_index_lists=frozen_lists, class_in_group_id=tuple(class_in_group))


def configure_roi_grouped_ce(
    model: object,
    loss_config: object,
    class_map: Optional[Dict[str, int]],
    *,
    num_foreground_classes: int,
    device: Optional["torch.device"] = None,
) -> bool:
    """Attach grouped CE schedule and group indices to a two-stage detector model.

    Returns True when grouped CE is enabled and configured.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for grouped CE configuration.")

    enabled = bool(getattr(loss_config, "roi_grouped_ce_enabled", False))
    groups = getattr(loss_config, "roi_grouped_ce_groups", None) or {}
    if not enabled or not groups:
        if hasattr(model, "clear_roi_grouped_ce"):
            model.clear_roi_grouped_ce()
        return False

    if not class_map:
        raise ValueError("roi_grouped_ce requires class_map from the training dataset")

    spec = build_grouped_ce_spec(groups, class_map, num_foreground_classes)
    if not any(spec.group_index_lists):
        raise ValueError("roi_grouped_ce_groups resolved to no class ids; check class names")

    if device is None:
        if hasattr(model, "parameters"):
            params = list(model.parameters())
            device = params[0].device if params else torch.device("cpu")
        else:
            device = torch.device("cpu")

    class_in_group_t = torch.tensor(spec.class_in_group_id, dtype=torch.long, device=device)

    if not hasattr(model, "set_roi_grouped_ce"):
        raise TypeError(f"{type(model).__name__} does not support ROI grouped CE")

    model.set_roi_grouped_ce(
        group_index_lists=list(spec.group_index_lists),
        class_in_group_id=class_in_group_t,
        schedule_type=getattr(loss_config, "roi_grouped_ce_schedule_type", None),
        schedule_start_epoch=int(getattr(loss_config, "roi_grouped_ce_schedule_start_epoch", 0) or 0),
        schedule_end_epoch=int(getattr(loss_config, "roi_grouped_ce_schedule_end_epoch", 0) or 0),
        schedule_power=float(getattr(loss_config, "roi_grouped_ce_schedule_power", 1.0) or 1.0),
    )
    return True
