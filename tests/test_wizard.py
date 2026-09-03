"""Wizard recommendations are geometry-based (class-agnostic)."""

from io import StringIO
from contextlib import redirect_stdout

import numpy as np

from oriented_det.train.config import ModelConfig, TrainingExperimentConfig
from tools.train import (
    _fpn_level_name,
    _wizard_fcos_box_reg_lines,
    _wizard_fcos_fpn_recommendation,
    print_wizard_recommendations,
)


def _widths_with_percentiles(p10: float, p50: float, n: int = 1000) -> np.ndarray:
    """Array whose np.percentile p10/p50 match the requested pixel widths."""
    n10 = int(np.ceil(0.10 * (n - 1))) + 1
    widths = np.full(n, p50, dtype=np.float32)
    widths[:n10] = p10
    return widths


def test_fpn_level_name_from_stride():
    assert _fpn_level_name(4) == "P2"
    assert _fpn_level_name(8) == "P3"
    assert _fpn_level_name(16) == "P4"
    assert _fpn_level_name(7) == "finest"


def test_fpn_advice_planes_like_sizes_is_class_agnostic():
    """Planes-scale GT (~3.5/8.5 cells on P3) still gets kfiou advice, not DOTA class names."""
    widths = _widths_with_percentiles(28.3, 67.9)
    line, small = _wizard_fcos_fpn_recommendation(8, widths)
    assert small
    assert "P3" in line
    assert "28.3" in line and "67.9" in line
    assert "3.5" in line and "8.5" in line
    assert "kfiou" in line
    assert "keep P3" in line
    assert "vehicle" not in line.lower()
    assert "ship" not in line.lower()
    assert "plane" not in line.lower()
    assert "1–2 cells" not in line
    assert "1-2 cells" not in line


def test_fpn_advice_tiny_boxes_reports_share_not_class():
    widths = np.concatenate(
        [np.full(20, 12.0, dtype=np.float32), np.full(80, 40.0, dtype=np.float32)]
    )
    line, small = _wizard_fcos_fpn_recommendation(8, widths)
    assert small
    assert "20% of boxes occupy <2 cells" in line
    assert "kfiou" in line
    assert "vehicle" not in line.lower()


def test_fpn_advice_large_boxes_skips_kfiou_nudge():
    widths = _widths_with_percentiles(80.0, 160.0)
    line, small = _wizard_fcos_fpn_recommendation(8, widths)
    assert not small
    assert "looks adequate" in line
    assert "kfiou" not in line


def test_box_reg_l1_small_objects_suggests_kfiou_aux_without_class_names():
    lines = _wizard_fcos_box_reg_lines(
        "l1",
        None,
        0.0,
        small_vs_stride=True,
        aux_angle_weight=1.0,
        aux_angle_lambda=1.0,
    )
    joined = "\n".join(lines)
    assert "kfiou" in joined
    assert "aux_loss_type=kfiou" in joined
    assert "vehicle" not in joined.lower()
    assert "ship" not in joined.lower()
    assert "storage-tank" not in joined
    assert "dota_le90" not in joined


def test_box_reg_l1_large_objects_does_not_scare_about_small_classes():
    lines = _wizard_fcos_box_reg_lines(
        "l1",
        None,
        0.0,
        small_vs_stride=False,
        aux_angle_weight=1.0,
        aux_angle_lambda=1.0,
    )
    joined = "\n".join(lines)
    assert "encoded L1" in joined
    assert "aux_loss_type=kfiou" not in joined
    assert "small classes" not in joined.lower()


def _fcos_config(**model_kw) -> TrainingExperimentConfig:
    return TrainingExperimentConfig(
        model_type="rotated_fcos",
        model=ModelConfig(
            fpn_strides=[8, 16, 32, 64, 128],
            box_reg_loss_type="l1",
            max_detections_per_image=2000,
            fcos_center_sample_radius=1.5,
            **model_kw,
        ),
    )


def _stats_from_widths(widths: np.ndarray, objects_per_image: float = 2.0) -> dict:
    n_obj = int(widths.size)
    n_img = max(1, int(round(n_obj / objects_per_image)))
    heights = widths.copy()
    return {
        "num_images": n_img,
        "num_objects": n_obj,
        "per_image_objects": np.full(n_img, objects_per_image, dtype=np.float32),
        "widths": widths,
        "heights": heights,
        "aspects": np.ones(n_obj, dtype=np.float32),
        "areas": widths * heights,
        "angles": np.zeros(n_obj, dtype=np.float32),
    }


def test_print_wizard_planes_recipe_stdout_is_class_agnostic():
    widths = _widths_with_percentiles(28.3, 67.9)
    buf = StringIO()
    with redirect_stdout(buf):
        print_wizard_recommendations(_stats_from_widths(widths), _fcos_config(), world_size=1)
    out = buf.getvalue()
    assert "fpn_strides: finest=8px" in out
    assert "GT width p10/p50=28.3/67.9px" in out
    assert "cells on P3" in out
    assert "prefer decoded `kfiou`" in out
    for banned in ("small-vehicle", "Small vehicles", "storage-tank", "ship/small"):
        assert banned not in out
