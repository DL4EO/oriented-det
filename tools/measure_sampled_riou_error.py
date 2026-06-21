#!/usr/bin/env python3
"""Compare sampled GPU rIoU vs exact Shapely polygon IoU.

Generates stratified oriented box pairs (squares, elongated ships, thin slivers),
computes exact IoU on CPU (Shapely) and sampling-based IoU via
``oriented_box_iou_gpu``, then reports absolute / relative error statistics.

Use this to validate geometry sampling defaults in ``oriented_det/ops/gpu_ops.py``:

  target_spacing_px=2.0, min_samples=25, max_samples=1024, min_points_short=3

Examples:
    python tools/measure_sampled_riou_error.py
    python tools/measure_sampled_riou_error.py --pairs 5000 --seed 0
    python tools/measure_sampled_riou_error.py --sweep-spacing 2 3 4 5 6 8
    python tools/measure_sampled_riou_error.py --target-spacing 4 --max-samples 1024 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from oriented_det.geometry import RBox
from oriented_det.ops.gpu_ops import (
    _DEFAULT_MAX_SAMPLES,
    _DEFAULT_MIN_SAMPLES,
    _DEFAULT_TARGET_SPACING_PX,
    _MIN_POINTS_ALONG_SHORT_SIDE,
    geometry_sample_count_for_boxes,
    oriented_box_iou_gpu,
)
from oriented_det.ops.iou import rbox_iou
from oriented_det.ops.utils import SHAPELY_AVAILABLE


@dataclass(frozen=True)
class GeometryParams:
    target_spacing_px: float = _DEFAULT_TARGET_SPACING_PX
    min_samples: int = _DEFAULT_MIN_SAMPLES
    max_samples: int = _DEFAULT_MAX_SAMPLES
    min_points_short: int = _MIN_POINTS_ALONG_SHORT_SIDE


@dataclass(frozen=True)
class PairRecord:
    category: str
    w1: float
    h1: float
    w2: float
    h2: float
    exact: float
    sampled: float
    num_samples: int

    @property
    def abs_error(self) -> float:
        return abs(self.sampled - self.exact)

    @property
    def signed_error(self) -> float:
        return self.sampled - self.exact

    @property
    def rel_error_pct(self) -> float:
        if self.exact <= 1e-8:
            return float("nan")
        return 100.0 * self.abs_error / self.exact


def _require_shapely() -> None:
    if not SHAPELY_AVAILABLE:
        print(
            "Shapely is required for exact IoU. Install oriented-det with shapely.",
            file=sys.stderr,
        )
        sys.exit(1)


def _box_tensor(cx: float, cy: float, w: float, h: float, angle: float) -> torch.Tensor:
    return torch.tensor([cx, cy, w, h, angle], dtype=torch.float32)


def _exact_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = RBox(*a.tolist())
    rb = RBox(*b.tolist())
    return float(rbox_iou(ra, rb, intersection_backend="shapely"))


def _sampled_iou_batch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    *,
    device: torch.device,
    num_samples: int,
) -> torch.Tensor:
    b1 = boxes1.to(device)
    b2 = boxes2.to(device)
    mat = oriented_box_iou_gpu(b1, b2, num_samples=num_samples)
    return torch.diagonal(mat).cpu()


def _resolve_num_samples(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    params: GeometryParams,
) -> int:
    combined = torch.cat([boxes1, boxes2], dim=0)
    return geometry_sample_count_for_boxes(
        combined,
        target_spacing_px=params.target_spacing_px,
        min_samples=params.min_samples,
        max_samples=params.max_samples,
        min_points_short=params.min_points_short,
    )


def _random_box(rng: np.random.Generator, spec: str) -> torch.Tensor:
    if spec == "tiny_square":
        side = rng.uniform(4.0, 16.0)
        w = h = side
    elif spec == "small_square":
        side = rng.uniform(16.0, 32.0)
        w = h = side
    elif spec == "vehicle":
        # Cars / trucks in DOTA tiles (~10–25 px short side)
        side = rng.uniform(10.0, 25.0)
        w = h = side
    elif spec == "medium_square":
        side = rng.uniform(32.0, 128.0)
        w = h = side
    elif spec == "large_square":
        side = rng.uniform(128.0, 512.0)
        w = h = side
    elif spec == "elongated":
        short = rng.uniform(8.0, 40.0)
        aspect = rng.uniform(3.0, 8.0)
        w, h = short, short * aspect
        if rng.random() < 0.5:
            w, h = h, w
    elif spec == "elongated_thin":
        short = rng.uniform(4.0, 12.0)
        aspect = rng.uniform(8.0, 40.0)
        w, h = short, short * aspect
        if rng.random() < 0.5:
            w, h = h, w
    else:
        w = rng.uniform(4.0, 512.0)
        h = rng.uniform(4.0, 512.0)

    angle = rng.uniform(-math.pi / 2, math.pi / 2)
    cx = rng.uniform(0.0, 1024.0)
    cy = rng.uniform(0.0, 1024.0)
    return _box_tensor(cx, cy, w, h, angle)


def _offset_for_target_iou(
    rng: np.random.Generator,
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    """Shift box_b center to vary overlap (rough control via fraction of max side)."""
    overlap_frac = rng.uniform(0.0, 1.2)
    max_side = max(float(box_a[2]), float(box_a[3]), float(box_b[2]), float(box_b[3]))
    dist = (1.0 - overlap_frac) * max_side
    angle = rng.uniform(0.0, 2.0 * math.pi)
    cx = float(box_a[0]) + dist * math.cos(angle)
    cy = float(box_a[1]) + dist * math.sin(angle)
    out = box_b.clone()
    out[0] = cx
    out[1] = cy
    return out


def generate_pairs(
    rng: np.random.Generator,
    *,
    n_per_category: int,
    categories: Sequence[str],
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    for cat in categories:
        for _ in range(n_per_category):
            a = _random_box(rng, cat)
            b = _random_box(rng, cat)
            b = _offset_for_target_iou(rng, a, b)
            yield cat, a, b


DEFAULT_CATEGORIES = (
    "tiny_square",
    "vehicle",
    "small_square",
    "medium_square",
    "large_square",
    "elongated",
    "elongated_thin",
)


def evaluate_pairs(
    pairs: Sequence[Tuple[str, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    params: GeometryParams,
    chunk_size: int = 256,
) -> List[PairRecord]:
    records: List[PairRecord] = []
    n = len(pairs)
    for start in range(0, n, chunk_size):
        chunk = pairs[start : start + chunk_size]
        boxes1 = torch.stack([p[1] for p in chunk])
        boxes2 = torch.stack([p[2] for p in chunk])
        num_samples = _resolve_num_samples(boxes1, boxes2, params)
        sampled = _sampled_iou_batch(
            boxes1, boxes2, device=device, num_samples=num_samples
        )
        for i, (cat, a, b) in enumerate(chunk):
            exact = _exact_iou(a, b)
            records.append(
                PairRecord(
                    category=cat,
                    w1=float(a[2]),
                    h1=float(a[3]),
                    w2=float(b[2]),
                    h2=float(b[3]),
                    exact=exact,
                    sampled=float(sampled[i].item()),
                    num_samples=num_samples,
                )
            )
    return records


@dataclass
class ErrorStats:
    count: int
    mean_abs: float
    p50_abs: float
    p90_abs: float
    p99_abs: float
    max_abs: float
    mean_signed: float
    mean_rel_pct: float
    p90_rel_pct: float
    frac_gt_5pct: float
    frac_gt_10pct: float
    frac_gt_20pct: float
    mean_samples: float


def _summarize(records: Sequence[PairRecord]) -> ErrorStats:
    if not records:
        return ErrorStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    abs_err = np.array([r.abs_error for r in records], dtype=np.float64)
    signed = np.array([r.signed_error for r in records], dtype=np.float64)
    rel = np.array([r.rel_error_pct for r in records], dtype=np.float64)
    rel = rel[np.isfinite(rel)]
    exact = np.array([r.exact for r in records], dtype=np.float64)
    samples = np.array([r.num_samples for r in records], dtype=np.float64)

    def _frac(th: float) -> float:
        mask = exact > 1e-6
        if not mask.any():
            return 0.0
        return float((abs_err[mask] > th).mean())

    return ErrorStats(
        count=len(records),
        mean_abs=float(abs_err.mean()),
        p50_abs=float(np.percentile(abs_err, 50)),
        p90_abs=float(np.percentile(abs_err, 90)),
        p99_abs=float(np.percentile(abs_err, 99)),
        max_abs=float(abs_err.max()),
        mean_signed=float(signed.mean()),
        mean_rel_pct=float(rel.mean()) if rel.size else float("nan"),
        p90_rel_pct=float(np.percentile(rel, 90)) if rel.size else float("nan"),
        frac_gt_5pct=_frac(0.05),
        frac_gt_10pct=_frac(0.10),
        frac_gt_20pct=_frac(0.20),
        mean_samples=float(samples.mean()),
    )


def _fmt_stats(label: str, stats: ErrorStats) -> str:
    if stats.count == 0:
        return f"{label:16s}  (no pairs)"
    return (
        f"{label:16s}  n={stats.count:5d}  "
        f"|err| mean={stats.mean_abs:.4f} p50={stats.p50_abs:.4f} "
        f"p90={stats.p90_abs:.4f} p99={stats.p99_abs:.4f} max={stats.max_abs:.4f}  "
        f"rel% mean={stats.mean_rel_pct:6.1f} p90={stats.p90_rel_pct:6.1f}  "
        f">5%={stats.frac_gt_5pct:.1%} >10%={stats.frac_gt_10pct:.1%} >20%={stats.frac_gt_20pct:.1%}  "
        f"grid≈{stats.mean_samples:.0f}"
    )


def _print_report(
    records: List[PairRecord],
    params: GeometryParams,
    *,
    title: str = "Sampled vs Shapely rIoU",
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"Geometry: spacing={params.target_spacing_px}px  "
        f"min={params.min_samples}  max={params.max_samples}  "
        f"min_short_pts={params.min_points_short}"
    )
    overall = _summarize(records)
    print(_fmt_stats("ALL", overall))

    by_cat: dict[str, List[PairRecord]] = {}
    for r in records:
        by_cat.setdefault(r.category, []).append(r)
    for cat in sorted(by_cat):
        print(_fmt_stats(cat, _summarize(by_cat[cat])))

    # IoU bins (exact)
    bins = [(0.0, 0.0), (0.0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 1.0 + 1e-9)]
    labels = ["exact=0", "0-0.3", "0.3-0.7", "0.7-1", "exact=1"]
    print("\nBy exact IoU bin:")
    for (lo, hi), lab in zip(bins, labels):
        if lo == hi == 0.0:
            subset = [r for r in records if r.exact <= 1e-8]
        elif lo >= 1.0:
            subset = [r for r in records if r.exact >= 1.0 - 1e-6]
        else:
            subset = [r for r in records if lo < r.exact <= hi]
        print(f"  {_fmt_stats(lab, _summarize(subset))}")


def _write_csv(path: Path, records: Sequence[PairRecord], params: GeometryParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "category",
                "w1",
                "h1",
                "w2",
                "h2",
                "exact_iou",
                "sampled_iou",
                "abs_error",
                "signed_error",
                "rel_error_pct",
                "num_samples",
                "target_spacing_px",
                "min_samples",
                "max_samples",
                "min_points_short",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.category,
                    f"{r.w1:.4f}",
                    f"{r.h1:.4f}",
                    f"{r.w2:.4f}",
                    f"{r.h2:.4f}",
                    f"{r.exact:.6f}",
                    f"{r.sampled:.6f}",
                    f"{r.abs_error:.6f}",
                    f"{r.signed_error:.6f}",
                    f"{r.rel_error_pct:.4f}",
                    r.num_samples,
                    params.target_spacing_px,
                    params.min_samples,
                    params.max_samples,
                    params.min_points_short,
                ]
            )


def _build_pairs(
    rng: np.random.Generator,
    *,
    pairs: int,
    categories: Sequence[str],
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    n_per = max(1, pairs // len(categories))
    out = list(generate_pairs(rng, n_per_category=n_per, categories=categories))
    if len(out) < pairs:
        extra = pairs - len(out)
        out.extend(
            generate_pairs(rng, n_per_category=1, categories=categories[:extra])
        )
    return out[:pairs]


def main(argv: Optional[Sequence[str]] = None) -> int:
    _require_shapely()

    parser = argparse.ArgumentParser(
        description="Measure sampling error vs exact Shapely rIoU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", type=int, default=3000, help="Number of box pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        choices=list(DEFAULT_CATEGORIES),
        help="Strata to sample",
    )
    parser.add_argument("--target-spacing", type=float, default=_DEFAULT_TARGET_SPACING_PX)
    parser.add_argument("--min-samples", type=int, default=_DEFAULT_MIN_SAMPLES)
    parser.add_argument("--max-samples", type=int, default=_DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--min-points-short", type=int, default=_MIN_POINTS_ALONG_SHORT_SIDE
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for oriented_box_iou_gpu",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--csv", type=Path, default=None, help="Write per-pair CSV")
    parser.add_argument(
        "--sweep-spacing",
        type=float,
        nargs="+",
        default=None,
        help="Sweep target_spacing_px values (overrides --target-spacing)",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    pairs = _build_pairs(rng, pairs=args.pairs, categories=args.categories)

    spacings: List[float]
    if args.sweep_spacing:
        spacings = list(args.sweep_spacing)
    else:
        spacings = [args.target_spacing]

    all_records: List[PairRecord] = []
    for spacing in spacings:
        params = GeometryParams(
            target_spacing_px=spacing,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            min_points_short=args.min_points_short,
        )
        title = f"spacing={spacing}px"
        records = evaluate_pairs(
            pairs, device=device, params=params, chunk_size=args.chunk_size
        )
        _print_report(records, params, title=title)
        if args.csv and len(spacings) == 1:
            _write_csv(args.csv, records, params)
        all_records = records

    if args.csv and len(spacings) > 1:
        print("\nNote: --csv ignored during multi-value --sweep-spacing", file=sys.stderr)

    if len(spacings) == 1 and all_records:
        worst = sorted(all_records, key=lambda r: r.abs_error, reverse=True)[:10]
        print("\nTop 10 |error| pairs:")
        for r in worst:
            print(
                f"  {r.category:14s}  box1={r.w1:.1f}x{r.h1:.1f}  box2={r.w2:.1f}x{r.h2:.1f}  "
                f"exact={r.exact:.4f}  sampled={r.sampled:.4f}  |err|={r.abs_error:.4f}  "
                f"grid={r.num_samples}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
