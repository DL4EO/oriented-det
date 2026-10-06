#!/usr/bin/env python3
"""Latency benchmark: single-tile forward (eval) on fixed H×W (default 1024).

Compares detectors under the same canvas / device / warmup protocol. Times the
full ``model([image])`` path (backbone + head + decode + NMS), not preprocess.

Examples::

  python tools/bench_latency.py \\
    --pair rtmdet=configs/rotated_rtmdet/dota_le90_3x_s.json:pretrained/rotated_rtmdet_cspnext_s_dota_le90_3x.pth \\
    --pair fcos=configs/rotated_fcos/dota_le90_1x.json:pretrained/rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.pth

  python tools/bench_latency.py --size 1024 --warmup 20 --iters 100 --device cuda:0
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _parse_pair(spec: str) -> Tuple[str, str, str]:
    """``name=config:checkpoint`` (checkpoint may contain ``:`` on Windows paths — split once from the right after ``=``)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected name=config:ckpt, got {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError(f"Expected name=config:ckpt, got {spec!r}")
    # Split on last colon so Windows drive letters don't break; Linux paths rarely have ':'
    config, ckpt = rest.rsplit(":", 1)
    name = name.strip()
    config = config.strip()
    ckpt = ckpt.strip()
    if not name or not config or not ckpt:
        raise argparse.ArgumentTypeError(f"Empty field in {spec!r}")
    return name, config, ckpt


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.inference_mode()
def _timed_forwards(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    synchronize: bool,
) -> List[float]:
    """Return per-iter wall times in milliseconds (excludes warmup)."""
    device = image.device

    def _sync() -> None:
        if synchronize and device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(max(0, warmup)):
        _ = model([image])
        _sync()

    times_ms: List[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        _ = model([image])
        _sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def _summarize(times_ms: List[float]) -> Dict[str, float]:
    ordered = sorted(times_ms)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    mean = statistics.fmean(times_ms)
    return {
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": statistics.pstdev(times_ms) if n > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def _bench_one(
    name: str,
    config_path: str,
    checkpoint_path: str,
    *,
    size: int,
    device: str,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    from oriented_det.runtime.checkpoint import load_model_from_checkpoint

    model, config, _ = load_model_from_checkpoint(checkpoint_path, config_path, device=device)
    model.eval()
    total, trainable = _count_params(model)
    # ImageNet-ish float tensor in [0,1] (models expect that range via training preprocess)
    image = torch.rand(3, size, size, device=device, dtype=torch.float32)
    sync = device.startswith("cuda")
    times = _timed_forwards(model, image, warmup=warmup, iters=iters, synchronize=sync)
    stats = _summarize(times)
    # Peak allocated during timed section
    peak_mb = None
    if sync:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "name": name,
        "model_type": getattr(config, "model_type", "?"),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "params_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "size": size,
        "warmup": warmup,
        "iters": iters,
        "peak_mb": peak_mb,
        **stats,
    }


def _default_pairs(repo: Path) -> List[Tuple[str, str, str]]:
    hub_s = repo / "pretrained/rotated_rtmdet_cspnext_s_dota_le90_3x-3f6a1a0b.pth"
    hub_s_cfg = repo / "configs/rotated_rtmdet/dota_le90_3x_s.json"
    pairs = [
        (
            "rtmdet_s",
            str(hub_s_cfg),
            str(hub_s),
        ),
        (
            "fcos_r50_1x",
            str(repo / "configs/rotated_fcos/dota_le90_1x.json"),
            str(repo / "pretrained/rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.pth"),
        ),
        (
            "oriented_rcnn_r50_1x",
            str(repo / "configs/oriented_rcnn/dota_le90_1x.json"),
            str(repo / "pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-725c244f.pth"),
        ),
    ]
    out = []
    for name, cfg, ckpt in pairs:
        if Path(cfg).is_file() and Path(ckpt).is_file():
            out.append((name, cfg, ckpt))
        else:
            print(f"skip {name}: missing {cfg if not Path(cfg).is_file() else ckpt}", file=sys.stderr)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=_parse_pair,
        help="name=config:checkpoint (repeatable). Default: RTMDet / FCOS 1× / RetinaNet 3× if present.",
    )
    parser.add_argument("--size", type=int, default=1024, help="Square tile size (default 1024).")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    pairs = args.pair or _default_pairs(repo)
    if not pairs:
        print("No model pairs to benchmark.", file=sys.stderr)
        return 1

    print(f"device={args.device}  size={args.size}²  warmup={args.warmup}  iters={args.iters}")
    if args.device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(args.device)}")
    print()

    rows: List[Dict[str, Any]] = []
    for name, cfg, ckpt in pairs:
        print(f"=== {name} ===")
        print(f"  config: {cfg}")
        print(f"  ckpt:   {ckpt}")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)
        row = _bench_one(
            name,
            cfg,
            ckpt,
            size=args.size,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(row)
        peak = f"{row['peak_mb']:.0f} MiB" if row["peak_mb"] is not None else "n/a"
        print(
            f"  params={row['params_m']:.2f}M  "
            f"mean={row['mean_ms']:.2f} ms  median={row['median_ms']:.2f} ms  "
            f"std={row['std_ms']:.2f}  fps={row['fps']:.1f}  peak_alloc={peak}"
        )
        print()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("Summary (mean latency, full eval forward incl. NMS)")
    print(f"{'name':20s} {'params_M':>8s} {'ms':>8s} {'fps':>7s} {'peak_MiB':>9s}")
    for r in rows:
        peak = f"{r['peak_mb']:.0f}" if r["peak_mb"] is not None else "-"
        print(f"{r['name']:20s} {r['params_m']:8.2f} {r['mean_ms']:8.2f} {r['fps']:7.1f} {peak:>9s}")

    if len(rows) >= 2:
        print()
        # Pairwise vs first model (usually RTMDet)
        base = rows[0]
        for r in rows[1:]:
            if r["mean_ms"] <= 0 or base["mean_ms"] <= 0:
                continue
            ratio = r["mean_ms"] / base["mean_ms"]
            if base["mean_ms"] < r["mean_ms"]:
                print(
                    f"  {base['name']} is {ratio:.2f}× faster than {r['name']} "
                    f"({base['mean_ms']:.1f} vs {r['mean_ms']:.1f} ms)"
                )
            else:
                print(
                    f"  {r['name']} is {1/ratio:.2f}× faster than {base['name']} "
                    f"({r['mean_ms']:.1f} vs {base['mean_ms']:.1f} ms)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
