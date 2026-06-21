#!/usr/bin/env python3
"""
Free GPU memory by terminating processes that are using the GPU.

Use this before re-launching training when the GPU is still occupied (e.g. after
a crash or when a previous run did not release memory).

Usage:
  python tools/free_gpu.py              # List GPU processes, prompt before killing
  python tools/free_gpu.py --force     # Kill all GPU processes without prompting
  python tools/free_gpu.py --dry-run    # Only list processes, do not kill
  python tools/free_gpu.py --python-only  # Only kill Python processes (default: kill all)
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_nvidia_smi_query(query: str, format: str = "csv") -> str:
    """Run nvidia-smi with a query and return stdout."""
    cmd = [
        "nvidia-smi",
        f"--query-compute-apps={query}",
        f"--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return (result.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi failed: {e}", file=sys.stderr)
        return ""
    except FileNotFoundError:
        print("nvidia-smi not found. Is the NVIDIA driver installed?", file=sys.stderr)
        return ""


def get_gpu_pids(python_only: bool = False) -> list[tuple[int, str]]:
    """
    Return list of (pid, process_name) for processes using the GPU.
    If python_only is True, only include processes whose name contains 'python'.
    """
    # pid,process_name,used_memory
    out = run_nvidia_smi_query("pid,process_name,used_memory", format="csv")
    if not out:
        return []

    entries: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            name = parts[1] if len(parts) > 1 else ""
            if python_only and "python" not in name.lower():
                continue
            entries.append((pid, name))
        except ValueError:
            continue
    return entries


def kill_pid(pid: int) -> bool:
    """Send SIGKILL to the process. Return True if successful."""
    try:
        subprocess.run(["kill", "-9", str(pid)], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Free GPU memory by killing processes using the GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list processes using the GPU; do not kill any.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Kill processes without asking for confirmation.",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only kill Python processes (default: kill all GPU processes).",
    )
    args = parser.parse_args()

    entries = get_gpu_pids(python_only=args.python_only)
    if not entries:
        print("No GPU processes found (or nvidia-smi returned nothing).")
        return 0

    print("Processes using the GPU:")
    for pid, name in entries:
        print(f"  PID {pid}: {name or '(unknown)'}")

    if args.dry_run:
        print("Dry-run: no processes killed.")
        return 0

    if not args.force:
        try:
            reply = input("Kill these processes? [y/N]: ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 0

    killed = 0
    for pid, name in entries:
        if kill_pid(pid):
            print(f"Killed PID {pid} ({name or 'unknown'})")
            killed += 1
        else:
            print(f"Failed to kill PID {pid}", file=sys.stderr)

    print(f"Freed GPU memory by terminating {killed} process(es).")
    return 0 if killed == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
