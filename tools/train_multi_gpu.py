#!/usr/bin/env python3
"""Multi-GPU training wrapper for train.py.

This script launches train.py with torchrun for distributed training across multiple GPUs.
Configuration is read from JSON config files; pass --config <path> and other train.py arguments.

Usage:
    python train_multi_gpu.py [--nproc-per-node N] --config <config.json> [train.py arguments...]

Examples:
    # Use all available GPUs with default config
    python train_multi_gpu.py --config configs/oriented_rcnn/dota_le90_1x.json

    # Use 4 GPUs
    python train_multi_gpu.py --nproc-per-node 4 --config configs/oriented_rcnn/dota_le90_1x.json

    # Use 2 GPUs with custom arguments
    python train_multi_gpu.py --nproc-per-node 2 --config configs/.../config.json --use-amp

Checkpoint loading is configured only in the JSON ``checkpoint`` section (same as ``train.py``).

The script automatically:
- Detects the number of available GPUs
- Launches train.py with torchrun
- Forwards all arguments (including --config) to train.py
- Sets up proper environment variables for distributed training
"""

import subprocess
import sys
import os
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install PyTorch first.")
    sys.exit(1)


def get_num_gpus():
    """Get the number of available GPUs."""
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU training.")
        return 0
    return torch.cuda.device_count()


def find_torchrun():
    """Find torchrun executable."""
    # Try torchrun first (PyTorch 1.9+)
    try:
        result = subprocess.run(
            ["torchrun", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "torchrun"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fallback to python -m torch.distributed.launch
    try:
        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.launch", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "python -m torch.distributed.launch"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-GPU training wrapper for train.py (config-based training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all available GPUs (pass --config and any train.py options)
  python train_multi_gpu.py --config configs/oriented_rcnn/dota_le90_1x.json

  # Use 4 GPUs
  python train_multi_gpu.py --nproc-per-node 4 --config configs/.../config.json

  # Use 2 GPUs with custom arguments
  python train_multi_gpu.py --nproc-per-node 2 --config configs/.../config.json --use-amp --debug
        """,
    )
    
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Number of processes per node (default: number of available GPUs)",
    )
    
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)",
    )
    
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default=None,
        help="Distributed backend (default: nccl for single-node, faster than Gloo with P2P/SHM enabled)",
    )
    
    # Parse known args to separate wrapper args from train.py args
    args, unknown_args = parser.parse_known_args()
    
    # Get number of GPUs
    num_gpus = get_num_gpus()
    
    if num_gpus == 0:
        print("Error: No GPUs available. Cannot run multi-GPU training.")
        print("Please use train.py directly for CPU or single-GPU training.")
        sys.exit(1)
    
    # Determine number of processes
    nproc_per_node = args.nproc_per_node or num_gpus
    
    if nproc_per_node > num_gpus:
        print(f"Warning: Requested {nproc_per_node} GPUs but only {num_gpus} available.")
        print(f"Using {num_gpus} GPUs instead.")
        nproc_per_node = num_gpus
    
    if nproc_per_node < 1:
        print("Error: nproc-per-node must be at least 1.")
        sys.exit(1)
    
    # Find torchrun
    torchrun_cmd = find_torchrun()
    if torchrun_cmd is None:
        print("Error: Could not find torchrun or torch.distributed.launch.")
        print("Please ensure PyTorch is properly installed.")
        sys.exit(1)
    
    # Train via installable module (odet train / oriented_det.cli.train)
    train_module = "oriented_det.cli.train"
    extra_args = unknown_args
    if torchrun_cmd == "torchrun":
        cmd = [
            "torchrun",
            "--nproc-per-node", str(nproc_per_node),
            "--master-port", str(args.master_port),
            "-m", train_module,
        ] + extra_args
    else:
        cmd = [
            sys.executable,
            "-m", "torch.distributed.launch",
            "--nproc-per-node", str(nproc_per_node),
            "--master-port", str(args.master_port),
            train_module,
        ] + extra_args
    
    # Set up environment for single-node multi-GPU training
    env = os.environ.copy()
    
    # Short TMPDIR to avoid Gloo "File name too long" (store paths can embed hostname)
    if "TMPDIR" not in env:
        for d in ("/dev/shm", "/tmp"):
            if os.path.isdir(d):
                env["TMPDIR"] = d
                break
    
    # Backend: default NCCL for all (faster with P2P/SHM enabled on single-node A100)
    # With PyTorch 2.3+ and proper configuration, NCCL works reliably on single-node
    if args.backend:
        env["DIST_BACKEND"] = args.backend
    else:
        env.setdefault("DIST_BACKEND", "nccl")  # NCCL is faster than Gloo on single-node multi-GPU
    
    # Ensure MASTER_ADDR and MASTER_PORT are set for distributed init
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(args.master_port))
    
    socket_ifname = None
    use_gib = False

    # Configure NCCL for GCP. gIB is only available on A3 Ultra, A4, A4X instances.
    # A2 instances (A100) do NOT support gIB - use socket-only NCCL (NVLink intra-node, sockets/gVNIC inter-node).
    if env.get("DIST_BACKEND", "nccl") == "nccl":
        # Check instance type to determine if gIB is available
        try:
            instance_type = subprocess.run(
                ["curl", "-s", "-H", "Metadata-Flavor: Google",
                 "http://metadata.google.internal/computeMetadata/v1/instance/machine-type"],
                capture_output=True, text=True, timeout=2
            )
            if instance_type.returncode == 0:
                instance_type = instance_type.stdout.strip().split("/")[-1]
                # gIB is only available on A3 Ultra, A4, A4X (not A2)
                if instance_type.startswith(("a3-ultra", "a4", "a4x")):
                    gib_script = "/usr/local/gib/scripts/set_nccl_env.sh"
                    if os.path.exists(gib_script):
                        use_gib = True
                        env["NCCL_NET"] = "gIB"
                        env["NCCL_CROSS_NIC"] = "0"
                        env["NCCL_NET_GDR_LEVEL"] = "PIX"
                        env["NCCL_P2P_NET_CHUNKSIZE"] = "131072"
                        env["NCCL_NVLS_CHUNKSIZE"] = "524288"
                        env["NCCL_IB_ADAPTIVE_ROUTING"] = "1"
                        env["NCCL_IB_QPS_PER_CONNECTION"] = "4"
                        env["NCCL_IB_TC"] = "52"
                        env["NCCL_IB_FIFO_TC"] = "84"
                        env["NCCL_TUNER_CONFIG_PATH"] = "/usr/local/gib/configs/tuner_config_a3u.txtpb"
                        print(f"Using GCP gIB backend (instance: {instance_type})")
                    else:
                        print(f"Warning: gIB script not found on {instance_type}, using socket-only NCCL")
                else:
                    print(f"Instance {instance_type} does not support gIB (A2/A100), using socket-only NCCL")
            else:
                print("Could not detect instance type, using socket-only NCCL")
        except Exception:
            print("Could not detect instance type, using socket-only NCCL")

        # A2 / socket-only: recommended NCCL settings (GDR off, socket tuning, NCCL 2.12.7+)
        if not use_gib:
            env.pop("NCCL_NET", None)  # Ensure no inherited gIB on A2
            env["NCCL_NET_GDR_LEVEL"] = "0"  # Disable GPUDirect RDMA (unsupported on A2)
            env["NCCL_SOCKET_NTHREADS"] = "4"
            env["NCCL_NSOCKETS_PER_PEER"] = "4"
            env.setdefault("NCCL_DEBUG", "WARN")  # WARN: minimal output; use INFO for debugging
            
            # Remove GCP gIB shim from LD_LIBRARY_PATH on A2 (shim enforces PIX, incompatible with A2)
            # The shim (libnccl-net_internal.so) checks guest_config.txtpb and enforces gIB settings
            ld_path = env.get("LD_LIBRARY_PATH", "")
            if ld_path:
                # Remove /usr/local/gib/lib64 to prevent shim from loading
                paths = [p for p in ld_path.split(":") if p and not p.rstrip("/").endswith("/usr/local/gib/lib64")]
                if paths:
                    env["LD_LIBRARY_PATH"] = ":".join(paths)
                else:
                    env.pop("LD_LIBRARY_PATH", None)
            print("Removed GCP gIB shim from LD_LIBRARY_PATH (A2 uses socket-only NCCL)")

        # Detect and set network interface for single-node communication (required for socket-only NCCL)
        socket_ifname = env.get("NCCL_SOCKET_IFNAME")
        if not socket_ifname:
            try:
                result = subprocess.run(["ip", "route", "get", "8.8.8.8"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'dev' in line:
                            parts = line.split()
                            if 'dev' in parts:
                                idx = parts.index('dev')
                                if idx + 1 < len(parts):
                                    socket_ifname = parts[idx + 1]
                                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass
        if not socket_ifname:
            for ifname in ["ens8", "ens4", "eth0", "eno1"]:
                try:
                    with open(f"/sys/class/net/{ifname}/operstate", "r") as f:
                        if "up" in f.read().lower():
                            socket_ifname = ifname
                            break
                except (FileNotFoundError, IOError):
                    continue
        if socket_ifname:
            env["NCCL_SOCKET_IFNAME"] = socket_ifname
    
    # For Gloo backend: set GLOO_SOCKET_IFNAME (reuse socket_ifname if already detected for NCCL)
    if env.get("DIST_BACKEND", "nccl") == "gloo":
        if not socket_ifname:
            try:
                result = subprocess.run(["ip", "route", "get", "8.8.8.8"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'dev' in line:
                            parts = line.split()
                            if 'dev' in parts:
                                idx = parts.index('dev')
                                if idx + 1 < len(parts):
                                    socket_ifname = parts[idx + 1]
                                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass
            if not socket_ifname:
                for ifname in ["ens8", "ens4", "eth0", "eno1"]:
                    try:
                        with open(f"/sys/class/net/{ifname}/operstate", "r") as f:
                            if "up" in f.read().lower():
                                socket_ifname = ifname
                                break
                    except (FileNotFoundError, IOError):
                        continue
    
    # Optimize Gloo backend settings for better performance
    if env.get("DIST_BACKEND", "nccl") == "gloo":
        # Use proper network interface for Gloo
        if socket_ifname:
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        # Increase Gloo timeout (default 30s, increase for large models)
        env.setdefault("GLOO_TIMEOUT_SECONDS", "1800")  # 30 minutes
        # Use async operations where possible (if supported)
        # Note: GLOO_ASYNC may not be available in all PyTorch versions
    
    # Print info
    print("=" * 80)
    print("Multi-GPU Training Launcher")
    print("=" * 80)
    print(f"GPUs available: {num_gpus}")
    print(f"GPUs to use: {nproc_per_node}")
    print(f"Module: {train_module}")
    proj = env.get("ORIENTED_DET_PROJECT_ROOT")
    if proj:
        print(f"ORIENTED_DET_PROJECT_ROOT: {proj}")
    from oriented_det.pretrained import get_pretrained_dir

    pre = env.get("ORIENTED_DET_PRETRAINED_DIR")
    print(f"Pretrained cache: {pre or get_pretrained_dir()}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    print("Distributed Training Configuration:")
    backend = env.get('DIST_BACKEND', 'nccl')
    print(f"  Backend: {backend} (default: NCCL, use --backend gloo for Gloo)")
    print(f"  MASTER_ADDR: {env.get('MASTER_ADDR', '127.0.0.1')}")
    print(f"  MASTER_PORT: {env.get('MASTER_PORT', '29500')}")
    if backend == "nccl":
        net = env.get("NCCL_NET", "socket (default)")
        ifname = env.get("NCCL_SOCKET_IFNAME", "not set")
        if net == "gIB":
            gdr_level = env.get("NCCL_NET_GDR_LEVEL", "not set")
            print(f"  NCCL_NET: {net} (gIB backend)")
            print(f"  NCCL_NET_GDR_LEVEL: {gdr_level}")
        else:
            print(f"  NCCL_NET: {net} (socket-only, A2)")
            print(f"  NCCL_NET_GDR_LEVEL: {env.get('NCCL_NET_GDR_LEVEL', 'not set')}")
            print(f"  NCCL_SOCKET_NTHREADS: {env.get('NCCL_SOCKET_NTHREADS', 'not set')}")
            print(f"  NCCL_NSOCKETS_PER_PEER: {env.get('NCCL_NSOCKETS_PER_PEER', 'not set')}")
            print(f"  NCCL_DEBUG: {env.get('NCCL_DEBUG', 'not set')}")
        print(f"  NCCL_SOCKET_IFNAME: {ifname}")
    print("=" * 80)
    print()
    
    # Launch training
    try:
        # Run the command with environment variables and forward stdout/stderr
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error launching training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
