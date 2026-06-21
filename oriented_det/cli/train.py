"""``odet train`` / ``torchrun -m oriented_det.cli.train`` entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# tools/ lives at the oriented-det repo root (editable install); same bootstrap as oriented_det.cli._invoke
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tools.train import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
