"""Progress bar streams: show tqdm on the real console when stderr is piped (e.g. ``2>&1 | tee log``)."""

from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Optional, TextIO

# Lazily opened /dev/tty (or Windows CON) for tqdm; closed at exit.
_tqdm_tty: Optional[TextIO] = None
_tqdm_tty_open_failed: bool = False
_devnull: Optional[TextIO] = None


def _close_tqdm_tty() -> None:
    global _tqdm_tty
    if _tqdm_tty is not None:
        try:
            _tqdm_tty.close()
        except Exception:
            pass
        _tqdm_tty = None


def _get_devnull() -> TextIO:
    global _devnull
    if _devnull is None:
        _devnull = open(os.devnull, "w")
    return _devnull


def tqdm_progress_stream() -> Any:
    """File object for ``tqdm(..., file=...)``, aligned with :func:`tools.train` / ``progress_stream`` on the training loop.

    * If ``sys.stderr`` is a TTY, use it (normal interactive run).
    * If not (shell redirect / ``| tee``), try ``/dev/tty`` (POSIX) or ``CON`` (Windows) so the bar
      still updates on the user’s terminal while other output is captured in a file.
    * If none of that works, use ``os.devnull`` so logs are not flooded with control characters.
    """
    global _tqdm_tty, _tqdm_tty_open_failed
    if sys.stderr.isatty():
        return sys.stderr
    if _tqdm_tty is not None:
        return _tqdm_tty
    if not _tqdm_tty_open_failed:
        if os.name == "posix":
            try:
                # Still works when e.g. ``python ... 2>&1 | tee x.log`` (stderr is a pipe, not a TTY).
                _tqdm_tty = open("/dev/tty", "w", encoding="utf-8", buffering=1)
                atexit.register(_close_tqdm_tty)
                return _tqdm_tty
            except OSError:
                _tqdm_tty_open_failed = True
        elif os.name == "nt":
            try:
                _tqdm_tty = open("CON", "w", encoding="utf-8", errors="replace", buffering=1)
                atexit.register(_close_tqdm_tty)
                return _tqdm_tty
            except OSError:
                _tqdm_tty_open_failed = True
    return _get_devnull()
