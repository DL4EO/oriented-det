"""Ensure oriented_det/configs/ stays in sync with repo configs/."""

from __future__ import annotations

from tools.sync_vendored_configs import sync_vendored_configs


def test_vendored_configs_match_repo_manifest() -> None:
    assert sync_vendored_configs(check=True) == 0
