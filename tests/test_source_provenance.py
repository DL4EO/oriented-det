"""Tests for git/package provenance stamping at train start."""

from __future__ import annotations

from pathlib import Path

from oriented_det.train.utils import capture_source_provenance, get_framework_source_root


def test_capture_source_provenance_in_git_repo():
    root = get_framework_source_root()
    prov = capture_source_provenance(root)
    assert prov["source_code_root"] == str(root.resolve())
    # This workspace is a git checkout during development.
    assert prov["git_commit"] is not None
    assert len(prov["git_commit"]) >= 7
    assert prov["git_describe"] is not None
    assert isinstance(prov["git_dirty"], bool)
    assert prov["git_branch"] is not None
    assert prov["git_commit_date"] is not None


def test_capture_source_provenance_non_git_dir(tmp_path: Path):
    prov = capture_source_provenance(tmp_path)
    assert prov["source_code_root"] == str(tmp_path.resolve())
    assert prov["git_commit"] is None
    assert prov["git_describe"] is None
    assert prov["git_dirty"] is None
