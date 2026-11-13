"""Harness contract: three presets are declared and keep their isolation rules."""

from __future__ import annotations

import subprocess

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.toolchain import node_exe

PROFILE_PATCH = (
    PROJECT_ROOT
    / "sub_root"
    / "deps_management"
    / "deps"
    / "node"
    / "modules"
    / "dsh_home"
    / "profiles"
    / "web"
    / "cordis.patch.yml"
)
EXTRACT = (
    PROJECT_ROOT
    / "sub_root"
    / "deps_management"
    / "deps_manager"
    / "scripts"
    / "extract_presets.mjs"
)


def _patch_text() -> str:
    return PROFILE_PATCH.read_text(encoding="utf-8")


def test_presets_declared():
    text = _patch_text()
    for preset in ("pysci-worker", "pysci-leader", "pysci-meta"):
        assert f"id: {preset}" in text
    assert "selectedDefault: pysci-worker" in text


def test_subagent_allowlists():
    text = _patch_text()
    assert "allowedAgentPresets: [pysci-worker]" in text
    assert "allowedAgentPresets: [pysci-worker, pysci-leader, pysci-meta]" in text


def test_leader_fork_and_workflow_disabled():
    text = _patch_text()
    assert "id: tool-subagent-fork" in text
    assert "id: tool-workflow" in text
    assert text.count("disabled: true") >= 2


def test_skill_roots_isolated():
    text = _patch_text()
    assert "- .agents/skills" in text
    assert "- .agents/leader-skills" in text
    assert "- .agents/meta-skills" in text
    assert "- .agents/shared-skills" in text
    assert "includeDefaultRoots: false" in text


@pytest.mark.skipif(not PROFILE_PATCH.is_file(), reason="project DSH profile not installed")
def test_extract_presets_in_sync():
    if not EXTRACT.is_file():
        pytest.skip("extract_presets.mjs missing")
    result = subprocess.run(
        [node_exe(), str(EXTRACT), "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
