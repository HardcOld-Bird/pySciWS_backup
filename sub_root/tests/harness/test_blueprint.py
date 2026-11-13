"""Harness contract: blueprint budgets, skill frontmatter, links, no dead Qoder paths."""

from __future__ import annotations

import re
from pathlib import Path

from pysci.paths import PROJECT_ROOT

BUDGET_BYTES = 8192
SKILL_ROOTS = (
    PROJECT_ROOT / ".agents" / "skills",
    PROJECT_ROOT / ".agents" / "leader-skills",
    PROJECT_ROOT / ".agents" / "meta-skills",
)
SHARED = PROJECT_ROOT / ".agents" / "shared-skills" / "blueprint-canon"
META_VIEW = PROJECT_ROOT / ".agents" / "meta-skills" / "blueprint-canon" / "SKILL.md"

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AssertionError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise AssertionError(f"unterminated frontmatter: {path}")
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip("'\"")
    return fields


def test_agents_budget():
    for name in ("AGENTS.md", "AGENTS.leader.md", "AGENTS.meta.md"):
        size = (PROJECT_ROOT / name).stat().st_size
        assert size <= BUDGET_BYTES, f"{name} = {size} > {BUDGET_BYTES}"


def test_blueprint_budget():
    for path in SHARED.rglob("*.md"):
        size = path.stat().st_size
        assert size <= BUDGET_BYTES, f"{path} = {size} > {BUDGET_BYTES}"
    size = META_VIEW.stat().st_size
    assert size <= BUDGET_BYTES, f"{META_VIEW} = {size} > {BUDGET_BYTES}"


def test_skill_frontmatter_and_names():
    for root in SKILL_ROOTS:
        names: set[str] = set()
        for path in root.glob("*/SKILL.md"):
            fields = _frontmatter(path)
            name = fields.get("name", "")
            assert _NAME_RE.match(name), f"invalid skill name {name!r} in {path}"
            assert fields.get("description"), f"empty description in {path}"
            assert name not in names, f"duplicate skill {name!r} under {root}"
            names.add(name)


def test_blueprint_links_exist():
    for path in (SHARED / "SKILL.md", META_VIEW):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"broken link {target!r} in {path}"


def test_meta_view_points_to_shared_canon():
    text = META_VIEW.read_text(encoding="utf-8")
    assert "../../shared-skills/blueprint-canon/" in text


def test_no_dead_qoder_orchestration_paths():
    forbidden = (".qoder/", "pysci.skills.orchestration", "pysci-orch")
    roots = [*SKILL_ROOTS, PROJECT_ROOT / "src" / "pysci", PROJECT_ROOT / "sub_root" / "tests"]
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".py", ".toml", ".ps1", ".mjs"}:
                continue
            if path == Path(__file__) or path.name == "test_blueprint.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                assert token not in text, f"dead path {token!r} in {path}"
