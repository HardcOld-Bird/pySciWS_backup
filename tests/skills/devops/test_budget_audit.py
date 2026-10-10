"""harness 预算三档审计（backlog 20261010-budget-doctor）+ doctor 接线回归。

锁定：① 三档扫描对象齐全且按落盘面计量；② 超限者被 report 列为 [!] 且 cmd_doctor 计入
退出码 2；③ 常驻暴露档按 description 行合计判；④ 真实仓库现状全部在预算内（即任务书
「find … -size +8192c 输出为空」的机械版）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pysci.skills.orchestration.tools.sync as sync_mod
from pysci.skills.devops.tools import budget as budget_mod
from pysci.skills.devops.tools import dev


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """把三档根全部指到 tmp_path 下的假树，返回可写的目录句柄。"""
    skills = tmp_path / "skills"
    rules = tmp_path / "rules"
    pods = tmp_path / "pods"
    for d in (skills, rules, pods):
        d.mkdir()
    monkeypatch.setattr(budget_mod, "SKILLS_ROOT", skills)
    monkeypatch.setattr(budget_mod, "RULES_ROOT", rules)
    monkeypatch.setattr(budget_mod, "PODS_ROOT", pods)
    monkeypatch.setattr(dev, "PODS_ROOT", pods)
    monkeypatch.setattr(sync_mod, "sync", lambda check_only=False: [])
    return SimpleNamespace(skills=skills, rules=rules, pods=pods)


def _mk(path, size: int, header: str = "") -> None:
    """写出恰好 ``size`` 字节的文本文件（header 长于 size 时按 header 长度落盘）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pad = "x" * max(0, size - len(header.encode("utf-8")))
    path.write_text(header + pad, encoding="utf-8")


def test_three_tiers_scanned(tree):
    _mk(tree.skills / "lit" / "SKILL.md", 500, "---\ndescription: 检索文献。\n---\n")
    _mk(tree.skills / "lit" / "references" / "search-01.md", 700)
    _mk(tree.rules / "basic.md", 900)
    _mk(tree.pods / "lit" / "AGENTS.md", 800)
    _mk(tree.pods / "lit" / ".qoder" / "rules" / "charter.md", 600)
    _mk(tree.pods / "lit" / ".qoder" / "rules" / "my-notes.md", 400)
    tiers = {f.tier for f in budget_mod.audit()}
    assert tiers == {"全量注入", "按需"}
    expected = {
        "SKILL.md",
        "search-01.md",
        "basic.md",
        "AGENTS.md",
        "charter.md",
        "my-notes.md",
    }
    assert expected <= {f.path.name for f in budget_mod.audit()}


def test_over_limit_flagged_and_sized_on_disk(tree):
    _mk(tree.skills / "lit" / "references" / "big.md", budget_mod.LIMIT + 1)
    findings = budget_mod.audit()
    (bad,) = [f for f in findings if f.over]
    assert bad.size == budget_mod.LIMIT + 1
    assert bad.size == bad.path.stat().st_size
    (line,) = [ln for ln in budget_mod.report(findings, []) if ln.startswith("[!]")]
    assert "超预算" in line and "big.md" in line and "按需" in line


def test_within_limit_reports_clean(tree):
    _mk(tree.skills / "lit" / "references" / "ok.md", budget_mod.LIMIT)
    lines = budget_mod.report(budget_mod.audit(), [])
    assert lines[0] == "  [√] 三档全部在预算内"
    assert not [ln for ln in lines if ln.startswith("[!]")]


def test_description_total_over_budget_flagged(tree):
    for name in ("a", "b"):
        _mk(
            tree.skills / name / "SKILL.md",
            500,
            f"---\ndescription: {'y' * 5000}\n---\n",
        )
    desc = budget_mod.description_findings()
    assert [d.size for d in desc] == [5013, 5013]  # 5000 正文 + 13 字节 "description: "
    lines = budget_mod.report(budget_mod.audit(), desc)
    assert any("description 合计" in ln and "超预算" in ln for ln in lines)


def test_missing_roots_degrade_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(budget_mod, "SKILLS_ROOT", tmp_path / "nope")
    monkeypatch.setattr(budget_mod, "RULES_ROOT", tmp_path / "nope")
    monkeypatch.setattr(budget_mod, "PODS_ROOT", tmp_path / "nope")
    monkeypatch.setattr(dev, "PODS_ROOT", tmp_path / "nope")
    assert budget_mod.audit() == [] and budget_mod.description_findings() == []


def test_doctor_exits_2_on_budget_violation(tree, capsys):
    _mk(tree.skills / "lit" / "references" / "huge.md", budget_mod.LIMIT + 5000)
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    out = capsys.readouterr().out
    assert rc == 2
    assert "harness 预算审计" in out and "huge.md" in out


def test_doctor_reports_clean_budget(tree, capsys):
    _mk(tree.skills / "lit" / "SKILL.md", 400, "---\ndescription: 短。\n---\n")
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "三档全部在预算内" in out


def test_repo_state_within_budget():
    """真实仓库（默认根，不打桩）：真本/根 rules/pod 文档每文件 ≤8192B，
    description 合计 ≤8192B——任务书 ③「find … -size +8192c 输出为空」的机械哨兵。
    """
    over = [f for f in budget_mod.audit() if f.over]
    assert not over, [(str(f.path), f.size) for f in over][:5]
    assert sum(d.size for d in budget_mod.description_findings()) <= budget_mod.LIMIT
