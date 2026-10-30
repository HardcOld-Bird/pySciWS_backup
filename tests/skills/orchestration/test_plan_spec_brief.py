"""orch 引导一致性（tb-20261017-211704-2-2 F1/F3/F4/F5）。

- F1 ``plan spec``：环节已入队/待证据后仍能回打「粗描述 + evidence 标题段 + 收尾命令」，
  且不改状态；标题段与 ``_check_evidence`` 同源（``ROLE_EVIDENCE_SECTIONS``）。
- F3 ``status --brief``：≤30 行，含计划进度/队列在途/事件水位/大任务队首四要素。
- F4 引导文案：收尾命令一律裸 ``pysci-orch``（basic.md §3 首选裸命令）。
- F5 ``plan reassign --step``：显示名（``B.b2``）与短 id（``b2``）同效。
"""

from __future__ import annotations

import argparse

import pytest

from pysci.skills.orchestration.tools import orch, plans

PLAN_SRC = """id: p1
goal: 引导一致性演示
steps:
  - id: A
    mode: serial
    brief: 组 A
    children:
      - id: a1
        kind: self
        role: cause
        brief: 成因分析环节
      - id: a2
        member: devops
        brief: 后续派发环节
  - id: B
    mode: serial
    brief: 组 B
    children:
      - id: b1
        member: devops
        brief: 组内前置
      - id: b2
        member: figure
        brief: 组内待改派
        member_placeholder: true
"""


@pytest.fixture
def plan(qface, tmp_path, capsys):
    src = tmp_path / "p1.plan-src.yaml"
    src.write_text(PLAN_SRC, encoding="utf-8")
    assert (
        plans.cmd_plan_new(argparse.Namespace(file=str(src), id="p1", force=False)) == 0
    )
    capsys.readouterr()
    return "p1"


def _status(pid: str, sid: str) -> str:
    st = plans._load_state(pid)
    return next(str(r["status"]) for r in st["steps"] if str(r["id"]) == sid)


# ---------------------------------------------------------------------------
# F1：plan spec 在任何状态回打规格，且不改状态
# ---------------------------------------------------------------------------
def test_spec_reprints_self_step_after_awaiting(plan, capsys):
    """self 环节 advance 后（self_awaiting），spec 仍回打粗描述/标题段/收尾命令。"""
    assert orch.main(["plan", "advance", plan]) == 0
    capsys.readouterr()
    assert _status(plan, "A.a1") == plans.SELF_AWAITING

    assert orch.main(["plan", "spec", plan]) == 0
    out = capsys.readouterr().out
    assert "粗描述原文：成因分析环节" in out, out
    # 标题段单一真源：role=cause → ROLE_EVIDENCE_SECTIONS["cause"]
    for sec in plans.ROLE_EVIDENCE_SECTIONS["cause"]:
        assert sec in out, out
    assert f"pysci-orch plan advance {plan} --evidence" in out, out
    assert "uv run" not in out, "引导面须裸命令（basic.md §3）"
    assert _status(plan, "A.a1") == plans.SELF_AWAITING, "spec 是查询，不得改状态"


def test_spec_step_accepts_short_id_and_shows_dispatch_guidance(plan, capsys):
    assert orch.main(["plan", "spec", plan, "--step", "a2"]) == 0
    out = capsys.readouterr().out
    assert "环节 A.a2" in out, out
    assert "粗描述原文：" in out and "后续派发环节" in out, out
    assert 'pysci-orch plan advance p1 --text "<任务书全文>"' in out, out


def test_spec_unknown_step_reports_forms(plan, capsys):
    assert orch.main(["plan", "spec", plan, "--step", "zz"]) == 2
    out = capsys.readouterr().out
    assert "可用环节" in out and "B.b2 或 b2" in out, out


# ---------------------------------------------------------------------------
# F3：status --brief 一页四要素（≤30 行）
# ---------------------------------------------------------------------------
def test_status_brief_is_one_page_with_four_elements(plan, iso_state, capsys):
    assert orch.main(["status", "--brief"]) == 0
    out = capsys.readouterr().out
    lines = out.rstrip("\n").splitlines()
    assert len(lines) <= orch._BRIEF_MAX_LINES, out
    assert any("▶ p1" in ln and "0/4（当前环节 A.a1）" in ln for ln in lines), out
    assert any("在途=" in ln for ln in lines), out
    assert any("事件水位" in ln for ln in lines), out
    assert any("队首：" in ln for ln in lines), out


# ---------------------------------------------------------------------------
# F4：引导打印面不得残留 uv run 前缀
# ---------------------------------------------------------------------------
def test_guidance_faces_use_bare_pysci_orch(plan, capsys):
    assert orch.main(["plan", "spec", plan, "--step", "a2"]) == 0
    assert "uv run" not in capsys.readouterr().out
    assert orch.main(["plan", "spec", plan, "--step", "a1"]) == 0
    assert "uv run" not in capsys.readouterr().out


def test_no_uv_prefix_in_orchestration_sources():
    """静态面：tools 里不再有旧前缀引导（防止新加提示语回退）。"""
    from pysci.paths import PROJECT_ROOT

    tools = PROJECT_ROOT / "src" / "pysci" / "skills" / "orchestration"
    hits = [
        f"{p.relative_to(PROJECT_ROOT).as_posix()}"
        for p in tools.rglob("*.py")
        if "uv run --no-sync pysci-orch" in p.read_text(encoding="utf-8")
    ]
    assert not hits, hits


# ---------------------------------------------------------------------------
# F5：reassign --step 显示名与短 id 同效
# ---------------------------------------------------------------------------
def test_reassign_short_id_and_display_name_equivalent(plan, capsys):
    assert (
        orch.main(["plan", "reassign", plan, "--step", "b2", "--member", "writing"])
        == 0
    )
    out = capsys.readouterr().out
    assert "环节 B.b2 成员：figure → writing" in out, out
    leaf = plans._leaf_map(plans._load_yaml(plan))["B.b2"]
    assert leaf["member"] == "writing" and "member_placeholder" not in leaf

    assert (
        orch.main(["plan", "reassign", plan, "--step", "B.b2", "--member", "figure"])
        == 0
    )
    assert plans._leaf_map(plans._load_yaml(plan))["B.b2"]["member"] == "figure"
