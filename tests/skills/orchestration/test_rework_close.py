"""PSEP_3 pd2：返工流**闭环**——历次展示 / 复盘段 lint / 全流程报告组装 / TUI 卡片 / 循环 e2e。

钉新面（§4.3 #4/#5 + §4.5）：① rework_round≥2 的 a1 引导**全文展开**前轮报告；
② 本轮成因分析报告须含「前次改进复盘」段（lint）；③ ``taskbooks report`` 组装八段大报告；
④ TUI 返工卡片（轮次/模式/优先级/裁决书摘要/改进清单/superstep 进度树）；⑤ 循环端到端。
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from pysci import paths as _paths
from pysci.skills.orchestration.tools import (
    jury,
    orch,
    plans,
    queueworker,
    taskbooks,
)

REPORT_SECTIONS = (
    "任务概述",
    "历轮摘要",
    "本轮成因分析",
    "解决方案",
    "裁决书",
    "监察判断",
    "改进清单",
    "返工产物报告",
)


@pytest.fixture
def cface(iso_state, tmp_path, monkeypatch):
    """闭环夹具：``iso_state``（全状态面隔离，含 QUEUES_DIR/wake stub 前置修复）之上补三面。"""
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（测试不消化）")
    )
    return tmp_path


def _args(plan_id: str, **over) -> Namespace:
    base = dict(
        id=plan_id,
        text=None,
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
        file="",
        force=True,
        evidence=None,
        dry=False,
        mode="full",
        goal=None,
        plan_id=None,
    )
    base.update(over)
    return Namespace(**base)


def _deliver(tid: str) -> None:
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")


def _ev(tmp: Path, name: str, sections: list[str], marker: str = "") -> str:
    p = tmp / name
    body = "\n".join(f"# {s}\n内容 {marker}-{s}\n" for s in sections)
    p.write_text(body, encoding="utf-8")
    return str(p)


def _jury(case_id: str, verdict: str, text: str = "") -> None:
    p = jury.record_path(case_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "verdict": verdict,
                "final_verdict": verdict,
                "verdict_summary": {"text": text or f"裁决={verdict}"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _guide_and_finish(plan_id: str, ev_path: str, tmp: Path):
    """引导当前 self 环节 → 提交 evidence 收尾。"""
    assert plans.cmd_plan_advance(_args(plan_id)) == 0  # 引导
    return plans.cmd_plan_advance(_args(plan_id, evidence=ev_path))


def _drive_A(cface: Path, plan_id: str, tb_id: str, *, round_no: int = 1) -> None:
    """推进 A.a1..a4（mock 送审 A）。a1 段按轮次选标题。"""
    a1_secs = (
        ["前次改进复盘", "成因分析", "结论"] if round_no >= 2 else ["成因分析", "结论"]
    )
    for step, secs in (
        ("a1", a1_secs),
        ("a2", ["解决方案", "结论"]),
        ("a3", ["送审结论"]),
    ):
        ev = _ev(cface, f"{plan_id}-{step}.md", secs, marker=f"{plan_id}{step}")
        assert _guide_and_finish(plan_id, ev, cface) == 0, step
    _jury(f"{tb_id}-jury", "A", "裁决=A（jury 定案；计票 A:2 B:1）")
    ev4 = _ev(cface, f"{plan_id}-a4.md", ["拆解派单"], marker=f"{plan_id}a4")
    assert _guide_and_finish(plan_id, ev4, cface) == 0


# ===========================================================================
#  ① 历次改进强制展示 + ② 复盘段 lint
# ===========================================================================
def test_a1_expands_previous_reports_full_text(cface, capsys):
    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", note="第一轮")["id"]
    _deliver(tb2)
    tb3 = taskbooks.create_rework(
        tb2, mode="no_inspector", priority="high", note="第二轮"
    )["id"]
    # 前两轮报告留痕（组装件）
    taskbooks.report_path(tb1).parent.mkdir(parents=True, exist_ok=True)
    taskbooks.report_path(tb1).write_text(
        "# R1\n 甲轮报告 MARKER_ALPHA\n", encoding="utf-8"
    )
    taskbooks.report_path(tb2).write_text(
        "# R2\n 乙轮报告 MARKER_BETA\n", encoding="utf-8"
    )

    plans.cmd_plan_rework(_args("x", id=tb3, mode="no_inspector"))
    plan_id = f"rework-{tb3}"
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args(plan_id)) == 0  # 引导 a1
    out = capsys.readouterr().out
    assert "历次改进强制展示" in out
    assert "MARKER_ALPHA" in out and "MARKER_BETA" in out, (
        "前轮报告须**全文**展开（非仅路径）"
    )
    assert "前次改进复盘" in out, "a1 引导须点名复盘段要求"


def test_a1_requires_review_section_lint(cface, tmp_path, capsys):
    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", note="r1")["id"]
    _deliver(tb2)
    tb3 = taskbooks.create_rework(tb2, mode="no_inspector", note="r2")["id"]
    plans.cmd_plan_rework(_args("x", id=tb3, mode="no_inspector"))
    plan_id = f"rework-{tb3}"
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args(plan_id)) == 0
    bad = _ev(tmp_path, "bad.md", ["成因分析", "结论"])  # 缺「前次改进复盘」
    assert plans.cmd_plan_advance(_args(plan_id, evidence=bad)) == 2
    assert "前次改进复盘" in capsys.readouterr().out
    good = _ev(tmp_path, "good.md", ["前次改进复盘", "成因分析", "结论"])
    assert plans.cmd_plan_advance(_args(plan_id, evidence=good)) == 0


# ===========================================================================
#  ③ 全流程报告组装器
# ===========================================================================
def test_build_report_has_all_sections(cface, capsys):
    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", priority="high", note="r1")[
        "id"
    ]
    plans.cmd_plan_rework(_args("x", id=tb2, mode="no_inspector"))
    plan_id = f"rework-{tb2}"
    taskbooks.link_plan(tb2, plan_id)
    capsys.readouterr()
    _drive_A(cface, plan_id, tb2, round_no=1)

    text = taskbooks.build_report(tb2)
    for sec in REPORT_SECTIONS:
        assert f"## {sec}" in text, f"缺段：{sec}"
    assert "MARKER" in text or "拆解派单" in text
    assert "裁决=A" in text, "裁决书段须含送审裁决"
    assert taskbooks.report_path(tb2).exists()
    assert taskbooks.load_state(tb2)["report"].endswith(f"{tb2}.report.md")
    # orch 门面
    assert orch.main(["taskbooks", "report", tb2]) == 0
    assert "全流程报告已组装" in capsys.readouterr().out


# ===========================================================================
#  ④ TUI 返工卡片
# ===========================================================================
def test_rework_card_content(cface, capsys):
    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", priority="high", note="r1")[
        "id"
    ]
    plans.cmd_plan_rework(_args("x", id=tb2, mode="no_inspector"))
    plan_id = f"rework-{tb2}"
    taskbooks.link_plan(tb2, plan_id)
    capsys.readouterr()
    _drive_A(cface, plan_id, tb2, round_no=1)

    card = "\n".join(taskbooks.rework_card(tb2))
    assert "轮次：第 1 轮返工" in card
    assert "模式：no_inspector" in card and "优先级：high" in card
    assert "各轮裁决书摘要" in card and "裁决=A" in card
    assert "改进清单" in card and "superstep 进度树" in card
    assert "A [serial]" in card and "a1 (self=done)" in card
    assert taskbooks.rework_card(tb1) == [], "普通任务书不出返工卡片"


def test_tui_enter_shows_rework_card(cface, capsys):
    import asyncio

    from textual.widgets import ListView

    from pysci.skills.orchestration.tools.tui import app as tui

    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", priority="high", note="r1")[
        "id"
    ]
    plans.cmd_plan_rework(_args("x", id=tb2, mode="rework_only"))
    taskbooks.link_plan(tb2, f"rework-{tb2}")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            lv = app.query_one("#tb-list", ListView)
            idx = next(i for i, d in enumerate(app.tb_items) if d["id"] == tb2)
            lv.index = idx
            await pilot.pause()
            for _ in range(3):  # 键驱动有竞态：按 Enter 直到详情页落定
                await pilot.press("enter")
                await pilot.pause()
                if app.view == "tb-detail":
                    break
            assert app.view == "tb-detail"
            assert "返工卡片" in app.tb_detail_body
            assert "第 1 轮返工" in app.tb_detail_body

    asyncio.run(_go())


# ===========================================================================
#  ⑤ 循环语义端到端
# ===========================================================================
def test_loop_reject_generate_expand_report(cface, tmp_path, capsys):
    # 1) 首次交付 → 驳回 → tb2(round1)
    tb1 = taskbooks.create("大任务", "简介")["id"]
    _deliver(tb1)
    tb2 = taskbooks.create_rework(
        tb1, mode="no_inspector", priority="normal", note="初轮驳回"
    )["id"]
    assert tb2 and taskbooks.load_state(tb2)["rework_round"] == 1
    plans.cmd_plan_rework(_args("x", id=tb2, mode="no_inspector"))
    plan2 = f"rework-{tb2}"
    taskbooks.link_plan(tb2, plan2)
    capsys.readouterr()
    _drive_A(cface, plan2, tb2, round_no=1)  # 模板生成 + mock 送审 A 推进
    taskbooks.build_report(tb2)  # 第 1 轮全流程报告

    # 2) 再驳回 → tb3(round2)
    _deliver(tb2)
    tb3 = taskbooks.create_rework(
        tb2, mode="no_inspector", priority="high", note="第二轮驳回"
    )["id"]
    st3 = taskbooks.load_state(tb3)
    assert st3["rework_round"] == 2 and st3["rework_of"] == tb2
    plans.cmd_plan_rework(_args("x", id=tb3, mode="no_inspector"))
    plan3 = f"rework-{tb3}"
    capsys.readouterr()
    # 3) 第 2 轮 a1 引导含第 1 轮报告全文 + 复盘段
    assert plans.cmd_plan_advance(_args(plan3)) == 0
    out = capsys.readouterr().out
    assert "历次改进强制展示" in out and "第 1 轮" in out
    assert "## 任务概述" in out, "展开的须是第 1 轮**全流程报告全文**"
    # 4) 报告组装含各段
    taskbooks.link_plan(tb3, plan3)
    _drive_A(cface, plan3, tb3, round_no=2)
    text = taskbooks.build_report(tb3)
    for sec in REPORT_SECTIONS:
        assert f"## {sec}" in text
