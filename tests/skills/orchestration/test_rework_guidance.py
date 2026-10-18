"""返工标题／引导／active 接线（用户 2026-10-15 改进建议 3+4 + 组长实锤缺陷）。

钉四件事（对应任务书 B/C/D 的 success_criteria）：
① **返工标题**（B）：``create_rework`` 自动增补「（返工第 n 轮）」，且**逐轮不叠加**；
   仓内存量返工件（已入库数据面）标题同样带轮次后缀；
② **引导补全**（C）：``command-surface.md`` 有 ``plan rework`` 行；``TASKBOOK-NEW`` 事件对
   返工件给**专属** ``plan rework`` 引导（非返工仍走通用 link 引导）；``plan rework`` 输出
   带 [NEXT]；``orch next`` 卡片对返工件给 ``plan rework`` 命令（不再说 ``plan new``）；
   TUI 驳回落账即提示下一步命令；
③ **active 接线**（D）：plan 首环节开始执行（入队 / ``kind:self`` 引导）→ 所链任务书
   ``planning → active``；幂等、不误伤别的 plan；
④ **[NEXT] 不回退**：advance 的三栏工作清单仍在（原本就有，别被新引导挤掉）。
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace

import pytest
import yaml
from textual.widgets import TextArea

from pysci import paths as _paths
from pysci.skills.orchestration.tools import (
    events,
    nextroute,
    orch,
    plans,
    queueworker,
    taskbooks,
)
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(iso_state, tmp_path, monkeypatch):
    """plans / taskbooks / data 三面 + 队列唤醒桩（同 test_rework_flow.rface 口径）。"""
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


def _add_plan(tmp, plan_id: str, steps: list[dict]) -> str:
    p = tmp / "plans" / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(
            {"id": plan_id, "goal": f"目标 {plan_id}", "steps": steps},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert plans.cmd_plan_new(_args(plan_id, file=str(p))) == 0
    return plan_id


def _rework_taskbook(orig_title: str = "目标大任务", *, round_no: int = 0) -> str:
    """造一件返工任务书（round_no=0 时直接 create，round_no>0 走 create_rework 链）。"""
    if round_no <= 0:
        return taskbooks.create(
            orig_title, "简介", mode="full", rework_of="tb-orig", rework_round=1
        )["id"]
    orig = taskbooks.create(orig_title, "简介")["id"]
    taskbooks.link_plan(orig, "p-x")
    taskbooks.transition(orig, "active")
    taskbooks.transition(orig, "delivered")
    return taskbooks.create_rework(orig, mode="full", note="驳回")["id"]


# ===========================================================================
# ① 返工标题（B）
# ===========================================================================
def test_rework_title_appends_round_suffix():
    assert taskbooks.rework_title("目标大任务", 1) == "目标大任务（返工第 1 轮）"
    assert taskbooks.rework_title("目标大任务", 3) == "目标大任务（返工第 3 轮）"
    # 逐轮不叠加：原件本身已是上一轮的返工件时，先剥旧后缀再补新的
    assert (
        taskbooks.rework_title("目标大任务（返工第 1 轮）", 2)
        == "目标大任务（返工第 2 轮）"
    )


def test_create_rework_titles_each_round_once(face):
    """success_criteria：返工任务书标题含「（返工第 n 轮）」——第 1 轮与链式第 2 轮各一次。"""
    orig = taskbooks.create("目标大任务", "简介")["id"]
    taskbooks.transition(orig, "active")
    taskbooks.transition(orig, "delivered")
    r1 = taskbooks.create_rework(orig, mode="full", note="r1")
    assert r1["title"] == "目标大任务（返工第 1 轮）"
    taskbooks.link_plan(r1["id"], "p-r1")
    taskbooks.transition(r1["id"], "active")
    taskbooks.transition(r1["id"], "delivered")
    r2 = taskbooks.create_rework(r1["id"], mode="no_inspector", note="r2")
    assert r2["title"] == "目标大任务（返工第 2 轮）", "逐轮不叠加"
    assert r2["rework_round"] == 2
    # 落盘面同样带后缀（列表/详情/GUI 都直读真源，不必各视图重实现）
    state = json.loads(
        (taskbooks.TASKBOOKS_DIR / f"{r2['id']}.state.json").read_text(encoding="utf-8")
    )
    assert state["title"] == "目标大任务（返工第 2 轮）"
    assert "目标大任务（返工第 2 轮）" in (
        taskbooks.TASKBOOKS_DIR / f"{r2['id']}.taskbook.md"
    ).read_text(encoding="utf-8")


def test_shipped_rework_taskbooks_carry_round_suffix():
    """**存量补迁移**（数据面）：仓内已入库的返工件标题都带轮次后缀（tb-20261015-091655-1）。

    读面是 shipped ``data/taskbooks``（**只读**，哨兵只拦写）；该目录里没有返工件时，
    该不变量自然成立（空集），不会因日后清理数据而变红。
    """
    for st in taskbooks.list_all():
        if int(st.get("rework_round") or 0) <= 0:
            continue
        rnd = int(st["rework_round"])
        assert f"（返工第 {rnd} 轮）" in str(st.get("title") or ""), st


# ===========================================================================
# ② 引导补全（C）
# ===========================================================================
def test_command_surface_lists_plan_rework():
    """success_criteria：``command-surface.md``（技能真本）含 ``plan rework`` 行。"""
    path = (
        _paths.PROJECT_ROOT
        / "orchestration/skills/orchestration/references/command-surface.md"
    )
    text = path.read_text(encoding="utf-8")
    assert "plan rework" in text, "命令全表漏了 plan rework（组长实测按 grep 才找到）"
    assert "--mode" in text.split("plan rework", 1)[1].split("\n", 1)[0]


def test_taskbook_new_event_gives_rework_specific_guidance(face):
    """success_criteria：返工任务书的 TASKBOOK-NEW 事件给**专属**引导。"""
    tid = _rework_taskbook("目标大任务", round_no=1)
    evs = [
        e
        for e in events.read_all()
        if e["event"] == "TASKBOOK-NEW" and e["subject"] == tid
    ]
    assert len(evs) == 1
    ev = evs[0]
    assert ev["subject"] == tid
    assert "返工" in ev["detail"] and "第 1 轮" in ev["detail"]
    assert f"plan rework {tid}" in ev["action"], ev["action"]
    assert "改进项按任务书标志" in ev["action"], "改进项随任务书标志，无需 --mode"
    assert "--mode" not in ev["action"], "三选一 mode 已退役，不该再出现在引导里"
    assert "taskbooks link <id> --plan <pid>" not in ev["action"], (
        "返工件不该只给通用 link 引导"
    )


def test_taskbook_new_event_keeps_generic_guidance_for_plain(face):
    """非返工任务书仍走通用引导（专属引导是**加**不是**替**普通件的）。"""
    tid = taskbooks.create("普通大任务", "正文")["id"]
    ev = [e for e in events.read_all() if e["event"] == "TASKBOOK-NEW"][-1]
    assert ev["subject"] == tid
    assert "用户投递了新任务书" in ev["detail"]
    assert "taskbooks link <id> --plan <pid>" in ev["action"]


def test_plan_rework_output_has_next_guidance(face, capsys):
    """``plan rework`` 登记后给 [NEXT]：链接任务书 → 推进首环节（原样命令，不靠 grep）。"""
    tb = _rework_taskbook("目标大任务")
    capsys.readouterr()
    assert plans.cmd_plan_rework(_args(tb, mode="full")) == 0
    out = capsys.readouterr().out
    pid = f"rework-{tb}"
    assert "[NEXT]" in out
    assert f"orch taskbooks link {tb} --plan {pid}" in out
    assert f"orch plan advance {pid}" in out
    assert "返工计划已登记" in out  # 既有内容不回退


def test_next_route_card_offers_plan_rework_for_rework_taskbook(face):
    """``orch next`` 卡片：返工件（未建 plan）的下一步是 ``plan rework``，不是 ``plan new``。"""
    tb = _rework_taskbook("目标大任务")
    acts = [a for a in nextroute.collect_actions() if a.get("tb_id") == tb]
    assert len(acts) == 1
    act = acts[0]
    assert "返工骨架" in act["action"]
    assert f"plan rework {tb} --mode full" in act["command"]
    assert "plan new" not in act["command"]


def test_tui_reject_message_shows_next_command(face):
    """TUI 三段式驳回落账即提示下一步命令（reject 输出的引导补全）。"""
    tid = taskbooks.create("待驳回大任务", "正文")["id"]
    taskbooks.link_plan(tid, "p-x")
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            assert app.view == "tb-reject"
            app.query_one("#tb-note", TextArea).text = "重新来"
            await pilot.click("#tb-reject-submit")
            return app.last_rework, app.tb_reject_msg

    new_id, msg = _run(_go())
    assert new_id.startswith("tb-")
    assert f"orch plan rework {new_id} 生成返工骨架（改进项按任务书标志）" in msg, msg
    assert "--mode" not in msg, "三选一 mode 已退役，提示语不该再出现"
    assert taskbooks.load_state(tid)["status"] == "rejected"


# ===========================================================================
# ③ active 接线（D）
# ===========================================================================
def test_sync_active_moves_planning_to_active(face):
    tid = taskbooks.create("大任务", "正文")["id"]
    taskbooks.link_plan(tid, "p-1")
    assert taskbooks.load_state(tid)["status"] == "planning"
    assert taskbooks.sync_active("p-1") == [tid]
    assert taskbooks.load_state(tid)["status"] == "active"
    assert taskbooks.sync_active("p-1") == []  # 幂等


def test_sync_active_ignores_other_plans_and_unlinked(face):
    a = taskbooks.create("甲", "正文")["id"]
    b = taskbooks.create("乙", "正文")["id"]
    taskbooks.link_plan(a, "p-a")
    taskbooks.link_plan(b, "p-b")
    assert taskbooks.sync_active("p-a") == [a]
    assert taskbooks.load_state(b)["status"] == "planning", "别的 plan 的任务书不动"
    assert taskbooks.sync_active("p-none") == []


def test_first_dispatch_step_marks_taskbook_active(face, capsys):
    """success_criteria：plan 首环节**入队**即置任务书 active（可观察）。"""
    tb = taskbooks.create("大任务", "正文")["id"]
    taskbooks.link_plan(tb, "p-act")
    _add_plan(face, "p-act", [{"id": "s1", "member": "sim", "brief": "甲"}])
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("p-act", text="任务书正文")) == 0
    out = capsys.readouterr().out
    assert taskbooks.load_state(tb)["status"] == "active"
    assert "[NEXT] 工作清单" in out, "advance 的三栏工作清单不回退"
    # 可观察：`orch taskbooks list` 的徽标面（active=执行中）
    assert orch.main(["taskbooks", "list"]) == 0
    listing = capsys.readouterr().out
    assert f"[active] {tb}" in listing


def test_first_self_step_marks_taskbook_active(face):
    """kind:self 首环节（不入队列，只做引导）同样算「开始执行」。"""
    tb = taskbooks.create("大任务", "正文")["id"]
    taskbooks.link_plan(tb, "p-self")
    _add_plan(face, "p-self", [{"id": "a1", "kind": "self", "brief": "成因分析"}])
    assert plans.cmd_plan_advance(_args("p-self")) == 0
    assert taskbooks.load_state(tb)["status"] == "active"


def test_plan_run_first_step_marks_taskbook_active(face, capsys, monkeypatch):
    """``plan run``（阻塞兼容形态）同样接线——两分支一个单点实现。"""
    from pysci.skills.orchestration.tools import queues, workq

    def _fake_worker(member: str, **kw):
        """假 worker：把队列条目直接标 done（否则 run 的阻塞等待会真睡到超时）。"""
        qp = queues.queue_path(member)
        for it in workq.load(qp):
            if str(it.get("status")) == workq.STATUS_DONE:
                continue
            workq.mutate(
                qp,
                it["id"],
                status=workq.STATUS_DONE,
                result={
                    "code": 0,
                    "kind": "result",
                    "body": "done",
                    "checks": [],
                    "member": member,
                },
            )
        return (True, "stub worker")

    monkeypatch.setattr(queueworker, "wake_member", _fake_worker)
    tb = taskbooks.create("大任务", "正文")["id"]
    taskbooks.link_plan(tb, "p-run")
    _add_plan(face, "p-run", [{"id": "s1", "member": "sim", "brief": "甲"}])
    capsys.readouterr()
    plans.cmd_plan_run(_args("p-run", text="任务书正文"))
    assert taskbooks.load_state(tb)["status"] == "active", capsys.readouterr().out
