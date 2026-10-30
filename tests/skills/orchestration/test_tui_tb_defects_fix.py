"""任务书界面 4 缺陷修复（用户 2026-10-15 驳回流程实测；backlog 20261015-230248）。

① 驳回批语**多行**：控件 `Input` → `TextArea`；提交后 `state.json` 的 note **含换行**，
   报告与历史完整展示（不再只存首行）。
② 状态**快照刷新**：终态变更时重生成 `<id>.report.md`——否则交付物永远写着 sync 前的
   `状态：active`（实测：transition 后文件仍是 active，而 state.json 已是 delivered）。
③ 确认即**关页回列表**：投递成功回列表（防重复提交）；驳回落盘同样回列表（既有行为，一并钉住）。
④ ESC 再进**无残留**：RadioSet 复位改两段式（先全清再置目标）——逐个写 `value=(i==index)`
   会留下 `[True,False,True]`（残留值与默认值同时被选中=双绿）。
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from textual.widgets import RadioButton, RadioSet, SelectionList, TextArea

from pysci import paths as _paths
from pysci.skills.orchestration.tools import ledger, plans, queueworker, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui

PLAN_ID = "p-demo"
NOTE = "第一行：明显不达标问题：标题量纲与色标不符\n第二行：请补齐色标上限并核对 p_t\n第三行：外部 logo 需清除"


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离四面：任务书目录 / 计划目录 / 交付留档 / 台账；＋settings（表单默认值手改面）。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    taskbooks.TASKBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    monkeypatch.setattr(queueworker, "DELIVERIES_DIR", tmp_path / "deliveries")
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    # 表单默认值隔离：不 patch 就读到用户手改的真源 toml（2026-10-17 实案：默认值漂移打红断言）
    from pysci.skills.orchestration.tools import settings as _settings

    monkeypatch.setattr(_settings, "SETTINGS_PATH", tmp_path / "no-settings.toml")
    return tmp_path


def _write_plan() -> None:
    (plans.PLANS_DIR / f"{PLAN_ID}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": PLAN_ID,
                "goal": "demo",
                "steps": [
                    {"id": "s1", "member": "lit", "brief": "一"},
                    {"id": "s2", "member": "writing", "brief": "二"},
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plans._save_state(
        PLAN_ID,
        {
            "plan_id": PLAN_ID,
            "created": "2026-10-15T12:00:00+08:00",
            "goal": "demo",
            "steps": [
                {"id": "s1", "status": "done", "member": "lit"},
                {"id": "s2", "status": "queued", "member": "writing"},
            ],
        },
    )


def _mk(title: str, status: str) -> str:
    """造一件处于指定状态的任务书（走合法迁移，不直接改 state.json）。"""
    tid = taskbooks.create(title, f"{title} 的正文")["id"]
    taskbooks.link_plan(tid, PLAN_ID)
    taskbooks.transition(tid, "active")
    if status != "active":
        taskbooks.transition(tid, status)
    return tid


def _run(coro):
    return asyncio.run(coro)


def _state(tid: str) -> dict:
    return json.loads(
        (taskbooks.TASKBOOKS_DIR / f"{tid}.state.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# ② 终态变更 → 报告快照刷新
# ---------------------------------------------------------------------------
def test_terminal_transition_refreshes_report_snapshot(face):
    _write_plan()
    tid = _mk("快照任务书", "active")
    taskbooks.build_report(tid)  # sync 前生成：写着 active
    assert "状态：active" in taskbooks.report_path(tid).read_text(encoding="utf-8")

    taskbooks.transition(tid, "delivered")

    txt = taskbooks.report_path(tid).read_text(encoding="utf-8")
    assert "状态：delivered" in txt
    assert "状态：active" not in txt  # 交付物不再留旧口径


def test_terminal_transition_does_not_conjure_report(face):
    """没有报告文件时**不凭空造**（刷新 ≠ 生成）。"""
    _write_plan()
    tid = _mk("无报告件", "active")
    taskbooks.transition(tid, "delivered")
    assert not taskbooks.report_path(tid).exists()


def test_sync_delivered_also_refreshes(face):
    """`sync_delivered` 走同一 transition 收口 → 同样刷新（口径只有一份）。"""
    _write_plan()
    tid = _mk("sync 件", "active")
    taskbooks.build_report(tid)
    plans._save_state(
        PLAN_ID,
        {
            **plans._load_state(PLAN_ID),
            "completed_at": "2026-10-15T13:00:00+08:00",
        },
    )

    assert taskbooks.sync_delivered() == [tid]

    assert "状态：delivered" in taskbooks.report_path(tid).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ① 多行批语：控件 + 落盘全文 + 展示完整
# ---------------------------------------------------------------------------
def test_reject_note_widget_is_multiline_textarea(face):
    _write_plan()
    _mk("控件件", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            return app.query_one("#tb-note")

    assert isinstance(_run(_go()), TextArea)


def test_multiline_note_stored_whole_and_shown(face):
    """criteria：多行批语提交 → state.json note 全文含换行 → 报告/历史完整显示。"""
    _write_plan()
    tid = _mk("多行件", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            app.query_one("#tb-note", TextArea).text = NOTE
            await pilot.pause()
            await app._submit_reject()
            await pilot.pause()
            return app.last_rework

    new_id = _run(_go())

    # 落盘：全文含换行（不再只存首行）
    assert _state(tid)["rejections"][0]["note"] == NOTE
    # 新返工任务书正文回抄批语全文
    assert NOTE in taskbooks.read_body(new_id)
    # 详情页（历史/批复段）完整展示三行
    detail = "\n".join(taskbooks.detail_lines(tid))
    for line in NOTE.splitlines():
        assert line in detail, f"详情缺批语行：{line}"


# ---------------------------------------------------------------------------
# ③ 确认即关页回列表
# ---------------------------------------------------------------------------
def test_submit_new_taskbook_returns_to_list(face):
    """投递成功 → 关表单回列表（防重复提交；用户点名）。"""
    _write_plan()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert app.view == "tb-new"
            app.query_one("#tb-new-title-input").value = "新任务书"
            await pilot.pause()
            await app._submit_new_taskbook()
            await pilot.pause()
            return app.view, app.last_created

    view, created = _run(_go())
    assert created
    assert view == "taskbooks"  # 回列表


def test_submit_reject_returns_to_list(face):
    _write_plan()
    _mk("驳件", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            app.query_one("#tb-note", TextArea).text = "批语"
            await pilot.pause()
            await app._submit_reject()
            await pilot.pause()
            return (
                app.view,
                app.query_one("#tb-reject", tui.Vertical).display,
            )

    view, panel_shown = _run(_go())
    assert view == "taskbooks" and panel_shown is False


# ---------------------------------------------------------------------------
# ④ ESC 再进无残留
# ---------------------------------------------------------------------------
def test_reopen_after_esc_has_no_stale_selection(face):
    """criteria：ESC 再进无选中残留——改进项/优先级回 [taskbooks] 默认、批语清空。"""
    _write_plan()
    _mk("残留件", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            # 留一个**非默认**勾选（只勾组长改进）证明重开被重置
            app._set_selections(
                "#tb-reject-improve",
                {"infra": False, "leader": True, "inspector": False},
            )
            app._set_radio("#tb-reject-priority", 0)
            app.query_one("#tb-note", TextArea).text = "残留批语"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            sel = app.query_one("#tb-reject-improve", SelectionList)
            prio = app.query_one("#tb-reject-priority", RadioSet)
            # 字形值感知（用户 2026-10-17「要对勾不要错叉」）：默认勾 infra/inspector
            marks = [
                "".join(seg.text for seg in sel.render_line(y))[:3] for y in range(3)
            ]
            return (
                set(sel.selected),
                [b.value for b in prio.query(RadioButton)],
                app.query_one("#tb-note", TextArea).text,
                marks,
            )

    sel_vals, prio_vals, note, marks = _run(_go())
    assert marks[0] == "▐√▌" and marks[2] == "▐√▌", f"选中须显对勾：{marks}"
    assert marks[1] == "▐ ▌", f"未选须空白（不是叉）：{marks}"
    from pysci.skills.orchestration.tools import settings as _settings

    form, _warns = _settings.taskbook_form_defaults()
    expected = {
        k
        for k, v in (
            ("infra", form.reject_infra),
            ("leader", form.reject_leader),
            ("inspector", form.reject_inspector),
        )
        if v
    }
    assert sel_vals == expected, f"改进项残留：{sel_vals}（应回默认 {expected}）"
    assert prio_vals == [False, True, False], (
        f"优先级残留（应回默认「中」）：{prio_vals}"
    )
    assert note == "", "批语未清空"
