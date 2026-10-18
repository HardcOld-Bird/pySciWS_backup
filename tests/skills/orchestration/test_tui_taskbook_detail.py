"""TUI 大报告**完成状态与详情页**（用户改进点 2026-10-15）。

用户原话：「已经存在的大报告**不显示是否完成**，也**没有详情页**（完成后的交付报告）……
完成后我看不到交付报告，我该参考什么批复呢？」

本文件钉四件事（对应任务书 success_criteria）：
① 列表每行含 **status 徽标**（七态字形可辨）；
② Enter 进**详情页**：任务书正文 + **流程报告**（所链 plan 的环节清单：状态/成员/耗时 +
   交付正文留档路径与**页内摘要**）；
③ delivered 态「批复参考」置顶（用户批复前必看）；
④ 返工卡片不回归（并与流程报告合并展示）。
另钉「单一报告实现」：详情页的流程报告 = :func:`..taskbooks.report_sections` 的输出，
且进详情**不落盘**（``<id>.report.md`` 不被读操作写出）。
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from textual.widgets import ListView

from pysci import paths as _paths
from pysci.skills.orchestration.tools import ledger, plans, queueworker, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui

PLAN_ID = "p-demo"
S1_JOB = f"lit-20261015-120000-{PLAN_ID}-s1"
S2_JOB = f"writing-20261015-121000-{PLAN_ID}-s2"

#: 交付留档正文里的哨兵行——详情页必须**在页内**看得见（不必翻文件）。
DELIVERY_MARK = "交付正文哨兵行：环节 s1 已交付"
S2_MARK = "交付正文哨兵行：环节 s2 已完成"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离四个读数面：任务书目录 / 计划目录 / 交付留档目录 / 台账。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    monkeypatch.setattr(queueworker, "DELIVERIES_DIR", tmp_path / "deliveries")
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def _write_plan(*, completed: bool = False) -> None:
    """落一份两环节计划：s1 done（有 job/交付留档/耗时），s2 queued。"""
    plans.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    (plans.PLANS_DIR / f"{PLAN_ID}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": PLAN_ID,
                "goal": "演示大任务",
                "steps": [
                    {"id": "s1", "member": "lit", "brief": "阅读材料并写总结"},
                    {"id": "s2", "member": "writing", "brief": "润色成稿"},
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (plans.PLANS_DIR / f"{PLAN_ID}.state.json").write_text(
        json.dumps(
            {
                "plan_id": PLAN_ID,
                "goal": "演示大任务",
                "steps": [
                    {"id": "s1", "status": "done", "job_id": S1_JOB, "kind": "result"},
                    {"id": "s2", "status": "queued", "job_id": S2_JOB, "kind": None},
                ],
                "completed_at": "2026-10-15T12:30:00+08:00" if completed else None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_delivery(
    job_id: str, mark: str, *, errors: bool = False, ms: int = 0
) -> None:
    """按 ``queueworker.archive_delivery`` 的落盘形状写一份交付全文留档。"""
    d = queueworker.DELIVERIES_DIR
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 交付全文留档：{job_id}",
        "",
        "- member: lit",
        "- kind: result",
        f"- duration_ms: {ms}",
        "",
        "## 交付正文",
        "",
        mark,
        "第二行内容",
    ]
    if errors:
        lines += ["", "## 错误正文", "", "不该出现在页内摘要里"]
    (d / f"{job_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_flow_steps_tolerates_legacy_polluted_plan_rows(face):
    """历史台账行 ``plan`` 是席表对象（2026-10-13 遮蔽窗口，约 107 行）→ 读数**不得崩**，
    只是那条耗时读不到（归属缺失，不冒充 0 之外的数）。"""
    _write_plan()
    ledger.append(
        ledger.LedgerEntry(
            ts="",
            member="lit",
            session_id="sid-legacy",
            kind="result",
            duration_ms=999_000,
            plan={"tier": "main", "usable": ["DeepSeek-Flash"]},
            step="s1",
        )
    )
    steps = taskbooks.flow_steps(PLAN_ID)
    assert [s["id"] for s in steps] == ["s1", "s2"]
    assert steps[0]["duration_ms"] == 0  # 归属读不到 → 显示「—」，不谎报


def test_flow_steps_falls_back_to_delivery_header_duration(face):
    """台账无归属（历史遮蔽窗口）→ 耗时回落**留档头** ``- duration_ms``（逐 job 写，与归属无关）。"""
    _write_plan()
    _write_delivery(S1_JOB, DELIVERY_MARK, ms=42_000)
    assert taskbooks.flow_steps(PLAN_ID)[0]["duration_ms"] == 42_000


def test_flow_steps_prefers_ledger_duration(face):
    """台账有归属则用台账（可累积重派多跳），留档头只兜底。"""
    _write_plan()
    _write_delivery(S1_JOB, DELIVERY_MARK, ms=42_000)
    _write_ledger(ms=125_000)
    assert taskbooks.flow_steps(PLAN_ID)[0]["duration_ms"] == 125_000


def _write_ledger(*, plan: str = PLAN_ID, step: str = "s1", ms: int = 125_000) -> None:
    ledger.append(
        ledger.LedgerEntry(
            ts="",
            member="lit",
            session_id="sid-x",
            kind="result",
            duration_ms=ms,
            plan=plan,
            step=step,
        )
    )


def _mk(title: str, status: str, *, plan: str = "") -> str:
    """造一件处于指定状态的任务书（走合法迁移，不直接改 state.json）。

    ``plan`` 非空则先 ``link_plan``（new → planning）——**必须在推进状态之前**：状态机里
    ``delivered → planning`` 非法。
    """
    tid = taskbooks.create(title, f"{title} 的正文")["id"]
    if status == "new" and not plan:
        return tid
    taskbooks.link_plan(tid, plan or PLAN_ID)
    if status == "planning":
        return tid
    taskbooks.transition(tid, "active")
    if status == "active":
        return tid
    if status == "failed":
        taskbooks.transition(tid, "failed")
        return tid
    taskbooks.transition(tid, "delivered")
    if status == "delivered":
        return tid
    if status == "accepted":
        taskbooks.accept(tid)
        return tid
    if status == "rejected":
        taskbooks.transition(tid, "rejected")
        return tid
    raise AssertionError(status)


async def _select_row(app, tb_id: str) -> int:
    """按**行模型**选中该任务书所在行（s5：列表分组，测试禁裸下标）。"""
    lv = app.query_one("#tb-list", ListView)
    index = next(i for i, r in enumerate(app.tb_rows) if r["tb_id"] == tb_id)
    lv.index = index
    return index


async def _open_panel(app, pilot, index: int = 0) -> None:
    await pilot.press("t")
    await pilot.pause()
    lv = app.query_one("#tb-list", ListView)
    lv.index = index
    await pilot.pause()


async def _enter_detail(app, pilot, index: int = 0) -> None:
    """进详情（s5：Enter 是展开/收起，`o` 才是进选中行目标详情）。"""
    await _open_panel(app, pilot, index)
    await pilot.press("o")
    await pilot.pause()
    assert app.view == "tb-detail", app.view


# ===========================================================================
# ① 列表每行含 status 徽标
# ===========================================================================
def test_list_rows_show_status_badge_for_all_seven_states(face):
    """success_criteria：任务书列表每行含 status 徽标（七态可辨）。"""
    states = (
        "new",
        "planning",
        "active",
        "delivered",
        "accepted",
        "rejected",
        "failed",
    )
    for s in states:
        _mk(f"任务书-{s}", s)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _open_panel(app, pilot)
            return [(r["badge"], r["title"]) for r in app.tb_rows]

    rows = _run(_go())
    assert len(rows) == len(states)
    glyphs = set()
    for status in states:
        badge = taskbooks.status_badge(status)
        hit = [t for b, t in rows if t.startswith(f"任务书-{status}")]
        assert len(hit) == 1, f"{status} 的行不唯一/缺失：{rows}"
        assert any(b == badge and t.startswith(f"任务书-{status}") for b, t in rows), (
            f"{status} 行缺徽标：{rows}"
        )
        glyphs.add(badge.split(" ")[0])
    # 七态字形两两可辨（不是同一个符号）
    assert len(glyphs) == len(states), glyphs


def test_status_badge_unknown_state_is_echoed_not_swallowed():
    assert taskbooks.status_badge("delivered") == "◆ delivered"
    assert taskbooks.status_badge("weird") == "weird"  # 未知态原样回显（不吞）


# ===========================================================================
# ② Enter 详情页：正文 + 流程报告
# ===========================================================================
def test_enter_opens_detail_with_body_and_flow_report(face):
    """success_criteria：Enter 详情页含任务书正文 + 流程报告（环节清单 + 留档摘要/引用）。"""
    _write_plan()
    _write_delivery(S1_JOB, DELIVERY_MARK)
    _write_delivery(S2_JOB, S2_MARK)
    _write_ledger()
    tid = taskbooks.create("演示大任务", "任务书正文甲乙丙")["id"]
    taskbooks.link_plan(tid, PLAN_ID)
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            return app.tb_detail_id, app.tb_detail_body, app.tb_detail_title

    got_id, body, title = _run(_go())
    assert got_id == tid and tid in title
    assert "任务书正文甲乙丙" in body, "详情页须含任务书正文"
    # 流程报告：环节清单逐项
    assert "环节完成清单" in body
    assert "s1" in body and "[done]" in body and "lit" in body
    assert "s2" in body and "[queued]" in body and "writing" in body
    assert "耗时 125s" in body, "环节耗时须来自台账（plan+step 归属）"
    # 交付正文留档：路径引用 + 页内摘要（用户「不必翻文件」）
    assert f"{S1_JOB}.md" in body, "须给出留档路径引用"
    assert DELIVERY_MARK in body, "交付正文须有**页内摘要**"
    assert "不该出现在页内摘要里" not in body, "「错误正文」段不混进交付摘要"


def test_detail_shows_rejection_history(face):
    tid = taskbooks.create("被驳回过的大任务", "正文")["id"]
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")
    taskbooks.create_rework(tid, improve={}, note="拆得不对，重来")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            return app.tb_detail_body

    body = _run(_go())
    assert "批语历史" in body and "拆得不对，重来" in body


def test_any_status_can_enter_detail(face):
    """任何状态的任务书都能进详情页（new 也要能看）。"""
    tid = _mk("新任务书", "new")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            return app.tb_detail_body

    body = _run(_go())
    assert tid in body and "新任务书 的正文" in body


# ===========================================================================
# ③ delivered 态「批复参考」突出
# ===========================================================================
def test_delivered_puts_flow_report_first_as_review_reference(face):
    """success_criteria：delivered 态批复参考突出展示（流程报告置顶）。"""
    _write_plan()
    _write_delivery(S1_JOB, DELIVERY_MARK)
    _write_ledger()
    tid = _mk("待批复大任务", "delivered", plan=PLAN_ID)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            return app.tb_detail_body

    body = _run(_go())
    assert "★ 批复参考" in body
    assert body.index("★ 批复参考") < body.index("## 任务书正文"), "批复参考须置顶"
    assert body.index("★ 批复参考") < body.index("环节完成清单")


def test_non_delivered_keeps_report_after_body(face):
    """非 delivered：先正文后流程报告（不冒充「批复参考」）。"""
    _write_plan()
    tid = _mk("在做大任务", "active", plan=PLAN_ID)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            return app.tb_detail_body

    body = _run(_go())
    assert "★ 批复参考" not in body
    assert body.index("## 任务书正文") < body.index("环节完成清单")


# ===========================================================================
# ④ 返工卡片不回归
# ===========================================================================
def test_rework_card_still_shown_and_merged(face):
    """success_criteria：返工卡片内容不回归（与流程报告合并展示）。"""
    _write_plan()
    tb1 = _mk("原大任务", "delivered")
    tb2 = taskbooks.create_rework(tb1, mode="no_inspector", note="初轮驳回")["id"]
    taskbooks.link_plan(tb2, PLAN_ID)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _open_panel(app, pilot)
            lv = app.query_one("#tb-list", ListView)
            lv.index = next(i for i, r in enumerate(app.tb_rows) if r["tb_id"] == tb2)
            await pilot.pause()
            await pilot.press("o")  # s5：Enter 是展开/收起，`o` 才是进详情
            await pilot.pause()
            return app.tb_detail_body

    body = _run(_go())
    assert "返工卡片" in body and "第 1 轮返工" in body
    assert "no_inspector" in body and "superstep 进度树" in body
    assert "环节完成清单" in body, "返工任务书同样有流程报告（合并展示）"


# ===========================================================================
# 单一报告实现（复用 build_report 的组装器，非第二套）
# ===========================================================================
def test_detail_report_is_the_shared_report_assembler(face):
    _write_plan()
    _write_delivery(S1_JOB, DELIVERY_MARK)
    _write_ledger()
    tid = _mk("单实现大任务", "delivered", plan=PLAN_ID)

    detail = "\n".join(taskbooks.detail_lines(tid))
    shared = "\n".join(taskbooks.report_sections(tid))
    assert shared.strip() and shared in detail, "详情页的流程报告必须就是组装器输出"
    # build_report = 同一组装器 + 落盘
    assert taskbooks.build_report(tid).strip() == shared.strip()


def test_plain_taskbook_report_omits_rework_only_sections(face):
    _write_plan()
    tid = _mk("普通大任务", "delivered", plan=PLAN_ID)
    text = "\n".join(taskbooks.report_sections(tid))
    assert "## 任务概述" in text and "## 环节完成清单" in text
    for sec in ("历轮摘要", "本轮成因分析", "解决方案", "监察判断", "返工产物报告"):
        assert f"## {sec}" not in text, f"普通任务书不该有返工专属段：{sec}"


def test_opening_detail_does_not_write_report_file(face):
    """详情页只读：看一眼不该写出 ``<id>.report.md``。"""
    tid = _mk("只读大任务", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)

    _run(_go())
    assert not taskbooks.report_path(tid).exists()
    assert "report" not in (taskbooks.load_state(tid) or {})


# ===========================================================================
# 交互：批复就地可用 + 导航
# ===========================================================================
def test_accept_from_detail_view(face):
    """详情页可批复（读完流程报告就地 accept）。"""
    tid = _mk("就地批复大任务", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            await pilot.press("a")
            await pilot.pause()
            return app.view, app.tb_msg

    view, msg = _run(_go())
    assert taskbooks.load_state(tid)["status"] == "accepted"
    assert "accepted" in msg
    assert view == "taskbooks", "批复后回列表（重建）"


def test_escape_from_detail_returns_to_list(face):
    _mk("导航大任务", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            return app.view

    assert _run(_go()) == "taskbooks"


def test_detail_view_footer_matches_guard(face):
    """详情页 footer 广告 ↔ check_action 一致；且 `n`（新增）在详情页不可用。"""
    _mk("键位大任务", "delivered")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await _enter_detail(app, pilot)
            keys = set(app.screen.active_bindings)
            return app, keys

    app, keys = _run(_go())
    assert app.check_action("tb_new", ()) is False, "详情页不广告/不响应 `n`"
    assert "n" not in keys
    for k in ("a", "r"):
        assert k in keys, f"详情页应广告 `{k}`"
    for k in ("t", "j"):
        assert k not in keys, f"s4 键位收敛：详情页不再广告 `{k}`（t/j 只在主菜单）"
