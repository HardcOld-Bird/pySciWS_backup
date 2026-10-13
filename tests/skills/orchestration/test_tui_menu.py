"""`pysci` 主菜单框架 + 更新水位 + 模块归并（REFACTOR.md §2.2/§2.4/§2.5，M2）。

Pilot 无头驱动交互；水位落盘面用**临时 seen 文件**、模块读数用**测试替身**（不碰真实 state）。
覆盖验收：菜单导航/展开/详情进出/进详情写水位/徽标随水位翻转；模块归并（7 模块、宪章归盲审、
决策全貌迁入）；口述动作分流；非 tty 通道不回归。
"""

from __future__ import annotations

import asyncio
import json

from textual.widgets import Input, ListView, SelectionList

from pysci.skills.orchestration.tools import mirror
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui import watermark
from pysci.skills.orchestration.tools.tui.modules import Module, modules


def _run(coro):
    return asyncio.run(coro)


def _mod(key: str, *, title: str = "", detail=None, count: int = 0) -> Module:
    """测试替身模块：固定读数、可控计数（隔离真实 state）。"""
    lines = detail or [f"{key} 读数行"]
    return Module(
        key=key,
        title=title or key,
        summary=lambda: list(lines),
        detail=lambda: list(lines),
        cursor=lambda: str(count),
        updates_since=lambda old: max(0, count - int(old or 0)),
    )


# ---------------------------------------------------------------------------
# 模块归并（§2.4）
# ---------------------------------------------------------------------------
def test_seven_modules_in_clause_order_with_titles():
    mods = modules()
    assert [m.key for m in mods] == [
        "taskbooks",
        "plans",
        "queues",
        "review",
        "system",
        "decisions",
        "pending",
    ]
    assert [m.title for m in mods] == [
        "任务书",
        "计划与派发",
        "队列与运行",
        "盲审与审查",
        "系统与路由",
        "决策审计",
        "待决与验收",
    ]


def test_charter_reading_belongs_to_review_module():
    """组长钉死：宪章读数归「盲审与审查」（不再择一）。"""
    review = next(m for m in modules() if m.key == "review")
    assert mirror._line_charter() in review.summary()


def test_decisions_detail_reuses_mirror_data_layer():
    """现 cmd_decisions 全貌迁入详情——**共用** mirror 纯函数，不复制代码（§2.5）。"""
    dec = next(m for m in modules() if m.key == "decisions")
    assert dec.detail() == mirror.decisions_detail_lines()


def test_placeholder_modules_are_labelled():
    """占位如实标注；任务书自 m3 起是**真身**（不再是占位）。"""
    plans = next(m for m in modules() if m.key == "plans")
    assert any("占位" in ln for ln in plans.detail())
    tb = next(m for m in modules() if m.key == "taskbooks")
    assert any("现有大任务" in ln for ln in tb.detail())


def test_badge_count_is_explainable_and_first_run_counts_all():
    """徽标数字 = 相对旧水位的**新增条数**；首次未看过（空水位）→ 全量（可解释）。"""
    dec = next(m for m in modules() if m.key == "decisions")
    cur = int(dec.cursor())
    assert dec.updates_since("") == cur  # 首次 = 全量
    assert dec.updates_since(str(cur)) == 0  # 看过 = 无更新


# ---------------------------------------------------------------------------
# 水位文件（§2.2）
# ---------------------------------------------------------------------------
def test_seen_load_fail_open(tmp_path):
    p = tmp_path / "seen.json"
    assert watermark.load_seen(p) == {}  # 缺失
    p.write_text("{坏 json", encoding="utf-8")
    assert watermark.load_seen(p) == {}  # 坏 JSON
    p.write_text('{"version": 1, "modules": []}', encoding="utf-8")
    assert watermark.load_seen(p) == {}  # 形状不符


def test_seen_save_is_stable_and_roundtrips(tmp_path):
    p = tmp_path / "seen.json"
    watermark.save_seen({"b": "2", "a": "1"}, p)
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n") and json.loads(text)["modules"] == {"a": "1", "b": "2"}
    assert text.index('"a"') < text.index('"b"')  # 键排序 → 稳定
    assert watermark.load_seen(p) == {"a": "1", "b": "2"}


def test_enter_detail_writes_watermark_and_badge_flips(tmp_path):
    """进详情 → 落盘水位；徽标随之翻灰（§2.2 的核心闭环）。"""
    seen = tmp_path / "seen.json"
    mods = [_mod("queues", title="队列与运行", count=3)]

    async def _go():
        app = tui.PysciTUI(mods=mods, seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "3" in app.badge_texts["queues"]  # 首跑 = 全量 3 条
            await pilot.press("enter")  # 进详情
            assert app.view == "detail"
            await pilot.press("escape")  # 回菜单（徽标已刷新）
            assert app.view == "menu"
            assert app.badge_texts["queues"] == "·"

    _run(_go())
    assert watermark.load_seen(seen) == {"queues": "3"}  # 落盘且格式稳定


# ---------------------------------------------------------------------------
# 口述动作（§2.4：journal 编号分流菜单迁入 TUI）
# ---------------------------------------------------------------------------
class _RouteSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[int]]] = []

    def __call__(self, text, selects):
        self.calls.append((text, list(selects)))
        return {
            "text": text,
            "preference": {} if mirror.CHOICE_PREFERENCE in selects else None,
            "directive": {} if mirror.CHOICE_DIRECTIVE in selects else None,
        }


def test_j_opens_journal_and_submit_routes_default_preference(monkeypatch):
    spy = _RouteSpy()
    monkeypatch.setattr(mirror, "route", spy)

    async def _go():
        app = tui.PysciTUI(mods=[_mod("pending")])
        async with app.run_test() as pilot:
            await pilot.press("j")
            assert app.view == "journal"
            app.query_one("#journal-text", Input).value = "以后少用 emoji"
            await pilot.press("enter")
            assert "已记录" in app.journal_result

    _run(_go())
    assert spy.calls == [
        ("以后少用 emoji", [mirror.CHOICE_PREFERENCE])
    ]  # 缺省 ①（同 CLI）


def test_journal_selection_routes_both(monkeypatch):
    spy = _RouteSpy()
    monkeypatch.setattr(mirror, "route", spy)

    async def _go():
        app = tui.PysciTUI(mods=[_mod("pending")])
        async with app.run_test() as pilot:
            await pilot.press("j")
            select = app.query_one("#journal-select", SelectionList)
            select.select(mirror.CHOICE_DIRECTIVE)
            app.query_one("#journal-text", Input).value = "把 X 派给 devops"
            await pilot.press("enter")

    _run(_go())
    assert spy.calls == [("把 X 派给 devops", [mirror.CHOICE_DIRECTIVE])]


def test_journal_empty_text_warns_and_does_not_route(monkeypatch):
    spy = _RouteSpy()
    monkeypatch.setattr(mirror, "route", spy)

    async def _go():
        app = tui.PysciTUI(mods=[_mod("pending")])
        async with app.run_test() as pilot:
            await pilot.press("j")
            await pilot.press("enter")  # 空文本
            assert "缺少口述内容" in app.journal_result

    _run(_go())
    assert spy.calls == []


# ---------------------------------------------------------------------------
# 通道不回归（§2.1）
# ---------------------------------------------------------------------------
def test_non_tty_still_equals_mirror_status(monkeypatch, capsys):
    monkeypatch.setattr(tui, "_is_tty", lambda: False)
    assert tui.main([]) == 0
    out = capsys.readouterr().out
    mirror.main(["status"])
    assert out == capsys.readouterr().out


def test_menu_renders_seven_rows():
    async def _go():
        app = tui.PysciTUI()  # 真注册表
        async with app.run_test() as pilot:
            assert len(app.query_one("#menu", ListView).children) == 7

    _run(_go())
