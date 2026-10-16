"""详情页**滚动**（用户 2026-10-15 实测反馈 1+2）。

用户原话：「① 详情页无法滚动（最大化窗口仍看不全）；② 大任务列表疑同样无滚动
（满页即不可用，请查证）」。

两条判据：
① 两个详情视图（模块详情 `detail` / 任务书详情 `tb-detail`）正文**可滚到底**，且有
   **位置指示**（`顶` / `n%` / `底`）——修复前是 ``Static`` + ``height: 1fr``，内容被
   静默裁掉（实测 ``max_scroll_y == 0``：超出的行既不动也读不到）；
② 大任务列表**本来就能滚**（用户担忧证伪）：以 ``max_scroll_y`` 与跟随光标的
   ``scroll_y`` 实测读数为证——本文件把该读数钉住，防日后回归成不可滚动。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from textual.containers import VerticalScroll
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui.modules import Module


def _run(coro):
    return asyncio.run(coro)


def _long_module(key: str = "fake", rows: int = 300) -> Module:
    lines = [f"读数行 {i}" for i in range(rows)]
    return Module(
        key=key,
        title="长读数模块",
        summary=lambda: list(lines[:3]),
        detail=lambda: list(lines),
        cursor=lambda: "0",
        updates_since=lambda old: 0,
    )


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    """任务书详情页读的是**真实 state** → 隔离，免随开发者机器上的任务书漂流。"""
    data = tmp_path / "data"
    d = data / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return d


@pytest.fixture
def seen(tmp_path):
    """水位落点隔离（不隔离则 `_mark_seen` 去写真实 ``state/mirror-seen.json``）。"""
    return tmp_path / "seen.json"


def _write_taskbook(tb_id: str, *, title: str = "长正文任务书", rows: int = 400) -> str:
    """手写一件「正文很长」的任务书（只造读数面，不经 create——免触发落盘即提交）。"""
    d = taskbooks.TASKBOOKS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tb_id}.taskbook.md").write_text(
        "---\n"
        f"id: {tb_id}\nts: 2026-10-15T00:00:00+08:00\ntitle: {title}\n"
        "status: new\nsource: mirror-tui\n---\n\n"
        + "\n".join(f"正文第 {i} 行" for i in range(rows)),
        encoding="utf-8",
    )
    (d / f"{tb_id}.state.json").write_text(
        json.dumps(
            {
                "id": tb_id,
                "ts": "2026-10-15T00:00:00+08:00",
                "title": title,
                "status": "new",
                "source": "mirror-tui",
                "priority": "normal",
                "mode": "",
                "rework_of": "",
                "rework_round": 0,
                "plan_id": "",
                "rejections": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tb_id


async def _down_to_bottom(pilot, app, pane_sel: str, key: str = "pagedown") -> None:
    """一路翻页到底（防死循环：每轮都读一次 scroll_y，卡住即断言失败）。"""
    pane = app.query_one(pane_sel, VerticalScroll)
    for _ in range(200):
        if pane.scroll_y >= pane.max_scroll_y:
            return
        await pilot.press(key)
        await pilot.pause()
    raise AssertionError("翻页 200 次仍未到底（滚动不生效？）")


# ===========================================================================
# ① 模块详情页可滚动（用户反馈 1）
# ===========================================================================
def test_module_detail_scrolls_to_bottom_with_position_mark(seen):
    """success_criteria：详情页可滚动到底（含位置指示）。"""
    mods = [_long_module()]

    async def _go():
        app = tui.PysciTUI(mods=mods, seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.view == "detail"
            pane = app.query_one("#detail-body", VerticalScroll)
            inner = app.query_one("#detail-body-text", Static)
            # 内容确实超出可视区（修复前 max_scroll_y 恒 0：内容被静默裁掉）
            assert pane.is_scrollable and pane.max_scroll_y > 0
            assert inner.virtual_size.height > pane.size.height
            assert app.detail_scroll_mark.endswith(f"顶（第 0/{pane.max_scroll_y} 行）")
            await pilot.press("pagedown")
            await pilot.pause()
            assert pane.scroll_y > 0 and "%" in app.detail_scroll_mark
            await _down_to_bottom(pilot, app, "#detail-body")
            assert pane.scroll_y == pane.max_scroll_y
            assert app.detail_scroll_mark.endswith(
                f"底（第 {pane.max_scroll_y}/{pane.max_scroll_y} 行）"
            )

    _run(_go())


def test_detail_scroll_keys_home_end_and_single_line(seen):
    """↑↓ 逐行、Home/End（g/G）到顶到底——键位与 TUI 既有单字母风格同族。"""
    mods = [_long_module()]

    async def _go():
        app = tui.PysciTUI(mods=mods, seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            pane = app.query_one("#detail-body", VerticalScroll)
            await pilot.press("down")
            await pilot.press("down")
            await pilot.pause()
            assert pane.scroll_y == 2, "↓ 应逐行下移"
            await pilot.press("up")
            await pilot.pause()
            assert pane.scroll_y == 1
            await pilot.press("end")
            await pilot.pause()
            assert pane.scroll_y == pane.max_scroll_y, "End 到底"
            await pilot.press("home")
            await pilot.pause()
            assert pane.scroll_y == 0, "Home 到顶"
            await pilot.press("G")
            await pilot.pause()
            assert pane.scroll_y == pane.max_scroll_y, "G 到底（vim 风格）"
            await pilot.press("g")
            await pilot.pause()
            assert pane.scroll_y == 0, "g 到顶"

    _run(_go())


def test_detail_scroll_keys_are_gated_to_detail_views():
    """滚动键只在 ``_SCROLL_PANES`` 所列视图生效（s4 收编 taskbooks）；菜单里按 ↑↓
    是**列表光标**语义，不被抢。"""
    app = tui.PysciTUI(mods=[_long_module()])
    for action in sorted(tui._SCROLL_ACTIONS):
        app.view = "menu"
        assert app.check_action(action, ()) is False, action
        for view in ("detail", "tb-detail", "taskbooks"):
            app.view = view
            assert app.check_action(action, ()) is True, (view, action)


def test_menu_updown_still_moves_list_cursor(seen):
    """菜单里 ↑↓ 仍是列表光标（滚动闸不得抢走导航）。"""

    async def _go():
        app = tui.PysciTUI(mods=[_long_module(), _long_module("other")], seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", ListView)
            lv.index = 0
            await pilot.press("down")
            await pilot.pause()
            assert lv.index == 1, "菜单 ↓ 应为列表下一个"

    _run(_go())


# ===========================================================================
# ② 任务书详情页可滚动（用户反馈 1 的另一半：交付报告最长）
# ===========================================================================
def test_tb_detail_scrolls_to_bottom(tbdir, seen):
    _write_taskbook("tb-long-1")

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.view == "tb-detail"
            pane = app.query_one("#tb-detail-body", VerticalScroll)
            assert pane.max_scroll_y > 0, "交付报告应超出可视区"
            assert app.tb_detail_scroll_mark.startswith(tui._SCROLL_MARK_HEAD)
            got_top = app.tb_detail_scroll_mark
            await _down_to_bottom(pilot, app, "#tb-detail-body")
            assert pane.scroll_y == pane.max_scroll_y
            assert "底" in app.tb_detail_scroll_mark
            # 退出再进 → 滚动复位到顶（新内容从头部读起）
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert pane.scroll_y == 0 and app.tb_detail_scroll == 0
            return got_top

    assert "顶" in _run(_go())


def test_tb_detail_short_content_says_whole_screen(tbdir, seen):
    """一屏装得下时位置指示如实说「整屏（无需滚动）」——不谎报百分比。"""
    _write_taskbook("tb-short-1", title="短任务书", rows=1)

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test(size=(80, 50)) as pilot:  # 高窗口：内容一屏装得下
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.view == "tb-detail"
            pane = app.query_one("#tb-detail-body", VerticalScroll)
            assert pane.max_scroll_y == 0
            return app.tb_detail_scroll_mark

    assert "整屏" in _run(_go())


# ===========================================================================
# ③ 大任务列表**本来就能滚**（用户担忧证伪，读数留档）
# ===========================================================================
def test_large_taskbook_list_is_already_scrollable(tbdir, seen):
    """实测读数证伪用户担忧 2：``#tb-list`` 是 ``ListView``（``is_scrollable``），
    满页时 ``max_scroll_y > 0``，且光标下移**自动跟随滚动**。"""
    for i in range(40):
        _write_taskbook(f"tb-many-{i:03d}", title=f"批量任务书 {i}", rows=1)

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            lv = app.query_one("#tb-list", ListView)
            assert len(lv.children) == 40
            assert lv.is_scrollable and lv.max_scroll_y > 0, "满页列表必须可滚"
            for _ in range(30):
                await pilot.press("down")
            await pilot.pause()
            return lv.index, lv.scroll_y, lv.max_scroll_y

    index, scroll_y, max_y = _run(_go())
    assert index == 30 and 0 < scroll_y <= max_y, (index, scroll_y, max_y)


def test_menu_list_is_a_listview_too(seen):
    """主菜单列表同为 ``ListView``（同一滚动基座）——模块装不下时同样可滚。"""

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            return app.query_one("#menu", ListView).is_scrollable

    assert _run(_go()) is True
