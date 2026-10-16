"""TUI 改版 s4（task tui-overhaul-20261016-s4-app-frame）：app.py 通用面。

钉五件事：

1. **footer 键位收敛**：`t`/`j` 只在主菜单广告；detail/taskbooks/tb-detail 视图二者不出现
   （`check_action` 与 `screen.active_bindings` 双面）；
2. **滚动接线**：`pane.scroll_y = N`（滚轮/滚动条同一条 reactive 通路）→ 位置指示**立即**
   更新；任务书列表底部 `#tb-scroll` 在场并随列表滚动变化；
3. **提示退役**：读面（`detail_text`/`tb_detail_hint`）不含「Esc/← 返回」「批复入口」
   「口述动作」；`_split_detail`/`_DETAIL_HINT_MARKERS` 已从源码消失；
4. **概览板**：`#summary` 首行＝「〔模块名〕概览」标题行；
5. **水位写盘**：`_mark_seen` 的 cursor 未变则不落盘（每秒 tick 不再写 seen 文件）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from textual.containers import Vertical
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui.modules import Module


def _run(coro):
    return asyncio.run(coro)


def _long_module(
    key: str = "fake", *, title: str = "长读数模块", rows: int = 300
) -> Module:
    lines = [f"读数行 {i}" for i in range(rows)]
    return Module(
        key=key,
        title=title,
        summary=lambda: list(lines[:3]),
        detail=lambda: list(lines),
        cursor=lambda: "0",
        updates_since=lambda old: 0,
    )


@pytest.fixture
def seen(tmp_path):
    return tmp_path / "seen.json"


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d = data / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return d


def _write_taskbook(tb_id: str, *, title: str = "批量任务书", rows: int = 1) -> None:
    d = taskbooks.TASKBOOKS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tb_id}.taskbook.md").write_text(
        "---\n"
        f"id: {tb_id}\nts: 2026-10-16T00:00:00+08:00\ntitle: {title}\n"
        "status: new\nsource: mirror-tui\n---\n\n"
        + "\n".join(f"正文第 {i} 行" for i in range(rows)),
        encoding="utf-8",
    )
    (d / f"{tb_id}.state.json").write_text(
        json.dumps(
            {
                "id": tb_id,
                "ts": "2026-10-16T00:00:00+08:00",
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


# ---------------------------------------------------------------------------
# ① footer 键位收敛（t/j 只在主菜单）
# ---------------------------------------------------------------------------
def test_menu_only_advertises_t_and_j():
    app = tui.PysciTUI()
    assert tui._ACTIONS_BY_VIEW["menu"] == frozenset({"taskbooks", "journal"})
    for view in ("detail", "taskbooks", "tb-detail", "journal", "tb-new", "tb-reject"):
        app.view = view
        for action in ("taskbooks", "journal"):
            assert app.check_action(action, ()) is False, (view, action)
    # 面板仍广告自己的键（taskbooks: n/a/r；tb-detail: a/r）
    app.view = "taskbooks"
    for action in ("tb_new", "tb_accept", "tb_reject"):
        assert app.check_action(action, ()) is True, action
    app.view = "tb-detail"
    for action in ("tb_accept", "tb_reject"):
        assert app.check_action(action, ()) is True, action
    assert app.check_action("tb_new", ()) is False


def test_t_and_j_not_advertised_outside_menu():
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            keys = set(app.screen.active_bindings)
            assert "t" in keys and "j" in keys, "主菜单应广告 t/j"
            lv = app.query_one("#menu", ListView)
            lv.index = 2  # 队列与运行（plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            keys = set(app.screen.active_bindings)
            assert "t" not in keys and "j" not in keys, "详情视图不得广告 t/j"
            await pilot.press("escape")
            await pilot.press("t")
            assert app.view == "taskbooks"
            keys = set(app.screen.active_bindings)
            assert "t" not in keys and "j" not in keys, "任务书面板不得广告 t/j"
            for k in ("n", "a", "r"):
                assert k in keys, f"任务书面板应广告 `{k}`"

    _run(_go())


# ---------------------------------------------------------------------------
# ② 滚动位置指示接线（滚轮/滚动条导航实时更新）
# ---------------------------------------------------------------------------
def test_scroll_y_assignment_updates_mark_immediately(seen):
    """`scroll_y = N` 是滚轮/滚动条拖动的同一条通路 → 位置指示立即更新（s4 修复的实测缺陷）。"""

    async def _go():
        app = tui.PysciTUI(mods=[_long_module()], seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.view == "detail"
            pane = app.query_one("#detail-body")
            assert pane.max_scroll_y > 0
            before = app.detail_scroll_mark
            pane.scroll_y = 7  # 模拟滚轮/滚动条（未经键盘路径）
            await pilot.pause()
            assert app.detail_scroll_mark != before, "scroll_y 变化必须立即刷新位置指示"
            assert f"（第 7/{int(pane.max_scroll_y)} 行）" in app.detail_scroll_mark

    _run(_go())


def test_tb_list_scroll_mark_present_and_tracks_scroll(tbdir, seen):
    for i in range(40):
        _write_taskbook(f"tb-{i:03d}", title=f"批量任务书 {i}")

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            assert app.view == "taskbooks"
            assert app.query("#tb-scroll"), "任务书列表底部必须有位置行"
            lv = app.query_one("#tb-list", ListView)
            assert lv.max_scroll_y > 0
            assert app.tb_scroll_mark.startswith(tui._SCROLL_MARK_HEAD)
            lv.scroll_y = 3  # 模拟滚轮/滚动条（ListView 只在 index 变化时自行滚动）
            await pilot.pause()
            assert f"（第 3/{int(lv.max_scroll_y)} 行）" in app.tb_scroll_mark
            await pilot.press("pagedown")  # 键盘路径（App 绑定，App 闸在此视图放行）
            await pilot.pause()
            assert (
                f"（第 {int(lv.scroll_y)}/{int(lv.max_scroll_y)} 行）"
                in app.tb_scroll_mark
            )

    _run(_go())


def test_scroll_mark_ignores_hidden_pane():
    """隐藏面板的光标跟随滚动不得串台（只认当前视图的面板）。"""
    app = tui.PysciTUI(mods=[_long_module()])
    app.view = "menu"
    app.tb_scroll_mark = "未动"
    app.on_pane_scrolled(object())  # 非当前视图面板（且当前视图无可滚面板）
    assert app.tb_scroll_mark == "未动"


# ---------------------------------------------------------------------------
# ③ 提示机制退役
# ---------------------------------------------------------------------------
def test_hint_mechanism_retired_from_source():
    src = Path(tui.__file__).read_text(encoding="utf-8")
    for gone in ("_DETAIL_HINT_MARKERS", "_split_detail"):
        assert gone not in src, f"{gone} 应已从 app.py 源码消失"


def test_read_faces_have_no_prompt_words(tbdir, seen):
    _write_taskbook("tb-hint-1", title="提示退役用例")

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("o")  # s5：Enter 是展开/收起，`o` 才是进详情
            await pilot.pause()
            assert app.view == "tb-detail"
            assert app.tb_detail_hint == "a 接受　r 驳回"
            joined = (
                f"{app.tb_detail_title}\n{app.tb_detail_body}\n{app.tb_detail_hint}"
            )
            for gone in ("Esc/← 返回", "批复入口", "口述动作"):
                assert gone not in joined, gone
            # 模块详情读面同样干净
            await pilot.press("escape")  # → taskbooks
            await pilot.press("escape")  # → menu
            lv = app.query_one("#menu", ListView)
            lv.index = 2  # 队列与运行（plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            for gone in ("Esc/← 返回", "批复入口", "口述动作"):
                assert gone not in app.detail_text, gone

    _run(_go())


# ---------------------------------------------------------------------------
# ④ 概览板（标题行 + 主题化）
# ---------------------------------------------------------------------------
def test_summary_board_title_line_follows_selection(seen):
    async def _go():
        app = tui.PysciTUI(
            mods=[_long_module("a", title="甲模块"), _long_module("b", title="乙模块")],
            seen_path=seen,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#summary", Vertical)
            assert list(box.children)[0].id == "summary-title", "标题行须为概览板首行"
            assert (
                str(app.query_one("#summary-title", Static).render())
                == "〔甲模块〕概览"
            )
            body = str(app.query_one("#summary-body", Static).render())
            assert "读数行 0" in body
            app.query_one("#menu", ListView).index = 1
            await pilot.pause()
            assert (
                str(app.query_one("#summary-title", Static).render())
                == "〔乙模块〕概览"
            )

    _run(_go())


def test_summary_board_title_uses_real_module_title():
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert (
                str(app.query_one("#summary-title", Static).render())
                == "〔任务书〕概览"
            )

    _run(_go())


# ---------------------------------------------------------------------------
# ⑤ 水位写盘（同 cursor 不写）
# ---------------------------------------------------------------------------
def test_mark_seen_skips_write_when_cursor_unchanged(seen, monkeypatch):
    calls: list[int] = []
    real = tui.watermark.save_seen

    def spy(data, path=None):
        calls.append(1)
        return real(data, path)

    monkeypatch.setattr(tui.watermark, "save_seen", spy)

    async def _go():
        app = tui.PysciTUI(mods=[_long_module()], seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._mark_seen(app._mods[0])
            assert len(calls) == 1, "首次进入应落盘水位"
            app._mark_seen(app._mods[0])  # 同 cursor（每秒 tick 的形态）
            assert len(calls) == 1, "cursor 未变不得重写 seen 文件"

    _run(_go())
