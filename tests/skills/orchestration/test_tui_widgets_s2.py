"""TUI 共享层 s2（task tui-overhaul-20261016-s2-widgets）：theme 唯一出处 + widgets 组件。

钉四件事：

1. theme 9 常量一字不改迁入（app.py 行为零变化的前提）；
2. ``ScrollMarkMixin``：``scroll_y`` 任一来源（含程序化赋值）→ 宿主 ``on_pane_scrolled``；
   宿主缺席不塌滚动（fail-soft）；
3. ``RowListView``：Enter → ``RowActivated(source="key", chain=1)`` 且**不发 Selected**；
   ``Pilot.click(times=2)`` → chain 1、2 两条 ``source="click"`` 激活；
4. ``ChoiceCard``：点击发 ``Chosen``（value 正确、不自改选中态）、``-selected`` 高亮
   开关正确；``card/section_title/hint/make_row`` 构造即过。
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Static

from pysci.skills.orchestration.tools.tui import theme, widgets


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ① theme 常量迁入完整
# ---------------------------------------------------------------------------
def test_theme_constants_migrated_intact():
    for name, expect in (
        ("_T_BORDER", "border" + ": round $accent;"),
        ("_T_BORDER_SOFT", "border" + ": round $warning;"),
        ("_T_INSET", "padding: 0 1;"),
        ("_T_TITLE", "text-style: bold; color" + ": $accent; background: $panel;"),
        ("_T_SUBTLE", "color" + ": $text-muted;"),
        ("_T_BODY", "color" + ": $text;"),
        ("_T_HINT", "color" + ": $warning; text-style: italic;"),
        ("_T_HL", "background: $accent; text-style: bold;"),
        ("_T_BG", "background: $panel;"),
    ):
        assert getattr(theme, name) == expect, name


# ---------------------------------------------------------------------------
# ② ScrollMarkMixin：scroll_y → on_pane_scrolled（含 fail-soft）
# ---------------------------------------------------------------------------
class _ScrollApp(App):
    """挂 ``on_pane_scrolled`` 的宿主：滚动知情通道的对端（s4 的接线形态）。"""

    def __init__(self) -> None:
        super().__init__()
        self.scrolled: list = []

    def on_pane_scrolled(self, pane) -> None:
        self.scrolled.append(pane)

    def compose(self) -> ComposeResult:
        yield widgets.MarkedVerticalScroll(
            Static("\n".join(f"行{i}" for i in range(300)), id="tall"), id="pane"
        )


def test_scroll_mark_mixin_fires_on_scroll_y():
    async def _go():
        app = _ScrollApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one("#pane", widgets.MarkedVerticalScroll)
            before = len(app.scrolled)
            pane.scroll_y = 5  # 程序化赋值＝与滚轮/滚动条同一条 reactive 通路
            await pilot.pause()
            assert len(app.scrolled) > before, "scroll_y 变化必须触发 on_pane_scrolled"
            assert app.scrolled[-1] is pane

    _run(_go())


def test_scroll_mark_mixin_failsoft_without_handler():
    class _Bare(App):
        def compose(self) -> ComposeResult:
            yield widgets.MarkedVerticalScroll(Static("x" * 200), id="pane")

    async def _go():
        app = _Bare()
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one("#pane", widgets.MarkedVerticalScroll)
            pane.scroll_y = 1  # 宿主无 on_pane_scrolled：不抛即过（fail-soft）
            await pilot.pause()

    _run(_go())


# ---------------------------------------------------------------------------
# ③ RowListView：Enter/单击/双击 → 同型 RowActivated，不发 Selected
# ---------------------------------------------------------------------------
class _RowApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[widgets.RowActivated] = []
        self.selected_seen: list = []

    def on_row_activated(self, event: widgets.RowActivated) -> None:
        self.events.append(event)

    def on_list_view_selected(self, event) -> None:
        self.selected_seen.append(event)

    def compose(self) -> ComposeResult:
        yield widgets.RowListView(
            widgets.ChainListItem(Static("第一行"), id="row-a"),
            widgets.ChainListItem(Static("第二行"), id="row-b"),
            widgets.ChainListItem(Static("第三行"), id="row-c"),
            id="rows",
        )


def test_row_list_view_enter_posts_key_activated_only():
    async def _go():
        app = _RowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#rows", widgets.RowListView)
            lv.focus()
            lv.index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            (ev,) = app.events
            assert ev.source == "key" and ev.chain == 1
            assert ev.index == 0 and ev.item_id == "row-a"
            assert ev.control is lv
            assert app.selected_seen == [], "Enter 必须只发 RowActivated、不发 Selected"

    _run(_go())


def test_row_list_view_click_posts_chain():
    async def _go():
        app = _RowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#rows", widgets.RowListView)
            lv.focus()
            await pilot.pause()
            await pilot.click("#row-b", times=2)
            await pilot.pause()
            clicks = [e for e in app.events if e.source == "click"]
            assert [e.chain for e in clicks] == [1, 2], "双击=chain 1、2 两次激活"
            assert all(e.item_id == "row-b" for e in clicks)
            assert clicks[-1].index == 1
            assert app.selected_seen == [], "点击也必须只发 RowActivated、不发 Selected"
            await pilot.click("#row-c")
            await pilot.pause()
            last = app.events[-1]
            assert last.source == "click" and last.chain == 1
            assert last.item_id == "row-c" and last.index == 2

    _run(_go())


# ---------------------------------------------------------------------------
# ④ ChoiceCard + 视觉件构造
# ---------------------------------------------------------------------------
class _CardApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.chosen: list[str] = []

    def on_choice_card_chosen(self, event) -> None:
        self.chosen.append(event.value)

    def compose(self) -> ComposeResult:
        yield widgets.ChoiceCard("A", "甲方案（选项 A）", id="card-a")


def test_choice_card_click_and_highlight():
    async def _go():
        app = _CardApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            c = app.query_one("#card-a", widgets.ChoiceCard)
            assert not c.selected and "-selected" not in c.classes
            c.toggle_selected()
            assert c.selected and "-selected" in c.classes
            c.toggle_selected()
            assert not c.selected and "-selected" not in c.classes
            await pilot.click("#card-a")
            await pilot.pause()
            assert app.chosen == ["A"], "点击只上报 value"
            assert not c.selected, "点击不自改选中态（流程代码驱动高亮）"

    _run(_go())


def test_visual_factories_compose():
    class _V(App):
        def compose(self) -> ComposeResult:
            yield widgets.section_title("分区标题")
            yield widgets.card("行一", "行二", id="c")
            yield widgets.hint("提示行")
            yield widgets.make_row("◆", "标题", meta="3/5", indent=2)
            yield widgets.MarkedTree("根", id="tree")
            yield widgets.MarkedList(
                widgets.ChainListItem(Static("项"), id="it"), id="ml"
            )

    async def _go():
        app = _V()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query(widgets.SectionTitle)
            assert app.query(widgets.HintLine)
            assert app.query(widgets.Row)
            assert app.query_one("#c", widgets.Card)
            assert app.query_one("#tree", widgets.MarkedTree)
            assert app.query_one("#ml", widgets.MarkedList)
            cells = list(app.query_one(widgets.Row).query(Static))
            assert [str(cell.render()) for cell in cells] == ["◆", "  标题", "3/5"]

    _run(_go())
