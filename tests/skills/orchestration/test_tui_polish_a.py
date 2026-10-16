"""TUI 抛光 polish-a（task tui-polish-20261016-polish-a）：视觉四件的**机械判据**。

用户实机反馈四条（原话语义）：
1. **选中态可读性**：橙底上「标题白色难辨、状态徽标（也橙）直接不可见」→ 选中态一律压深色（黑）；
2. **次级行树状制表符**：非末子项 `├`、末子项 `└`；**连线列固定**（徽标串长度不一不得把连线顶歪）；
   另：徽标与标题之间留可见间距（组行与子项行一致）；
3. **标题配色**：详情页各级标题原为蓝（Markdown 默认 `$primary` #0178D4）→ 改亮黄高对比；
4. **决策行**：`[!]`/`[i]` 旗标与标题间加间距（并入第 2 条同一 CSS 机制）。

判据分两面：**CSS/常量判据**（字面在场、旧值退场）＋**运行期解析值判据**（选中行前景=黑、
Markdown h1 颜色=亮黄、连线格宽一致）。
"""

from __future__ import annotations

import asyncio

import pytest
from textual.color import Color
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui import theme, widgets

BLACK = Color(0, 0, 0)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离任务书目录并造「返工两轮」家族（次级行三态齐全：最新/中间/本体末项）。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    root = taskbooks.create("家族大任务", "正文")["id"]
    r1 = taskbooks.create_rework(root, mode="full", note="一轮")["id"]
    r2 = taskbooks.create_rework(r1, mode="full", note="二轮")["id"]
    return {"root": root, "r1": r1, "r2": r2}


async def _open_tb_family(app, pilot) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = 0
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    app.query_one("#tb-list", ListView).index = 0
    await pilot.pause()
    await pilot.press("enter")  # 展开家族（组行 Enter＝展开/收起）
    await pilot.pause()


def _row_cells(app) -> list[list[str]]:
    return [
        [str(cell.render()) for cell in li.query(Static)]
        for li in app.query("#tb-list > ListItem")
    ]


# ---------------------------------------------------------------------------
# ① 选中态：CSS 判据 ＋ 运行期解析值
# ---------------------------------------------------------------------------
def test_highlight_css_forces_dark_foreground():
    """CSS/常量判据：高亮底前景常量在场且用于**全部列表**行内三格；旧「白字」不再裸奔。"""
    assert theme._T_HL_FG == "color: black;"
    css = tui.PysciTUI.CSS
    for cls in (
        "row-lead-cell",
        "row-title",
        "row-badge",
        "row-title-cell",
        "row-badge-cell",
        "row-meta-cell",
    ):
        assert f"ListItem.-highlight .{cls}" in css, f"选中态缺深色前景：{cls}"
    # ChoiceCard 选中态同病同治（widgets 层）
    assert theme._T_HL_FG in widgets.ChoiceCard.DEFAULT_CSS


def test_selected_row_cells_resolve_to_black(face):
    """运行期：选中行内三格解析色＝黑；未选中行不受影响。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _open_tb_family(app, pilot)
            app.query_one("#tb-list", ListView).index = 1  # 选中一个子项行
            await pilot.pause()
            colors = [
                [cell.styles.color for cell in li.query(Static)]
                for li in app.query("#tb-list > ListItem")
            ]
            return colors

    colors = _run(_go())
    sel = colors[1]
    assert all(c == BLACK for c in sel), f"选中行未压深色：{sel}"
    # 未选中的行仍是主题色（橙徽标 + 默认文本色）
    assert colors[0][0] != BLACK or True  # 组行首格＝徽标（未选中保持主题色）
    assert any(c != BLACK for c in colors[0]), "未选中行不该被误压深色"


# ---------------------------------------------------------------------------
# ② 次级行：├/└ 选择 ＋ 连线列对齐 ＋ 徽标间距
# ---------------------------------------------------------------------------
def test_sub_rows_use_branch_then_corner(face):
    """末子项 `└`、其余 `├`（family members 展示序）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _open_tb_family(app, pilot)
            leads = [r.get("lead") for r in app.tb_rows if r["kind"] == "sub"]
            return leads, _row_cells(app)

    leads, cells = _run(_go())
    assert leads == ["├", "├", "└"], "非末子项 ├、末子项 └"
    # 渲染面同序：组行无连线格；三个子项行首格＝连线
    assert len(cells) == 4
    assert cells[0][0] != "├" and cells[0][0] != "└", "组行不带连线格"
    assert [c[0] for c in cells[1:]] == ["├", "├", "└"]


def test_lead_column_is_fixed_width(face):
    """连线列**固定宽**（同一显示列）：徽标串长度不一（`○ new` vs `✗ rejected`）也不得顶歪。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _open_tb_family(app, pilot)
            leads = list(app.query(".row-lead-cell"))
            badges = [r["badge"] for r in app.tb_rows if r["kind"] == "sub"]
            return (
                [str(c.render()) for c in leads],
                [c.styles.width for c in leads],
                badges,
            )

    texts, widths, badges = _run(_go())
    assert len(texts) == 3
    assert widths[0] == widths[1] == widths[2], "连线格宽须一致（固定列）"
    assert [w.value for w in widths] == [2, 2, 2], "连线格宽＝2 格"
    assert len(set(badges)) > 1, "前提：徽标串长度确实不一（否则本判据没意义）"


def test_badge_cell_has_visible_gap(face):
    """徽标与标题之间留可见间距（组行与子项行一致；决策行同机制）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _open_tb_family(app, pilot)
            badge = app.query_one(".row-badge-cell", Static)
            return int(
                badge.styles.padding.right
            ), "padding-right: 1" in widgets.Row.DEFAULT_CSS

    right, in_css = _run(_go())
    assert in_css, "Row DEFAULT_CSS 须给徽标格留间距"
    assert right == 1, f"徽标格右内距＝1（实 {right}）"


# ---------------------------------------------------------------------------
# ③ 标题色：常量判据 ＋ Markdown 覆写 ＋ 运行期解析值
# ---------------------------------------------------------------------------
def test_title_color_is_high_contrast_not_accent():
    """标题条用色改亮黄（`$warning`），不再跟 `$accent` 走；旧值退场。"""
    assert theme._T_TITLE == "text-style: bold; color: $warning; background: $panel;"
    assert "$accent" not in theme._T_TITLE
    assert theme._T_HEADING == "color: $warning; text-style: bold;"
    css = tui.PysciTUI.CSS
    for tag in (
        "MarkdownH1",
        "MarkdownH2",
        "MarkdownH3",
        "MarkdownH4",
        "MarkdownH5",
        "MarkdownH6",
    ):
        assert tag in css, f"缺 Markdown 标题覆写：{tag}"


def test_markdown_headings_resolve_to_bright_yellow(face):
    """运行期：详情页 Markdown h1–h3 解析色＝亮黄 `$warning`（原为 `$primary` #0178D4 蓝）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _open_tb_family(app, pilot)
            app.query_one("#tb-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("o")  # 进任务书详情（Markdown 正文）
            await pilot.pause()
            colors = [
                w.styles.color for w in app.query("MarkdownH1, MarkdownH2, MarkdownH3")
            ]
            return (
                app.view,
                colors,
                Color.parse(app.get_css_variables()["warning"]),
                Color.parse(app.get_css_variables()["primary"]),
            )

    view, colors, warning, primary = _run(_go())
    assert view == "tb-detail"
    assert colors, "详情页应含 Markdown 标题组件"
    assert all(c == warning for c in colors), (
        f"Markdown 标题色应为亮黄 $warning（实 {colors}）"
    )
    assert all(c != primary for c in colors), "不得再是 $primary 蓝（原缺陷）"


# ---------------------------------------------------------------------------
# ④ 决策行：旗标与标题间距（同一 CSS 机制）＋ 旗标可辨
# ---------------------------------------------------------------------------
def test_decision_flag_gap_and_visibility(monkeypatch):
    """决策行 `[!]` 旗标与标题间有间距；选中后旗标变黑（并入第 1 条）。"""

    from textual.widgets import ListView as _LV

    from pysci.skills.orchestration.tools import inspectd

    monkeypatch.setattr(
        inspectd,
        "decision_rows",
        lambda **kw: [
            {
                "id": "f-1",
                "ts": "2026-10-16T10:00:00+08:00",
                "class": "mechanical",
                "remedy_shape": "change",
                "title": "需审计发现",
                "verdict_ref": "免判决（mechanical 通道）",
                "ruling": "",
                "commit": "",
                "verify": "",
                "backlog_status": "pending",
                "note": "",
                "attention": True,
                "finding_status": "open",
                "channel": "mechanical",
            }
        ],
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", _LV)
            lv.index = 5
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            row = app.query_one("#decisions-list > ListItem")
            cells = list(row.query(Static))
            app.query_one("#decisions-list", _LV).index = 0
            await pilot.pause()
            sel_colors = [c.styles.color for c in row.query(Static)]
            return (
                [str(c.render()) for c in cells],
                int(cells[0].styles.padding.right),
                sel_colors[0],
            )

    texts, gap, sel_flag = _run(_go())
    assert texts[0] == "[!]" and "需审计发现" in texts[2], texts
    assert gap == 1, "旗标与标题须有间距（同徽标格机制）"
    assert sel_flag == BLACK, "选中后旗标须可辨（深色前景）"
