"""TUI 改版 s7（task tui-overhaul-20261016-s7-boards-ui）：决策审计列表→详情（Markdown）。

钉四件事：

1. **列表**（`#decisions`）：行＝attention 旗标（``[!]``/``[i]``）＋ ts ＋ class/remedy_shape
   ＋ title（截断）；**attention 置顶**；空态「（暂无决策记录）」；
2. **详情**（`#decision-detail`）：头卡（id/ts/class/shape/attention）＋ **Markdown 正文**
   （发现/判决/实施与核验/留痕；无值字段省略）；
3. 回退链：`decision-detail → decisions → menu`；`#decisions-scroll`/`#decision-detail-scroll`
   位置行在场（滚动三路已由 s4/s5 内建）；
4. 读面无「Esc/← 返回主菜单」「批复入口」「口述动作」。
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import ListView, Markdown, Static

from pysci.skills.orchestration.tools import inspectd
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


_ATTENTION_ROW = {
    "id": "f-mech",
    "ts": "2026-10-16T09:00:00+08:00",
    "class": "mechanical",
    "remedy_shape": "change",
    "title": "机械发现：免判决改了策略面",
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

_PLAIN_ROW = {
    "id": "f-judg",
    "ts": "2026-10-16T11:00:00+08:00",
    "class": "judgment",
    "remedy_shape": "restore",
    "title": "判断发现：恢复旧行为",
    "verdict_ref": "J 案卷 evals/jury-cases/f-judg.md",
    "ruling": "裁示：恢复旧行为",
    "commit": "abc1234",
    "verify": "stage-b",
    "backlog_status": "done",
    "note": "留痕一行",
    "attention": False,
    "finding_status": "open",
    "channel": "jury",
}


@pytest.fixture
def dface(monkeypatch):
    """决策行替身（真 inspectd 读盘面不动）：attention 行 + 普通行。"""
    rows = [dict(_ATTENTION_ROW), dict(_PLAIN_ROW)]
    monkeypatch.setattr(inspectd, "decision_rows", lambda **kw: [dict(r) for r in rows])
    return rows


async def _open_decisions(app, pilot) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = 5  # 决策审计
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _row_texts(app) -> list[str]:
    lv = app.query_one("#decisions-list", ListView)
    return [
        " ".join(str(cell.render()) for cell in li.query(Static)) for li in lv.children
    ]


# ---------------------------------------------------------------------------
# ① 列表
# ---------------------------------------------------------------------------
def test_decision_list_rows_and_attention_first(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            return app.view, [r["id"] for r in app.decision_rows], _row_texts(app)

    view, ids, texts = _run(_go())
    assert view == "decisions"
    assert ids == ["f-mech", "f-judg"], "attention 行置顶（数据层序照搬）"
    assert texts[0].startswith("[!]"), "attention 旗标 [!]"
    assert "2026-10-16T09:00:00" in texts[0]
    assert "mechanical/change" in texts[0] and "机械发现" in texts[0]
    assert texts[1].startswith("[i]") and "judgment/restore" in texts[1]


def test_decision_list_empty_state(monkeypatch):
    monkeypatch.setattr(inspectd, "decision_rows", lambda **kw: [])

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            return app.view, app.decision_rows, _row_texts(app)

    view, rows, texts = _run(_go())
    assert view == "decisions" and rows == []
    assert any("（暂无决策记录）" in t for t in texts)


# ---------------------------------------------------------------------------
# ② 详情（头卡＋Markdown）
# ---------------------------------------------------------------------------
def test_decision_detail_head_and_markdown(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            app.query_one(
                "#decisions-list", ListView
            ).index = 1  # 普通行（有 ruling/commit）
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return (
                app.view,
                app.decision_detail_id,
                app.decision_detail_title,
                app.decision_detail_head,
                app.decision_detail_md,
            )

    view, did, title, head, md = _run(_go())
    assert view == "decision-detail" and did == "f-judg" and did in title
    assert "2026-10-16T11:00:00+08:00" in head and "judgment" in head
    assert "restore" in head and "常规" in head
    assert "# 决策 f-judg" in md
    assert "## 发现" in md and "判断发现：恢复旧行为" in md
    assert "## 判决" in md and "裁示：恢复旧行为" in md and "J 案卷" in md
    assert "`abc1234`" in md and "stage-b" in md and "done" in md
    assert "## 留痕" in md and "留痕一行" in md


def test_decision_detail_attention_flag_and_optional_fields_omitted(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            app.query_one(
                "#decisions-list", ListView
            ).index = 0  # attention 行（无 ruling/commit）
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return app.decision_detail_head, app.decision_detail_md

    head, md = _run(_go())
    assert "免判决+change" in head and "最需人工审计" in head
    assert "## 发现" in md
    assert "- 依据：免判决（mechanical 通道）" in md
    assert "裁示" not in md, "无 ruling 则不写裁示行（无值字段省略）"
    assert "## 实施与核验" in md and "- backlog：pending" in md
    assert "## 留痕" not in md and "实施 commit" not in md, "无值字段省略"


def test_decision_detail_renders_markdown_widget(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            app.query_one("#decisions-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return app.query_one("#decision-detail-md", Markdown)

    assert isinstance(_run(_go()), Markdown)


# ---------------------------------------------------------------------------
# ③ 回退链 + 滚动位置行 + 禁语
# ---------------------------------------------------------------------------
def test_decision_back_chain_and_scroll_lines(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            in_list = bool(app.query("#decisions-scroll"))
            app.query_one("#decisions-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            in_detail = bool(app.query("#decision-detail-scroll"))
            await pilot.press("escape")
            await pilot.pause()
            back1 = app.view
            await pilot.press("escape")
            await pilot.pause()
            return in_list, in_detail, back1, app.view

    in_list, in_detail, back1, back2 = _run(_go())
    assert in_list and in_detail, "两条位置行在场"
    assert back1 == "decisions" and back2 == "menu"


def test_decision_faces_have_no_removed_prompts(dface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            await _open_decisions(app, pilot)
            list_face = "\n".join(_row_texts(app))
            app.query_one("#decisions-list", ListView).index = 1
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            detail_face = "\n".join([app.decision_detail_head, app.decision_detail_md])
            return list_face, detail_face

    list_face, detail_face = _run(_go())
    for banned in ("Esc/← 返回主菜单", "批复入口", "口述动作"):
        assert banned not in list_face, banned
        assert banned not in detail_face, banned
