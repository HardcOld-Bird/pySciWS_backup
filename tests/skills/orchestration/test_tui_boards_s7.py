"""TUI 改版 s7：三看板卡片化（队列/系统/待决）。

钉四件事：

1. **队列卡**：`queues.QUEUE_MEMBERS` **每成员恰一卡**（含全零「（空）」）＋ 支撑席卡 ＋
   事件水位卡；在途条目**带成员归属**（`· [m] 在途 …`）；
2. **系统首卡=倒计时**，且 `_tick_detail` 只 **update 文本**（倒计时／开关路由／配置三卡
   含每秒读数；卡控件对象一律不重建——polish-c 起三处统一实时）；
3. **待决卡**：待决事项／待批复交付／偏差报告（强制处理队列）三区；
4. 读面（`detail_body`）无「Esc/← 返回主菜单」「批复入口」「口述动作」；「支撑席存量」字样保留。
"""

from __future__ import annotations

import asyncio

from textual.widgets import ListView, Static

from pysci.skills.orchestration.tools import queues, workq
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui import cards as tui_cards


def _run(coro):
    return asyncio.run(coro)


async def _open_module(app, pilot, index: int) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = index
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _card_texts(app) -> list[str]:
    """卡文本（``Card`` 是 ``Static`` 子类：读其**自身** render，不查子孙）。"""
    return [str(c.render()) for c in app.query("#detail-cards > Card")]


# ---------------------------------------------------------------------------
# ① 队列看板（每成员一卡，含空队；在途带成员归属）
# ---------------------------------------------------------------------------
def test_queues_board_one_card_per_member_including_empty(tmp_path, monkeypatch):
    qdir = tmp_path / "queues"
    qdir.mkdir()
    monkeypatch.setattr(queues, "QUEUES_DIR", qdir)  # 空队列根：全员皆空

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 2)  # 队列与运行
            return app.view, _card_texts(app), list(app.query("#detail-cards > Card"))

    view, texts, cards = _run(_go())
    assert view == "detail"
    assert len(cards) == len(queues.QUEUE_MEMBERS) + 2, "每成员一卡 + 支撑席卡 + 事件卡"
    joined = "\n".join(texts)
    for m in queues.QUEUE_MEMBERS:
        assert f"[{m}]" in joined, f"缺成员卡：{m}"
    assert "（空）" in joined, "全零成员须明示「（空）」"
    assert "支撑席存量" in joined, (
        "「支撑席存量」字样保留（test_live_config_panel 钉住）"
    )
    assert "事件水位" in joined


def test_queues_board_inflight_row_carries_member_name(tmp_path, monkeypatch):
    qdir = tmp_path / "queues"
    qdir.mkdir()
    monkeypatch.setattr(queues, "QUEUES_DIR", qdir)
    queues.enqueue("sim", text="甲", plan="p-x", step="s1")
    qp = queues.queue_path("sim")
    workq.mutate(
        qp,
        workq.load(qp)[0]["id"],
        status=workq.STATUS_IN_PROGRESS,
        started_at="2026-10-16T10:00:00+08:00",
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 2)
            return _card_texts(app)

    joined = "\n".join(_run(_go()))
    assert "· [sim] 在途" in joined, "在途条目须带成员归属"
    assert "（p-x·s1）" in joined and "起于 2026-10-16T10:00:00" in joined


# ---------------------------------------------------------------------------
# ② 系统看板（首卡=倒计时；tick 只刷倒计时卡）
# ---------------------------------------------------------------------------
def test_system_board_countdown_is_first_card(monkeypatch):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 4)  # 系统与路由
            cards = list(app.query("#detail-cards > Card"))
            return (
                app.view,
                [c.id for c in cards],
                app.detail_countdown_text,
            )

    view, card_ids, cd = _run(_go())
    assert view == "detail"
    assert card_ids[0] == "detail-countdown", "首卡＝倒计时卡"
    assert "倒计时" in cd


def test_tick_only_refreshes_countdown_card(monkeypatch):
    """`_tick_detail` 只刷倒计时卡文本：其余卡控件对象不被重建（防闪烁）。"""
    seq = {"n": 0}

    def fake_countdown() -> str:
        seq["n"] += 1
        return f"  倒计时：距下次切换还有 {seq['n']} 分 0 秒（模拟时钟 → 入谷）"

    monkeypatch.setattr(tui_cards, "countdown_text", fake_countdown)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 4)
            card_widgets = list(app.query("#detail-cards > Card"))
            before_ids = [id(c) for c in card_widgets]
            first_text = app.detail_countdown_text
            app._tick_detail()  # 模拟一秒到达
            await pilot.pause()
            after_widgets = list(app.query("#detail-cards > Card"))
            return (
                first_text,
                app.detail_countdown_text,
                before_ids,
                [id(c) for c in after_widgets],
                str(app.query_one("#detail-countdown", Static).render()),
            )

    first, second, before_ids, after_ids, card_text = _run(_go())
    assert first != second, "倒计时卡文本须随 tick 变化"
    assert "2 分 0 秒" in second or "秒" in second
    assert before_ids == after_ids, "其余卡（含倒计时卡本身）不得被重建"
    assert "倒计时" in card_text, "倒计时卡控件文本已就地更新"


# ---------------------------------------------------------------------------
# ③ 待决看板（三区，含偏差报告）
# ---------------------------------------------------------------------------
def test_pending_board_has_three_cards_with_deviations():
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 6)  # 待决与验收
            return (
                app.view,
                [c.id for c in app.query("#detail-cards > Card")],
                _card_texts(app),
            )

    view, card_ids, texts = _run(_go())
    assert view == "detail"
    assert card_ids == ["card-pending", "card-review", "card-deviations"]
    joined = "\n".join(texts)
    assert "待决事项" in joined and "待批复交付" in joined
    assert "偏差报告（强制处理队列）" in joined


# ---------------------------------------------------------------------------
# ④ 读面干净 + 文本面保留
# ---------------------------------------------------------------------------
def test_board_faces_clean_and_detail_body_keeps_support_seat(tmp_path, monkeypatch):
    qdir = tmp_path / "queues"
    qdir.mkdir()
    monkeypatch.setattr(queues, "QUEUES_DIR", qdir)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            faces = []
            for idx in (2, 4, 6):  # queues / system / pending
                await _open_module(app, pilot, idx)
                faces.append((app.view, app.detail_body, _card_texts(app)))
                await pilot.press("escape")
                await pilot.pause()
            return faces

    faces = _run(_go())
    for view, body, texts in faces:
        assert view == "detail" and body.strip()
        joined = body + "\n" + "\n".join(texts)
        for banned in ("Esc/← 返回主菜单", "批复入口", "口述动作"):
            assert banned not in joined, banned
    # 队列文本面（徽标指纹用）里的钉串仍在
    assert "支撑席存量" in faces[0][1]


def test_non_board_modules_still_render_text_face():
    """无卡片规格的模块仍走**文本面**（卡片栈隐藏、文本 Static 在场）。

    s8 起七个模块各有看板或专用面板，故用**替身模块**（key 无卡片规格）钉这条兜底路径。
    """
    from pysci.skills.orchestration.tools.tui.modules import Module

    def _stub() -> Module:
        return Module(
            key="fake",
            title="替身模块",
            summary=lambda: ["读数一"],
            detail=lambda: ["读数行一", "读数行二"],
            cursor=lambda: "0",
            updates_since=lambda old: 0,
        )

    async def _go():
        app = tui.PysciTUI(mods=[_stub()])
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 0)  # 唯一模块
            return (
                app.view,
                app.query_one("#detail-body-text", Static).display,
                app.query_one("#detail-cards").display,
                app.detail_body,
            )

    view, text_shown, cards_shown, body = _run(_go())
    assert view == "detail" and text_shown is True and cards_shown is False
    assert "读数行一" in body
