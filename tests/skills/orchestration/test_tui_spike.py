"""`pysci` TUI 骨架 spike（README §12，M1）：Pilot 无头键盘导航 + 入口分派。

Pilot 覆盖的是**交互逻辑**（选中态 / 视图切换 / Esc 返回 / 干净退出）；实机终端
（Git Bash/mintty、Windows Terminal、PowerShell 5.1）的渲染与键盘事件是 Pilot **够不到**的
残余风险——结论里如实标注，不在测试里谎称「全部兼容」。
"""

from __future__ import annotations

import asyncio

from pysci.skills.orchestration.tools import mirror
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


def test_module_list_matches_refactor_clause():
    """主菜单 = README §12 的七模块（顺序与标题）。"""
    mods = tui.modules()
    assert [m.key for m in mods] == [
        "taskbooks",
        "plans",
        "queues",
        "review",
        "system",
        "decisions",
        "pending",
    ]
    assert mods[2].title == "队列与运行"


def test_pilot_arrow_navigation_updates_selection():
    """↑↓ 切换选中态（Pilot 无头驱动真实键盘事件）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            assert app.view == "menu" and app.selected == 0
            await pilot.press("down")
            assert app.selected == 1
            await pilot.press("down")
            assert app.selected == 2
            await pilot.press("up")
            assert app.selected == 1

    _run(_go())


def test_pilot_enter_opens_detail_and_escape_returns():
    """Enter 进详情（全屏该模块）、Esc 返回主菜单。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("down")  # → 计划与派发
            await pilot.press("enter")
            assert app.view == "detail"
            assert "计划与派发" in app.detail_text
            await pilot.press("escape")
            assert app.view == "menu" and app.detail_text == ""

    _run(_go())


def test_pilot_starts_renders_and_exits_cleanly():
    """启动→渲染（七行菜单）→干净退出（主菜单 Esc = 退出），无异常、无残留终端状态。"""

    async def _go():
        from textual.widgets import ListView

        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            assert len(app.query_one("#menu", ListView).children) == 7
            await pilot.press("escape")  # 主菜单 Esc = 退出
        assert not app.is_running  # 退出后无残留运行态

    _run(_go())


# ---------------------------------------------------------------------------
# 入口分派（README §12）
# ---------------------------------------------------------------------------
def test_non_tty_no_subcommand_is_status(monkeypatch, capsys):
    """非 tty + 无子命令 → 等价 `pysci status` 静态输出，退出码 0，绝不阻塞等输入。"""
    monkeypatch.setattr(tui, "_is_tty", lambda: False)
    rc = tui.main([])
    out = capsys.readouterr().out
    mirror.main(["status"])
    ref = capsys.readouterr().out
    assert rc == 0
    assert out.strip() and out == ref  # 逐字等价 = 「复用 cmd_status，未重写」


def test_subcommand_delegates_to_static_channel(monkeypatch, capsys):
    """有子命令 → 静态命令面（LLM 通道），即便 tty 也不进 TUI。"""
    monkeypatch.setattr(tui, "_is_tty", lambda: True)
    assert tui.main(["status"]) == 0
    assert capsys.readouterr().out.strip()
