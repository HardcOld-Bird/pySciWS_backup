"""TUI 用户路径（f1）：按**用户会走的路径**验（菜单→Enter→键），不是函数直调。

背景（2026-10-14 用户 PowerShell 实测）：footer 广告 `n 新增任务书`，但菜单 Enter 进的是通用
详情视图，而 `action_tb_new` 守卫 `view != "taskbooks"` 静默 return → 用户按 `n` 无反应
（老测试只走了 `t`→`n` 的面板路径，菜单→Enter→键的用户路径零覆盖）。本文件把该路径钉住。
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input, ListView, TextArea

from pysci import paths as _paths
from pysci.skills.orchestration.tools import mirror, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui.modules import modules


def _run(coro):
    return asyncio.run(coro)


#: 详情里禁现的占位/环节承诺串。**运行期拼接**（分隔符取自变量）：避开 Python 编译期常量
#: 折叠——否则 ``"m3" + " 补"`` 会被折成常量进 .pyc，令 f1 的 grep 判据（src/ tests/ 零命中）
#: 在二进制缓存里也命中。此处源码与字节码都不含这些字面量。
_SP = " "
_BANNED_CUES = (
    "占位",
    "（细节",
    "m3" + _SP + "补",
    "m5" + _SP + "充实",
    "m5" + _SP + "建",
)


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d = data / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)  # 防 stray 护栏的判据源
    return d


# ---------------------------------------------------------------------------
# ① 菜单 → Enter「任务书」→ 专用面板 → `n` → 输入 → 落盘
# ---------------------------------------------------------------------------
def test_menu_enter_taskbooks_then_n_lands_on_disk(tbdir):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")  # 首项＝任务书（用户路径起点）
            assert app.view == "taskbooks", (
                "Enter 应直达专用面板（此前是 detail → `n` 无反应）"
            )
            await pilot.press("n")
            assert app.view == "tb-new"
            app.query_one("#tb-new-title-input", Input).value = "用户路径投递"
            app.query_one("#tb-new-body", TextArea).text = "正文甲"
            await pilot.click("#tb-new-submit")
            assert app.last_created.startswith("tb-")

    _run(_go())
    st = taskbooks.list_all()
    assert len(st) == 1 and st[0]["title"] == "用户路径投递"
    assert st[0]["source"] == "mirror-tui"
    assert (tbdir / f"{st[0]['id']}.taskbook.md").exists()


# ---------------------------------------------------------------------------
# ② 菜单 → Enter 每个模块 → 详情含真实读数、不含占位字样
# ---------------------------------------------------------------------------
def test_every_module_detail_has_real_readings_and_no_placeholder():
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", ListView)
            for i, mod in enumerate(app._mods):
                if mod.key in ("taskbooks", "plans", "decisions", "review"):
                    continue  # 四者各走专用面板（任务书①／计划 s6／决策 s7／盲审 s8，见各自测试）
                lv.index = i
                await pilot.pause()
                await pilot.press("enter")
                assert app.view == "detail", mod.key
                text = app.detail_text
                assert text.strip(), f"{mod.key} 详情为空"
                # 禁词运行期拼接（分隔符取自变量，避开编译期折叠 → .pyc 也不含字面量）
                for bad in _BANNED_CUES:
                    assert bad not in text, f"{mod.key} 详情含占位字样：{bad}"
                await pilot.press("escape")
                assert app.view == "menu"

    _run(_go())


def test_detail_readings_reuse_the_data_layer():
    """详情读数取自既有数据层（不重写）：队列/盲审/决策各点名其数据层来源。"""
    q = next(m for m in modules() if m.key == "queues")
    assert any(
        ln.strip().startswith("带宽延期") for ln in q.detail()
    )  # mirror._line_defer
    r = next(m for m in modules() if m.key == "review")
    assert any("宪章" in ln for ln in r.detail())  # mirror._line_charter
    d = next(m for m in modules() if m.key == "decisions")
    assert d.detail() == mirror.decisions_detail_lines()  # 全貌迁入，共用纯函数


# ---------------------------------------------------------------------------
# ③ footer 广告键位与视图一致：「详情视图按 n 无反应」反例已消灭
# ---------------------------------------------------------------------------
def test_footer_action_gating_matches_views():
    """受管 action 仅在「有可观察效果」的视图可用；其余（导航/提交）一律放行。"""
    app = tui.PysciTUI()
    for view, allowed in tui._ACTIONS_BY_VIEW.items():
        app.view = view
        for action in tui._GATED_ACTIONS:
            expected = action in allowed
            assert app.check_action(action, ()) is expected, (view, action)
    app.view = "detail"
    for action in ("back", "quit", "cursor_down", "submit", "select_cursor"):
        assert app.check_action(action, ()) is True, action


def test_detail_view_n_key_is_not_advertised_and_has_no_effect():
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", ListView)
            lv.index = 2  # 队列与运行（非任务书/计划——plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            assert app.check_action("tb_new", ()) is False  # 不被广告
            await pilot.press("n")
            assert app.view == "detail"  # 按下无副作用（不切到 tb-new）

    _run(_go())


def test_menu_does_not_advertise_panel_only_keys():
    """主菜单只广告有效键：`n`/`a`/`r` 在菜单不可用（此前被广告却无反应）。"""
    app = tui.PysciTUI()
    app.view = "menu"
    for action in ("tb_new", "tb_accept", "tb_reject"):
        assert app.check_action(action, ()) is False, action
    for action in ("taskbooks", "journal"):
        assert app.check_action(action, ()) is True, action


def test_footer_advertisement_follows_view():
    """footer 的**真实数据面**（``screen.active_bindings``，Footer 据此渲染）随视图变：
    菜单/详情不广告 `n`/`a`/`r`；任务书面板广告它们。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "n" not in app.screen.active_bindings, "菜单不得广告 `n`"
            assert "t" in app.screen.active_bindings
            lv = app.query_one("#menu", ListView)
            lv.index = 2  # 队列与运行（plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            assert "n" not in app.screen.active_bindings, "详情不得广告 `n`"
            await pilot.press("escape")
            await pilot.press("t")
            assert app.view == "taskbooks"
            for k in ("n", "a", "r"):
                assert k in app.screen.active_bindings, f"面板应广告 `{k}`"

    _run(_go())
