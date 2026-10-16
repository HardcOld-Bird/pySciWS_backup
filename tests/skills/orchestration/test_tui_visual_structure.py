"""TUI 美学统一 + 键位矩阵 + 视觉结构回归（tui-fixup f2）。

三条判据（**不做像素比对**）：
1. **共享主题常量**：同一视觉参数只在共享常量里硬编码一次（判据按 20261014-075230-devops
   约定：**运行期拼接**，字面量不进源码；s2 起判据扫 `tui/` 目录全 .py 聚合文本）。
2. **键位矩阵**：每个视图下，Footer 的**真实数据面**（`screen.active_bindings`，Footer 据此
   渲染）与 `check_action` 守卫**一致**；广告的键逐一按下有可观察效果；未广告的键无副作用。
3. **视觉结构**：主题常量被各面板引用、组件 id 齐全、七模块 + 任务书面板 + 口述面全走一遍
   无崩溃。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.binding import Binding

from pysci import paths as _paths
from pysci.skills.orchestration.tools import taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui.modules import modules


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    """任务书面板/详情页的读数是**真实 state**——不隔离则该模块用例随开发者机器上的
    任务书而变（也让「任务书详情页」这条视图走不进：空列表按 Enter 无处可去）。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return data


#: action 名 → 键（从 BINDINGS 反推，避免手工表漂移）。
#: BINDINGS 允许混用「三元组」与 ``Binding`` 实例（Textual ``Binding.make_bindings`` 同规则）
#: ——详情页滚动键用 ``Binding(..., show=False)`` 声明（**不进 footer 广告面**，其可见性由
#: 位置指示行承担），故这里统一按 Textual 自己的归一化读。
KEY_OF_ACTION: dict[str, str] = {
    b.action: b.key for b in Binding.make_bindings(tui.PysciTUI.BINDINGS)
}

#: 详情页三区组件。
_DETAIL_IDS = ("#detail", "#detail-title", "#detail-body", "#detail-hint")
#: 任务书详情页三区组件（Enter 进；与模块详情页同构）。
_TB_DETAIL_IDS = (
    "#tb-detail",
    "#tb-detail-title",
    "#tb-detail-body",
    "#tb-detail-hint",
)
_PANEL_IDS = (
    "#menu",
    "#summary",
    "#journal",
    "#journal-title",
    "#journal-select",
    "#journal-text",
    "#journal-result",
    "#taskbook",
    "#tb-title",
    "#tb-list",
    "#tb-note",
    "#tb-msg",
    "#tb-new",
    "#tb-new-title",
    "#tb-new-title-input",
    "#tb-new-body",
    "#tb-new-submit",
    "#tb-new-msg",
)


# ---------------------------------------------------------------------------
# ① 共享主题常量：同一视觉参数只硬编码一次
# ---------------------------------------------------------------------------
def test_visual_constants_defined_once_in_source():
    """同一视觉参数在 ``tui/`` 目录源码里**只出现一次**。

    s2 共享层起唯一出处 = ``theme.py``：判据从「扫 app.py 单文件」升级为「扫 ``tui/``
    目录全部 .py 聚合文本」——纪律不变（各面板 CSS 一律插值引用共享常量），覆盖面更严
    （widgets.py 等新面同样受钉）。
    """
    tui_dir = Path(tui.__file__).resolve().parent
    src = "".join(p.read_text(encoding="utf-8") for p in sorted(tui_dir.glob("*.py")))
    for name, fragment in (
        ("面板边框", "border" + ": round $accent"),
        ("提示边框", "border" + ": round $warning"),
        ("标题条", "text-style: bold; color" + ": $accent; background: $panel"),
        ("弱文案", "color" + ": $text-muted"),
    ):
        assert src.count(fragment) == 1, (
            f"{name} 被硬编码 {src.count(fragment)} 次（应仅 theme.py 1 次）"
        )


def _css_rules(css: str) -> dict[str, str]:
    """把本 TUI 的 CSS（无嵌套规则）解析成 ``{selector: rule-body}``。"""
    out: dict[str, str] = {}
    for chunk in css.split("}"):
        if "{" not in chunk:
            continue
        sel, body = chunk.split("{", 1)
        sel = sel.strip()
        if sel:
            out[sel] = body.strip()
    return out


def test_every_panel_references_shared_constants():
    """视觉结构：主菜单/详情/口述/任务书与四个标题条都来自共享常量（同一段字面）。"""
    border = "border" + ": round $accent;"
    title = "text-style: bold; color" + ": $accent; background: $panel;"
    rules = _css_rules(tui.PysciTUI.CSS)
    for selector in ("#menu", "#detail", "#journal", "#taskbook"):
        assert selector in rules, f"缺面板样式：{selector}"
        assert border in rules[selector], f"{selector} 未引用共享边框常量"
    for selector in ("#detail-title", "#journal-title", "#tb-title", "#tb-new-title"):
        assert selector in rules and title in rules[selector], (
            f"{selector} 未引用共享标题常量"
        )
    # 选中态视觉语言同源（主菜单与任务书列表的高亮一致）
    assert "background: $accent" in rules["#menu > ListItem.-highlight"]
    assert (
        rules["#menu > ListItem.-highlight"] == rules["#tb-list > ListItem.-highlight"]
    )


# ---------------------------------------------------------------------------
# ② 键位矩阵：footer 广告 ↔ check_action 守卫，全 action 表一致
# ---------------------------------------------------------------------------
def _advertised(app) -> set[str]:
    """Footer 的**真实数据面**（``screen.active_bindings`` 的键）映射回 action 名。"""
    keys = set(app.screen.active_bindings)
    return {a for a in KEY_OF_ACTION if KEY_OF_ACTION[a] in keys}


async def _each_view(pilot, app):
    """把应用带进**每一个视图**（走用户会走的路径），逐个 yield 视图名。"""
    yield "menu"
    lv = app.query_one("#menu", tui.ListView)
    lv.index = 1
    await pilot.pause()
    await pilot.press("enter")
    yield "detail"
    await pilot.press("escape")
    await pilot.press("t")
    yield "taskbooks"
    if app.tb_rows:  # 空列表按 `o` 无处可去（无任务书时跳过详情页）
        await pilot.press("o")  # s5：Enter 是展开/收起，`o` 才是进详情
        await pilot.pause()
        yield "tb-detail"
        await pilot.press("escape")
        yield "taskbooks"
    await pilot.press("n")
    yield "tb-new"
    await pilot.press("escape")
    await pilot.press("j")
    yield "journal"
    await pilot.press("escape")


def test_key_matrix_matches_guard_for_all_actions(tbdir):
    """全 action 表 × 全视图：Footer 广告集合 == `check_action` 守卫集合（不许漂移）。"""
    taskbooks.create("键位矩阵用任务书", "正文")
    app = tui.PysciTUI()

    async def _go():
        async with app.run_test() as pilot:
            await pilot.pause()
            async for view in _each_view(pilot, app):
                app.view = view
                guarded = {
                    a for a in tui._GATED_ACTIONS if app.check_action(a, ()) is True
                }
                advertised = _advertised(app) & set(tui._GATED_ACTIONS)
                assert advertised == guarded, (
                    f"{view}: footer 广告 {sorted(advertised)} ≠ 守卫 {sorted(guarded)}"
                )
                # 受管 action 之外（back/quit）永远可用
                for action in ("back", "quit"):
                    assert app.check_action(action, ()) is True, (view, action)

    _run(_go())


def test_every_advertised_key_has_an_observable_effect():
    """广告的键必须有可观察效果（s4 键位收敛：`t`/`j` **只在主菜单**广告）。

    菜单：`t`/`j` 逐一按下都切视图；详情视图：二者不再出现在 Footer 数据面、
    按下也无副作用（留在详情）。
    """
    effects: dict[str, str] = {}

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("t")
            effects["t"] = app.view
            await pilot.press("escape")
            await pilot.press("j")
            effects["j"] = app.view
            await pilot.press("escape")
            assert app.view == "menu"
            lv = app.query_one("#menu", tui.ListView)
            lv.index = 2  # 队列与运行（plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            keys = set(app.screen.active_bindings)
            for k in ("t", "j", "n", "a", "r"):
                assert k not in keys, f"详情视图不得广告 `{k}`"
            for k in ("t", "j"):
                await pilot.press(k)
                assert app.view == "detail", f"详情视图按 `{k}` 不得有副作用"

    _run(_go())
    assert effects["t"] == "taskbooks", "菜单广告的 `t` 必须有效果"
    assert effects["j"] == "journal", "菜单广告的 `j` 必须有效果"


def test_unadvertised_key_has_no_side_effect():
    """反向：未广告的键在该视图按下**无副作用**（f1「详情按 n 无反应」反例已消灭）。"""
    seen: dict[str, str] = {}

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", tui.ListView)
            lv.index = 2  # 队列与运行（plans 已走专用面板，s6）
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            for key in ("n", "a", "r"):
                await pilot.press(key)
                seen[key] = app.view
            await pilot.press("escape")

    _run(_go())
    assert seen == {"n": "detail", "a": "detail", "r": "detail"}, seen


# ---------------------------------------------------------------------------
# ③ 视觉结构回归 + 全走一遍不崩溃
# ---------------------------------------------------------------------------
def test_component_ids_complete():
    """组件 id 齐全：菜单/概览/详情三区/口述/任务书/新增面板/任务书详情三区一次性查全。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            for sel in (*_PANEL_IDS, *_DETAIL_IDS, *_TB_DETAIL_IDS):
                assert app.query(sel), f"缺组件：{sel}"

    _run(_go())


def test_walk_every_surface_without_crash():
    """七模块 + 任务书面板 + 口述面全走一遍：无异常、视图切换正确。"""
    visited: list[str] = []

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            for i, mod in enumerate(app._mods):
                lv = app.query_one("#menu", tui.ListView)
                lv.index = i
                await pilot.pause()
                await pilot.press("enter")
                if mod.key == "taskbooks":
                    assert app.view == "taskbooks"
                    await pilot.press("n")
                    assert app.view == "tb-new"
                    await pilot.press(
                        "escape"
                    )  # tb-new → 菜单（一次即回，再按就退出了）
                elif mod.key == "plans":
                    assert app.view == "plans"  # s6：计划面板
                    await pilot.press("escape")
                elif mod.key == "decisions":
                    assert app.view == "decisions"  # s7：决策审计面板
                    await pilot.press("escape")
                else:
                    assert app.view == "detail"
                    await pilot.press("escape")
                visited.append(mod.key)
                assert app.view == "menu"
            await pilot.press("j")
            assert app.view == "journal"
            visited.append("journal")
            await pilot.press("escape")
            assert app.view == "menu"

    _run(_go())
    assert visited[:7] == [m.key for m in modules()]
    assert "journal" in visited


def test_detail_partitioning_three_zones():
    """详情页三区（标题 / 读数 / 操作提示）：提示由 TUI 生成，**不含 Esc 字样**（s4：Esc 归 Footer）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            lv = app.query_one("#menu", tui.ListView)
            lv.index = 6  # 待决与验收
            await pilot.pause()
            await pilot.press("enter")
            assert app.view == "detail"
            assert "详情" in app.detail_title
            assert "待决" in app.detail_body  # 读数在读数区
            assert "偏差报告" in app.detail_body  # s3：偏差报告（强制处理队列）区块
            for gone in ("Esc/← 返回", "口述动作", "批复入口"):
                assert gone not in app.detail_text, gone

    _run(_go())


def test_non_tty_entry_unchanged(monkeypatch, capsys):
    """非 tty 裸 `pysci` 语义不回归（f1/f2 都动过呈现层）。"""
    monkeypatch.setattr(tui, "_is_tty", lambda: False)
    assert tui.main([]) == 0
    assert capsys.readouterr().out.strip()
