"""TUI 改版 s6（task tui-overhaul-20261016-s6-plans-ui）：计划与派发面板。

钉四件事：

1. **列表**：行＝状态徽标（图标带空格）＋plan id＋进度 done/total＋goal（截断），序＝
   `plans.overview_rows()`（新→旧）；单击/Enter → `plan-detail`；空态一行「（暂无计划）」；
2. **详情**：头卡（goal/状态/进度/时间）＋ **可折叠 Tree**（节点数与 `plan_tree_nodes`
   一致、superstep/叶子 label 形态正确、默认全展开）＋ **Markdown 面板联动**（选中节点
   → 面板含该节点 brief）；空 plan/缺 state 不崩；
3. **性能钉**：`plans._load_yaml` 抛异常时列表视图仍能打开（UI 只走 state.json 缓存）；
4. **禁语钉**：列表/详情读面不含「推进：」「多 plan 并行」。
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from textual.widgets import ListView, Markdown

from pysci.skills.orchestration.tools import plans
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pface(tmp_path, monkeypatch):
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    return pdir


def _write_plan(
    pdir,
    pid: str,
    *,
    goal: str = "目标",
    created: str = "2026-10-16T10:00:00+08:00",
    steps: list | None = None,
    state_steps: list | None = None,
    completed_at: str = "",
) -> None:
    (pdir / f"{pid}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": pid,
                "goal": goal,
                "steps": steps
                or [
                    {"id": "s1", "member": "sim", "brief": "环节一"},
                    {"id": "s2", "member": "figure", "brief": "环节二"},
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (pdir / f"{pid}.state.json").write_text(
        json.dumps(
            {
                "plan_id": pid,
                "created": created,
                "goal": goal,
                "steps": state_steps
                or [
                    {"id": "s1", "status": "done"},
                    {"id": "s2", "status": "pending"},
                ],
                "completed_at": completed_at or None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


async def _open_plans(app, pilot) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = 1  # 计划与派发
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _row_texts(app) -> list[str]:
    """列表行渲染文本（每行三格拼起来）——与行模型同序。"""
    from textual.widgets import Static

    lv = app.query_one("#plans-list", ListView)
    return [
        " ".join(str(cell.render()) for cell in li.query(Static)) for li in lv.children
    ]


# ---------------------------------------------------------------------------
# ① 列表：序 / 内容 / 空态
# ---------------------------------------------------------------------------
def test_plan_list_rows_newest_first_with_badge_progress_goal(pface):
    _write_plan(pface, "p-old", created="2026-10-16T09:00:00+08:00", goal="旧计划")
    _write_plan(
        pface,
        "p-new",
        created="2026-10-16T11:00:00+08:00",
        goal="新计划",
        state_steps=[
            {"id": "s1", "status": "done"},
            {"id": "s2", "status": "done"},
        ],
        completed_at="2026-10-16T12:00:00+08:00",
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            return app.view, app.plan_rows, _row_texts(app)

    view, rows, texts = _run(_go())
    assert view == "plans"
    assert [r["id"] for r in rows] == ["p-new", "p-old"], "序＝新→旧"
    assert rows[0]["mark"] == "✔ 完成" and (rows[0]["done"], rows[0]["total"]) == (2, 2)
    assert len(texts) == 2
    assert "✔ 完成" in texts[0] and "p-new" in texts[0] and "2/2" in texts[0]
    assert "新计划" in texts[0]
    assert "▶ 进行中" in texts[1] and "1/2" in texts[1]


def test_plan_list_empty_state(pface):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            return app.view, app.plan_rows, _row_texts(app)

    view, rows, texts = _run(_go())
    assert view == "plans" and rows == []
    assert any("（暂无计划）" in t for t in texts)


# ---------------------------------------------------------------------------
# ② 详情：头卡 / Tree / Markdown 联动
# ---------------------------------------------------------------------------
def _tree_steps_plan() -> tuple[list, list]:
    steps = [
        {
            "id": "A",
            "mode": "serial",
            "brief": "综合改进",
            "children": [
                {"id": "a1", "kind": "self", "brief": "成因分析"},
                {"id": "a2", "member": "devops", "brief": "改进实施"},
            ],
        },
        {"id": "s1", "member": "sim", "brief": "扁平环节"},
    ]
    state = [
        {"id": "A.a1", "status": "done", "evidence": "ev/a1.md"},
        {"id": "A.a2", "status": "queued", "session_id": "sess-a2", "job_id": "job-a2"},
        {"id": "s1", "status": "failed", "task_file": "tb/task.md"},
    ]
    return steps, state


def test_enter_opens_plan_detail_with_head_card(pface):
    _write_plan(pface, "p1", goal="演示目标")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return (
                app.view,
                app.plan_detail_id,
                app.plan_detail_title,
                app.plan_detail_head,
            )

    view, pid, title, head = _run(_go())
    assert view == "plan-detail" and pid == "p1" and "p1" in title
    assert (
        "演示目标" in head
        and "进度：1/2" in head
        and "创建：2026-10-16T10:00:00+08:00" in head
    )


def test_tree_matches_plan_tree_nodes_and_labels(pface):
    steps, state = _tree_steps_plan()
    _write_plan(pface, "p-tree", steps=steps, state_steps=state)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            tree = app.query_one("#plan-detail-tree")
            return (
                tree,
                [str(c.label) for c in tree.root.children],
                str(tree.root.label),
            )

    tree, labels, root_label = _run(_go())
    nodes = plans.plan_tree_nodes("p-tree")
    assert len(tree.root.children) == len(nodes) == 2
    assert labels[0].startswith("〔serial〕")
    assert "A" in labels[0]
    # 叶子：图标＋空格＋id＋[status]＋member｜brief
    assert labels[1].startswith("✗ s1 [failed] sim｜扁平环节")
    # 默认全展开：superstep 的两个子节点在场且 label 形态正确
    subs = [str(c.label) for c in tree.root.children[0].children]
    assert subs[0].startswith("✔ a1 [done]") and "成因分析" in subs[0]
    assert subs[1].startswith("· a2 [queued] devops｜改进实施")
    assert root_label, "根节点有 label（计划 id）"


def test_tree_selection_updates_markdown_panel(pface):
    steps, state = _tree_steps_plan()
    _write_plan(pface, "p-md", goal="联动目标", steps=steps, state_steps=state)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            tree = app.query_one("#plan-detail-tree")
            first_node = tree.cursor_node
            first_md = app.plan_detail_md
            await pilot.press("down")  # 树光标下移 → 选中联动
            await pilot.pause()
            second_md = app.plan_detail_md
            return tree, first_node, first_md, second_md

    tree, first_node, first_md, second_md = _run(_go())
    # 默认选中＝当前进行中环节（A.a2 queued）
    assert str(first_node.label).startswith("· a2 [queued]")
    assert "改进实施" in first_md and "`sess-a2`" in first_md and "`job-a2`" in first_md
    # ↓ 到 s1 → 面板换成该节点详情（含 task_file 留痕与 failed 状态）
    assert second_md != first_md
    assert "# 环节 s1" in second_md and "`tb/task.md`" in second_md
    assert "failed" in second_md
    assert isinstance(tree, tui.MarkedTree)


def test_superstep_node_markdown_lists_children(pface):
    steps, state = _tree_steps_plan()
    _write_plan(pface, "p-super", steps=steps, state_steps=state)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            tree = app.query_one("#plan-detail-tree")
            tree.select_node(tree.root.children[0])  # 选中 superstep A
            await pilot.pause()
            return app.plan_detail_md

    md = _run(_go())
    assert "superstep" in md and "子节点概览" in md
    assert "A.a1" in md and "A.a2" in md


def test_missing_state_and_unknown_plan_do_not_crash(pface):
    # 只有 YAML、没有 state（s1 概览会回落 YAML 取 goal）
    (pface / "p-nostate.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "p-nostate",
                "goal": "无 state 计划",
                "steps": [{"id": "s1", "member": "sim"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            unknown_head, unknown_md = app.plan_detail_head, app.plan_detail_md
            app._open_plan_detail("nope-gone")  # 不存在的 plan：不崩、如实报
            await pilot.pause()
            return app.view, app.plan_detail_head, unknown_head, unknown_md

    view, head_missing, head_ok, md_ok = _run(_go())
    assert view == "plan-detail"
    assert "读数缺失" in head_missing, "不存在的 plan 头卡如实报缺"
    assert isinstance(head_ok, str) and head_ok and isinstance(md_ok, str) and md_ok


# ---------------------------------------------------------------------------
# ③ 性能钉（列表只读 state.json 缓存）
# ---------------------------------------------------------------------------
def test_plan_list_opens_without_parsing_yaml(pface, monkeypatch):
    for i in range(6):
        _write_plan(pface, f"p{i}", created=f"2026-10-16T1{i}:00:00+08:00")

    def boom(*a, **k):
        raise AssertionError("计划列表不得解析 YAML")

    monkeypatch.setattr(plans, "_load_yaml", boom)
    monkeypatch.setattr(plans, "plan_tree_nodes", boom)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            return app.view, len(app.plan_rows)

    view, n = _run(_go())
    assert view == "plans" and n == 6, (
        "列表只走 overview_rows（state.json 缓存），不碰 YAML"
    )


# ---------------------------------------------------------------------------
# ④ 禁语钉（s3 已删数据面；UI 面再钉不出现）
# ---------------------------------------------------------------------------
def test_plans_views_have_no_removed_prompts(pface):
    _write_plan(pface, "p1")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            list_face = "\n".join(_row_texts(app))
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            detail_face = "\n".join(
                [app.plan_detail_head, app.plan_detail_md, app.plan_detail_title]
            )
            return list_face, detail_face

    list_face, detail_face = _run(_go())
    for banned in ("推进：", "多 plan 并行"):
        assert banned not in list_face, banned
        assert banned not in detail_face, banned


def test_plan_detail_md_widget_present(pface):
    _write_plan(pface, "p1")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_plans(app, pilot)
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return app.query_one("#plan-detail-md", Markdown)

    assert isinstance(_run(_go()), Markdown)
