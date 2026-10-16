"""TUI 改版 s5（task tui-overhaul-20261016-s5-taskbooks-ui）：任务书分组列表＋Markdown 详情。

钉用户原始诉求（逐条）：

1. 主列表**每组只展示最新一条**（无返工=本体），初始**全部收起**；
2. 单击组行 → 展开/收起（次级行＝各轮返工 + 本体，**从新到旧**、缩进、label 正确）；
3. 双击组行 → 直接进**最新项**详情；`o` 键同效；次级行 Enter → 进该项详情；
4. `a`/`r` 目标解析：组行→最新项、次级行→该项；
5. tb-detail ＝ 头卡 ＋ **Markdown 正文**（delivered 态含「★ 批复参考」）；VSCode 按钮
   成功/失败两分支（polish-b 后：mock ``resolve_vscode_launcher`` / ``subprocess.Popen``，
   解析链单测见 ``test_tui_polish_b.py``）。

行选择一律走 :attr:`PysciTUI.tb_rows` **行模型**（禁裸下标语义猜测）。
"""

from __future__ import annotations

import asyncio

import pytest
import yaml
from textual.widgets import ListView, Markdown, TextArea

from pysci import paths as _paths
from pysci.skills.orchestration.tools import plans, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    taskbooks.TASKBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    # 返工件详情会读**所链计划**（返工卡片/流程报告）——隔离并落一份最简计划。
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    (pdir / "p-demo.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "p-demo",
                "goal": "演示",
                "steps": [{"id": "s1", "member": "lit", "brief": "一"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data


def _family() -> tuple[str, str, str]:
    """造返工两轮的家族：本体 → 返工第 1 轮 → 返工第 2 轮；返回 ``(root, r1, r2)``。"""
    root = taskbooks.create("家族大任务", "本体正文")["id"]
    r1 = taskbooks.create_rework(root, mode="full", note="第一轮批语")["id"]
    r2 = taskbooks.create_rework(r1, mode="full", note="第二轮批语")["id"]
    return root, r1, r2


def _to_delivered(tid: str) -> None:
    taskbooks.link_plan(tid, "p-demo")
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")


async def _open_panel(app, pilot) -> None:
    await pilot.press("t")
    await pilot.pause()


def _row_index(app, *, kind: str, tb_id: str = "", label: str = "") -> int:
    """按**行模型**定位行下标（禁裸下标猜测语义）。"""
    for i, r in enumerate(app.tb_rows):
        if r["kind"] != kind:
            continue
        if tb_id and r["tb_id"] != tb_id:
            continue
        if label and r.get("label") != label:
            continue
        return i
    raise AssertionError(
        f"行不存在：kind={kind} tb_id={tb_id} label={label} / {app.tb_rows}"
    )


# ---------------------------------------------------------------------------
# ① 初始收起：每组只出最新一行
# ---------------------------------------------------------------------------
def test_family_collapsed_to_single_group_row(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            return app.tb_rows, app.tb_expanded, list(app.query("#tb-list > ListItem"))

    rows, expanded, items = _run(_go())
    assert len(rows) == 1 and rows[0]["kind"] == "group"
    assert rows[0]["tb_id"] == r2, "组行＝最新（返工轮次最大者）"
    assert rows[0]["root"] == root and rows[0]["size"] == 3 and rows[0]["round"] == 2
    assert expanded == {}, "初始全部收起"
    assert len(items) == 1, "收起时主列表一屏只见每组最新/本体"


def test_single_taskbook_group_is_one_row(face):
    tid = taskbooks.create("单件任务书", "正文")["id"]

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            return app.tb_rows

    rows = _run(_go())
    assert len(rows) == 1
    assert (
        rows[0]["kind"] == "group" and rows[0]["tb_id"] == tid and rows[0]["size"] == 1
    )


# ---------------------------------------------------------------------------
# ② 展开/收起：次级行从新到旧（组行 Enter；单击走延迟，见 ③）
# ---------------------------------------------------------------------------
def test_enter_expands_then_collapses_group(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("enter")  # 组行 Enter → 立即展开
            await pilot.pause()
            expanded_rows = [
                (r["label"], r["tb_id"], r["indent"])
                for r in app.tb_rows
                if r["kind"] == "sub"
            ]
            n_items = len(app.query("#tb-list > ListItem"))
            await pilot.press("enter")  # 再击 → 收起
            await pilot.pause()
            return expanded_rows, n_items, app.tb_rows, app.tb_expanded

    expanded_rows, n_items, rows, expanded = _run(_go())
    assert [t for t, _, _ in expanded_rows] == ["返工第 2 轮", "返工第 1 轮", "本体"]
    assert [i for _, i, _ in expanded_rows] == [r2, r1, root], "从新到旧、本体殿后"
    assert [ind for _, _, ind in expanded_rows] == [1, 1, 1], "次级行须缩进"
    assert n_items == 4, "组行 + 3 个次级行"
    assert len(rows) == 1 and expanded[root] is False, "再击收起回一行"


# ---------------------------------------------------------------------------
# ③ 单击（延迟切换）/ 双击（直接进最新项）
# ---------------------------------------------------------------------------
def test_single_click_toggles_after_delay(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            await pilot.click(f"#tbrow-{r2}-group")
            await pilot.pause(0.5)  # 越过单击窗口（0.35s）
            return app.tb_rows

    rows = _run(_go())
    assert [r["kind"] for r in rows] == ["group", "sub", "sub", "sub"], "单击展开"


def test_double_click_opens_latest_detail_without_toggling(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            await pilot.click(f"#tbrow-{r2}-group", times=2)
            await pilot.pause(0.5)
            return app.view, app.tb_detail_id, app.tb_rows

    view, tid, rows = _run(_go())
    assert view == "tb-detail" and tid == r2, "双击直接进最新项详情"
    assert [r["kind"] for r in rows] == ["group"], "双击不触发单击的展开"


# ---------------------------------------------------------------------------
# ④ `o` 键 / 次级行 Enter 的目标解析
# ---------------------------------------------------------------------------
def test_o_key_opens_highlighted_target(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("o")
            await pilot.pause()
            first = app.tb_detail_id
            await pilot.press("escape")  # 回列表
            await pilot.pause()
            return first, app.view

    first, back_view = _run(_go())
    assert first == r2, "组行 o → 最新项"
    assert back_view == "taskbooks"


def test_sub_row_enter_opens_that_item(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            lv = app.query_one("#tb-list", ListView)
            lv.index = 0
            await pilot.press("enter")  # 展开
            await pilot.pause()
            lv.index = _row_index(app, kind="sub", label="本体")
            await pilot.press("enter")  # 次级行 Enter → 立即进该项详情
            await pilot.pause()
            return app.tb_detail_id

    assert _run(_go()) == root


# ---------------------------------------------------------------------------
# ⑤ a / r 目标：组行→最新项；次级行→该项（create_rework 挂到该项）
# ---------------------------------------------------------------------------
def test_accept_targets_latest_of_group_row(face):
    root, r1, r2 = _family()
    _to_delivered(r2)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("a")
            await pilot.pause()
            return app.tb_msg

    msg = _run(_go())
    assert taskbooks.load_state(r2)["status"] == "accepted", msg
    assert taskbooks.load_state(r1)["status"] != "accepted"


def test_reject_on_sub_row_hangs_rework_on_that_item(face):
    root, r1, r2 = _family()  # r1 已被 r2 的生成置 rejected（返工链常态）

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            lv = app.query_one("#tb-list", ListView)
            lv.index = 0
            await pilot.press("enter")  # 展开
            await pilot.pause()
            lv.index = _row_index(app, kind="sub", label="返工第 1 轮")
            await pilot.press("r")  # r → 该次级行目标
            await pilot.pause()
            target = app.reject_target
            app.query_one("#tb-note", TextArea).text = "甲批语"
            await app._submit_reject()
            await pilot.pause()
            return target, app.last_rework

    target, new_id = _run(_go())
    assert target == r1, "r 的作用目标＝次级行那一项"
    assert taskbooks.load_state(new_id)["rework_of"] == r1, "返工件挂到该项"


# ---------------------------------------------------------------------------
# ⑥ tb-detail：Markdown 正文（含批复参考段）＋ VSCode 按钮两分支
# ---------------------------------------------------------------------------
def test_tb_detail_body_is_markdown_with_review_reference(face):
    root, r1, r2 = _family()
    _to_delivered(r2)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("o")
            await pilot.pause()
            return app.tb_detail_id, app.tb_detail_body, app.tb_detail_title

    tid, body, title = _run(_go())
    assert tid == r2 and r2 in title
    assert "★ 批复参考" in body, "delivered 态批复参考段"
    assert "## " in body and "# 任务书" in body, "Markdown 结构符（标题层级）"


def test_tb_detail_renders_markdown_widget(face):
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            await pilot.press("o")
            await pilot.pause()
            return app.query_one("#tb-detail-md", Markdown)

    assert isinstance(_run(_go()), Markdown), "tb-detail 正文容器须是 Markdown 组件"


def _vscode_msg(monkeypatch, *, popen, launcher: str, tried=None) -> tuple[str, str]:
    """建家族 → 打开**最新项**详情 → 点 VSCode 按钮 → 返回 ``(最新项 id, 消息行)``。

    polish-b：启动器解析已抽成 :func:`tui.resolve_vscode_launcher`（其单测见
    ``test_tui_polish_b.py``），此处注入其读数 ``(launcher, tried)``，只验按钮→消息行接线。
    """
    root, r1, r2 = _family()
    monkeypatch.setattr(tui.subprocess, "Popen", popen)
    monkeypatch.setattr(
        tui, "resolve_vscode_launcher", lambda **k: (launcher, list(tried or []))
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            await pilot.press("o")
            await pilot.pause()
            await pilot.click("#tb-open-vscode")
            await pilot.pause()
            return r2, app.tb_detail_msg

    return _run(_go())


def test_vscode_button_missing_cli(face, monkeypatch):
    tried = ["code（PATH/PATHEXT）", r"C:\nope（常见安装位）"]
    _tid, msg = _vscode_msg(
        monkeypatch, popen=lambda *a, **k: None, launcher="", tried=tried
    )
    assert msg.startswith("[!] 打开失败"), msg
    assert "已尝试" in msg and r"C:\nope（常见安装位）" in msg, msg


def test_vscode_button_success_uses_cmd_c_for_cmd_launcher(face, monkeypatch):
    calls: list[list[str]] = []
    tid, msg = _vscode_msg(
        monkeypatch,
        popen=lambda cmd, *a, **k: calls.append(list(cmd)),
        launcher="C:/fake/code.cmd",
        tried=["code（PATH/PATHEXT）"],
    )
    assert msg.startswith("[√] 已在 VSCode 打开"), msg
    assert "C:/fake/code.cmd" in msg, "成功消息须显实际启动器路径"
    # .cmd 必须经 cmd /c（直呼 .cmd → WinError 2）
    assert calls and calls[0][:3] == ["cmd", "/c", "C:/fake/code.cmd"], calls
    assert str(taskbooks.md_path(tid)) in calls[0][3], "须打开详情项原文路径"


def test_vscode_button_popen_error(face, monkeypatch):
    def boom(*a, **k):
        raise OSError("no vscode")

    _tid, msg = _vscode_msg(
        monkeypatch, popen=boom, launcher="C:/fake/code.exe", tried=["code"]
    )
    assert msg.startswith("[!] 打开失败"), msg
    assert "C:/fake/code.exe" in msg, "失败消息须带启动器路径"


def test_state_face_matches_rendered_list(face):
    """验收 #4：``tb_rows``/``tb_expanded`` 状态面与界面实际渲染**一致**。"""
    root, r1, r2 = _family()

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            before = (
                len(app.tb_rows),
                len(app.query("#tb-list > ListItem")),
                dict(app.tb_expanded),
            )
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("enter")
            await pilot.pause()
            after = (
                len(app.tb_rows),
                len(app.query("#tb-list > ListItem")),
                dict(app.tb_expanded),
            )
            return before, after

    before, after = _run(_go())
    assert before == (1, 1, {}), "收起：行模型与列表都是一行"
    assert after == (4, 4, {root: True}), "展开：行模型与列表都是 组行+3 次级"
