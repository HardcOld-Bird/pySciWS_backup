"""TUI polish-b（task-20261016-224655-tui-polish-20261016-polish-b）：行为两件。

① **VSCode 启动器稳健化**（用户实测 `WinError 2`）：Windows 的 `code` 实为 `code.cmd`，
   `subprocess.Popen` 直呼 `.cmd` 不经 cmd 解释器 → WinError 2。修法＝**多级解析链**
   （`which`→PATH 目录扫描→常见安装位兜底）＋ **`.cmd`/`.bat` 经 `cmd /c`**＋失败消息带
   **尝试过的路径清单**、成功消息显**实际启动器路径**。此文件验**纯函数分支**（不真启动）。
② **Esc 路由**：详情页进驳回面板 → Esc 回**详情页**；列表进驳回面板 → Esc 回**列表**；
   新增面板 → Esc 回**列表**；提交成功去向不变（回列表）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from pysci import paths as _paths
from pysci.skills.orchestration.tools import plans, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui

PLAN_ID = "p-demo"


# ---------------------------------------------------------------------------
# ① 启动器解析链（纯函数，注入 which/env/is_file）
# ---------------------------------------------------------------------------
def _norm(p: object) -> str:
    return str(p).replace("\\", "/").rstrip("/").lower()


def test_resolve_which_hit_returns_full_path():
    """① ``which("code")`` 命中（PATHEXT 展开出 `.CMD`）→ 直接取该绝对路径。"""
    launcher, tried = tui.resolve_vscode_launcher(
        which=lambda name: (
            r"D:\XiGPrograms\vsCode\base\bin\code.CMD" if name == "code" else None
        ),
        env={"PATH": ""},
        is_file=lambda p: False,
    )
    assert _norm(launcher) == "d:/xigprograms/vscode/base/bin/code.cmd", launcher
    assert tried == ["code（PATH/PATHEXT）"], tried


def test_resolve_path_scan_finds_cmd():
    """② ``which`` 落空 → 逐个 PATH 目录扫 `code.cmd/.exe/.bat`。"""
    target = "c:/vs/bin/code.cmd"
    launcher, tried = tui.resolve_vscode_launcher(
        which=lambda name: None,
        env={"PATH": r"C:\vs\bin;C:\other"},
        is_file=lambda p: _norm(p) == target,
    )
    assert _norm(launcher) == target, launcher
    assert any("PATH 目录扫描" in t for t in tried), tried
    assert any("vs/bin" in _norm(t) for t in tried), tried


def test_resolve_fallback_dir_hit():
    """③ PATH 扫描也落空 → 常见安装位兜底（`%LOCALAPPDATA%\\Programs\\...`）。"""
    launcher, tried = tui.resolve_vscode_launcher(
        which=lambda name: None,
        env={"PATH": "", "LOCALAPPDATA": r"C:\Users\x\AppData\Local"},
        is_file=lambda p: _norm(p).endswith("microsoft vs code/bin/code.cmd"),
    )
    assert "microsoft vs code/bin/code.cmd" in _norm(launcher), launcher
    assert any("常见安装位" in t for t in tried), tried


def test_resolve_nonstandard_d_drive_fallback():
    """③b 用户实测非标准安装位（D 盘自定目录）在兜底链里。"""
    launcher, _tried = tui.resolve_vscode_launcher(
        which=lambda name: None,
        env={"PATH": ""},
        is_file=lambda p: _norm(p).endswith("xigprograms/vscode/base/bin/code.exe"),
    )
    assert _norm(launcher) == "d:/xigprograms/vscode/base/bin/code.exe", launcher


def test_resolve_all_missing_lists_tried():
    """全链落空 → 回 ``("", 清单)``，清单**非空**且含候选名与兜底位（自诊断面）。"""
    launcher, tried = tui.resolve_vscode_launcher(
        which=lambda name: None,
        env={
            "PATH": r"C:\a;C:\b",
            "LOCALAPPDATA": r"C:\Users\x\AppData\Local",
            "ProgramFiles": r"C:\Program Files",
        },
        is_file=lambda p: False,
    )
    assert launcher == "", launcher
    assert "code（PATH/PATHEXT）" in tried and "code.cmd（PATH/PATHEXT）" in tried, (
        tried
    )
    assert any("PATH 目录扫描" in t for t in tried), tried
    assert any("常见安装位" in t for t in tried), tried


def test_resolve_which_exception_treated_as_miss():
    """``which`` 抛异常按**未命中**处理（观察面不炸），后续候选照常接力。"""

    def which(name: str):
        if name == "code":
            raise OSError("boom")
        if name == "code.cmd":
            return r"C:\x\code.cmd"
        return None

    launcher, tried = tui.resolve_vscode_launcher(which=which, env={"PATH": ""})
    assert _norm(launcher) == "c:/x/code.cmd", launcher
    assert tried == ["code（PATH/PATHEXT）", "code.cmd（PATH/PATHEXT）"], tried


# ---------------------------------------------------------------------------
# ① launch_editor：`.cmd`/`.bat` 经 cmd /c，`.exe` 直呼
# ---------------------------------------------------------------------------
def test_launch_editor_cmd_goes_through_cmd_c():
    calls: list[list[str]] = []
    ok, info = tui.launch_editor(
        r"C:\fake\code.cmd", Path(r"C:\tb\x.md"), popen=calls.append
    )
    assert ok is True and info == " ".join(calls[0])
    assert calls[0][:3] == ["cmd", "/c", r"C:\fake\code.cmd"], calls
    assert calls[0][3] == r"C:\tb\x.md", calls


def test_launch_editor_bat_uppercase_goes_through_cmd_c():
    calls: list[list[str]] = []
    tui.launch_editor(r"C:\fake\Code.BAT", Path("t.md"), popen=calls.append)
    assert calls[0][0] == "cmd" and calls[0][1] == "/c", calls


def test_launch_editor_exe_called_directly():
    calls: list[list[str]] = []
    tui.launch_editor(r"C:\fake\Code.EXE", Path("t.md"), popen=calls.append)
    assert calls[0] == [r"C:\fake\Code.EXE", "t.md"], calls


def test_launch_editor_reports_spawn_error():
    def boom(*a, **k):
        raise OSError("nope")

    ok, info = tui.launch_editor(r"C:\fake\code.exe", Path("t.md"), popen=boom)
    assert ok is False and info.startswith("OSError:"), (ok, info)


# ---------------------------------------------------------------------------
# ② Esc 路由（进入前视图）
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    taskbooks.TASKBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    (pdir / f"{PLAN_ID}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": PLAN_ID,
                "goal": "演示",
                "steps": [{"id": "s1", "member": "lit", "brief": "一"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data


def _mk_delivered(title: str) -> str:
    tid = taskbooks.create(title, f"{title} 正文")["id"]
    taskbooks.link_plan(tid, PLAN_ID)
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")
    return tid


def test_esc_from_reject_panel_returns_to_detail(face):
    """详情页进驳回面板 → Esc → **回详情页**（用户点名）。"""
    tid = _mk_delivered("详情页驳件")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("o")  # 组行 → 最新项详情
            await pilot.pause()
            assert app.view == "tb-detail" and app.tb_detail_id == tid
            await pilot.press("r")  # 详情页内进驳回面板
            await pilot.pause()
            assert app.view == "tb-reject", app.view
            await pilot.press("escape")
            await pilot.pause()
            return (
                app.view,
                app.tb_detail_id,
                app.query_one("#tb-reject", tui.Vertical).display,
            )

    view, detail_id, panel_shown = _run(_go())
    assert view == "tb-detail" and detail_id == tid, (view, detail_id)
    assert panel_shown is False


def test_esc_from_reject_panel_entered_via_list_returns_to_list(face):
    """列表进驳回面板 → Esc → **回任务书列表**（非主菜单）。"""
    _mk_delivered("列表驳件")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("r")  # 列表内进驳回面板
            await pilot.pause()
            assert app.view == "tb-reject", app.view
            await pilot.press("escape")
            await pilot.pause()
            return app.view

    assert _run(_go()) == "taskbooks"


def test_esc_from_tb_new_returns_to_taskbooks(face):
    """新增面板 → Esc → **回任务书列表**（非主菜单）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert app.view == "tb-new", app.view
            await pilot.press("escape")
            await pilot.pause()
            return app.view

    assert _run(_go()) == "taskbooks"


def test_reject_submit_from_detail_still_returns_to_list(face):
    """a/r 提交成功后的去向**维持现状**（回列表重建），不受 Esc 路由改动影响。"""
    _mk_delivered("提交驳回件")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            app.query_one("#tb-note", tui.TextArea).text = "批语"
            await pilot.pause()
            await app._submit_reject()
            await pilot.pause()
            return app.view, app.last_rework

    view, new_id = _run(_go())
    assert new_id
    assert view == "taskbooks", view
