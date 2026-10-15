"""脏跟踪面哨兵（backlog 20261012-080410-devops 方案 C）回归。

实锤形状：运行时自愈把 git 跟踪的 ``orchestration/state/**`` 周期改脏后，组长只在组员
交上来的 ``<blocked>``（「合并前检查发现 main 有未提交改动」）里才发现——一个本该看得见
的事实变成每单都要撞一次的墙。本组钉住搬进观测面（``orch status`` / ``pysci-dev doctor``）
后的三件事：① 解析正确（含改名取新面、引号包裹路径）；② **fail-open**（非仓库/git 缺失/
超时/race 非 0 → 空列表，哨兵失灵不得拖累主流程）；③ 脏时出声且附归因提示，干净时不打扰。
"""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import orch
from pysci.skills.orchestration.tools import registry as reg_mod

REG = "orchestration/state/registry.json"


def _fake_run(monkeypatch, *, rc=0, stdout=""):
    """把 registry 的 spawn 口径（``procs.run_hidden``）换成定值回放（不真起 git）。"""

    def run(cmd, *a, **kw):
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")

    monkeypatch.setattr(reg_mod, "procs", SimpleNamespace(run_hidden=run))


def test_dirty_lines_parsed_is_order_kept(monkeypatch):
    _fake_run(
        monkeypatch,
        stdout=(
            f" M {REG}\n"
            "A  orchestration/state/backlog.json\n"
            "R  orchestration/state/old.json -> orchestration/state/new.json\n"
            ' M "orchestration/state/中文名.json"\n'
            "\n"
        ),
    )
    assert reg_mod.dirty_tracked_state() == [
        REG,
        "orchestration/state/backlog.json",
        "orchestration/state/new.json",
        "orchestration/state/中文名.json",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{"stdout": ""}, {"stdout": "\n"}, {"rc": 128, "stdout": f" M {REG}\n"}],
    ids=["clean", "blank-only", "git-rc-nonzero"],
)
def test_no_dirty_returns_empty(monkeypatch, kwargs):
    _fake_run(monkeypatch, **kwargs)
    assert reg_mod.dirty_tracked_state() == []


def test_git_unavailable_fails_open(monkeypatch):
    """git 不在 PATH / 超时 / 任何异常 → 空列表（观测失灵≠缺陷报警）。"""

    def boom(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(reg_mod, "procs", SimpleNamespace(run_hidden=boom))
    assert reg_mod.dirty_tracked_state() == []


def test_scoped_to_state_path_only(monkeypatch):
    """路径限定在 orchestration/state 下：别处的未提交改动与本哨兵无关。"""
    seen = {}

    def run(cmd, *a, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(reg_mod, "procs", SimpleNamespace(run_hidden=run))
    reg_mod.dirty_tracked_state()
    assert seen["cmd"][-2:] == ["--", reg_mod.STATE_REL_PATH]
    assert "--untracked-files=no" in seen["cmd"]


def test_real_repo_dirty_then_clean(tmp_path, monkeypatch):
    """真 git 仓库阳性：提交后干净、改一个跟踪文件即报、未跟踪文件不报。"""

    def git(*args):
        subprocess.run(
            ["git", *args],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )

    git("init", "-q")
    state = tmp_path / "orchestration" / "state"
    state.mkdir(parents=True)
    (state / "backlog.json").write_text(
        json.dumps({"version": 1, "items": []}), encoding="utf-8"
    )
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed")
    monkeypatch.setattr(reg_mod, "PROJECT_ROOT", tmp_path)
    assert reg_mod.dirty_tracked_state() == []

    (state / "backlog.json").write_text(
        json.dumps({"version": 1, "items": [{"id": "x"}]}), encoding="utf-8"
    )
    (state / "inbox-note.md").write_text("未跟踪", encoding="utf-8")
    assert reg_mod.dirty_tracked_state() == ["orchestration/state/backlog.json"], (
        "跟踪面改动报出；未跟踪文件（--untracked-files=no）不报"
    )


# ---------------------------------------------------------------------------
# 消费面：orch status
# ---------------------------------------------------------------------------
def _status_out(iso_state, monkeypatch, capsys, dirty):
    monkeypatch.setattr(orch._drain, "lock_status", lambda: {"state": "idle"})
    monkeypatch.setattr(orch._drain, "cancel_status", lambda: None)
    monkeypatch.setattr(orch._drain, "latest_done", lambda: None)
    monkeypatch.setattr(orch, "dirty_tracked_state", lambda **kw: list(dirty))
    assert orch.cmd_status(Namespace()) == 0
    return capsys.readouterr().out


def test_status_shows_dirty_face_and_next_hint(
    iso_state, iso_registry, monkeypatch, capsys
):
    out = _status_out(iso_state, monkeypatch, capsys, [REG])
    assert "跟踪 state 脏面" in out
    assert f"[!] {REG}" in out
    assert "组员只能交 <blocked>" in out  # [NEXT] 段也出声


def test_status_quiet_when_tracked_state_clean(
    iso_state, iso_registry, monkeypatch, capsys
):
    out = _status_out(iso_state, monkeypatch, capsys, [])
    assert "脏面" not in out
