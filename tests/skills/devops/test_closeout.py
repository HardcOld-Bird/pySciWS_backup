"""devops 收尾面硬化（backlog 20261013-215911-leader）：merge 自动摘面 / worktree clean /
push 留痕。

用户提案：「devops 的常用功能该整理并硬化进它的基础设施；每次工作正常结束后自动提交并 push」。
三处软纪律/半自动的收尾动作收成机械步骤：① merge 成功后自动摘除**已完成**的隔离面（失败/
blocked 保留供抢救）；② `worktree clean` 列/清已合入的遗留、拒绝未合入的；③ `push` 跨平台
超时 + 成败**留痕**（best-effort 但失败不静默）。合并与清理都跑**真 git**（临时仓），因为
本组的病灶正是「隔离面残留」这类 git 状态。
"""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

from pysci.skills.devops.tools import dev


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@e", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )
    return proc.stdout.strip()


def _main_repo(tmp_path: Path) -> Path:
    """真 git 仓（worktrees 与运行态 gitignore，避免 merge 的 cp/自检被无关脏面干扰）。"""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / ".gitignore").write_text("/.qoder/worktrees/\n", encoding="utf-8")
    (main / "a.txt").write_text("base\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    return main


def _bind(monkeypatch, main: Path) -> None:
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    monkeypatch.setattr(dev, "WORKTREES_ROOT", main / ".qoder" / "worktrees")
    # 隔离面外的运行态读数为常量（本组只验 git 面；真读数在 worktree-merge 组已钉）
    monkeypatch.setattr(dev, "_runtime_session_total", lambda *a, **k: 1)
    monkeypatch.setattr(dev, "_cp_ignored_state", lambda *a, **k: [])


def _add_face(main: Path, name: str, *, work: str = "work\n") -> Path:
    """建一个 wt-<name> 隔离面并提交一处改动。"""
    wt = main / ".qoder" / "worktrees" / name
    _git(main, "worktree", "add", "-q", "-b", f"wt-{name}", str(wt))
    (wt / "a.txt").write_text(work, encoding="utf-8")
    _git(wt, "add", "a.txt")
    _git(wt, "commit", "-q", "-m", f"work {name}")
    return wt


def _worktree_names(main: Path) -> str:
    return _git(main, "worktree", "list")


# ---------------------------------------------------------------------------
# ① merge 成功 → 自动摘面；失败/blocked → 保留
# ---------------------------------------------------------------------------
def test_merge_success_removes_face(tmp_path, monkeypatch):
    """建→改→merge→worktree list 不含该面、分支也没了（success_criteria 原文钉法）。"""
    main = _main_repo(tmp_path)
    wt = _add_face(main, "foo")
    _bind(monkeypatch, main)
    assert dev.cmd_worktree_merge("foo", wt) == 0
    assert "foo" not in _worktree_names(main)
    assert _git(main, "branch", "--list", "wt-foo") == ""
    # 改动确已进 main（摘面不丢工作）
    assert (main / "a.txt").read_text(encoding="utf-8") == "work\n"


def test_merge_keep_preserves_face(tmp_path, monkeypatch):
    main = _main_repo(tmp_path)
    wt = _add_face(main, "foo")
    _bind(monkeypatch, main)
    assert dev.cmd_worktree_merge("foo", wt, keep=True) == 0
    assert "foo" in _worktree_names(main)


def test_merge_blocked_path_keeps_face(tmp_path, monkeypatch):
    """交集阻断（rc=2）时面与分支必须保留——那是待抢救/待清面的进度。"""
    main = _main_repo(tmp_path)
    wt = _add_face(main, "foo")
    (main / "a.txt").write_text("main WIP\n", encoding="utf-8")  # 与提交面相交的脏文件
    _bind(monkeypatch, main)
    assert dev.cmd_worktree_merge("foo", wt) == 2
    assert "foo" in _worktree_names(main)
    assert _git(main, "branch", "--list", "wt-foo") != ""


# ---------------------------------------------------------------------------
# ② worktree clean：列/清已合入的、拒绝未合入的（非零退出）
# ---------------------------------------------------------------------------
def _two_faces(tmp_path, monkeypatch) -> Path:
    """一个已合入（done）、一个未合入（pending）的遗留面。"""
    main = _main_repo(tmp_path)
    _add_face(main, "done", work="done work\n")
    _git(main, "merge", "--no-ff", "-q", "-m", "merge done", "wt-done")
    _add_face(main, "pending", work="pending work\n")
    _bind(monkeypatch, main)
    return main


def test_clean_dry_run_lists_and_refuses_unmerged(tmp_path, monkeypatch, capsys):
    main = _two_faces(tmp_path, monkeypatch)
    rc = dev.cmd_worktree(Namespace(action="clean", name=None, apply=False))
    out = capsys.readouterr().out
    assert "done" in out and "pending" in out  # 列名
    assert rc == 2  # 存在未合入 → 非零
    assert "done" in _worktree_names(main)  # dry-run 不真删


def test_clean_apply_removes_merged_keeps_unmerged(tmp_path, monkeypatch):
    main = _two_faces(tmp_path, monkeypatch)
    rc = dev.cmd_worktree(Namespace(action="clean", name=None, apply=True))
    assert rc == 2  # 未合入者仍在 → 非零
    names = _worktree_names(main)
    assert "done" not in names and "pending" in names
    assert _git(main, "branch", "--list", "wt-done") == ""


def test_clean_all_merged_returns_zero(tmp_path, monkeypatch):
    main = _main_repo(tmp_path)
    _add_face(main, "only")
    _git(main, "merge", "--no-ff", "-q", "-m", "m", "wt-only")
    _bind(monkeypatch, main)
    assert dev.cmd_worktree(Namespace(action="clean", name=None, apply=True)) == 0
    assert "only" not in _worktree_names(main)


def test_clean_empty_is_zero(tmp_path, monkeypatch):
    main = _main_repo(tmp_path)
    monkeypatch.setattr(dev, "WORKTREES_ROOT", main / ".qoder" / "worktrees")
    assert dev.cmd_worktree(Namespace(action="clean", name=None, apply=False)) == 0


# ---------------------------------------------------------------------------
# ③ push：跨平台超时 + 成败留痕（失败不静默）
# ---------------------------------------------------------------------------
def _with_remote(tmp_path, monkeypatch, url: str) -> Path:
    main = _main_repo(tmp_path)
    _git(main, "remote", "add", "origin", url)
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    monkeypatch.setattr(dev, "PUSH_LOG_PATH", tmp_path / "push-log.jsonl")
    return main


def _last_trace(tmp_path: Path) -> dict:
    lines = (
        (tmp_path / "push-log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    return json.loads(lines[-1])


def test_push_success_is_traced(tmp_path, monkeypatch, capsys):
    bare = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    _with_remote(tmp_path, monkeypatch, str(bare))
    assert dev.cmd_push(Namespace(remote="origin", branch="main", timeout=25)) == 0
    assert _last_trace(tmp_path)["ok"] is True
    assert "push 成功" in capsys.readouterr().out


def test_push_failure_is_traced_not_silent(tmp_path, monkeypatch, capsys):
    """伪 remote（不存在的裸仓路径）→ push 失败，但**留痕**且出声（成功完成的工作静默地
    没被推送，正是本单要治的「失败静默」）。"""
    _with_remote(tmp_path, monkeypatch, str(tmp_path / "nope.git"))
    rc = dev.cmd_push(Namespace(remote="origin", branch="main", timeout=25))
    assert rc != 0
    assert _last_trace(tmp_path)["ok"] is False
    out = capsys.readouterr().out
    assert "push 失败" in out and "留痕" in out
