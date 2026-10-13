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


# ---------------------------------------------------------------------------
# ④ 分类—动作一致（backlog 20261013-222531-leader）：不许「说可清却删不掉」
# ---------------------------------------------------------------------------
def _merged_then_dirty(tmp_path, monkeypatch) -> Path:
    """一个已合入 main、但盘内有未跟踪文件的温面（正是本单的实形）。"""
    main = _main_repo(tmp_path)
    wt = _add_face(main, "dirty", work="dirty work\n")
    _git(main, "merge", "--no-ff", "-q", "-m", "m", "wt-dirty")
    (wt / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    _bind(monkeypatch, main)
    return main


def _clean(main: Path, *, apply: bool, discard: bool = False) -> int:
    return dev.cmd_worktree(
        Namespace(action="clean", name=None, apply=apply, discard=discard)
    )


def test_clean_refuses_merged_but_dirty_by_default(tmp_path, monkeypatch, capsys):
    """已合入但有脏 → 归「待确认」、默认不删、rc 非 0（分类=拒绝，与动作一致）。"""
    main = _merged_then_dirty(tmp_path, monkeypatch)
    rc = _clean(main, apply=True)
    out = capsys.readouterr().out
    assert "待确认" in out and "dirty" in out
    assert rc != 0
    assert "dirty" in _worktree_names(main)  # 确实没删


def test_clean_force_removes_merged_but_dirty(tmp_path, monkeypatch):
    """--force 才真删脏的已合入面（分类=可清(force)，与动作一致）。"""
    main = _merged_then_dirty(tmp_path, monkeypatch)
    assert _clean(main, apply=True, discard=True) == 0
    assert "dirty" not in _worktree_names(main)


def test_clean_dry_run_never_lies_about_cleanable(tmp_path, monkeypatch, capsys):
    """dry-run 的「可清」必须个个真删得掉——本单的病灶就是这两者不一致。"""
    main = _merged_then_dirty(tmp_path, monkeypatch)
    _clean(main, apply=False)
    out = capsys.readouterr().out
    assert "[√] 可清" not in out  # 唯一的已合入面是脏的 → 不该出现在「可清」桶
    assert "待确认" in out


def test_clean_skips_in_flight_face(tmp_path, monkeypatch, capsys):
    """新建未产出的面（分支 tip == main HEAD）不算「已合入的遗留」——别摘正在作业的面。"""
    main = _main_repo(tmp_path)
    _git(
        main,
        "worktree",
        "add",
        "-q",
        "-b",
        "wt-fresh",
        str(main / ".qoder" / "worktrees" / "fresh"),
    )
    _bind(monkeypatch, main)
    rc = _clean(main, apply=True, discard=True)
    out = capsys.readouterr().out
    assert "在途" in out and "fresh" in out
    assert rc == 0
    assert "fresh" in _worktree_names(main)  # 不动


def test_remove_worktree_keeps_branch_on_failure(tmp_path, monkeypatch, capsys):
    """脏面删不掉时**不**谎报分支已删（此前打「分支删除跳过：used by worktree」）。"""
    main = _merged_then_dirty(tmp_path, monkeypatch)
    wt = main / ".qoder" / "worktrees" / "dirty"
    assert dev._remove_worktree("dirty", wt) != 0
    out = capsys.readouterr().out
    assert "未能摘除" in out and "分支已删除" not in out
    assert _git(main, "branch", "--list", "wt-dirty") != ""  # 分支保留


def test_clean_guidance_names_the_real_flag(tmp_path, monkeypatch, capsys):
    """指引文本必须报**真旗标名** `--discard`——`--force` 已被跨树自检全局占用，
    报错旗标会让组长敲一个空动作（本单自查实测到的文本缺陷）。"""
    main = _merged_then_dirty(tmp_path, monkeypatch)
    _clean(main, apply=False)
    out = capsys.readouterr().out
    assert "--discard" in out and "--force" not in out


# ---------------------------------------------------------------------------
# ⑤ 未注册目录面（backlog 20261013-231756-leader）：clean 不只看 git worktree list
# ---------------------------------------------------------------------------
def _ghost(main: Path, name: str, files: dict[str, str] | None = None) -> Path:
    """造一个**未注册**的陈旧目录（不经 `git worktree add`，故 git/prune 都够不着）。"""
    d = main / ".qoder" / "worktrees" / name
    d.mkdir(parents=True)
    for rel, text in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


def test_clean_lists_and_clears_empty_unregistered_dir(tmp_path, monkeypatch, capsys):
    """空未注册目录：被 clean 列出（不依赖 git worktree list）且 `--apply` 可清。"""
    main = _main_repo(tmp_path)
    _ghost(main, "ghost-empty")
    _bind(monkeypatch, main)

    assert _clean(main, apply=False) == 0
    out = capsys.readouterr().out
    assert "ghost-empty" in out and "未注册" in out

    assert _clean(main, apply=True) == 0
    assert not (main / ".qoder" / "worktrees" / "ghost-empty").exists()


def test_clean_unregistered_with_content_confirms_and_surfaces_artifacts(
    tmp_path, monkeypatch, capsys
):
    """有内容的未注册目录 → 待确认 + 顶出**未记录产物清单**；`--discard` 才删。"""
    main = _main_repo(tmp_path)
    _ghost(main, "stale", {"orchestration/state/suggestions/LOST-1.md": "# 遗失工单\n"})
    _bind(monkeypatch, main)

    _clean(main, apply=False)
    out = capsys.readouterr().out
    assert "未注册" in out and "stale" in out
    assert "orchestration/state/suggestions/LOST-1.md" in out  # 未记录产物被顶出

    assert _clean(main, apply=True) != 0  # 无 --discard → 拒绝、rc 非 0
    assert (main / ".qoder" / "worktrees" / "stale").exists()

    assert _clean(main, apply=True, discard=True) == 0
    assert not (main / ".qoder" / "worktrees" / "stale").exists()


def test_unregistered_artifact_detection_ignores_recorded_files(tmp_path, monkeypatch):
    """已进 main 的建议文件不算「未记录」——不该误报（判据＝main 树里同名不存在）。"""
    main = _main_repo(tmp_path)
    (main / "orchestration" / "state" / "suggestions").mkdir(parents=True)
    (main / "orchestration" / "state" / "suggestions" / "KNOWN.md").write_text(
        "x\n", encoding="utf-8"
    )
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "known suggestion")
    _ghost(main, "stale", {"orchestration/state/suggestions/KNOWN.md": "x\n"})
    _bind(monkeypatch, main)
    assert dev._undocumented_artifacts(main / ".qoder" / "worktrees" / "stale") == []
