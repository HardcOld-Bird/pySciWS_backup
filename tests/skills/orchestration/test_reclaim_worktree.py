"""孤儿 worktree 清理 + 崩溃信号（backlog 20261010-reclaim-worktree-cleanup）。

死 worker 在 worktree 内中途被杀 → 残留 wt-<id> worktree+branch，重派 worktree add 撞名
失败。本组钉住：take_first 记 worktree 名入条目；reclaim 孤儿后 drain 顺带 force-remove
worktree + branch -D + prune 兜底（只清孤儿名，不碰活跃/手动 worktree）；lock_status 据
「stale 锁有无对应 .done」置 crashed，供 orch status 区分崩溃 vs 干净完成。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch, drain, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog


# ---------------------------------------------------------------------------
# take_first 记名 + 任务书指示
# ---------------------------------------------------------------------------
def test_take_first_records_worktree_name(iso_state):
    seed_backlog(iso_state, [{"id": "wid", "status": "pending"}])
    item = workflow.backlog_take_first()
    assert item["worktree"] == "wid"  # = 条目 id（与 dispatch slug 一致）
    on_disk = read_backlog(iso_state)[0]
    assert on_disk["worktree"] == "wid" and on_disk["status"] == "in_progress"


def test_task_text_instructs_worktree_name():
    text = workflow.backlog_task_text({"id": "20261010-abc", "summary": "s"})
    assert "pysci-dev worktree add 20261010-abc" in text
    assert "wt-20261010-abc" in text


# ---------------------------------------------------------------------------
# cleanup_orphan_worktrees（mock _run_git，不触真实 git）
# ---------------------------------------------------------------------------
def test_cleanup_removes_each_name_and_prunes(iso_state, monkeypatch):
    calls = []
    monkeypatch.setattr(drain, "_run_git", lambda *a: calls.append(a) or (0, ""))
    cleaned = drain.cleanup_orphan_worktrees(["a1", "a2"])
    assert cleaned == ["a1", "a2"]
    assert ("branch", "-D", "wt-a1") in calls
    assert ("branch", "-D", "wt-a2") in calls
    removes = [c for c in calls if c[:3] == ("worktree", "remove", "--force")]
    assert len(removes) == 2
    assert all(r[3].replace("\\", "/").endswith(("/a1", "/a2")) for r in removes)
    assert ("worktree", "prune") in calls  # 兜底清悬挂管理项


def test_cleanup_fail_open_when_nothing_removed(iso_state, monkeypatch):
    """worktree 与 branch 都不存在（git 全非零）→ 不计入 cleaned、不抛。"""
    monkeypatch.setattr(drain, "_run_git", lambda *a: (1, "fatal: not a worktree"))
    assert drain.cleanup_orphan_worktrees(["ghost"]) == []


def test_cleanup_skips_empty_names(iso_state, monkeypatch):
    calls = []
    monkeypatch.setattr(drain, "_run_git", lambda *a: calls.append(a) or (0, ""))
    assert drain.cleanup_orphan_worktrees(["", "x"]) == ["x"]
    assert ("branch", "-D", "wt-x") in calls
    assert not any(c == ("branch", "-D", "wt-") for c in calls)  # 空名被跳过


def test_run_git_fail_open_on_oserror(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git binary")

    monkeypatch.setattr(drain.subprocess, "run", boom)
    rc, out = drain._run_git("status")
    assert rc == 1 and "no git binary" in out


# ---------------------------------------------------------------------------
# drain 集成：非 dry 清理、dry 跳过
# ---------------------------------------------------------------------------
def test_drain_non_dry_cleans_orphan_worktrees(iso_state, monkeypatch):
    seed_backlog(
        iso_state, [{"id": "orph", "status": "in_progress", "started_at": "x"}]
    )
    git_calls = []
    monkeypatch.setattr(drain, "_run_git", lambda *a: git_calls.append(a) or (0, ""))
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body="done commit fff6666"
        ),
    )
    summary = drain.drain_backlog(drain.new_run_log())  # 非 dry
    assert summary["reclaimed"] == ["orph"]
    assert summary["worktrees_cleaned"] == ["orph"]
    assert ("branch", "-D", "wt-orph") in git_calls
    assert ("worktree", "prune") in git_calls


def test_drain_dry_skips_worktree_cleanup(iso_state, monkeypatch):
    seed_backlog(
        iso_state, [{"id": "orph", "status": "in_progress", "started_at": "x"}]
    )
    git_calls = []
    monkeypatch.setattr(drain, "_run_git", lambda *a: git_calls.append(a) or (0, ""))
    summary = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert summary["reclaimed"] == ["orph"]
    assert "worktrees_cleaned" not in summary
    assert git_calls == []  # dry 演练不触碰 git


# ---------------------------------------------------------------------------
# lock_status.crashed（stale 锁有无 .done）
# ---------------------------------------------------------------------------
def _write_stale_lock(iso_state, monkeypatch, tmp_path):
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(drain, "PROJECT_ROOT", tmp_path)
    (iso_state / "devops.lock").write_text(
        json.dumps(
            {
                "pid": 999999,
                "started": drain._now(),
                "run_log": "state/devops-runs/r.log",
            }
        ),
        encoding="utf-8",
    )


def test_lock_status_crashed_when_no_done(iso_state, monkeypatch, tmp_path):
    _write_stale_lock(iso_state, monkeypatch, tmp_path)
    st = drain.lock_status()
    assert st["state"] == "stale" and st["crashed"] is True


def test_lock_status_clean_when_done_exists(iso_state, monkeypatch, tmp_path):
    _write_stale_lock(iso_state, monkeypatch, tmp_path)
    done = tmp_path / "state" / "devops-runs" / "r.done"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("{}", encoding="utf-8")
    st = drain.lock_status()
    assert st["state"] == "stale" and st["crashed"] is False


def test_lock_status_running_has_no_crashed(iso_state, monkeypatch):
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    st = drain.lock_status()
    assert st["state"] == "running" and "crashed" not in st
