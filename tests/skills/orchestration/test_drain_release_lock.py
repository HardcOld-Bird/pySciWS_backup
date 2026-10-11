"""锁释放失败不留黑洞（README §6 锁面；backlog 20261011-225933-devops）。

旧实现 `release_lock` 把 ``os.unlink`` 的 OSError 吞成 ``pass``：Windows 上他方仍持该
文件句柄时 DeleteFile 报共享冲突，锁原地留着自称「pid 活着」，``_lock_live`` 遂把残骸
判为活锁 → 下一任 drain ``skipped=lock_busy``，最长阻塞到 ``LOCK_STALE_S``（6 h）。
本文件锁死新语义：**失败不静默**（run 日志 ``[!]``）+ **失败不挡路**（released 墓碑被
``lock_status`` 判非 running、被 ``acquire_lock`` 就地接管）。
"""

from __future__ import annotations

import json
import os

from pysci.skills.orchestration.tools import drain, orch

from .conftest import seed_backlog

#: 真实共享冲突的形状（Windows：文件被他方读句柄占用，DeleteFile 报 EACCES/32）。
SHARING = OSError(
    13, "The process cannot access the file because it is being used by another process"
)


def _unlink_fails(monkeypatch):
    """让一切 unlink 失败——正是要复现「删不掉的锁」。"""

    def boom(path, *a, **kw):
        raise SHARING

    monkeypatch.setattr(os, "unlink", boom)


def _seed_tombstone(state, *, pid, run_log="orchestration/state/devops-runs/r.log"):
    state.mkdir(parents=True, exist_ok=True)
    (state / "devops.lock").write_text(
        json.dumps(
            {
                "released": True,
                "pid": pid,
                "started": drain._now(),
                "run_log": run_log,
                "released_at": drain._now(),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 阳性三断言（任务书修复面 3）
# ---------------------------------------------------------------------------
def test_unlink_failure_marks_tombstone_and_leaves_trace(iso_state, monkeypatch):
    rec = drain.acquire_lock("orchestration/state/devops-runs/r1.log")
    assert rec is not None
    run_log = iso_state / "devops-runs" / "r1.log"
    _unlink_fails(monkeypatch)

    assert drain.release_lock(run_log) is False

    # (a) run 日志必须有一行 [!]，写明失败原因与已写标记
    text = run_log.read_text(encoding="utf-8")
    assert "[!] 锁释放失败（共享句柄），已写 released 标记" in text
    assert "another process" in text

    # 锁记录被改写为墓碑，且保留溯源字段
    data = drain.read_lock()
    assert data["released"] is True
    assert data["pid"] == rec["pid"] == os.getpid()
    assert data["run_log"] == rec["run_log"]
    assert data["released_at"]


def test_residue_is_not_reported_as_running(iso_state, monkeypatch):
    """(b) 锁面不再骗人：``lock_status`` 不报 running，也不报崩溃。"""
    rec = drain.acquire_lock("orchestration/state/devops-runs/r2.log")
    assert rec is not None
    _unlink_fails(monkeypatch)
    drain.release_lock(iso_state / "devops-runs" / "r2.log")

    st = drain.lock_status()
    assert st["state"] != "running"
    assert st["state"] == "stale"
    assert st["released"] is True
    # 崩溃者走不到 release_lock，故墓碑绝非 crashed（防 watch 误报 CRASH）
    assert st["crashed"] is False


def test_next_drain_takes_over_undeletable_residue(iso_state, monkeypatch):
    """(c) 残骸挡不住下一任：就地改写接管，且接管后不再是墓碑。"""
    _seed_tombstone(iso_state, pid=4242)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 墓碑 pid 恰好「活着」
    assert drain._lock_live(drain.read_lock()) is False

    rec = drain.acquire_lock("orchestration/state/devops-runs/r3.log")
    assert rec is not None and rec["pid"] == os.getpid()
    assert not drain.read_lock().get("released")
    assert drain.lock_status()["state"] == "running"


def test_full_drain_twice_not_blocked_by_residue(iso_state, monkeypatch):
    """端到端：一次 drain 收尾删不掉锁，下一次仍照常消化（不 ``skipped=lock_busy``）。"""
    _unlink_fails(monkeypatch)  # 全程删不动任何文件，锁只能靠墓碑 + 就地接管自愈
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    first = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert first["count"] == 1

    # 首个 run 结束后锁成了删不掉的残骸，但不再自称在跑
    assert drain.LOCK_PATH.exists()
    assert drain.read_lock()["released"] is True
    assert drain.lock_status()["state"] != "running"

    seed_backlog(iso_state, [{"id": "b2", "status": "pending", "summary": "s2"}])
    second = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert second.get("skipped") != "lock_busy"
    assert second["count"] == 1


# ---------------------------------------------------------------------------
# 旧路径：unlink 成功时一切照旧（无标记、无噪音）
# ---------------------------------------------------------------------------
def test_unlink_success_still_removes_lock_quietly(iso_state):
    drain.acquire_lock("orchestration/state/devops-runs/ok.log")
    run_log = iso_state / "devops-runs" / "ok.log"

    assert drain.release_lock(run_log) is True
    assert not drain.LOCK_PATH.exists()
    assert drain.lock_status()["state"] == "idle"
    text = run_log.read_text(encoding="utf-8") if run_log.exists() else ""
    assert "[!]" not in text  # 成功路径一行噪音也不留


def test_non_owner_release_never_touchs_foreign_residue(iso_state, monkeypatch):
    """不是自己的锁就一律不改（含别人的墓碑）——清除归下一任 acquire。"""
    _seed_tombstone(iso_state, pid=999999)
    _unlink_fails(monkeypatch)
    before = drain.read_lock()

    assert drain.release_lock(iso_state / "devops-runs" / "x.log") is False
    assert drain.read_lock() == before


def test_no_run_log_still_marks_but_stays_silent(iso_state, monkeypatch):
    """无 run 日志可写时也必须打标——留痕是尽力，标记是硬要求。"""
    drain.acquire_lock("orchestration/state/devops-runs/s.log")
    _unlink_fails(monkeypatch)

    assert drain.release_lock() is False
    assert drain.read_lock()["released"] is True
    assert drain.lock_status()["state"] != "running"


def test_marker_write_failure_is_also_logged(iso_state, monkeypatch):
    """连墓碑都写不进的极端情形仍不留黑洞：日志说明「标记亦写入失败」。"""
    drain.acquire_lock("orchestration/state/devops-runs/t.log")
    run_log = iso_state / "devops-runs" / "t.log"
    _unlink_fails(monkeypatch)
    monkeypatch.setattr(drain, "_overwrite_lock", lambda payload: False)

    assert drain.release_lock(run_log) is False
    assert "released 标记亦写入失败" in run_log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 观察面：orch status 呈现「残留已标记」
# ---------------------------------------------------------------------------
def test_orch_status_shows_marked_residue(iso_state, iso_registry, capsys):
    _seed_tombstone(iso_state, pid=4242)
    assert drain.lock_status()["released"] is True

    orch.cmd_status(None)
    out = capsys.readouterr().out
    assert "残留已标记" in out
    assert "▶ running" not in out
    assert "⊘ 崩溃" not in out
