"""drain 锁的获取 / stale 接管 / 释放路径（README §6 自动唤醒的并发防护）。

覆盖任务书验证要求 1：直接调用锁函数模拟获取、存活 worker 拒抢、死进程/超龄接管、
仅释放自有锁。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import drain


def _write_lock(state, pid, started=None, run_log="r.log"):
    state.mkdir(parents=True, exist_ok=True)
    (state / "devops.lock").write_text(
        json.dumps(
            {"pid": pid, "started": started or drain._now(), "run_log": run_log}
        ),
        encoding="utf-8",
    )


def test_acquire_when_idle_then_release(iso_state):
    rec = drain.acquire_lock("state/devops-runs/x.log")
    assert rec is not None
    assert rec["pid"] == os.getpid()
    assert drain.LOCK_PATH.exists()
    # 同进程重复获取：锁已被本进程（存活）持有 → busy
    assert drain.acquire_lock("other.log") is None
    drain.release_lock()
    assert not drain.LOCK_PATH.exists()


def test_live_worker_blocks_second_acquire(iso_state, monkeypatch):
    _write_lock(iso_state, pid=999999)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    assert drain.acquire_lock("b.log") is None
    # 未持锁进程 release 不得删除他人锁（pid 不匹配）
    drain.release_lock()
    assert drain.LOCK_PATH.exists()


def test_stale_takeover_dead_pid(iso_state, monkeypatch):
    _write_lock(iso_state, pid=999999)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)  # 持有者已死
    rec = drain.acquire_lock("b.log")
    assert rec is not None
    assert rec["pid"] == os.getpid()


def test_stale_takeover_over_age(iso_state, monkeypatch):
    old = (
        datetime.now().astimezone() - timedelta(seconds=drain.LOCK_STALE_S + 60)
    ).isoformat(timespec="seconds")
    _write_lock(iso_state, pid=os.getpid(), started=old)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 进程活但锁超龄
    rec = drain.acquire_lock("b.log")
    assert rec is not None  # 超龄 → stale 接管


def test_corrupt_lock_is_taken_over(iso_state):
    iso_state.mkdir(parents=True, exist_ok=True)
    (iso_state / "devops.lock").write_text("{ not json", encoding="utf-8")
    assert drain.read_lock() is None  # 损坏 → 视为无锁
    rec = drain.acquire_lock("b.log")
    assert rec is not None and rec["pid"] == os.getpid()


def test_lock_status_states(iso_state, monkeypatch):
    assert drain.lock_status()["state"] == "idle"
    _write_lock(iso_state, pid=4242)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    st = drain.lock_status()
    assert st["state"] == "running" and st["pid"] == 4242
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    assert drain.lock_status()["state"] == "stale"
