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


# ---------------------------------------------------------------------------
# 损坏锁的判读语义对齐（backlog insp-20261013-045910-mechanical，remedy_shape=restore）
# ---------------------------------------------------------------------------
def test_unknown_age_with_live_pid_reads_running_not_crashed(iso_state, monkeypatch):
    """`started` 不可解析 + 持有者**存活** → running，且不得同时报 crashed。

    修复前这里报 (stale, crashed=True, _lock_live=True)：观察面宣告「崩溃残骸、下一任
    可接管」，锁面却因判活拒绝起锁——操作员照观察面去清锁/杀进程，杀的正是活 worker。
    按 :func:`_lock_age_s` 的既定语义（未知 ≠ 超龄），未知龄不得单独构成 stale。
    """
    _write_lock(iso_state, pid=4242, started="not-a-date")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)

    st = drain.lock_status()
    live = drain._lock_live(drain.read_lock(drain._devops_paths()))

    assert (st["state"], st.get("crashed"), live) != ("stale", True, True), (
        "这条组合就是本单要修的矛盾（观察面报崩溃残骸、锁面判活拒锁）"
    )
    assert st["state"] == "running", (
        "未知龄＋存活进程按 running 报（判不出龄≠判得出死）"
    )
    assert st.get("crashed") is None and live is True


def test_unknown_age_with_dead_pid_still_stale_and_crashed(iso_state, monkeypatch):
    """另一方向不变：未知龄 + 持有者**已死** → 仍报 stale+crashed，且此时 `_lock_live`
    为假——「真残骸被宣告崩溃」是正确读数，不构成矛盾。"""
    _write_lock(iso_state, pid=4242, started="not-a-date")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)

    st = drain.lock_status()
    live = drain._lock_live(drain.read_lock(drain._devops_paths()))

    assert st["state"] == "stale" and st.get("crashed") is True
    assert live is False, "死进程的锁不是活锁——与观察面一致"


def test_lock_live_true_implies_running_across_matrix(iso_state, monkeypatch):
    """**判据的不变量**（success_criteria 原文）：`_lock_live` 为 True 时 state 必为
    running、crashed 不得为 True。跨「锁龄 × 存活」四格逐一验，防只修了样例那一格。"""
    for started, alive in (
        ("not-a-date", True),  # 未知龄 + 活（本单的 bug 格）
        (drain._now(), True),  # 正常龄 + 活
        ("not-a-date", False),  # 未知龄 + 死
        (drain._now(), False),  # 正常龄 + 死
    ):
        _write_lock(iso_state, pid=4242, started=started)
        monkeypatch.setattr(drain, "_pid_alive", lambda pid, _a=alive: _a)
        st = drain.lock_status()
        if drain._lock_live(drain.read_lock(drain._devops_paths())):
            assert st["state"] == "running" and not st.get("crashed"), (
                f"活锁却被读成 {st['state']}/crashed={st.get('crashed')}（started={started!r}）"
            )


def test_bencher_paths_share_the_fixed_semantics(iso_state, monkeypatch):
    """bencher 经 ``BENCH_PATHS`` 复用同一套判读——修在 drain 一处，双面同愈。"""
    from pysci.skills.orchestration.tools import benchd

    paths = benchd.BENCH_PATHS
    paths.lock.parent.mkdir(parents=True, exist_ok=True)
    paths.lock.write_text(
        json.dumps({"pid": 4242, "started": "not-a-date", "run_log": "r.log"}),
        encoding="utf-8",
    )
    try:
        monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
        st = drain.lock_status(paths)
        assert st["state"] == "running" and not st.get("crashed")
    finally:
        paths.lock.unlink(missing_ok=True)
