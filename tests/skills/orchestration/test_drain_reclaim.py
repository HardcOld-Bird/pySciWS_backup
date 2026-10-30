"""drain 启动回收孤儿 in_progress 条目（backlog 20261010-drain-orphan-reclaim）。

场景：worker 在 backlog_take_first 与 backlog_complete 之间被杀，条目卡在
in_progress（take_first 只找 pending → 永久不再被消费）。新 drain 获锁即回收。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import drain, workflow

from .conftest import read_backlog, seed_backlog


def test_reclaim_resets_only_in_progress(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "orph", "status": "in_progress", "started_at": "x"},
            {"id": "pend", "status": "pending"},
            {"id": "dn", "status": "done"},
            {"id": "nl", "status": "needs_leader"},
        ],
    )
    ids = workflow.backlog_reclaim_orphans()
    assert ids == ["orph"]
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["orph"]["status"] == "pending"
    assert "started_at" not in by_id["orph"]
    assert "回收" in by_id["orph"]["note"] and "started_at=x" in by_id["orph"]["note"]
    assert by_id["pend"]["status"] == "pending"
    assert by_id["dn"]["status"] == "done"
    assert by_id["nl"]["status"] == "needs_leader"


def test_reclaim_no_backlog_file(iso_state):
    assert workflow.backlog_reclaim_orphans() == []


def test_drain_startup_reclaims_then_consumes_orphan(iso_state):
    """孤儿条目被回收后立即进入本轮 FIFO 消化（不再永久卡死）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "orph", "status": "in_progress", "started_at": "x"},
            {"id": "pend", "status": "pending", "summary": "s"},
        ],
    )
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log, dry=True)
    assert summary["reclaimed"] == ["orph"]
    assert {i["id"] for i in summary["items"]} == {"orph", "pend"}
    assert all(i["status"] == "done" for i in read_backlog(iso_state))
    assert "回收孤儿" in run_log.read_text(encoding="utf-8")


def test_lock_busy_does_not_reclaim(iso_state, monkeypatch):
    """锁忙（存活 worker 正在处理）时不得回收——那是在途条目不是孤儿。"""
    seed_backlog(iso_state, [{"id": "live", "status": "in_progress"}])
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    summary = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert summary.get("skipped") == "lock_busy"
    assert "reclaimed" not in summary
    assert read_backlog(iso_state)[0]["status"] == "in_progress"


def test_stale_lock_takeover_reclaims(iso_state, monkeypatch):
    """死锁持有者被接管时回收其遗留孤儿（本 bug 的核心场景）。"""
    seed_backlog(iso_state, [{"id": "orph", "status": "in_progress"}])
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 999999, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    summary = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert summary["reclaimed"] == ["orph"]
    assert read_backlog(iso_state)[0]["status"] == "done"
