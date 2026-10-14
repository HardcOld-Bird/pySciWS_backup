"""backlog 派发并入成员队列（backlog-fifo-20261014）的机械验收面。

- approve → 计划器入队 devops FIFO → worker 消化 → 销账（端到端，mock do_dispatch）；
- 组包批次入队形态（backlog 条目 → 队列条目：kind / 元数据 / 批次正文 / 种子声明）；
- 锁忙不复现「丢弃待拾」：计划器在跑时不 spawn，条目在持久 backlog 里排队；
- 巡更判据精化：有 devops 队列 worker 在跑即 PENDING-WORK 降噪；STALL 叠加心跳判活
  （慢而活不报 / 死而僵报，双向）。
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import (
    backpressure,
    dispatch,
    drain,
    events,
    queues,
    queueworker,
    watch,
    workflow,
    workq,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog, seed_suggestion


def _sync_planner(monkeypatch):
    """把计划器的分离派生换成同步执行（argv 原样，含 --dry 约定）。"""

    def run_planner(argv, **kw):
        ns = Namespace(run_log=argv[argv.index("--run-log") + 1], dry="--dry" in argv)
        drain.cmd_drain_devops(ns)

    monkeypatch.setattr(drain, "_spawn_detached", run_planner)


def _run_worker_once(iso_state):
    wpaths = queueworker.member_paths("devops")
    wpaths.runs.mkdir(parents=True, exist_ok=True)
    return queueworker.run_worker("devops", queueworker.new_run_log(wpaths))


# ---------------------------------------------------------------------------
# ① approve → 入队 → worker 消化 → 销账（端到端，mock do_dispatch）
# ---------------------------------------------------------------------------
def test_approve_to_fifo_settle_end_to_end(iso_state, monkeypatch):
    sid = seed_suggestion(iso_state)
    _sync_planner(monkeypatch)

    rc = workflow.cmd_approve(
        Namespace(suggestion_id=sid, note="采纳", no_wake=False, override="测试自裁")
    )
    assert rc == 0
    # 队列面板见 devops 批次条目（backlog 条目 → 队列条目的形态转换）
    qitems = queues.items("devops")
    assert len(qitems) == 1
    assert qitems[0]["kind"] == drain.BACKLOG_BATCH_KIND
    assert qitems[0]["backlog"]["seed_id"] == sid
    assert qitems[0]["status"] == workq.STATUS_PENDING
    assert read_backlog(iso_state)[-1]["status"] == workq.STATUS_IN_PROGRESS

    # worker 消化（假派发）→ 销账
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda m, **kw: DispatchOutcome(
            code=0, kind="result", body="done commit cafe123"
        ),
    )
    summary = _run_worker_once(iso_state)
    assert summary["count"] == 1
    assert summary["items"][0]["backlog_results"][0]["status"] == "done"
    new_item = next(i for i in read_backlog(iso_state) if i["id"] == sid)
    assert new_item["status"] == workq.STATUS_DONE
    assert queues.items("devops")[0]["status"] == workq.STATUS_DONE
    # 完成唤醒走统一事件流的 QUEUE-EMPTY（执行侧发，backlog-fifo 迁移点）
    names = [str(e.get("event")) for e in events.read_after(0)]
    assert events.QUEUE_EMPTY in names


# ---------------------------------------------------------------------------
# ② 组包批次入队形态
# ---------------------------------------------------------------------------
def test_batch_queue_shape_and_grouping(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "summary": "种子", "member": "devops"},
            {"id": "g1", "status": "pending", "summary": "组包一", "member": "devops"},
        ],
    )
    plan = drain.drain_backlog(drain.new_run_log())
    assert plan["kind"] == "plan" and plan["count"] == 1
    qitems = queues.items("devops")
    assert len(qitems) == 1
    it = qitems[0]
    assert it["kind"] == drain.BACKLOG_BATCH_KIND
    assert it["backlog"]["seed_id"] == "s1"
    assert it["backlog"]["ids"] == ["s1", "g1"]  # 同 member 组包
    assert "backlog id=s1" in it["task"]
    assert "g1" in it["task"]  # 菜单列出组包项
    assert it["effort"] in ("", None)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == workq.STATUS_IN_PROGRESS  # 种子已声明（在途）
    assert (
        by_id["g1"]["status"] == workq.STATUS_PENDING
    )  # 组包项保持 pending（承诺后置）


# ---------------------------------------------------------------------------
# ③ 锁忙不复现「丢弃待拾」：条目在持久 backlog 里排队
# ---------------------------------------------------------------------------
def test_lock_busy_keeps_item_queued_not_dropped(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    _sync_planner(monkeypatch)
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)

    spawned, msg = drain.wake_devops()
    assert spawned is False and "计划器正在跑" in msg
    assert read_backlog(iso_state)[0]["status"] == "pending", (
        "条目留在持久 backlog，不丢"
    )
    assert queues.items("devops") == []

    # 锁释放后下一次唤醒即入队（FIFO 排队而非丢弃待拾）
    (iso_state / "devops.lock").unlink()
    spawned, _ = drain.wake_devops()
    assert spawned is True
    assert queues.items("devops")[0]["backlog"]["seed_id"] == "b1"


# ---------------------------------------------------------------------------
# ④ 巡更降噪：有 devops 队列 worker 在跑即不报 backlog pending
# ---------------------------------------------------------------------------
def test_pending_work_noise_suppressed_when_worker_live(iso_state):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    press = watch.queue_pressure()
    assert any("backlog" in r for r in press["reasons"]), (
        "无 worker：有活没人干，必须报"
    )

    wp = queueworker.member_paths("devops")
    wpaths = drain.acquire_lock("worker-live", wp)
    assert wpaths is not None
    try:
        press = watch.queue_pressure()
        assert not any("backlog" in r for r in press["reasons"]), (
            "devops FIFO worker 在跑：backlog pending 是正常在途，降噪（144 注记场景）"
        )
    finally:
        drain.release_lock(wp.runs / "worker-live.log", wp)


# ---------------------------------------------------------------------------
# ⑤ STALL 判活精化（双向）：慢而活不报 / 死而僵报
# ---------------------------------------------------------------------------
def _old_running_sim_item() -> str:
    """造一个「sim 队列里 in_progress 且 started_at 超期」的条目，返回其 id。"""
    it = queues.enqueue("sim", text="重型环节", slug="heavy")
    qp = queues.queue_path("sim")
    workq.take_next(qp)
    old = (datetime.now().astimezone() - timedelta(hours=3)).isoformat(
        timespec="seconds"
    )
    workq.mutate(qp, str(it["id"]), started_at=old)
    return str(it["id"])


def test_stall_reported_when_member_session_dead(iso_state):
    _old_running_sim_item()
    press = watch.queue_pressure()
    assert any("in_progress" in s for s in press["stall"]), "无活会话：死而僵必须报"


def test_stall_suppressed_when_member_session_live(iso_state):
    _old_running_sim_item()
    sess = backpressure.LiveSession(
        "sim", sid="sim-live", tier="research", start_thread=False
    ).start()
    sess.heartbeat_now()
    try:
        press = watch.queue_pressure()
        assert press["stall"] == [], (
            "心跳新鲜的活会话：慢而不死，不报（2026-14 晚实测误报）"
        )
    finally:
        sess.stop()
    assert any("in_progress" in s for s in watch.queue_pressure()["stall"]), (
        "会话退场后恢复报警（双向）"
    )
