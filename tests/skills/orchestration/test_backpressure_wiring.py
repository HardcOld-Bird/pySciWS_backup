"""M6：backpressure 写侧接线 + 全队列 defer 延期语义（README §12 硬验收）。

覆盖任务书验收：worker 活跃时 active-sessions 非空、完成清零；长 job 心跳续命不被判死；
带宽失败→成员队列挂 defer_until、到期重取；次数/壁钟闸触顶→停止并上报；devops backlog
与成员队列**两条链都验证**；观察面（mirror/_line_defer / orch status）可读延期读数。

真 LLM 派发一律 mock ``dispatch.do_dispatch``；注册表/队列/事件全部落 tmp 隔离面。
"""

from __future__ import annotations

import time
from argparse import Namespace

import pytest

from pysci.skills.orchestration.tools import (
    backpressure,
    dispatch,
    drain,
    events,
    mirror,
    orch,
    queues,
    queueworker,
    workflow,
    workq,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import drain_once, read_backlog, seed_backlog

MIMO_TEXT = "Custom model service error, please retry later or switch to another model."

#: 假时钟基准（**不能用 0**：Windows 下 ``datetime.fromtimestamp(0)`` 越界报 OSError）。
BASE = 1_700_000_000.0


class _Ok:
    code, kind, body, error, error_text = 0, "result", "ok", "", ""


def _chan_out() -> DispatchOutcome:
    return DispatchOutcome(
        code=2,
        kind="channel_error",
        member="figure",
        body=MIMO_TEXT,
        error=MIMO_TEXT,
        error_text=MIMO_TEXT,
    )


@pytest.fixture
def state(iso_state, monkeypatch):
    """成员队列 / worker 根落 iso_state 面（settings 已隔离，_seed_tier 可读）。"""
    root = iso_state
    monkeypatch.setattr(queues, "QUEUES_DIR", root / "queues")
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", root)
    return root


def _no_wait(monkeypatch):
    """worker 的「睡到到期」在测试里恒返 0（免死循环）；到期重取本身由 take_ready 直测。"""
    monkeypatch.setattr(queues, "next_defer_s", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# ① 写侧接线：登记 / 心跳续命 / 注销
# ---------------------------------------------------------------------------
def test_active_session_registered_during_job_and_cleared_after(state, monkeypatch):
    seen: dict[str, object] = {}

    def fake(member, **kw):
        seen["live"] = backpressure.active_count(now=time.time())
        seen["who"] = [e.get("worker") for e in backpressure.read_registry()]
        return _Ok()

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    queues.enqueue("figure", text="x", root=state)
    queueworker.run_worker("figure", state / "r.log", root=state)

    assert seen["live"] == 1, "job 进行中注册表非空"
    assert seen["who"] == ["figure"]
    assert backpressure.read_registry() == [], "退场注销（清零）"


def test_long_job_heartbeat_keeps_session_alive(state):
    """长 job（>判死阈值）不被误判死：假时钟下心跳续命（测试注入口，免真等）。"""
    path = backpressure.REGISTRY_PATH
    sess = backpressure.LiveSession(
        "figure",
        sid="j1",
        tier="research",
        purpose="t",
        path=path,
        start_thread=False,
        now=lambda: BASE,
    ).start()
    # 700s 无心跳（>STALE_S 600）→ 判死
    assert backpressure.active_count(now=BASE + 700, path=path) == 0
    sess.heartbeat_now(BASE + 700)  # 心跳续命
    assert backpressure.active_count(now=BASE + 701, path=path) == 1
    sess.stop()
    assert backpressure.read_registry(path=path) == []


def test_registry_prune_stale_drops_dead_entries(state):
    path = backpressure.REGISTRY_PATH
    backpressure.note_session(
        "figure", sid="a", tier="research", purpose="t", now=BASE, path=path
    )
    backpressure.note_session(
        "writing", sid="b", tier="research", purpose="t", now=BASE + 1000, path=path
    )
    assert backpressure.prune_stale(now=BASE + 1001, path=path) == 1  # a 死、b 活
    assert [e["worker"] for e in backpressure.read_registry(path=path)] == ["writing"]


# ---------------------------------------------------------------------------
# ② 全队列 defer：成员队列链
# ---------------------------------------------------------------------------
def test_member_bandwidth_failure_defers_and_retakes_on_expiry(state, monkeypatch):
    _no_wait(monkeypatch)
    monkeypatch.setattr(dispatch, "do_dispatch", lambda m, **kw: _chan_out())
    item = queues.enqueue("figure", text="x", root=state)

    queueworker.run_worker("figure", state / "r.log", root=state)

    it = queues.items("figure", root=state)[0]
    assert it["status"] == "pending" and it["defer_until"] > time.time()
    assert it["defer_count"] == 1
    assert queues.take_ready("figure", root=state) is None, "未到期不被取"
    got = queues.take_ready("figure", now=it["defer_until"] + 1, root=state)
    assert got is not None and got["id"] == item["id"], "到期重取"


def test_member_defer_gate_exhausted_stops_and_reports(state, monkeypatch):
    _no_wait(monkeypatch)
    monkeypatch.setattr(dispatch, "do_dispatch", lambda m, **kw: _chan_out())
    item = queues.enqueue("figure", text="x", root=state)
    workq.mutate(
        queues.queue_path("figure", root=state),
        item["id"],
        defer_count=backpressure.DEFER_MAX_ATTEMPTS,  # 次数闸已满
    )
    queues.enqueue("figure", text="y", root=state)  # 第二条不该被吃

    summary = queueworker.run_worker("figure", state / "r.log", root=state)

    assert summary["stopped"] == "channel_saturated", "延期触顶 → 停止本队列并上报"
    by_id = {i["id"]: i for i in queues.items("figure", root=state)}
    assert by_id[item["id"]]["status"] == "failed"
    assert by_id[item["id"]]["result"]["saturated"] is True
    evs = [e for e in events.read_all() if e["event"] == "CHANNEL-STOP"]
    assert evs and item["id"] in evs[0]["subject"], "上报进统一事件层"


# ---------------------------------------------------------------------------
# ③ 全队列 defer：devops backlog 链（既有语义不动，回归确认）
# ---------------------------------------------------------------------------
def test_devops_backlog_chain_defers(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [{"id": "b1", "status": "pending", "summary": "s", "member": "devops"}],
    )
    monkeypatch.setattr(dispatch, "do_dispatch", lambda m, **kw: _chan_out())
    monkeypatch.setattr(workflow, "next_defer_s", lambda **k: 0.0)
    monkeypatch.setattr(drain, "_sleep", lambda s: None)

    drain_once()

    b = read_backlog(iso_state)[0]
    assert b["status"] == "pending" and b["defer_until"] > 0 and b["defer_count"] == 1


# ---------------------------------------------------------------------------
# ④ 观察面：mirror 延期读数（devops + 成员队列合并）+ orch status
# ---------------------------------------------------------------------------
def test_line_defer_reflects_member_queue(state):
    it = queues.enqueue("figure", text="x", root=state)
    queues.defer(
        "figure", it["id"], until=time.time() + 300, count=2, elapsed_s=120, root=state
    )
    line = mirror._line_defer()
    assert "带宽延期" in line and "1" in line, "成员队列的延期要进概览读数"


def test_status_shows_defer_line(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    assert orch.cmd_status(Namespace()) == 0
    assert "带宽延期" in capsys.readouterr().out
