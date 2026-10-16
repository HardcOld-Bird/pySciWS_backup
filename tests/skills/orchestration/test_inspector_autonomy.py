"""inspector 自主巡逻 + 队列入队超时告警（autonomy-fix a2）。

背景（2026-10-14 用户实测）：02:30–07:30 inspector **零巡检**、02:21 入队的 patrol 项
`pending=1` 积压整夜——「闲时自主巡逻」名不副实（inspector 是「orch 命令拉起」式，
无自调度；唤醒链断（a1 已修）+ 无命令 = 永久沉睡）。

纪律：**不造第二套调度/唤醒循环**——自调度与超时告警都挂在 a1 看门狗的 tick 上
（唯一耦合点：``watchdog.run_watchdog`` → ``inspectd.self_patrol_tick`` /
``inspectd.notify_stale_queue``），让路/节流/cooldown/预算闸仍在 inspectd 内部。
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from pysci.skills.orchestration.tools import (
    drain,
    events,
    watch,
    watchdog,
    workflow,
    workq,
)
from pysci.skills.orchestration.tools import (
    inspectd as _insp,
)
from pysci.skills.orchestration.tools.tui.modules import modules


@pytest.fixture
def state(tmp_path, monkeypatch):
    insp = tmp_path / "inspector"
    root = tmp_path / "state"
    monkeypatch.setattr(_insp, "STATE_DIR", insp)
    monkeypatch.setattr(_insp, "FINDINGS_PATH", insp / "findings.jsonl")
    monkeypatch.setattr(_insp, "THROTTLE_PATH", insp / "throttle.json")
    monkeypatch.setattr(_insp, "QUEUE_PATH", insp / "queue.json")
    monkeypatch.setattr(_insp, "EVIDENCE_DIR", insp / "evidence")
    monkeypatch.setattr(_insp, "ENABLE_PATH", root / "inspector.enabled")
    monkeypatch.setattr(
        _insp,
        "INSPECT_PATHS",
        drain.WorkerPaths(
            lock=root / "inspector.lock",
            runs=root / "inspector-runs",
            cancel=root / "inspector.cancel",
        ),
    )
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", root / "events-seen.json")
    monkeypatch.setattr(watch, "WATCH_PATH", root / "watch.json")
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=root / "watchdog.lock",
            runs=root / "watchdog-runs",
            cancel=root / "watchdog.cancel",
        ),
    )
    monkeypatch.setattr(workflow, "BACKLOG_PATH", root / "backlog.json")
    monkeypatch.setattr(workflow, "SUGGESTIONS_DIR", root / "suggestions")
    (root / "suggestions").mkdir(parents=True, exist_ok=True)
    (root / "backlog.json").write_text(
        '{"version": 1, "items": []}\n', encoding="utf-8"
    )
    return root


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    )


# ---------------------------------------------------------------------------
# ① 闲时自调度：组长 5h 零操作 → 仍有 ≥ floor(5h/cooldown) 次自主巡检
# ---------------------------------------------------------------------------
def test_leader_idle_5h_still_patrols(state, monkeypatch):
    """假时钟 + 零 orch 命令：仅靠看门狗 tick 驱动 self_patrol_tick。"""
    cooldown = int(_insp.params()["cooldown_s"])
    tick, span = 60, 5 * 3600
    clock = {"t": time.time()}
    runs: list[float] = []
    last_done: dict[str, float] = {"completed": None}

    monkeypatch.setattr(_insp, "enabled", lambda: True)
    monkeypatch.setattr(
        _insp, "lock_status", lambda paths=None, **kw: {"state": "idle"}
    )
    # 冷却判据走假时钟（模拟「上轮 run 结束于假时钟那一刻」）
    monkeypatch.setattr(
        _insp,
        "last_run_age_s",
        lambda done=None: (
            None
            if last_done["completed"] is None
            else clock["t"] - last_done["completed"]
        ),
    )

    def _simulate_worker(argv, **kw):
        # 消化当前 pending（worker 真跑会 take_next→finish），并把 .done 时刻钉在假时钟
        while workq.take_next(_insp.QUEUE_PATH):
            pass
        last_done["completed"] = clock["t"]

    monkeypatch.setattr(_insp, "_spawn_detached", _simulate_worker)

    for _ in range(span // tick):
        clock["t"] += tick
        if _insp.self_patrol_tick()["spawned"]:
            runs.append(clock["t"])

    assert len(runs) >= span // cooldown, (
        f"5h 内自主巡检 {len(runs)} 次 < floor(5h/cooldown)"
    )
    assert all(runs[i + 1] - runs[i] >= cooldown for i in range(len(runs) - 1)), (
        "自主巡检不得违反 cooldown（不是每个 tick 都烧一发）"
    )


def test_self_patrol_respects_switch_and_running_worker(state, monkeypatch):
    monkeypatch.setattr(_insp, "enabled", lambda: False)
    assert _insp.self_patrol_tick()["reason"] == "全局开关关闭"
    monkeypatch.setattr(_insp, "enabled", lambda: True)
    monkeypatch.setattr(
        _insp, "lock_status", lambda paths=None, **kw: {"state": "running"}
    )
    assert _insp.self_patrol_tick()["reason"] == "worker 在跑"
    # 队列已有 pending → 不重复入队（但会尝试唤醒）
    monkeypatch.setattr(
        _insp, "lock_status", lambda paths=None, **kw: {"state": "idle"}
    )
    _insp.enqueue("patrol", target="code-audit")
    monkeypatch.setattr(_insp, "wake_inspector", lambda **kw: (False, "已拉起"))
    out = _insp.self_patrol_tick()
    assert out["enqueued"] is None and workq.pending_count(_insp.QUEUE_PATH) == 1


def test_watchdog_tick_is_the_coupling_point(state, monkeypatch):
    """耦合点：看门狗 tick 敲 inspector（不造第二套调度循环）。"""
    spawned_inspector: list[bool] = []
    monkeypatch.setattr(_insp, "enabled", lambda: True)
    monkeypatch.setattr(
        _insp, "lock_status", lambda paths=None, **kw: {"state": "idle"}
    )
    monkeypatch.setattr(
        _insp,
        "last_run_age_s",
        lambda done=None: None,  # 无历史 → 无冷却
    )
    monkeypatch.setattr(
        _insp, "_spawn_detached", lambda argv, **kw: spawned_inspector.append(True)
    )
    # 重臂替身：看门狗本身不真去拉 bash（否则会抢真实 token）
    monkeypatch.setattr(watchdog, "_rearm", lambda *a, **k: True)

    clock = {"t": time.time()}
    stops = {"n": 0}

    def fake_sleep(_s):
        stops["n"] += 1
        clock["t"] += 1
        if stops["n"] >= 2:
            monkeypatch.setattr(watchdog, "_consume_stop_request", lambda *a, **k: True)

    # 心跳新鲜 → 看门狗不重臂，但仍敲 inspector tick
    import pathlib

    pathlib.Path(watchdog.WATCHDOG_PATHS.runs).mkdir(parents=True, exist_ok=True)
    watch.save_state({"version": 1, "heartbeat_at": _iso(clock["t"])})
    summary = watchdog.run_watchdog(
        state / "wd.log",
        interval_s=1,
        stale_s=5,
        sleeps=fake_sleep,
        now=lambda: clock["t"],
    )
    assert summary["checks"] >= 1
    assert spawned_inspector, "看门狗 tick 必须驱动 inspector 自调度（耦合点）"


# ---------------------------------------------------------------------------
# ② 队列入队超时告警：事件发出 + 去重 + TUI 读数可见（不许静默积压）
# ---------------------------------------------------------------------------
def test_stale_queue_item_emits_event_and_tui_reading(state):
    it = _insp.enqueue("patrol", target="code-audit")
    # 把入队时刻回拨 35min（> QUEUE_TIMEOUT_S）
    workq.mutate(_insp.QUEUE_PATH, it["id"], enqueued_at=_iso(time.time() - 35 * 60))

    assert [i["id"] for i in _insp.stale_queue_items()] == [it["id"]]

    out = _insp.notify_stale_queue()
    assert [i["id"] for i in out] == [it["id"]]
    evs = [e for e in events.read_all() if e["event"] == "QUEUE-TIMEOUT"]
    assert len(evs) == 1 and evs[0]["subject"] == it["id"]
    assert events.wake_label("QUEUE-TIMEOUT")  # 词表单一源（events.WAKE）
    assert "30min" in events.WAKE["QUEUE-TIMEOUT"]

    assert _insp.notify_stale_queue() == [], (
        "每项只报一次（否则每 60s tick 重报淹掉事件流）"
    )

    q = next(m for m in modules() if m.key == "queues")
    assert any("队列超时" in ln for ln in q.detail()), "队列与运行详情须点名超时项"
    r = next(m for m in modules() if m.key == "review")
    assert not any("超时项" in ln for ln in r.detail()), (
        "s3：inspector 超时明细归队列模块（用户点名从盲审与审查面删除）"
    )


def test_orch_status_shows_inspector_timeout(state, monkeypatch):
    monkeypatch.setattr(_insp, "enabled", lambda: True)  # 关闭态会提前 return，不看队列
    it = _insp.enqueue("patrol", target="code-audit")
    workq.mutate(_insp.QUEUE_PATH, it["id"], enqueued_at=_iso(time.time() - 35 * 60))
    lines = "\n".join(_insp.status_lines())
    assert "队列超时" in lines and it["id"] in lines
