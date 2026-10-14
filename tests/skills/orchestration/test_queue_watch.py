"""队列感知唤醒（adhoc-20261014-124521-s1，用户设计 2026-10-14）。

三件：
① **迟到唤醒双消费缺陷**（生产实锤 12:44）：同一底层事件被两个 watch 实例各唤醒一次
   ——被取代那任仍持**自己的**旧基线，命中时照唤醒。修法 = **单一消费水位**
   （``watch.json`` 的 ``consumed`` 账，读盘为准、跨实例跨挂载共享），命中前先过它。
② ``PENDING-WORK`` / ``STALL`` **伪事件**（条件型，同一 detect→echo→exit 2 链）+ 30min
   节流；队列全空 → 纯 TIMEOUT 循环**零唤醒**（= 手工巡更 cron 自动取消的语义）。
③ 水位/节流**跨挂载持久化**（``arm()`` 整体覆写 watch.json，不搬就丢）。
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    drain,
    events,
    queues,
    queueworker,
    registry,
    taskbooks,
    watch,
    workq,
)


@pytest.fixture
def face(iso_state, monkeypatch):
    """把队列/worker/任务书面也重定向到 tmp（``iso_state`` 只覆盖 backlog/lock/watch/events）。"""
    monkeypatch.setattr(queues, "QUEUES_DIR", iso_state / "queues")
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", iso_state)
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", iso_state / "taskbooks")
    return iso_state


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def _pend(member: str, n: int = 1) -> None:
    for i in range(n):
        queues.enqueue(member, text=f"job {i}")


def _armed_base() -> dict:
    return dict(watch.load_state()["base"])


def _emit_into(state: dict) -> tuple[bool, list[str]]:
    out: list[str] = []
    evs, verdict = watch.detect(state["base"], watch.snapshot())
    return watch._emit(state, evs, out.append), out


# ---------------------------------------------------------------------------
# ① 单一消费水位：同一底层事件只唤醒一次
# ---------------------------------------------------------------------------
def test_same_event_not_woken_twice_across_instances(face):
    """生产回归（2026-10-14 12:44）：两个实例共享消费账 → 第二次不再唤醒。"""
    events.emit(events.JOB_DONE, "figure-1", "figure job 收尾：done", "advance")
    st_a = watch.arm()
    events.emit(
        events.QUEUE_EMPTY, "run-123354", "devops 队列消化完：处理 1 项", "drain"
    )

    # 实例 A：命中 → 记账
    woken, out = _emit_into(st_a)
    assert woken is True
    assert out and out[0].startswith(events.QUEUE_EMPTY)
    # 实例 B（被取代那任，内存基线是旧的）：同一事件仍被它算作「新」
    stale, _ = watch.detect(st_a["base"], watch.snapshot())
    assert [e["event"] for e in stale] == [events.QUEUE_EMPTY]
    # 但单一消费水位挡住第二次唤醒
    assert watch.filter_consumed(stale) == []
    assert watch._emit(st_a, stale, [].append) is False
    assert len(watch.load_state()["fired"]) == 1, "同一事件只允许进 fired 一次"


def test_late_consumption_then_immediate_remount_no_duplicate(face):
    """生产形态：带〔迟到唤醒〕的消费后**立即重挂载** → 零重复唤醒。"""
    watch.arm()  # 首任：种子基线
    events.emit(
        events.QUEUE_EMPTY, "run-123354", "devops 队列消化完：处理 1 项", "drain"
    )
    st = watch.arm()  # 新持有者：继承持久水位 → catchup（迟到）
    assert st["catchup"] is True

    woken, out = _emit_into(st)
    assert woken is True and any("迟到" in ln for ln in out)

    st2 = watch.arm()  # 立即重挂载（Stop 钩子下一次接管）
    again, _ = watch.detect(st2["base"], watch.snapshot())
    assert watch._emit(st2, again, [].append) is False, "重挂载不得重复唤醒同一事件"


def test_consumed_and_throttle_survive_remount(face):
    """重启语义：消费账与节流账跨挂载继承（arm 整体覆写 watch.json，不搬就丢）。"""
    st = watch.arm()
    st["consumed"] = {"QUEUE-EMPTY|#7": "2026-10-14T00:00:00+08:00"}
    st["throttle"] = {events.PENDING_WORK: 123.0}
    watch.save_state(st)

    st2 = watch.arm()
    assert st2["consumed"] == st["consumed"]
    assert st2["throttle"] == st["throttle"]


# ---------------------------------------------------------------------------
# ② PENDING-WORK：队列非空（有活却没人干）
# ---------------------------------------------------------------------------
def test_pending_work_fires_when_queue_waits_without_worker(face):
    watch.arm()
    _pend("figure")
    evs, verdict = watch.detect(_armed_base(), watch.snapshot())
    assert [e["event"] for e in evs] == [events.PENDING_WORK]
    assert "figure 队列 1 项 pending 且无 worker" in evs[0]["detail"]
    assert verdict["throttle"][events.PENDING_WORK] > 0


def test_pending_work_silent_when_worker_live(face):
    """worker 在场时队列有活是常态：不该叫醒组长。"""
    drain.acquire_lock("w", queueworker.member_paths("figure"))
    watch.arm()
    _pend("figure")
    evs, _ = watch.detect(_armed_base(), watch.snapshot())
    assert evs == []


def test_pending_work_silent_when_everything_empty(face):
    """队列全空 → 条件不成立（这是「cron 自动取消」的判据面）。"""
    watch.arm()
    evs, _ = watch.detect(_armed_base(), watch.snapshot())
    assert evs == []
    assert watch.queue_pressure()["pending"] is False


def test_pending_work_fires_on_taskbook_new(face):
    """用户投了单没人拆解 → 也算「有活」。"""
    watch.arm()
    tb_dir = face / "taskbooks"
    tb_dir.mkdir(parents=True, exist_ok=True)
    (tb_dir / "tb-x.state.json").write_text(
        json.dumps(
            {"id": "tb-x", "status": "new", "ts": "2026-10-14T00:00:00+08:00"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    evs, _ = watch.detect(_armed_base(), watch.snapshot())
    assert [e["event"] for e in evs] == [events.PENDING_WORK]
    assert "taskbooks" in evs[0]["detail"]


# ---------------------------------------------------------------------------
# ② PENDING-WORK：节流（同条件 ≤1 次/30min）与窗口过期
# ---------------------------------------------------------------------------
def test_throttle_suppresses_within_window_then_reports_after_expiry(face):
    """节流防抖：窗口内抑制（不重复唤醒）；窗口过后条件仍在 → 再报一次（信号不丢）。"""
    watch.arm()
    _pend("figure")
    evs, verdict = watch.detect(_armed_base(), watch.snapshot())
    assert [e["event"] for e in evs] == [events.PENDING_WORK]
    throttle = dict(verdict["throttle"])
    assert throttle[events.PENDING_WORK] > 0

    # 窗口内：条件仍在（甚至又多一项）→ 抑制，不记新账
    _pend("figure")
    evs2, v2 = watch.detect(_armed_base(), watch.snapshot(), throttle=throttle)
    assert evs2 == [] and v2["throttle"] == {}

    # 窗口过期：同一条件再报一次
    throttle[events.PENDING_WORK] -= watch.PENDING_THROTTLE_S + 1
    evs3, _ = watch.detect(_armed_base(), watch.snapshot(), throttle=throttle)
    assert [e["event"] for e in evs3] == [events.PENDING_WORK]


def test_pending_work_fires_on_mount_when_already_nonempty(face):
    """挂载时队列**已经**非空也要报一次（那正是「没人干」的现场）——条件型判据，非边沿。"""
    _pend("figure")
    out: list[str] = []
    rc = watch.run_watch(
        timeout_s=60,
        interval_s=60,
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
        echo=out.append,
    )
    assert rc == 0
    assert out and out[0].startswith(events.PENDING_WORK), out


# ---------------------------------------------------------------------------
# ② STALL：in_progress 超期
# ---------------------------------------------------------------------------
def test_stall_fires_on_overdue_in_progress(face, monkeypatch):
    item = queues.enqueue("figure", text="job")
    workq.mutate(
        queues.queue_path("figure"),
        item["id"],
        status="in_progress",
        started_at=_iso(1_000_000.0),
    )
    monkeypatch.setattr(
        watch,
        "_watch_cfg",
        lambda: {"pending_throttle_s": 1800, "stall_threshold_s": 3600},
    )
    pressure = watch.queue_pressure(now=1_000_000.0 + 4000)
    assert pressure["stall"], "超期 in_progress 必须报"
    assert "figure" in pressure["stall"][0]


def test_stall_silent_below_threshold(face, monkeypatch):
    item = queues.enqueue("figure", text="job")
    workq.mutate(
        queues.queue_path("figure"),
        item["id"],
        status="in_progress",
        started_at=_iso(1_000_000.0),
    )
    monkeypatch.setattr(
        watch,
        "_watch_cfg",
        lambda: {"pending_throttle_s": 1800, "stall_threshold_s": 3600},
    )
    assert watch.queue_pressure(now=1_000_000.0 + 600)["stall"] == []


# ---------------------------------------------------------------------------
# ② 队列全空 → 纯 TIMEOUT 循环零唤醒
# ---------------------------------------------------------------------------
def test_empty_queues_yield_zero_wakes_over_many_cycles(face, monkeypatch):
    fired: list[object] = []
    monkeypatch.setattr(watch, "_emit", lambda *a, **k: fired.append(a) or True)
    clock = {"t": 1_000_000.0}

    def fake_sleep(seconds):
        clock["t"] += float(seconds)

    rc = watch.run_watch(
        timeout_s=300,
        interval_s=60,
        sleep=fake_sleep,
        monotonic=lambda: clock["t"],
        echo=lambda *_a: None,
    )
    assert rc == 0 and fired == [], "队列全空 → 一个周期都不该唤醒"


# ---------------------------------------------------------------------------
# ③ 阈值可配（registry 顶层 watch 块，逐键 fail-open）
# ---------------------------------------------------------------------------
def test_watch_cfg_reads_registry_block(face, monkeypatch):
    monkeypatch.setattr(
        registry.Registry,
        "load",
        classmethod(
            lambda cls: SimpleNamespace(data={"watch": {"stall_threshold_s": 120}})
        ),
    )
    cfg = watch._watch_cfg()
    assert cfg["stall_threshold_s"] == 120
    assert cfg["pending_throttle_s"] == watch.PENDING_THROTTLE_S


def test_watch_cfg_fail_open_on_broken_block(face, monkeypatch):
    monkeypatch.setattr(
        registry.Registry,
        "load",
        classmethod(lambda cls: SimpleNamespace(data={"watch": "坏"})),
    )
    assert watch._watch_cfg()["stall_threshold_s"] == watch.STALL_THRESHOLD_S
