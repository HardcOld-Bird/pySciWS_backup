"""迟到唤醒**归因细分**（backlog 20261016-215945-devops）。

实测教训（2026-10-16 唤醒链排障）：补报一律打「〔迟到唤醒：发生于挂载前（聋期）〕」，
但 s6/s7 两例事件发生时 **listener 心跳新鲜**（watchdog-runs 19:54/19:59 心跳 age≈11s）——
真成因是「被去重账/水位吞掉、事后补报」，不是没人听。该标注把归因引向「watcher 不在场」，
第一轮判断先怀疑 watch 被杀/看门狗失效（误导实证 30+ 分钟）。

判据：事件时刻**有在岗 listener 心跳**（判据与 `watch.mounted` 同源宽限窗）→ 不得含「聋期」；
读不到心跳（或心跳远离事件）→ 保持原「聋期」措辞（保守，不冒充判断）。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import watch


def _ev(ts: str) -> dict:
    return {
        "ts": ts,
        "event": "JOB-DONE",
        "subject": "figure-1",
        "detail": "figure 完工",
        "action": "advance",
    }


def _state(*, armed: str, prior_hb: str, interval: int = 60) -> dict:
    return {
        "armed_at": armed,
        "prior_heartbeat_at": prior_hb,
        "interval_s": interval,
        "catchup": True,  # 继承上一任水位＝本任首轮（补报形态）
    }


def test_hb_after_event_means_listener_was_up():
    """心跳晚于事件（listener 之后仍在跑）⇒ 事件时刻必有在岗 listener：不得说「聋期」。"""
    ev = _ev("2026-10-16T20:00:00+08:00")
    watch._mark_late(
        [ev],
        _state(armed="2026-10-16T20:10:00+08:00", prior_hb="2026-10-16T20:05:00+08:00"),
    )
    assert ev["late"] is True
    assert "迟到" in ev["detail"], ev["detail"]
    assert "聋期" not in ev["detail"], "有在岗 listener 却说「聋期」＝归因误导"
    assert "补报" in ev["detail"] and "在岗" in ev["detail"]


def test_hb_within_grace_before_event_is_covered():
    """心跳在事件前但落在宽限窗内（interval×3+60s）⇒ 覆盖该时刻。"""
    ev = _ev("2026-10-16T20:00:00+08:00")
    watch._mark_late(
        [ev],
        _state(armed="2026-10-16T20:11:00+08:00", prior_hb="2026-10-16T19:59:00+08:00"),
    )
    assert "聋期" not in ev["detail"], ev["detail"]


def test_stale_hb_far_before_event_keeps_deafness_wording():
    """心跳远在事件之前（窗外）⇒ 维持「聋期」措辞（该时刻确实没人听）。"""
    ev = _ev("2026-10-16T20:00:00+08:00")
    watch._mark_late(
        [ev],
        _state(armed="2026-10-16T20:11:00+08:00", prior_hb="2026-10-16T18:00:00+08:00"),
    )
    assert "聋期" in ev["detail"], ev["detail"]
    assert "迟到" in ev["detail"]


def test_no_hb_keeps_deafness_wording():
    """读不到前任心跳 ⇒ 不冒充判断，维持「聋期」。"""
    ev = _ev("2026-10-16T20:00:00+08:00")
    watch._mark_late([ev], _state(armed="2026-10-16T20:11:00+08:00", prior_hb=""))
    assert "聋期" in ev["detail"], ev["detail"]


def test_ts_less_event_keeps_deafness_wording():
    """差分型事件（无 ts）⇒ 无从判「当时有没有 listener」，维持「聋期」。"""
    ev = {"event": "QUEUE-COMPLETE", "subject": "r", "detail": "d", "action": "a"}
    watch._mark_late(
        [ev],
        _state(armed="2026-10-16T20:11:00+08:00", prior_hb="2026-10-16T20:05:00+08:00"),
    )
    assert ev["late"] is True and "聋期" in ev["detail"]
