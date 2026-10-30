"""inspector 常闭态零噪声（用户 2026-10-22 裁决）：关闭态零入队、零唤醒 + 诊断字段。

无头案（insp-20261022-151947-patrol）：toml 开关=false，但条目仍入队且
``created_at/enqueued_at`` 全 None——根因是旧「toml 缺席/读失败回落 legacy true」的
瞬时置真；修后：真源读不出=关（常闭）、关闭态 patrol 滞留不触发 QUEUE-TIMEOUT、
新条目带 ``created_at``/``enabled_at_enqueue`` 诊断字段。
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from pysci.skills.orchestration.tools import (
    events,
    inspectd,
    settings,
    workq,
)


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    )


@pytest.fixture
def closed_state(iso_state):
    """真源 toml 显式写 enabled=false（模拟用户真源；SETTINGS_PATH 已被 iso_state 指向 tmp）。"""
    settings.SETTINGS_PATH.write_text(
        "[inspector]\nenabled = false\n", encoding="utf-8"
    )
    return iso_state


def test_enabled_fails_closed_when_toml_unreadable(iso_state):
    """真源读不出 + legacy "true" 残留 → 仍判关（不复活）。"""
    state = settings.SETTINGS_PATH
    legacy = inspectd.ENABLE_PATH
    legacy.write_text("true\n", encoding="utf-8")
    state.unlink(missing_ok=True)
    assert inspectd.enabled() is False, "toml 缺席必须判关（常闭）"


def test_closed_tick_enqueues_nothing(closed_state, monkeypatch):
    monkeypatch.setattr(inspectd, "lock_status", lambda *a, **k: {"state": "idle"})
    monkeypatch.setattr(inspectd, "last_run_age_s", lambda *a, **k: None)
    monkeypatch.setattr(inspectd, "wake_inspector", lambda **kw: (False, "closed"))
    before = workq.pending_count(inspectd.QUEUE_PATH)
    out = inspectd.self_patrol_tick()
    assert out["reason"] == "全局开关关闭"
    assert workq.pending_count(inspectd.QUEUE_PATH) == before, "关闭态零入队"


def test_stale_patrol_silent_assignment_loud_when_closed(closed_state):
    """关闭态：滞留 patrol 不发 QUEUE-TIMEOUT；组长手工 assignment 照常报。"""
    patrol = inspectd.enqueue("patrol", target="code-walk")
    assignment = inspectd.enqueue("assignment", brief="核查某面")
    workq.mutate(
        inspectd.QUEUE_PATH,
        patrol["id"],
        enqueued_at=_iso(time.time() - 40 * 60),
        timeout_notified_at=None,
    )
    workq.mutate(
        inspectd.QUEUE_PATH,
        assignment["id"],
        enqueued_at=_iso(time.time() - 40 * 60),
        timeout_notified_at=None,
    )

    out = inspectd.notify_stale_queue()
    assert [it["id"] for it in out] == [assignment["id"]], "patrol 项须被豁免"
    evs = [e for e in events.read_all() if e["event"] == "QUEUE-TIMEOUT"]
    assert [e["subject"] for e in evs] == [assignment["id"]]


def test_enqueue_records_created_at_and_switch_reading(closed_state, monkeypatch):
    monkeypatch.setattr(inspectd, "enabled", lambda: False)
    item = inspectd.enqueue("assignment", brief="x")
    assert item["created_at"] and item["created_at"] == item["enqueued_at"]
    assert item["enabled_at_enqueue"] is False

    monkeypatch.setattr(inspectd, "enabled", lambda: True)
    on_item = inspectd.enqueue("patrol", target="y")
    assert on_item["enabled_at_enqueue"] is True


def test_item_epoch_falls_back_to_created_at(closed_state):
    q = closed_state / "inspector" / "queue.json"
    workq.save(
        q,
        [
            {
                "id": "x-1",
                "kind": "patrol",
                "status": "pending",
                "created_at": _iso(time.time() - 90 * 60),
            }
        ],
    )
    it = workq.load(q)[0]
    assert workq.item_epoch(it) is not None, "created_at 应可作入队时刻兜底"
