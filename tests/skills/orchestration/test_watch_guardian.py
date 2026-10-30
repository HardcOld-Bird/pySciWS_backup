"""看门狗补位（看护模式）与 token 竞争修复回归（wake-repair-20261027）。

2026-10-27 事故实测：会话附着 watch 被补位实例顶掉后只剩**无 harness 附着**的替身——
命中事件时「stderr + exit 2」喊给空气（无人接收），事件永远滞留、组长永远不醒；替身
退出→看门狗再臂→再命中→再退出，死循环空转。本文件钉死三条：

① 看护模式（``orch watch --guardian``，看门狗补位实例专用）命中事件**不退出、不发唤醒**，
   基线**不推进**——事件留持久水位，owner 下次挂载迟到唤醒（唤醒权只属于会话附着实例）；
② 同一事件的重复检出按 ``guard_seen`` 去重（基线不动时每轮都会再检出，无去重即刷屏）；
③ 手动兜底挂载（非 guardian）维持「命中即返回」语义——其唤醒通道=后台任务完成通知，
   且命中但未消费时基线同样不落盘推进（迟到唤醒靠它）。
"""

from __future__ import annotations

import pytest

from pysci.skills.orchestration.tools import events, watch


class Clock:
    """假时钟：sleep 直接推进 monotonic，测试免真等。"""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


@pytest.fixture
def gstate(iso_state, monkeypatch):
    """iso_state 之外补**事件流**隔离（EVENTS_LOG/SEEN_PATH 不在 iso_state 面内）。"""
    monkeypatch.setattr(events, "EVENTS_LOG", iso_state / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", iso_state / "events-seen.json")
    return iso_state


def _seed_late_event(name: str = "job-1") -> None:
    """基线之后落一条生产者直发事件（聋期事件的形状）。"""
    watch.arm()
    events.emit(events.JOB_DONE, name, "s1 交付完成", "组长审阅")


def test_guardian_hit_keeps_watermark_and_reports_once(gstate):
    """看护命中：不退出不发唤醒、基线不推进、重复检出去重、不记消费账。"""
    _seed_late_event()
    out: list[str] = []
    clock = Clock()
    rc = watch.run_watch(
        timeout_s=300,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=out.append,
        queue_watch=False,
        guardian=True,
    )
    assert rc == 0
    text = "\n".join(out)
    assert text.startswith("JOB-DONE job-1"), "命中事件照常呈报"
    assert text.count("JOB-DONE job-1") == 1, "同一事件只留痕一次（guard_seen 去重）"
    assert "看护模式命中" in text
    assert "唤醒组长" not in text
    assert text.startswith("TIMEOUT") is False  # 命中与超时同帧时先呈报命中
    st = watch.load_state()
    assert st["base"]["events_seq"] == 0, "基线不得推进——迟到唤醒的持久水位靠它"
    assert not st.get("consumed"), (
        "看护实例不得记消费账（exit 2 无人接收，记账=吞事件）"
    )
    assert st["guard_seen"], "命中身份入 guard_seen（跨轮去重、跨挂载继承）"
    assert st["last_exit"] == "timeout"


def test_guardian_watermark_serves_owner_late_wakeup(gstate, monkeypatch):
    """看护留下的水位由 owner 下次挂载迟到唤醒并消费（不丢事件）。"""
    _seed_late_event("job-2")
    clock = Clock()
    watch.run_watch(
        timeout_s=60,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=lambda *a: None,
        queue_watch=False,
        guardian=True,
    )
    # owner 会话下次挂载：认领唤醒权、带身份挂载（看护实例恒无身份——watchdog._rearm 剥离）
    watch.save_owner(watch._new_owner("test-owner"))
    monkeypatch.setenv(watch.WATCH_OWNER_ENV, "test-owner")
    out: list[str] = []
    clock2 = Clock()
    rc = watch.run_watch(
        timeout_s=600,
        interval_s=60,
        sleep=clock2.sleep,
        monotonic=clock2.monotonic,
        echo=out.append,
        queue_watch=False,
    )
    assert rc == 0
    assert out[0].startswith("JOB-DONE job-2"), "owner 挂载必须迟到唤醒到看护期事件"
    st = watch.load_state()
    assert st["base"]["events_seq"] == 1, "owner 消费后基线才推进"
    assert st["consumed"], "owner 侧消费入账（同一事件只唤醒一次）"


def test_manual_unowned_hit_returns_and_keeps_watermark(gstate):
    """手动兜底（非 guardian）无主命中：命中即返回（退出=后台通知唤醒），基线不落盘推进。"""
    _seed_late_event("job-3")
    out: list[str] = []
    clock = Clock()
    rc = watch.run_watch(
        timeout_s=600,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=out.append,
        queue_watch=False,
    )
    assert rc == 0
    assert clock.t == 60, "命中即返回（首轮）——手动挂载的唤醒=进程退出通知"
    assert any("非 owner 侧" in line for line in out)
    st = watch.load_state()
    assert st["base"]["events_seq"] == 0, "未消费的命中不得把基线落盘推进"
    assert not st.get("consumed")


def test_guard_seen_carries_across_mounts(gstate):
    """guard_seen 是跨挂载账：arm() 整体覆写 watch.json，不搬就丢（去重退化成刷屏）。"""
    st = watch.arm()
    st["guard_seen"] = {"JOB-DONE|abc": watch._now()}
    watch.save_state(st)
    st2 = watch.arm()
    assert "JOB-DONE|abc" in st2.get("guard_seen", {})
