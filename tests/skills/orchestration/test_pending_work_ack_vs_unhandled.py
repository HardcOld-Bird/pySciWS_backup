"""PENDING-WORK 报警：区分「已处理未销账」与「真没人处理」（backlog 20261017-001500）。

现象（实证通宵反复）：消费账是两段式——watch 唤醒（叫人）≠ `events drain` 销账（盖章）。
组长把事件**处理完毕**却没盖章时，30 分钟后巡更照报「有活却没人干：统一事件流 1 条超
30min 未排空」＝**错怪**，还逼出一轮纯记账性唤醒。

修法：对超期的 ``JOB-DONE``/``STEP-REVIEW-DONE`` 查**旁证**（subject 对应队列条目已终态／
交付留档在盘）——有旁证 → 「已处理疑未销账」降级为 **status 注记**（不唤醒）；无旁证或
类型无旁证可查（如 ``NEEDS-LEADER``）→ 维持唤醒（兜底防漏唤）。
"""

from __future__ import annotations

import time

from pysci.skills.orchestration.tools import (
    events,
    queues,
    queueworker,
    watch,
    workq,
)


def _done_item(member: str = "figure") -> str:
    """造一件已落终态的队列条目，返回其 id。"""
    it = queues.enqueue(member, text="job")
    iid = str(it["id"])
    workq.finish(
        queues.queue_path(member), iid, workq.STATUS_DONE, {"code": 0, "kind": "result"}
    )
    return iid


def _future_clock(monkeypatch, *, ahead_s: int = 3600) -> None:
    """本帧按「未来时刻」读队列压力：让刚发的事件也算**超期未排空**（不睡等 30min）。"""
    real = watch.queue_pressure
    monkeypatch.setattr(
        watch, "queue_pressure", lambda **k: real(now=time.time() + ahead_s)
    )


def _stale_texts(press: dict) -> tuple[list[str], list[str]]:
    """(真积压理由里的「未排空」行, 注记里的「疑未销账」行)。"""
    reason = [r for r in press["reasons"] if "未排空" in r]
    note = [n for n in press["notes"] if "已处理疑未销账" in n]
    return reason, note


# ---------------------------------------------------------------------------
# ① 已处理未销账 → 不唤醒（降级 status 注记）
# ---------------------------------------------------------------------------
def test_handled_but_unacked_is_downgraded_to_note(iso_state, monkeypatch):
    iid = _done_item()
    events.emit(events.JOB_DONE, subject=iid, detail="figure job 收尾：done（result）")
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is False, press["reasons"]
    reason, note = _stale_texts(press)
    assert reason == [], "有旁证就不该进「真积压」理由"
    assert note and "请 `orch events drain` 盖章" in note[0], press["notes"]
    assert iid in note[0], note


def test_handled_event_emits_no_pending_work(iso_state, monkeypatch):
    """唤醒面：`detect` 不产出 PENDING-WORK 伪事件。"""
    iid = _done_item()
    events.emit(events.JOB_DONE, subject=iid, detail="收尾 done")
    _future_clock(monkeypatch)
    watch.arm()

    evs, _verdict = watch.detect(
        watch.load_state()["base"], watch.snapshot(), queue_watch=True
    )

    assert [e["event"] for e in evs] == [], evs


def test_delivery_archive_is_evidence_too(iso_state, monkeypatch):
    """旁证②：条目已被裁剪，但交付留档还在盘 → 同样判「疑已处理」。"""
    orphan_id = "figure-20260101-000000-orphan"
    (queueworker.DELIVERIES_DIR).mkdir(parents=True, exist_ok=True)
    (queueworker.DELIVERIES_DIR / f"{orphan_id}.md").write_text(
        "# 交付全文留档\n", encoding="utf-8"
    )
    events.emit(events.JOB_DONE, subject=orphan_id, detail="收尾 done")
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is False, press["reasons"]
    _reason, note = _stale_texts(press)
    assert note and "交付留档已在盘" in note[0], press["notes"]


# ---------------------------------------------------------------------------
# ② 真未处理 → 照常唤醒（文案仍是「有活却没人干」）
# ---------------------------------------------------------------------------
def test_stale_job_done_without_evidence_still_wakes(iso_state, monkeypatch):
    events.emit(
        events.JOB_DONE, subject="figure-查无此条", detail="figure job 收尾：done"
    )
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is True
    reason, note = _stale_texts(press)
    assert reason == ["统一事件流 1 条超 30min 未排空"], press["reasons"]
    assert note == [], press["notes"]


def test_stale_job_done_for_still_pending_item_still_wakes(iso_state, monkeypatch):
    """条目**还在队里排队**（没有旁证）→ 维持唤醒（别把没干的活静默掉）。"""
    it = queues.enqueue("figure", text="job")
    events.emit(events.JOB_DONE, subject=str(it["id"]), detail="收尾 done")
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is True
    reason, _note = _stale_texts(press)
    assert len(reason) == 1, press["reasons"]


def test_no_evidence_channel_types_still_wake(iso_state, monkeypatch):
    """NEEDS-LEADER 属人工事项、无旁证可查 → 一律维持唤醒。"""
    events.emit(
        events.NEEDS_LEADER, subject="backlog-x", detail="条目需裁决", action="处置"
    )
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is True
    reason, _note = _stale_texts(press)
    assert reason == ["统一事件流 1 条超 30min 未排空"], press["reasons"]


def test_pending_work_detail_keeps_the_true_backlog_wording(iso_state, monkeypatch):
    """「有活却没人干」只留给真积压（伪事件 detail 前缀不变）。"""
    events.emit(events.NEEDS_LEADER, subject="backlog-x", detail="需裁决")
    _future_clock(monkeypatch)
    watch.arm()

    evs, _ = watch.detect(
        watch.load_state()["base"], watch.snapshot(), queue_watch=True
    )

    pw = [e for e in evs if e["event"] == events.PENDING_WORK]
    assert pw and pw[0]["detail"].startswith("有活却没人干："), evs


# ---------------------------------------------------------------------------
# ③ 两类同刻并存 → 各归各位
# ---------------------------------------------------------------------------
def test_mixed_stale_events_partition_correctly(iso_state, monkeypatch):
    iid = _done_item()
    events.emit(events.JOB_DONE, subject=iid, detail="收尾 done")
    events.emit(events.NEEDS_LEADER, subject="backlog-y", detail="需裁决")
    _future_clock(monkeypatch)

    press = watch.queue_pressure()

    assert press["pending"] is True, "并存时真积压仍要唤醒"
    reason, note = _stale_texts(press)
    assert reason == ["统一事件流 1 条超 30min 未排空"], press["reasons"]
    assert len(note) == 1 and "1 条**已处理疑未销账**" in note[0], press["notes"]
    assert iid in note[0], note


# ---------------------------------------------------------------------------
# ④ 注记仍可见（`orch status` 的 [i] 行读的就是 notes 通道）
# ---------------------------------------------------------------------------
def test_note_is_visible_on_status_channel(iso_state, monkeypatch, capsys):
    from argparse import Namespace

    from pysci.skills.orchestration.tools import orch

    iid = _done_item()
    events.emit(events.JOB_DONE, subject=iid, detail="收尾 done")
    _future_clock(monkeypatch)

    orch.cmd_status(
        Namespace(id="", limit=5, days=0, stats=False, tail=10, member=None)
    )
    out = capsys.readouterr().out

    assert "[i] 统一事件流 1 条**已处理疑未销账**" in out, out
    assert "有活却没人干" not in out, "已处理未销账不该打成真积压"
