"""唤醒链抗回卷（backlog 20261016-203600-devops）：merge 覆盖事件流＋seq 身份撞车致静默漏报。

**实锤事故**（s6 收尾组长零唤醒，靠人工 `orch status` 才发现）：worktree merge 把 main 的
``events.jsonl`` 从 45 条覆盖成 6 条（**seq 回卷**），消费账里 `JOB-DONE|#6`（旧流身份）与
回卷后的新 `#6` 撞车 → ``filter_consumed`` 静默吞掉真事件；``events-seen.json`` 水位 51 对
新流无意义 → ``unacked()`` 恒空（「待排空 0」假绿、PENDING-WORK 判据失效）。

三层防线各自钉住（每层都能独立止血）：
1. **同步面**（devops）：唤醒链文件**永不跨树 cp**——根因一（见
   ``tests/skills/devops/test_worktree_merge.py::test_cp_never_moves_wake_chain_files``）；
2. **身份面**（watch）：流事件身份＝**内容身份**，抗 seq 回卷（根因二）；
3. **水位面**（events）：``ack_seq`` 检测回卷即**自愈**并显式点名，不许假绿。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import events, watch


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离事件流 / 水位 / watch 账三面。"""
    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", root / "events-seen.json")
    monkeypatch.setattr(watch, "WATCH_PATH", root / "watch.json")
    return root


def _write_stream(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ③ 水位面：ack 抗回卷自愈
# ---------------------------------------------------------------------------
def test_ack_self_heals_when_stream_rolls_back(face):
    """流被覆盖回卷（旧水位 > 新流最大 seq）→ ``unacked()`` 不得假绿，水位归零重排并点名。"""
    for i in range(6):
        events.emit(events.JOB_DONE, f"j{i}", f"d{i}", "a")
    assert events.drain() and events.ack_seq() == 6

    # 模拟 merge 覆盖：main 活流（seq≤6）被 worktree 旧快照替换（seq 回到 1..2）
    _write_stream(
        events.EVENTS_LOG,
        [
            {
                "ts": "2026-10-16T19:55:41+08:00",
                "seq": 1,
                "event": "JOB-DONE",
                "subject": "n1",
                "detail": "x",
                "action": "y",
            },
            {
                "ts": "2026-10-16T19:55:42+08:00",
                "seq": 2,
                "event": "QUEUE-EMPTY",
                "subject": "n2",
                "detail": "x",
                "action": "y",
            },
        ],
    )
    assert events.latest_seq() == 2

    unacked = events.unacked()
    assert [e["subject"] for e in unacked] == ["n1", "n2"], (
        "回卷后新流事件必须可见（此前恒空）"
    )
    assert events.ack_seq() == 0, "自愈：旧水位对新编号无意义 → 归零"
    lines = "\n".join(events.status_lines())
    assert "待排空 2" in lines, "不得报「待排空 0」假绿"
    assert "回卷已自愈" in lines and "水位曾到 6" in lines, "自愈须显式点名（不静默）"


def test_ack_stays_put_when_stream_advances_normally(face):
    """正常推进（水位 ≤ 流最大 seq）不受自愈逻辑影响。"""
    events.emit(events.JOB_DONE, "j1", "d", "a")
    assert events.ack_seq() == 0  # 未排空
    events.drain()
    assert events.ack_seq() == 1
    events.emit(events.JOB_DONE, "j2", "d", "a")
    assert events.ack_seq() == 1 and len(events.unacked()) == 1


# ---------------------------------------------------------------------------
# ② 身份面：内容身份抗 seq 回卷
# ---------------------------------------------------------------------------
def test_stream_identity_is_content_based(face):
    """同一 seq、不同事实 → 身份不同；同一事实、seq 变 → 身份不变。"""
    old = {
        "ts": "2026-10-15T00:33:46+08:00",
        "seq": 6,
        "event": events.JOB_DONE,
        "subject": "j-old",
        "detail": "旧流第 6 条",
        "action": "advance",
    }
    new = {
        "ts": "2026-10-16T19:55:43+08:00",
        "seq": 6,
        "event": events.JOB_DONE,
        "subject": "j-new",
        "detail": "今日第 6 条",
        "action": "advance",
    }
    assert watch._identity(old) != watch._identity(new), (
        "seq 相同但事实不同 → 不得同身份"
    )
    assert watch._identity({**new, "seq": 999}) == watch._identity(new), (
        "身份与 seq 无关"
    )
    assert watch._identity(dict(new)) == watch._identity(new), (
        "同一事实身份稳定（幂等去重）"
    )


def test_old_seq_ledger_does_not_swallow_new_event(face):
    """旧消费账里的 seq 身份（bug 当刻形状）不得吞掉回卷后的新事件。"""
    rows = [
        {
            "ts": f"2026-10-16T19:55:{i:02d}+08:00",
            "seq": i,
            "event": events.JOB_DONE if i == 6 else events.QUEUE_EMPTY,
            "subject": "j-new" if i == 6 else f"filler-{i}",
            "detail": "d",
            "action": "advance",
        }
        for i in range(1, 7)
    ]
    _write_stream(events.EVENTS_LOG, rows)
    # 昨晚旧流的消费身份（实锤里正是它撞车）
    watch.save_state({"consumed": {"JOB-DONE|#6": "2026-10-15T00:33:46+08:00"}})

    evs = watch.detect_events({"events_seq": 0}, {"events_seq": events.latest_seq()})
    kept = watch.filter_consumed(evs)
    assert any(e["subject"] == "j-new" for e in kept), (
        "新事实被旧 seq 身份吞掉＝静默漏报"
    )


# ---------------------------------------------------------------------------
# ③ 端到端：merge 覆盖 → 新 JOB-DONE 仍唤醒（现必静默）
# ---------------------------------------------------------------------------
def test_watch_still_wakes_after_stream_overwrite(face):
    """基线水位（旧世界 45）> 当前流最大 seq（回卷后 3）→ 从 0 重扫，新事实仍报出。"""
    _write_stream(
        events.EVENTS_LOG,
        [
            {
                "ts": "2026-10-16T19:55:41+08:00",
                "seq": 1,
                "event": events.QUEUE_EMPTY,
                "subject": "a",
                "detail": "d",
                "action": "x",
            },
            {
                "ts": "2026-10-16T19:55:42+08:00",
                "seq": 2,
                "event": events.QUEUE_EMPTY,
                "subject": "b",
                "detail": "d",
                "action": "x",
            },
            {
                "ts": "2026-10-16T19:55:43+08:00",
                "seq": 3,
                "event": events.JOB_DONE,
                "subject": "figure-1",
                "detail": "s6 完工",
                "action": "advance",
            },
        ],
    )
    base = {"events_seq": 45}  # 回卷前的水位（新流里根本没有 45 号）
    cur = {"events_seq": events.latest_seq()}

    evs = watch.detect_events(base, cur)
    assert any(e.get("subject") == "figure-1" for e in evs), (
        "回卷后新事件必须报出（此前静默）"
    )

    watch.save_state({"consumed": {"JOB-DONE|#3": "2026-10-15T00:00:00+08:00"}})
    kept = watch.filter_consumed(evs)
    assert any(e.get("subject") == "figure-1" for e in kept), "旧 seq 账不得吞新事件"


def test_identity_ledger_dedupes_same_fact_across_reads(face):
    """回卷重扫不引入重复唤醒：同一事实读两遍身份相同 → 消费账只记一条。"""
    events.emit(events.JOB_DONE, "figure-1", "收尾", "advance")
    evs = watch.detect_events({"events_seq": 0}, {"events_seq": events.latest_seq()})
    st = {"consumed": {}}
    watch.mark_consumed(evs, st)
    assert len(st["consumed"]) == 1
    watch.save_state(st)  # 消费账读盘为准
    # 回卷后重扫同一事实（内容身份不变）→ 被消费账过滤
    assert watch.filter_consumed(evs) == []
