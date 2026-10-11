"""QUEUE-COMPLETE 识别强化 + 抑制帧不吞账 + 黑匣子（backlog 20261011-220622-leader）。

2026-10-11 实测故障：drain run 20261011-210557 收尾后，挂载中的 watch 轮询了十几帧
却 ``fired=[]``，其后 ``orch status`` 补账也报「无新事件」——因为 ``same_run`` 抑制帧
照样推进基线，被压住的完成事实再不参与比较。成因帧是**锁自称在跑该 run**（``release_lock``
的 unlink 在 Windows 上可能静默失败 + ``_pid_alive`` 保守判活/pid 回收），而旧判定把锁的
自称看得比 ``.done`` 这块收尾硬证据更重。

本文件按建议书要求 5 逐类钉住：pid 复用假象、``.done`` 半写、同 run ``.done`` 重写、
running→idle 转移帧、抑制/空值帧不推进基线；外加黑匣子 ``recent``/``exit_reason`` 与
「上一任遗言」的可诊断面。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import drain, watch

from .test_watch import (
    Clock,
    _seed_done,
    _seed_lock,
    _seq,
    _snap,
)


# ---------------------------------------------------------------------------
# 要求 2：完成事实优先于锁的自称（脱靶主因）
# ---------------------------------------------------------------------------
def test_pid_reuse_illusion_still_reports(iso_state, monkeypatch):
    """锁残留 + 判活为真（pid 回收/tasklist 超时保守判活）+ ``.done`` 已在 → 必须报。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    _seed_done(iso_state, "r1", {"completed": "2026-10-11T21:42:41+08:00", "count": 3})
    _seed_lock(iso_state, "r1")
    cur = watch.snapshot()
    assert cur["lock_state"] == "running" and cur["lock_done"]  # 锁在说谎，摘要在作证
    ev = watch.detect_events(_snap(), cur)
    assert [e["event"] for e in ev] == ["QUEUE-COMPLETE"]
    assert ev[0]["subject"] == "r1"


def test_genuinely_running_run_is_not_reported(iso_state, monkeypatch):
    """反面对照：锁真在跑 r2（只有 .log、无 .done）→ 在跑的 run 不算完成，不虚报。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    _seed_done(iso_state, "r1", {"completed": "T1", "count": 1})
    _seed_lock(iso_state, "r2")
    cur = watch.snapshot()
    assert cur["lock_run"] == "r2" and not cur["lock_done"] and cur["done"] == "r1"
    base = _snap(done="r1", done_completed="T1", done_count=1)  # r1 已入基线
    assert watch.detect_events(base, cur) == []


def test_running_to_idle_transition_reports_completion(iso_state, monkeypatch):
    """两场帧：arm 时锁真在跑 rC（尚无摘要）→ 收尾时 unlink 失败、锁自称仍在跑 →
    该帧照报；锁随后真释放的转移帧也不会把这笔账漏掉（要求 2）。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    _seed_lock(iso_state, "rC")
    base = watch.snapshot()  # 进行中：没有完成事实可报
    assert watch.detect_events(base, base) == []
    _seed_done(iso_state, "rC", {"completed": "2026-10-11T21:00:00+08:00", "count": 2})
    cur = watch.snapshot()
    assert cur["lock_state"] == "running"  # 锁残留 + 判活说谎，仍在自称
    assert [e["event"] for e in watch.detect_events(base, cur)] == ["QUEUE-COMPLETE"]
    (iso_state / "devops.lock").unlink()
    idle = watch.snapshot()
    held = watch.advance_base(base, cur, watch.detect(base, cur)[1]["hold"])
    assert watch.detect_events(held, idle) == []  # 已报身份入基线，不重复报


def test_transition_frame_reports_never_reported_run(iso_state):
    """``.done`` 名额被更晚的 run 占了也要补报：基线里锁自称在跑 rA 且 rA 已收尾，
    而 latest_done 是别的 run —— rA 的完成从未进过基线，转移帧必须点名补报。"""
    _seed_done(iso_state, "rA", {"completed": "2026-10-11T21:00:00+08:00", "count": 2})
    rb = dict(done="rB", done_completed="2026-10-11T22:00:00+08:00", done_count=1)
    base = _snap(
        lock_state="running",
        lock_run="rA",
        lock_done=watch._key("rA", "2026-10-11T21:00:00+08:00", 2),
        **rb,
    )
    cur = _snap(**rb)  # 锁已释放，最新摘要属于更晚的 rB
    ev = watch.detect_events(base, cur)
    assert [e["subject"] for e in ev] == ["rA"]
    assert "2 项" in ev[0]["detail"] and "转移帧" in ev[0]["detail"]


# ---------------------------------------------------------------------------
# 要求 1：完成身份 = (run 名, completed, count) 三元组
# ---------------------------------------------------------------------------
def test_same_run_done_rewrite_re_reports(iso_state):
    """同 run 的 ``.done`` 被重写（配额/协作停止后续跑完）→ 三元组变 → 再报。"""
    _seed_done(iso_state, "r7", {"completed": "2026-10-11T20:00:00+08:00", "count": 1})
    before = watch.snapshot()
    assert watch.detect_events(_snap(), before)[0]["event"] == "QUEUE-COMPLETE"
    _seed_done(iso_state, "r7", {"completed": "2026-10-11T21:00:00+08:00", "count": 4})
    after = watch.snapshot()
    ev = watch.detect_events(before, after)
    assert [e["event"] for e in ev] == ["QUEUE-COMPLETE"]
    assert "4 项" in ev[0]["detail"]


def test_identical_done_readings_never_repeat(iso_state):
    """三元组未变（同一收尾事实）→ 不重复打扰，即便 run 名与基线同串也要按住。"""
    _seed_done(iso_state, "r8", {"completed": "2026-10-11T20:00:00+08:00", "count": 1})
    cur = watch.snapshot()
    base = dict(cur)
    assert watch.detect_events(base, cur) == []


# ---------------------------------------------------------------------------
# 要求 3：抑制帧与瞬态空值不得推进基线（吞账根因）
# ---------------------------------------------------------------------------
def test_suppressed_frame_does_not_swallow_completion():
    """真进行中而摘要未落 → 抑制；但基线的完成身份必须按住，下一帧释放即报。"""
    base = _snap()
    cur = _snap(
        done="r2",
        done_completed="T2",
        done_count=2,
        lock_state="running",
        lock_run="r2",
    )
    events, verdict = watch.detect(base, cur)
    assert events == [] and "done" in verdict["hold"]
    assert verdict["why"] == "lock-claims-running"
    held = watch.advance_base(base, cur, verdict["hold"])
    assert held["done"] == "" and held["done_count"] == 0  # 身份未被洗进基线
    assert held["lock_run"] == "r2"  # 观察面照常推进
    after = dict(cur, lock_state="idle", lock_run="", lock_done="")
    ev = watch.detect_events(held, after)
    assert [e["subject"] for e in ev] == ["r2"]


def test_half_written_done_does_not_wipe_identity():
    """``.done`` 半写/读不出 → 瞬态空值不得覆盖已确立的完成身份，也不得据此重复报。"""
    base = _snap(done="r1", done_completed="T1", done_count=3)
    gap = _snap()
    events, verdict = watch.detect(base, gap)
    assert events == [] and "done" in verdict["hold"]
    held = watch.advance_base(base, gap, verdict["hold"])
    assert held["done"] == "r1" and held["done_completed"] == "T1"
    assert held["done_count"] == 3
    assert (
        watch.detect_events(held, _snap(done="r1", done_completed="T1", done_count=3))
        == []
    )
    again = watch.detect_events(
        held, _snap(done="r1", done_completed="T2", done_count=5)
    )
    assert [e["event"] for e in again] == ["QUEUE-COMPLETE"]


def test_corrupt_done_file_reads_as_no_evidence(iso_state):
    """半写文件按「无收尾证据」处理（不抛、不谎报），空身份串不进基线比较。"""
    runs = iso_state / "devops-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "r9.done").write_text('{"count": 2, "elap', encoding="utf-8")
    assert watch._done_file("r9") == {}
    assert watch._lock_done_key("r9") == ""
    assert watch._done_key(_snap()) == ""


def test_transient_empty_crash_and_quota_do_not_re_open_events():
    """崩溃/额度身份同受保护：瞬态空值不洗基线，否则同一 run 会被重复报。"""
    base = _snap(crash_run="r7", quota_run="r7")
    events, verdict = watch.detect(base, _snap())
    assert events == []
    assert {"crash_run", "quota_run"} <= verdict["hold"]
    held = watch.advance_base(base, _snap(), verdict["hold"])
    assert held["crash_run"] == "r7" and held["quota_run"] == "r7"


# ---------------------------------------------------------------------------
# 端到端：那场脱靶的复现（挂载中撞上残留锁也必须唤醒）
# ---------------------------------------------------------------------------
def test_run_watch_wakes_on_residue_frame(iso_state, monkeypatch):
    """arm 时无摘要无锁 → 收尾时释放失败残留 + 判活为真 → 挂载中的 watch 当场报。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    armed = watch.snapshot()  # 此刻还没有 .done：基线干净
    _seed_done(
        iso_state,
        "20261011-210557",
        {"completed": "2026-10-11T21:42:41+08:00", "count": 3},
    )
    _seed_lock(iso_state, "20261011-210557")
    hit = watch.snapshot()
    monkeypatch.setattr(watch, "snapshot", _seq(armed, hit))
    out: list[str] = []
    clock = Clock()
    rc = watch.run_watch(
        timeout_s=600,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=out.append,
    )
    assert rc == 0
    assert out[0].startswith("QUEUE-COMPLETE 20261011-210557")
    st = watch.load_state()
    assert st["exit_reason"] == "event:QUEUE-COMPLETE"
    assert st["recent"][-1]["done"].startswith("20261011-210557")
    assert watch.detect_events(st["base"], hit) == []  # 已报身份入基线，不重复


# ---------------------------------------------------------------------------
# 要求 4：黑匣子（recent 环形 + exit_reason + 上一任遗言）
# ---------------------------------------------------------------------------
def test_recent_ring_and_exit_reason_persist(iso_state, monkeypatch):
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(pending=1), _snap(pending=2)))
    clock = Clock()
    watch.run_watch(
        timeout_s=120,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=lambda *a: None,
    )
    st = watch.load_state()
    assert st["last_exit"] == "timeout" and st["exit_reason"] == "timeout"
    assert len(st["recent"]) == st["polls"] == 3  # 每轮（含超时收拍）都留一帧
    assert {"ts", "lock_state", "lock_run", "done", "why"} <= set(st["recent"][0])
    assert watch.fmt_frame(st["recent"][1]).startswith(str(st["recent"][1]["ts"])[:19])


def test_push_recent_caps_ring():
    st: dict = {"recent": []}
    for i in range(watch.RECENT_MAX + 5):
        watch.push_recent(st, watch._frame(_snap(ts=f"T{i}")))
    assert len(st["recent"]) == watch.RECENT_MAX
    assert st["recent"][-1]["ts"] == f"T{watch.RECENT_MAX + 4}"


def test_arm_inherits_black_box_of_killed_predecessor(iso_state, monkeypatch):
    """arm 全局覆写 watch.json——先把上一任的现场折进 ``prev``，被杀的那任才留得下遗言。"""
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(), _snap()))
    first = watch.arm()
    watch.push_recent(first, watch._frame(_snap(done="rZ"), "lock-claims-running"))
    watch.save_state(first)
    fresh = watch.arm()
    assert fresh["prev"]["exit_reason"] == watch.KILLED
    assert fresh["prev"]["recent"][-1]["why"] == "lock-claims-running"
    lines = watch.last_words_lines(fresh)
    assert any("没有退出记录" in line for line in lines)
    assert any("lock-claims-running" in line for line in lines)


def test_last_words_silent_for_clean_exit(iso_state, monkeypatch):
    """正常命中/超时都有账可查，status 不该为此刷屏。"""
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(), _snap()))
    st = watch.arm()
    st["last_exit"] = "timeout"
    st["exit_reason"] = "timeout"
    watch.save_state(st)
    assert watch.last_words_lines() == []
    assert watch.last_words_lines(watch.load_state()) == []


def test_reconcile_of_killed_incumbent_reports_and_signs(iso_state, monkeypatch):
    """被杀那任的基线仍带着未报的完成事实 → status 补账照报，并追认「无退出记录」。"""
    monkeypatch.setattr(watch, "snapshot", _seq(_snap()))
    st = watch.arm()
    stale = (
        datetime.now().astimezone() - timedelta(seconds=st["interval_s"] * 5)
    ).isoformat(timespec="seconds")
    st["heartbeat_at"] = stale  # 心跳断供：等待器死了，且没写退出原因
    watch.save_state(st)
    monkeypatch.setattr(
        watch,
        "snapshot",
        _seq(_snap(done="rD", done_completed="T", done_count=2)),
    )
    events, settled = watch.reconcile()
    assert settled and [e["event"] for e in events] == ["QUEUE-COMPLETE"]
    assert watch.load_state()["exit_reason"] == watch.KILLED
    assert "没有退出记录" in "\n".join(watch.status_lines())
