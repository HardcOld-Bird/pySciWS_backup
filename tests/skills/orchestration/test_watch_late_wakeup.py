"""watch 唤醒链加固（autonomy-fix a1）：迟到唤醒 + 看门狗自愈 + 水位单调。

背景（2026-10-14 用户实测，02:30–07:30 五小时唤醒真空）：
1. watcher 被**无声杀死**（watcher 的所有**正常**退出路径都留痕 ⇒ 零留痕只能是外部终止）；
2. `arm()` 又把对比基线重置为**挂载瞬间快照**——聋期（无监听期间）产生的 QUEUE-EMPTY
   被新基线吸收，永不触发。
本文件把三条机制钉死：① 聋期事件→新挂载 ≤1 轮询周期迟到唤醒；② listener 死→看门狗阈值内
重臂（留痕）；③ 水位单调、重复挂载不重复唤醒。④ 看门狗自启动（hook 每回合幂等拉起，冷启动
后即在位）——backlog 20261014-104738。
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import (
    drain,
    events,
    watch,
    watchdog,
)


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / "state"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(watch, "WATCH_PATH", root / "watch.json")
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", root / "events-seen.json")
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=root / "watchdog.lock",
            runs=root / "watchdog-runs",
            cancel=root / "watchdog.cancel",
        ),
    )
    return root


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    )


# ---------------------------------------------------------------------------
# ① 聋期事件 → 新挂载 ≤1 个轮询周期迟到唤醒
# ---------------------------------------------------------------------------
def test_deaf_period_event_fires_on_next_mount(state):
    watch.arm()  # 首次挂载：无历史基线 → 当下快照种子（无可迟发项）
    # 聋期：watcher 已死、无人监听时产生一条生产者直发事件（02:30 那起 QUEUE-EMPTY 的形状）
    events.emit(
        events.QUEUE_EMPTY, "run-022113", "devops 队列消化完：处理 3 项", "drain"
    )

    st = watch.arm()  # 聋期后的新挂载（07:40 那种）
    assert st["catchup"] is True, "继承持久水位 → 本任是迟到接续"

    evs, verdict = watch.detect(st["base"], watch.snapshot())
    assert [e["event"] for e in evs] == ["QUEUE-EMPTY"], "聋期事件必须被读出"
    watch._mark_late(evs, st)
    assert evs[0].get("late") is True and "迟到" in evs[0]["detail"]


def test_first_arm_has_nothing_to_late_report(state):
    """首次挂载（无历史基线）用当下快照种子：不该凭空造出迟发事件。"""
    st = watch.arm()
    assert st["catchup"] is False
    assert watch.detect(st["base"], watch.snapshot())[0] == []


# ---------------------------------------------------------------------------
# ③ 水位单调：重复挂载不重复唤醒
# ---------------------------------------------------------------------------
def test_watermark_monotonic_no_double_fire(state):
    watch.arm()
    events.emit(
        events.JOB_DONE, "figure-1", "figure job 收尾：done（result）", "advance"
    )

    st = watch.arm()
    prev, cur = st["base"], watch.snapshot()
    evs, verdict = watch.detect(prev, cur)
    assert [e["event"] for e in evs] == ["JOB-DONE"]
    st["base"] = watch.advance_base(prev, cur, verdict["hold"])
    watch.save_state(st)  # 等价 _touch/_emit 的落账

    st2 = watch.arm()  # 重复挂载（无新事件）
    assert watch.detect(st2["base"], watch.snapshot())[0] == [], "水位单调，不重复消费"


# ---------------------------------------------------------------------------
# ② listener 被杀 → 看门狗阈值内重臂（留痕）；活着则不打扰
# ---------------------------------------------------------------------------
def _fake_sleep(monkeypatch, clock: dict, *, bump_s: int, stop_after: int):
    """假睡：推进假时钟（不真等）；第 ``stop_after`` 次自查后协作停止，让循环干净退出。"""
    calls: list[float] = []

    def fake_sleep(seconds):
        calls.append(float(seconds))
        clock["t"] += float(bump_s)
        if len(calls) >= stop_after:
            monkeypatch.setattr(watchdog, "_consume_stop_request", lambda *a, **k: True)

    return fake_sleep


def test_watchdog_rearms_when_listener_dies(state, monkeypatch):
    """心跳 stale（listener 被无声杀死）→ 看门狗在阈值内重臂监听并留痕。"""
    clock = {"t": time.time()}
    watch.save_state(
        {"version": 1, "heartbeat_at": _iso(clock["t"] - 400)}
    )  # 400s 无心跳
    rearms: list[str] = []
    monkeypatch.setattr(
        watchdog, "_rearm", lambda run_log=None: rearms.append(str(run_log)) or True
    )
    fake_sleep = _fake_sleep(monkeypatch, clock, bump_s=60, stop_after=2)

    run_log = state / "wd.log"
    summary = watchdog.run_watchdog(
        run_log, interval_s=60, stale_s=300, sleeps=fake_sleep, now=lambda: clock["t"]
    )

    assert summary["rearms"] == 1 and len(rearms) == 1, "心跳 stale 必须重臂"
    assert "重臂" in run_log.read_text(encoding="utf-8")
    assert summary["stopped"] == "cooperative"


def test_watchdog_stays_quiet_when_listener_live(state, monkeypatch):
    """心跳新鲜 → 看门狗不重臂（不打扰在跑的 listener）。"""
    clock = {"t": time.time()}
    watch.save_state({"version": 1, "heartbeat_at": _iso(clock["t"])})
    rearms: list[str] = []
    monkeypatch.setattr(
        watchdog, "_rearm", lambda run_log=None: rearms.append(str(run_log)) or True
    )
    fake_sleep = _fake_sleep(monkeypatch, clock, bump_s=60, stop_after=3)

    summary = watchdog.run_watchdog(
        state / "wd.log",
        interval_s=60,
        stale_s=300,
        sleeps=fake_sleep,
        now=lambda: clock["t"],
    )

    assert summary["rearms"] == 0 and rearms == []


def test_heartbeat_age_and_listener_live_helpers(state):
    watch.save_state({"version": 1, "heartbeat_at": _iso(time.time())})
    assert watchdog.listener_live() is True
    watch.save_state({"version": 1, "heartbeat_at": _iso(time.time() - 400)})
    assert watchdog.listener_live(stale_s=300) is False
    watch.save_state({"version": 1})  # 无心跳 → 无 listener
    assert watchdog.heartbeat_age_s() == float("inf")


# ---------------------------------------------------------------------------
# ③b 日志完整化（hook 侧）：token 自抢 / 环境失败 / TERM 都留痕
# ---------------------------------------------------------------------------
def test_hook_script_logs_its_exit_paths():
    text = (PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh").read_text(
        encoding="utf-8"
    )
    assert "接任 watch" in text  # token 写入即留痕
    assert "token 缺失 → 自抢" in text  # 自抢分支也留痕（此前静默）
    assert "被取代" in text  # 让位退出留痕
    assert "环境缺失" in text  # python/orch 缺失留痕
    assert "TERM" in text  # TERM/INT 陷阱留痕（SIGKILL 无法拦，交看门狗）


def test_orch_exposes_watchdog_command():
    from pysci.skills.orchestration.tools import orch

    assert orch.main(["watchdog", "status"]) == 0


# ---------------------------------------------------------------------------
# ③c 看门狗自启动（backlog 20261014-104738）：hook 每回合幂等拉起，冷启动后即在位
# ---------------------------------------------------------------------------
def test_hook_script_self_starts_watchdog():
    """hook 脚本每回合调用 `orch watchdog start`（幂等）——冷启动/重启无需手工敲。"""
    text = (PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh").read_text(
        encoding="utf-8"
    )
    assert "watchdog start" in text, "hook 未自启动看门狗 → 冷启动后「被杀即重臂」空转"
    assert "已后台启动" in text, "仅在实际拉起时留痕（在场则秒返回不留噪）"
    assert "自启动失败" in text, "失败要留痕（否则冷启动静默失败）"


def test_watchdog_self_start_idempotent_then_relaunch(state, monkeypatch):
    """(a) 缺锁→拉起；(b) 第二次→「已在跑」不重复派生；(c) 被杀（锁清）→ 下一跳重新拉起。"""
    wp = watchdog.WATCHDOG_PATHS
    wp.runs.mkdir(parents=True, exist_ok=True)
    spawned: list[list[str]] = []

    def fake_spawn(argv, *, env, stdout):
        spawned.append(argv)
        drain.acquire_lock("wd-sim", wp)  # 模拟被拉起的看门狗子进程持有守护者锁

    monkeypatch.setattr(watchdog, "_spawn_detached", fake_spawn)

    ok1, _ = watchdog.start_watchdog()
    assert ok1 is True and len(spawned) == 1, "(a) 缺锁即拉起"

    ok2, msg2 = watchdog.start_watchdog()
    assert ok2 is False and "已在跑" in msg2 and len(spawned) == 1, "(b) 幂等不重复派生"

    drain.release_lock(wp.runs / "wd-sim.log", wp)  # (c) 看门狗被杀 → 锁释放
    ok3, _ = watchdog.start_watchdog()
    assert ok3 is True and len(spawned) == 2, "(c) 被杀后下一跳重新拉起"


# ---------------------------------------------------------------------------
# ④ 看门狗生命周期：start 幂等（已在跑不重复派生）、stop 空闲不误报、无孤儿
# ---------------------------------------------------------------------------
def test_start_watchdog_refuses_when_already_running(state, monkeypatch):
    """幂等：per-守护者锁在手（= 有活着的看门狗）时，start 不再派生新进程。"""
    wp = watchdog.WATCHDOG_PATHS
    wp.runs.mkdir(parents=True, exist_ok=True)
    assert drain.acquire_lock("held", wp) is not None  # 模拟已在跑

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        watchdog, "_spawn_detached", lambda argv, *, env, stdout: spawned.append(argv)
    )
    ok, msg = watchdog.start_watchdog()
    assert ok is False and spawned == [] and "已在跑" in msg
    drain.release_lock(wp.runs / "held.log", wp)


def test_stop_watchdog_broadcasts_even_when_lock_idle(state):
    """空闲时 stop 改为**广播纪元**（backlog 20261014-135935-devops）：锁空闲也投递。

    旧语义「空闲不写信号」在孤儿场景下失效——不持锁的 ``_watchdog-loop`` 永远收不到停
    信（2026-10-14 实测：两实例同跑、一次 stop 只停一个）。现广播恒写、按启动时刻比较，
    新实例不受旧纪元误停（该性质由 test_watchdog_stop.py 单测）。
    """
    wp = watchdog.WATCHDOG_PATHS
    wp.runs.mkdir(parents=True, exist_ok=True)
    ok, msg = watchdog.stop_watchdog()
    assert ok is True and "广播" in msg
    data = json.loads(wp.cancel.read_text(encoding="utf-8"))
    assert data["scope"] == "all" and data["requested_at"]
