"""worker **静默卡死**的取证埋点与防线（task-20261015-worker-stall，用户点名高危）。

实案（2026-10-15 08:02）：devops 队列 worker 获锁后静默卡死 31+ 分钟——run 日志 0 字节、
零会话登记、零 token、条目假挂 ``in_progress``，而**心跳线程照跳**故三道旧闸全判它「活着」。
本文件按任务书 A/B/C 三段钉：

* **A 取证**：``run worker → do_dispatch`` 各阶段真写 run 日志（带时间戳；``quiet=True`` 不抑制）；
* **B 防线**：跳级 watchdog 判死→复位→出声→杀进程树；孤儿自愈不依赖下次获锁；
  ``communicate`` 超时杀子树；``queue_pressure`` 补「有心跳无进展」判据；超龄锁的心跳免抢；
* 时间相关一律用注入口压缩等待（假时钟 / 注入监护 / 注入杀器），不真等。
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import (
    backpressure,
    drain,
    events,
    procs,
    queues,
    queueworker,
    runner,
    stallwatch,
    watch,
    workq,
)
from pysci.skills.orchestration.tools.registry import Member

from .conftest import envelope, fake_headless

OK = "<result>完工，commit abc1234 已合并。</result>"
MEMBER = "quotamember"

_STAGE_TS_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+\-]\d{2}:\d{2}\]"
)


def _iso(minutes_ago: float = 0) -> str:
    return (datetime.now().astimezone() - timedelta(minutes=minutes_ago)).isoformat(
        timespec="seconds"
    )


def _stages(run_log) -> dict[str, str]:
    """run 日志里的 ``[stage] <名> …`` 行 → ``{阶段名: 整行}``。"""
    out: dict[str, str] = {}
    for ln in run_log.read_text(encoding="utf-8").splitlines():
        if "] [stage] " not in ln:
            continue
        name = ln.split("] [stage] ", 1)[1].split(" ", 1)[0]
        out[name] = ln
    return out


# ===========================================================================
# A. 取证埋点：阶段真写 run 日志
# ===========================================================================
def test_stage_trace_really_writes_run_log(iso_state, iso_dispatch, monkeypatch):
    """success_criteria：run worker→do_dispatch 阶段埋点真写 run_log（含时间戳，quiet 不抑制）。

    真跑 ``do_dispatch``（只把 ``run_headless`` 换成假实现），断言从获锁到跳收尾的每一段
    都在 run 日志里有带毫秒时间戳的行——这正是 08:02 实案里**完全缺失**的那一维
    （``quiet=True`` 吃掉 stdout，run 日志 0 字节 ⇒ 事后钉不死卡在哪一段）。
    """
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    item = queues.enqueue(MEMBER, text="出图", session="new")
    wlog = queueworker.new_run_log(queueworker.member_paths(MEMBER))

    summary = queueworker.run_worker(MEMBER, wlog)

    assert summary["items"] and summary["items"][0]["status"] == workq.STATUS_DONE
    stages = _stages(wlog)
    must = (
        "lock_acquired",
        "prune_stale",
        "reset_orphans_begin",
        "take_ready",
        "item_taken",
        "dispatch_begin",
        "seat_resolved",
        "session_selected",
        "taskbook_written",
        "popen_begin",
        "spawn",
        "popen_returned",
        "hop_settled",
    )
    missing = [s for s in must if s not in stages]
    assert not missing, f"缺阶段埋点：{missing}（实得 {sorted(stages)}）"
    # 每行都带毫秒时间戳（钉「事后能排出卡在哪一段、卡了多久」）
    for name, ln in stages.items():
        assert _STAGE_TS_RE.match(ln), f"阶段 {name} 缺时间戳：{ln!r}"
    # 派生瞬间的 pid 落进埋点（判死时要点名杀谁）
    assert "pid=" in stages["spawn"], stages["spawn"]
    assert str(item["id"]) in stages["item_taken"]


def test_log_stage_is_fail_open_and_accepts_none(tmp_path):
    """埋点写失败绝不改变派发结局：``None`` 日志、父路径是文件、坏字符路径都静默通过。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    drain.log_stage(None, "x")  # 不抛
    drain.log_stage(blocker / "sub" / "r.log", "x", k=1)  # 不抛（父目录建不出来）
    drain.log_stage(tmp_path / "bad\0name.log", "x")  # 不抛（坏路径）
    p = tmp_path / "r.log"
    drain.log_stage(p, "stage_a", member="m", empty="")
    text = p.read_text(encoding="utf-8")
    assert "[stage] stage_a member=m" in text and "empty=" not in text


# ===========================================================================
# B1. 跳级 watchdog：判据（纯函数 + 假时钟）
# ===========================================================================
def test_job_verdict_no_spawn_then_no_growth():
    """双判据：先「未派生超 N」，派生后转「字符零增长超 M」；探不到水位则弃权。"""
    prog = stallwatch.JobProgress(item_id="j1", member="m", started_at=1000.0)
    assert stallwatch.job_verdict(prog, now=1000 + 599) == ""
    assert stallwatch.job_verdict(prog, now=1000 + 601) == "no_spawn"

    prog.spawned_at = 1700.0
    prog.chars_at = 1700.0
    prog.probed = True
    prog.pid, prog.sid = 4242, "s" * 8
    assert stallwatch.job_verdict(prog, now=1700 + 1799) == ""
    assert stallwatch.job_verdict(prog, now=1700 + 1801) == "no_growth"

    prog.chars_at = 1700 + 1800  # 观测到增长 → 计时归零
    assert stallwatch.job_verdict(prog, now=1700 + 1801) == ""

    prog.probed = False  # 探不到会话水位 → 不据此判死
    assert stallwatch.job_verdict(prog, now=1700 + 99999) == ""


def test_stall_watch_fires_once_with_verdict_text():
    """监护线程的判定体：命中即回调一次并记 ``fired``（防重复动作），此后不再触发。"""
    clock = {"t": 1000.0}
    fired: list[tuple[str, str]] = []
    w = stallwatch.StallWatch(
        on_stall=lambda v, r, p: fired.append((v, r)),
        now=lambda: clock["t"],
        probe=lambda member, sid: 0,
        start_thread=False,
    )
    w.arm("j1", "devops")
    assert w.tick() == ""  # 未到阈值
    clock["t"] += stallwatch.NO_SPAWN_S + 1
    assert w.tick() == "no_spawn"
    assert len(fired) == 1 and "判卡死" in fired[0][1] and "devops" in fired[0][1]
    clock["t"] += 10_000
    assert w.tick() == "" and len(fired) == 1  # 已触发过，不重复
    w.disarm()
    assert w.tick() == ""  # 已收尾的 job 不再判死


def test_stall_watch_growth_resets_timer():
    """会话仍在长字符 ⇒ 是「慢」不是「卡死」（实测单跳 p90=41min，绝不能误杀）。"""
    clock = {"t": 1000.0}
    chars = {"n": 0}
    fired: list[str] = []
    w = stallwatch.StallWatch(
        on_stall=lambda v, r, p: fired.append(v),
        now=lambda: clock["t"],
        probe=lambda member, sid: chars["n"],
        start_thread=False,
    )
    w.arm("j1", "devops")
    w.note("spawn", pid=1, sid="abcd1234")
    for _ in range(4):  # 每 20min 有增长 → 永不判死
        clock["t"] += 20 * 60
        chars["n"] += 500
        assert w.tick() == ""
    clock["t"] += stallwatch.NO_GROWTH_S + 1  # 停止增长 → 判死
    assert w.tick() == "no_growth" and fired == ["no_growth"]


# ===========================================================================
# B1'. watchdog 的动作：复位条目 + 出声 + 杀进程树（注入监护/杀器，不真杀）
# ===========================================================================
class _Stop(BaseException):
    """模拟 ``os._exit``：解栈退出 worker，但不被 ``except Exception`` 吞掉。"""


class _FakeGuard:
    """假监护：条目一取到就判死（把时序从测试里摘出去；**动作仍走真实现**）。"""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.notes: list[str] = []
        self.prog = stallwatch.JobProgress()

    def start(self):
        return self

    def stop(self) -> None:
        pass

    def arm(self, item_id: str, member: str) -> None:
        self.prog = stallwatch.JobProgress(
            item_id=str(item_id), member=str(member), started_at=1.0
        )

    def disarm(self) -> None:
        pass

    def note(self, stage: str, **fields) -> None:
        self.notes.append(stage)
        if stage == "item_taken":
            self.handler("no_spawn", "注入：出队后未派生即卡死", self.prog)


def test_watchdog_kill_resets_item_and_emits_event(
    iso_state, iso_dispatch, monkeypatch
):
    """success_criteria：卡死注入后**条目回 pending + 出声**（杀进程树的动作一并断言）。

    注入的是「判死」这一步（真监护的判定体已单测）；复位/出声/杀树的**动作**全走
    ``_make_stall_handler`` 真实现，只把杀器与硬退出换成记录器。
    """
    seen: dict[str, object] = {}

    def fake_kill(pid: int):
        seen.setdefault("kills", []).append(pid)
        return True, "fake"

    def fake_exit(code: int):
        # 记录「判死那一刻」的现场：条目状态与事件流（真进程此刻已被杀，写盘已成事实）
        seen["exit_code"] = code
        seen["status_at_exit"] = [i["status"] for i in queues.items(MEMBER)]
        seen["events_at_exit"] = [e["event"] for e in events.read_all()]
        raise _Stop()

    monkeypatch.setattr(stallwatch, "kill_tree", fake_kill)
    monkeypatch.setattr(stallwatch, "hard_exit", fake_exit)
    queues.enqueue(MEMBER, text="出图", session="new")
    wlog = queueworker.new_run_log(queueworker.member_paths(MEMBER))
    q = queues.queue_path(MEMBER)
    guard = _FakeGuard(queueworker._make_stall_handler(MEMBER, q, wlog))

    with pytest.raises(_Stop):
        queueworker.run_worker(MEMBER, wlog, watch=guard)

    assert "item_taken" in guard.notes, "埋点须同时喂给监护（判据的输入面）"
    assert seen["status_at_exit"] == [workq.STATUS_PENDING], (
        "卡死条目必须复位回 pending"
    )
    assert events.NEEDS_LEADER in seen["events_at_exit"], "静默变有声：须发事件"
    assert (
        int(os.getpid()) in seen["kills"]
    )  # 杀 worker 自身进程树（主线程卡在 communicate）
    assert seen["exit_code"] == queueworker.STALL_EXIT_CODE
    assert not events.read_all()[-1].get("subject", "").startswith("?")
    # 判死处置在 run 日志里留痕（事后可查为什么被杀）
    assert "stall_detected" in _stages(wlog)
    assert "watchdog 判卡死" in queues.items(MEMBER)[0].get("note", "")


# ===========================================================================
# B2. 孤儿自愈：不依赖下次获锁
# ===========================================================================
def _seed_orphan(member: str, *, age_min: float = 30) -> str:
    """造一条「在途」孤儿条目（无活 worker），返回其 id。"""
    item = queues.enqueue(member, text="出图", session="new")
    workq.mutate(
        queues.queue_path(member),
        str(item["id"]),
        status=workq.STATUS_IN_PROGRESS,
        started_at=_iso(age_min),
        created=_iso(10),
    )
    return str(item["id"])


def test_reclaim_orphans_resets_and_wakes(iso_state, monkeypatch):
    """success_criteria：孤儿回收去鸡生蛋——巡更面复位 pending **并唤醒 worker**。"""
    iid = _seed_orphan("figure")
    woken: list[str] = []
    monkeypatch.setattr(
        queueworker, "wake_member", lambda m, **k: (woken.append(m) or True, "stub")
    )

    out = queueworker.reclaim_orphans()

    assert out["reclaimed"].get("figure") == [iid]
    assert queues.items("figure")[0]["status"] == workq.STATUS_PENDING
    assert woken == ["figure"], "复位只是把活放回队列，还得有人来干"
    assert events.NEEDS_LEADER in [e["event"] for e in events.read_all()]
    # 幂等：已复位、worker 已叫过 → 第二轮空手
    assert queueworker.reclaim_orphans()["reclaimed"] == {}


def test_reclaim_orphans_leaves_live_worker_alone(iso_state, monkeypatch):
    """有活 worker（锁 running）时那些 in_progress 是它的在途，绝不碰。"""
    iid = _seed_orphan("figure")
    paths = queueworker.member_paths("figure")
    paths.lock.parent.mkdir(parents=True, exist_ok=True)
    assert drain.acquire_lock("state/queues-runs/figure/x.log", paths) is not None

    woken: list[str] = []
    monkeypatch.setattr(
        queueworker, "wake_member", lambda m, **k: (woken.append(m) or True, "stub")
    )
    out = queueworker.reclaim_orphans()

    assert out["reclaimed"] == {} and woken == []
    assert queues.items("figure")[0]["status"] == workq.STATUS_IN_PROGRESS
    assert queues.items("figure")[0]["id"] == iid
    drain.release_lock(paths=paths)


def test_reclaim_orphans_wake_cooldown(iso_state, monkeypatch):
    """刚叫过（run 目录有新鲜日志）就不再叫——防「起不来 → 每 tick 派生一次」暴走。"""
    _seed_orphan("figure")
    runs = queueworker.member_paths("figure").runs
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "20261015-090000.log").write_text(
        "[stage] lock_acquired\n", encoding="utf-8"
    )

    woken: list[str] = []
    monkeypatch.setattr(
        queueworker, "wake_member", lambda m, **k: (woken.append(m) or True, "stub")
    )
    out = queueworker.reclaim_orphans()

    assert out["reclaimed"], "孤儿仍要复位（这部分与唤醒冷却无关）"
    assert woken == [] and any("刚派生过" in n for n in out["notes"])


# ===========================================================================
# B3. communicate 超时杀子树
# ===========================================================================
def test_run_headless_timeout_kills_process_tree(monkeypatch):
    """success_criteria：``communicate`` 有超时且超时**杀子树**（不只杀直接子进程）。"""
    killed: list[int] = []
    monkeypatch.setattr(
        procs,
        "terminate_tree",
        lambda pid: (killed.append(int(pid)) or True, "fake"),
    )
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]

    rc, out, err = runner.run_headless(
        cmd, cwd=os.getcwd(), env=dict(os.environ), timeout_s=1
    )

    assert rc == -1 and "TIMEOUT" in err and "已杀进程树" in err
    assert killed and killed[0] > 0, f"超时未走 terminate_tree：{killed}"
    assert out == ""


# ===========================================================================
# B4. 巡更判据：有心跳却无进展 → PENDING-WORK
# ===========================================================================
def test_queue_pressure_flags_heartbeat_but_no_progress(iso_state, iso_registry):
    """success_criteria：queue_pressure 补「in_progress 超期且无进展」→ 疑似 worker 卡死。"""
    iid = _seed_orphan("figure", age_min=120)
    # 心跳新鲜（慢而不死面）：正是实案里「三道旧闸全判活着」的那一维
    backpressure.note_session(
        "figure", sid="sid-1", tier="t", purpose="p", now=time.time()
    )

    read = watch.queue_pressure()

    assert read["pending"] is True
    joined = "；".join(read["reasons"])
    assert iid in joined and "疑似 worker 卡死" in joined, read["reasons"]
    assert not read["stall"], "有心跳不该走旧 STALL 支（那是另一维）"


def test_queue_pressure_quiet_when_session_has_progress(
    iso_state, iso_registry, monkeypatch
):
    """在途会话**仍在长字符** ⇒ 不报（防误杀/误报：实测单跳可跑 130min）。"""
    _seed_orphan("figure", age_min=120)
    backpressure.note_session(
        "figure", sid="sid-1", tier="t", purpose="p", now=time.time()
    )
    monkeypatch.setattr(
        stallwatch, "inflight_sessions", lambda m: [{"sid": "sid-1", "pid": 999999}]
    )
    monkeypatch.setattr(stallwatch, "session_idle_s", lambda m, s, now: 5.0)

    read = watch.queue_pressure()

    assert "疑似 worker 卡死" not in "；".join(read["reasons"])
    assert read["pending"] is False and read["stall"] == []


# ===========================================================================
# B4'. 超龄锁的心跳免抢（stale_s 下调的前提）
# ===========================================================================
def _lock_file(tmp_path, *, age_min: float) -> drain.WorkerPaths:
    paths = drain.WorkerPaths(
        lock=tmp_path / "figure.lock",
        runs=tmp_path / "runs",
        cancel=tmp_path / "figure.cancel",
    )
    paths.lock.write_text(
        f'{{"pid": {os.getpid()}, "started": "{_iso(age_min)}", '
        '"run_log": "state/queues-runs/figure/x.log"}\n',
        encoding="utf-8",
    )
    return paths


def test_stale_lock_is_not_stealable_while_holder_heartbeats(tmp_path, monkeypatch):
    """超龄（>45min）**但持有者仍在心跳** ⇒ 慢而非死，不许抢锁（抢了就是双 worker）。

    这正是「stale_s 下调」得以安全成立的前提：没有这条免抢判据，45min 的一刀切会把
    10% 的正常跳（实测 p95=58min、max=130min）判死并放进第二个消费者。
    """
    import pysci.skills.orchestration.tools.settings as _settings

    monkeypatch.setattr(_settings, "SETTINGS_PATH", tmp_path / "s.toml")
    paths = _lock_file(tmp_path, age_min=50)

    assert drain.lock_status(paths)["state"] == "stale", "无心跳的超龄锁 = 死的"
    assert drain.acquire_lock("state/queues-runs/figure/x.log", paths) is not None

    paths2 = _lock_file(tmp_path, age_min=50)
    backpressure.note_session(
        "figure", sid="x", tier="t", purpose="queue-worker", now=time.time()
    )
    assert drain.lock_status(paths2)["state"] == "running", "心跳在 ⇒ 只是慢"
    assert drain.acquire_lock("state/queues-runs/figure/x.log", paths2) is None, (
        "心跳在场时 acquire 必须让位（防双 worker）"
    )


def test_short_stale_bound_within_member_timeout_ordering():
    """成文理由的两条不变式：卡死阈值 ≪ 锁龄上限 ≪ 成员 timeout（谁先动手说得清）。"""
    assert stallwatch.NO_SPAWN_S < stallwatch.NO_GROWTH_S < drain.LOCK_STALE_S
    default_timeout = Member(member_id="x", pod=Path("p")).timeout_s
    assert drain.LOCK_STALE_S <= default_timeout / 2, (
        "锁龄上限须显著低于成员 timeout，否则「超龄抢锁」会与「跳还在合法跑」打架"
    )


def test_watchdog_lock_keeps_long_stale_bound():
    """看门狗自己的锁不随 45min 下调（常驻监督者，无心跳证据可免抢 ⇒ 会被自己抢成双份）。"""
    from pysci.skills.orchestration.tools import watchdog as _wd

    assert _wd.WATCHDOG_PATHS.stale_s >= 6 * 3600
    assert _wd.WATCHDOG_LOCK_STALE_S == 6 * 3600


# ===========================================================================
# 回归护栏：默认阈值不得被调成「分钟级误杀」
# ===========================================================================
def test_default_thresholds_are_conservative():
    """阈值下限护栏：实测单跳 p90=41min/p95=58min/max=130min，
    「无增长」阈值不得低于 15min，否则正常长跳会被误杀（丢整跳工作）。"""
    assert stallwatch.NO_GROWTH_S >= 15 * 60
    assert stallwatch.NO_SPAWN_S >= 5 * 60  # 边界避让本身可正当睡满 300s
    assert stallwatch.PROBE_INTERVAL_S < stallwatch.NO_SPAWN_S
