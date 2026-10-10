"""orch watch 事件等待器（backlog 20261010-orch-watch）。

覆盖四类事件的**代码判定**（差分规则 + 去重身份）、阻塞轮询的命中/超时两条出口、
``state/watch.json`` 基线与心跳、``orch status`` 补账的一次性清算语义，以及三处
[NEXT] 挂载指令注入（wake_devops / approve / dispatch --from-backlog）。

不真等：:func:`watch.run_watch` 的 ``sleep``/``monotonic`` 为注入口，配合假时钟；
不碰真实 state：``iso_state`` 夹具已把 lock/runs/backlog/watch.json 全部重定向。
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import drain, orch, watch, workflow

from .conftest import read_backlog, seed_backlog, seed_suggestion


# ---------------------------------------------------------------------------
# 构造观察面（快照字段与 watch.snapshot 输出同构）
# ---------------------------------------------------------------------------
def _snap(**over) -> dict:
    """中性快照基座：空闲锁、无摘要、无 needs_leader、无额度。"""
    base = {
        "ts": watch._now(),
        "lock_state": "idle",
        "lock_pid": 0,
        "lock_run": "",
        "crash_run": "",
        "done": "",
        "done_stopped": "",
        "done_count": 0,
        "done_elapsed_s": "?",
        "needs_leader": [],
        "pending": 0,
        "quota_run": "",
    }
    base.update(over)
    return base


def _seq(*items):
    """假 snapshot 供给器：依次给出 items，耗尽后重复最后一项（免测试数调用次数）。"""

    def _f():
        _seq.i = getattr(_seq, "i", 0)
        idx = _seq.i
        _seq.i = idx + 1
        return items[idx] if idx < len(items) else items[-1]

    _seq.i = 0
    return _f


class Clock:
    """假时钟：sleep 直接推进 monotonic，测试免真等。"""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _seed_done(state, name: str, data: dict) -> dict:
    """落 ``devops-runs/<name>.done`` 摘要，返回写入的数据。"""
    d = state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"count": 1, "elapsed_s": 42.0, "items": [], **data}
    (d / f"{name}.done").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _seed_run_log(state, name: str, text: str) -> None:
    d = state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.log").write_text(text, encoding="utf-8")


def _seed_lock(state, run: str = "r1", *, pid: int = 4242, started=None) -> str:
    """落 drain 锁（run_log 用 tmp 绝对路径串：``PROJECT_ROOT / 绝对路径`` 即其本身）。"""
    log = state / "devops-runs" / f"{run}.log"
    (state / "devops.lock").write_text(
        json.dumps(
            {
                "pid": pid,
                "started": started or drain._now(),
                "run_log": str(log),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(log)


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------
def test_snapshot_idle_state(iso_state):
    seed_backlog(
        iso_state, [{"id": "a", "status": "pending"}, {"id": "b", "status": "x"}]
    )
    s = watch.snapshot()
    assert s["lock_state"] == "idle"
    assert s["crash_run"] == "" and s["done"] == "" and s["quota_run"] == ""
    assert s["pending"] == 1 and s["needs_leader"] == []


def test_snapshot_reads_needs_leader_and_crash(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "stuck", "status": "needs_leader"}])
    _seed_lock(iso_state, "r9")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)  # 持有者已死且无 .done
    s = watch.snapshot()
    assert s["needs_leader"] == ["stuck"]
    assert s["lock_state"] == "stale" and s["crash_run"] == "r9"


def test_snapshot_stale_with_done_is_not_crash(iso_state, monkeypatch):
    _seed_lock(iso_state, "r8")
    _seed_done(iso_state, "r8", {})  # 该 run 干净收尾，只是锁残留
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    assert watch.snapshot()["crash_run"] == ""


# ---------------------------------------------------------------------------
# 事件判定（差分 + 去重身份）
# ---------------------------------------------------------------------------
def test_detect_queue_complete_on_new_done(iso_state):
    ev = watch.detect_events(_snap(), _snap(done="20261010-1", done_count=3))
    assert [e["event"] for e in ev] == ["QUEUE-COMPLETE"]
    assert ev[0]["subject"] == "20261010-1"
    assert "3 项" in ev[0]["detail"]


def test_detect_no_repeat_for_known_done(iso_state):
    cur = _snap(done="r1", done_count=2)
    assert watch.detect_events(cur, dict(cur)) == []


def test_detect_queue_complete_suppressed_when_that_run_still_locked(iso_state):
    # 摘要与在持锁同名 → 该 run 尚未收尾（不应算完成）
    cur = _snap(done="r1", lock_state="running", lock_run="r1")
    assert watch.detect_events(_snap(), cur) == []


def test_detect_needs_leader_only_new_ids(iso_state):
    base = _snap(needs_leader=["old"])
    cur = _snap(needs_leader=["old", "new1", "new2"])
    ev = watch.detect_events(base, cur)
    assert [e["subject"] for e in ev] == ["new1", "new2"]
    assert all(e["event"] == "NEEDS-LEADER" for e in ev)


def test_detect_crash(iso_state):
    ev = watch.detect_events(
        _snap(), _snap(crash_run="r7", lock_state="stale", lock_pid=99)
    )
    assert ev[0]["event"] == "CRASH" and ev[0]["subject"] == "r7"
    assert "99" in ev[0]["detail"]
    # 同一崩溃 run 已入基线 → 不重复打扰
    assert watch.detect_events(_snap(crash_run="r7"), _snap(crash_run="r7")) == []
    # 新的崩溃 run → 再报
    assert (
        watch.detect_events(_snap(crash_run="r7"), _snap(crash_run="r8"))[0]["event"]
        == "CRASH"
    )


def test_detect_quota_from_done_summary(iso_state):
    ev = watch.detect_events(
        _snap(),
        _snap(quota_run="r3", done="r3", done_stopped="quota_exhausted"),
    )
    quota = [e for e in ev if e["event"] == "QUOTA-STOP"]
    assert len(quota) == 1 and quota[0]["subject"] == "r3"
    # 已入基线的同一 run 不再重复报
    assert (
        watch.detect_events(
            _snap(quota_run="r3", done="r3", done_stopped="quota_exhausted"),
            _snap(quota_run="r3", done="r3", done_stopped="quota_exhausted"),
        )
        == []
    )


def test_detect_quota_from_run_log_before_done(iso_state, monkeypatch):
    """drain 打 [Q] 行在前、写 .done 在后 → watch 早一拍就能唤醒（不必等摘要落盘）。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 锁持有者仍存活
    _seed_run_log(iso_state, "r5", "[16:00:00] [Q] 批次 x 额度耗尽 → 触发全局停止\n")
    # 无锁、无摘要时无从归属，不误报（否则陈旧日志会让每次 arm 都立刻命中）
    assert watch.snapshot()["quota_run"] == ""
    _seed_lock(iso_state, "r5")  # 锁指向该 run → 认定额度类全局停止正在发生
    s = watch.snapshot()
    assert s["quota_run"] == "r5"
    assert watch.detect_events(_snap(), s)[0]["event"] == "QUOTA-STOP"


def test_quota_re_only_matches_quota_signals(iso_state, monkeypatch):
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    _seed_run_log(iso_state, "r6", "[16:00:00] [!] 批次 x 失败 → 种子 needs_leader\n")
    _seed_lock(iso_state, "r6")
    assert watch.snapshot()["quota_run"] == ""


def test_detect_priority_crash_beats_rest(iso_state):
    cur = _snap(
        crash_run="r1",
        quota_run="r1",
        needs_leader=["n1"],
        done="r1",
        lock_state="stale",
    )
    ev = watch.detect_events(_snap(), cur)
    assert [e["event"] for e in ev] == [
        "CRASH",
        "QUOTA-STOP",
        "NEEDS-LEADER",
        "QUEUE-COMPLETE",
    ]


def test_format_event_is_single_line():
    line = watch.format_event(
        {"event": "CRASH", "subject": "r1", "detail": "d", "action": "a"}
    )
    assert "\n" not in line and line.startswith("CRASH r1 ")


# ---------------------------------------------------------------------------
# 轮询：命中 / 超时
# ---------------------------------------------------------------------------
def test_run_watch_returns_event_line_and_persists(iso_state, monkeypatch):
    monkeypatch.setattr(
        watch,
        "snapshot",
        _seq(_snap(), _snap(done="run-a", done_count=2, lock_state="idle")),
    )
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
    assert out[0].startswith("QUEUE-COMPLETE run-a")
    assert any("重挂：pysci-orch watch" in line for line in out)  # 按需重挂提示
    st = watch.load_state()
    assert st["last_exit"] == "event"
    assert st["fired"][0]["event"] == "QUEUE-COMPLETE"
    assert st["base"]["done"] == "run-a"  # 基线推进到事件当帧


def test_run_watch_timeout_exits_zero(iso_state, monkeypatch):
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(pending=2)))
    out: list[str] = []
    clock = Clock()
    rc = watch.run_watch(
        timeout_s=120,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=out.append,
    )
    assert rc == 0
    assert out[0].startswith("TIMEOUT")
    assert "pending=2" in out[0]
    assert watch.load_state()["last_exit"] == "timeout"
    assert clock.t == 120.0  # 恰好睡满，不超时一秒


def test_run_watch_no_event_does_not_record_fired(iso_state, monkeypatch):
    monkeypatch.setattr(watch, "snapshot", _seq(_snap()))
    clock = Clock()
    watch.run_watch(
        timeout_s=60,
        interval_s=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        echo=lambda *a: None,
    )
    assert watch.load_state()["fired"] == []


def test_clamp_interval_bounds():
    assert watch.clamp_interval(None) == watch.DEFAULT_INTERVAL_S
    assert watch.clamp_interval(600) == watch.MAX_INTERVAL_S  # 超上限会漏短命崩溃窗口
    assert watch.clamp_interval(0) == watch.DEFAULT_INTERVAL_S
    assert watch.clamp_interval(-5) == watch.DEFAULT_INTERVAL_S
    assert watch.clamp_interval(1) == watch.MIN_INTERVAL_S


# ---------------------------------------------------------------------------
# 状态落盘 / 心跳 / status 补账
# ---------------------------------------------------------------------------
def test_arm_writes_readable_state(iso_state):
    st = watch.arm(timeout_s=90, interval_s=30)
    raw = json.loads((iso_state / "watch.json").read_text(encoding="utf-8"))
    assert raw["timeout_s"] == 90 and raw["interval_s"] == 30
    assert set(st["base"]) >= {"lock_state", "done", "needs_leader", "quota_run"}


def test_mounted_follows_heartbeat(iso_state):
    st = watch.arm()
    assert watch.mounted(st) is True
    st["heartbeat_at"] = (
        datetime.now().astimezone() - timedelta(seconds=st["interval_s"] * 4)
    ).isoformat(timespec="seconds")
    assert watch.mounted(st) is False
    assert watch.mounted({}) is False


def test_reconcile_skipped_while_mounted(iso_state, monkeypatch):
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(), _snap(done="run-x")))
    watch.arm()  # 心跳新鲜 = watch 仍挂载，事件归它的原生通知负责
    events, settled = watch.reconcile()
    assert (events, settled) == ([], False)
    assert watch.load_state()["base"]["done"] == ""  # 不抢账、不动基线


def test_reconcile_settles_once_after_watch_died(iso_state, monkeypatch):
    monkeypatch.setattr(
        watch, "snapshot", _seq(_snap(), _snap(done="run-y", done_count=1))
    )
    st = watch.arm()
    st["heartbeat_at"] = ""  # 会话结束、watch 已死
    watch.save_state(st)
    events, settled = watch.reconcile()
    assert settled is True
    assert [e["event"] for e in events] == ["QUEUE-COMPLETE"]
    # 一次性清算：基线已推进，再 status 不重复刷屏
    assert watch.reconcile() == ([], True)


def test_reconcile_without_state_arms_baseline(iso_state, monkeypatch):
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(pending=3)))
    assert watch.reconcile() == ([], False)
    assert watch.load_state()["base"]["pending"] == 3


def test_status_lines_reports_backlog_events(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(
        watch,
        "snapshot",
        _seq(_snap(), _snap(needs_leader=["stuck-1"])),
    )
    st = watch.arm()
    st["heartbeat_at"] = ""
    watch.save_state(st)
    for line in watch.status_lines():
        print(line)
    out = capsys.readouterr().out
    assert "未挂载" in out and "补账 1 起事件" in out and "NEEDS-LEADER stuck-1" in out


def test_status_lines_shows_mounted_watch(iso_state):
    watch.arm()
    assert any("已挂载" in line for line in watch.status_lines())


def test_status_first_baseline_is_not_reported_as_mounted(
    iso_state, monkeypatch, capsys
):
    """status 自建基线 ≠ 有人挂载：不得谎报「已挂载」（会让组长以为不用等通知）。"""
    monkeypatch.setattr(watch, "snapshot", _seq(_snap(), _snap()))
    assert "首次运行" in watch.status_lines()[0]
    second = watch.status_lines()[0]
    assert "未挂载" in second and "已挂载" not in second
    assert watch.mounted(watch.load_state()) is False


def test_status_lines_just_exited_is_not_mounted(iso_state, monkeypatch):
    """心跳新鲜但已命中退出 → 不能说「已挂载」（会把「刚结束」读成「还在等」）。"""
    monkeypatch.setattr(watch, "snapshot", _seq(_snap()))
    st = watch.arm()
    st["last_exit"] = "event"
    watch.save_state(st)
    line = watch.status_lines()[0]
    assert "已结束" in line and "已挂载" not in line and "重挂" in line
    st = watch.load_state()
    st["last_exit"] = ""  # 新一轮 arm 后的正常挂载态
    watch.save_state(st)
    assert "已挂载" in watch.status_lines()[0]


def test_load_state_tolerates_corruption(iso_state):
    (iso_state / "watch.json").write_text("{not json", encoding="utf-8")
    assert watch.load_state() == {}
    assert watch.mounted() is False


# ---------------------------------------------------------------------------
# [NEXT] 挂载指令注入（唤醒/采纳/交办三处，代码保证不靠纪律）
# ---------------------------------------------------------------------------
def test_wake_devops_message_carries_mount_lines(iso_state, monkeypatch):
    monkeypatch.setattr(drain, "_spawn_detached", lambda argv, **kw: None)
    spawned, msg = drain.wake_devops()
    assert spawned is True
    assert "pysci-orch watch --timeout" in msg
    assert "[NEXT]" in msg and "run_in_background" in msg


def test_wake_devops_mount_lines_even_when_busy(iso_state, monkeypatch):
    _seed_lock(iso_state, "busy")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    spawned, msg = drain.wake_devops()
    assert spawned is False and "pysci-orch watch" in msg


def test_approve_output_injects_mount_command(iso_state, monkeypatch, capsys):
    sid = seed_suggestion(iso_state)
    monkeypatch.setattr(drain, "_spawn_detached", lambda argv, **kw: None)
    rc = workflow.cmd_approve(
        Namespace(suggestion_id=sid, note="采纳", no_wake=False, effort=None)
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pysci-orch watch --timeout" in out
    assert "QUEUE-COMPLETE" in out  # 事件类型一并告知


def test_dispatch_from_backlog_injects_mount_command(iso_state, monkeypatch, capsys):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    calls: list[dict] = []

    class _Stub:
        code = 0
        kind = "result"
        body = "done"
        error = ""

    def fake_dispatch(member, **kw):
        calls.append({"member": member, **kw})
        return _Stub()

    monkeypatch.setattr(orch, "do_dispatch", fake_dispatch)
    rc = orch.cmd_dispatch(
        Namespace(
            member="devops",
            text=None,
            task=None,
            from_backlog=True,
            name=None,
            session="latest",
            effort=None,
            model_tier=None,
            max_turns=None,
            timeout=None,
            dirs=None,
            plan=None,
            step=None,
            no_checks=False,
            no_retry=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pysci-orch watch --timeout" in out
    assert read_backlog(iso_state)[0]["status"] == "done"


def test_mount_lines_header_toggle_and_indent():
    headed = watch.mount_lines()
    bare = watch.mount_lines(header=False, indent="  ")
    assert headed[0].startswith("[NEXT]")
    assert not bare[0].startswith("[NEXT]") and bare[0].startswith("  挂载事件等待器")
    assert all(line.startswith("  ") for line in bare)


# ---------------------------------------------------------------------------
# CLI 接线
# ---------------------------------------------------------------------------
def test_cli_watch_subcommand_routes_with_defaults(iso_state, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(watch, "run_watch", lambda **kw: seen.append(kw) or 0)
    assert orch.main(["watch"]) == 0
    assert seen[0] == {
        "timeout_s": watch.DEFAULT_TIMEOUT_S,
        "interval_s": watch.DEFAULT_INTERVAL_S,
    }
    assert orch.main(["watch", "--timeout", "7", "--interval", "3"]) == 0
    assert seen[1] == {"timeout_s": 7, "interval_s": 3}


def test_cli_watch_appears_in_command_surface(capsys):
    """命令面自证：``--help`` 的子命令清单含 watch（组长/文档据此寻址）。"""
    import pytest

    with pytest.raises(SystemExit):
        orch.main(["--help"])
    assert "watch" in capsys.readouterr().out
