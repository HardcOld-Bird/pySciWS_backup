"""watchdog stop 语义（backlog 20261014-135935-devops）：**一次 stop 停全部实例**。

实测病体（2026-10-14 19:1x 现场）：``_watchdog-loop`` 会存在多个实例同跑（锁持有者 +
不持锁的**孤儿**，实测两实例共用同一 run-log）。旧 ``stop`` 只对锁持有者 pid 投递、
锁空闲时干脆不写信号——孤儿会把信号当「陈旧」清掉继续跑，于是「stop 一次」停不全。

修复 = **广播停止纪元**（``{"scope": "all", "requested_at": …}``）：
- 一次停止覆盖同刻**全部**实例（含孤儿）；锁空闲照样投递；
- 按各实例 ``started_at`` 比较——晚于纪元启动的**新实例不受影响**（看门狗调度器/下一跳
  照常拉起，即验收里的「或 1 后立刻 start」）；
- 消费方**不删文件**（删了会抢掉其他实例的信号——第一个醒来的实例把纪元吞掉，其余照旧
  存活），单 pid 路径的「消费即删」语义原样保留给 drain/队列。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import drain, watchdog


def _paths(iso_state, monkeypatch) -> drain.WorkerPaths:
    """watchdog 生命周期面落 iso_state（与 test_watch_late_wakeup 同构）。"""
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=iso_state / "watchdog.lock",
            runs=iso_state / "watchdog-runs",
            cancel=iso_state / "watchdog.cancel",
        ),
    )
    return watchdog.WATCHDOG_PATHS


def test_one_stop_reaches_every_instance_spares_newcomers(iso_state, monkeypatch):
    paths = _paths(iso_state, monkeypatch)
    ok, msg = drain.request_stop(paths, broadcast=True)
    assert ok is True and "停止纪元" in msg
    data = json.loads(paths.cancel.read_text(encoding="utf-8"))
    assert data["scope"] == "all" and data["requested_at"]

    # 早于纪元启动的两任实例：都消费（旧实现里只有持锁那位能收到）
    assert (
        drain._consume_stop_request(paths, started_at="2000-01-01T00:00:00+08:00")
        is True
    )
    # 消费方不删文件（否则第一个醒来的实例把纪元吞掉，其余收不到）
    assert paths.cancel.exists()
    assert (
        drain._consume_stop_request(paths, started_at="2000-01-01T00:00:00+08:00")
        is True
    )

    # 晚于纪元启动的新实例：不受影响，且不吞纪元（陈旧纪元不误停接棒者）
    assert (
        drain._consume_stop_request(paths, started_at="2099-01-01T00:00:00+08:00")
        is False
    )
    assert paths.cancel.exists()


def test_stop_when_lock_idle_still_delivers_to_orphans(iso_state, monkeypatch):
    """锁空闲（无持有者，孤儿可能在跑）也必须投递——旧实现此处返回 False 且不写信号。"""
    paths = _paths(iso_state, monkeypatch)
    assert not paths.lock.exists()
    ok, msg = watchdog.stop_watchdog()
    assert ok is True and "广播" in msg
    assert json.loads(paths.cancel.read_text(encoding="utf-8"))["scope"] == "all"
    # 孤儿（起于纪元前）收到停止
    assert (
        drain._consume_stop_request(paths, started_at="2000-01-01T00:00:00+08:00")
        is True
    )


def test_run_watchdog_honours_epoch_and_ignores_stale_one(iso_state, monkeypatch):
    """循环级：未来纪元 → 干净退出；陈旧纪元 → 不误停（状态非 cooperative、纪元未被吞）。"""
    paths = _paths(iso_state, monkeypatch)
    paths.runs.mkdir(parents=True, exist_ok=True)

    # (a) 纪元晚于本实例启动 → 立刻干净退出
    paths.cancel.write_text(
        json.dumps({"scope": "all", "requested_at": "2099-12-31T23:59:59+08:00"}),
        encoding="utf-8",
    )
    summary = watchdog.run_watchdog(paths.runs / "wd-a.log", sleeps=lambda s: None)
    assert summary.get("stopped") == "cooperative"

    # (b) 陈旧纪元（早于本实例启动）→ 不误停；用哨兵异常结束循环以断言未被纪元停掉
    paths.cancel.write_text(
        json.dumps({"scope": "all", "requested_at": "2000-01-01T00:00:00+08:00"}),
        encoding="utf-8",
    )

    class _Sentinel(Exception):
        pass

    def boom_sleep(_seconds):
        raise _Sentinel("terminate-loop")

    summary2 = watchdog.run_watchdog(paths.runs / "wd-b.log", sleeps=boom_sleep)
    assert summary2.get("stopped") != "cooperative", "陈旧纪元不得误停新实例"
    assert paths.cancel.exists(), "陈旧纪元不得被新实例吞掉"
    assert "terminate-loop" in str(summary2.get("error", ""))
