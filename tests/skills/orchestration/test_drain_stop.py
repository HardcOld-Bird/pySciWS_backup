"""drain 协作式停止（backlog 20261010-drain-cooperative-stop）。

orch drain-stop 写 state/devops.cancel（指向当前锁持有者 pid）；drain 每项间隙消费
信号→干净退出（释放锁、写部分 .done 含 stopped=cooperative、cancel 自删）；陈旧信号
（指向他方 pid）清除且不误停新 worker。
"""

from __future__ import annotations

import json
import os

from pysci.skills.orchestration.tools import dispatch, drain, orch
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog


def _seed_running_lock(iso_state, monkeypatch, pid=4242):
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": pid, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda p: True)


def test_request_stop_running_writes_signal(iso_state, monkeypatch):
    _seed_running_lock(iso_state, monkeypatch)
    ok, msg = drain.request_stop()
    assert ok is True and "协作停止" in msg
    data = json.loads(drain.CANCEL_PATH.read_text(encoding="utf-8"))
    assert data["pid"] == 4242 and data["requested_at"]


def test_request_stop_idle_writes_nothing(iso_state):
    ok, msg = drain.request_stop()
    assert ok is False and "无需停止" in msg
    assert not drain.CANCEL_PATH.exists()  # 空闲不写——防残留信号误停下个 worker


def test_cancel_status(iso_state):
    assert drain.cancel_status() is None
    drain.CANCEL_PATH.write_text(
        json.dumps({"pid": 7, "requested_at": "t"}), encoding="utf-8"
    )
    assert drain.cancel_status()["pid"] == 7


def test_consume_stop_self_pid_true_and_deletes(iso_state):
    drain.CANCEL_PATH.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert drain._consume_stop_request() is True
    assert not drain.CANCEL_PATH.exists()


def test_consume_stop_foreign_pid_false_and_cleans(iso_state):
    drain.CANCEL_PATH.write_text(
        json.dumps({"pid": os.getpid() + 999_999}), encoding="utf-8"
    )
    assert drain._consume_stop_request() is False
    assert not drain.CANCEL_PATH.exists()


def test_consume_stop_corrupt_file_false_and_cleans(iso_state):
    drain.CANCEL_PATH.write_text("not json", encoding="utf-8")
    assert drain._consume_stop_request() is False
    assert not drain.CANCEL_PATH.exists()


def test_drain_stops_between_items_cleanly(iso_state, monkeypatch):
    """条目间隙消费信号：已处理项落账，未取项仍 pending，无孤儿 in_progress。"""
    seed_backlog(
        iso_state,
        [
            {"id": "b1", "status": "pending", "summary": "s1"},
            {"id": "b2", "status": "pending", "summary": "s2"},
        ],
    )
    calls = {"n": 0}

    def fake_dispatch(member, **kw):
        calls["n"] += 1
        if calls["n"] == 1:  # 处理 b1 期间组长敲了 drain-stop（目标=本 worker）
            drain.CANCEL_PATH.write_text(
                json.dumps({"pid": os.getpid()}), encoding="utf-8"
            )
        return DispatchOutcome(code=0, kind="result", body="done commit aaa1111")

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    assert summary["stopped"] == "cooperative"
    assert summary["count"] == 1
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["b1"]["status"] == "done"
    assert by_id["b2"]["status"] == "pending"
    assert not drain.CANCEL_PATH.exists()  # 信号已消费自删
    assert not drain.LOCK_PATH.exists()  # 锁已释放
    done = json.loads(run_log.with_suffix(".done").read_text(encoding="utf-8"))
    assert done["stopped"] == "cooperative"
    assert "协作停止信号" in run_log.read_text(encoding="utf-8")


def test_stale_signal_does_not_stop_new_worker(iso_state, monkeypatch):
    """被 kill worker 留下的陈旧信号：新 worker 清除之并照常消化全队列。"""
    drain.CANCEL_PATH.write_text(
        json.dumps({"pid": os.getpid() + 999_999}), encoding="utf-8"
    )
    seed_backlog(
        iso_state,
        [
            {"id": "b1", "status": "pending", "summary": "s1"},
            {"id": "b2", "status": "pending", "summary": "s2"},
        ],
    )
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body="done commit bbb2222"
        ),
    )
    summary = drain.drain_backlog(drain.new_run_log())
    assert "stopped" not in summary
    assert summary["count"] == 2
    assert not drain.CANCEL_PATH.exists()


def test_cli_drain_stop_wired(iso_state, monkeypatch):
    """orch drain-stop 端到端：CLI → request_stop → 信号落盘。"""
    _seed_running_lock(iso_state, monkeypatch)
    rc = orch.main(["drain-stop"])
    assert rc == 0
    assert json.loads(drain.CANCEL_PATH.read_text(encoding="utf-8"))["pid"] == 4242


def test_cli_drain_stop_idle_rc(iso_state, capsys):
    rc = orch.main(["drain-stop"])
    assert rc == 0
    assert "无需停止" in capsys.readouterr().out
