"""approve 自动唤醒：spawn 派生 / 锁忙不重复 spawn / --no-wake 跳过 / 全链到 .done。

覆盖任务书验证要求 2（approve→唤醒→drain→.done 全链，以 _spawn_detached 同步模拟
分离子进程 + --dry 避免真派发）与要求 3（锁忙时不 spawn 第二 worker）。
"""

from __future__ import annotations

import json
from argparse import Namespace

from pysci.skills.orchestration.tools import drain, workflow

from .conftest import read_backlog, seed_backlog, seed_suggestion


def _record_spawns(monkeypatch):
    calls = []
    monkeypatch.setattr(drain, "_spawn_detached", lambda argv, **kw: calls.append(argv))
    return calls


def test_wake_spawns_detached_when_idle(iso_state, monkeypatch):
    calls = _record_spawns(monkeypatch)
    spawned, msg = drain.wake_devops()
    assert spawned is True
    assert "已后台唤醒" in msg and "devops-runs" in msg
    assert len(calls) == 1
    argv = calls[0]
    assert "_drain-devops" in argv and "--run-log" in argv
    # 内部命令经 sys.executable -m 调用（从任意 cwd 可用）
    assert argv[1] == "-m"
    assert argv[2] == "pysci.skills.orchestration.tools.orch"


def test_wake_skips_when_lock_busy(iso_state, monkeypatch):
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    calls = _record_spawns(monkeypatch)
    spawned, msg = drain.wake_devops()
    assert spawned is False
    assert calls == []  # 锁忙 → 不重复 spawn
    assert "正在处理队列" in msg


def test_wake_passes_dry_flag(iso_state, monkeypatch):
    calls = _record_spawns(monkeypatch)
    monkeypatch.setenv(drain.DRY_ENV, "1")
    drain.wake_devops()
    assert "--dry" in calls[0]


def test_approve_wakes_and_enqueues(iso_state, monkeypatch, capsys):
    sid = seed_suggestion(iso_state)
    calls = _record_spawns(monkeypatch)
    rc = workflow.cmd_approve(Namespace(suggestion_id=sid, note="采纳", no_wake=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "已后台唤醒" in out
    assert len(calls) == 1
    items = read_backlog(iso_state)
    assert items[-1]["id"] == sid and items[-1]["status"] == "pending"


def test_approve_no_wake_skips_spawn(iso_state, monkeypatch, capsys):
    sid = seed_suggestion(iso_state)
    calls = _record_spawns(monkeypatch)
    rc = workflow.cmd_approve(Namespace(suggestion_id=sid, note="x", no_wake=True))
    assert rc == 0
    assert calls == []
    assert "已后台唤醒" not in capsys.readouterr().out
    # 仍入队（只是不唤醒）
    assert read_backlog(iso_state)[-1]["status"] == "pending"


def test_approve_full_chain_to_done(iso_state, monkeypatch):
    """approve → 自动唤醒 → drain（dry）消化全部 pending → 写 .done。"""
    sid = seed_suggestion(iso_state)
    seed_backlog(iso_state, [{"id": "pre", "status": "pending", "summary": "既有项"}])

    def fake_spawn(argv, **kw):
        # 同步模拟分离子进程立即跑完 drain（dry 由 DRY_ENV 注入 argv）
        ns = Namespace(run_log=argv[argv.index("--run-log") + 1], dry="--dry" in argv)
        drain.cmd_drain_devops(ns)

    monkeypatch.setattr(drain, "_spawn_detached", fake_spawn)
    monkeypatch.setenv(drain.DRY_ENV, "1")

    rc = workflow.cmd_approve(Namespace(suggestion_id=sid, note="采纳", no_wake=False))
    assert rc == 0

    statuses = {i["id"]: i["status"] for i in read_backlog(iso_state)}
    assert statuses["pre"] == "done"  # 既有 pending 被消化
    assert statuses[sid] == "done"  # approve 新增项亦被同一 drain 消化
    dones = list((iso_state / "devops-runs").glob("*.done"))
    assert len(dones) == 1
    summary = json.loads(dones[0].read_text(encoding="utf-8"))
    assert summary["count"] == 2
    # drain 结束已释放锁
    assert not (iso_state / "devops.lock").exists()
