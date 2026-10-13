"""组长面两条 CLI：`orch requeue` / `orch resolve`（backlog 20261013-215909-leader）。

病灶：组长处理 needs_leader 条目缺干净 CLI——「重派」只能直调 drain 内部 `wake_devops`，
「销账」只能直改 `state/backlog.json`（drain 在跑时有并发覆盖风险）。本组钉住两条正门：
**起点状态受控**（只对 needs_leader 生效）、**唤醒被调用**、**原 blocked 事实不丢**。
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from pysci.skills.orchestration.tools import drain, orch

from .conftest import read_backlog, seed_backlog


def _seed(state, **kw) -> dict:
    item = {"id": "b1", "member": "leader", "summary": "s", "status": "needs_leader"}
    item.update(kw)
    seed_backlog(state, [item])
    return item


def _stub_wake(monkeypatch) -> list:
    """替换 drain.wake_devops（cmd_requeue 在函数内 `from .drain import wake_devops` 取）。"""
    calls: list = []
    monkeypatch.setattr(
        drain,
        "wake_devops",
        lambda **kw: (calls.append(kw), (True, "devops 已后台唤醒（stub）"))[1],
    )
    return calls


# ---------------------------------------------------------------------------
# ① requeue：needs_leader → pending 并唤醒 drain
# ---------------------------------------------------------------------------
def test_requeue_transitions_and_wakes(iso_state, monkeypatch):
    _seed(iso_state)
    calls = _stub_wake(monkeypatch)
    assert orch.main(["requeue", "b1"]) == 0
    assert read_backlog(iso_state)[0]["status"] == "pending"
    assert len(calls) == 1  # 唤醒确被调用（success_criteria 要的两件事之一）


def test_requeue_writes_note(iso_state, monkeypatch):
    _seed(iso_state, note="原失败全文")
    _stub_wake(monkeypatch)
    orch.main(["requeue", "b1", "--note", "授权已补"])
    assert read_backlog(iso_state)[0]["note"] == "授权已补"


def test_requeue_keeps_note_when_absent(iso_state, monkeypatch):
    """不传 --note 就保留原附注（drain 的失败全文）——不当成「清空」。"""
    _seed(iso_state, note="原失败全文")
    _stub_wake(monkeypatch)
    orch.main(["requeue", "b1"])
    assert read_backlog(iso_state)[0]["note"] == "原失败全文"


def test_requeue_preserves_awaiting_decision_marker(iso_state, monkeypatch):
    """回排不清 awaiting_decision/blocked_report：重排任务书的「原点名↔本次补齐」靠它们。"""
    _seed(iso_state, awaiting_decision=True, blocked_report="点名 A/B")
    _stub_wake(monkeypatch)
    orch.main(["requeue", "b1"])
    it = read_backlog(iso_state)[0]
    assert it["status"] == "pending"
    assert it["awaiting_decision"] is True and it["blocked_report"] == "点名 A/B"


def test_requeue_rejects_non_needs_leader_and_does_not_wake(iso_state, monkeypatch):
    """起点状态受控：pending 条目回排应 rc=2、状态不动、**不**唤醒。"""
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    calls = _stub_wake(monkeypatch)
    assert orch.main(["requeue", "b1"]) == 2
    assert read_backlog(iso_state)[0]["status"] == "pending"
    assert calls == []


def test_requeue_unknown_id(iso_state, monkeypatch):
    _seed(iso_state)
    _stub_wake(monkeypatch)
    assert orch.main(["requeue", "nope"]) == 2


# ---------------------------------------------------------------------------
# ② resolve：needs_leader → done（须 --note，原事实保留）
# ---------------------------------------------------------------------------
def test_resolve_transitions_to_done_preserving_facts(iso_state):
    _seed(
        iso_state,
        note="drain 跳过（交付=blocked）…",
        needs_leader_at="2026-10-13T00:00:00+08:00",
        blocked_report="点名 A/B",
    )
    assert orch.main(["resolve", "b1", "--note", "已由 20261013-xxxx 单覆盖"]) == 0
    it = read_backlog(iso_state)[0]
    assert it["status"] == "done"
    assert it["note"] == "已由 20261013-xxxx 单覆盖"
    # 原 blocked/needs_leader 事实不丢
    assert it["needs_leader_at"] == "2026-10-13T00:00:00+08:00"
    assert it["blocked_report"] == "点名 A/B"
    assert it.get("done_at")


def test_resolve_rejects_non_needs_leader(iso_state):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    assert orch.main(["resolve", "b1", "--note", "x"]) == 2
    assert read_backlog(iso_state)[0]["status"] == "pending"


def test_resolve_unknown_id(iso_state):
    _seed(iso_state)
    assert orch.main(["resolve", "nope", "--note", "x"]) == 2


def test_resolve_requires_note(iso_state):
    """--note 必填（销账留痕）：缺则 argparse 非零退出、状态不动。"""
    _seed(iso_state)
    with pytest.raises(SystemExit):
        orch.main(["resolve", "b1"])
    assert read_backlog(iso_state)[0]["status"] == "needs_leader"


# ---------------------------------------------------------------------------
# ③ 发现性：orch status 的 needs_leader 段把两条正门摆出来
# ---------------------------------------------------------------------------
def test_status_surfaces_requeue_resolve(iso_state, monkeypatch, capsys):
    _seed(iso_state)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "orch requeue" in out and "orch resolve" in out
