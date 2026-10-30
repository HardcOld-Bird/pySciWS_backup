"""plan 续作会话选择：显式 ``--session`` 无条件优先（task-20261022-234407）。

缺陷形状：``cmd_plan_run`` / ``cmd_plan_advance`` 的会话选择写成
``sel = cur["session_id"][:8] if cur.get("session_id") else (args.session or "latest")``
——环节已有 session_id（blocked/failed 续作）时，显式传入的 ``--session new`` 被
无条件无视，死会话被复用。实证：job devops-20261022-231123 派发命令带 ``--session new``，
run 日志仍 ``resume=<旧 sid>``，随即再撞模型流故障（run_failed 后原会话本就不可复用）。

修复语义（两处同改）：显式 ``args.session`` 无条件优先（含 ``"new"``）；仅当未显式
传入时才维持续作复用 ``cur.session_id``，否则默认 ``latest``。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    events,
    plans,
    queues,
    queueworker,
    workq,
)

#: 假 worker 回填的会话 id（前 8 位 = 续作复用值）。
SID = "abcdef123456"


@pytest.fixture
def qface(iso_state, tmp_path, monkeypatch):
    """计划目录落 tmp（队列/worker 面由 iso_state 隔离）。"""
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    return plans_dir


def _write_plan(plans_dir: Path, plan_id: str, steps: list[dict]) -> None:
    import yaml

    (plans_dir / f"{plan_id}.plan.yaml").write_text(
        yaml.safe_dump(
            {"id": plan_id, "goal": "g", "steps": steps}, allow_unicode=True
        ),
        encoding="utf-8",
    )


def _run_args(plan_id: str, **over) -> SimpleNamespace:
    base = dict(
        id=plan_id,
        text="任务书全文",
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _fake_blocked_worker(member: str, **kw):
    """把**最新** pending 项以 blocked 收尾（模拟 blocked 续作场景，回填 sid）。

    队列侧口径同 queueworker：blocked 交付 = 条目 ``failed`` 终态 + ``result.kind=blocked``
    （plans 的结账按 kind 判环节 blocked）。
    """
    qp = queues.queue_path(member)
    pend = [i for i in workq.load(qp) if str(i.get("status")) == workq.STATUS_PENDING]
    if not pend:
        return (False, f"{member} 队列无 pending（无 job）")
    it = pend[-1]
    workq.mutate(qp, it["id"], status=workq.STATUS_IN_PROGRESS)
    it = workq.mutate(
        qp,
        it["id"],
        status=workq.STATUS_FAILED,
        result={
            "code": 2,
            "kind": "blocked",
            "body": "needs_decision",
            "checks": [],
            "member": member,
            "sid": SID,
            "duration_ms": 900,
        },
    )
    events.emit(
        events.JOB_DONE,
        subject=str(it["id"]),
        detail=f"{member} job 收尾：blocked",
        action="orch plan advance 推进",
    )
    return (True, f"{member} worker 已消化（stub）")


def _blocked_plan(qface: Path, plan_id: str) -> None:
    """建一份单环节计划并推进到 blocked（环节状态带 session_id）。"""
    _write_plan(
        qface,
        plan_id,
        [{"id": "s1", "member": "devops", "brief": "基建环节", "dirs": []}],
    )
    # blocked 交付 → 同步原语按「检出问题」返回 2（非 0 不代表没入队/没结账）
    assert plans.cmd_plan_run(_run_args(plan_id)) == 2
    state = json.loads((qface / f"{plan_id}.state.json").read_text(encoding="utf-8"))
    assert state["steps"][0]["status"] == "blocked"
    assert state["steps"][0]["session_id"] == SID


def _last_session(member: str) -> str:
    return str(queues.items(member)[-1].get("session") or "")


# ---------------------------------------------------------------------------
# (a) 环节有 session_id 且显式 --session new → 入队 session 为 "new"
# ---------------------------------------------------------------------------
def test_explicit_session_new_wins_over_recorded_session_on_run(qface, monkeypatch):
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_blocked_worker(member)
    )
    _blocked_plan(qface, "ps1")
    # 续作：显式 --session new 必须优先于旧 sid（run_failed 恢复清单推荐用法）
    assert plans.cmd_plan_run(_run_args("ps1", text="续作任务书", session="new")) == 2
    assert _last_session("devops") == "new"


def test_explicit_session_new_wins_over_recorded_session_on_advance(qface, monkeypatch):
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_blocked_worker(member)
    )
    _blocked_plan(qface, "ps2")
    assert (
        plans.cmd_plan_advance(_run_args("ps2", text="续作任务书", session="new")) == 0
    )
    assert _last_session("devops") == "new"


# ---------------------------------------------------------------------------
# (b) 不传 --session → 仍复用旧 sid（防语义回归）
# ---------------------------------------------------------------------------
def test_absent_session_still_resumes_recorded_session(qface, monkeypatch):
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_blocked_worker(member)
    )
    _blocked_plan(qface, "ps3")
    assert plans.cmd_plan_run(_run_args("ps3", text="续作任务书", session=None)) == 2
    assert _last_session("devops") == SID[:8]


def test_absent_session_advance_still_resumes_recorded_session(qface, monkeypatch):
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_blocked_worker(member)
    )
    _blocked_plan(qface, "ps4")
    assert (
        plans.cmd_plan_advance(_run_args("ps4", text="续作任务书", session=None)) == 0
    )
    assert _last_session("devops") == SID[:8]
