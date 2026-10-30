"""devops 收编为 plan 可派发队列成员（task-20261014-devops-gap-d1）。

补 PSEP_2 点名的缺口（plan 不支持派发 devops 环节）：devops 进 QUEUE_MEMBERS，plan run /
plan advance / queues enqueue 三路全通；非队列成员拒绝保留；**双通道并存**（plan 环节
直派 kind=produce 与 backlog 组包 kind=backlog-batch 非二选一）——backlog-batch 既有行为
由 test_backlog_fifo / test_batch_digest 全量回归覆盖，此处只钉双通道的事件分流与 ADR 注记。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    events,
    orch,
    plans,
    queues,
    queueworker,
    workq,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome


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


def _fake_worker(member: str, **kw):
    """唤醒的等价物：把队首 pending 项收尾（done + JOB-DONE），与 test_queue_dispatch 同构。"""
    qp = queues.queue_path(member)
    items = workq.load(qp)
    if not items:
        return (False, f"{member} 队列为空（无 job）")
    it = items[0]
    if str(it.get("status")) == workq.STATUS_PENDING:
        workq.mutate(qp, it["id"], status=workq.STATUS_IN_PROGRESS)
        it = workq.mutate(
            qp,
            it["id"],
            status=workq.STATUS_DONE,
            result={
                "code": 0,
                "kind": "result",
                "body": "ok",
                "checks": [],
                "member": member,
                "sid": "devops-sid",
                "duration_ms": 900,
            },
        )
    events.emit(
        events.JOB_DONE,
        subject=str(it["id"]),
        detail=f"{member} job 收尾：done（result）",
        action="orch plan advance 推进",
    )
    return (True, f"{member} worker 已消化（stub）")


# ---------------------------------------------------------------------------
# ① QUEUE_MEMBERS 含 devops + ADR 注记（双通道语义与推翻理由成文）
# ---------------------------------------------------------------------------
def test_queue_members_include_devops_with_adr():
    assert "devops" in queues.QUEUE_MEMBERS
    src = (
        PROJECT_ROOT
        / "src"
        / "pysci"
        / "skills"
        / "orchestration"
        / "tools"
        / "queues.py"
    ).read_text(encoding="utf-8")
    assert "ADR" in src and "双通道" in src and "非二选一" in src
    assert "kind=produce" in src and "backlog-batch" in src
    assert "推翻旧决策" in src and "为什么" in src


# ---------------------------------------------------------------------------
# ② plan advance 对 member=devops 环节：入队 + 结账回读
# ---------------------------------------------------------------------------
def test_plan_devops_step_enqueues_and_settles(qface, monkeypatch):
    _write_plan(
        qface,
        "pd1",
        [
            {"id": "s1", "member": "devops", "brief": "基建环节", "dirs": []},
            {"id": "s2", "member": "devops", "brief": "第二环节", "dirs": []},
        ],
    )
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )
    assert plans.cmd_plan_run(_run_args("pd1")) == 0
    items = queues.items("devops")
    assert len(items) == 1
    assert str(items[0].get("plan")) == "pd1" and str(items[0].get("step")) == "s1"
    assert items[0]["status"] == workq.STATUS_DONE
    state = json.loads((qface / "pd1.state.json").read_text(encoding="utf-8"))
    assert state["steps"][0]["status"] == "done"
    assert state["steps"][0]["session_id"] == "devops-sid"

    # advance：结账回读（_queue_item_for_step）+ 入队下一 devops 环节
    assert plans.cmd_plan_advance(_run_args("pd1", text="下一环节任务书")) == 0
    assert [str(i.get("step")) for i in queues.items("devops")] == ["s1", "s2"]
    state = json.loads((qface / "pd1.state.json").read_text(encoding="utf-8"))
    assert state["steps"][1]["status"] == "queued"


# ---------------------------------------------------------------------------
# ③ queues enqueue devops 放行（旧闸 rc=2 退场）
# ---------------------------------------------------------------------------
def test_queues_enqueue_devops_accepted(qface, capsys):
    rc = orch.main(["queues", "enqueue", "devops", "--text", "一条 devops 环节任务"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "非队列成员" not in out
    assert any(
        str(i.get("task")) == "一条 devops 环节任务" for i in queues.items("devops")
    )


# ---------------------------------------------------------------------------
# ④ 非队列成员仍被拒（两条闸都保留）
# ---------------------------------------------------------------------------
def test_bogus_member_rejected_by_both_gates(qface, monkeypatch, capsys):
    rc = orch.main(["queues", "enqueue", "bogus", "--text", "x"])
    assert rc == 2
    assert "非队列成员" in capsys.readouterr().out

    _write_plan(
        qface, "pb", [{"id": "s1", "member": "bogus", "brief": "b", "dirs": []}]
    )
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )
    assert plans.cmd_plan_run(_run_args("pb")) == 2
    assert "非队列成员" in capsys.readouterr().out
    assert queues.items("devops") == [] and queues.items("bogus") == []


# ---------------------------------------------------------------------------
# ⑤ 双通道事件分流：plan 环节 job 走 JOB-DONE；backlog 批次保留 QUEUE-EMPTY
# ---------------------------------------------------------------------------
def _run_worker_once(iso_state):
    wpaths = queueworker.member_paths("devops")
    wpaths.runs.mkdir(parents=True, exist_ok=True)
    return queueworker.run_worker("devops", queueworker.new_run_log(wpaths))


def test_plan_step_job_emits_job_done_not_queue_empty(iso_state, monkeypatch):
    queues.enqueue("devops", text="环节任务", kind="produce", slug="step")
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda m, **k: DispatchOutcome(code=0, kind="result", body="ok"),
    )
    _run_worker_once(iso_state)
    names = [str(e.get("event")) for e in events.read_after(0)]
    assert events.JOB_DONE in names
    assert events.QUEUE_EMPTY not in names, (
        "plan 环节 job 不叠发 QUEUE-EMPTY（双通道分流）"
    )


def test_backlog_batch_run_keeps_queue_empty(iso_state, monkeypatch):
    queues.enqueue(
        "devops",
        text="批次任务",
        kind=drain.BACKLOG_BATCH_KIND,
        slug="batch",
        extra={
            "backlog": {
                "seed": {"id": "b1", "summary": "s", "member": "devops"},
                "group": [],
                "ids": ["b1"],
                "seed_id": "b1",
            }
        },
    )
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda m, **k: DispatchOutcome(code=0, kind="result", body="ok commit abc"),
    )
    _run_worker_once(iso_state)
    names = [str(e.get("event")) for e in events.read_after(0)]
    assert events.QUEUE_EMPTY in names, "backlog 批次消化完的知会保留（不回归）"


def test_devops_step_with_review_spawns_reviewer_child(iso_state, monkeypatch):
    """全链一致性：devops 环节带 review:true 时照常投 reviewer 子 job（generic 链路）。"""
    host = queues.enqueue(
        "devops", text="带审环节", kind="produce", review=True, slug="rv"
    )
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda m, **k: DispatchOutcome(code=0, kind="result", body="ok"),
    )
    _run_worker_once(iso_state)
    kids = queues.review_children(host["id"], host_member="devops")
    assert kids and kids[0].get("kind") == "review"
    assert str(kids[0].get("parent")) == str(host["id"])
