"""派发路径统一（用户裁定 2026-10-14）：plan run/adhoc 一律经成员队列。

`state/queues/` 自 m4 建成后长期空转——`plan run`/`plan adhoc` 仍是直派阻塞路径，
于是队列面板「皆空」、active-sessions live=0、STALL 判据全在说谎。本文件钉三件事：

1. **直派退役**：`cmd_plan_run` 入队 → 唤醒 worker → 等收尾 → 从条目还原交接报告
   （对外仍是同步的「推进到交接暂停」）；`do_dispatch` 直调只剩队列 worker 一个入口。
2. **监测覆盖**：面板计数含 `in_progress`（在途显名+起跑时刻），「皆空」只在双零时出现。
3. **事件显示**：已消费事件标注消费态、不再整行复读「建议处置」。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pysci.skills.orchestration.tools import (
    dispatch,
    events,
    plans,
    queues,
    queueworker,
    workq,
)
from pysci.skills.orchestration.tools.tui import modules

OK_BODY = "<result>交付 OK</result>"


def _ok_outcome():
    from pysci.skills.orchestration.tools.runner import Envelope

    return dispatch.DispatchOutcome(
        code=0,
        kind="result",
        member="sim",
        sid="sim-sid",
        body=OK_BODY,
        envelope=Envelope(duration_ms=1200),
    )


@pytest.fixture
def qface(tmp_path, monkeypatch):
    """计划目录 + 成员队列目录都落 tmp（成员队列原先指向真 state，测试会写脏运行面）。"""
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    qdir = tmp_path / "state" / "queues"
    qdir.mkdir(parents=True)
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(queues, "QUEUES_DIR", qdir)
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", tmp_path / "state")
    return plans_dir


def _write_plan(plans_dir: Path, plan_id: str, steps: list[dict]) -> None:
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
    """唤醒的等价物：把队首 pending 项**按 worker 的语义**收尾并发出完成事件。"""
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
                "body": OK_BODY,
                "checks": [],
                "member": member,
                "sid": "sim-sid",
                "duration_ms": 1200,
            },
        )
    events.emit(
        events.JOB_DONE,
        subject=str(it["id"]),
        detail=f"{member} job 收尾：done（result）",
        action="orch events drain 后接续下一份工作；orch plan advance 推进",
    )
    return (True, f"{member} worker 已消化（stub）")


def test_plan_run_goes_through_member_queue_end_to_end(qface, monkeypatch, capsys):
    """端到端：入队 → worker 消化 → 完成交尾 → plan run 还原结果并报交接（同步体验不变）。"""
    _write_plan(
        qface,
        "p1",
        [
            {"id": "s1", "member": "sim", "brief": "b1", "dirs": []},
            {"id": "s2", "member": "sim", "brief": "b2", "dirs": []},
        ],
    )
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )

    assert plans.cmd_plan_run(_run_args("p1")) == 0
    items = queues.items("sim")
    assert len(items) == 1, "plan run 必须入队（直派退役）"
    assert str(items[0].get("plan")) == "p1" and str(items[0].get("step")) == "s1"
    assert items[0]["status"] == workq.STATUS_DONE
    assert items[0]["result"]["code"] == 0

    state = json.loads((qface / "p1.state.json").read_text(encoding="utf-8"))
    assert state["steps"][0]["status"] == "done"
    assert state["steps"][0]["session_id"] == "sim-sid"

    out = capsys.readouterr().out
    assert "计划 p1：环节 s1 → done" in out and "交接决策点" in out


def test_plan_run_success_then_advance_queues_next_step(qface, monkeypatch):
    """事件驱动闭环不回归：run 收尾后 advance 落结账并**入队**下一环节（不直派）。"""
    _write_plan(
        qface,
        "p1",
        [
            {"id": "s1", "member": "sim", "brief": "b1", "dirs": []},
            {"id": "s2", "member": "sim", "brief": "b2", "dirs": []},
        ],
    )
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )
    plans.cmd_plan_run(_run_args("p1"))

    called: list[str] = []
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda member, **k: called.append(member) or _ok_outcome(),
    )
    assert plans.cmd_plan_advance(_run_args("p1", text="下一环节任务书")) == 0
    assert called == [], "advance 只准入队，不许直派"
    state = json.loads((qface / "p1.state.json").read_text(encoding="utf-8"))
    assert state["steps"][1]["status"] == "queued"
    assert [str(i.get("step")) for i in queues.items("sim")] == ["s1", "s2"]


def test_plan_run_no_direct_dispatch_call(qface, monkeypatch, capsys):
    """直派退役的**负面判据**：全程不许碰到 do_dispatch（只剩队列 worker 一个入口）。"""
    _write_plan(qface, "p1", [{"id": "s1", "member": "sim", "brief": "b1", "dirs": []}])
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda member, **k: (_ for _ in ()).throw(
            AssertionError("plan run 仍直派 do_dispatch")
        ),
    )
    assert plans.cmd_plan_run(_run_args("p1")) == 0


# ---------------------------------------------------------------------------
# ② 面板：在途条目显名+起跑时刻；「皆空」只在双零时出现
# ---------------------------------------------------------------------------
def test_water_lines_names_in_progress_with_start_time(qface):
    queues.enqueue("sim", text="a")
    qp = queues.queue_path("sim")
    workq.mutate(
        qp,
        workq.load(qp)[0]["id"],
        status=workq.STATUS_IN_PROGRESS,
        started_at="2026-10-14T14:05:00+08:00",
    )
    lines = "\n".join(queues.water_lines())
    assert "在途合计：pending=0 in_progress=1" in lines
    assert "在途 " in lines and "起于 2026-10-14T14:05:00+08:00" in lines
    assert "双零" not in lines, "有在途条目时绝不能说「皆空」"


def test_water_lines_double_zero_is_the_only_all_clear(qface):
    queues.enqueue("sim", text="a")
    qp = queues.queue_path("sim")
    workq.mutate(qp, workq.load(qp)[0]["id"], status=workq.STATUS_DONE)
    lines = "\n".join(queues.water_lines())
    assert "pending=0 in_progress=0" in lines and "双零" in lines
    assert "在途 " not in lines


def test_queues_count_includes_in_progress(qface, monkeypatch):
    """徽标水位要算在途条目（直派退役后成员队列=唯一在途真相面）。"""
    from pysci.skills.orchestration.tools import inspectd, workflow

    queues.enqueue("sim", text="a")
    qp = queues.queue_path("sim")
    workq.mutate(qp, workq.load(qp)[0]["id"], status=workq.STATUS_IN_PROGRESS)
    monkeypatch.setattr(workflow, "backlog_pending", lambda: [])
    monkeypatch.setattr(workflow, "deferred_count", lambda: 0)
    monkeypatch.setattr(inspectd, "stale_queue_items", lambda **k: [])
    assert modules._queues_count() >= 1


# ---------------------------------------------------------------------------
# ③ 事件显示：区分已消费/待排空，已消费不再复读「建议处置」
# ---------------------------------------------------------------------------
def test_event_status_marks_consumed_and_drops_action(iso_events):
    events.emit(events.JOB_DONE, "j1", "收尾", "orch plan advance 推进")
    fresh = events.status_lines()
    assert any("[待排空]" in ln and "orch plan advance 推进" in ln for ln in fresh)

    events.drain()
    digested = events.status_lines()
    assert any("[已消费]" in ln for ln in digested)
    assert not any("建议处置" in ln for ln in digested if "[已消费]" in ln), (
        "已消费事件不该再挂「建议处置」（历史指引别包装成待办）"
    )
    assert not any("[待排空]" in ln for ln in digested), "排空后不再有待排空行"
