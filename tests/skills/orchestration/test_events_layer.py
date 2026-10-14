"""统一事件/唤醒层 + plan advance 事件驱动（README §12，M5）。

覆盖任务书验收：worker 完成→JOB-DONE；宿主+review 合并只发一次 STEP-REVIEW-DONE；
队空发 QUEUE-EMPTY（devops/bencher）；events drain 一次排空且幂等；plan advance 入队即
返回；双 plan 并行（同成员队列串行、异成员并行）；orch status 读事件水位。

真 LLM 派发一律 mock ``dispatch.do_dispatch``；事件/队列/计划全部落 tmp 隔离面。
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from pysci.skills.orchestration.tools import (
    benchd,
    dispatch,
    drain,
    events,
    orch,
    plans,
    queues,
    queueworker,
    watch,
)

from .conftest import drain_once, seed_backlog


class _Out:
    """dispatch.DispatchOutcome 的最小替身。"""

    def __init__(self, code: int = 0, kind: str = "result", body: str = "ok"):
        self.code, self.kind, self.body, self.error = code, kind, body, ""


@pytest.fixture
def state(tmp_path, monkeypatch):
    """隔离：成员队列 / 计划目录 / worker 根 / 事件流四面同落 tmp。"""
    root = tmp_path / "state"
    monkeypatch.setattr(queues, "QUEUES_DIR", root / "queues")
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", root)
    monkeypatch.setattr(plans, "PLANS_DIR", root / "plans")
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", root / "events-seen.json")
    return root


def _mock_dispatch(monkeypatch, *, code: int = 0, kind: str = "result"):
    calls: list[str] = []

    def fake(member, **kw):
        calls.append(str(member))
        return _Out(code=code, kind=kind if code == 0 else "blocked")

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    return calls


def _write_plan(root, pid, steps):
    """落一份计划 YAML（steps: [(id, member, review), ...]）。"""
    (root / "plans").mkdir(parents=True, exist_ok=True)
    body = [f"id: {pid}", f"goal: {pid} 目标", "steps:"]
    for sid, member, review in steps:
        body += [
            f"  - id: {sid}",
            f"    member: {member}",
            f"    brief: {sid} 粗描述",
            f"    review: {'true' if review else 'false'}",
        ]
    (root / "plans" / f"{pid}.plan.yaml").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# ① 生产者直发：JOB-DONE / STEP-REVIEW-DONE
# ---------------------------------------------------------------------------
def test_job_done_emitted_on_worker_completion(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    queues.enqueue("figure", text="甲", root=state)
    queueworker.run_worker("figure", state / "r.log", root=state)
    evs = events.read_all()
    assert [e["event"] for e in evs] == ["JOB-DONE"]
    assert evs[0]["subject"].startswith("figure-")
    assert "figure" in evs[0]["detail"]


def test_review_merge_emits_step_review_done_exactly_once(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    host = queues.enqueue("writing", text="宿主", review=True, root=state)
    queueworker.run_worker("writing", state / "a.log", root=state)
    # 宿主持 review：交付后**不**发 JOB-DONE（合并单元未完成）
    assert [e["event"] for e in events.read_all()] == []

    queueworker.run_worker("reviewer", state / "b.log", root=state)
    evs = events.read_all()
    assert [e["event"] for e in evs] == ["STEP-REVIEW-DONE"]
    assert evs[0]["subject"] == host["id"]

    # 幂等：再消化一次空 reviewer 队列不再重复发
    queueworker.run_worker("reviewer", state / "c.log", root=state)
    assert len([e for e in events.read_all() if e["event"] == "STEP-REVIEW-DONE"]) == 1


def test_failed_job_still_emits_job_done(state, monkeypatch):
    _mock_dispatch(monkeypatch, code=2)
    queues.enqueue("theory", text="会失败", root=state)
    queueworker.run_worker("theory", state / "r.log", root=state)
    assert [e["event"] for e in events.read_all()] == ["JOB-DONE"]


# ---------------------------------------------------------------------------
# ② 队空发 QUEUE-EMPTY（devops / bencher）
# ---------------------------------------------------------------------------
def test_devops_drain_emits_queue_empty(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s"}])
    # 完成唤醒由**执行侧**（devops FIFO worker）在退出时发出（backlog-fifo-20261014）：
    # 计划器的 dry 支不发消费事实，故驱动一次真消化（假派发）让 worker 排空队列。
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: dispatch.DispatchOutcome(
            code=0, kind="result", body="done commit abc1234def"
        ),
    )
    drain_once()
    evs = [e for e in events.read_all() if e["event"] == "QUEUE-EMPTY"]
    assert len(evs) == 1 and "devops" in evs[0]["detail"]


def test_bencher_drain_emits_queue_empty(iso_bench):
    benchd.queue_save([])
    benchd.bench_drain(benchd.new_run_log(benchd.BENCH_PATHS), dry=True)
    evs = [e for e in events.read_all() if e["event"] == "QUEUE-EMPTY"]
    assert len(evs) == 1 and "bencher" in evs[0]["detail"]


# ---------------------------------------------------------------------------
# ③ events drain：一次排空 + 幂等
# ---------------------------------------------------------------------------
def test_events_drain_empties_all_and_is_idempotent(state):
    events.emit(events.JOB_DONE, "j1", "a", "x")
    events.emit(events.QUEUE_EMPTY, "r1", "b", "y")
    first = events.drain()
    assert [e["event"] for e in first] == ["JOB-DONE", "QUEUE-EMPTY"]
    assert events.drain() == []  # 幂等
    assert events.unacked() == []


def test_events_cli_drain_clears_all(state):
    events.emit(events.JOB_DONE, "j1", "a", "x")
    assert orch.main(["events", "drain"]) == 0
    assert events.unacked() == []
    assert orch.main(["events", "list"]) == 0  # 观察面可读


# ---------------------------------------------------------------------------
# ④ watch 从统一事件流唤醒（TASKBOOK-NEW 收编 + 生产者事件）
# ---------------------------------------------------------------------------
def test_watch_fires_on_stream_events(state):
    events.emit(events.JOB_DONE, "figure-1", "figure job 收尾", "advance")
    cur = {"events_seq": events.latest_seq()}
    out = watch.detect_events({"events_seq": 0}, cur)
    assert [e["event"] for e in out] == ["JOB-DONE"]
    # 水位线未推进 → 不重复报
    assert watch.detect_events(cur, cur) == []


def test_watch_ignores_mirrored_legacy_types_in_stream(state):
    """差分已覆盖的 legacy 类型即使出现在流里也不经流回读（防双触发）。"""
    events.emit(events.CRASH, "run-x", "死了", "接管")
    out = watch.detect_events({"events_seq": 0}, {"events_seq": events.latest_seq()})
    assert out == []


# ---------------------------------------------------------------------------
# ⑤ plan advance：入队即返回 + 事件驱动闭环 + 多 plan 并行
# ---------------------------------------------------------------------------
class _Args:
    def __init__(self, **kw):
        self.id = kw["id"]
        self.text = kw.get("text", "任务书")
        self.task = kw.get("task")
        self.session = kw.get("session", "latest")
        self.model_tier = kw.get("model_tier")
        self.effort = kw.get("effort")
        self.dirs = kw.get("dirs")
        self.dry = kw.get("dry", True)


def test_plan_advance_enqueues_and_returns(state, monkeypatch):
    calls = _mock_dispatch(monkeypatch)
    _write_plan(state, "p1", [("s1", "figure", False)])
    assert plans.cmd_plan_advance(_Args(id="p1")) == 0
    assert calls == []  # 入队即返回：不真派发
    items = queues.items("figure", root=state)
    assert len(items) == 1 and items[0]["step"] == "s1" and items[0]["plan"] == "p1"
    assert [s["status"] for s in plans.load_state("p1")["steps"]] == ["queued"]


def test_plan_advance_loop_finalizes_and_moves_on(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    _write_plan(state, "p1", [("s1", "figure", False), ("s2", "writing", False)])
    plans.cmd_plan_advance(_Args(id="p1"))
    queueworker.run_worker("figure", state / "a.log", root=state)  # 消化 s1
    assert [e["event"] for e in events.read_all()] == ["JOB-DONE"]
    events.drain()

    assert plans.cmd_plan_advance(_Args(id="p1")) == 0  # 收尾 s1 → 入队 s2
    steps = {s["id"]: s["status"] for s in plans.load_state("p1")["steps"]}
    assert steps == {"s1": "done", "s2": "queued"}
    assert [i["step"] for i in queues.items("writing", root=state)] == ["s2"]


def test_plan_advance_waits_while_job_in_flight(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    _write_plan(state, "p1", [("s1", "figure", False), ("s2", "writing", False)])
    plans.cmd_plan_advance(_Args(id="p1"))
    # job 还在 pending → 再 advance 不推进（等事件）
    assert plans.cmd_plan_advance(_Args(id="p1")) == 0
    assert len(queues.items("figure", root=state)) == 1
    assert queues.items("writing", root=state) == []


def test_plan_advance_wires_review_attach(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    _write_plan(state, "p1", [("s1", "writing", True)])
    plans.cmd_plan_advance(_Args(id="p1"))
    item = queues.items("writing", root=state)[0]
    assert item["review"] is True and item["kind"] == "produce"


def test_two_plans_parallel_same_member_serializes(state, monkeypatch):
    calls = _mock_dispatch(monkeypatch)
    _write_plan(state, "pA", [("s1", "figure", False)])
    _write_plan(state, "pB", [("s1", "writing", False)])
    _write_plan(state, "pC", [("s1", "figure", False)])
    assert plans.cmd_plan_advance(_Args(id="pA")) == 0
    assert plans.cmd_plan_advance(_Args(id="pB")) == 0
    assert plans.cmd_plan_advance(_Args(id="pC")) == 0
    assert calls == []  # 各自入队即返回，互不阻塞
    # 不同成员并行（各自队列各一条）；同成员 FIFO 串行（figure 两条堆积，不互斥）
    assert len(queues.items("figure", root=state)) == 2
    assert len(queues.items("writing", root=state)) == 1


def test_plan_advance_rejects_non_queue_member(state):
    _write_plan(state, "p1", [("s1", "devops", False)])
    assert plans.cmd_plan_advance(_Args(id="p1")) == 2


# ---------------------------------------------------------------------------
# ⑥ orch status 读事件水位（观察面接入）
# ---------------------------------------------------------------------------
def test_status_reads_event_water(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    events.emit(events.JOB_DONE, "figure-1", "figure job 收尾", "advance")
    assert orch.cmd_status(Namespace()) == 0
    out = capsys.readouterr().out
    assert "== 统一事件流" in out and "事件水位 seq=1" in out
