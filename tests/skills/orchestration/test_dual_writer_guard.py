"""双写者防护（backlog 20261015-140209-f3-writer-guard）。

实案（2026-10-15 job 20261015-122406）：批次派发**崩在「已派发、未结账」**处（归档 reply
FileNotFoundError，f1 已修 fail-soft）→ 条目坠落**重派** → 同一份工单出现**两个活跃写者**
（会话 6304 与重派会话抢写同一份 TUI 改进包），靠执行端自己认出才没互踩。本单补结构性防护：

- **派发前在途检测**：`writer_occupancy` 复用 sid 在途闸（pid 判活 + argv 核 sid），命中即
  **不派发**（延期回队 + 证据 note + NEEDS_LEADER 出声）；
- **孤儿复位前核验写者死活**：`orphans_guarded` 活者不复位不重派、死者正常复位；
- 两处决策都留痕（条目 note + run 日志埋点），可事后审计。

判活桩法沿用 ``test_sid_gate_audit``：``write_runtime`` 预置带 ``pid``/``pid_task`` 的会话 +
``procs.pid_alive``/``pid_is_session`` 双桩（只给 pid 不给 argv 核对＝把闸测成哑巴）。
"""

from __future__ import annotations

import json
import time

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    plans,
    queues,
    registry,
    workq,
)
from pysci.skills.orchestration.tools import (
    queueworker as qw,
)

from .conftest import FAKE_HOP_PID, write_runtime

MEMBER = "devops"
SID = "6f0cc8e5-1111-2222-3333-444455556666"
PLAN, STEP = "p-writer", "s1"
TASKBOOK = f"task-20261015-140000-{PLAN}-{STEP}.md"


@pytest.fixture
def face(iso_state, qface, tmp_path, monkeypatch):
    """隔离：注册表（含 devops 成员）/ 队列 / 计划 / 运行态会话池。"""
    reg = iso_state / "registry.json"
    reg.write_text(
        json.dumps(
            {
                "version": 1,
                "members": {
                    MEMBER: {
                        "pod": str(tmp_path / "pods" / MEMBER),
                        "max_turns": 3,
                        "timeout_s": 5,
                    }
                },
                "checks": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg)
    return iso_state


def _alive(monkeypatch, *, alive: bool) -> None:
    monkeypatch.setattr(
        dispatch.procs, "pid_alive", lambda pid: alive and pid == FAKE_HOP_PID
    )
    monkeypatch.setattr(
        dispatch.procs, "pid_is_session", lambda pid, sid: alive and sid == SID
    )


def _seed_session_pool(state, *, task: str = TASKBOOK) -> None:
    """预置一具带 pid/pid_task 的会话（＝派发瞬间 ``note_hop_pid`` 写下的事实）。"""
    write_runtime(
        state,
        {
            "version": 1,
            "members": {
                MEMBER: {
                    "sessions": [
                        {
                            "sid": SID,
                            "status": "active",
                            "last_active": "9999",
                            "hops": 2,
                            "pid": FAKE_HOP_PID,
                            "pid_task": task,
                            "ctx_ratio": 0.2,
                        }
                    ]
                }
            },
        },
    )


def _seed_queue(
    state, *, item_id="j-new", status=workq.STATUS_PENDING, session=SID
) -> dict:
    """入队一条 plan/step 条目（可指定状态/会话），返回条目。"""
    item = queues.enqueue(
        MEMBER, plan=PLAN, step=STEP, task_file="", session=session, item_id=item_id
    )
    if status != workq.STATUS_PENDING:
        workq.mutate(queues.queue_path(MEMBER), item_id, status=status)
    return item


def _q() -> object:
    return queues.queue_path(MEMBER)


# ---------------------------------------------------------------------------
# ① 派发前在途检测：活写者 → 占用证据（三锚之一：会话池 pid_task）
# ---------------------------------------------------------------------------
def test_occupancy_found_via_session_pool_pid_task(face, monkeypatch):
    _seed_session_pool(face)
    _alive(monkeypatch, alive=True)
    item = _seed_queue(face)

    occ = qw.writer_occupancy(MEMBER, item)

    assert occ is not None and occ["via"] == "session-pool"
    assert occ["pid"] == FAKE_HOP_PID and occ["sid"] == SID and occ["task"] == TASKBOOK


def test_dead_writer_is_not_occupancy(face, monkeypatch):
    """死写者（pid 判活不过）→ 不判占用（fail-open，不冤枉正常重派）。"""
    _seed_session_pool(face)
    _alive(monkeypatch, alive=False)
    item = _seed_queue(face)
    assert qw.writer_occupancy(MEMBER, item) is None


def test_argv_mismatch_is_not_occupancy(face, monkeypatch):
    """pid 活着但 argv 对不上 sid（pid 被复用）→ 不判占用（同 sid 在途闸口径）。"""
    _seed_session_pool(face)
    monkeypatch.setattr(dispatch.procs, "pid_alive", lambda pid: True)
    monkeypatch.setattr(dispatch.procs, "pid_is_session", lambda pid, sid: False)
    item = _seed_queue(face)
    assert qw.writer_occupancy(MEMBER, item) is None


def test_planless_item_is_never_flagged(face, monkeypatch):
    """计划外条目（adhoc/backlog 组包）无 plan/step 归属 → 不做此判（单写者路径不回归）。"""
    _seed_session_pool(face)
    _alive(monkeypatch, alive=True)
    item = queues.enqueue(MEMBER, task_file="", item_id="adhoc-1")
    assert qw.writer_occupancy(MEMBER, item) is None


def test_occupancy_via_other_inflight_queue_item(face, monkeypatch):
    """锚 2：同队列里**另一条同 plan/step 的在途条目**（其 session 指向活写者）。

    会话池里那条的 pid_task 故意不指向本环节 → 锚 1 不命中，证明锚 2 独立成立。
    """
    _seed_session_pool(face, task="task-other-thing.md")
    _alive(monkeypatch, alive=True)
    _seed_queue(face, item_id="j-old", status=workq.STATUS_IN_PROGRESS)
    fresh = _seed_queue(face, item_id="j-new")

    occ = qw.writer_occupancy(MEMBER, fresh)

    assert occ is not None and occ["via"] == "queue-item" and occ["sid"] == SID


def test_occupancy_via_plan_state_row(face, monkeypatch):
    """锚 3：plan state 该环节行的 ``session_id``（上一次派发实际用的整条 sid）。

    本条 session 留空、池里 pid_task 也不指向本环节 → 只有 plan 行能给出 sid。
    """
    _seed_session_pool(face, task="task-other-thing.md")
    _alive(monkeypatch, alive=True)
    (plans.PLANS_DIR / f"{PLAN}.state.json").write_text(
        json.dumps(
            {
                "plan_id": PLAN,
                "steps": [{"id": STEP, "status": "queued", "session_id": SID}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    item = _seed_queue(face, session="latest")

    occ = qw.writer_occupancy(MEMBER, item)

    assert occ is not None and occ["via"] == "plan-state"


# ---------------------------------------------------------------------------
# ② 拒绝派发：延期回队 + 证据 note（不丢、不抢、可审计）
# ---------------------------------------------------------------------------
def test_hold_defers_the_item_with_evidence_note(face, monkeypatch):
    _seed_session_pool(face)
    _alive(monkeypatch, alive=True)
    item = _seed_queue(face)
    occ = qw.writer_occupancy(MEMBER, item)
    before = time.time()

    qw._hold_for_writer(_q(), item, occ)

    row = next(i for i in workq.load(_q()) if i["id"] == "j-new")
    assert row["status"] == workq.STATUS_PENDING, "拒绝＝延期回队，不占 in_progress"
    assert float(row["defer_until"]) > before + qw.WRITER_HOLD_DEFER_S - 5
    for frag in (str(FAKE_HOP_PID), SID, TASKBOOK, "双写者防护"):
        assert frag in row["note"], f"note 必须留占用证据：{frag}"


# ---------------------------------------------------------------------------
# ③ 孤儿复位前核验：活者不复位、死者正常复位
# ---------------------------------------------------------------------------
def test_orphan_reset_deferred_when_writer_alive(face, monkeypatch):
    _seed_session_pool(face)
    _alive(monkeypatch, alive=True)
    _seed_queue(face, item_id="j-inflight", status=workq.STATUS_IN_PROGRESS)
    traced: list[str] = []

    reset, held = qw.orphans_guarded(
        _q(), MEMBER, trace=lambda s, **k: traced.append(s)
    )

    assert reset == [] and [h["id"] for h in held] == ["j-inflight"]
    assert traced == ["orphan_held_writer_alive"]
    row = next(i for i in workq.load(_q()) if i["id"] == "j-inflight")
    assert row["status"] == workq.STATUS_IN_PROGRESS, "活写者：**不复位不重派**"
    assert str(FAKE_HOP_PID) in row["note"] and "暂缓" in row["note"], "决策须留痕"


def test_orphan_reset_proceeds_when_writer_dead(face, monkeypatch):
    _seed_session_pool(face)
    _alive(monkeypatch, alive=False)
    _seed_queue(face, item_id="j-dead", status=workq.STATUS_IN_PROGRESS)

    reset, held = qw.orphans_guarded(_q(), MEMBER)

    assert reset == ["j-dead"] and held == []
    row = next(i for i in workq.load(_q()) if i["id"] == "j-dead")
    assert row["status"] == workq.STATUS_PENDING and "孤儿回收" in row["note"]


def test_reclaim_orphans_holds_when_writer_alive(face, monkeypatch):
    """巡更自愈整链：活写者的在途项**不被复位**，且 notes 出声（不重派）。"""
    _seed_session_pool(face)
    _alive(monkeypatch, alive=True)
    _seed_queue(face, item_id="j-inflight", status=workq.STATUS_IN_PROGRESS)

    out = qw.reclaim_orphans(wake=False)

    assert out["reclaimed"][MEMBER] == []
    assert [h["id"] for h in out["held"][MEMBER]] == ["j-inflight"]
    assert any("未复位" in n for n in out["notes"]), out["notes"]
    row = next(i for i in workq.load(_q()) if i["id"] == "j-inflight")
    assert row["status"] == workq.STATUS_IN_PROGRESS


def test_reclaim_orphans_resets_when_writer_dead(face, monkeypatch):
    """对照面：写者已死 → 照旧复位（自愈不回归）。"""
    _seed_session_pool(face)
    _alive(monkeypatch, alive=False)
    _seed_queue(face, item_id="j-dead", status=workq.STATUS_IN_PROGRESS)

    out = qw.reclaim_orphans(wake=False)

    assert out["reclaimed"][MEMBER] == ["j-dead"]
    assert "held" not in out
    row = next(i for i in workq.load(_q()) if i["id"] == "j-dead")
    assert row["status"] == workq.STATUS_PENDING


# ---------------------------------------------------------------------------
# ④ 接线：派发循环里真的过这道闸（改回裸派发即红）
# ---------------------------------------------------------------------------
def test_dispatch_loop_is_wired_to_the_guard():
    from pathlib import Path

    src = Path(qw.__file__).read_text(encoding="utf-8")
    assert "writer_occupancy(member, item" in src, "派发前须过在途写者闸"
    assert "_hold_for_writer(q, item, occ)" in src, "命中须延期回队而非派发"
    assert "workq.reset_orphans(q)" not in src, (
        "复位点须走 orphans_guarded（写者核验合一）"
    )
