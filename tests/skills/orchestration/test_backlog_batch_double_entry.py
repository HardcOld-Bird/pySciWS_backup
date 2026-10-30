"""backlog 组包双记账根治（backlog 20261016-224211）。

现象（实证 2026-10-16 22:3x）：一张组包 job 一次消化了 215944/215945/215946（交付留档
明示三项均已实施并合并），但同刻 devops 队列里 215945/215946 又各自有**独立条目**
（22:08:24 / 22:17:04，均晚于组包 job 开跑）——同一 id 被执行两次，各烧一跳容量。

根因：**组包项按设计保持 pending**（不连坐，留待下轮自选），于是下一轮计划器把它们当
**新种子**再取一次。三面合一（建议单的三条方向都落一处实现）：

① 计划器取批前排除「在飞批次已覆盖」的 id —— 源头不再产生重复条目；
② 组包入队时把仍**待跑**的重复条目就地销账（方向①: 撤销其独立条目）；
③ worker 取条目时若覆盖 id **已全部终态** → 就地销账、**不派生会话**（方向③，兜存量）。
"""

from __future__ import annotations

import time

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    queues,
    queueworker,
    workq,
)

from .conftest import read_backlog, seed_backlog


def _qpath():
    return queues.queue_path("devops")


def _enqueue_batch(ids, *, seed_id: str, status: str = "pending") -> str:
    """造一条 ``backlog-batch`` 队列条目（携覆盖 id 元数据），返回条目 id。"""
    item = queues.enqueue(
        "devops",
        kind=drain.BACKLOG_BATCH_KIND,
        text="批次正文",
        slug=seed_id,
        extra={
            "backlog": {
                "seed": {"id": seed_id},
                "group": [],
                "ids": list(ids),
                "seed_id": seed_id,
            }
        },
    )
    if status != workq.STATUS_PENDING:
        workq.mutate(_qpath(), str(item["id"]), status=status)
    return str(item["id"])


def _item(item_id: str) -> dict:
    return next(i for i in workq.load(_qpath()) if str(i["id"]) == item_id)


# ---------------------------------------------------------------------------
# ① 在飞批次覆盖面读数
# ---------------------------------------------------------------------------
def test_inflight_batch_ids_collects_coverage(iso_state):
    """只收 ``pending``/``in_progress`` 的批次条目（终态条目已不承诺任何 id）。"""
    _enqueue_batch(["X", "Y"], seed_id="X")
    _enqueue_batch(["Z"], seed_id="Z", status=workq.STATUS_IN_PROGRESS)
    _enqueue_batch(["W"], seed_id="W", status=workq.STATUS_DONE)
    assert drain.inflight_batch_ids() == {"X", "Y", "Z"}


def test_deferred_batch_entry_does_not_block_next_seed(iso_state):
    """**延期待重试**的条目不算在飞：它已把覆盖 id 交还 backlog（否则同队后续条目被挡住）。

    实测回归（backlog 20261016-224211 的自伤面）：把延期条目仍计作承诺面，
    ``test_bandwidth_defers_and_keeps_going`` 的 b2 不再被取 → 延期支停链。
    """
    iid = _enqueue_batch(["X"], seed_id="X")
    workq.defer_item(
        _qpath(), iid, until=time.time() + 300, count=1, elapsed_s=1.0, note="t"
    )
    assert drain.inflight_batch_ids() == set()


def test_inflight_batch_ids_ignores_non_batch_kind(iso_state):
    """普通 produce 条目没有覆盖面（不误伤 plan 直派条目）。"""
    queues.enqueue("devops", text="普通一跳")
    assert drain.inflight_batch_ids() == set()


# ---------------------------------------------------------------------------
# ② 计划器：排除在飞批次已覆盖的 id（复现缺陷 + 修好）
# ---------------------------------------------------------------------------
def _seed_two(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "X", "status": "pending", "member": "m"},
            {"id": "Y", "status": "pending", "member": "m"},
        ],
    )


def test_take_batch_without_exclude_repeats_inflight_seed(iso_state):
    """**缺陷复现**：不排除 → 被在飞批次覆盖的 X 又被当新种子取走（双记账的成因）。"""
    _seed_two(iso_state)
    _enqueue_batch(["X"], seed_id="X")  # X 已被一个在飞批次承诺
    from pysci.skills.orchestration.tools import workflow

    seed, _group = workflow.backlog_take_batch(4)
    assert seed is not None and seed["id"] == "X", "旧行为：重复取同一 id"


def test_take_batch_with_exclude_skips_inflight_seed(iso_state):
    """**修好**：排除后种子跳到 Y，X 保持 pending（不重复承诺）。"""
    _seed_two(iso_state)
    _enqueue_batch(["X"], seed_id="X")
    from pysci.skills.orchestration.tools import workflow

    seed, _group = workflow.backlog_take_batch(4, exclude=drain.inflight_batch_ids())
    assert seed is not None and seed["id"] == "Y", seed
    on_disk = {i["id"]: i["status"] for i in read_backlog(iso_state)}
    assert on_disk["X"] == "pending", "被在飞批次覆盖的 id 不得被再次拉成 in_progress"


def test_take_batch_exclude_also_filters_group_members(iso_state):
    """排除面同时作用于**组包**侧（否则 X 会作为组包项混进另一批）。"""
    _seed_two(iso_state)
    _enqueue_batch(["Y"], seed_id="Y")
    from pysci.skills.orchestration.tools import workflow

    seed, group = workflow.backlog_take_batch(4, exclude=drain.inflight_batch_ids())
    assert seed is not None and seed["id"] == "X"
    assert [g["id"] for g in group] == [], "Y 已被在飞批次覆盖，不得再组包"


# ---------------------------------------------------------------------------
# ③ 入队去重：仍待跑的重复条目就地销账（不派跳）
# ---------------------------------------------------------------------------
def test_supersede_settles_duplicate_pending_entry(iso_state):
    dup = _enqueue_batch(["X"], seed_id="X")
    keep = _enqueue_batch(["X", "Y"], seed_id="X")
    dropped = drain.supersede_batch_entries({"X"}, keep=keep)
    assert dropped == [dup]
    it = _item(dup)
    assert it["status"] == workq.STATUS_DONE
    assert it["result"]["deduped"] == ["X"]
    assert _item(keep)["status"] == workq.STATUS_PENDING, "新批条目不动"


def test_supersede_does_not_touch_in_progress(iso_state):
    """在跑的条目**不杀**（杀了即造孤儿）——其 worker 已持锁。"""
    running = _enqueue_batch(["X"], seed_id="X", status=workq.STATUS_IN_PROGRESS)
    assert drain.supersede_batch_entries({"X"}) == []
    assert _item(running)["status"] == workq.STATUS_IN_PROGRESS


def test_supersede_ignores_disjoint_entries(iso_state):
    other = _enqueue_batch(["Z"], seed_id="Z")
    assert drain.supersede_batch_entries({"X"}) == []
    assert _item(other)["status"] == workq.STATUS_PENDING


# ---------------------------------------------------------------------------
# ④ 端到端：重复条目**不派生会话**，且不计入待办（criterion ②/③）
# ---------------------------------------------------------------------------
def _run_worker():
    wpaths = queueworker.member_paths("devops")
    wpaths.runs.mkdir(parents=True, exist_ok=True)
    return queueworker.run_worker("devops", queueworker.new_run_log(wpaths))


def test_duplicate_entry_is_settled_without_dispatch(iso_state, monkeypatch):
    """覆盖 id 已全部 done 的重复条目 → 就地销账、**do_dispatch 零调用**。"""
    seed_backlog(iso_state, [{"id": "X", "status": "done", "member": "m"}])
    dup = _enqueue_batch(["X"], seed_id="X")

    called: list = []

    def boom(*a, **k):
        called.append((a, k))
        raise AssertionError("重复条目不得派跳")

    monkeypatch.setattr(dispatch, "do_dispatch", boom)
    _run_worker()

    assert called == [], "重复条目派生了一次会话"
    it = _item(dup)
    assert it["status"] == workq.STATUS_DONE
    assert "重复批次条目" in str(it["result"]["body"])


def test_deduped_duplicate_not_counted_as_pending(iso_state, monkeypatch):
    """criterion ③：销账后 `queues list` 水位不把它算作「待办」（待=0）。"""
    seed_backlog(iso_state, [{"id": "X", "status": "done", "member": "m"}])
    _enqueue_batch(["X"], seed_id="X")
    monkeypatch.setattr(dispatch, "do_dispatch", lambda *a, **k: None)
    _run_worker()
    line = next(ln for ln in queues.water_lines() if ln.strip().startswith("[devops]"))
    assert "待=0" in line, line


def test_legit_batch_still_dispatches(iso_state, monkeypatch):
    """**反例护栏**：覆盖 id 仍处 pending 的正常批次照常派发（别把好条目也销了）。"""
    seed_backlog(iso_state, [{"id": "X", "status": "pending", "member": "m"}])
    _enqueue_batch(["X"], seed_id="X")
    called: list = []

    def fake(*a, **k):
        called.append(k)
        return None  # worker 侧不校验交付形状（本用例只看「是否派跳」）

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    _run_worker()
    assert len(called) == 1, "正常批次必须派发一次"


@pytest.mark.parametrize("status", [workq.STATUS_FAILED, "blocked"])
def test_partially_terminal_ids_still_dispatch(iso_state, monkeypatch, status):
    """覆盖 id 中**任一**非 done（failed/blocked）→ 不判重复、照常派发（不静默丢活）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "X", "status": "done", "member": "m"},
            {"id": "Y", "status": status, "member": "m"},
        ],
    )
    _enqueue_batch(["X", "Y"], seed_id="Y")
    called: list = []
    monkeypatch.setattr(
        dispatch, "do_dispatch", lambda *a, **k: called.append(k) or None
    )
    _run_worker()
    assert len(called) == 1, f"Y={status} 未终态，必须派发"
