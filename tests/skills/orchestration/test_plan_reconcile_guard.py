"""``_reconcile`` 账实保护（backlog 20261015-135000-f2-ledger）。

实案：``c4`` 已派发（queued）却被 amend 对账**回滚成 pending**——「已完成的工作显示待派发」，
组长只能手工账实修正（state.json 里那句「账实修正（组长 2026-10-15）…状态被 amend 对账误重置」
即现场留痕）。

本文件钉对账的**允许动作集合**：增补新叶子（pending）／删幽灵行／**已 started 一律带过**。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import plans


def _plan_nested(parent: str, inner: str, leaf: str) -> dict:
    """``<parent>.<inner>.<leaf>`` 三层嵌套叶子（点路径 = 参数拼起来的形态）。"""
    return {
        "id": "p",
        "goal": "g",
        "steps": [
            {
                "id": parent,
                "mode": "serial",
                "children": [
                    {
                        "id": inner,
                        "mode": "serial",
                        "children": [{"id": leaf, "member": "devops"}],
                    }
                ],
            }
        ],
    }


def test_started_status_survives_a_path_rescope():
    """c4 同款：superstep 重排（``C.c4`` → ``BC.C.c4``）后**不得**回滚成 pending。"""
    plan = _plan_nested("BC", "C", "c4")
    state = {
        "plan_id": "p",
        "steps": [
            {
                "id": "C.c4",
                "status": "queued",
                "session_id": "sid-x",
                "job_id": "job-1",
            }
        ],
    }

    row = plans._reconcile(plan, state)["steps"][0]

    assert row["id"] == "BC.C.c4", "点路径随 YAML 走"
    assert row["status"] == "queued", "已 started 的状态不许被对账打回 pending"
    assert row["session_id"] == "sid-x" and row["job_id"] == "job-1", (
        "会话/作业归属也是账实"
    )


def test_ambiguous_same_leaf_name_does_not_guess():
    """同名歧义（``X.c4`` 与 ``Y.c4`` 都 started → 新路径 ``A.c4``/``B.c4``）→ 保守 pending。"""
    plan = {
        "id": "p",
        "goal": "g",
        "steps": [
            {
                "id": "G",
                "mode": "parallel",
                "children": [
                    {
                        "id": "A",
                        "mode": "serial",
                        "children": [{"id": "c4", "member": "x"}],
                    },
                    {
                        "id": "B",
                        "mode": "serial",
                        "children": [{"id": "c4", "member": "y"}],
                    },
                ],
            }
        ],
    }
    state = {
        "plan_id": "p",
        "steps": [
            {"id": "X.c4", "status": "queued"},
            {"id": "Y.c4", "status": "done"},
        ],
    }

    rows = {r["id"]: r for r in plans._reconcile(plan, state)["steps"]}

    assert rows["G.A.c4"]["status"] == "pending", "歧义不猜：宁可显示待派发也不冒领"
    assert rows["G.B.c4"]["status"] == "pending"


def test_matched_id_keeps_status_and_new_leaf_is_pending():
    plan = _plan_nested("G", "A", "a1")
    state = {"plan_id": "p", "steps": [{"id": "G.A.a1", "status": "done"}]}

    row = plans._reconcile(plan, state)["steps"][0]

    assert row["status"] == "done"
    plan2 = {
        "id": "p",
        "goal": "g",
        "steps": [
            {
                "id": "G",
                "mode": "serial",
                "children": [
                    {
                        "id": "A",
                        "mode": "serial",
                        "children": [{"id": "a1", "member": "x"}],
                    },
                    {
                        "id": "B",
                        "mode": "serial",
                        "children": [{"id": "b1", "member": "y"}],
                    },
                ],
            }
        ],
    }
    rows = {r["id"]: r for r in plans._reconcile(plan2, state)["steps"]}
    assert rows["G.A.a1"]["status"] == "done"
    assert rows["G.B.b1"]["status"] == "pending", "新叶子置 pending"


def test_ghost_rows_are_dropped():
    """YAML 里已不存在的行不留在 state（对账只增补/删幽灵）——未 started 的幽灵才允许被删。"""
    plan = _plan_nested("G", "A", "a1")
    state = {
        "plan_id": "p",
        "steps": [
            {"id": "G.A.a1", "status": "done"},
            {"id": "G.A.ghost", "status": "pending"},
        ],
    }

    ids = [r["id"] for r in plans._reconcile(plan, state)["steps"]]

    assert ids == ["G.A.a1"]


def test_extra_fields_are_carried_not_rebuilt():
    """evidence/job_id/note 也是账实：对账**整体带过**既有行（原实现只挑五个键重建，
    每次对账都静默丢掉自办环节的 evidence）。"""
    plan = _plan_nested("G", "A", "a1")
    state = {
        "plan_id": "p",
        "steps": [
            {
                "id": "G.A.a1",
                "status": "done",
                "evidence": "reports/a1.md",
                "job_id": "job-9",
                "note": "账实修正",
            }
        ],
    }

    row = plans._reconcile(plan, state)["steps"][0]

    assert row["evidence"] == "reports/a1.md"
    assert row["job_id"] == "job-9"
    assert row["note"] == "账实修正"
