"""superstep 嵌套编排引擎（PSEP_3 §2，M-A）：schema 扁平化 / 递归就绪谓词 / 多 job 收尾 / 三栏清单。

存量扁平 plan 行为不变由既有套件全量回归（1504 passed）保证；本文件钉新面：

① PlanTree 谓词单测（serial=前兄弟 done〔完成级〕、parallel=前兄弟 started〔**派发级**〕、
   递归向上、聚合状态由叶子推导）；② 嵌套 plan 的 state.json 叶子点路径扁平化与 amend 增删
   兼容；③ advance/run 按就绪谓词推进（含「平行分支派发后即可派发靠后分支」）；④ 多 job
   乱序收尾；⑤ [NEXT] 三栏工作清单；⑥ run 等**全部在途** job 收尾。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from pysci.skills.orchestration.tools import (
    events,
    plans,
    queues,
    queueworker,
    workq,
)
from pysci.skills.orchestration.tools.planflow import PlanTree

OK = "改动完成 commit abc1234 已合并回 main"


@pytest.fixture
def qface(iso_state, tmp_path, monkeypatch):
    d = tmp_path / "plans"
    d.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    return d


def _write_plan(d: Path, plan_id: str, steps: list[dict]) -> None:
    (d / f"{plan_id}.plan.yaml").write_text(
        yaml.safe_dump(
            {"id": plan_id, "goal": "g", "steps": steps}, allow_unicode=True
        ),
        encoding="utf-8",
    )


def _args(plan_id: str, **over) -> Namespace:
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
        file="",
        force=False,
    )
    base.update(over)
    return Namespace(**base)


def _fake_worker(member: str, **kw):
    """把该成员队列里**全部 pending 条目**收尾（done + JOB-DONE）——多 job 并行时队列会积压多条。"""
    qp = queues.queue_path(member)
    pend = [i for i in workq.load(qp) if str(i.get("status")) == workq.STATUS_PENDING]
    if not pend:
        return (False, f"{member} 队列无 pending")
    for it in pend:
        workq.mutate(qp, it["id"], status=workq.STATUS_IN_PROGRESS)
        workq.mutate(
            qp,
            it["id"],
            status=workq.STATUS_DONE,
            result={
                "code": 0,
                "kind": "result",
                "body": OK,
                "checks": [],
                "member": member,
            },
        )
        events.emit(
            events.JOB_DONE,
            subject=str(it["id"]),
            detail=f"{member} job 收尾：done（result）",
            action="orch plan advance 推进",
        )
    return (True, f"{member} worker 已消化（stub）")


def _finish(plan_id: str, step_id: str, member: str) -> None:
    """手工把某环节的队列条目收尾为 done（乱序/顺序由调用方决定）。"""
    item = plans._queue_item_for_step(plan_id, step_id, member)
    qp = queues.queue_path(member)
    workq.mutate(
        qp,
        str(item["id"]),
        status=workq.STATUS_DONE,
        result={
            "code": 0,
            "kind": "result",
            "body": OK,
            "checks": [],
            "member": member,
        },
    )


def _state(plan_id: str) -> dict:
    return plans._load_state(plan_id)


def _statuses(plan_id: str) -> dict[str, str]:
    return {str(s["id"]): str(s["status"]) for s in _state(plan_id)["steps"]}


# ---------------------------------------------------------------------------
# ① 递归就绪谓词（纯函数面）
# ---------------------------------------------------------------------------
NESTED = [
    {"id": "a1", "member": "sim", "brief": "前置"},
    {
        "id": "A",
        "mode": "parallel",
        "children": [
            {
                "id": "P1",
                "mode": "serial",
                "children": [
                    {"id": "p1-1", "member": "sim", "brief": "串链一"},
                    {"id": "p1-2", "member": "sim", "brief": "串链二"},
                ],
            },
            {
                "id": "P2",
                "mode": "serial",
                "children": [{"id": "p2-1", "member": "figure"}],
            },
        ],
    },
]


def test_predicate_recursive_ready_and_aggregate():
    tree = PlanTree(NESTED)
    assert tree.leaf_paths() == ["a1", "A.P1.p1-1", "A.P1.p1-2", "A.P2.p2-1"]
    st: dict[str, str] = {}

    def ready(p):
        return tree.ready(p, lambda x: st.get(x, "pending"))

    # 顶层隐式 serial：a1 就绪，其余全部等 a1 done
    assert ready("a1") and not ready("A.P1.p1-1") and not ready("A.P2.p2-1")
    st["a1"] = "done"
    # a1 done → 并行组 A 的首个分支（P1）首叶就绪；P2 仍等 P1 started
    assert ready("A.P1.p1-1") and not ready("A.P2.p2-1")
    # serial 组内：p1-2 必须等 p1-1 **done**（完成级）
    assert not ready("A.P1.p1-2")
    # **派发级依赖**：p1-1 一旦入队（started，未 done），P2 即可派发
    st["A.P1.p1-1"] = "queued"
    assert not ready("A.P1.p1-2"), "serial 兄弟须 done（queued 不算）"
    assert ready("A.P2.p2-1"), "parallel 兄弟 started 即解锁（派发级）"
    # 串链推进：p1-1 done → p1-2 就绪
    st["A.P1.p1-1"] = "done"
    assert ready("A.P1.p1-2")
    # 聚合由叶子推导
    assert tree.aggregate("A.P1", lambda x: st.get(x, "pending")) == {
        "started": True,
        "done": False,
    }
    st["A.P1.p1-2"] = "done"
    assert tree.done("A.P1", lambda x: st.get(x, "pending")) is True


def test_dispatchable_orders_by_leaf_path_dfs():
    tree = PlanTree(NESTED)
    st = {"a1": "done", "A.P1.p1-1": "queued"}
    disp = tree.dispatchable(lambda x: st.get(x, "pending"))
    assert disp == ["A.P2.p2-1"], "DFS 序 + 就绪过滤（p1-2 未就绪被排除）"


# ---------------------------------------------------------------------------
# ② 嵌套 plan：state 扁平化 + amend 增删兼容
# ---------------------------------------------------------------------------
def test_nested_plan_flattens_leaf_paths_and_amend(qface):
    _write_plan(qface, "n1", NESTED)
    assert (
        plans.cmd_plan_new(_args("n1", file=str(qface / "n1.plan.yaml"), force=True))
        == 0
    )
    st = _state("n1")
    assert [s["id"] for s in st["steps"]] == [
        "a1",
        "A.P1.p1-1",
        "A.P1.p1-2",
        "A.P2.p2-1",
    ]
    assert all(s["status"] == "pending" for s in st["steps"])

    # 推进一叶后 amend：加一叶（旧状态保留）/ 删一叶（状态行移除）
    st["steps"][0]["status"] = "done"
    plans._save_state("n1", st)
    NESTED2 = [*NESTED]
    NESTED2[1] = {
        "id": "A",
        "mode": "parallel",
        "children": [
            NESTED[1]["children"][0],
            {
                "id": "P3",
                "mode": "serial",
                "children": [{"id": "p3-1", "member": "lit"}],
            },
        ],
    }
    _write_plan(qface, "n1", NESTED2)
    assert plans.cmd_plan_amend(_args("n1")) == 0
    st = _state("n1")
    assert [s["id"] for s in st["steps"]] == [
        "a1",
        "A.P1.p1-1",
        "A.P1.p1-2",
        "A.P3.p3-1",
    ]
    assert st["steps"][0]["status"] == "done", "既有叶子状态按 id 对账保留"
    assert st["steps"][3]["status"] == "pending", "新增叶子置 pending"


# ---------------------------------------------------------------------------
# ③ advance 按递归就绪谓词推进（含派发级依赖）
# ---------------------------------------------------------------------------
def test_advance_follows_ready_order_with_nesting(qface, monkeypatch, capsys):
    _write_plan(qface, "n2", NESTED)
    plans.cmd_plan_new(_args("n2", file=str(qface / "n2.plan.yaml"), force=True))
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: _fake_worker(member)
    )
    capsys.readouterr()

    # 入队即返回、**下一次 advance 才结账**（既有事件驱动语义）：逐轮推进并断言顺序
    assert (
        plans.cmd_plan_advance(_args("n2", text="t1")) == 0
    )  # → a1 queued（stub 已消化条目）
    assert _statuses("n2")["a1"] == "queued"
    assert (
        plans.cmd_plan_advance(_args("n2", text="t2")) == 0
    )  # 结 a1 → 派并行组首分支首叶
    assert _statuses("n2") == {
        "a1": "done",
        "A.P1.p1-1": "queued",
        "A.P1.p1-2": "pending",
        "A.P2.p2-1": "pending",
    }
    assert (
        plans.cmd_plan_advance(_args("n2", text="t3")) == 0
    )  # 结 p1-1 → 派 serial 链下一叶
    st = _statuses("n2")
    assert st["A.P1.p1-1"] == "done" and st["A.P1.p1-2"] == "queued"
    assert (
        plans.cmd_plan_advance(_args("n2", text="t4")) == 0
    )  # 结 p1-2 → 派 P2（派发级已解锁）
    st = _statuses("n2")
    assert st["A.P1.p1-2"] == "done" and st["A.P2.p2-1"] == "queued"
    assert "A" not in st, "superstep 聚合不落盘（state 只有叶子）"
    assert plans.cmd_plan_advance(_args("n2")) == 0  # 结 p2-1 → 全部完成
    assert "全部环节完成" in capsys.readouterr().out


def test_serial_sibling_waits_done_but_parallel_unlocks_early(
    qface, monkeypatch, capsys
):
    """派发级 vs 完成级对照：不 stub 收尾，直接观察第二次 advance 的可派发项。"""
    _write_plan(qface, "n3", NESTED)
    plans.cmd_plan_new(_args("n3", file=str(qface / "n3.plan.yaml"), force=True))
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("n3", text="t1")) == 0  # a1 queued（未 done）
    # a1 未 done → 后续无可派发
    assert plans.cmd_plan_advance(_args("n3", text="tx")) == 0
    assert "可派发：（无）" in capsys.readouterr().out
    _finish("n3", "a1", "sim")
    assert plans.cmd_plan_advance(_args("n3", text="t2")) == 0  # 收尾 a1 → 派 P1 首叶
    st = _statuses("n3")
    assert st["a1"] == "done" and st["A.P1.p1-1"] == "queued"
    # p1-1 在途（未 done）：serial 兄弟 p1-2 不可派发；**parallel 兄弟 P2 可派发**
    assert plans.cmd_plan_advance(_args("n3", text="t3")) == 0
    st = _statuses("n3")
    assert st["A.P2.p2-1"] == "queued", "派发级：靠前分支在途即解锁靠后分支"
    assert st["A.P1.p1-2"] == "pending", "完成级：serial 兄弟须 done 才解锁"


# ---------------------------------------------------------------------------
# ④ 多 job 乱序收尾
# ---------------------------------------------------------------------------
def test_multi_job_out_of_order_settlement(qface, monkeypatch, capsys):
    steps = [
        {
            "id": "g",
            "mode": "parallel",
            "children": [
                {"id": "l1", "member": "sim", "brief": "甲"},
                {"id": "l2", "member": "figure", "brief": "乙"},
            ],
        },
    ]
    _write_plan(qface, "n4", steps)
    plans.cmd_plan_new(_args("n4", file=str(qface / "n4.plan.yaml"), force=True))
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("n4", text="t1")) == 0  # l1 queued
    assert plans.cmd_plan_advance(_args("n4", text="t2")) == 0  # l2 queued（派发级）
    assert _statuses("n4") == {"g.l1": "queued", "g.l2": "queued"}

    # 乱序：**后派的 l2 先收尾** → 一次 advance 把已终态的 l2 结掉，l1 保持 in-flight
    _finish("n4", "g.l2", "figure")
    assert plans.cmd_plan_advance(_args("n4")) == 0
    st = _statuses("n4")
    assert st["g.l2"] == "done" and st["g.l1"] == "queued", "已终态的先结，在途的保持"
    out = capsys.readouterr().out
    assert "环节 g.l2 → done" in out, "逐项交接（乱序即到达序）"
    assert "在途 1 项" in out

    _finish("n4", "g.l1", "sim")
    assert plans.cmd_plan_advance(_args("n4")) == 0
    assert _statuses("n4") == {"g.l1": "done", "g.l2": "done"}
    assert "全部环节完成" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ⑤ [NEXT] 三栏工作清单
# ---------------------------------------------------------------------------
def test_next_worklist_three_panes(qface, monkeypatch, capsys):
    _write_plan(qface, "n5", NESTED)
    plans.cmd_plan_new(_args("n5", file=str(qface / "n5.plan.yaml"), force=True))
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    capsys.readouterr()
    plans.cmd_plan_advance(_args("n5", text="t1"))  # a1 queued（在途）
    out = capsys.readouterr().out
    assert "[NEXT] 工作清单（可派发 / 在途 / 待验收）：" in out
    assert "在途：a1" in out
    assert "可派发：（无）" in out, "a1 未 done：后继不可派发"
    assert "待验收：（无）" in out

    _finish("n5", "a1", "sim")
    plans.cmd_plan_advance(_args("n5", text="t2"))
    out = capsys.readouterr().out
    assert "可派发（按 id 序，一次引导一个）：A.P1.p1-1" in out
    assert "在途：A.P1.p1-1" in out


# ---------------------------------------------------------------------------
# ⑥ run 等全部在途 job 收尾
# ---------------------------------------------------------------------------
def test_run_waits_all_inflight(qface, monkeypatch, capsys):
    steps = [
        {
            "id": "g",
            "mode": "parallel",
            "children": [
                {"id": "l1", "member": "sim", "brief": "甲"},
                {"id": "l2", "member": "figure", "brief": "乙"},
            ],
        },
        {"id": "tail", "member": "sim", "brief": "收尾"},
    ]
    _write_plan(qface, "n6", steps)
    plans.cmd_plan_new(_args("n6", file=str(qface / "n6.plan.yaml"), force=True))
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    capsys.readouterr()
    plans.cmd_plan_advance(_args("n6", text="t1"))  # l1 queued
    plans.cmd_plan_advance(_args("n6", text="t2"))  # l2 queued
    _finish("n6", "g.l1", "sim")
    _finish("n6", "g.l2", "figure")

    # run：无新任务书 → 先等**全部在途**收尾（两项都结），再引导 tail
    rc = plans.cmd_plan_run(_args("n6", text=None))
    out = capsys.readouterr().out
    st = _statuses("n6")
    assert st["g.l1"] == "done" and st["g.l2"] == "done", "run 等全部在途收尾"
    assert "环节 g.l1 → done" in out and "环节 g.l2 → done" in out
    assert st["tail"] == "pending" and rc == 2, (
        "收尾后引导下一环节（缺任务书 → 需书写）"
    )
