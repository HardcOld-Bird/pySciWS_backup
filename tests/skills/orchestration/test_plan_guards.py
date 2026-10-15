"""PSEP_3 pa2 护栏与拆解指导：kind:self / amend 冻结 / 软提示 / 拆解必读三级。

钉新面（pa1 谓词测试见 test_plan_superstep.py）：

① ``kind: self`` 组长环节：不入成员队列，``plan advance --evidence`` 收尾（文件存在 +
   标题段 lint 两侧）；无 ``kind`` 等价 ``dispatch``；
② ``amend`` 冻结（PSEP_3 §2.4）：已 started 子树内节点删/挪拒绝、未派发部分放行、追加放行；
③ 软提示（不阻断）：并行多分支同成员 / 宽度>3 / 深度>3；
④ 拆解必读三级：``plan new`` 头部打印 + ``read:`` 标记行 lint（含 superstep 或环节≥5）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from pysci.skills.orchestration.tools import plans, queues, queueworker, workq
from pysci.skills.orchestration.tools.planflow import (
    PlanTree,
    freeze_violations,
)

OK = "改动完成 commit abc1234 已合并回 main"


def _write_plan(d: Path, plan_id: str, steps: list[dict], **top) -> None:
    doc = {"id": plan_id, "goal": "g", "steps": steps}
    doc.update(top)
    (d / f"{plan_id}.plan.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8"
    )


def _args(plan_id: str, **over) -> Namespace:
    base = dict(
        id=plan_id,
        text=None,
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
        file="",
        force=True,
        evidence=None,
        dry=False,
    )
    base.update(over)
    return Namespace(**base)


def _new(qface: Path, plan_id: str, steps: list[dict], **top) -> int:
    _write_plan(qface, plan_id, steps, **top)
    return plans.cmd_plan_new(_args(plan_id, file=str(qface / f"{plan_id}.plan.yaml")))


def _statuses(plan_id: str) -> dict[str, str]:
    return {s["id"]: s["status"] for s in plans._load_state(plan_id)["steps"]}


def _finish(plan_id: str, step_id: str, member: str) -> None:
    item = plans._queue_item_for_step(plan_id, step_id, member)
    workq.mutate(
        queues.queue_path(member),
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


# ---------------------------------------------------------------------------
# ① kind:self 组长环节
# ---------------------------------------------------------------------------
def test_self_step_guided_then_evidence_done(qface, monkeypatch, capsys, tmp_path):
    steps = [
        {
            "id": "s1",
            "kind": "self",
            "brief": "成因分析报告",
            "evidence_sections": ["成因分析", "结论"],
        },
        {"id": "s2", "member": "sim", "brief": "实施"},
    ]
    assert _new(qface, "g1", steps) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    # 引导：self 环节不入队列、置 self_awaiting（started 族）
    assert plans.cmd_plan_advance(_args("g1")) == 0
    out = capsys.readouterr().out
    assert "kind: self" in out and "self·待证据" in out
    st = _statuses("g1")
    assert st["s1"] == "self_awaiting" and st["s2"] == "pending"
    assert queues.items("sim") == [], "self 环节不入成员队列"

    # evidence 缺标题段 → 拒绝（rc=2，状态保持 self_awaiting）
    bad = tmp_path / "ev_bad.md"
    bad.write_text("只有正文，没有标题段\n", encoding="utf-8")
    assert plans.cmd_plan_advance(_args("g1", evidence=str(bad))) == 2
    assert "缺标题段" in capsys.readouterr().out
    assert _statuses("g1")["s1"] == "self_awaiting"

    # evidence 齐全 → done，后继 dispatch 环节解锁
    good = tmp_path / "ev.md"
    good.write_text("# 成因分析\n…\n## 结论\n…\n", encoding="utf-8")
    assert plans.cmd_plan_advance(_args("g1", evidence=str(good))) == 0
    assert "收尾" in capsys.readouterr().out
    assert _statuses("g1")["s1"] == "done"


def test_self_step_missing_file_rejected(qface, monkeypatch, capsys, tmp_path):
    steps = [{"id": "s1", "kind": "self", "brief": "报告"}]
    _new(qface, "g2", steps)
    capsys.readouterr()
    monkeypatch.setattr(queueworker, "wake_member", lambda member, **k: (True, "stub"))
    plans.cmd_plan_advance(_args("g2"))  # 引导
    assert plans.cmd_plan_advance(_args("g2", evidence=str(tmp_path / "nope.md"))) == 2
    assert "不存在" in capsys.readouterr().out


def test_no_kind_is_dispatch(qface, monkeypatch, capsys):
    """无 kind 等价 dispatch：入成员队列（既有行为）。"""
    steps = [{"id": "s1", "member": "sim", "brief": "普通环节"}]
    _new(qface, "g3", steps)
    capsys.readouterr()
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    assert plans.cmd_plan_advance(_args("g3", text="t")) == 0
    assert _statuses("g3")["s1"] == "queued"
    assert len(queues.items("sim")) == 1


# ---------------------------------------------------------------------------
# ② amend 冻结
# ---------------------------------------------------------------------------
def test_amend_rejects_removal_in_started_subtree(qface, monkeypatch, capsys):
    steps = [
        {"id": "a1", "member": "sim", "brief": "前置"},
        {
            "id": "A",
            "mode": "serial",
            "children": [
                {"id": "p1", "member": "sim", "brief": "一"},
                {"id": "p2", "member": "sim", "brief": "二"},
            ],
        },
    ]
    _new(qface, "m1", steps)
    capsys.readouterr()
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    plans.cmd_plan_advance(_args("m1", text="t1"))  # a1 queued
    _finish("m1", "a1", "sim")
    plans.cmd_plan_advance(_args("m1", text="t2"))  # A.p1 queued（A started）
    _finish("m1", "A.p1", "sim")
    plans.cmd_plan_advance(_args("m1", text="t3"))  # 结 p1 → A 已 started
    capsys.readouterr()

    # 删 A.p2（已 started 子树内未派发节点）→ 拒绝
    _write_plan(
        qface,
        "m1",
        [
            {"id": "a1", "member": "sim", "brief": "前置"},
            {
                "id": "A",
                "mode": "serial",
                "children": [
                    {"id": "p1", "member": "sim", "brief": "一"},
                ],
            },
        ],
    )
    assert plans.cmd_plan_amend(_args("m1")) == 2
    out = capsys.readouterr().out
    assert "结构冻结" in out and "A.p2" in out


def test_amend_allows_pending_edit_and_append(qface, monkeypatch, capsys):
    steps = [
        {"id": "a1", "member": "sim", "brief": "前置"},
        {
            "id": "A",
            "mode": "serial",
            "children": [
                {"id": "p1", "member": "sim", "brief": "一"},
                {"id": "p2", "member": "sim", "brief": "二"},
            ],
        },
    ]
    _new(qface, "m2", steps)
    capsys.readouterr()
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    plans.cmd_plan_advance(_args("m2", text="t1"))
    _finish("m2", "a1", "sim")
    plans.cmd_plan_advance(_args("m2", text="t2"))
    _finish("m2", "A.p1", "sim")
    plans.cmd_plan_advance(_args("m2", text="t3"))  # A started（p1 done）
    capsys.readouterr()

    # 改未派发叶子 p2 的 brief + 追加新叶子 p3 → 放行（rc=0）
    _write_plan(
        qface,
        "m2",
        [
            {"id": "a1", "member": "sim", "brief": "前置"},
            {
                "id": "A",
                "mode": "serial",
                "children": [
                    {"id": "p1", "member": "sim", "brief": "一"},
                    {"id": "p2", "member": "sim", "brief": "二改"},
                    {"id": "p3", "member": "sim", "brief": "三新增"},
                ],
            },
        ],
    )
    assert plans.cmd_plan_amend(_args("m2")) == 0
    assert _statuses("m2")["A.p3"] == "pending"


def test_freeze_violations_unit():
    # 已 started 叶子被删
    assert freeze_violations({"A.p1": "done", "A.p2": "pending"}, ["A.p2"])
    # 未 started 子树可随意删
    assert freeze_violations({"a1": "pending", "a2": "pending"}, []) == []
    # 追加放行
    assert freeze_violations({"A.p1": "queued"}, ["A.p1", "A.p2"]) == []


# ---------------------------------------------------------------------------
# ③ 软提示（不阻断）
# ---------------------------------------------------------------------------
def _hints(steps: list[dict]) -> list[str]:
    return PlanTree(steps).soft_hints()


def test_soft_hint_same_member_parallel():
    steps = [
        {
            "id": "G",
            "mode": "parallel",
            "children": [
                {"id": "l1", "member": "sim"},
                {"id": "l2", "member": "sim"},
            ],
        }
    ]
    hints = _hints(steps)
    assert any("多分支同用成员" in h and "sim" in h for h in hints)


def test_soft_hint_width_gt3():
    steps = [
        {
            "id": "G",
            "mode": "parallel",
            "children": [{"id": f"l{i}", "member": f"m{i}"} for i in range(4)],
        }
    ]
    assert any("宽度 4 > 3" in h for h in _hints(steps))


def test_soft_hint_depth_gt3():
    steps = [
        {
            "id": "A",
            "mode": "serial",
            "children": [
                {
                    "id": "B",
                    "mode": "serial",
                    "children": [
                        {
                            "id": "C",
                            "mode": "serial",
                            "children": [
                                {
                                    "id": "D",
                                    "mode": "serial",
                                    "children": [
                                        {"id": "l", "member": "sim"},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
    ]
    assert any("嵌套深度 4 > 3" in h for h in _hints(steps))


def test_soft_hint_non_blocking(qface, capsys):
    # 软提示只打印、不影响 plan new 成功（rc=0）
    steps = [
        {
            "id": "G",
            "mode": "parallel",
            "children": [
                {"id": "l1", "member": "sim"},
                {"id": "l2", "member": "sim"},
            ],
        }
    ]
    assert _new(qface, "h1", steps) == 0
    assert "软提示，不阻断" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ④ 拆解必读三级
# ---------------------------------------------------------------------------
def test_plan_new_prints_read_hint(qface, capsys):
    steps = [{"id": "s1", "member": "sim"}]
    _new(qface, "r1", steps)
    out = capsys.readouterr().out
    assert plans.DECOMP_READ_HINT in out
    assert "plan-decomposition.md" in out


def test_read_marker_lint_when_superstep(qface, capsys):
    steps = [
        {
            "id": "G",
            "mode": "parallel",
            "children": [
                {"id": "l1", "member": "sim"},
                {"id": "l2", "member": "figure"},
            ],
        }
    ]
    _new(qface, "r2", steps)  # 无 read: 标记 → lint 提示
    assert "read: plan-decomposition.md" in capsys.readouterr().out


def test_read_marker_lint_satisfied(qface, capsys):
    steps = [{"id": f"s{i}", "member": "sim"} for i in range(5)]
    _new(qface, "r3", steps, read="plan-decomposition.md")
    out = capsys.readouterr().out
    assert "read: plan-decomposition.md" not in out, "已有标记行：不再提示"


def test_read_marker_flat_five_leaves(qface, capsys):
    steps = [{"id": f"s{i}", "member": "sim"} for i in range(5)]
    _new(qface, "r4", steps)  # 扁平但 ≥5 环节 → 仍提示
    assert "read: plan-decomposition.md" in capsys.readouterr().out
