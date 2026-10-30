"""PSEP_3 pd1：返工流**骨架**——字段/状态机 / 三段式 / 模板三档 / 必留 lint / 送审 gate。

钉新面（PSEP_3 §4.1/§4.2/§4.3）：① taskbooks 四字段 + ``failed`` 终态；② TUI reject 三段式
（模式→优先级→批语，Pilot 见 test_taskbooks）；③ plan rework 模板三档骨架；④ 必留节点 lint
（缺失/乱序 advance 拒绝）+ 模板节点 amend 冻结；⑤ 送审 gate（verdict=A 才派单）；⑥ 两次送审
不过 → plan abandoned + 任务书 failed。
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from pysci import paths as _paths
from pysci.skills.orchestration.tools import jury, plans, queueworker, taskbooks
from pysci.skills.orchestration.tools.planflow import PlanTree


@pytest.fixture
def rface(iso_state, tmp_path, monkeypatch):
    """在 ``iso_state``（全状态面隔离）之上补 plans/taskbooks/data 三面。

    前置缺陷修复（pd2 §0）：原夹具漏 ``queues.QUEUES_DIR``，``plan advance`` 入队真写仓库
    ``orchestration/state/queues/sim.json``（组长 2026-10-15 复现实锤）。改锚 ``iso_state``
    ——它已 patch ``QUEUES_DIR``/``QUEUE_WORKER_ROOT``/``ORCH_STATE_ROOT``/jury/journal…；
    另 stub ``wake_member`` 防起真 worker。
    """
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（测试不消化）")
    )
    return tmp_path


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
        mode="full",
        goal=None,
        plan_id=None,
    )
    base.update(over)
    return Namespace(**base)


def _write_plan(tmp: Path, plan_id: str, plan: dict) -> Path:
    plan = {**plan, "id": plan_id}
    p = tmp / "plans" / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return p


def _jury_record(case_id: str, verdict: str) -> None:
    p = jury.record_path(case_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "verdict": verdict,
                "final_verdict": verdict,
                "verdict_summary": {"text": f"裁决={verdict}"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ===========================================================================
#  ① taskbooks 四字段 + failed 终态
# ===========================================================================
def test_create_stores_new_fields(rface):
    st = taskbooks.create(
        "大任务",
        "正文",
        priority="high",
        mode="full",
        rework_of="tb-old",
        rework_round=2,
    )
    assert st["priority"] == "high" and st["mode"] == "full"
    assert st["rework_of"] == "tb-old" and st["rework_round"] == 2
    md = (taskbooks.TASKBOOKS_DIR / f"{st['id']}.taskbook.md").read_text(
        encoding="utf-8"
    )
    for frag in (
        "priority: high",
        "mode: full",
        "rework_of: tb-old",
        "rework_round: 2",
    ):
        assert frag in md, frag


def test_create_defaults_priority_normal(rface):
    assert taskbooks.create("t", "b")["priority"] == "normal"


def test_create_rejects_bad_priority_or_mode(rface):
    with pytest.raises(ValueError):
        taskbooks.create("t", "b", priority="urgent")
    with pytest.raises(ValueError):
        taskbooks.create("t", "b", mode="partial")


def test_failed_is_terminal_from_active(rface):
    tid = taskbooks.create("t", "b")["id"]
    taskbooks.transition(tid, "active")
    assert taskbooks.transition(tid, "failed")["status"] == "failed"
    with pytest.raises(ValueError):  # failed 终态
        taskbooks.transition(tid, "new")


def test_create_rework_marks_original_and_inherits(rface):
    orig = taskbooks.create("目标大任务", "简介文本")["id"]
    taskbooks.transition(orig, "active")
    taskbooks.transition(orig, "delivered")
    ne = taskbooks.create_rework(
        orig, mode="no_inspector", priority="low", note="拆得不对"
    )
    o = taskbooks.load_state(orig)
    assert o["status"] == "rejected"  # 原任务书终态留档
    assert o["rejections"][-1]["note"] == "拆得不对"
    assert ne["rework_of"] == orig and ne["rework_round"] == 1
    assert ne["mode"] == "no_inspector" and ne["priority"] == "low"
    assert ne["title"] == "目标大任务（返工第 1 轮）"  # 继承名称 + 轮次后缀
    assert "简介文本" in taskbooks.read_body(ne["id"])  # 继承简介
    # 第 2 轮：round 自增
    taskbooks.transition(ne["id"], "active")
    taskbooks.transition(ne["id"], "delivered")
    ne2 = taskbooks.create_rework(
        ne["id"], mode="rework_only", priority="high", note="x"
    )
    assert ne2["rework_round"] == 2 and ne2["rework_of"] == ne["id"]


# ===========================================================================
#  ③ plan rework 模板三档骨架
# ===========================================================================
def test_build_rework_plan_full():
    plan = plans.build_rework_plan("tb-x", mode="full")
    assert plan["template"] == "rework" and plan["rework_mode"] == "full"
    assert plan["taskbook"] == "tb-x"
    leaves = PlanTree(plan["steps"]).leaf_paths()
    assert leaves == list(plans.REWORK_REQUIRED["full"])
    assert plans.template_violations(plan) == []
    # gate 落在 a4 / c3
    gate_ids = {
        leaf.yaml.get("gate_jury")
        for leaf in PlanTree(plan["steps"]).leaves
        if leaf.yaml.get("gate_jury")
    }
    assert gate_ids == {"tb-x-jury", "tb-x-discoverability"}


def test_build_rework_plan_no_inspector_and_rework_only():
    ni = plans.build_rework_plan("tb-x", mode="no_inspector")
    assert [s["id"] for s in ni["steps"]] == ["A", "B"]
    assert PlanTree(ni["steps"]).leaf_paths() == list(
        plans.REWORK_REQUIRED["no_inspector"]
    )
    ro = plans.build_rework_plan("tb-x", mode="rework_only")
    assert [s["id"] for s in ro["steps"]] == ["B"]
    assert PlanTree(ro["steps"]).leaf_paths() == list(
        plans.REWORK_REQUIRED["rework_only"]
    )


def test_cmd_plan_rework_registers(rface, capsys):
    rc = plans.cmd_plan_rework(_args("tb-x", mode="full"))
    assert rc == 0
    assert (rface / "plans" / "rework-tb-x.plan.yaml").exists()
    state = plans.load_state("rework-tb-x")
    assert len(state["steps"]) == 12  # 叶子扁平化


# ===========================================================================
#  ④ 必留节点 lint（缺失/乱序 → advance 拒绝）+ amend 冻结
# ===========================================================================
def _plan_with_leaves(plan: dict, keep: list[str]) -> dict:
    """按点路径过滤叶子（保留必留节点顺序），重建 steps 树（保留原始组结构）。"""

    def _flt(items: list[dict], prefix: str) -> list[dict]:
        out: list[dict] = []
        for it in items:
            path = f"{prefix}.{it['id']}" if prefix else it["id"]
            kids = it.get("children")
            if kids:
                sub = _flt(kids, path)
                if sub:
                    out.append({**it, "children": sub})
            elif path in keep:
                out.append(it)
        return out

    return {**plan, "steps": _flt(plan["steps"], "")}


def test_advance_refuses_missing_required_node(rface, capsys):
    plan = plans.build_rework_plan("tb-x", mode="rework_only")
    bad = _plan_with_leaves(plan, ["B.b1", "B.b_last"])  # 删 b2
    _write_plan(rface, "rw1", bad)
    assert plans.cmd_plan_advance(_args("rw1", text="t")) == 2
    assert "必留节点缺失" in capsys.readouterr().out


def test_advance_refuses_out_of_order(rface, capsys):
    plan = plans.build_rework_plan("tb-x", mode="rework_only")
    # 把 b2 挪到 b1 之前 → 乱序
    b = plan["steps"][0]
    b["children"] = [b["children"][1], b["children"][0], b["children"][2]]
    _write_plan(rface, "rw2", plan)
    assert plans.cmd_plan_advance(_args("rw2", text="t")) == 2
    assert "乱序" in capsys.readouterr().out


def test_amend_refuses_template_node_removal(rface, capsys):
    plan = plans.build_rework_plan("tb-x", mode="rework_only")
    _write_plan(rface, "rw3", plan)
    plans.cmd_plan_new(_args("rw3", file=str(rface / "plans" / "rw3.plan.yaml")))
    capsys.readouterr()
    _write_plan(rface, "rw3", _plan_with_leaves(plan, ["B.b1", "B.b_last"]))  # 删 b2
    assert plans.cmd_plan_amend(_args("rw3")) == 2
    assert "必留节点" in capsys.readouterr().out


# ===========================================================================
#  ⑤⑥ 送审 gate 与两次不过中止
# ===========================================================================
def _gate_plan(rface, plan_id: str, tb_id: str) -> None:
    _write_plan(
        rface,
        plan_id,
        {
            "goal": "gate 测试",
            "taskbook": tb_id,
            "steps": [
                {"id": "g", "kind": "self", "brief": "拆解派单", "gate_jury": "case-1"}
            ],
        },
    )


def test_gate_blocks_until_verdict_a(rface, capsys):
    _gate_plan(rface, "g1", "tb-g")
    assert plans.cmd_plan_advance(_args("g1")) == 2  # 案卷未落 → 拒
    assert "未落/未裁决" in capsys.readouterr().out
    _jury_record("case-1", "A")
    assert plans.cmd_plan_advance(_args("g1")) == 0  # verdict=A → 放行（引导 self）
    assert "kind: self" in capsys.readouterr().out


def test_gate_two_b_attempts_abort_and_fail_taskbook(rface, capsys):
    tb = taskbooks.create("大任务", "b")["id"]
    taskbooks.transition(tb, "active")
    _gate_plan(rface, "g2", tb)
    _jury_record("case-1", "B")
    assert plans.cmd_plan_advance(_args("g2")) == 2  # 第 1 次 B → 拒
    assert "第 1 次" in capsys.readouterr().out
    assert plans.cmd_plan_advance(_args("g2")) == 2  # 第 2 次 B → 中止
    out = capsys.readouterr().out
    assert "已中止" in out and "failed" in out
    st = plans.load_state("g2")
    assert st["abandoned_at"] and st["abandon_reason"]
    assert len(st["gate_rulings"]) == 2  # 附两轮裁决书
    assert taskbooks.load_state(tb)["status"] == "failed"


def test_gate_not_applied_without_gate_jury(rface, capsys):
    _write_plan(
        rface,
        "g3",
        {"goal": "无 gate", "steps": [{"id": "a1", "member": "sim", "brief": "普通"}]},
    )
    assert (
        plans.cmd_plan_advance(_args("g3", text="t")) == 0
    )  # 无 gate_jury 字段 → 不拦截
