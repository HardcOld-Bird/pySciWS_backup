"""L（组长改进）环节：link 自动追加 / l1 正式要求文本 / --skip 空清单收尾 / 报告段。

用户 2026-10-17 裁决：初次/返工 plan 末尾可选 L 环节（l1 反思 → l2 拆解改进派单 →
l3 devops 实施），**工具自动追加、组长零手写**；空改进清单合法（``plan advance --skip
L.l3`` 收尾）。
"""

from __future__ import annotations

from argparse import Namespace

import pytest
import yaml

from pysci import paths as _paths
from pysci.skills.orchestration.tools import plans, taskbooks


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
        mode=None,
        goal=None,
        plan_id=None,
        limit=5,
        skip="",
        improve="",
    )
    base.update(over)
    return Namespace(**base)


@pytest.fixture
def face(qface, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return tmp_path


def _register(plan_id: str, steps: list[dict]) -> str:
    p = plans.PLANS_DIR / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(
            {"id": plan_id, "goal": "目标", "steps": steps},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    assert plans.cmd_plan_new(_args(plan_id, file=str(p))) == 0
    return plan_id


def _one_step():
    return [{"id": "s1", "member": "devops", "brief": "干活"}]


# ---------------------------------------------------------------------------
# ① link 自动追加（幂等、不动已 started）
# ---------------------------------------------------------------------------
def test_link_appends_leader_step_once(face):
    pid = _register("p-l", _one_step())
    tb = taskbooks.create("大任务", "正文", improve={"leader": True})["id"]
    taskbooks.link_plan(tb, pid)
    plan = plans._load_yaml(pid)
    assert [s["id"] for s in plan["steps"]][-1] == "L", "L 追加在顶层末尾"
    roles = [leaf.yaml.get("role") for leaf in plans._tree(plan).leaves]
    assert "leader_reflection" in roles and "leader_improvement_plan" in roles
    taskbooks.link_plan(tb, pid)  # 幂等：二次 link 不重复追加
    plan = plans._load_yaml(pid)
    assert (
        sum(
            1
            for leaf in plans._tree(plan).leaves
            if leaf.yaml.get("role") == "leader_reflection"
        )
        == 1
    )


def test_link_without_flag_appends_nothing(face):
    pid = _register("p-n", _one_step())
    tb = taskbooks.create("大任务", "正文")["id"]
    taskbooks.link_plan(tb, pid)
    plan = plans._load_yaml(pid)
    assert all(
        leaf.yaml.get("role") != "leader_reflection"
        for leaf in plans._tree(plan).leaves
    )
    assert plans.ensure_leader_step(pid) is True, (
        "工具面直接调用仍可追加（幂等键=role）"
    )
    assert plans.ensure_leader_step(pid) is False


def test_ensure_leader_step_preserves_started_rows(face):
    pid = _register("p-k", _one_step())
    state = plans._load_state(pid)
    state["steps"][0].update({"status": "done", "evidence": "x.md"})
    plans._save_state(pid, state)
    assert plans.ensure_leader_step(pid) is True
    rows = {s["id"]: s for s in plans._load_state(pid)["steps"]}
    assert rows["s1"]["status"] == "done" and rows["s1"]["evidence"] == "x.md"
    assert rows["L.l1"]["status"] == "pending"


# ---------------------------------------------------------------------------
# ② l1 引导：正式要求文本全文打印
# ---------------------------------------------------------------------------
def test_l1_guidance_prints_formal_prompt(face, capsys):
    pid = _register("p-g", _one_step())
    state = plans._load_state(pid)
    state["steps"][0]["status"] = "done"
    plans._save_state(pid, state)
    plans.ensure_leader_step(pid)
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args(pid)) == 0
    out = capsys.readouterr().out
    assert "组长改进环节" in out and "harnessing" in out
    assert "立单改进" in out and "--skip L.l3" in out
    assert "摩擦反思、改进清单、结论" in out, "l1 evidence 特征段随引导附送"


# ---------------------------------------------------------------------------
# ③ --skip：空改进清单合法收尾
# ---------------------------------------------------------------------------
def test_skip_pending_leaf_closes_plan(face, capsys):
    pid = _register(
        "p-s",
        [
            {"id": "s1", "member": "devops", "brief": "干活"},
            {"id": "s2", "member": "devops", "brief": "不需要"},
        ],
    )
    state = plans._load_state(pid)
    state["steps"][0]["status"] = "done"
    plans._save_state(pid, state)
    assert plans.cmd_plan_advance(_args(pid, skip="s2")) == 0
    out = capsys.readouterr().out
    assert "s2 置 skipped" in out
    pstate = plans.load_state(pid)
    rows = {s["id"]: s for s in pstate["steps"]}
    assert rows["s2"]["status"] == "skipped" and pstate.get("completed_at")


def test_skip_rejects_started_and_unknown(face, capsys):
    pid = _register("p-r2", _one_step())
    state = plans._load_state(pid)
    state["steps"][0].update({"status": "queued", "job_id": "j1"})
    plans._save_state(pid, state)
    assert plans.cmd_plan_advance(_args(pid, skip="s1")) == 2
    assert "只对 pending" in capsys.readouterr().out
    assert plans.cmd_plan_advance(_args(pid, skip="nope")) == 2


# ---------------------------------------------------------------------------
# ④ 报告「组长改进」段
# ---------------------------------------------------------------------------
def test_report_leader_improvement_section(face, tmp_path):
    tb = taskbooks.create("大任务", "正文", improve={"leader": True})["id"]
    pid = _register(
        "p-rep",
        [
            {"id": "l1", "kind": "self", "role": "leader_reflection", "brief": "反思"},
            {
                "id": "l2",
                "kind": "self",
                "role": "leader_improvement_plan",
                "brief": "派单",
            },
        ],
    )
    taskbooks.link_plan(tb, pid)
    ev1 = tmp_path / "e1.md"
    ev1.write_text(
        "## 摩擦反思\n摩擦A\n\n## 改进清单\n单1\n\n## 结论\nOK\n", encoding="utf-8"
    )
    ev2 = tmp_path / "e2.md"
    ev2.write_text("## 拆解\n改进派单正文\n", encoding="utf-8")
    state = plans._load_state(pid)
    for row, ev in zip(state["steps"], (ev1, ev2), strict=True):
        row.update({"status": "done", "evidence": str(ev)})
    plans._save_state(pid, state)
    report = "\n".join(taskbooks.report_sections(tb))
    assert "## 组长改进" in report
    assert "摩擦A" in report and "单1" in report and "改进派单正文" in report
    # 非返工件也出（段在提前 return 之前）
    assert "## 返工产物" not in report
