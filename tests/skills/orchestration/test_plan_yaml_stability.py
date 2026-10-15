"""plan YAML 稳定序列化 + 未编辑面零漂移（backlog 20261015-235100-devops）。

成因：``yaml.safe_dump`` 默认 ``width=80`` 把长 ``brief`` 折行——任何**重新序列化**（plan
new/edit/amend/rework/adhoc 都走 ``safe_dump``）都会让**未编辑的环节**也变（实案：手写长行
计划出现「4 insert, 2 del 语义零变化」的折行 diff，脏掉跟踪 state 面）。修复 = ``_dump_plan``
**不折行** → 重序列化幂等、未改动计划的字节不变。

另含同批菜单项 20261015-235101-devops 的两处引导缺口（role 证据段 + 骨架占位标注）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from pysci import paths as _paths
from pysci.skills.orchestration.tools import plans, taskbooks
from pysci.skills.orchestration.tools.planflow import PlanTree


@pytest.fixture
def pface(qface, tmp_path, monkeypatch):
    """qface（plans 隔离）之上补 taskbooks/data 两面——``cmd_plan_rework`` 会读任务书状态。"""
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tmp_path / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", tmp_path / "data")
    return tmp_path


#: 远超 PyYAML 默认 width=80 的 brief（默认会折行；不折行修复应让它整行保留）。
LONG_BRIEF = (
    "三机制缺口修复（用户点名）——①shutil.move 归档 reply 源缺失 fail-soft（dispatch.py:1596，"
    "双写者根因）②任务书白名单声明自动注入 PYSCI_TASK_DIRS（声明面与注入面合一）③机械验收 "
    "orch-tests 判据防哨兵留痕污染；各带回归测试"
)


def _plan_two_steps(pid: str = "p1") -> dict:
    return {
        "id": pid,
        "goal": "g",
        "steps": [
            {"id": "f1", "member": "devops", "brief": LONG_BRIEF},
            {"id": "f2", "member": "devops", "brief": LONG_BRIEF},
        ],
    }


def _seed(pface: Path, pid: str, plan: dict) -> Path:
    p = pface / "plans" / f"{pid}.plan.yaml"
    p.write_text(plans._dump_plan(plan), encoding="utf-8")
    return p


def _edit_args(pid: str, **over) -> Namespace:
    base = dict(id=pid, step="f2", brief="短", append_brief=None, dirs=None)
    base.update(over)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# ① _dump_plan：不折行 + 幂等 + round-trip 保真
# ---------------------------------------------------------------------------
def test_dump_plan_does_not_wrap_long_brief():
    text = plans._dump_plan(_plan_two_steps())
    assert f"  brief: {LONG_BRIEF}\n" in text, "长 brief 被折行——重序列化即漂移"


def test_dump_plan_is_idempotent():
    d1 = plans._dump_plan(_plan_two_steps())
    d2 = plans._dump_plan(yaml.safe_load(d1))
    assert d1 == d2, "重序列化不幂等"


def test_dump_plan_roundtrip_preserves_canonical_text():
    """规范形态（不折行）文本 → load → dump 字节不变（round-trip 保真）。"""
    canonical = plans._dump_plan(_plan_two_steps())
    assert plans._dump_plan(yaml.safe_load(canonical)) == canonical


# ---------------------------------------------------------------------------
# ② 未编辑面零漂移（success_criteria：B 计划字节不变 / 未编辑环节不折行）
# ---------------------------------------------------------------------------
def test_plan_edit_does_not_reflow_untouched_step(pface):
    """编辑 f2 的 brief → f1（未编辑）的长 brief 仍整行不折。"""
    p = _seed(pface, "p1", _plan_two_steps())
    assert plans.cmd_plan_edit(_edit_args("p1", step="f2", brief="改短的 f2")) == 0
    text = p.read_text(encoding="utf-8")
    assert f"  brief: {LONG_BRIEF}\n" in text, "未编辑的 f1 被折行（工具重序列化漂移）"
    assert plans._dump_plan(yaml.safe_load(text)) == text


def test_edit_on_plan_a_leaves_plan_b_bytes_unchanged(pface):
    _seed(pface, "pa", _plan_two_steps("pa"))
    pb = _seed(pface, "pb", _plan_two_steps("pb"))
    before = pb.read_bytes()
    assert plans.cmd_plan_edit(_edit_args("pa", step="f1", brief="x")) == 0
    assert pb.read_bytes() == before, "编辑 A 却改动了无关的 B"


def test_read_paths_do_not_write_plans(pface):
    p = _seed(pface, "p1", _plan_two_steps())
    before = p.read_bytes()
    plans.list_plans()
    plans.plan_tree_lines("p1")
    plans.load_state("p1")
    plans._load_yaml("p1")
    assert p.read_bytes() == before, "读路径落盘改写了计划 YAML"


# ---------------------------------------------------------------------------
# ③ 菜单项 20261015-235101-devops：role 证据段引导 + 骨架占位标注
# ---------------------------------------------------------------------------
def test_self_guidance_prints_role_default_sections(capsys):
    """role-only 的 self 环节：引导打印 role 缺省特征段（与 _check_evidence 同源）。"""
    plans._print_self_guidance(
        "p1", {"id": "a4"}, {"role": "improvement_plan", "brief": "拆解改进派单"}
    )
    out = capsys.readouterr().out
    assert "拆解" in out, "role 缺省证据段未打印——组长仍靠猜关键词"


def test_self_guidance_prefers_explicit_sections(capsys):
    plans._print_self_guidance(
        "p1",
        {"id": "a1"},
        {"role": "cause", "evidence_sections": ["成因分析", "结论"]},
    )
    out = capsys.readouterr().out
    assert "成因分析" in out and "结论" in out


def test_rework_skeleton_marks_placeholder_members():
    plan = plans.build_rework_plan("tb-x", mode="full")
    ph = {
        leaf.path
        for leaf in PlanTree(plan["steps"]).leaves
        if leaf.yaml.get("member_placeholder")
    }
    assert ph == {"A.a5", "BC.B.b2", "BC.C.c4"}


def test_rework_print_annotates_placeholder_member(pface, capsys):
    rc = plans.cmd_plan_rework(_edit_args("tb-x", mode="full", plan_id="rework-tb-x"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "占位" in out and "按 §5 分工" in out
