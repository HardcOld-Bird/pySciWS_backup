"""`plan edit`：结构化改环节字段，免手编折行（backlog 20261015-222248）。

现场：`plan new` 把 YAML 交给 `yaml.safe_dump` **重排折行**，于是照自己写入时的折行写
`old_string` 的 Edit **逐字对不上**（实测连续两次 `0 occurrences found`），每次都要先 `sed`
读回实际折行再改。本命令让人直接说「改什么」。
"""

from __future__ import annotations

import argparse

import pytest
import yaml

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import orch, plans

PLAN_SRC = """id: p1
goal: demo
steps:
  - id: a1
    member: devops
    brief: 原始环节
    dirs: src
  - id: a2
    member: figure
    brief: 后续环节
"""

CMDSURF = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "command-surface.md"
)


@pytest.fixture
def plan(qface, tmp_path, capsys):
    """登记一份两环节计划（经真命令：YAML 会被重排折行——正是本单要绕开的那件事）。"""
    src = tmp_path / "p1.plan-src.yaml"
    src.write_text(PLAN_SRC, encoding="utf-8")
    assert (
        plans.cmd_plan_new(argparse.Namespace(file=str(src), id="p1", force=False)) == 0
    )
    capsys.readouterr()
    return "p1"


def _leaf(pid: str, sid: str) -> dict:
    return plans._leaf_map(plans._load_yaml(pid))[sid]


def test_append_brief_needs_no_read_back(plan):
    """核心用例：往既有叶子 brief **追加一句**——不必先读回折行后的文本。"""
    rc = orch.main(["plan", "edit", plan, "--step", "a1", "--append-brief", "追加一句"])

    assert rc == 0
    assert _leaf(plan, "a1")["brief"] == "原始环节；追加一句"
    assert _leaf(plan, "a2")["brief"] == "后续环节"  # 只动点名环节


def test_replace_brief_and_dirs(plan):
    assert orch.main(["plan", "edit", plan, "--step", "a2", "--brief", "新描述"]) == 0
    assert _leaf(plan, "a2")["brief"] == "新描述"

    assert (
        orch.main(["plan", "edit", plan, "--step", "a2", "--dirs", "tests,data/x"]) == 0
    )
    assert _leaf(plan, "a2")["dirs"] == ["tests", "data/x"]


def test_progress_is_preserved(plan):
    """对账像 amend 一样**按环节 id 保状态**——改字段不动进度。"""
    st = plans._load_state(plan)
    st["steps"][1]["status"] = "queued"
    plans._save_state(plan, st)

    assert orch.main(["plan", "edit", plan, "--step", "a1", "--append-brief", "x"]) == 0

    assert plans._load_state(plan)["steps"][1]["status"] == "queued"


def test_amend_reconciles_green_after_edit(plan):
    """criteria：改后 `plan amend` 对账绿。"""
    assert orch.main(["plan", "edit", plan, "--step", "a1", "--append-brief", "x"]) == 0
    assert orch.main(["plan", "amend", plan]) == 0


def test_unknown_step_lists_available(plan, capsys):
    rc = orch.main(["plan", "edit", plan, "--step", "nope", "--brief", "x"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "无环节 nope" in out and "a1" in out  # 给出可用环节，不裸报错


def test_no_edit_item_refused(plan, capsys):
    rc = orch.main(["plan", "edit", plan, "--step", "a1"])
    assert rc == 2
    assert "未给任何编辑项" in capsys.readouterr().out


def test_unknown_plan_refused(qface, capsys):
    rc = orch.main(["plan", "edit", "nope", "--step", "a1", "--brief", "x"])
    assert rc == 2
    assert "计划不存在" in capsys.readouterr().out


def test_yaml_stays_loadable_and_written(plan, tmp_path):
    """写盘后 YAML 仍可解析（工具重排折行是它的事，人不用管）。"""
    orch.main(["plan", "edit", plan, "--step", "a1", "--append-brief", "再来一句"])
    p = tmp_path / "plans" / "p1.plan.yaml"  # qface 把 PLANS_DIR 指到 tmp_path/plans
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["goal"] == "demo"


def test_command_surface_documents_plan_edit():
    md = CMDSURF.read_text(encoding="utf-8")
    assert "plan edit" in md
    assert "重排折行" in md  # 明示折行事实（criteria：文档/[NEXT] 明示）


# ---------------------------------------------------------------------------
# 叶子简写（tb-3 R4 L.l3 F5）：--step f3 → B.f3；错误列表给真实路径
# ---------------------------------------------------------------------------
NESTED_PLAN = """id: pn
goal: demo
steps:
  - id: A
    mode: serial
    brief: 分析
    children:
      - id: a1
        kind: self
        brief: 分析叶
  - id: B
    mode: serial
    brief: 返工
    children:
      - id: f1
        member: figure
        brief: 图一
      - id: f2
        member: figure
        brief: 图二
"""


@pytest.fixture
def nested_plan(qface, tmp_path):
    p = tmp_path / "plans" / "pn.plan.yaml"
    p.write_text(NESTED_PLAN, encoding="utf-8")
    return "pn"


def test_edit_accepts_short_leaf_id(nested_plan):
    assert (
        orch.main(["plan", "edit", nested_plan, "--step", "f2", "--brief", "新图二"])
        == 0
    )
    assert _leaf(nested_plan, "B.f2")["brief"] == "新图二"


def test_edit_unknown_step_lists_full_leaf_paths(nested_plan, capsys):
    rc = orch.main(["plan", "edit", nested_plan, "--step", "zz", "--brief", "x"])
    out = capsys.readouterr().out
    assert rc == 2 and "无环节 zz" in out and "B.f2" in out


def test_reassign_accepts_short_leaf_id(nested_plan):
    rc = orch.main(
        ["plan", "reassign", nested_plan, "--step", "f2", "--member", "devops"]
    )
    assert rc == 0
    assert _leaf(nested_plan, "B.f2")["member"] == "devops"
