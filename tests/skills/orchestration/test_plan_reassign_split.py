"""`plan reassign` / `plan split`：占位符改派与单叶拆分的命令面（backlog 20261017-104504）。

现场：返工模板的 b2 是 ``member: devops + member_placeholder: true``（「按 §5 分工于拆解
环节改派」），但改派/拆叶此前只能手编 plan YAML 再 advance——必留叶子 lint 的容忍边界与
YAML 缩进错误都要试了才知道。两个命令内部走 amend 同款护栏（模板必留节点 / started 冻结），
并额外对**已 started 叶子**直接拒绝。
"""

from __future__ import annotations

import argparse

import pytest

from pysci.skills.orchestration.tools import orch, plans

PLAN_SRC = """id: p1
goal: demo
steps:
  - id: A
    mode: serial
    brief: 串行组
    children:
      - id: a1
        member: devops
        brief: 占位环节
        member_placeholder: true
      - id: a2
        member: figure
        brief: 后续环节
"""


@pytest.fixture
def plan(qface, tmp_path, capsys):
    src = tmp_path / "p1.plan-src.yaml"
    src.write_text(PLAN_SRC, encoding="utf-8")
    assert (
        plans.cmd_plan_new(argparse.Namespace(file=str(src), id="p1", force=False)) == 0
    )
    capsys.readouterr()
    return "p1"


def _leaf(pid: str, sid: str) -> dict:
    return plans._leaf_map(plans._load_yaml(pid))[sid]


def _set_status(pid: str, sid: str, status: str) -> None:
    st = plans._load_state(pid)
    for row in st["steps"]:
        if row["id"] == sid:
            row["status"] = status
    plans._save_state(pid, st)


def test_reassign_placeholder_then_advance_ok(plan, capsys):
    """占位叶改派：member 更新、占位符清除，改派后 advance 正常（criteria ①）。"""
    rc = orch.main(["plan", "reassign", plan, "--step", "A.a1", "--member", "writing"])
    out = capsys.readouterr().out
    assert rc == 0, out
    leaf = _leaf(plan, "A.a1")
    assert leaf["member"] == "writing"
    assert "member_placeholder" not in leaf, "占位符应在改派后清除"
    assert "占位符已清" in out

    assert (
        orch.main(["plan", "advance", plan, "--text", "占位环节任务书", "--dry"]) == 0
    )


def test_reassign_started_leaf_refused(plan, capsys):
    """已 started（含 done）的叶子改派被拒，且 YAML 不动（criteria ③）。"""
    _set_status(plan, "A.a1", "queued")

    rc = orch.main(["plan", "reassign", plan, "--step", "A.a1", "--member", "writing"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "已 queued" in out and "reassign 被拒" in out
    assert _leaf(plan, "A.a1")["member"] == "devops"


def test_split_inserts_new_leaf_before(plan, capsys):
    """拆叶：新叶插在原叶之前，原叶 id/描述不变，状态对账出新叶 pending。"""
    rc = orch.main(
        [
            "plan",
            "split",
            plan,
            "--step",
            "A.a1",
            "--before",
            "--member",
            "sim",
            "--brief",
            "前置环节",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    tree = plans._tree(plans._load_yaml(plan))
    assert tree.leaf_paths() == ["A.a1_pre", "A.a1", "A.a2"]
    assert _leaf(plan, "A.a1_pre") == {
        "id": "a1_pre",
        "member": "sim",
        "brief": "前置环节",
    }
    rows = {s["id"]: s for s in plans._load_state(plan)["steps"]}
    assert rows["A.a1_pre"]["status"] == "pending"
    assert rows["A.a1"]["status"] == "pending"


def test_split_keeps_rework_required_leaves(qface, capsys):
    """拆分必留叶（rework 模板的 B.b2）后，必留叶子仍齐全且顺序不变（criteria ②）。"""
    assert (
        plans.cmd_plan_rework(
            argparse.Namespace(
                id="tb-x", mode="full", goal="", plan_id="rework-tb-x", force=False
            )
        )
        == 0
    )
    capsys.readouterr()

    rc = orch.main(
        [
            "plan",
            "split",
            "rework-tb-x",
            "--step",
            "BC.B.b2",
            "--before",
            "--member",
            "sim",
            "--brief",
            "预拆解",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    paths = plans._tree(plans._load_yaml("rework-tb-x")).leaf_paths()
    assert paths.index("BC.B.b2_pre") + 1 == paths.index("BC.B.b2")
    required = plans.REWORK_REQUIRED[plans.MODE_FULL]
    assert [p for p in paths if p in required] == list(required)


def test_split_started_leaf_refused(plan, capsys):
    _set_status(plan, "A.a1", "done")

    rc = orch.main(
        [
            "plan",
            "split",
            plan,
            "--step",
            "A.a1",
            "--after",
            "--member",
            "sim",
            "--brief",
            "x",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "已 done" in out and "split 被拒" in out
    assert plans._tree(plans._load_yaml(plan)).leaf_paths() == ["A.a1", "A.a2"]


def test_commands_documented_in_command_surface_and_help(capsys):
    """发现面：两个新命令在命令全表与 `plan -h` 里都在。"""
    from pysci.paths import ORCHESTRATION_ROOT

    md = (
        ORCHESTRATION_ROOT
        / "skills"
        / "orchestration"
        / "references"
        / "command-surface.md"
    ).read_text(encoding="utf-8")
    assert "plan reassign" in md and "plan split" in md

    with pytest.raises(SystemExit):
        orch.main(["plan", "-h"])
    out = capsys.readouterr().out
    assert "reassign" in out and "split" in out


def test_split_duplicate_new_id_refused(plan, capsys):
    rc = orch.main(
        [
            "plan",
            "split",
            plan,
            "--step",
            "A.a1",
            "--after",
            "--member",
            "sim",
            "--brief",
            "x",
            "--new-id",
            "a2",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "已存在环节 a2" in out
