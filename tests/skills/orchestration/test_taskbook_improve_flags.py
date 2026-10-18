"""改进项三标志（improve）schema：legacy 映射 / 新旧签名互斥 / 落盘字段面。

用户 2026-10-17 裁决：reject 三选一（full/no_inspector/rework_only）改三复选
（infra/leader/inspector，8 组合皆合法）。真源 = ``state["improve"]``；``mode`` 降级为
派生兼容字段（:func:`legacy_mode_of`），存量任务书按旧三档映射回落（:func:`improve_flags`）。
（8 标志组合的返工骨架断言见 test_rework_flow / test_taskbook_improve_flags 的骨架段。）
"""

from __future__ import annotations

import json

import pytest

from pysci import paths as _paths
from pysci.skills.orchestration.tools import plans, taskbooks


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d = data / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return d


# ---------------------------------------------------------------------------
# ① 读侧：legacy mode 三档 → 标志
# ---------------------------------------------------------------------------
def test_improve_flags_legacy_mode_mapping():
    assert taskbooks.improve_flags({"mode": "full"}) == {
        "infra": True,
        "leader": False,
        "inspector": True,
    }
    assert taskbooks.improve_flags({"mode": "no_inspector"}) == {
        "infra": True,
        "leader": False,
        "inspector": False,
    }
    assert taskbooks.improve_flags({"mode": "rework_only"}) == {
        "infra": False,
        "leader": False,
        "inspector": False,
    }
    assert taskbooks.improve_flags({}) == {
        "infra": False,
        "leader": False,
        "inspector": False,
    }
    assert taskbooks.improve_flags(None)["leader"] is False


def test_improve_flags_takes_precedence_over_mode():
    st = {"mode": "full", "improve": {"leader": True}}
    assert taskbooks.improve_flags(st) == {
        "infra": False,
        "leader": True,
        "inspector": False,
    }, "improve 是真源；mode 只是派生兼容显示"


def test_legacy_mode_of_derives_three_modes():
    f = lambda i, ldr, s: {"infra": i, "leader": ldr, "inspector": s}  # noqa: E731
    assert taskbooks.legacy_mode_of(f(True, False, True)) == "full"
    assert taskbooks.legacy_mode_of(f(True, True, False)) == "no_inspector"
    assert taskbooks.legacy_mode_of(f(False, True, False)) == "rework_only"
    assert taskbooks.legacy_mode_of(f(True, True, True)) == "full", "leader 不参与派生"


# ---------------------------------------------------------------------------
# ② 写侧：create / create_rework 新旧签名
# ---------------------------------------------------------------------------
def test_create_writes_improve_truth_and_derived_mode(tbdir):
    st = taskbooks.create(
        "返工件",
        "正文",
        improve={"infra": True, "leader": True, "inspector": False},
        rework_of="tb-old",
        rework_round=1,
    )
    assert st["improve"] == {"infra": True, "leader": True, "inspector": False}
    assert st["mode"] == "no_inspector"  # 旧三档派生
    md = (tbdir / f"{st['id']}.taskbook.md").read_text(encoding="utf-8")
    assert "improve: infra,leader" in md
    state = json.loads((tbdir / f"{st['id']}.state.json").read_text(encoding="utf-8"))
    assert state["improve"]["leader"] is True


def test_create_non_rework_keeps_mode_blank(tbdir):
    """非返工件 mode 恒留空——「仅返工」挂到新任务书上是误导显示。"""
    st = taskbooks.create("新件", "正文", improve={"leader": True})
    assert st["mode"] == ""
    assert st["improve"] == {"infra": False, "leader": True, "inspector": False}
    assert taskbooks.improve_flags(st)["leader"] is True


def test_create_legacy_mode_entry_maps_to_flags(tbdir):
    st = taskbooks.create(
        "旧口", "正文", mode="full", rework_of="tb-old", rework_round=1
    )
    assert st["improve"] == {"infra": True, "leader": False, "inspector": True}
    assert st["mode"] == "full"


def test_create_rejects_improve_mode_both_or_unknown_key(tbdir):
    with pytest.raises(ValueError):
        taskbooks.create("x", "y", improve={"infra": True}, mode="full")
    with pytest.raises(ValueError):
        taskbooks.create("x", "y", improve={"infra": True, "nope": True})
    with pytest.raises(ValueError):
        taskbooks.create("x", "y", mode="deluxe")


def test_create_rework_new_and_legacy_entries(tbdir):
    orig = taskbooks.create("原", "正文")["id"]
    rw = taskbooks.create_rework(orig, improve={"leader": True}, note="批语")
    assert rw["improve"] == {"infra": False, "leader": True, "inspector": False}
    assert rw["mode"] == "rework_only"
    assert rw["rework_of"] == orig and rw["rework_round"] == 1

    orig2 = taskbooks.create("原2", "正文")["id"]
    rw2 = taskbooks.create_rework(orig2, mode="no_inspector", note="批语")
    assert rw2["improve"] == {"infra": True, "leader": False, "inspector": False}


def test_create_rework_requires_one_of_improve_or_mode(tbdir):
    orig = taskbooks.create("原", "正文")["id"]
    with pytest.raises(ValueError):
        taskbooks.create_rework(orig, improve={"infra": True}, mode="full")
    with pytest.raises(ValueError):
        taskbooks.create_rework(orig)


def test_rejection_entries_carry_round(tbdir):
    a = taskbooks.create("原", "正文")["id"]
    rw1 = taskbooks.create_rework(a, improve={}, note="第一轮批语")["id"]
    rw2 = taskbooks.create_rework(rw1, improve={}, note="第二轮批语")["id"]
    assert taskbooks.load_state(a)["rejections"][-1] == {
        **taskbooks.load_state(a)["rejections"][-1],
        "note": "第一轮批语",
        "round": 1,
    }
    assert taskbooks.load_state(rw1)["rejections"][-1]["note"] == "第二轮批语"
    assert taskbooks.load_state(rw1)["rejections"][-1]["round"] == 2
    assert taskbooks.load_state(rw2)["rejections"] == []


# ---------------------------------------------------------------------------
# ③ 返工骨架 8 组合（_rework_layout 单源；用户 2026-10-17 三复选）
# ---------------------------------------------------------------------------
A_LEAVES = ("A.a1", "A.a2", "A.a3", "A.a4", "A.a5")
B_LEAVES = ("B.b1", "B.b2", "B.b_last")
BC_B_LEAVES = ("BC.B.b1", "BC.B.b2", "BC.B.b_last")
C_LEAVES = ("BC.C.c1", "BC.C.c2", "BC.C.c3", "BC.C.c4")
L_LEAVES = ("L.l1", "L.l2", "L.l3")


def _f(i: bool = False, lead: bool = False, s: bool = False) -> dict:
    return {"infra": i, "leader": lead, "inspector": s}


@pytest.mark.parametrize(
    "flags,top,required",
    [
        (_f(), ["B"], B_LEAVES),
        (_f(i=True), ["A", "B"], A_LEAVES + B_LEAVES),
        (_f(s=True), ["BC"], BC_B_LEAVES + C_LEAVES),
        (_f(i=True, s=True), ["A", "BC"], A_LEAVES + BC_B_LEAVES + C_LEAVES),
        (_f(lead=True), ["B", "L"], B_LEAVES + L_LEAVES),
        (_f(i=True, lead=True), ["A", "B", "L"], A_LEAVES + B_LEAVES + L_LEAVES),
        (_f(s=True, lead=True), ["BC", "L"], BC_B_LEAVES + C_LEAVES + L_LEAVES),
        (
            _f(i=True, lead=True, s=True),
            ["A", "BC", "L"],
            A_LEAVES + BC_B_LEAVES + C_LEAVES + L_LEAVES,
        ),
    ],
)
def test_layout_per_flag_combo(flags, top, required):
    plan = plans.build_rework_plan("tb-x", improve=flags)
    assert [s["id"] for s in plan["steps"]] == top
    assert plans.rework_required_leaves(flags) == tuple(required)
    assert plan["improve"] == flags and plan["template"] == "rework"
    assert plans.template_violations(plan) == []


def test_legacy_mode_tables_derive_from_layout():
    """REWORK_REQUIRED 旧三档表＝layout 派生（单源不漂开）；--mode 入口映射同源。"""
    assert plans.REWORK_REQUIRED["full"] == A_LEAVES + BC_B_LEAVES + C_LEAVES
    assert plans.REWORK_REQUIRED["no_inspector"] == A_LEAVES + B_LEAVES
    assert plans.REWORK_REQUIRED["rework_only"] == B_LEAVES
    legacy = plans.build_rework_plan("tb-x", mode="full")
    assert legacy["improve"] == _f(i=True, s=True)
    assert (
        plans.rework_required_leaves(legacy["improve"]) == plans.REWORK_REQUIRED["full"]
    )


def test_build_rework_plan_rejects_both_sources():
    with pytest.raises(ValueError):
        plans.build_rework_plan("tb-x", improve={"infra": True}, mode="full")
    with pytest.raises(ValueError):
        plans.build_rework_plan("tb-x", mode="deluxe")


def test_leader_step_ready_only_after_body_done():
    """L 恒居顶层末尾：前序全 done 才就绪（「主体完成后才进组长改进」机械保证）。"""
    plan = plans.build_rework_plan("tb-x", improve={"leader": True})
    tree = plans._tree(plan)
    status = {p: "pending" for p in tree.leaf_paths()}
    assert tree.dispatchable(lambda p: status.get(p, "pending")) == ["B.b1"]
    for p in tree.leaf_paths():
        if not p.startswith("L."):
            status[p] = "done"
    assert tree.dispatchable(lambda p: status.get(p, "pending")) == ["L.l1"]


# ---------------------------------------------------------------------------
# ④ plan rework 改进项三来源（--improve ＞ --mode ＞ 任务书标志）
# ---------------------------------------------------------------------------
def _rework_args(tb_id: str, **over):
    from argparse import Namespace

    base = dict(id=tb_id, improve="", mode=None, goal="", plan_id=None, force=True)
    base.update(over)
    return Namespace(**base)


def test_plan_rework_resolves_three_flag_sources(qface, tbdir, capsys):
    """qface 先落（iso_state 会改 TASKBOOKS_DIR），tbdir 后置——防 stray 护栏判据错位。"""
    orig = taskbooks.create("原", "正文")["id"]
    rw = taskbooks.create_rework(orig, improve={"leader": True}, note="n")["id"]
    capsys.readouterr()
    assert plans.cmd_plan_rework(_rework_args(rw)) == 0
    out = capsys.readouterr().out
    assert "标志来源：任务书标志" in out
    assert plans._load_yaml(f"rework-{rw}")["improve"] == {
        "infra": False,
        "leader": True,
        "inspector": False,
    }
    assert plans.cmd_plan_rework(_rework_args(rw, mode="no_inspector")) == 0
    out = capsys.readouterr().out
    assert "标志来源：兼容 --mode 映射" in out
    assert plans._load_yaml(f"rework-{rw}")["improve"]["infra"] is True
    assert plans.cmd_plan_rework(_rework_args(rw, improve="infra,leader")) == 0
    out = capsys.readouterr().out
    assert "标志来源：显式 --improve" in out
    assert plans._load_yaml(f"rework-{rw}")["improve"] == {
        "infra": True,
        "leader": True,
        "inspector": False,
    }
    assert plans.cmd_plan_rework(_rework_args(rw, improve="nope")) == 2
    assert "未知改进标志" in capsys.readouterr().out
