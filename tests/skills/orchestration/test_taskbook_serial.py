"""大任务串行执行模式（用户 2026-10-17 裁决）：队列谓词 / TASKBOOK-NEW 排队语义 / 热切换。

语义钉住三件事：按任务书**产生时序**严格排队、一次只允许一个大任务进行；解锁点＝
更早任务书离开 new/planning/active（delivered/failed/accepted/rejected 均解锁，用户
批复与下一任务并行）；返工件**不继承家族位置**（ts=新生时刻，天然排队尾）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pysci import paths as _paths
from pysci.skills.orchestration.tools import events, settings, taskbooks

T0 = datetime(2026, 10, 17, 8, 0, 0)


def _at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def serial_env(tmp_path, monkeypatch):
    """隔离任务书目录 + settings 真源；返回改模式的 helper（默认 serial）。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    path = tmp_path / "pysci_settings.toml"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)

    def _set_mode(mode: str) -> None:
        path.write_text(f'[taskbooks]\nexecution_mode = "{mode}"\n', encoding="utf-8")

    _set_mode("serial")
    return _set_mode


# ---------------------------------------------------------------------------
# ① 队列谓词
# ---------------------------------------------------------------------------
def test_queue_rows_follow_creation_order(serial_env):
    a = taskbooks.create("A", "a", now=_at(0))["id"]
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    c = taskbooks.create("C", "c", now=_at(2))["id"]
    rows = taskbooks.queue_rows()
    assert [r["id"] for r in rows] == [a, b, c]
    assert [r["queue_no"] for r in rows] == [1, 2, 3]
    assert all(r["queue_total"] == 3 for r in rows)
    assert taskbooks.serial_head()["id"] == a


def test_rework_taskbook_lands_at_queue_tail(serial_env):
    """返工件不继承家族位置：即使驳回的是队首，返工件也按产生时序排队尾（用户裁决）。"""
    a = taskbooks.create("A", "a", now=_at(0))["id"]
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    c = taskbooks.create("C", "c", now=_at(2))["id"]
    rw = taskbooks.create_rework(a, improve={}, note="返工", now=_at(3))["id"]
    rows = taskbooks.queue_rows()
    assert [r["id"] for r in rows] == [b, c, rw], "原件 rejected 离队；返工件落队尾"


def test_unlock_when_earlier_taskbooks_settle(serial_env):
    a = taskbooks.create("A", "a", now=_at(0))["id"]
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    assert not taskbooks.is_unlocked(b)
    taskbooks.transition(a, "planning")
    assert not taskbooks.is_unlocked(b), "planning 仍占队"
    taskbooks.transition(a, "active")
    assert not taskbooks.is_unlocked(b), "active 仍占队"
    taskbooks.transition(a, "delivered")
    assert taskbooks.is_unlocked(b), "delivered 即解锁（用户批复与下一任务并行）"


@pytest.mark.parametrize("settle", ["delivered", "failed", "accepted", "rejected"])
def test_all_settled_states_unlock(serial_env, settle):
    a = taskbooks.create("A", "a", now=_at(0))["id"]
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    taskbooks.transition(a, "active")
    if settle == "delivered":
        taskbooks.transition(a, "delivered")
    elif settle == "failed":
        taskbooks.transition(a, "failed")
    elif settle == "accepted":
        taskbooks.transition(a, "delivered")
        taskbooks.accept(a)
    else:
        taskbooks.transition(a, "delivered")
        taskbooks.create_rework(a, improve={}, note="x", now=_at(9))
    assert taskbooks.is_unlocked(b), settle


def test_unknown_id_is_unlocked_fail_soft(serial_env):
    assert taskbooks.is_unlocked("tb-none") is True


# ---------------------------------------------------------------------------
# ② 硬闸文案
# ---------------------------------------------------------------------------
def test_gate_blocks_with_position_and_passes_head(serial_env):
    a = taskbooks.create("A", "a", now=_at(0))["id"]
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    gate = taskbooks.queue_gate(b)
    assert "串行执行模式" in gate and "队列第 2 位" in gate
    assert f"队首={a}" in gate and "未执行任何变更" in gate
    assert taskbooks.queue_gate(a) == "", "队首放行"


def test_gate_absent_in_parallel_mode(serial_env):
    serial_env("parallel")
    taskbooks.create("A", "a", now=_at(0))
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    assert taskbooks.queue_gate(b) == "", "parallel 无闸"


def test_hot_switch_takes_effect_immediately(serial_env):
    """热切换保证：改 toml 下一次调用即生效（读数无缓存，谓词无状态）。"""
    taskbooks.create("A", "a", now=_at(0))
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    assert taskbooks.queue_gate(b) != ""
    serial_env("parallel")
    assert taskbooks.queue_gate(b) == ""
    serial_env("serial")
    assert taskbooks.queue_gate(b) != ""


# ---------------------------------------------------------------------------
# ③ TASKBOOK-NEW 排队语义（组长侧零规程零决策）
# ---------------------------------------------------------------------------
def test_create_event_queue_guidance_when_serial(serial_env):
    taskbooks.create("A", "a", now=_at(0))
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    evs = [e for e in events.read_all() if e["event"] == "TASKBOOK-NEW"]
    assert evs[-1]["subject"] == b
    assert "已排队第 2 位" in evs[-1]["action"]
    assert "无需行动" in evs[-1]["action"]
    assert "taskbooks link" in evs[0]["action"], "队首仍推正常拆解待办"


def test_create_event_normal_guidance_when_parallel(serial_env):
    serial_env("parallel")
    taskbooks.create("A", "a", now=_at(0))
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    evs = [e for e in events.read_all() if e["event"] == "TASKBOOK-NEW"]
    assert evs[-1]["subject"] == b
    assert "已排队" not in evs[-1]["action"] and "taskbooks link" in evs[-1]["action"]


# ---------------------------------------------------------------------------
# ④ 硬闸（rc=2 拒绝 + 放行面；「才允许进行」的机械执行）
# ---------------------------------------------------------------------------
def _mk_plan(tmp_path, monkeypatch, plan_id: str, tb_id: str):
    import yaml

    from pysci.skills.orchestration.tools import plans

    d = tmp_path / "plans"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    p = d / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "id": plan_id,
                "goal": "g",
                "taskbook": tb_id,
                "steps": [{"id": "s1", "member": "devops", "brief": "b"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return p


def _rework_args(tb_id: str, **over):
    from argparse import Namespace

    base = dict(
        id=tb_id,
        improve="",
        mode=None,
        goal="",
        plan_id=None,
        force=True,
        text=None,
        task=None,
        evidence=None,
        skip="",
        session=None,
        dirs=None,
        name=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dry=False,
        file="",
    )
    base.update(over)
    return Namespace(**base)


def test_gate_blocks_advance_link_rework_new_for_queued(
    serial_env, tmp_path, monkeypatch, capsys
):
    from pysci.skills.orchestration.tools import orch, plans

    taskbooks.create("A", "a", now=_at(0))
    b = taskbooks.create("B", "b", now=_at(1))["id"]
    p = _mk_plan(tmp_path, monkeypatch, "p-b", b)
    assert plans.cmd_plan_advance(_rework_args("p-b", skip="")) == 2
    out = capsys.readouterr().out
    assert "串行执行模式" in out and "未执行任何变更" in out

    assert orch.main(["taskbooks", "link", b, "--plan", "p-b"]) == 2
    assert "串行执行模式" in capsys.readouterr().out

    assert plans.cmd_plan_rework(_rework_args(b)) == 2
    assert "串行执行模式" in capsys.readouterr().out

    assert plans.cmd_plan_new(_rework_args("p-b2", file=str(p))) == 2
    assert "串行执行模式" in capsys.readouterr().out


def test_gate_passes_head_and_parallel(serial_env, tmp_path, monkeypatch):
    from pysci.skills.orchestration.tools import orch, plans

    a = taskbooks.create("A", "a", now=_at(0))["id"]
    taskbooks.create("B", "b", now=_at(1))
    p = _mk_plan(tmp_path, monkeypatch, "p-a", a)
    assert orch.main(["taskbooks", "link", a, "--plan", "p-a"]) == 0, "队首放行"
    assert plans.cmd_plan_new(_rework_args("p-a2", file=str(p))) == 0, "队首放行"
    serial_env("parallel")
    b = [s for s in taskbooks.list_all() if s["id"] != a][0]["id"]
    assert orch.main(["taskbooks", "link", b, "--plan", "p-b"]) == 0, "parallel 无闸"
