"""组长引导面修缮（backlog adhoc-guidance-fixes）：status 夜航读数 / plan 收尾 taskbook 联动 /
blocked×capacity 的 --session new。三条各自独立可验（本条任务书「逐条自证」）。

④ 追加（源起 backlog 20261016-215946，20261016-220500 定稿）：`plan new`/`plan advance` 的
`[NEXT]` 推进引导只讲事件驱动 `plan advance`（入队即返回、长任务首选）＋计划外 `plan adhoc`。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from pysci.skills.orchestration.tools import (
    orch,
    plans,
    queues,
    queueworker,
    settings,
    workq,
)


def _args(**over):
    from argparse import Namespace

    base = dict(id="p1", limit=5, days=0, stats=False, tail=10, member=None)
    base.update(over)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# ① orch status 夜航读数（两态 + settings 坏掉不崩）
# ---------------------------------------------------------------------------
def _settings(tmp: Path, night: str) -> Path:
    p = tmp / "pysci_settings.toml"
    p.write_text(
        f'[model]\npreset = "mimo"\n\n[night]\nenabled = {night}\n', encoding="utf-8"
    )
    return p


def test_status_shows_night_off(iso_state, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "SETTINGS_PATH", _settings(tmp_path, "false"))
    orch.cmd_status(_args())
    out = capsys.readouterr().out
    assert "== 夜航 ==" in out and "夜航：关闭" in out


def test_status_shows_night_on_with_behavior_hint(
    iso_state, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(settings, "SETTINGS_PATH", _settings(tmp_path, "true"))
    orch.cmd_status(_args())
    out = capsys.readouterr().out
    assert "夜航：开启" in out, out
    for frag in ("决策留痕", "AskUserQuestion", "patrol-watchdog.md §5"):
        assert frag in out, f"ON 态须给行为提示与细则索引：{frag}"


def test_status_survives_broken_settings(iso_state, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "pysci_settings.toml"
    bad.write_text("[night\nenabled = = =\n", encoding="utf-8")  # 坏 toml
    monkeypatch.setattr(settings, "SETTINGS_PATH", bad)
    orch.cmd_status(_args())  # 不得抛
    assert "夜航" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ② plan 完成 [NEXT] 的 taskbook 联动（关联给命令 / 未关联不加噪声）
# ---------------------------------------------------------------------------
def _complete(tmp_path, monkeypatch, *, linked: bool) -> str:
    from pysci import paths as _paths
    from pysci.skills.orchestration.tools import taskbooks

    monkeypatch.setattr(plans, "PLANS_DIR", tmp_path / "plans")
    (tmp_path / "plans").mkdir(exist_ok=True)
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tmp_path / "data" / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", tmp_path / "data")
    if linked:
        taskbooks.create("被关联的大任务", "正文")
        tid = taskbooks.list_all()[0]["id"]
        taskbooks.link_plan(tid, "p1")
        return tid
    return ""


def test_plan_complete_adds_taskbook_commands_when_linked(
    iso_state, tmp_path, monkeypatch, capsys
):
    tid = _complete(tmp_path, monkeypatch, linked=True)
    plans._report_plan_complete("p1", {"completed_at": "2026-10-15T18:00:00+08:00"})
    out = capsys.readouterr().out
    assert "pysci-orch taskbooks sync" in out, out
    assert f"pysci-orch taskbooks report {tid}" in out, out


def test_plan_complete_silent_when_unlinked(iso_state, tmp_path, monkeypatch, capsys):
    _complete(tmp_path, monkeypatch, linked=False)
    plans._report_plan_complete("p1", {"completed_at": "2026-10-15T18:00:00+08:00"})
    out = capsys.readouterr().out
    assert "taskbooks" not in out, f"未关联不加噪声：{out}"


# ---------------------------------------------------------------------------
# ③ blocked×capacity → [NEXT] 必须 --session new
# ---------------------------------------------------------------------------
def _blocked_scene(tmp_path, monkeypatch, *, tags: list[str], body: str = "") -> dict:
    step = {"id": "s1", "member": "devops", "session_id": "abc12345-dead-beef"}
    item = {
        "id": "j1",
        "plan": "p1",
        "step": "s1",
        "status": "done",
        "result": {"kind": "blocked", "tags": tags, "body": body},
    }
    monkeypatch.setattr(plans, "PLANS_DIR", tmp_path / "plans")
    (tmp_path / "plans").mkdir(exist_ok=True)
    monkeypatch.setattr(plans, "_queue_item_for_step", lambda *a, **k: item)
    monkeypatch.setattr(queues, "items", lambda member, **k: [item], raising=False)
    return step


def test_capacity_return_tells_next_to_use_new_session(
    iso_state, tmp_path, monkeypatch, capsys
):
    step = _blocked_scene(tmp_path, monkeypatch, tags=["model_capability"])
    plans._print_blocked_options("p1", step, {"member": "devops"})
    out = capsys.readouterr().out
    assert "--session new" in out, out
    assert "余量已尽" in out and "不复用" in out, out


def test_plain_blocked_keeps_session_reuse(iso_state, tmp_path, monkeypatch, capsys):
    step = _blocked_scene(tmp_path, monkeypatch, tags=["whitelist_conflict"])
    plans._print_blocked_options("p1", step, {"member": "devops"})
    out = capsys.readouterr().out
    assert "--session new" not in out, out
    assert "续作自动复用" in out, out


# ---------------------------------------------------------------------------
# ④ [NEXT] 推进引导只走 advance（backlog 20261016-215946 → 20261016-220500）
#   现象：new/advance 的 [NEXT] 曾只推阻塞式同步子命令，照抄即把长任务跑成阻塞、撞 Bash 超时。
#   220500 后：推进引导只讲 `plan advance`（事件驱动）＋ `plan adhoc`（计划外同步）。
# ---------------------------------------------------------------------------
def _plan_args(plan_id: str, **over) -> Namespace:
    base = dict(
        id=plan_id,
        file="",
        force=True,
        text=None,
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
        evidence=None,
        dry=False,
    )
    base.update(over)
    return Namespace(**base)


def _plan_face(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "plans"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    return d


def _mk_plan(d: Path, plan_id: str, *, steps=None) -> str:
    p = d / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "id": plan_id,
                "goal": f"目标 {plan_id}",
                "steps": steps or [{"id": "s1", "member": "sim", "brief": "环节一"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    plans.cmd_plan_new(_plan_args(plan_id, file=str(p)))
    return plan_id


def _fake_worker(member: str, **kw):
    qp = queues.queue_path(member)
    for it in workq.load(qp):
        if str(it.get("status")) == workq.STATUS_DONE:
            continue
        workq.mutate(
            qp,
            it["id"],
            status=workq.STATUS_DONE,
            result={
                "code": 0,
                "kind": "result",
                "body": "done",
                "checks": [],
                "member": member,
            },
        )
    return (True, "stub worker")


#: 已废弃的同步子命令名（运行期拼接：防常量折叠让含二进制的 grep 误命中，任务书 grep 约定）。
_DEAD_SYNC = "plan " + "run"


def _advance_only(out: str, plan_id: str) -> None:
    """推进引导只讲 advance（＋ adhoc 指路），无已废子命令残留。"""
    assert f"plan advance {plan_id}" in out, out
    assert "入队即返回" in out, out
    assert "plan adhoc" in out, out
    assert _DEAD_SYNC not in out, out


def test_plan_new_next_uses_advance_only(iso_state, tmp_path, monkeypatch, capsys):
    d = _plan_face(tmp_path, monkeypatch)
    capsys.readouterr()
    _mk_plan(d, "p-new")
    _advance_only(capsys.readouterr().out, "p-new")


def test_plan_advance_next_uses_advance_only(iso_state, tmp_path, monkeypatch, capsys):
    d = _plan_face(tmp_path, monkeypatch)
    _mk_plan(d, "p-adv")
    capsys.readouterr()
    # 未给任务书 → 只印引导并 rc=2（延迟书写）
    assert plans.cmd_plan_advance(_plan_args("p-adv")) == 2
    _advance_only(capsys.readouterr().out, "p-adv")


def test_handoff_next_uses_advance_only(iso_state, tmp_path, monkeypatch, capsys):
    """交接决策点 [NEXT] 同样只讲 advance。"""
    d = _plan_face(tmp_path, monkeypatch)
    _mk_plan(
        d,
        "p-run",
        steps=[
            {"id": "s1", "member": "sim", "brief": "甲"},
            {"id": "s2", "member": "sim", "brief": "乙"},
        ],
    )
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: _fake_worker(m))
    capsys.readouterr()
    # 阻塞推进原语（内部；`plan adhoc` 走它）——交接面照常打印 [NEXT]
    plans.cmd_plan_run(_plan_args("p-run", text="t"))
    out = capsys.readouterr().out
    assert "交接决策点" in out, out
    assert "plan advance p-run" in out, out
    assert _DEAD_SYNC not in out, out


# ---------------------------------------------------------------------------
# ⑤ `plan advance` 承接已废子命令的旗标与续作语义（backlog 20261016-220500）
# ---------------------------------------------------------------------------
def _spy_enqueue(monkeypatch) -> dict:
    seen: dict = {}
    real = queues.enqueue

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(queues, "enqueue", spy)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    return seen


def test_plan_advance_absorbs_name_and_max_turns(
    iso_state, tmp_path, monkeypatch, capsys
):
    """`--name`/`--max-turns`（原同步子命令独有）由 advance 透传到 enqueue。"""
    d = _plan_face(tmp_path, monkeypatch)
    _mk_plan(d, "p-abs")
    seen = _spy_enqueue(monkeypatch)
    capsys.readouterr()
    rc = plans.cmd_plan_advance(
        _plan_args("p-abs", text="t", name="custom-slug", max_turns=7)
    )
    assert rc == 0, rc
    assert seen.get("slug") == "custom-slug", seen
    assert seen.get("max_turns") == 7, seen


def test_plan_advance_resumes_step_session(iso_state, tmp_path, monkeypatch, capsys):
    """续作 blocked 环节复用该环节会话（承接原同步子命令的续作语义）。"""
    d = _plan_face(tmp_path, monkeypatch)
    _mk_plan(d, "p-res")
    st = plans.load_state("p-res")
    st["steps"][0]["status"] = "blocked"
    st["steps"][0]["session_id"] = "abcd1234-dead-beef"
    plans._save_state("p-res", st)
    seen = _spy_enqueue(monkeypatch)
    capsys.readouterr()
    assert plans.cmd_plan_advance(_plan_args("p-res", text="补充")) == 0
    assert seen.get("session") == "abcd1234", seen
