"""组长引导面修缮（backlog adhoc-guidance-fixes）：status 夜航读数 / plan 收尾 taskbook 联动 /
blocked×capacity 的 --session new。三条各自独立可验（本条任务书「逐条自证」）。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.orchestration.tools import orch, plans, queues, settings


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
