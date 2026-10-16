"""夜航模式 n2（task-20261014-psep2-n2）：判据层——AskUserQuestion 拦截 + queue_pressure 夜航判据。

a）PreToolUse 拦截：夜航 ON → exit 2 + stderr「转 twin 终裁」指令（提问工具成不可能）；
   OFF → exit 0 放行零变化；读开关异常 → **放行**（fail-open：不能因配置损坏废掉提问）。
b）巡更判据：夜航 ON 且 ``state/mirror-pending.json`` 非空 → 并入 PENDING-WORK（detail
   带夜航提示）；节流复用既有 30min 窗口；非夜航**不触发**（白天等用户正常）。
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import events, mirror, night, settings, watch

HOOK = PROJECT_ROOT / ".qoder" / "hooks" / "night-gate.sh"


@pytest.fixture
def face(tmp_path, monkeypatch):
    """开关面（settings.SETTINGS_PATH）+ 待决面（mirror.PENDING_PATH）都落 tmp。"""
    toml = tmp_path / "pysci_settings.toml"
    pending = tmp_path / "mirror-pending.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", toml)
    monkeypatch.setattr(mirror, "PENDING_PATH", pending)

    def set_switch(on: bool) -> None:
        toml.write_text(
            f"[night]\nenabled = {'true' if on else 'false'}\n", encoding="utf-8"
        )

    def set_pending(items: list[str]) -> None:
        pending.write_text(json.dumps({"items": items}), encoding="utf-8")

    return set_switch, set_pending


# ---------------------------------------------------------------------------
# a）拦截判定与 CLI（exit 2 / 0；fail-open=放行）
# ---------------------------------------------------------------------------
def test_gate_denies_with_twin_directive_when_night_on():
    deny = night.gate_deny_message({"night": {"enabled": True}})
    assert deny is not None
    assert "夜航模式" in deny and "twin", "consult" in deny and "twin 终裁" in deny
    assert "仍不等用户" in deny


def test_gate_allows_when_night_off_or_key_absent():
    assert night.gate_deny_message({"night": {"enabled": False}}) is None
    assert night.gate_deny_message({}) is None


def test_gate_fail_open_when_switch_read_raises(monkeypatch):
    def boom(data=None):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(night, "night_enabled", boom)
    assert night.gate_deny_message(None) is None, "读开关异常必须放行（fail-open）"


def test_gate_cli_two_states(tmp_path, capsys):
    toml = tmp_path / "pysci_settings.toml"
    # ON → rc 2 + stderr 指令
    toml.write_text("[night]\nenabled = true\n", encoding="utf-8")
    assert night.main(["--gate", "--settings", str(toml)]) == 2
    err = capsys.readouterr().err
    assert "夜航模式" in err and "twin", "consult" in err
    # OFF → rc 0 + 零 stderr
    toml.write_text("[night]\nenabled = false\n", encoding="utf-8")
    assert night.main(["--gate", "--settings", str(toml)]) == 0
    assert capsys.readouterr().err == ""


def test_gate_cli_fail_open_on_broken_or_absent_settings(tmp_path, capsys):
    broken = tmp_path / "broken.toml"
    broken.write_text("这不是 TOML [[[", encoding="utf-8")
    assert night.main(["--gate", "--settings", str(broken)]) == 0, "配置损坏 → 放行"
    assert capsys.readouterr().err == ""
    assert night.main(["--gate", "--settings", str(tmp_path / "absent.toml")]) == 0, (
        "配置缺席 → 放行"
    )


def test_gate_hook_script_fails_open_without_env(tmp_path):
    """bash 级：假根无 .venv → rc 0（放行）+ stderr 留 fail-open 痕，零副作用。"""
    if shutil.which("bash") is None:
        pytest.skip("需 bash（Git Bash）")
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=b'{"tool_name":"AskUserQuestion"}',
        capture_output=True,
        env={"QODER_PROJECT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert proc.returncode == 0
    assert b"[night-gate fail-open]" in proc.stderr


# ---------------------------------------------------------------------------
# b）queue_pressure 夜航判据（含节流与非夜航不触发）
# ---------------------------------------------------------------------------
def _night_reasons(face) -> list[str]:
    press = watch.queue_pressure()
    return [r for r in press["reasons"] if "夜航" in r]


def test_pressure_night_waiting_user_triggers_reason(face):
    set_switch, set_pending = face
    set_switch(True)
    set_pending(["选型待裁决"])
    reasons = _night_reasons(face)
    assert reasons and "不得等用户" in reasons[0] and "twin" in reasons[0]
    assert watch.queue_pressure()["pending"] is True


def test_pressure_night_silent_when_no_pending(face):
    set_switch, set_pending = face
    set_switch(True)
    set_pending([])
    assert _night_reasons(face) == []


def test_pressure_daytime_waiting_user_does_not_trigger(face):
    set_switch, set_pending = face
    set_switch(False)
    set_pending(["选型待裁决"])
    assert _night_reasons(face) == [], "白天等用户是正常的——非夜航不加此判据"


def test_pressure_night_reason_reaches_pending_work_event(face):
    """判据进既有 PENDING-WORK 单点：events 事件 detail 带夜航提示（不动消费面）。"""
    set_switch, set_pending = face
    set_switch(True)
    set_pending(["选型待裁决"])
    cur = watch.snapshot()
    evs, _verdict = watch.detect({}, cur, queue_watch=True)
    pw = [e for e in evs if e["event"] == events.PENDING_WORK]
    assert pw and "夜航" in pw[0]["detail"] and "不得等用户" in pw[0]["detail"]


def test_pressure_night_reason_throttled_by_existing_window(face):
    """节流复用既有 30min 账：同签名首帧出事件，窗口内第二帧被吞。"""
    set_switch, set_pending = face
    set_switch(True)
    set_pending(["选型待裁决"])
    cur = watch.snapshot()
    evs, verdict = watch.detect({}, cur, queue_watch=True)
    assert [e["event"] for e in evs if e["event"] == events.PENDING_WORK]
    throttle = dict(verdict["throttle"])
    assert events.PENDING_WORK in throttle, "首帧须记节流账（既有机制）"
    evs2, _v = watch.detect({}, watch.snapshot(), throttle=throttle, queue_watch=True)
    assert not [e for e in evs2 if e["event"] == events.PENDING_WORK], (
        "窗口内同条件不得重复告警（30min 节流既有实现）"
    )
