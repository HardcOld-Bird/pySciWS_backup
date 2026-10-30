"""唤醒候选的会话种类过滤（用户 2026-10-29 裁决；backlog 20261029-212328-leader）。

事故：空位补位把唤醒权判给 **QoderCN 桌面版应用会话**（b70794b5），真 CLI 组长会话按
「永不抢占」永远进不了门——TASKBOOK-NEW/JOB-DONE 全被 rewake 注入桌面会话。
规则（leader-only.md §4）：候选仅限 qodercn CLI 会话；桌面版应用会话、未运行 CLI 的
普通终端、判别不了者一律不入候选（fail-close）；无合格 CLI 会话时不认领（事件留水位），
CLI 会话出现即自动认领；owner=CLI 时桌面会话 Stop 不监听不抢。
"""

from __future__ import annotations

import json

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import live_sessions, watch

#: 真实种类判别函数（collection 期捕获——autouse fixture 会把模块属性换成 cli 替身）。
REAL_SESSION_KIND = live_sessions.session_kind


def _payload(sid: str, event: str = "Stop") -> str:
    return json.dumps(
        {
            "session_id": sid,
            "hook_event_name": event,
            "cwd": str(PROJECT_ROOT),
            "transcript_path": "/tmp/t.jsonl",
        },
        ensure_ascii=False,
    )


def _table(*rows: tuple[int, int, str]) -> dict[int, dict]:
    """进程快照：``(pid, ppid, name)`` 行 → ``session_kind`` 的 table 形。"""
    return {
        pid: {"ppid": ppid, "name": name, "create_time": None}
        for pid, ppid, name in rows
    }


def _kind(monkeypatch, kind: str, reason: str = "") -> None:
    """覆盖 autouse 的 cli 缺省（本文件专测桌面/未知路径）。"""
    monkeypatch.setattr(
        live_sessions, "session_kind", lambda *a, **k: (kind, reason or kind)
    )


# ---------------------------------------------------------------------------
# 种类判别（纯函数：进程表注入）
# ---------------------------------------------------------------------------
def test_session_kind_cli_desktop_unknown():
    cli = _table((10, 20, "bash.exe"), (20, 0, "qoderclicn.exe"))
    assert REAL_SESSION_KIND(10, table=cli)[0] == "cli"

    # 桌面版内嵌 agent：链上既有 qoderclicn 又有 Qoder CN.exe → 仍判 desktop（先于 CLI）
    desktop = _table(
        (10, 20, "bash.exe"), (20, 30, "qoderclicn.exe"), (30, 0, "Qoder CN.exe")
    )
    assert REAL_SESSION_KIND(10, table=desktop)[0] == "desktop"

    plain = _table((10, 20, "bash.exe"), (20, 0, "WindowsTerminal.exe"))
    assert REAL_SESSION_KIND(10, table=plain)[0] == "unknown"
    assert REAL_SESSION_KIND(10, table={})[0] == "unknown"


def test_desktop_exe_matcher():
    assert live_sessions.is_desktop_exe("Qoder CN.exe")
    assert not live_sessions.is_desktop_exe("qoderclicn.exe")


# ---------------------------------------------------------------------------
# 候选过滤与归属（hook_decision 单一实现）
# ---------------------------------------------------------------------------
def test_desktop_only_never_claimed(iso_state, monkeypatch):
    """桌面会话独存 → 不认领（rc=3）、候选集空、owner 记录不写。"""
    _kind(monkeypatch, "desktop", "祖先含桌面版应用（Qoder CN.exe）")
    code, reason = watch.hook_decision(_payload("desk-1"))
    assert code == 3 and "非 qodercn CLI" in reason
    assert live_sessions.ordered_candidates() == []
    assert str(watch.load_owner().get("session_id") or "") == ""


def test_cli_appears_after_desktop_auto_claims(iso_state, monkeypatch):
    """桌面独存时不认领；CLI 出现后自动认领（事件留水位语义）。"""
    _kind(monkeypatch, "desktop")
    assert watch.hook_decision(_payload("desk-1"))[0] == 3
    _kind(monkeypatch, "cli")
    code, reason = watch.hook_decision(_payload("cli-1"))
    assert code == 0 and "adopt" in reason
    assert watch.load_owner()["session_id"] == "cli-1"


def test_cli_owner_denies_desktop_stop(iso_state, monkeypatch):
    """owner=CLI 时桌面会话 Stop → rc=3 不监听不抢，owner 不变。"""
    _kind(monkeypatch, "cli")
    assert watch.hook_decision(_payload("cli-1"))[0] == 0
    _kind(monkeypatch, "desktop")
    code, reason = watch.hook_decision(_payload("desk-1"))
    assert code == 3 and "不监听" in reason
    assert watch.load_owner()["session_id"] == "cli-1"


def test_unknown_kind_is_fail_closed(iso_state):
    """判别不了的登记条目一律不入候选（fail-close）。"""
    live_sessions.register_session(
        "mystery", str(PROJECT_ROOT), kind="unknown", kind_reason="判别不了"
    )
    assert live_sessions.ordered_candidates() == []


def test_desktop_owner_record_not_kept(iso_state, monkeypatch):
    """历史脏记录：owner 指向桌面会话（存活）→ 不再保持，CLI 候选在即补位。"""
    _kind(monkeypatch, "desktop")
    watch.hook_decision(_payload("desk-1"))  # 只登记（rc=3，不写 owner）
    watch.save_owner(watch._new_owner("desk-1"))  # 模拟历史脏 owner 记录
    _kind(monkeypatch, "cli")
    code, reason = watch.hook_decision(_payload("cli-1"))
    assert code == 0 and "adopt" in reason
    assert watch.load_owner()["session_id"] == "cli-1"
