"""唤醒/夜航绑定主会话（wake-bind-20261018 s2；twin 裁决 A）。

用户要求（2026-10-18）：watchdog/夜航的**自动唤醒始终绑定主会话**；子会话（主会话跑大
任务期间另开的第二个组长会话）绝不自动唤醒；pod cwd 的组员直聊会话永不被认领/监听；
**取消全部自动转让**（stale 接管、SessionEnd 自动升主），迁移只走人工 ``--force``。

本文件钉住四组机械判据：

1. 双会话：仅 owner 收到自动唤醒；子会话在归属闸/夜航差分/夜航 gate/政策注入全部零触发；
2. pod cwd：即使被历史遗留记成 owner，也不认领/不监听；夜航同样不生效；
3. **事件不丢**（twin 点名验收件）：owner 缺席 + 无主替身在岗时，替身**不记消费账、
   不推进水位**；owner 回归挂载后该事件以「迟到唤醒」补上；
4. 夜航 gate/政策对非 owner（子会话/pod cwd）零输出。
"""

from __future__ import annotations

import io
import json

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import events, night, watch

POD_CWD = str(PROJECT_ROOT / "orchestration" / "pods" / "devops")
ROOT_CWD = str(PROJECT_ROOT)


def _payload(sid: str, *, cwd: str = ROOT_CWD, event: str = "Stop") -> str:
    return json.dumps(
        {"session_id": sid, "hook_event_name": event, "cwd": cwd},
        ensure_ascii=False,
    )


def _seed_owner(sid: str = "owner-A") -> None:
    watch.save_owner(watch._new_owner(sid))


# ---------------------------------------------------------------------------
# 双会话：仅 owner 收自动唤醒
# ---------------------------------------------------------------------------
def test_dual_session_only_owner_is_woken(iso_state):
    _seed_owner("owner-A")
    code_a, why_a = watch.hook_decision(_payload("owner-A"))
    code_b, why_b = watch.hook_decision(_payload("sub-B"))
    assert code_a == 0 and why_a.startswith("owner")
    assert code_b == 3 and why_b.startswith("deny")
    assert watch.load_owner()["session_id"] == "owner-A", "子会话不得改写真源"


def test_dual_session_night_diff_only_for_owner(iso_state, monkeypatch, capsys):
    """夜航差分（watch-arm 的 rewake 通道）：子会话零输出（= 零 rewake）。"""
    _seed_owner("owner-A")
    # 制造一次开关跳变：先落基线 off，再把开关打开
    night.record_baseline("owner-A", data={"night": {"enabled": False}})
    night.record_baseline("sub-B", data={"night": {"enabled": False}})
    on = {"night": {"enabled": True}}

    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("sub-B")))
    assert (
        night.main(["--diff", "--session", "sub-B", "--settings", _settings(on)]) == 0
    )
    assert capsys.readouterr().out == "", "子会话不得拿到夜航差分（否则 rewake 误投）"

    night.record_baseline("owner-A", data={"night": {"enabled": False}})
    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("owner-A")))
    assert (
        night.main(["--diff", "--session", "owner-A", "--settings", _settings(on)]) == 0
    )
    assert "夜航" in capsys.readouterr().out, "owner 的差分照常输出"


def _settings(data: dict) -> str:
    """把 ``{"night": {"enabled": bool}}`` 落成临时 TOML 路径。"""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp(prefix="wake-bind-")) / "settings.toml"
    enabled = "true" if data["night"]["enabled"] else "false"
    path.write_text(f"[night]\nenabled = {enabled}\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 夜航 gate / 政策注入：主会话绑定
# ---------------------------------------------------------------------------
def test_night_gate_and_policy_skip_non_owner(iso_state):
    _seed_owner("owner-A")
    on = {"night": {"enabled": True}}
    sub = {"session_id": "sub-B", "cwd": ROOT_CWD}
    assert night.gate_deny_message(on, payload=sub) is None, "子会话不受夜航拦截"
    assert night.policy_hook_output("SessionStart", data=on, payload=sub) is None

    owner = {"session_id": "owner-A", "cwd": ROOT_CWD}
    assert night.gate_deny_message(on, payload=owner), "owner 仍受夜航 gate"
    assert night.policy_hook_output("SessionStart", data=on, payload=owner)


def test_night_gate_and_policy_skip_pod_cwd_even_if_owner(iso_state):
    """pod 直聊会话即使（历史遗留）被记成 owner，夜航也不生效。"""
    _seed_owner("owner-A")
    on = {"night": {"enabled": True}}
    pod = {"session_id": "owner-A", "cwd": POD_CWD}
    assert night.gate_deny_message(on, payload=pod) is None
    assert night.policy_hook_output("SessionStart", data=on, payload=pod) is None


def test_night_main_gate_and_injection_respect_binding(iso_state, monkeypatch, capsys):
    _seed_owner("owner-A")
    settings_path = _settings({"night": {"enabled": True}})

    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("sub-B")))
    assert night.main(["--gate", "--settings", settings_path]) == 0, (
        "子会话放行、零 stderr"
    )
    assert capsys.readouterr().err == ""
    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("owner-A")))
    assert night.main(["--gate", "--settings", settings_path]) == 2, "owner 被 gate"
    assert "twin" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("sub-B")))
    assert night.main(["--settings", settings_path]) == 0
    assert capsys.readouterr().out == "", "子会话不注入夜航政策"
    monkeypatch.setattr("sys.stdin", io.StringIO(_payload("owner-A")))
    assert night.main(["--settings", settings_path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["additionalContext"], "owner 照常注入政策"


# ---------------------------------------------------------------------------
# 事件不丢（twin 点名验收件）：替身不吞账 → owner 迟到唤醒
# ---------------------------------------------------------------------------
def test_substitute_does_not_consume_and_owner_gets_late_wake(iso_state, monkeypatch):
    _seed_owner("owner-A")
    # 首任挂载（无 owner 环境 = 无主替身）建立持久水位
    monkeypatch.delenv(watch.WATCH_OWNER_ENV, raising=False)
    watch.arm()
    base_before = dict(watch.load_state()["base"])

    events.emit(events.QUEUE_EMPTY, "run-1", "devops 队列消化完：处理 1 项", "drain")

    # 看门狗重臂的无主替身：命中事件但**不得**消费
    sub = watch.arm()
    assert sub["armed_by"] == ""
    sub_events, _ = watch.detect(sub["base"], watch.snapshot(), queue_watch=False)
    assert sub_events and sub_events[0]["event"] == events.QUEUE_EMPTY
    out_sub: list[str] = []
    assert watch._emit(sub, sub_events, out_sub.append) is True, "替身退出（不挂死）"
    assert any("非 owner 侧" in line for line in out_sub), out_sub

    after_sub = watch.load_state()
    assert not after_sub.get("consumed"), "无主替身不得记消费账"
    assert after_sub["base"] == base_before, "无主替身不得推进持久水位"

    # owner 回归挂载：继承旧水位 → 同一事件作为「迟到唤醒」送达
    monkeypatch.setenv(watch.WATCH_OWNER_ENV, "owner-A")
    owner_state = watch.arm()
    assert owner_state["catchup"] is True
    late, _ = watch.detect(owner_state["base"], watch.snapshot(), queue_watch=False)
    assert [e["event"] for e in late] == [events.QUEUE_EMPTY]
    out_owner: list[str] = []
    assert watch._emit(owner_state, late, out_owner.append) is True
    assert any("QUEUE-EMPTY" in line for line in out_owner)
    assert any("迟到" in line for line in out_owner), "迟到唤醒必须标注来源"
    # 单一消费水位：owner 侧消费已入账（_mark_late 会就地改写 detail，需从账本键判断）
    consumed = watch.load_state().get("consumed") or {}
    assert any(k.startswith(events.QUEUE_EMPTY + "|") for k in consumed), (
        "owner 消费后单一水位生效（防双唤醒）"
    )


def test_replaced_owner_watch_cannot_consume(iso_state, monkeypatch):
    """被人工迁移取代的旧 owner watch（armed_by ≠ 当前 owner）同样不得消费。"""
    _seed_owner("owner-B")
    monkeypatch.setenv(watch.WATCH_OWNER_ENV, "owner-A")
    watch.arm()
    events.emit(events.JOB_DONE, "figure-1", "收尾", "advance")
    st = watch.load_state()
    evs, _ = watch.detect(st["base"], watch.snapshot(), queue_watch=False)
    assert evs
    assert watch._emit(st, evs, [].append) is True
    assert not watch.load_state().get("consumed"), "旧实例不得抢账"
