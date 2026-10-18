"""唤醒权自动归属（wake-owner，用户设计定稿 2026-10-14 晚）：session_id 识别 + 零人工认领。

事故现场：watch 钩子的 token 竞争是「最新在听者胜」（v2 修会话重启丢唤醒的规则）——
用户主会话工作时另开的**次要会话**一有回合就抢走唤醒权，drain/事件的 exit-2 被投到
次要会话（污染其上下文，正是用户要避免的）；凌晨「被取代却无接任」的谜之日志疑为同源。

本文件钉任务书的机械验收语义（全走单一实现 :func:`watch.hook_decision`——stdin
载荷即钩子要喂的载荷）：
① 首会话自动认领（仅无 owner 记录时）；② 异 id 不抢；③ **自动转让已取消**（stale 不
接管、SessionEnd 写墓碑不升主，迁移只走人工 --force）；④ pod cwd 会话永不认领/监听；
⑤ 载荷解析失败 fail-open。另钉：owner 缺席事件不丢（迟到唤醒续接；**不存在「转投
其他会话」语义**）、消费权只认 owner 侧（2026-10-18 wake-bind s2）与钩子脚本的分支
形状（文本级，防接线回退）。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import drain, events, orch, watch, watchdog

HOOK = PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh"


def _payload(sid: str, event: str = "Stop") -> str:
    """钩子 stdin 载荷的最小真实形（hook-config 文档：自带 session_id / hook_event_name）。"""
    return json.dumps(
        {
            "session_id": sid,
            "hook_event_name": event,
            "transcript_path": "/tmp/t.jsonl",
        },
        ensure_ascii=False,
    )


def _iso(seconds_ago: float) -> str:
    return (
        datetime.now().astimezone() - timedelta(seconds=abs(seconds_ago))
    ).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ① 首会话自动认领（零人工）+ owner==我 续心跳
# ---------------------------------------------------------------------------
def test_first_session_auto_claims(iso_state):
    assert watch.load_owner() == {}, "起手必须无 owner"

    code, reason = watch.hook_decision(_payload("sess-A"))
    assert code == 0 and reason.startswith("claim"), "无 owner → 本会话自动认领"
    owner = watch.load_owner()
    assert owner["session_id"] == "sess-A"
    assert owner["claimed_at"] and owner["last_turn_at"], (
        "真源三字段：session_id/claimed_at/last_turn_at"
    )

    # 续心跳：last_turn_at 推进、claimed_at 不动
    owner["last_turn_at"] = _iso(3600)
    watch.save_owner(owner)
    code, reason = watch.hook_decision(_payload("sess-A"))
    assert code == 0 and reason.startswith("owner")
    fresh = watch.load_owner()
    assert fresh["claimed_at"] == owner["claimed_at"], "认领时刻不被心跳覆盖"
    assert fresh["last_turn_at"] != owner["last_turn_at"], "心跳必须刷新"
    assert fresh["session_id"] == "sess-A"


# ---------------------------------------------------------------------------
# ② 异 id 不抢（次要会话自动排除；真源一字不动）
# ---------------------------------------------------------------------------
def test_other_session_never_preempts(iso_state):
    watch.hook_decision(_payload("sess-A"))
    before = watch.load_owner()

    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 3 and reason.startswith("deny"), "心跳新鲜的非 owner 必须被拒"
    assert watch.load_owner() == before, "非 owner 不得改写真源（永不抢占）"

    # 再轮到 owner：照常续心跳（被拒的 B 不留任何痕迹）
    code, _ = watch.hook_decision(_payload("sess-A"))
    assert code == 0
    assert watch.load_owner()["session_id"] == "sess-A"


def test_hook_script_gates_by_owner_and_fails_open():
    """钩子面机械判据：不再丢载荷；deny 分支唯一出口 exit 0、绝不 exit 2。"""
    text = HOOK.read_text(encoding="utf-8")
    assert "PAYLOAD=$(cat)" in text and "cat >/dev/null" not in text, "载荷不再被丢弃"
    assert '"session_id"' in text and "hook_event_name" in text, "两键都要解析"
    assert "wake-claim --check" in text, "归属判定必须走单一实现"
    assert "fail-open 按旧 token 语义监听" in text, "检查坏了不得塌唤醒链"
    assert "SessionEnd" in text and "PYSCI_WAKE_OWNER" in text, (
        "释放分支 + owner watch 身份"
    )
    assert "unset PYSCI_WAKE_OWNER" in text, (
        "无载荷路径必须剥掉继承的 owner 身份——否则看门狗重臂的替身 watch 会盖上 owner 的 "
        "armed_by、冒充 owner 侧消费事件（2026-10-14 17:53 实测的现场；2026-10-18 起还会吞掉"
        "应留给 owner 的迟到唤醒）"
    )
    branch = text.split('case "$wcr" in', 1)[1].split("esac", 1)[0]
    assert "3)" in branch and "exit 0" in branch
    assert "exit 2" not in branch, "非 owner 分支绝不 exit 2（不唤醒）"


# ---------------------------------------------------------------------------
# ③ stale 不再自动接管（twin 裁决 A，2026-10-18）：迁移只走人工 wake-claim --force
# ---------------------------------------------------------------------------
def test_stale_owner_is_not_auto_taken_over(iso_state):
    watch.hook_decision(_payload("sess-A"))
    owner = watch.load_owner()
    owner["last_turn_at"] = _iso(watch.OWNER_STALE_S + 600)
    watch.save_owner(owner)
    before = watch.load_owner()

    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 3 and reason.startswith("deny"), (
        f"stale owner 也不得被自动接管（只走人工）：{reason}"
    )
    assert "自动转让已取消" in reason and "wake-claim --force --session-id" in reason
    assert watch.load_owner() == before, "拒绝分支不得改写真源"

    # 人工迁移是唯一通道
    watch.wake_claim_force("sess-B")
    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 0 and reason.startswith("owner")


@pytest.mark.parametrize(
    "hb,armed_by",
    [
        ("fresh", None),
        ("stale", "sess-A"),
        ("stale", ""),
        ("stale", None),
    ],
    ids=["fresh-dead", "stale-owner-live", "stale-substitute", "stale-dead"],
)
def test_no_auto_takeover_under_any_staleness_combination(iso_state, hb, armed_by):
    """任何「心跳 × watch 在挂形态」组合都不得触发接管（旧双条件判据已整体退役）。"""
    hb_fresh = datetime.now().astimezone().isoformat(timespec="seconds")
    hb_stale = _iso(watch.OWNER_STALE_S + 600)
    watch.save_owner(
        {
            "session_id": "sess-A",
            "claimed_at": _iso(7200),
            "last_turn_at": hb_fresh if hb == "fresh" else hb_stale,
        }
    )
    if armed_by is not None:
        watch.save_state(
            {
                "version": 1,
                "interval_s": 60,
                "heartbeat_at": hb_fresh,
                "armed_by": armed_by,
            }
        )
    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 3 and reason.startswith("deny"), reason
    assert watch.load_owner()["session_id"] == "sess-A"


# ---------------------------------------------------------------------------
# ④ SessionEnd 释放：写墓碑、不自动升主；迁移只走人工 --force
# ---------------------------------------------------------------------------
def test_session_end_writes_tombstone_and_no_auto_promotion(iso_state):
    watch.hook_decision(_payload("sess-A"))

    # 非 owner 会话结束：不释放
    code, reason = watch.hook_decision(_payload("sess-B", event="SessionEnd"))
    assert code == 4 and reason.startswith("release")
    assert watch.load_owner().get("session_id") == "sess-A", (
        "非 owner 的 SessionEnd 不得动真源"
    )

    # owner 结束 → 墓碑（不是删除）：下一个会话不得自动升主
    code, reason = watch.hook_decision(_payload("sess-A", event="SessionEnd"))
    assert code == 4 and "释放墓碑" in reason and "不会自动升主" in reason
    owner = watch.load_owner()
    assert owner.get("session_id") == "" and owner.get("released_at"), owner

    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 3 and reason.startswith("deny"), (
        "SessionEnd 后任意会话自动升主必须取消"
    )
    assert "wake-claim --force --session-id" in reason

    # 人工迁移（唯一通道）后 B 成为 owner
    watch.wake_claim_force("sess-B")
    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 0 and reason.startswith("owner")


def test_manual_force_release_allows_first_session_claim(iso_state):
    """保留的认领语义：**无 owner 记录**（冷启动/人工 --force 释放）时首会话自动认领。"""
    watch.wake_claim_force("sess-A")
    assert watch.load_owner()["session_id"] == "sess-A"

    watch.wake_claim_force("")  # 人工释放 = 删除记录（不是墓碑）
    assert watch.load_owner() == {}
    code, reason = watch.hook_decision(_payload("sess-B"))
    assert code == 0 and reason.startswith("claim")


# ---------------------------------------------------------------------------
# ④b cwd 判据：pod 直聊会话永不认领/接管/监听
# ---------------------------------------------------------------------------
def _payload_cwd(sid: str, cwd: str, event: str = "Stop") -> str:
    return json.dumps(
        {"session_id": sid, "hook_event_name": event, "cwd": cwd},
        ensure_ascii=False,
    )


def test_pod_cwd_session_never_claims(iso_state):
    pod = str(PROJECT_ROOT / "orchestration" / "pods" / "devops")
    code, reason = watch.hook_decision(_payload_cwd("pod-sess", pod))
    assert code == 3 and "cwd 非项目根" in reason and "pod 直聊" in reason
    assert watch.load_owner() == {}, "pod 会话不得认领"


def test_pod_cwd_session_denied_even_if_recorded_owner(iso_state):
    """历史遗留把 pod 会话记成 owner 时，也不得监听/续心跳（一律 deny）。"""
    watch.save_owner(watch._new_owner("pod-sess"))
    before = watch.load_owner()
    pod = str(PROJECT_ROOT / "orchestration" / "pods" / "devops")
    code, reason = watch.hook_decision(_payload_cwd("pod-sess", pod))
    assert code == 3 and "cwd 非项目根" in reason
    assert watch.load_owner() == before


def test_project_root_cwd_claims_and_heartbeats(iso_state):
    root = str(PROJECT_ROOT)
    code, reason = watch.hook_decision(_payload_cwd("main-sess", root))
    assert code == 0 and reason.startswith("claim")
    code, reason = watch.hook_decision(_payload_cwd("main-sess", root))
    assert code == 0 and reason.startswith("owner")


def test_cwd_falls_back_to_hook_entry_env(iso_state, monkeypatch):
    """载荷无 cwd 时用钩子脚本 cd 前导出的 PYSCI_HOOK_CWD；缺省再退 QODER_PROJECT_DIR。"""
    pod = str(PROJECT_ROOT / "orchestration" / "pods" / "devops")
    monkeypatch.setenv("PYSCI_HOOK_CWD", pod)
    raw = json.dumps({"session_id": "pod-sess", "hook_event_name": "Stop"})
    code, reason = watch.hook_decision(raw)
    assert code == 3 and "cwd 非项目根" in reason

    monkeypatch.setenv("PYSCI_HOOK_CWD", str(PROJECT_ROOT))
    code, reason = watch.hook_decision(raw)
    assert code == 0 and reason.startswith("claim")


# ---------------------------------------------------------------------------
# ⑤ 载荷解析失败 fail-open（走旧 token 语义并留痕；不写真源）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "{broken json",
        '{"hook_event_name":"Stop"}',
        '"just-a-string"',
        "[1, 2, 3]",
    ],
    ids=["empty", "blank", "invalid-json", "no-session_id", "non-object", "array"],
)
def test_payload_parse_failure_fails_open(iso_state, bad):
    code, reason = watch.hook_decision(bad)
    assert code == 0, "fail-open = 放行（rc 0），走旧 token 语义"
    assert reason.startswith("fail-open"), reason
    assert watch.load_owner() == {}, "解析失败不得写真源（无身份即无归属）"


@pytest.mark.skipif(shutil.which("bash") is None, reason="需 bash（Git Bash）")
def test_hook_script_fails_open_on_garbage_payload(tmp_path):
    """bash 级端到端：垃圾载荷 → 解析留痕 → 无 .venv 环境缺失退出，全程 rc 0。"""
    hooks_dir = tmp_path / ".qoder" / "hooks"
    hooks_dir.mkdir(parents=True)
    env = {**os.environ, "QODER_PROJECT_DIR": str(tmp_path)}

    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=b"{broken json",
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, "钩子任何路径都必须 exit 0"
    log = (hooks_dir / "watch-arm.log").read_text(encoding="utf-8")
    assert "载荷解析失败/无 session_id → fail-open 走旧 token 语义" in log
    assert "环境缺失" in log, "假 ROOT 无 .venv → 环境检查即退出（零副作用）"

    # 对照：合法载荷能被 sed 解析出身份（同一 bash 路径）
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(
            {"session_id": "sess-E2E", "hook_event_name": "Stop"}
        ).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0
    log = (hooks_dir / "watch-arm.log").read_text(encoding="utf-8")
    assert "载荷解析 ok（session_id=sess-E2E event=Stop）" in log


# ---------------------------------------------------------------------------
# owner 缺席事件不丢：迟到唤醒在 owner 下次挂载时接续（a1 语义；无「转投」）
# ---------------------------------------------------------------------------
def test_owner_absent_events_woken_on_owner_next_mount(iso_state):
    watch.hook_decision(_payload("sess-A"))
    assert watch.hook_decision(_payload("sess-B"))[0] == 3, "次要会话被排除、不挂载"

    watch.arm()  # 无人监听的基线（首任）
    events.emit(
        events.QUEUE_EMPTY, "run-010000", "devops 队列消化完：处理 1 项", "drain"
    )

    # owner 下次挂载 → 继承持久水位 → 立即补唤醒（事件不因「不转投」而丢）
    st = watch.arm()
    assert st["catchup"] is True, "继承持久水位 → 本任是迟到接续"
    evs, _verdict = watch.detect(st["base"], watch.snapshot())
    assert [e["event"] for e in evs] == [events.QUEUE_EMPTY], "聋期事件必须被读出"
    watch._mark_late(evs, st)
    assert evs[0].get("late") is True and "迟到" in evs[0]["detail"]

    echoed: list[str] = []
    assert watch._emit(st, evs, echoed.append) is True, "owner 侧送达唤醒"
    cur = watch.snapshot()
    st["base"] = watch.advance_base(st["base"], cur, set())
    watch.save_state(st)
    st2 = watch.arm()
    assert watch.detect(st2["base"], watch.snapshot())[0] == [], "水位单调，不重复消费"


# ---------------------------------------------------------------------------
# CLI 面（降级后的 wake-claim：status 缺省 / --force 应急 / --check 钩子口）
# ---------------------------------------------------------------------------
def test_cli_status_and_force(iso_state, monkeypatch, capsys):
    assert orch.main(["wake-claim", "status"]) == 0
    assert "自动认领" in capsys.readouterr().out, "无 owner → status 提示零人工语义"

    watch.hook_decision(_payload("sess-A"))
    assert orch.main(["wake-claim"]) == 0, "缺省 action = status"
    out = capsys.readouterr().out
    assert "owner = sess-A" in out and "--force" in out

    assert orch.main(["wake-claim", "--force"]) == 0, "缺省 force = 释放"
    assert "强制释放" in capsys.readouterr().out
    assert watch.load_owner() == {}

    assert orch.main(["wake-claim", "--force", "--session-id", "sess-Z"]) == 0
    assert "强制迁移" in capsys.readouterr().out
    assert watch.load_owner()["session_id"] == "sess-Z"

    # --check（钩子口）：同身份放行、异身份 rc 3
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"session_id": "sess-Z", "hook_event_name": "Stop"})),
    )
    assert orch.main(["wake-claim", "--check"]) == 0
    assert "owner" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"session_id": "sess-Q", "hook_event_name": "Stop"})),
    )
    assert orch.main(["wake-claim", "--check"]) == 3, "非 owner → rc 3（钩子据此不抢）"


def test_watchdog_rearm_strips_inherited_owner_env(iso_state, monkeypatch):
    """看门狗重臂剥掉继承的 owner 身份：替身 watch 不得盖 owner 的 armed_by（挡接管）。"""
    monkeypatch.setenv(
        watch.WATCH_OWNER_ENV, "sess-X"
    )  # 模拟 owner 钩子 spawn 的看门狗
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=iso_state / "watchdog.lock",
            runs=iso_state / "watchdog-runs",
            cancel=iso_state / "watchdog.cancel",
        ),
    )
    captured: dict[str, dict] = {}

    def fake_spawn(argv, *, env, stdout):
        captured["env"] = dict(env)

    monkeypatch.setattr(watchdog, "_spawn_detached", fake_spawn)
    assert watchdog._rearm(iso_state / "wd.log") is True
    assert captured["env"], "重臂必须派生进程"
    assert watch.WATCH_OWNER_ENV not in captured["env"], (
        "重臂实例无主：env 里残留的 owner 身份会让替身 watch 冒充 owner 的 watch"
    )
