"""主会话程序化识别＋自动补位（wake-adopt-20261018 s1）。

用户 2026-10-18 直接裁决（高于 twin A）：候选集＝**存活 ∧ cwd==项目根**；主会话＝
**创建更早者**；owner 存活→保持/恢复；owner 缺席/死亡→最早存活者自动补位；晚创建者
在早者存活期间永不夺主；pod cwd 永不入候选。

本文件钉：
1. 登记表 + 创建时序锚定（OS create_time / first_seen 回退 / session_id 平局）与清理；
2. 存活性：**PID 优先于心跳**——心跳过期而 PID 存活 = 存活（长回合保护）；
3. 两会话并存（真实存活子进程）：早者恒 owner、晚者恒 deny；早者死亡→晚者补位；
   owner 记录跨重启恢复；SessionEnd 自动补位；pod cwd 永不入候选；冷启动单会话接管；
4. 归属判定走 `watch.hook_decision`（单一实现）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import live_sessions, watch

ROOT = str(PROJECT_ROOT)
POD = str(PROJECT_ROOT / "orchestration" / "pods" / "devops")


@pytest.fixture
def spawn():
    """起一个真实长驻子进程（PID 判活用）；用例结束统一杀掉。"""
    made: list[subprocess.Popen] = []

    def _spawn() -> subprocess.Popen:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        made.append(proc)
        return proc

    yield _spawn
    for proc in made:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except OSError:
            pass


def _seed(sid: str, *, pid: int | None, created: float, cwd: str = ROOT) -> dict:
    sessions = live_sessions.load_registry()
    sessions[sid] = {
        "session_id": sid,
        "cwd": cwd,
        "pid": pid,
        "created_at": created,
        "first_seen": live_sessions._now_iso(created),
        "last_seen": live_sessions._now_iso(),
    }
    live_sessions.save_registry(sessions)
    return sessions[sid]


def _payload(sid: str, *, cwd: str = ROOT, event: str = "Stop") -> str:
    import json

    return json.dumps(
        {"session_id": sid, "hook_event_name": event, "cwd": cwd},
        ensure_ascii=False,
    )


def _kill(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# 登记表与创建时序
# ---------------------------------------------------------------------------
def test_registry_roundtrip_and_creation_order(iso_state):
    _seed("s-late", pid=None, created=200)
    _seed("s-early", pid=None, created=100)
    # 无 pid 的条目按 last_seen 心跳判活（刚写入 → 新鲜）
    assert [e["session_id"] for e in live_sessions.ordered_candidates()] == [
        "s-early",
        "s-late",
    ]
    assert live_sessions.load_registry()["s-early"]["created_at"] == 100


def test_creation_tie_breaks_by_session_id(iso_state):
    _seed("s-b", pid=None, created=100)
    _seed("s-a", pid=None, created=100)
    assert [e["session_id"] for e in live_sessions.ordered_candidates()] == [
        "s-a",
        "s-b",
    ]


def test_first_seen_fallback_when_no_os_create_time(iso_state):
    _seed("s-x", pid=None, created=150)
    sessions = live_sessions.load_registry()
    sessions["s-x"].pop("created_at")
    live_sessions.save_registry(sessions)
    cands = live_sessions.ordered_candidates()
    assert [e["session_id"] for e in cands] == ["s-x"]
    assert live_sessions.creation_epoch(cands[0]) < float("inf"), "回退 first_seen"


def test_pid_alive_beats_stale_heartbeat(iso_state, spawn):
    """长回合保护：心跳停摆但 PID 活着 = 存活。"""
    proc = spawn()
    _seed("s-live", pid=proc.pid, created=100)
    sessions = live_sessions.load_registry()
    sessions["s-live"]["last_seen"] = live_sessions._now_iso(time.time() - 7200)
    live_sessions.save_registry(sessions)
    entry = live_sessions.load_registry()["s-live"]
    assert live_sessions.session_alive(entry) is True, "PID 存活优先于心跳"

    entry["pid"] = None  # 仅当 PID 缺失才看心跳
    assert live_sessions.session_alive(entry) is False


def test_dead_pid_is_pruned(iso_state, spawn):
    proc = spawn()
    _seed("s-dead", pid=proc.pid, created=100)
    _kill(proc)
    assert live_sessions.prune_dead() == ["s-dead"]
    assert live_sessions.load_registry() == {}


def test_process_create_time_and_anchor(iso_state, spawn):
    proc = spawn()
    created = live_sessions.process_create_time(proc.pid)
    assert created is not None and abs(created - time.time()) < 300

    cli = lambda name: "qoderclicn" in name.lower()  # noqa: E731
    table = {
        os.getpid(): {"ppid": 900, "name": "python.exe", "create_time": None},
        900: {"ppid": 800, "name": "bash.exe", "create_time": None},
        800: {"ppid": 700, "name": "qoderclicn.exe", "create_time": None},
    }
    assert live_sessions.anchor_session_pid(table=table, is_cli=cli) == 800
    assert live_sessions.anchor_session_pid(table=table) == 800


def test_pod_cwd_never_a_candidate(iso_state):
    _seed("s-pod", pid=None, created=1, cwd=POD)
    _seed("s-root", pid=None, created=2, cwd=ROOT)
    assert [e["session_id"] for e in live_sessions.ordered_candidates()] == ["s-root"]


# ---------------------------------------------------------------------------
# 归属判定（真实存活进程）
# ---------------------------------------------------------------------------
def test_earlier_session_owns_and_later_denies(iso_state, spawn):
    early, late = spawn(), spawn()
    _seed("s-early", pid=early.pid, created=100)
    _seed("s-late", pid=late.pid, created=200)

    code, reason = watch.hook_decision(_payload("s-early"))
    assert code == 0 and reason.startswith("adopt"), reason
    before = watch.load_owner()
    assert before["session_id"] == "s-early"

    code, reason = watch.hook_decision(_payload("s-late"))
    assert code == 3 and reason.startswith("deny"), reason
    assert watch.load_owner() == before, "晚创建者不得改写真源"


def test_long_turn_owner_keeps_owner_despite_stale_heartbeat(iso_state, spawn):
    """早者心跳过期（长回合）而晚者心跳新鲜：早者 PID 存活 → 仍保 owner。"""
    early, late = spawn(), spawn()
    _seed("s-early", pid=early.pid, created=100)
    _seed("s-late", pid=late.pid, created=200)
    assert watch.hook_decision(_payload("s-early"))[0] == 0

    owner = watch.load_owner()
    owner["last_turn_at"] = "2000-01-01T00:00:00+08:00"
    watch.save_owner(owner)
    sessions = live_sessions.load_registry()
    sessions["s-early"]["last_seen"] = "2000-01-01T00:00:00+08:00"
    sessions["s-late"]["last_seen"] = live_sessions._now_iso()
    live_sessions.save_registry(sessions)

    code, reason = watch.hook_decision(_payload("s-late"))
    assert code == 3 and reason.startswith("deny"), reason
    assert watch.load_owner()["session_id"] == "s-early"


def test_dead_owner_promotes_earliest_survivor(iso_state, spawn):
    early, late = spawn(), spawn()
    _seed("s-early", pid=early.pid, created=100)
    _seed("s-late", pid=late.pid, created=200)
    assert watch.hook_decision(_payload("s-early"))[0] == 0
    _kill(early)

    code, reason = watch.hook_decision(_payload("s-late"))
    assert code == 0 and reason.startswith("adopt"), reason
    assert watch.load_owner()["session_id"] == "s-late"


def test_owner_record_restored_after_watchdog_restart(iso_state, spawn):
    """记录指向者仍存活 → 保持/恢复（模拟看门狗/监听被杀重启后的下一次钩子）。"""
    early, late = spawn(), spawn()
    _seed("s-early", pid=early.pid, created=100)
    _seed("s-late", pid=late.pid, created=200)
    assert watch.hook_decision(_payload("s-early"))[0] == 0
    before = watch.load_owner()

    # 「重启」：无任何内存状态，只有磁盘记录与登记表
    code, reason = watch.hook_decision(_payload("s-late"))
    assert code == 3
    code, reason = watch.hook_decision(_payload("s-early"))
    assert code == 0 and reason.startswith("owner"), reason
    assert watch.load_owner()["session_id"] == before["session_id"] == "s-early"


def test_cold_start_single_session_adopts(iso_state, spawn):
    proc = spawn()
    _seed("s-only", pid=proc.pid, created=100)
    code, reason = watch.hook_decision(_payload("s-only"))
    assert code == 0 and reason.startswith("adopt"), reason
    assert watch.load_owner()["session_id"] == "s-only"


def test_session_end_unregisters_and_promotes_survivor(iso_state, spawn):
    early, late = spawn(), spawn()
    _seed("s-early", pid=early.pid, created=100)
    _seed("s-late", pid=late.pid, created=200)
    assert watch.hook_decision(_payload("s-early"))[0] == 0

    code, reason = watch.hook_decision(_payload("s-early", event="SessionEnd"))
    assert code == 4 and "自动补位" in reason and "s-late" in reason, reason
    assert "s-early" not in live_sessions.load_registry(), "SessionEnd 注销登记"
    assert watch.load_owner()["session_id"] == "s-late", "墓碑不阻止自动补位"

    code, reason = watch.hook_decision(_payload("s-late"))
    assert code == 0 and reason.startswith("owner"), reason


def test_pod_cwd_never_adopts_even_if_earliest(iso_state, spawn):
    pod, root = spawn(), spawn()
    _seed("s-pod", pid=pod.pid, created=1, cwd=POD)
    _seed("s-root", pid=root.pid, created=2)
    code, reason = watch.hook_decision(_payload("s-pod", cwd=POD))
    assert code == 3 and "不入候选" in reason, reason
    assert watch.load_owner() == {}, "pod 会话永不升主"

    code, reason = watch.hook_decision(_payload("s-root"))
    assert code == 0 and reason.startswith("adopt")
    assert watch.load_owner()["session_id"] == "s-root"
