"""跨组件 e2e 防回归（autonomy-fix a3）：真实联动链，不是单测拼凑。

四条场景对应 2026-10-14 的四个真实故障面：
① 聋期丢唤醒（02:30 现场）：无监听期产生的事件 → 下次挂载 ≤1 轮询周期迟到唤醒 + `late`
   标注 + 水位单调（重复挂载不重复消费）；
② watcher 被杀 → 心跳 stale（≤300s）→ 看门狗重臂且留痕；
③ **inspector 自调度链**：watchdog tick → `self_patrol_tick` → 队列 → worker。
   ⚠ 用户指令（2026-10-14）：**inspector 实测移夜间**——全局开关=off 保持不动，e2e 只在
   **内存里**（monkeypatch）模拟开启，**不真实拉起巡检**；
④ 队列超时告警链：入队 30min 未处理 → 事件层 → TUI 读数可见。

另附文档/技能与实现的一致性测试（README §4.5 三级防线、§6 自调度与超时、SKILL 索引）。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT
from pysci.skills.orchestration.tools import (
    drain,
    events,
    inspectd,
    settings,
    watch,
    watchdog,
    workq,
)
from pysci.skills.orchestration.tools.tui.modules import modules

_INSPECTOR_TOML = PROJECT_ROOT / "data" / "用户的inbox" / "pysci_settings.toml"
#: 真实巡检队列（e2e 只许读、不许写；insp-20261022-141842）。
REAL_QUEUE = ORCH_STATE_ROOT / "inspector" / "queue.json"


def _iso(epoch: float) -> str:
    from datetime import datetime

    return (
        datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    )


class _Clock:
    """假时钟：`now()` 给时刻，`sleep()` 只推进不真等（e2e 免等）。"""

    def __init__(self, t0: float | None = None) -> None:
        self.t = float(t0 if t0 is not None else time.time())

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def _state_paths(monkeypatch, tmp_path):
    root = tmp_path / "state"
    root.mkdir(exist_ok=True)
    insp = tmp_path / "inspector"
    monkeypatch.setattr(watch, "WATCH_PATH", root / "watch.json")
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    monkeypatch.setattr(events, "SEEN_PATH", root / "events-seen.json")
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=root / "watchdog.lock",
            runs=root / "watchdog-runs",
            cancel=root / "watchdog.cancel",
        ),
    )
    monkeypatch.setattr(inspectd, "STATE_DIR", insp)
    monkeypatch.setattr(inspectd, "FINDINGS_PATH", insp / "findings.jsonl")
    monkeypatch.setattr(inspectd, "THROTTLE_PATH", insp / "throttle.json")
    monkeypatch.setattr(inspectd, "QUEUE_PATH", insp / "queue.json")
    monkeypatch.setattr(inspectd, "EVIDENCE_DIR", insp / "evidence")
    monkeypatch.setattr(inspectd, "ENABLE_PATH", root / "inspector.enabled")
    monkeypatch.setattr(
        inspectd,
        "INSPECT_PATHS",
        drain.WorkerPaths(
            lock=root / "inspector.lock",
            runs=root / "inspector-runs",
            cancel=root / "inspector.cancel",
        ),
    )
    monkeypatch.setattr(events, "EVENTS_LOG", root / "events.jsonl")
    # 存活会话登记表同归 tmp（wake-adopt-20261018 s1）
    from pysci.skills.orchestration.tools import live_sessions as _live

    monkeypatch.setattr(_live, "SESSIONS_PATH", root / "sessions-live.json")
    return root


# ---------------------------------------------------------------------------
# ① 聋期丢唤醒 → 迟到唤醒（走 run_watch 真链路：arm→detect→_emit→echo）
# ---------------------------------------------------------------------------
def test_deaf_period_event_fires_on_next_mount(tmp_path, monkeypatch):
    root = _state_paths(monkeypatch, tmp_path)
    # owner 侧挂载身份（2026-10-18：只有 owner 自己的 watch 可消费事件）
    monkeypatch.setattr(watch, "WAKE_OWNER_PATH", root / "wake-owner.json")
    watch.save_owner(watch._new_owner("test-owner"))
    monkeypatch.setenv(watch.WATCH_OWNER_ENV, "test-owner")
    watch.arm()  # 首次挂载（基线=当下快照）
    # 聋期：watcher 已死，无人监听
    events.emit(
        events.QUEUE_EMPTY, "run-022113", "devops 队列消化完：处理 3 项", "drain"
    )

    clock = _Clock()
    echoed: list[str] = []
    watch.run_watch(
        timeout_s=10,
        interval_s=5,
        sleep=clock.sleep,
        monotonic=clock.now,
        echo=echoed.append,
        # 本例只验**纯差分链路**（聋期迟到唤醒 + 水位单调）：关掉队列感知巡更通道，
        # 免得运行期恰好存在的 backlog/队列存量挤进断言（判据得自己隔离自己的输入）。
        # 巡更通道的行为由 tests/skills/orchestration/test_queue_watch.py 覆盖。
        queue_watch=False,
    )

    assert echoed, "新挂载的首轮轮询必须命中聋期事件"
    assert echoed[0].startswith("QUEUE-EMPTY"), echoed[0]
    assert "迟到唤醒" in echoed[0], echoed[0]

    fired = watch.load_state().get("fired") or []
    assert fired and fired[-1].get("late") is True and "迟到" in fired[-1]["detail"]

    # 水位单调：再次挂载不重复消费
    st2 = watch.arm()
    assert watch.detect(st2["base"], watch.snapshot())[0] == []


# ---------------------------------------------------------------------------
# ② watcher 被杀 → 心跳 stale → 看门狗重臂且留痕
# ---------------------------------------------------------------------------
def test_killed_listener_rearms_with_log(tmp_path, monkeypatch):
    root = _state_paths(monkeypatch, tmp_path)
    clock = _Clock()
    monkeypatch.setattr(watchdog, "_rearm", lambda run_log=None: True)  # 不真拉 bash
    # 被杀：心跳停在 400s 前（> 300s 判死阈值）
    watch.save_state({"version": 1, "heartbeat_at": _iso(clock.now() - 400)})

    stops = {"n": 0}

    def fake_sleep(_s):
        stops["n"] += 1
        clock.sleep(60)
        if stops["n"] >= 2:
            monkeypatch.setattr(watchdog, "_consume_stop_request", lambda *a, **k: True)

    run_log = root / "watchdog-runs" / "wd.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    summary = watchdog.run_watchdog(
        run_log, interval_s=60, stale_s=300, sleeps=fake_sleep, now=clock.now
    )

    assert summary["rearms"] == 1, "心跳 stale 必须重臂"
    text = run_log.read_text(encoding="utf-8")
    assert "重臂监听循环" in text and "看门狗退出（已释放锁）" in text


# ---------------------------------------------------------------------------
# ③ inspector 自调度链：watchdog tick → self_patrol_tick → 队列 → worker
# ---------------------------------------------------------------------------
REAL_QUEUE = ORCH_STATE_ROOT / "inspector" / "queue.json"


def _real_queue_bytes() -> bytes:
    return REAL_QUEUE.read_bytes() if REAL_QUEUE.exists() else b""


def _is_test_leftover(it: dict) -> bool:
    """测试期新入队且非人手发起的 patrol 项＝测试遗留（真实开关=off，leader 不会 self）。"""
    return it.get("kind") == "patrol" and it.get("requested_by") == "self"


@pytest.fixture
def no_real_queue_pollution():
    """真实队列零遗留：teardown 兜底摘除测试期新入队的 self-patrol 项。

    现场（insp-20261022-114925-patrol）：巡检链若有任何路径漏到真实队列，开关=off 时
    无人消化 → QUEUE-TIMEOUT 每 30min 唤醒组长。兜底清理与下方不变量用例双保险；只摘
    「测试窗口内新出现且 requested_by=self」的项，不碰他人在飞写入。
    """
    before_ids = {str(it.get("id")) for it in workq.load(REAL_QUEUE)}
    yield
    try:
        items = workq.load(REAL_QUEUE)
        leftover = [
            it
            for it in items
            if str(it.get("id")) not in before_ids and _is_test_leftover(it)
        ]
        if leftover:
            workq.save(REAL_QUEUE, [it for it in items if it not in leftover])
            print(
                "[teardown] 摘除测试期漏入真实队列的巡检项："
                + ", ".join(str(it.get("id")) for it in leftover)
            )
    except Exception:  # noqa: BLE001 - 兜底清理不得影响测试结论
        pass


def _run_patrol_chain(tmp_path, monkeypatch) -> tuple[list[list[str]], Path]:
    """驱动 watchdog→self_patrol_tick→队列→worker 链（只内存开启）。返回 (spawns, run_log)。"""
    root = _state_paths(monkeypatch, tmp_path)
    # ⚠ 只在内存里模拟开启态（用户指令：真源开关=off，不得改）
    monkeypatch.setattr(inspectd, "enabled", lambda: True)
    monkeypatch.setattr(
        inspectd, "lock_status", lambda paths=None, **kw: {"state": "idle"}
    )
    monkeypatch.setattr(inspectd, "last_run_age_s", lambda done=None: None)  # 无冷却
    spawns: list[list[str]] = []
    monkeypatch.setattr(
        inspectd, "_spawn_detached", lambda argv, **kw: spawns.append(list(argv))
    )
    monkeypatch.setattr(watchdog, "_rearm", lambda *a, **k: True)  # 心跳新鲜，不重臂

    clock = _Clock()
    stops = {"n": 0}

    def fake_sleep(_s):
        stops["n"] += 1
        clock.sleep(1)
        if stops["n"] >= 2:  # 第一个 tick 之后再停，确保至少敲一次
            monkeypatch.setattr(watchdog, "_consume_stop_request", lambda *a, **k: True)

    watch.save_state({"version": 1, "heartbeat_at": _iso(clock.now())})
    run_log = root / "watchdog-runs" / "wd.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    # 一个 tick（interval 短 + 立刻停），链路必须走到 worker
    watchdog.run_watchdog(
        run_log, interval_s=1, stale_s=5, sleeps=fake_sleep, now=clock.now
    )
    return spawns, run_log


def test_inspector_chain_from_watchdog_tick_to_worker(
    tmp_path, monkeypatch, no_real_queue_pollution
):
    before = _INSPECTOR_TOML.read_text(encoding="utf-8")
    spawns, run_log = _run_patrol_chain(tmp_path, monkeypatch)

    assert workq.pending_count(inspectd.QUEUE_PATH) >= 1, (
        "tick → self_patrol_tick → 队列"
    )
    assert spawns, "tick → 队列 → worker（spawn）"
    assert "_inspect-worker" in spawns[0], spawns[0]
    assert "inspector tick 异常" not in run_log.read_text(encoding="utf-8")
    assert _INSPECTOR_TOML.read_text(encoding="utf-8") == before, "不得改写开关真源"


def test_patrol_chain_never_touches_real_inspector_queue(
    tmp_path, monkeypatch, no_real_queue_pollution
):
    """根治回归（insp-20261022-141842）：巡检链只写 tmp 队列，真实队列零字节变化。"""
    real_before = _real_queue_bytes()
    spawns, _ = _run_patrol_chain(tmp_path, monkeypatch)
    assert workq.pending_count(inspectd.QUEUE_PATH) >= 1, "tmp 队列必须拿到项"
    assert spawns, "worker spawn 链完好（不因隔离而丢覆盖）"
    assert _real_queue_bytes() == real_before, (
        "巡检链不得写真实 inspector 队列（泄漏→开关 off 时无人消化→QUEUE-TIMEOUT 唤醒组长）"
    )


# ---------------------------------------------------------------------------
# ④ 队列超时告警链：入队 30min → 事件层 → TUI 读数
# ---------------------------------------------------------------------------
def test_queue_timeout_chain_event_to_tui(tmp_path, monkeypatch):
    root = _state_paths(monkeypatch, tmp_path)
    # 常闭降噪后 patrol 的超时告警只在开关开启时发生——本链测「开」态的告警链路
    monkeypatch.setattr(inspectd, "enabled", lambda: True)
    it = inspectd.enqueue("patrol", target="code-audit")
    workq.mutate(inspectd.QUEUE_PATH, it["id"], enqueued_at=_iso(time.time() - 35 * 60))

    assert len(inspectd.notify_stale_queue()) == 1
    evs = events.read_all()
    assert [e["event"] for e in evs] == ["QUEUE-TIMEOUT"]
    assert evs[0]["subject"] == it["id"]
    assert events.WAKE["QUEUE-TIMEOUT"], "事件词表单一源（events.WAKE）"

    assert inspectd.notify_stale_queue() == [], "每项只报一次（去重）"

    q = next(m for m in modules() if m.key == "queues")
    assert any("队列超时" in ln for ln in q.detail()), "TUI 队列与运行须点名超时项"
    r = next(m for m in modules() if m.key == "review")
    assert not any("超时项" in ln for ln in r.detail()), (
        "s3：inspector 超时明细归队列模块（用户点名从盲审与审查面删除）"
    )
    monkeypatch.setattr(
        inspectd, "enabled", lambda: True
    )  # status 段：开关关会提前 return
    assert "队列超时" in "\n".join(inspectd.status_lines())


# ---------------------------------------------------------------------------
# ⑤ 用户指令：inspector 全局开关终检 = off，e2e 只在内存翻
# ---------------------------------------------------------------------------
def test_inspector_switch_stays_off():
    """用户指令（2026-10-14）：全局开关保持 off。

    这条红了只有两种可能——有测试翻了它（违规），或用户在真源里开了（那也该让任务书的
    「终检」口径随之更新，而不是悄悄过）。e2e 一律用 monkeypatch 在内存里模拟开启。
    """
    assert settings.inspector_flag() is False, (
        "inspector 全局开关须保持 off（用户指令）"
    )


# ---------------------------------------------------------------------------
# ⑥ 文档/技能与实现一致性（沿 test_readme_refactor_closeout 惯例）
# ---------------------------------------------------------------------------
def test_readme_s4_5_states_three_tier_defense():
    readme = _readme()
    sec = readme[readme.index("### 4.5") : readme.index("## 5.")]
    for marker in (
        "三级防线",
        "watchdog-scheduler.ps1",
        "-Mode install",
        "巡更",
        "pysci-watchdog",
    ):
        assert marker in sec, f"§4.5 缺三级防线要素：{marker}"


def test_readme_s6_states_self_patrol_and_timeout():
    readme = _readme()
    sec = readme[readme.index("## 6.") : readme.index("## 7.")]
    for marker in ("self_patrol_tick", "notify_stale_queue", "QUEUE-TIMEOUT", "1800"):
        assert marker in sec, f"§6 缺自调度/超时告警实现点名：{marker}"


def test_skill_indexes_patrol_reference():
    skill = (
        PROJECT_ROOT / "orchestration" / "skills" / "orchestration" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "references/patrol-watchdog.md" in skill
    ref = (
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "patrol-watchdog.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "CronCreate",
        "30 min",
        "全空闲",
        "watchdog-scheduler.ps1",
        "sleep-轮询",
        "test_autonomy_e2e",
    ):
        assert marker in ref, f"巡更规程缺要素：{marker}"


def _doc_corpus() -> str:
    """文档语料（README v3.0 拆分后口径）：`references/legacy-readme-v2.2.md`（v2.2 原文留档）
    + `README.md`（v3.0 正文）+ `references/*.md`（细节块）。doc-face 断言据此判「文档面是否声明 X」。
    """
    root = PROJECT_ROOT / "orchestration"
    parts: list[str] = []
    legacy = root / "references" / "legacy-readme-v2.2.md"
    if legacy.exists():
        parts.append(legacy.read_text(encoding="utf-8"))
    parts.append((root / "README.md").read_text(encoding="utf-8"))
    refs = root / "references"
    if refs.is_dir():
        for f in sorted(refs.glob("*.md")):
            if f.name != "legacy-readme-v2.2.md":
                parts.append(f.read_text(encoding="utf-8"))
    return (chr(10) * 2).join(parts)


def _readme() -> str:
    return _doc_corpus()
