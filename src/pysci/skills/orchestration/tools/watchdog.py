"""watch 看门狗（autonomy-fix a1）：独立于会话/钩子生命周期的 **watch 自愈监督者**。

**为什么需要它**（2026-10-14 组长取证，可复核）：watcher 进程曾被**无声杀死**——
``watch-arm.log`` 02:23:02「被取代」→ 02:23–07:40:30 **零日志** → 07:40:30 另一 token 接任。
watcher 的所有**正常**退出路径都留痕（命中 / 被取代 / `rc≠`），零留痕只能是**外部终止**
（详见 README §4.5 的取证）。而「每回合点火依赖组长会话活跃」——不被唤醒就没有下一次点火，
形成**死循环依赖**：被杀的 watcher 就此永远无人唤醒。

本模块是**监督者**（supervisor），与被监督的 listener（``.qoder/hooks/watch-arm.sh`` 的
监听循环）解耦：
- 每 ``interval_s`` 读一次 ``watch.json`` 的 ``heartbeat_at``；
- **心跳新鲜**（≤ ``stale_s``）⇒ 有活着的 listener，什么都不做；
- **心跳 stale/缺失** ⇒ **重臂**：拉起一条 ``watch-arm.sh`` 监听循环（它自抢 token 并循环
  ``orch watch``），落日志；带**冷却期**（≥ ``stale_s`` 不重复拉），防反复拉起堆积进程。

生命周期走 ``orch watchdog start|stop|status``：``start`` 幂等（per-守护者锁）拉起**分离
进程**，``stop`` 协作式停止（不留孤儿守护者进程），``status`` 报读数。看门狗自身**不依赖**
任何会话/钩子——OS 侧重启持久化可选，安装/卸载见
``orchestration/scripts/watchdog-scheduler.ps1``。

**a2 耦合点（唯一调度钩子）**：同一 tick 顺带敲 :func:`inspectd.self_patrol_tick`（闲时自
调度巡检）与 :func:`inspectd.notify_stale_queue`（队列入队超时告警）——让路/节流/cooldown
/预算闸全在 inspectd 内部，本模块不重算，纪律是**不造第二套调度循环**。
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import drain as _drain
from . import inspectd as _inspectd
from . import watch as _watch
from .drain import (
    STOP_SCOPE_ALL,
    WorkerPaths,
    _consume_stop_request,
    _spawn_detached,
    acquire_lock,
    cancel_status,
    lock_status,
    new_run_log,
    release_lock,
    request_stop,
)
from .registry import orchestration_rel

#: 看门狗自身的生命周期落点（锁/日志/协作停止信号）——复用 drain 原语，不另造一套。
#:
#: ``stale_s`` **显式取 6h**（不随 ``drain.LOCK_STALE_S`` 的 45min 下调走）：看门狗是
#: **常驻**监督者，其锁的「黏性」正是 ``start_watchdog`` 幂等判据（锁在跑就不重复派生）；
#: 而它此刻并无 LiveSession 心跳可作免抢证据（心跳面 :func:`listener_live` 判的是被监督的
#: watch listener，不是看门狗自己）。若按 45min 判死，一个跑了 2 小时的健康看门狗会被自己的
#: 下一任抢锁 → 双看门狗（与 drain 侧「误抢=双 worker」同族事故）。
WATCHDOG_LOCK_STALE_S: int = 6 * 3600

WATCHDOG_PATHS = WorkerPaths(
    lock=ORCH_STATE_ROOT / "watchdog.lock",
    runs=ORCH_STATE_ROOT / "watchdog-runs",
    cancel=ORCH_STATE_ROOT / "watchdog.cancel",
    stale_s=WATCHDOG_LOCK_STALE_S,
)

#: 自查周期（秒）。分钟级即可——判死阈值是分钟级的，查得再勤也不改变结论。
CHECK_INTERVAL_S: int = 60

#: watch 心跳判死阈值（秒）：> 它即认为**没有活着的 listener**，需重臂。
#: 取 300 s（≤5 min，§a1 建议值）且 **≥ watch 的挂载宽限**（``interval*3+60``，默认 240 s）
#: ——否则「刚 arm 完就被杀」的 watcher 会在宽限期内被误判成活着，重臂被推迟。
HEARTBEAT_STALE_S: int = 300

#: 监听循环脚本（token 竞争 + 循环 ``orch watch``）——看门狗**重臂的就是它**，不新造循环。
ARM_SCRIPT: Path = PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh"

#: 内部命令名（``orch _watchdog-loop``）。
_INTERNAL_CMD = "_watchdog-loop"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(line: str, run_log: Path | None = None) -> None:
    """看门狗日志（全部 run 共用一份 ``watchdog-runs/watchdog.log``，append-only）。"""
    try:
        target = run_log or (WATCHDOG_PATHS.runs / "watchdog.log")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {line}\n")
    except OSError:  # noqa: PERF203 - 日志失败不阻塞守护
        pass


# ---------------------------------------------------------------------------
# 心跳读数（判死的唯一依据）
# ---------------------------------------------------------------------------
def heartbeat_age_s(*, now: float | None = None) -> float:
    """``watch.json`` 的心跳距今秒数；读不出/缺失 → ``+inf``（= 无活监听）。"""
    st = _watch.load_state()
    hb = str(st.get("heartbeat_at") or "")
    moment = time.time() if now is None else float(now)
    try:
        dt = datetime.fromisoformat(hb)
    except (TypeError, ValueError):
        return float("inf")
    return float(moment - dt.timestamp())


def listener_live(
    *, now: float | None = None, stale_s: int = HEARTBEAT_STALE_S
) -> bool:
    """心跳是否新鲜（≤ ``stale_s``）——即「此刻有没有活着的 listener」。"""
    return heartbeat_age_s(now=now) <= stale_s


# ---------------------------------------------------------------------------
# 重臂：拉起一条监听循环（watch-arm.sh 自抢 token + 循环 orch watch）
# ---------------------------------------------------------------------------
def _rearm(run_log: Path | None = None) -> bool:
    """拉起监听循环；成功 True（失败只留痕，不抛——看门狗要活到下一轮）。"""
    bash = shutil.which("bash")
    if not bash:
        _log(
            "[!] 找不到 bash——无法重臂（监听循环在 .qoder/hooks/watch-arm.sh）", run_log
        )
        return False
    if not ARM_SCRIPT.is_file():
        _log(f"[!] 监听循环脚本缺失：{ARM_SCRIPT}", run_log)
        return False
    env = {**os.environ, "QODER_PROJECT_DIR": str(PROJECT_ROOT)}
    # 重臂实例无主：剥掉派生方携带的 owner 身份（owner 钩子 spawn 的看门狗会继承
    # ``PYSCI_WAKE_OWNER``——漏剥则替身 watch 盖上 owner 的 ``armed_by``、冒充「owner
    # 自己的 watch」把 stale 接管永久挡死；watch-arm.sh 侧另有 unset 双保险）。
    env.pop(_watch.WATCH_OWNER_ENV, None)
    spawn_log = (run_log or (WATCHDOG_PATHS.runs / "watchdog.log")).with_suffix(
        ".spawn.log"
    )
    try:
        spawn_log.parent.mkdir(parents=True, exist_ok=True)
        with spawn_log.open("ab") as fh:
            _spawn_detached([bash, str(ARM_SCRIPT)], env=env, stdout=fh)
    except OSError as exc:
        _log(f"[!] 重臂 spawn 失败：{exc}", run_log)
        return False
    return True


def run_watchdog(
    run_log: Path,
    *,
    interval_s: int = CHECK_INTERVAL_S,
    stale_s: int = HEARTBEAT_STALE_S,
    sleeps: Any = None,
    now: Any = time.time,
) -> dict[str, Any]:
    """监督循环（持锁运行）：心跳新鲜不动，stale 即重臂。**测试注入口** ``now``/``sleeps``。

    停止消费按**广播纪元**判定（``started_at`` = 本实例启动时刻）：纪元晚于它才退出，
    早于它的陈旧纪元不误停（backlog 20261014-135935-devops）。
    """
    if acquire_lock(orchestration_rel(run_log), WATCHDOG_PATHS) is None:
        return {"skipped": "lock_busy"}
    sleep = _drain._sleep if sleeps is None else sleeps
    started_iso = _now()
    summary: dict[str, Any] = {
        "started": started_iso,
        "checks": 0,
        "rearms": 0,
        "run_log": str(run_log),
        "interval_s": int(interval_s),
        "stale_s": int(stale_s),
    }
    last_live = listener_live(now=now(), stale_s=stale_s)
    last_rearm = 0.0
    _log(f"[▶] 看门狗启动（心跳{'新鲜' if last_live else 'stale'}）", run_log)
    try:
        while True:
            if _consume_stop_request(WATCHDOG_PATHS, started_at=started_iso):
                summary["stopped"] = "cooperative"
                break
            sleep(interval_s)
            live = listener_live(now=now(), stale_s=stale_s)
            summary["checks"] += 1
            if live != last_live:
                if live:
                    _log("[√] 监听心跳恢复新鲜（listener 已在跑）", run_log)
                else:
                    _log("[!] 监听心跳转 stale（listener 失踪）", run_log)
                last_live = live
            if not live:
                # 冷却：一次重臂后给它至少一个阈值的时间去拉起并刷新心跳，防堆进程。
                if float(now()) - last_rearm >= stale_s:
                    if _rearm(run_log):
                        summary["rearms"] += 1
                        _log(f"[→] 重臂监听循环（>{stale_s}s 无心跳）", run_log)
                    last_rearm = float(now())
            # a2 耦合点：同一 tick 顺带驱动 inspector 的**闲时自调度巡检**与**队列超时告警**
            # （纪律「不造第二套调度/唤醒循环」——细节与全部闸门全在 inspectd，这里只敲一下）。
            try:
                _inspectd.notify_stale_queue(now=now())
                _inspectd.self_patrol_tick()
            except Exception as exc:  # noqa: BLE001 - 看门狗要活到下一轮
                _log(f"[!] inspector tick 异常：{type(exc).__name__}: {exc}", run_log)
            # 队列**自愈**（task-20261015-worker-stall B2）：孤儿 in_progress 复位 + 有活无人干
            # 时唤醒 worker。为什么挂在看门狗而不是 worker 自己：worker 死了就没人再拉它
            # （队列无 pending → 无唤醒 → 鸡生蛋），而看门狗是**唯一**与队列状态无关的常驻面。
            try:
                from . import queueworker as _qw

                healed = _qw.reclaim_orphans(now=now())
                if healed.get("reclaimed") or healed.get("woken"):
                    _log(f"[→] 队列自愈：{healed}", run_log)
            except Exception as exc:  # noqa: BLE001 - 看门狗要活到下一轮
                _log(f"[!] 队列自愈异常：{type(exc).__name__}: {exc}", run_log)
            if (
                int(summary["checks"]) % max(1, int(stale_s) // max(1, int(interval_s)))
                == 0
            ):
                _log(
                    f"[i] 监督中 checks={summary['checks']} 心跳 age="
                    f"{heartbeat_age_s(now=now()):.0f}s listener={'活' if live else 'stale'}",
                    run_log,
                )
    except Exception as exc:  # noqa: BLE001 - 看门狗要活到下一轮
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"[!] 看门狗异常：{summary['error']}", run_log)
    finally:
        release_lock(run_log, WATCHDOG_PATHS)
        summary["completed"] = _now()
        _log("[√] 看门狗退出（已释放锁）", run_log)
    return summary


# ---------------------------------------------------------------------------
# 生命周期：start / stop / status（orch 命令面）
# ---------------------------------------------------------------------------
def start_watchdog(
    *, interval_s: int = CHECK_INTERVAL_S, stale_s: int = HEARTBEAT_STALE_S
) -> tuple[bool, str]:
    """幂等拉起看门狗（分离进程；已在跑则不动）。

    Returns:
        ``(spawned, message)``——``False`` = 已在跑（不重复派生）。
    """
    if lock_status(WATCHDOG_PATHS).get("state") == "running":
        return False, "看门狗已在跑（幂等：不重复派生）"
    WATCHDOG_PATHS.runs.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log(WATCHDOG_PATHS)
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        _INTERNAL_CMD,
        "--run-log",
        str(run_log),
        "--interval",
        str(interval_s),
        "--stale",
        str(stale_s),
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        with run_log.open("ab") as fh:
            _spawn_detached(argv, env=env, stdout=fh)
    except OSError as exc:
        return False, f"看门狗拉起失败：{exc}"
    return (
        True,
        f"看门狗已后台启动（日志 {orchestration_rel(WATCHDOG_PATHS.runs / 'watchdog.log')}）",
    )


def stop_watchdog() -> tuple[bool, str]:
    """协作式停止看门狗（**广播停止纪元**；各实例在下一轮自查点干净退出）。

    backlog 20261014-135935-devops：**一次 stop 停全部实例**（含不持锁的孤儿）——
    旧实现只对锁持有者 pid 投递、锁空闲时干脆不写信号，孤儿实例会把信号当「陈旧」
    清掉继续跑（2026-10-14 实测：两实例同跑，一次 stop 只停一个）。广播纪元不删文件、
    按启动时刻比较，故锁空闲也照投、新实例不受旧纪元误停。
    """
    return request_stop(WATCHDOG_PATHS, broadcast=True)


def status_lines(
    *, now: float | None = None, stale_s: int = HEARTBEAT_STALE_S
) -> list[str]:
    """看门狗/监听读数（`orch watchdog status`）。"""
    st = lock_status(WATCHDOG_PATHS)
    age = heartbeat_age_s(now=now)
    age_txt = "∞（无心跳）" if age == float("inf") else f"{age:.0f}s"
    out = [
        f"  看门狗 = {st.get('state')}",
        f"  watch 心跳 age = {age_txt}（判死阈值 {stale_s}s）",
        f"  listener = {'活着' if age <= stale_s else 'stale/无（看门狗将重臂）'}",
    ]
    epoch = cancel_status(WATCHDOG_PATHS)
    if epoch and str(epoch.get("scope") or "") == STOP_SCOPE_ALL:
        out.append(
            f"  停止纪元 = {epoch.get('requested_at', '?')}"
            "（早于它的实例将退出；新实例不受影响）"
        )
    return out


def cmd_watchdog_loop(args: Any) -> int:
    """内部命令 ``orch _watchdog-loop``（``orch watchdog start`` 派生）。"""
    run_log = Path(args.run_log) if getattr(args, "run_log", None) else None
    if run_log is None:
        WATCHDOG_PATHS.runs.mkdir(parents=True, exist_ok=True)
        run_log = new_run_log(WATCHDOG_PATHS)
    elif not run_log.is_absolute():
        run_log = PROJECT_ROOT / run_log
    run_watchdog(
        run_log,
        interval_s=int(getattr(args, "interval", CHECK_INTERVAL_S) or CHECK_INTERVAL_S),
        stale_s=int(getattr(args, "stale", HEARTBEAT_STALE_S) or HEARTBEAT_STALE_S),
    )
    return 0


def cmd_watchdog(args: Any) -> int:
    """``orch watchdog start|stop|status``：watch 自愈监督者的生命周期与读数。"""
    action = getattr(args, "w_action", "") or "status"
    if action == "start":
        ok, msg = start_watchdog()
        print(("[√] " if ok else "[=] ") + msg)
        return 0
    if action == "stop":
        ok, msg = stop_watchdog()
        print(("[√] " if ok else "[!] ") + msg)
        return 0 if ok else 2
    for line in status_lines():
        print(line)
    return 0


__all__ = [
    "ARM_SCRIPT",
    "CHECK_INTERVAL_S",
    "HEARTBEAT_STALE_S",
    "WATCHDOG_PATHS",
    "cmd_watchdog",
    "cmd_watchdog_loop",
    "heartbeat_age_s",
    "listener_live",
    "run_watchdog",
    "start_watchdog",
    "status_lines",
    "stop_watchdog",
]
