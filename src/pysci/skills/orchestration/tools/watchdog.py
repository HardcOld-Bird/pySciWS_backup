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
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import drain as _drain
from . import inspectd as _inspectd
from . import procs as _procs
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

#: ``start`` 的「查在跑 → 派生」互斥标记的存在上限（秒）：持有者若卡死/被杀，标记超龄即被
#: 回收，不永久挡住下一次 start。取 120s（远大于「派生 + 等它上锁」的真实耗时）。
SPAWN_MARKER_TTL_S: float = 120.0

#: 竞争者等标记释放的上限（秒）——等不到就如实回报「正在拉起中」，不重复派生。
SPAWN_MARKER_WAIT_S: float = 10.0

#: 派生后等子进程**持锁**的上限（秒）：只查锁不等待的话，紧随其后的第二次 start 会在
#: 子进程上锁前看到「不在跑」而各自再派生一个（同秒并发双生）。
SPAWN_LOCK_WAIT_S: float = 3.0

#: 上式与标记等待的轮询步长（秒）。
_SPAWN_POLL_S: float = 0.02


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
        # 留痕：抢不到锁的实例**秒退**，但它的 run_log 已被 new_run_log 建成空文件——
        # 不留一行就取证成谜（2026-10-15 双生单：双生之一或纯空跑看不出差别的现场）。
        _log("[=] 未取到锁（lock_busy）：另一实例在跑，本实例退出", run_log)
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}
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
    #: 「已提交未合并」出声去重集（本进程持有；见 tick 里的检出段）。
    unmerged_seen: set = set()
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
            # 存活会话登记表清理（wake-adopt-20261018 s1）：死会话（PID 不在且心跳过期）
            # 不留在候选集里；同一 tick 顺手做，不造第二套定时器。
            try:
                pruned = _watch.prune_live_sessions()
                if pruned:
                    _log(f"[→] 清理死会话登记 {len(pruned)} 个：{pruned}", run_log)
            except Exception as exc:  # noqa: BLE001 - 看门狗要活到下一轮
                _log(f"[!] 会话登记清理异常：{type(exc).__name__}: {exc}", run_log)
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
            # 「已提交未合并」检出（backlog 20261015-145045，f4a）：devops 尾段被杀/截断时
            # 成果会卡在 worktree 里静默躺平（实案 6304 卡 3 小时）。挂同一 tick（不造第二套
            # 定时器）；**只出声不自动合并**（护栏级写操作，留给组长有意授权）。去重集由本
            # 进程持有：同一面同一 head 只喊一次，head 变了再喊。
            try:
                fresh = _drain.announce_unmerged_worktrees(unmerged_seen)
                if fresh:
                    _log(f"[→] 检出已提交未合并 {len(fresh)} 个面：{fresh}", run_log)
            except Exception as exc:  # noqa: BLE001 - 看门狗要活到下一轮
                _log(f"[!] 未合并检出异常：{type(exc).__name__}: {exc}", run_log)
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
def _spawn_marker() -> Path:
    """派生互斥标记的落点（跟随 ``WATCHDOG_PATHS``——用例可整体改道到隔离面）。"""
    return WATCHDOG_PATHS.lock.with_name(WATCHDOG_PATHS.lock.stem + ".spawn")


def _take_spawn_marker(*, wait_s: float = SPAWN_MARKER_WAIT_S) -> bool:
    """原子占用派生标记（``O_EXCL`` 建成即得）；占不到就等到 ``wait_s`` 或标记超龄回收。

    判据只看**存在性**、不读内容——故不存在 ``acquire_lock`` 那种「半写载荷」窗口
    （内容不参与判定）。超龄（> :data:`SPAWN_MARKER_TTL_S`）视为持有者卡死，回收后重试。
    """
    mark = _spawn_marker()
    deadline = time.time() + wait_s
    mark.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(mark), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - mark.stat().st_mtime
            except OSError:
                continue  # 刚好被释放：下一轮直接抢
            if age > SPAWN_MARKER_TTL_S:
                try:
                    mark.unlink()
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                return False
            time.sleep(_SPAWN_POLL_S)
            continue
        except OSError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {_now()}\n")
        return True


def _release_spawn_marker() -> None:
    try:
        _spawn_marker().unlink()
    except OSError:
        pass


def _await_lock_running(*, wait_s: float = SPAWN_LOCK_WAIT_S) -> bool:
    """等看门狗锁变成 ``running``（派生成功的前兆）；超时返回 False（如实回报，不当失败）。"""
    deadline = time.time() + wait_s
    while True:
        if lock_status(WATCHDOG_PATHS).get("state") == "running":
            return True
        if time.time() >= deadline:
            return False
        time.sleep(_SPAWN_POLL_S)


def start_watchdog(
    *, interval_s: int = CHECK_INTERVAL_S, stale_s: int = HEARTBEAT_STALE_S
) -> tuple[bool, str]:
    """幂等拉起看门狗（分离进程；已在跑则不动）。

    「查在跑 → 派生」整段持 :func:`_take_spawn_marker` 互斥，且派生后**等子进程上锁**才
    归还标记：否则两次 start 相隔毫秒级（手工 start 与 watch-arm 钩子自启动撞车）会双双
    通过幂等判定、各派生一个（2026-10-15 双生单的竞态窗口）。

    Returns:
        ``(spawned, message)``——``False`` = 已在跑 / 他方正在拉起（不重复派生）。
    """
    if not _take_spawn_marker():
        return False, "看门狗正在拉起中（派生互斥标记被占，不重复派生）"
    try:
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
        _await_lock_running()  # 等它上锁，幂等判定才对紧随其后的 start 生效
        return (
            True,
            "看门狗已后台启动（日志 "
            f"{orchestration_rel(WATCHDOG_PATHS.runs / 'watchdog.log')}）",
        )
    finally:
        _release_spawn_marker()


def stop_watchdog() -> tuple[bool, str]:
    """协作式停止看门狗（**广播停止纪元**；各实例在下一轮自查点干净退出）。

    backlog 20261014-135935-devops：**一次 stop 停全部实例**（含不持锁的孤儿）——
    旧实现只对锁持有者 pid 投递、锁空闲时干脆不写信号，孤儿实例会把信号当「陈旧」
    清掉继续跑（2026-10-14 实测：两实例同跑，一次 stop 只停一个）。广播纪元不删文件、
    按启动时刻比较，故锁空闲也照投、新实例不受旧纪元误停。
    """
    return request_stop(WATCHDOG_PATHS, broadcast=True)


#: 从命令行取 ``--run-log`` 取值（``--run-log X`` / ``--run-log=X``；值可带引号）。
_RUN_LOG_ARG_RE = re.compile(r"--run-log[=\s]+(?:([\"'])(.+?)\1|(\S+))")


def watchdog_instances(
    processes: list[tuple[int, str]] | None, *, marker: str = _INTERNAL_CMD
) -> dict[str, Any] | None:
    """把「命令行表」归并成**逻辑看门狗实例**读数（纯函数，无副作用）。

    为什么不能数进程（backlog 20261015-182745-devops，2026-10-15 实测）：venv 的
    ``.venv/Scripts/python.exe`` 是**转发垫片**，一个逻辑看门狗恒产生**两个同 argv 进程**
    （父=垫片、子=uv 托管的真解释器，锁由子进程持有）——按「进程数」验收会把健康的单实例
    误判成双生（实测同刻：2 进程 / 1 个 ``--run-log``，实例数按进程口径**永远读不到 1**）。

    故口径定为 **distinct ``--run-log`` 数 = 逻辑实例数**；且只认**真 loop 进程**（命令行
    含标记**且**带 ``--run-log``）——仅「提到了这个字符串」的命令（grep / PowerShell 探针）
    不计入，防假阳性。

    Args:
        processes: ``[(pid, 命令行), …]``；``None`` = 进程面读不到（**透传 None**，
            绝不折算成 0——那会让验收假绿）。

    Returns:
        ``{"logical", "loop_processes", "run_logs", "mentioned"}``；入参为 ``None`` 时
        返回 ``None``。
    """
    if processes is None:
        return None
    logs: list[str] = []
    seen: set[str] = set()
    loop_processes = 0
    mentioned = 0
    for _pid, cmd in processes:
        if marker not in cmd:
            continue
        m = _RUN_LOG_ARG_RE.search(cmd)
        if m is None:
            mentioned += 1
            continue
        loop_processes += 1
        run_log = (m.group(2) or m.group(3) or "").strip()
        key = os.path.normcase(run_log)
        if key and key not in seen:
            seen.add(key)
            logs.append(run_log)
    return {
        "logical": len(logs),
        "loop_processes": loop_processes,
        "run_logs": logs,
        "mentioned": mentioned,
    }


def status_lines(
    *,
    now: float | None = None,
    stale_s: int = HEARTBEAT_STALE_S,
    instances: bool = True,
    scan: Any = None,
) -> list[str]:
    """看门狗/监听读数（`orch watchdog status`）。

    ``instances``（默认开）附**逻辑实例数**：口径 = distinct ``--run-log``
    （见 :func:`watchdog_instances`）——垫片+托管解释器两个进程算**一个**实例；按进程数
    验收会把健康的单实例误判成双生。进程面读不到时显式打 ``?`` 而非 ``0``（防假绿）。
    ``scan`` 是测试注入口（默认 :func:`procs.list_processes_matching`）。
    """
    st = lock_status(WATCHDOG_PATHS)
    age = heartbeat_age_s(now=now)
    age_txt = "∞（无心跳）" if age == float("inf") else f"{age:.0f}s"
    out = [
        f"  看门狗 = {st.get('state')}",
        f"  watch 心跳 age = {age_txt}（判死阈值 {stale_s}s）",
        f"  listener = {'活着' if age <= stale_s else 'stale/无（看门狗将重臂）'}",
    ]
    if instances:
        rows = (scan or _procs.list_processes_matching)(_INTERNAL_CMD)
        got = watchdog_instances(rows)
        if got is None:
            out.append(
                "  逻辑实例数 = ?（进程面读不到，不得按 0 记；口径：distinct --run-log）"
            )
        else:
            out.append(
                f"  逻辑实例数 = {got['logical']}（loop 进程 {got['loop_processes']}；"
                "口径：distinct --run-log——venv 垫片与托管解释器同 argv，算一个）"
            )
            if got["mentioned"]:
                out.append(
                    f"  （另有 {got['mentioned']} 个进程仅提及标记、非 loop，未计入）"
                )
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
