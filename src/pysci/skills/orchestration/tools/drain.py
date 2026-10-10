"""devops backlog 自动消化：锁 / 后台唤醒 / drain 循环（README §6）。

``orch approve`` 采纳建议入 backlog 后，orch 自动以**分离后台进程**唤醒 devops，
FIFO 串行消化整个 backlog——组长零阻塞、零轮询（对齐 README §4.5 等待模型）。

三块机制：

- **锁**（``state/devops.lock``，JSON: pid/started/run_log）：同刻至多一个 drain
  worker。原子 ``O_CREAT|O_EXCL`` 抢占；持有者进程已死或锁龄超 :data:`LOCK_STALE_S`
  判 stale 并接管（fail-open，防 worker 崩溃后永久死锁）。
- **唤醒**（:func:`wake_devops`）：approve 成功后调用；锁忙则不重复 spawn（本项由
  当前 drain 循环接手），空闲则 Popen 分离进程运行内部命令 ``_drain-devops``。
- **drain 循环**（:func:`drain_backlog`）：获锁后先回收孤儿 in_progress 条目（已死
  worker 遗留，重置 pending）；``backlog_take_first`` → 任务书 →
  ``do_dispatch("devops")`` → 销账；单项失败（blocked/run_failed）标记 needs_leader
  并**跳过**（一项卡住不阻塞全队列）；run 日志逐行落 ``state/devops-runs/<ts>.log``，
  全部完成写 ``<ts>.done``（JSON 摘要）。

内部命令 ``_drain-devops`` 下划线前缀，argparse help 标注为 ``[内部]``（勿手调）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from .registry import orchestration_rel
from .workflow import (
    backlog_complete,
    backlog_needs_leader,
    backlog_reclaim_orphans,
    backlog_take_first,
    backlog_task_text,
)

#: drain 互斥锁（JSON: pid/started/run_log）；仅 orch 与 devops drain 读写。
LOCK_PATH: Path = ORCH_STATE_ROOT / "devops.lock"

#: run 日志与 .done 摘要目录。
DEVOPS_RUNS_DIR: Path = ORCH_STATE_ROOT / "devops-runs"

#: 锁龄上限（秒）——超过即判 stale 接管（worker 崩溃兜底）。6h 覆盖最长 devops 任务。
LOCK_STALE_S: int = 6 * 3600

#: 干跑开关环境变量：置真值时 wake 派生的 drain 走 --dry（不真派发，演练/验证唤醒链用）。
DRY_ENV: str = "PYSCI_ORCH_DRAIN_DRY"

_COMMIT_RE = re.compile(r"commit[^0-9a-f]{0,16}\b([0-9a-f]{7,40})\b", re.IGNORECASE)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 锁
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """判断 pid 是否仍存活（Windows: tasklist；POSIX: os.kill 0 信号）。

    判活失败（命令不可用/超时）时**保守视为存活**——宁可让锁忙等下一轮，也不误抢
    正在运行的 worker（误抢会导致双 worker 并发消费 backlog）。
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return True
        return re.search(rf"\b{pid}\b", proc.stdout or "") is not None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock() -> dict[str, Any] | None:
    """读取锁记录；不存在或损坏返回 None（损坏锁将由 acquire 覆盖）。"""
    if not LOCK_PATH.exists():
        return None
    try:
        data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _lock_age_s(data: dict[str, Any]) -> float:
    """锁龄（秒）；started 不可解析时返回 -1（视为未知，不按超龄处理）。"""
    try:
        started = datetime.fromisoformat(str(data.get("started", "")))
    except (ValueError, TypeError):
        return -1.0
    return (datetime.now().astimezone() - started).total_seconds()


def _lock_live(data: dict[str, Any] | None) -> bool:
    """锁是否有效：持有者进程存活且锁龄未超上限。"""
    if not data:
        return False
    age = _lock_age_s(data)
    if age > LOCK_STALE_S:
        return False
    return _pid_alive(int(data.get("pid", 0) or 0))


def acquire_lock(run_log: str) -> dict[str, Any] | None:
    """尝试获取 drain 锁。

    Args:
        run_log: 本次 run 的日志相对路径（写入锁记录供 status 展示）。

    Returns:
        成功获取时返回锁记录 ``{pid, started, run_log}``；已被存活 worker 持有时
        返回 ``None``（调用方应优雅退出，不重复消费）。stale 锁（进程死/超龄）被接管。
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "started": _now(), "run_log": run_log}
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    for _ in range(3):
        if _lock_live(read_lock()):
            return None
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # 竞态：他方刚抢占。若其存活则退出；若仍是 stale 残留则清除后重试。
            if _lock_live(read_lock()):
                return None
            try:
                os.unlink(str(LOCK_PATH))
            except OSError:
                pass
            continue
        except OSError:
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return record
    return None


def release_lock() -> None:
    """释放锁——**仅当持有者是本进程**（防未获取到锁的进程误删他人锁）。"""
    data = read_lock()
    if data and int(data.get("pid", 0) or 0) == os.getpid():
        try:
            os.unlink(str(LOCK_PATH))
        except OSError:
            pass


def lock_status() -> dict[str, Any]:
    """status 命令用的锁态摘要：``{state: idle|running|stale, pid, started, ...}``。"""
    data = read_lock()
    if not data:
        return {"state": "idle"}
    pid = int(data.get("pid", 0) or 0)
    age = _lock_age_s(data)
    alive = _pid_alive(pid)
    state = "running" if (alive and 0 <= age <= LOCK_STALE_S) else "stale"
    return {
        "state": state,
        "pid": pid,
        "started": data.get("started", ""),
        "run_log": data.get("run_log", ""),
        "age_s": round(age, 1) if age >= 0 else -1,
    }


# ---------------------------------------------------------------------------
# run 日志 / .done
# ---------------------------------------------------------------------------
def _log(run_log: Path, line: str) -> None:
    """向 run 日志追加一行（带时间戳）。日志写入失败不阻塞 drain（fail-open）。"""
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
    except OSError:
        pass


def new_run_log() -> Path:
    """生成本次 run 的日志路径 ``devops-runs/<ts>.log``（不落盘）。"""
    return DEVOPS_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def latest_done() -> dict[str, Any] | None:
    """最近一次 run 的 .done 摘要（无则 None）；附 ``name`` 字段。"""
    if not DEVOPS_RUNS_DIR.exists():
        return None
    dones = sorted(DEVOPS_RUNS_DIR.glob("*.done"))
    if not dones:
        return None
    try:
        data = json.loads(dones[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        data["name"] = dones[-1].stem
    return data if isinstance(data, dict) else None


def _extract_commit(body: str) -> str:
    """从交付正文尽力提取 commit hash（``commit ... <hex>``）；无则空串。"""
    m = _COMMIT_RE.search(body or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# drain 循环
# ---------------------------------------------------------------------------
def _drain_one(item: dict[str, Any], run_log: Path, *, dry: bool) -> dict[str, Any]:
    """消化单个 backlog 条目：派发 devops → 销账（成功）/ needs_leader（失败跳过）。

    Returns:
        结果字典（写入 .done 的 items 列表）：``{id, status, commit?, kind?, elapsed_s}``。
    """
    item_id = str(item.get("id", ""))
    summary = str(item.get("summary", ""))[:60]
    _log(run_log, f"[→] 消化 {item_id}: {summary}")
    t0 = time.time()

    if dry:
        backlog_complete(item_id, note="drain --dry 干跑（未实派 devops）")
        _log(run_log, f"[dry] {item_id} 干跑销账完成")
        return {"id": item_id, "status": "dry", "elapsed_s": round(time.time() - t0, 1)}

    from .dispatch import (
        do_dispatch,  # 延迟导入：避免 drain↔dispatch 环（dispatch 不依赖 drain）
    )

    outcome = do_dispatch(
        "devops",
        text=backlog_task_text(item),
        slug=item_id,
        session="latest",
        quiet=True,
    )
    elapsed = round(time.time() - t0, 1)
    if outcome.code == 0:
        commit = _extract_commit(outcome.body)
        backlog_complete(item_id, note=f"drain 自动销账（commit {commit or '未报告'}）")
        _log(run_log, f"[√] {item_id} 完成（{elapsed}s，commit={commit or '-'}）")
        return {
            "id": item_id,
            "status": "done",
            "commit": commit,
            "elapsed_s": elapsed,
        }
    # 失败（blocked / run_failed / parse_error / 验收 FAIL）→ 跳过并标记 needs_leader
    err = (outcome.error or outcome.body[:160] or "").strip().replace("\n", " ")
    backlog_needs_leader(
        item_id, note=f"drain 跳过（交付={outcome.kind}）：{err[:160]}"
    )
    _log(run_log, f"[!] {item_id} 跳过→needs_leader（{outcome.kind}）：{err[:120]}")
    return {
        "id": item_id,
        "status": "needs_leader",
        "kind": outcome.kind,
        "error": err[:200],
        "elapsed_s": elapsed,
    }


def drain_backlog(run_log: Path, *, dry: bool = False) -> dict[str, Any]:
    """FIFO 串行消化整个 backlog（持锁运行；异常/结束均释放锁并写 .done）。

    Args:
        run_log: 本次 run 日志路径；同名 ``.done`` 为完成摘要。
        dry: True 时不真派发，仅模拟销账（唤醒链演练 / 测试）。

    Returns:
        .done 摘要字典：``{started, completed, count, elapsed_s, items, run_log}``。
    """
    done_path = run_log.with_suffix(".done")
    lock = acquire_lock(orchestration_rel(run_log))
    if lock is None:
        _log(run_log, "[i] 另一 drain worker 持锁运行中，本进程退出（不重复消费）。")
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}

    t0 = time.time()
    summary: dict[str, Any] = {
        "started": _now(),
        "run_log": orchestration_rel(run_log),
        "dry": dry,
        "items": [],
    }
    _log(run_log, f"[▶] drain 启动（pid={os.getpid()}, dry={dry}）")
    # 刚获锁 → 此刻的 in_progress 必是已死 worker 的孤儿，回收为 pending
    orphans = backlog_reclaim_orphans()
    if orphans:
        summary["reclaimed"] = orphans
        _log(run_log, f"[i] 回收孤儿 in_progress 条目：{', '.join(orphans)}")
    try:
        while True:
            item = backlog_take_first()
            if item is None:
                break
            summary["items"].append(_drain_one(item, run_log, dry=dry))
    except Exception as exc:  # 记录后仍写 .done（后台进程无人值守，不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _log(run_log, f"[!] drain 异常中断：{summary['error']}")
    finally:
        summary["count"] = len(summary["items"])
        summary["completed"] = _now()
        summary["elapsed_s"] = round(time.time() - t0, 1)
        release_lock()
        try:
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log(run_log, f"[!] .done 写入失败：{exc}")
    _log(
        run_log,
        f"[√] drain 结束：处理 {summary['count']} 项，耗时 {summary['elapsed_s']}s "
        f"→ {orchestration_rel(done_path)}",
    )
    return summary


def cmd_drain_devops(args) -> int:
    """内部命令 ``_drain-devops``：持锁消化 backlog，落 run 日志 + .done。"""
    run_log = Path(args.run_log) if getattr(args, "run_log", None) else new_run_log()
    if not run_log.is_absolute():
        run_log = PROJECT_ROOT / run_log
    run_log.parent.mkdir(parents=True, exist_ok=True)
    summary = drain_backlog(run_log, dry=bool(getattr(args, "dry", False)))
    if summary.get("skipped") == "lock_busy":
        return 0
    return 0


# ---------------------------------------------------------------------------
# 唤醒（approve 调用）
# ---------------------------------------------------------------------------
def _drain_argv(run_log: Path, *, dry: bool) -> list[str]:
    """构造运行内部命令的 argv（``sys.executable -m`` → 从任意 cwd 可用）。"""
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_drain-devops",
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


def _spawn_detached(argv: list[str], *, env: dict[str, str], stdout: Any) -> None:
    """以分离后台进程启动 argv（Windows: DETACHED_PROCESS；POSIX: 新会话）。"""
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": subprocess.STDOUT,
        "cwd": str(PROJECT_ROOT),
        "env": env,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(argv, **kwargs)


def wake_devops(*, force_dry: bool | None = None) -> tuple[bool, str]:
    """approve 后尝试唤醒后台 drain worker。

    Args:
        force_dry: 覆盖干跑判定（None → 读 :data:`DRY_ENV` 环境变量）。

    Returns:
        ``(spawned, message)``。``spawned=False`` 表示锁忙未唤醒（message 说明本项由
        当前 drain 循环接手）；``True`` 表示已派生分离进程（message 含日志路径）。
    """
    st = lock_status()
    if st["state"] == "running":
        return False, "devops 正在处理队列，本项将由当前 drain 循环接手（不重复唤醒）。"
    DEVOPS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log()
    dry = bool(os.environ.get(DRY_ENV)) if force_dry is None else bool(force_dry)
    argv = _drain_argv(run_log, dry=dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(argv, env=env, stdout=logf)
    return True, (
        f"devops 已后台唤醒（日志 {orchestration_rel(run_log)}）；"
        "结果见 orch status / 该目录 .done 文件。"
    )
