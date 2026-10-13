"""成员队列 worker（REFACTOR.md §3.1，M4）：每成员一个**非常驻** FIFO worker。

推广 devops drain 的生命周期模式——**抽取共用，不复制粘贴**（§纪律）：锁/墓碑/收尾证据/
run 名同秒顺延全部走 :mod:`drain` 的 ``WorkerPaths`` + ``acquire_lock``/``release_lock``/
``new_run_log``/``lock_status``（bencher 亦复用同一套），派生走 ``drain._spawn_detached``
（自带「测试进程不真派生」的护栏）。

流程：入队且无存活 worker → :func:`wake_member` 派生 ``_queue-worker <member>`` → worker 获锁
（**per-member 锁防双 worker**）→ 孤儿回收（``workq.reset_orphans``：刚获锁时的 in_progress
必是上任被杀留下的）→ 循环 { 出队（``workq.take_next``——**出队点单点**，m6 在此接
backpressure 延期检查，本环节只留位置）→ ``do_dispatch`` 执行 → 记结果 → 下一条 } →
**队空写 ``.done`` 摘要并退出销毁**。

review 附属（§3.1 第 3 件）：宿主 job（``review=True``）交付成功后**自动**投 reviewer 子 job；
review 子 job 收尾时尽力回填宿主记录。宿主的「完成」由 :func:`queues.completion` 派生。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from . import queues, workq
from .drain import (
    WorkerPaths,
    _consume_stop_request,  # drain 的实现——按 §纪律「抽取共用」直接引用，不复制
    _spawn_detached,
    acquire_lock,
    lock_status,
    new_run_log,
    release_lock,
)
from .registry import orchestration_rel

#: 成员 worker 的运行态根（锁/停止信号在 ``state/queues/`` 旁，run 日志独立成树）。
QUEUE_WORKER_ROOT: Path = ORCH_STATE_ROOT / "queues-runs"


def member_paths(member: str, *, root: Path | None = None) -> WorkerPaths:
    """该成员 worker 的生命周期落点（锁 / run 目录 / 协作停止信号）。

    每次**现读**模块常量，便于测试 monkeypatch 隔离面（同 drain 的理由）。
    """
    base = (root or ORCH_STATE_ROOT) / "queues"
    runs = (root or ORCH_STATE_ROOT) / "queues-runs" / member
    return WorkerPaths(
        lock=base / f"{member}.lock",
        runs=runs,
        cancel=base / f"{member}.cancel",
    )


def _worker_argv(member: str, run_log: Path, *, dry: bool) -> list[str]:
    """内部命令 argv（``sys.executable -m`` → 从任意 cwd 可用）。"""
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_queue-worker",
        member,
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


def wake_member(
    member: str, *, dry: bool = False, root: Path | None = None
) -> tuple[bool, str]:
    """入队后拉起该成员的 worker；锁忙（已有存活 worker）则不重复派生。

    Returns:
        ``(spawned, message)``——``False`` = 已有 worker 在跑，本条由当前循环接手。
    """
    paths = member_paths(member, root=root)
    st = lock_status(paths)
    if st.get("state") == "running":
        return False, f"{member} worker 正在跑队列，本条由当前循环接手（不重复派生）。"
    paths.runs.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log(paths)
    argv = _worker_argv(member, run_log, dry=dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(argv, env=env, stdout=logf)
    return True, f"{member} worker 已后台唤醒（日志 {orchestration_rel(run_log)}）"


def _execute(item: dict[str, Any], *, dry: bool) -> tuple[str, dict[str, Any]]:
    """执行一条 job（复用 :func:`dispatch.do_dispatch`，会话策略沿派发惯例）。"""
    if dry:
        return workq.STATUS_DONE, {"dry": True}
    from .dispatch import do_dispatch  # 延迟导入：dispatch 引 runner 链

    out = do_dispatch(
        str(item.get("member")),
        text=str(item.get("task") or "") or None,
        task=str(item.get("task_file") or "") or None,
        session=str(item.get("session") or "latest"),
        effort=str(item.get("effort") or "") or None,
        model_tier=str(item.get("model") or "") or None,
        dirs=list(item.get("dirs") or []),
        plan=str(item.get("plan") or ""),
        step=str(item.get("step") or ""),
        quiet=True,
        auto=True,
    )
    code = int(getattr(out, "code", 2))
    result: dict[str, Any] = {
        "kind": str(getattr(out, "kind", "")),
        "body": str(getattr(out, "body", ""))[:2000],
        "error": str(getattr(out, "error", ""))[:500],
        "task_file": str(item.get("task_file") or ""),
    }
    return (workq.STATUS_DONE if code == 0 else workq.STATUS_FAILED), result


def run_worker(
    member: str, run_log: Path, *, dry: bool = False, root: Path | None = None
) -> dict[str, Any]:
    """FIFO 消化该成员队列（持锁运行；异常/结束均释放锁并写 ``.done`` 摘要）。"""
    paths = member_paths(member, root=root)
    q = queues.queue_path(member, root=root)
    if acquire_lock(orchestration_rel(run_log), paths) is None:
        return {"skipped": "lock_busy"}
    summary: dict[str, Any] = {
        "member": member,
        "started": workq._now(),
        "items": [],
        "run_log": str(run_log),
    }
    try:
        orphans = workq.reset_orphans(q)
        if orphans:
            summary["reclaimed"] = orphans
        while True:
            if _consume_stop_request(paths):
                summary["stopped"] = "cooperative"
                break
            item = workq.take_next(q)  # **出队点单点**（m6 将在此接延期检查）
            if item is None:
                break
            status, result = _execute(item, dry=dry)
            workq.finish(q, str(item["id"]), status, result)
            summary["items"].append({"id": str(item["id"]), "status": status})
            if status == workq.STATUS_DONE:
                if str(item.get("kind")) == "review":
                    queues.backfill_host(item, result, root=root)
                else:
                    child = queues.spawn_review(item, root=root)
                    if child:
                        summary["items"].append(
                            {"id": str(child["id"]), "status": "queued"}
                        )
        workq.prune(q)
    except Exception as exc:  # noqa: BLE001 - 异常也要写 .done（不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        release_lock(run_log, paths)
        summary["completed"] = workq._now()
        summary["count"] = len(summary["items"])
        done = run_log.with_suffix(".done")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def cmd_queue_worker(args: Any) -> int:
    """内部命令 ``_queue-worker <member>``：持锁消化该成员队列，落 run 日志 + .done。"""
    run_log = Path(args.run_log) if getattr(args, "run_log", None) else None
    if run_log is None:
        paths = member_paths(str(args.member))
        paths.runs.mkdir(parents=True, exist_ok=True)
        run_log = new_run_log(paths)
    elif not run_log.is_absolute():
        run_log = Path(os.getcwd()) / run_log
    run_worker(str(args.member), run_log, dry=bool(getattr(args, "dry", False)))
    return 0
