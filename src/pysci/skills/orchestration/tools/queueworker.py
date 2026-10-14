"""成员队列 worker（README §12，M4）：每成员一个**非常驻** FIFO worker。

推广 devops drain 的生命周期模式——**抽取共用，不复制粘贴**（§纪律）：锁/墓碑/收尾证据/
run 名同秒顺延全部走 :mod:`drain` 的 ``WorkerPaths`` + ``acquire_lock``/``release_lock``/
``new_run_log``/``lock_status``（bencher 亦复用同一套），派生走 ``drain._spawn_detached``
（自带「测试进程不真派生」的护栏）。

流程：入队且无存活 worker → :func:`wake_member` 派生 ``_queue-worker <member>`` → worker 获锁
（**per-member 锁防双 worker**）→ 孤儿回收（``workq.reset_orphans``：刚获锁时的 in_progress
必是上任被杀留下的）→ 循环 { 出队（``workq.take_next``——**出队点单点**，m6 在此接
backpressure 延期检查，本环节只留位置）→ ``do_dispatch`` 执行 → 记结果 → 下一条 } →
**队空写 ``.done`` 摘要并退出销毁**。``kind=backlog-batch``（backlog-fifo-20261014：devops
backlog 批次）不走通用记账，改经 :func:`_settle_batch_item` → ``drain.settle_backlog_batch``
落 backlog 账务（销账/needs_leader/延期/系统性停止），完成后**续计划**下一批；devops 队列
排空/全局停止时由本 worker 发 ``QUEUE-EMPTY``/``CHANNEL-STOP``（完成唤醒的执行侧落点）。

review 附属（§3.1 第 3 件）：宿主 job（``review=True``）交付成功后**自动**投 reviewer 子 job；
review 子 job 收尾时尽力回填宿主记录。宿主的「完成」由 :func:`queues.completion` 派生。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from . import backpressure, events, queues, workq
from . import drain as _drain
from .drain import (
    BACKLOG_BATCH_KIND,
    DEFER_MAX_SLEEP_S,
    WorkerPaths,
    _consume_stop_request,  # drain 的实现——按 §纪律「抽取共用」直接引用，不复制
    _dry_settle_batch,
    _seed_tier,
    _spawn_detached,
    acquire_lock,
    lock_status,
    new_run_log,
    release_lock,
    settle_backlog_batch,
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


def _execute(item: dict[str, Any], *, dry: bool) -> tuple[str, dict[str, Any], Any]:
    """执行一条 job（复用 :func:`dispatch.do_dispatch`，会话策略沿派发惯例）。

    返回 ``(status, result, outcome)``；``outcome`` 原样带出（带宽类延期判据要用它的
    ``kind``/``error_text``，别在结果字典里丢掉分类原料）。
    """
    if dry:
        return workq.STATUS_DONE, {"dry": True}, None
    from .dispatch import do_dispatch  # 延迟导入：dispatch 引 runner 链

    out = do_dispatch(
        str(item.get("member")),
        text=str(item.get("task") or "") or None,
        task_file=str(item.get("task_file") or "") or None,
        session=str(item.get("session") or "latest"),
        effort=str(item.get("effort") or "") or None,
        model_tier=str(item.get("model") or "") or None,
        dirs=list(item.get("dirs") or []),
        plan=str(item.get("plan") or ""),
        step=str(item.get("step") or ""),
        max_turns=item.get("max_turns") or None,
        quiet=True,
        auto=True,
    )
    code = int(getattr(out, "code", 2))
    env = getattr(out, "envelope", None)
    # 结果字典要能**独立撑起 plan run 的交接报告**（直派退役后 plan run 不再拿
    # DispatchOutcome，而是从本条目还原）——分类/正文/验收/建议/时长/会话一并落盘。
    result: dict[str, Any] = {
        "kind": str(getattr(out, "kind", "")),
        "body": str(getattr(out, "body", ""))[:2000],
        "error": str(getattr(out, "error", ""))[:500],
        "task_file": str(item.get("task_file") or ""),
        "code": code,
        "member": str(getattr(out, "member", "") or item.get("member") or ""),
        "sid": str(getattr(out, "sid", "") or ""),
        "session_name": str(getattr(out, "session_name", "") or ""),
        "checks": [dict(c) for c in (getattr(out, "checks", None) or [])],
        "suggestion_id": str(getattr(out, "suggestion_id", "") or ""),
        "duration_ms": int(getattr(env, "duration_ms", 0) or 0) if env else 0,
    }
    return (workq.STATUS_DONE if code == 0 else workq.STATUS_FAILED), result, out


def _defer_plan(outcome: Any, item: dict[str, Any], now: float):
    """带宽类失败的延期判决（**复用** drain 的纯函数，不重写，§纪律）。

    成员队列与 devops backlog 走同一套分类/退避/双闸（``backpressure.plan_defer``）；
    devops 既有语义不动，此处只是把同一判据接到成员队列 worker 的出队处置上。
    """
    if outcome is None:
        return None
    from .drain import _bandwidth_defer_plan

    return _bandwidth_defer_plan(outcome, item, now)


def _notify(
    q: Path,
    item: dict[str, Any],
    status: str,
    result: dict[str, Any],
    member: str,
    *,
    root: Path | None = None,
) -> None:
    """worker 消化处**直发**统一事件（§3.2）——JOB-DONE / STEP-REVIEW-DONE。

    合并单元（§3.1）：带 ``review`` 的宿主 job 交付后**不**发 JOB-DONE，等 review 子 job
    收尾时按 :func:`queues.completion` 判「宿主+review 均收尾」才发**一次**
    ``STEP-REVIEW-DONE``。无 review 的普通 job 每件发 JOB-DONE。写失败 fail-open。
    """
    try:
        if str(item.get("kind")) == "review":
            host_id = str(item.get("parent") or "")
            host_member = str(item.get("parent_member") or "")
            state, why = queues.completion(host_id, host_member=host_member, root=root)
            if state == queues.COMPLETE:
                events.emit(
                    events.STEP_REVIEW_DONE,
                    subject=host_id,
                    detail=f"{host_member} 环节 + review 合并完成（{why}）",
                    action="orch events drain 后逐 plan 推进；orch status 看交付明细",
                )
            return
        if item.get("review"):
            state, why = queues.completion(
                str(item["id"]), host_member=member, root=root
            )
            if state == queues.COMPLETE:  # 无子 job 或子 job 已收尾（宿主失败/已完成）
                events.emit(
                    events.STEP_REVIEW_DONE,
                    subject=str(item["id"]),
                    detail=f"{member} 环节 + review 合并完成（{why}）",
                    action="orch events drain 后逐 plan 推进；orch status 看交付明细",
                )
            return  # AWAITING_REVIEW → 等 review 子 job 收尾时那一支发
        events.emit(
            events.JOB_DONE,
            subject=str(item["id"]),
            detail=f"{member} job 收尾：{status}（{result.get('kind', '')}）",
            action="orch events drain 后接续下一份工作；orch plan advance 推进",
        )
    except Exception:  # noqa: BLE001 - 事件面绝不阻塞生产（§0.6）
        pass


def _settle_batch_item(
    q: Path,
    member: str,
    item: dict[str, Any],
    outcome: Any,
    run_log: Path,
    summary: dict[str, Any],
    *,
    t0: float,
    dry: bool,
    error: str = "",
    root: Path | None = None,
) -> None:
    """backlog 批次条目的执行收尾：drain 账务（唯一实现）→ 队列条目终态。

    - done（销账完成）/ failed（needs_leader 或系统性停止）→ ``workq.finish``；
      systemic 时置 ``summary["stopped"]`` 由调用方停队（全局停止语义）。
    - deferred（带宽延期）→ 账务已把 backlog 条目写 ``defer_until``；本条目同步挂
      ``defer_until``（worker 睡到到期自动重取，**不停链**）。
    - dry（干跑演练）→ 模拟逐 id 销账（与计划器 dry 支同源）。
    - ``outcome is None``（**派发抛异常**，非 dry）→ 按 ``run_failed`` 合成结果结算——
      绝不能误走 dry 支把失败批次销账成 done（2026-10-14 子代理实测抓到的接线缺陷）。

    账务行落**本 worker 的 run 日志**（执行账与计划账分列，见 drain 模块 docstring）。
    """
    if dry:
        meta = dict(item.get("backlog") or {})
        seed = dict(meta.get("seed") or {})
        group = [dict(g) for g in (meta.get("group") or [])]
        results = _dry_settle_batch(seed, group, run_log)
        workq.finish(
            q, str(item["id"]), workq.STATUS_DONE, dict(results[0]) if results else {}
        )
        summary["items"].append(
            {"id": str(item["id"]), "status": "dry", "backlog_results": results}
        )
        return
    if outcome is None:
        from .dispatch import DispatchOutcome  # 延迟导入：dispatch 引 runner 链

        outcome = DispatchOutcome(
            code=2,
            kind="run_failed",
            body="",
            error=error or "批次派发异常",
            member=member,
        )
    work_status, results, stop = settle_backlog_batch(item, outcome, run_log, t0=t0)
    first = dict(results[0]) if results else {}
    if work_status == "deferred":
        delay = float(first.get("defer_s") or 0.0) or 60.0
        meta = dict(item.get("backlog") or {})
        seed = dict(meta.get("seed") or {})
        used = float(seed.get("defer_elapsed_s") or 0.0) + delay
        queues.defer(
            member,
            str(item["id"]),
            until=time.time() + delay,
            count=int(first.get("defer_count") or 1),
            elapsed_s=used,
            note="devops 批次带宽延期：等瞬态带宽恢复后重取（backlog 条目侧已同步 defer）",
            root=root,
        )
        summary["items"].append(
            {
                "id": str(item["id"]),
                "status": "deferred",
                "backlog_results": results,
            }
        )
        return
    status = workq.STATUS_DONE if work_status == "done" else workq.STATUS_FAILED
    workq.finish(q, str(item["id"]), status, first)
    summary["items"].append(
        {"id": str(item["id"]), "status": work_status, "backlog_results": results}
    )
    if stop:
        summary["stopped"] = stop
        summary["error"] = str(first.get("error", "") or "")


def run_worker(
    member: str, run_log: Path, *, dry: bool = False, root: Path | None = None
) -> dict[str, Any]:
    """FIFO 消化该成员队列（持锁运行；异常/结束均释放锁并写 ``.done`` 摘要）。

    §3.3：写侧接线——登记活跃会话 + 守护线程心跳（长 job 阻塞在 ``do_dispatch`` 里也有
    心跳、不被判死）；出队单点先过延期判据；带宽类失败回队挂 ``defer_until``，到期自动重取，
    双闸触顶则停止本队列并上报（发 CHANNEL-STOP 事件）。
    """
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
    t_worker = time.time()
    # devops 的 QUEUE-EMPTY = **backlog 消化完**的知会（旧 drain 语义）；plan 环节直派的
    # devops job 走通用 JOB-DONE 收账，不再叠发（task-20261014-devops-gap-d1 双通道对齐）。
    saw_batch = False
    # 死会话条目（注册表过期）与死队列条目（in_progress 孤儿）各清各的，不重复回收。
    backpressure.prune_stale(now=time.time())
    sess = backpressure.LiveSession(
        member,
        sid=run_log.stem,
        tier=_seed_tier({"member": member}),
        purpose="queue-worker",
    ).start()
    try:
        orphans = workq.reset_orphans(q)
        if orphans:
            summary["reclaimed"] = orphans
        while True:
            if _consume_stop_request(paths):
                summary["stopped"] = "cooperative"
                break
            item = queues.take_ready(member, root=root)  # 出队单点：过延期判据
            if item is None:
                wait = 0.0 if dry else queues.next_defer_s(member, root=root)
                if wait > 0:
                    _drain._sleep(min(wait, float(DEFER_MAX_SLEEP_S)))
                    continue
                break
            # 单条目异常**就地结账**（2026-10-14 实测：`task=` 参数名写错 → TypeError 逃出
            # 循环 → 条目永远卡 in_progress、plan run 干等到超时）。失败也要 finish：发
            # JOB-DONE 让 plan run/advance 拿到事件而不是空等；本队列其余条目继续。
            t0 = time.time()
            try:
                status, result, outcome = _execute(item, dry=dry)
            except Exception as exc:  # noqa: BLE001 - 一条目炸不得炸整队
                if str(item.get("kind")) == BACKLOG_BATCH_KIND:
                    # 派发异常：栈进 run 日志（旧 drain 的异常留痕语义——不留黑洞）
                    _drain._log_exception(run_log, exc, "批次派发异常")
                status = workq.STATUS_FAILED
                result = {
                    "kind": "run_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "code": 2,
                    "member": member,
                }
                outcome = None
                summary["item_error"] = result["error"]
            # devops backlog 批次（计划器入队）：账务=backlog 销账/needs_leader/延期/系统性
            # 停止，全部走 drain 的单一实现；本条目终态由结算结果映射（见 _settle_batch_item）。
            if str(item.get("kind")) == BACKLOG_BATCH_KIND:
                saw_batch = True
                _settle_batch_item(
                    q,
                    member,
                    item,
                    outcome,
                    run_log,
                    summary,
                    t0=t0,
                    dry=dry,
                    error=str(result.get("error") or ""),
                    root=root,
                )
                last = (
                    str(summary["items"][-1].get("status", ""))
                    if summary["items"]
                    else ""
                )
                if last != "deferred":  # 延期不是终态，不发收尾事件（同普通条目路径）
                    _notify(q, item, last, result, member, root=root)
                if summary.get("stopped"):  # 系统性全局停止：停队，剩余条目保持 pending
                    break
                # 停止优先于续计划：协作停止信号在场时不把更多 backlog 条目拉成在途
                # （「条目间隙停止：未取项仍 pending」语义的落点）。
                if _consume_stop_request(paths):
                    summary["stopped"] = "cooperative"
                    break
                # 续计划（backlog-fifo-20261014）：被卡批次里未选中的组包项留在 pending，
                # 旧 drain 循环会在下一轮把它们当新种子再取——执行搬到 worker 后由此补齐，
                # 否则「一项卡住不阻塞后续」退化成「等下一次 approve」。
                _drain.replan_from_backlog(dry=dry)
                continue
            plan = _defer_plan(outcome, item, time.time())
            if plan is not None and not plan.exhausted:
                nth = plan.attempt + 1
                used = float(item.get("defer_elapsed_s") or 0.0) + plan.delay_s
                queues.defer(
                    member,
                    str(item["id"]),
                    until=plan.until,
                    count=nth,
                    elapsed_s=used,
                    note=(
                        f"带宽延期第 {nth} 次（等 {plan.delay_s}s 后重试）"
                        "——带宽不足是瞬态，不换席不停链；预算触顶则停止本队列并上报"
                    ),
                    root=root,
                )
                summary["items"].append({"id": str(item["id"]), "status": "deferred"})
                continue
            if plan is not None and plan.exhausted:
                # 双闸触顶仍未愈 → 停止本队列并上报（沿 devops 语义；发渠道级停止事件）
                workq.finish(
                    q,
                    str(item["id"]),
                    workq.STATUS_FAILED,
                    {**result, "saturated": True, "exhausted": plan.reason},
                )
                summary["items"].append(
                    {"id": str(item["id"]), "status": "channel_saturated"}
                )
                summary["stopped"] = "channel_saturated"
                events.emit(
                    events.CHANNEL_STOP,
                    subject=str(item["id"]),
                    detail=f"{member} job 带宽延期触顶仍未愈（{plan.reason}）→ 停止本队列",
                    action="渠道恢复后重新入队/唤醒；orch status 看延期读数",
                )
                break
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
            _notify(q, item, status, result, member, root=root)
        workq.prune(q)
    except Exception as exc:  # noqa: BLE001 - 异常也要写 .done（不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sess.stop()  # 退场注销活跃会话（先于释放锁，读数面不留幽灵）
        release_lock(run_log, paths)
        summary["completed"] = workq._now()
        summary["count"] = len(summary["items"])
        summary["elapsed_s"] = round(time.time() - t_worker, 1)
        # blocked 成因档位分列（backlog 20261013-191213-devops 语义迁移到执行账）：
        # 计划器 items 已是「queued」无 reason，档位只在批次的账务行里——从 backlog_results
        # 聚合进 .done，供 orch status 分列「待裁决 / 待排期」。
        by_reason: dict[str, int] = {}
        for entry in summary["items"]:
            for row in entry.get("backlog_results") or []:
                r = str(row.get("reason") or "")
                if r:
                    by_reason[r] = by_reason.get(r, 0) + 1
        if by_reason:
            summary["by_reason"] = by_reason
        done = run_log.with_suffix(".done")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # devops 完成唤醒（backlog-fifo-20261014 迁移）：执行只留队列 worker 一份实现后，
        # 「消化完」的消费事实由**执行侧**发——统一事件流一条 QUEUE-EMPTY（旧 drain.finally
        # 的职责平移至此）；仅在消化过 backlog 批次的 run 发（plan 环节直派走 JOB-DONE 收账，
        # 双通道对齐见 task-20261014-devops-gap-d1）；带宽延期触顶回升的全局停止另发一条
        # 渠道级停止事件。写失败 fail-open。
        if member == "devops" and saw_batch:
            try:
                events.note_queue_empty(
                    "devops",
                    subject=run_log.stem,
                    count=summary["count"],
                    elapsed_s=summary.get("elapsed_s", "?"),
                    stopped=str(summary.get("stopped", "") or ""),
                )
            except Exception:  # noqa: BLE001 - 事件面绝不阻塞 worker（§0.6）
                pass
            if str(summary.get("stopped", "") or "") == "channel_saturated":
                try:
                    events.emit(
                        events.CHANNEL_STOP,
                        subject=run_log.stem,
                        detail=(
                            "devops 带宽延期触顶仍未愈 → 全局停止（"
                            + str(summary.get("error", "") or "")[:80]
                            + "）"
                        ),
                        action="渠道恢复后重新入队/唤醒；orch status 看延期读数",
                    )
                except Exception:  # noqa: BLE001
                    pass
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
