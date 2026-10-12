"""bencher 工作队列与进程生命周期（REFACTOR v1.0 §8；E4，backlog 20261012-120002-leader）。

**为什么要有它**：§3 的套件运行/基线捕获/副本 A/B 都是「跑一次要几分钟到几十分钟、
中途没有对话价值」的机械活。放在组长侧就变成组长手动守着 ``pysci-bench``；放在 devops
的 backlog 里就变成基础设施单。§8 的裁决是第三条路：**专职队列 + 非常驻 worker**——
非空拉起、消化完销毁，完成项进 ``orch status`` 的「待查看」面（§5.1-B 择机查看型通知，
不即时打断提请人）。

三块机制与 :mod:`drain` **同构但不复制**（§0.9）：锁 / 墓碑 / ``.done`` 收尾证据 / run
名同秒顺延 / 协作停止都是被三次事故打磨出来的原语，本模块通过 :class:`~drain.WorkerPaths`
把落点换成 bencher 的一套（``state/bencher.lock``、``state/bencher-runs/``、
``state/bencher.cancel``）后复用它们。差别只在**队列**：devops 的队列是 backlog.json
（组长审批面），这里的队列是 ``state/bencher/queue.json``（测量任务面，纯运行态）。

入队项四型（:data:`ITEM_KINDS`）：

- ``run`` —— 跑一批套件案（``cases`` 列表），复用 :func:`eval.run_case` 且
  ``runner=bencher``：读数进 ``state/ledger/evals.jsonl``，与组长抽检通道
  （``orch eval run``，``runner=leader``）在台账里可分开。
- ``capture`` —— 基线捕获（§3：记录 ref 的 commit + 归档 pods 的 gitignored harness 面）。
- ``ab`` —— 副本 A/B：两 ref 各建一份 git worktree 副本、各自跑案内案、出二元判决；
  窗口期污染 → ``inconclusive`` + ``escalate=twin``（§5.5）。
- ``check`` —— 组长或 inspector 指派的核查跑：真派一跳给 bencher 组员（它有自己的
  ``pysci-bench`` 工具面），只记录交付正文，不产出台账行。

内部命令 ``_bench-worker`` 下划线前缀（勿手调）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import eval as _eval
from . import workq
from .drain import (
    DRY_ENV as _DRY_ENV,
)
from .drain import (
    WorkerPaths,
    _consume_stop_request,
    _log,
    _now,
    _spawn_detached,
    acquire_lock,
    cancel_status,
    lock_status,
    new_run_log,
    release_lock,
    request_stop,
)
from .registry import orchestration_rel

#: 测量任务队列（纯运行态：项 = 一次测量动作，永久审计在 evals.jsonl 与 ab 判决文件）。
QUEUE_PATH: Path = ORCH_STATE_ROOT / "bencher" / "queue.json"

#: bencher worker 的生命周期落点（锁语义与 devops drain 完全共用一套原语）。
BENCH_PATHS = WorkerPaths(
    lock=ORCH_STATE_ROOT / "bencher.lock",
    runs=ORCH_STATE_ROOT / "bencher-runs",
    cancel=ORCH_STATE_ROOT / "bencher.cancel",
)

#: 入队项类型（§8 任务面）。
ITEM_KINDS: tuple[str, ...] = ("run", "capture", "ab", "check")

#: 队列项终态：``done``/``failed`` 都进「待查看」面，直到提请人 ``bench-seen``。
#: 状态词表与队列形状统一由 :mod:`workq` 定义（E4 时这套读写在本模块内部，E5a 抽库后
#: 这里只留测量业务的薄封装——两处各写一份状态机就是双源漂移的开始）。
TERMINAL_STATUSES: tuple[str, ...] = workq.TERMINAL_STATUSES

_ITEM_SEP = "-"


# ---------------------------------------------------------------------------
# 队列（薄封装：全部原语见 :mod:`workq`）
# ---------------------------------------------------------------------------
def queue_load() -> list[dict[str, Any]]:
    """读队列全部条目（按入队序）；文件缺失返回空表。"""
    return workq.load(QUEUE_PATH)


def queue_save(items: list[dict[str, Any]]) -> None:
    """整体重写队列（原子 replace）。"""
    workq.save(QUEUE_PATH, items)


def new_item_id(kind: str, *, stamp: str | None = None) -> str:
    """条目 id ``bench-<ts>-<kind>``（同秒重复则补 ``-002``，id 唯一是销账依据）。

    ``stamp`` 参数是测试注入口：脚本里不许用 ``datetime.now()`` 决定可复现的命名。
    """
    return workq.new_item_id(QUEUE_PATH, "bench", kind, stamp=stamp, sep=_ITEM_SEP)


def enqueue(
    kind: str,
    *,
    cases: list[str] | None = None,
    label: str = "",
    tier: str = "",
    ref: str = "",
    ref_a: str = "",
    ref_b: str = "",
    brief: str = "",
    requested_by: str = "leader",
    summary: str = "",
    stamp: str | None = None,
    backlog_id: str = "",
) -> dict[str, Any]:
    """入队一项测量任务（FIFO 尾部追加），返回该条目。

    Args:
        kind: :data:`ITEM_KINDS` 之一。
        cases: 案 id 列表（``run``/``ab``/``capture`` 用；空 = 全套）。
        label: 本轮标签（进 ``evals.jsonl`` 的分组面，也是 A/B 判决文件名）。
        tier: 覆盖案卷档位（空=案内 ``tier:``，压测默认 minimum）。
        ref: ``capture`` 的基线 ref（空=HEAD）。
        ref_a / ref_b: ``ab`` 两端 ref（旧→新，判决读作「a 改善到 b」）。
        brief: ``check`` 型的核查任务书正文（真派一跳给 bencher）。
        requested_by: 提请人（组长或 inspector），「待查看」面按此归位通知。
        summary: 人读的一行摘要（status 展示用）。
        stamp: 命名时间戳覆盖（测试注入）。
        backlog_id: 关联的 backlog 条目 id（E5b §5.6 交付后核验：``ab`` 判决落地到该
            条目的账——销账 / 自动立案 / 挂起）。空 = 普通测量，判决不外溢。

    Raises:
        ValueError: kind 不在词表，或 ``ab`` 少一侧 ref，或 ``capture`` 缺 case。
    """
    if kind not in ITEM_KINDS:
        raise ValueError(f"未知队列项类型 {kind!r}；可用：{', '.join(ITEM_KINDS)}")
    if kind == "ab" and not (ref_a and ref_b):
        raise ValueError("ab 项需要 --ref-a 与 --ref-b 两端（二元对局没有单边）")
    if kind == "capture" and not cases:
        raise ValueError(
            "capture 项需要 --case（基线按案归档：state/evals/baselines/<case>/）"
        )
    item: dict[str, Any] = {
        "id": new_item_id(kind, stamp=stamp),
        "kind": kind,
        "cases": list(cases or []),
        "label": label,
        "tier": tier,
        "ref": ref,
        "ref_a": ref_a,
        "ref_b": ref_b,
        "brief": brief,
        "requested_by": requested_by,
        "summary": summary or _default_summary(kind, cases or [], label),
        "status": "pending",
        "enqueued_at": _now(),
    }
    if backlog_id:
        item["backlog_id"] = backlog_id
    return workq.append(QUEUE_PATH, item)


def _default_summary(kind: str, cases: list[str], label: str) -> str:
    """无人填写 summary 时给 status 一行可读摘要（不猜语义，只报参数）。"""
    cs = ",".join(cases) if cases else "全套"
    return f"{kind}: {cs}" + (f" label={label}" if label else "")


def _mutate(item_id: str, **fields: Any) -> dict[str, Any] | None:
    """按 id 改写字段并落盘，返回改后的条目（找不到返回 None）。"""
    return workq.mutate(QUEUE_PATH, item_id, **fields)


def take_next() -> dict[str, Any] | None:
    """取队首 pending 项并标记 ``in_progress``（无则 None）。"""
    return workq.take_next(QUEUE_PATH)


def reset_orphans() -> list[str]:
    """把 ``in_progress`` 复位为 pending——刚获锁时它们是已死 worker 的孤儿。"""
    return workq.reset_orphans(QUEUE_PATH)


def finish(item_id: str, status: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """写终态（``done``/``failed``）与结构化结果；``reviewed_at`` 留空 = 待查看。"""
    return workq.finish(QUEUE_PATH, item_id, status, result)


def pending_review() -> list[dict[str, Any]]:
    """「待查看结果」面：已终态但提请人尚未确认的条目（§5.1-B 择机查看，不即时唤醒）。"""
    return workq.pending_review(QUEUE_PATH)


def mark_seen(item_id: str) -> dict[str, Any] | None:
    """提请人确认已查看（从「待查看」面移除；条目本身留在队列作审计，直到下次净化）。"""
    return workq.mark_seen(QUEUE_PATH, item_id)


def prune_seen(keep: int = workq.DEFAULT_KEEP) -> int:
    """裁掉已查看的历史条目，只留最近 ``keep`` 条（队列文件不许无界增长）。"""
    return workq.prune(QUEUE_PATH, keep)


def queue_counts() -> dict[str, int]:
    """按状态计数（status 一行读数）。"""
    return workq.counts(QUEUE_PATH)


# ---------------------------------------------------------------------------
# 条目执行
# ---------------------------------------------------------------------------
def _run_kind_run(item: dict[str, Any], run_log: Path, *, dry: bool) -> dict[str, Any]:
    """``run`` 项：逐案 :func:`eval.run_case`（runner=bencher），返回逐案读数。"""
    cases = _eval.load_cases()
    wanted = [str(c) for c in item.get("cases") or []]
    if wanted:
        cases = [c for c in cases if c.case_id in wanted]
        missing = sorted(set(wanted) - {c.case_id for c in cases})
        if missing:
            raise ValueError(f"案卷不存在：{missing}")
    rows: dict[str, Any] = {}
    quota = False
    for case in cases:
        if dry:
            rows[case.case_id] = {"verdict": "dry", "num_turns": 0}
            _log(run_log, f"[dry] {case.case_id} 演练（不实派）")
            continue
        row = _eval.run_case(
            case,
            tier=str(item.get("tier") or ""),
            label=str(item.get("label") or ""),
            runner=_eval.RUNNER_BENCHER,
        )
        body = " ".join(row.reasons)
        quota = quota or "quota_exhausted" in body
        rows[row.case] = {
            "verdict": row.verdict,
            "num_turns": row.num_turns,
            "max_turns": row.max_turns,
            "duration_s": round(row.duration_ms / 1000, 1),
            "reasons": row.reasons,
        }
        _log(
            run_log,
            f"[{'√' if row.verdict == 'pass' else '✗'}] {row.case}: {row.verdict} "
            f"turns={row.num_turns}/{row.max_turns} dur={row.duration_ms / 1000:.0f}s",
        )
    return {"cases": rows, "quota_exhausted": quota}


def _run_kind_capture(
    item: dict[str, Any], run_log: Path, *, dry: bool
) -> dict[str, Any]:
    """``capture`` 项：基线捕获（§3 副本机制的 gitignored 面归档）。"""
    from .replica import (
        capture_baseline,  # 延迟导入：replica 依赖 git 子进程，非队列常驻路径
    )

    case = str((item.get("cases") or [""])[0])
    ref = str(item.get("ref") or "HEAD")
    if dry:
        _log(run_log, f"[dry] capture {case or '（未指定案）'} @{ref} 演练")
        return {"case": case, "ref": ref, "dry": True}
    return capture_baseline(case, ref=ref)


def _run_kind_ab(item: dict[str, Any], run_log: Path, *, dry: bool) -> dict[str, Any]:
    """``ab`` 项：副本 A/B 二元判决（含窗口期污染闸）+ 交付后核验落地。"""
    from .replica import run_ab

    label = str(item.get("label") or item["id"])
    if dry:
        _log(run_log, f"[dry] ab {item.get('ref_a')}..{item.get('ref_b')} 演练")
        return {"label": label, "verdict": "dry"}
    doc = run_ab(
        str(item["ref_a"]),
        str(item["ref_b"]),
        [str(c) for c in item.get("cases") or []],
        label=label,
        tier=str(item.get("tier") or ""),
        log=lambda m: _log(run_log, m),
    )
    bid = str(item.get("backlog_id") or "")
    if bid:
        # §5.6 核验判决直接落到提案账上（无人再裁决）：测量已完成，落地失败只该告警，
        # 不该把整项判失败——重跑 A/B 的代价远高于补一次销账备注。
        from .verdict import (  # 延迟导入：verdict 顶层依赖本模块
            apply_ab_verdict,
        )

        try:
            out = apply_ab_verdict(item, doc)
            doc["post_verify"] = out
            _log(
                run_log,
                f"[√] 交付后核验 {bid}：{out['action']}（verdict={doc.get('verdict') or '空'}）",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            _log(
                run_log,
                f"[!] 交付后核验落地失败（{bid}）：{type(exc).__name__}: {exc}"
                "——判决读数已落 ab/ 目录，需人工销账",
            )
            doc["post_verify_error"] = f"{type(exc).__name__}: {exc}"
    return doc


def _run_kind_check(
    item: dict[str, Any], run_log: Path, *, dry: bool
) -> dict[str, Any]:
    """``check`` 项：真派一跳给 bencher 组员（组长/inspector 指派的核查跑）。"""
    from .dispatch import do_dispatch  # 延迟导入：与 drain 同样的环规避

    brief = str(item.get("brief") or "").strip()
    if not brief:
        raise ValueError("check 项缺 brief（核查任务书正文）")
    if dry:
        _log(run_log, "[dry] check 演练（不派组员）")
        return {"kind": "dry"}
    out = do_dispatch(
        "bencher",
        text=brief,
        slug=str(item["id"]),
        session="latest",
        quiet=True,
    )
    return {
        "delivery": out.kind,
        "code": out.code,
        "body": (out.body or "")[:4000],
        "quota_exhausted": out.kind == "quota_exhausted",
    }


_RUNNERS = {
    "run": _run_kind_run,
    "capture": _run_kind_capture,
    "ab": _run_kind_ab,
    "check": _run_kind_check,
}


def _headline(item: dict[str, Any], result: dict[str, Any]) -> str:
    """status「待查看」面的一句话：各 kind 的判决口径不同，但都必须一眼可读。"""
    if result.get("error"):
        return str(result["error"])[:80]
    kind = str(item.get("kind"))
    if kind == "run":
        cases = result.get("cases") or {}
        ok = sum(1 for r in cases.values() if r.get("verdict") == "pass")
        return f"通过 {ok}/{len(cases)}"
    if kind == "ab":
        return (
            f"{result.get('verdict', '?')}"
            f"（{result.get('ref_a', '?')}→{result.get('ref_b', '?')}）"
            + ("，escalate=twin" if result.get("escalate") else "")
        )
    if kind == "capture":
        return f"基线已捕获 {result.get('case', '?')} @{result.get('ref', 'HEAD')}"
    if kind == "check":
        return f"组员交付=<{result.get('delivery', '?')}>"
    return ""


def run_item(
    item: dict[str, Any], run_log: Path, *, dry: bool = False
) -> dict[str, Any]:
    """执行一个队列项，写终态，返回结果字典（附 ``status`` 字段供 .done 摘要）。

    异常不外抛：后台 worker 无人接盘，失败必须变成条目上的 ``failed`` + 原因，
    队列继续下一项（一项坏不许拖垮整轮）。
    """
    t0 = time.time()
    runner = _RUNNERS.get(str(item.get("kind")))
    if runner is None:
        result = {"error": f"未知类型 {item.get('kind')!r}", "status": "failed"}
    else:
        try:
            result = runner(item, run_log, dry=dry)
            result["status"] = "failed" if result.get("error") else "done"
        except Exception as exc:  # noqa: BLE001 — 后台无人值守：异常即终态，不留黑洞
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "failed",
            }
            _log(run_log, f"[!] {item.get('id')} 失败：{result['error']}")
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["headline"] = _headline(item, result)
    finish(str(item["id"]), result["status"], result)
    return result


# ---------------------------------------------------------------------------
# worker 循环
# ---------------------------------------------------------------------------
def bench_drain(run_log: Path, *, dry: bool = False) -> dict[str, Any]:
    """持锁 FIFO 消化整个测量队列；队列空即释放锁退出（非常驻，§8）。

    Returns:
        ``.done`` 摘要：``{started, completed, count, elapsed_s, items, run_log}``；
        锁忙附 ``skipped="lock_busy"``；协作停止附 ``stopped="cooperative"``；
        额度类系统性故障附 ``stopped="quota_exhausted"``。
    """
    done_path = run_log.with_suffix(".done")
    lock = acquire_lock(orchestration_rel(run_log), BENCH_PATHS)
    if lock is None:
        _log(run_log, "[i] 另一 bencher worker 持锁运行中，本进程退出（不重复消费）。")
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}

    t0 = time.time()
    summary: dict[str, Any] = {
        "started": _now(),
        "run_log": orchestration_rel(run_log),
        "dry": dry,
        "items": [],
    }
    _log(run_log, f"[▶] bencher worker 启动（pid={os.getpid()}, dry={dry}）")
    orphans = reset_orphans()
    if orphans:
        summary["reclaimed"] = orphans
        _log(run_log, f"[i] 回收孤儿在途条目：{', '.join(orphans)}")
    try:
        while True:
            if _consume_stop_request(BENCH_PATHS):
                summary["stopped"] = "cooperative"
                _log(run_log, "[s] 收到协作停止信号，干净退出（条目间隙，无在途消费）")
                break
            item = take_next()
            if item is None:
                break
            _log(run_log, f"[→] 消化 {item['id']}: {item.get('summary', '')[:60]}")
            result = run_item(item, run_log, dry=dry)
            summary["items"].append(
                {
                    "id": item["id"],
                    "kind": item.get("kind"),
                    "status": result.get("status"),
                    "elapsed_s": result.get("elapsed_s"),
                }
            )
            if result.get("quota_exhausted"):
                summary["stopped"] = "quota_exhausted"
                summary["error"] = (
                    "系统性故障：模型渠道额度耗尽（quota_exhausted）——全局停止，"
                    "剩余条目保持 pending 待渠道恢复"
                )
                _log(run_log, "[Q] 全局停止：额度耗尽，剩余条目保持 pending，释放锁")
                break
    except Exception as exc:  # noqa: BLE001 — 记录后仍写 .done（不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _log(run_log, f"[!] bencher worker 异常中断：{summary['error']}")
    finally:
        summary["count"] = len(summary["items"])
        summary["completed"] = _now()
        summary["elapsed_s"] = round(time.time() - t0, 1)
        release_lock(run_log, BENCH_PATHS)
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
        f"[√] bencher worker 结束：处理 {summary['count']} 项，"
        f"耗时 {summary['elapsed_s']}s → {orchestration_rel(done_path)}",
    )
    return summary


def _worker_argv(run_log: Path, *, dry: bool) -> list[str]:
    """构造内部 worker 的 argv（``sys.executable -m`` → 从任意 cwd 可用）。"""
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_bench-worker",
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


def wake_bencher(*, force_dry: bool | None = None) -> tuple[bool, str]:
    """拉起 bencher worker（空闲才派生；锁忙则说明本轮会接手队列）。

    Args:
        force_dry: 覆盖干跑判定（None → 读 :data:`~drain.DRY_ENV` 环境变量）。

    Returns:
        ``(spawned, message)``。与 :func:`drain.wake_devops` 的区别：**不附** watch 挂载
        指令——§8 明写「择机查看型通知，不即时打断」，测量完成走 ``orch status``
        的待查看面，watch 事件 ``BENCH-DONE`` 属可选挂载且默认不挂。
    """
    st = lock_status(BENCH_PATHS)
    if st["state"] == "running":
        return (
            False,
            "bencher worker 正在消化队列，本项由当前循环接手（不重复唤醒）。"
            "结果稍后见 orch status 的「待查看」面。",
        )
    BENCH_PATHS.runs.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log(BENCH_PATHS)
    dry = bool(os.environ.get(_DRY_ENV)) if force_dry is None else bool(force_dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(_worker_argv(run_log, dry=dry), env=env, stdout=logf)
    return True, (
        f"bencher worker 已后台唤醒（日志 {orchestration_rel(run_log)}）；"
        "完成项见 orch status「待查看」面，不即时打断。"
    )


#: 干跑开关：与 devops drain 同一个环境变量（演练语义应当一致，不给两套旋钮）。


def bench_status() -> dict[str, Any]:
    """status 用的 bencher 面读数：锁态 / 最近 .done / 队列计数 / 待查看条目。"""
    return {
        "lock": lock_status(BENCH_PATHS),
        "cancel": cancel_status(BENCH_PATHS),
        "counts": queue_counts(),
        "review": pending_review(),
    }


# ---------------------------------------------------------------------------
# CLI 面（orch bench-*）
# ---------------------------------------------------------------------------
def cmd_bench_add(args: argparse.Namespace) -> int:
    """``orch bench-add``：注入一个测量任务项（``--start`` 则顺手唤醒 worker）。"""
    try:
        item = enqueue(
            args.kind,
            cases=[c for c in (args.cases or "").replace("，", ",").split(",") if c],
            label=args.label or "",
            tier=args.tier or "",
            ref=args.ref or "",
            ref_a=getattr(args, "ref_a", "") or "",
            ref_b=getattr(args, "ref_b", "") or "",
            brief=args.brief or "",
            requested_by=args.requested_by or "leader",
            summary=args.summary or "",
        )
    except ValueError as exc:
        print(f"[✗] 入队被拒：{exc}")
        return 2
    print(f"[√] 已入队 {item['id']}（{item['kind']}）：{item['summary']}")
    if args.start:
        spawned, msg = wake_bencher()
        print(f"[{'▶' if spawned else 'i'}] {msg}")
    else:
        print("    [NEXT] `orch bench-up` 拉起 worker（或 --start 入队即拉起）。")
    return 0


def cmd_bench_up(args: argparse.Namespace) -> int:
    """``orch bench-up``：拉起后台 worker（dry-run 演练用 ``--dry``）。"""
    spawned, msg = wake_bencher(force_dry=True if args.dry else None)
    print(f"[{'▶' if spawned else 'i'}] {msg}")
    return 0


def cmd_bench_queue(args: argparse.Namespace) -> int:
    """``orch bench-queue``：列队列条目（``--all`` 含已查看的历史）。"""
    items = queue_load()
    if not items:
        print("  （队列为空：orch bench-add 注入一项）")
        return 0
    shown = (
        items
        if args.all
        else [
            i
            for i in items
            if i.get("status") not in TERMINAL_STATUSES or not i.get("reviewed_at")
        ]
    )
    for it in shown:
        seen = "·已查看" if it.get("reviewed_at") else ""
        print(
            f"  [{it.get('status', '?'):<11}] {it['id']} "
            f"{str(it.get('summary', ''))[:56]}{seen}"
        )
        res = it.get("result") or {}
        if res.get("error"):
            print(f"        [!] {str(res['error'])[:120]}")
    print(
        f"  计数：{queue_counts()}  锁态：{lock_status(BENCH_PATHS)['state']}"
        f"  （--all 看历史）"
    )
    if not args.all:
        drop = prune_seen()
        if drop:
            print(
                f"  [i] 已裁掉 {drop} 条历史条目（保留最近 20 条，队列文件不许无界增长）"
            )
    return 0


def cmd_bench_seen(args: argparse.Namespace) -> int:
    """``orch bench-seen <id>``：提请人确认已查看（从「待查看」面摘掉）。"""
    if mark_seen(args.id) is None:
        print(f"[✗] 队列里没有 {args.id}")
        return 2
    print(f"[√] {args.id} 已标记查看")
    return 0


def cmd_bench_stop(args: argparse.Namespace) -> int:
    """``orch bench-stop``：请求 worker 协作停止（条目间隙干净退出）。"""
    del args
    ok, msg = request_stop(BENCH_PATHS)
    print(f"[{'s' if ok else 'i'}] {msg}")
    return 0 if ok else 1


def cmd_bench_worker(args: argparse.Namespace) -> int:
    """内部命令 ``_bench-worker``：持锁消化测量队列，落 run 日志 + .done。"""
    run_log = (
        Path(args.run_log)
        if getattr(args, "run_log", None)
        else new_run_log(BENCH_PATHS)
    )
    if not run_log.is_absolute():
        run_log = PROJECT_ROOT / run_log
    run_log.parent.mkdir(parents=True, exist_ok=True)
    bench_drain(run_log, dry=bool(getattr(args, "dry", False)))
    return 0
