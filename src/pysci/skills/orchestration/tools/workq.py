"""成员工作队列的通用原语（README §1「共享逻辑做库，不做第二套实现」）。

bencher 的测量队列（§8）与 inspector 的巡检队列（§7）是**同一种东西**：FIFO 运行态 JSON
文件，条目走 ``pending → in_progress → done|failed`` 状态机。E4 把这套读写写在
:mod:`~.benchd` 内部；E5a 若照抄一份，就会有两处孤儿回收、两处同秒改名顺延、两处裁剪——
正是本项目反复踩的「双源漂移」族（同一语义两处实现，迟早只改一处）。

本模块只放**与业务无关的队列形状**：类型词表、条目字段、执行分派、判决语义都留在各成员
模块里（``benchd`` 管测量、``inspectd`` 管巡检）。队列文件是纯运行态：永久审计在各成员
自己的台账面（``ledger/evals.jsonl``、``inspector/findings.jsonl``），不在这里。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

#: 条目状态机的四个态（成员模块共用同一套词，status 面才读得出同一种「在途」）。
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
TERMINAL_STATUSES: tuple[str, ...] = (STATUS_DONE, STATUS_FAILED)

#: 裁剪默认保留条数（队列文件不许无界增长，但也不能裁掉正在被 status 面引用的条目）。
DEFAULT_KEEP = 20


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(queue_path: Path) -> list[dict[str, Any]]:
    """读队列全部条目（按入队序）；文件缺失 → 空表，坏条目跳过但不静默丢弃整队。

    fail-soft 而非 fail-open：坏行留在盘上会让人以为「队列清空了」，故返回的是「读得出的
    条目」，损坏原件由 :func:`save` 的整体重写在下一次入队时自然清除。
    """
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict) and i.get("id")]


def save(queue_path: Path, items: list[dict[str, Any]]) -> None:
    """整体重写队列（``.tmp`` + ``os.replace`` 原子落盘）。"""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = queue_path.with_name(queue_path.name + ".tmp")
    doc = {"version": 1, "updated_at": _now(), "items": items}
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(tmp), str(queue_path))


def new_item_id(
    queue_path: Path,
    prefix: str,
    kind: str,
    *,
    stamp: str | None = None,
    sep: str = "-",
) -> str:
    """条目 id ``<prefix>-<ts><sep><kind>``，同秒撞名则补 ``-002``（id 唯一是销账依据）。

    Args:
        stamp: 命名时间戳覆盖——**测试注入口**：脚本里不许用 ``datetime.now()`` 决定
            可复现的命名，同一次扫描立多项时更需要调用方控住时间。
    """
    base = f"{prefix}-{stamp or datetime.now().strftime('%Y%m%d-%H%M%S')}{sep}{kind}"
    taken = {str(i.get("id")) for i in load(queue_path)}
    if base not in taken:
        return base
    n = 2
    while f"{base}{sep}{n:03d}" in taken:
        n += 1
    return f"{base}{sep}{n:03d}"


def append(queue_path: Path, item: dict[str, Any]) -> dict[str, Any]:
    """尾部追加一项（FIFO），返回该条目。"""
    items = load(queue_path)
    items.append(item)
    save(queue_path, items)
    return item


def mutate(queue_path: Path, item_id: str, **fields: Any) -> dict[str, Any] | None:
    """按 id 改写字段并落盘，返回改后的条目（找不到返回 ``None``）。"""
    items = load(queue_path)
    for i, it in enumerate(items):
        if str(it.get("id")) == item_id:
            it.update(fields)
            items[i] = it
            save(queue_path, items)
            return it
    return None


def take_next(queue_path: Path, *, ready: Any = None) -> dict[str, Any] | None:
    """取队首 ``pending`` 项并标记 ``in_progress``（无则 ``None``）。

    ``ready``：可选**就绪判据**（入参为条目，返回是否可出队）——出队单点的唯一注入点
    （§3.3 延期检查：``ready=lambda it: not backpressure.item_deferred(it, now)``）。
    缺省 ``None`` = 旧行为（只看 pending）。
    """
    for it in load(queue_path):
        if it.get("status") != STATUS_PENDING:
            continue
        if ready is not None and not ready(it):
            continue
        return mutate(
            queue_path,
            str(it["id"]),
            status=STATUS_IN_PROGRESS,
            started_at=_now(),
        )
    return None


def defer_item(
    queue_path: Path,
    item_id: str,
    *,
    until: float,
    count: int,
    elapsed_s: float,
    note: str = "",
) -> dict[str, Any] | None:
    """把条目复位 ``pending`` 并挂延期（带宽类失败的主处置；语义同 backlog_defer）。

    条目**带账单回队**（``defer_until`` / ``defer_count`` / ``defer_elapsed_s``），调度侧
    据 :func:`~pysci.skills.orchestration.tools.backpressure.item_deferred` 跳过、睡到到期
    再取。``started_at`` 清空——它不再是「在跑」，而是「在等」。
    """
    fields: dict[str, Any] = {
        "status": STATUS_PENDING,
        "started_at": "",
        "defer_until": float(until),
        "defer_count": int(count),
        "defer_elapsed_s": float(elapsed_s),
    }
    if note:
        fields["note"] = note
    return mutate(queue_path, item_id, **fields)


def reset_orphans(queue_path: Path) -> list[str]:
    """把 ``in_progress`` 复位为 ``pending``——刚获锁时它们是已死 worker 的孤儿。

    单 worker 互斥由锁保证，故获锁一刻仍在 ``in_progress`` 的条目必是上一任被杀死时
    留下的，不能让它永久卡在「正在跑」。返回被复位的 id 列表（进 ``.done`` 摘要）。
    """
    out: list[str] = []
    items = load(queue_path)
    for i, it in enumerate(items):
        if it.get("status") != STATUS_IN_PROGRESS:
            continue
        started = it.get("started_at", "?")
        it = dict(it)
        it.update(
            status=STATUS_PENDING,
            started_at="",
            reclaimed_at=_now(),
            note=f"孤儿回收（{started} 起的在途项）",
        )
        items[i] = it
        out.append(str(it.get("id")))
    if out:
        save(queue_path, items)
    return out


def finish(
    queue_path: Path, item_id: str, status: str, result: dict[str, Any]
) -> dict[str, Any] | None:
    """写终态与结构化结果；``reviewed_at`` 留空 = 待查看（择机查看型通知，§5.1-B）。"""
    return mutate(
        queue_path,
        item_id,
        status=status,
        result=result,
        finished_at=_now(),
        reviewed_at="",
    )


def pending_review(queue_path: Path) -> list[dict[str, Any]]:
    """「待查看结果」面：已终态但提请人尚未确认的条目。"""
    return [
        i
        for i in load(queue_path)
        if i.get("status") in TERMINAL_STATUSES and not i.get("reviewed_at")
    ]


def mark_seen(queue_path: Path, item_id: str) -> dict[str, Any] | None:
    """提请人确认已查看（从「待查看」面摘掉；条目本身留作审计直到下次裁剪）。"""
    return mutate(queue_path, item_id, reviewed_at=_now())


def counts(queue_path: Path) -> dict[str, int]:
    """按状态计数（status 一行读数；调用方用它做积压判断）。"""
    out: dict[str, int] = {}
    for it in load(queue_path):
        st = str(it.get("status", "?"))
        out[st] = out.get(st, 0) + 1
    return out


def pending_count(queue_path: Path) -> int:
    """待产出条目数（积压闸的读数口径：``pending`` 才算积压，``in_progress`` 是正在做）。"""
    return sum(1 for it in load(queue_path) if it.get("status") == STATUS_PENDING)


#: 条目「入队时刻」的候选字段（各成员队列写法不同；取首个能解析的）。缺该字段的条目
#: **不参与**超时判据——拿不到时间戳就别发告警（坏数据不该变成告警噪声）。
_TS_KEYS: tuple[str, ...] = ("enqueued_at", "created")


def item_epoch(it: dict[str, Any]) -> float | None:
    """条目的入队时刻（epoch 秒）；解析不出 → ``None``。"""
    for key in _TS_KEYS:
        raw = it.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw)).timestamp()
        except (TypeError, ValueError):
            continue
    return None


def stale_pending(
    queue_path: Path, *, timeout_s: int, now: float | None = None
) -> list[dict[str, Any]]:
    """``pending`` 且**入队超时**（距今 > ``timeout_s`` 秒）的条目。

    观察面：**不许静默积压**（autonomy-fix a2）——2026-10-14 那起 02:21 入队、整夜 pending
    的巡检项就是「没人取走也没人报警」的形态。给**各队列共用**（成员队列/inspector 队列）
    一份判据，别每队各写一套。
    """
    moment = datetime.now().timestamp() if now is None else float(now)
    out: list[dict[str, Any]] = []
    for it in load(queue_path):
        if it.get("status") != STATUS_PENDING:
            continue
        epoch = item_epoch(it)
        if epoch is None:
            continue
        if moment - epoch > int(timeout_s):
            out.append(it)
    return out


def prune(queue_path: Path, keep: int = DEFAULT_KEEP) -> int:
    """裁掉最早的条目，只留最近 ``keep`` 条，返回裁掉数。"""
    items = load(queue_path)
    if len(items) <= keep:
        return 0
    dropped = items[:-keep]
    save(queue_path, items[-keep:])
    return len(dropped)
