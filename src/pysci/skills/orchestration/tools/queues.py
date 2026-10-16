"""成员工作队列（README §12，M4）：全员统一 FIFO 的**队列层**。

每成员一队 ``orchestration/state/queues/<member>.json``，**队列形状全部走** :mod:`workq`
（纪律：同一能力不留两份实现）。本模块只定义「条目长什么样、入队/查队/父子关联怎么算」；
worker 生命周期在 :mod:`queueworker`。

**覆盖范围**：7 专职（sim/figure/writing/theory/lit/drawing/model3d）+ deputy + reviewer
+ devops。devops **双通道并存**（非二选一，见 :data:`QUEUE_MEMBERS` 的 ADR）：plan 环节
直派（``kind=produce``）与 backlog 组包（``kind=backlog-batch``，worker 经 drain 账务结算）
同队列共用 FIFO/worker；bencher、inspector 各有既有队列，**不在此列**。

**组包边界（§3.1）**：生产组员 job **一律不组包**（worker 每次只取一条 = batch_size 恒 1）；
devops backlog 的组包/延期/needs_leader 语义在 :mod:`workflow`/计划器侧保留，执行落到本队列。

**reviewer 附属（§3.1 第 3 件）**：带 ``review`` 位的宿主 job 交付后，worker 自动向 ``reviewer``
队列投一条 ``kind="review"`` 的**子 job**（``parent``=宿主 id、``parent_member``=宿主的成员）；
宿主 job 的「完成」= 宿主终态 **且** 其 review 子 job 终态（:func:`completion` 派生，不等跨队
回写——跨队写有并发丢更新的风险，派生读没有）。review 子 job 收尾时**尽力**回填宿主记录
（``review_result``），回填失败不影响完成判定。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from . import backpressure, workq

#: 成员队列根（运行态；gitignore 语义与其余 state 同族）。
QUEUES_DIR: Path = ORCH_STATE_ROOT / "queues"

#: 建队的成员（§3.1）：7 专职 + deputy + reviewer（reviewer 只收 review 子 job）+ devops。
#:
#: **ADR（task-20261014-devops-gap-d1，用户裁决 2026-10-14）**——推翻旧决策「基建/测量类
#: 条目走 devops backlog、不进 plan」：
#: - **选了什么**：devops 收编为 plan 系统可直接派发的队列成员（plan run / plan advance /
#:   queues enqueue 三路全通），与既有 backlog FIFO 组包通道**并存**（双通道，**非二选一**：
#:   ``kind=produce`` = plan 环节直派，worker 走通用 ``do_dispatch`` 结账；``kind=
#:   backlog-batch`` = backlog 组包入队，worker 经 ``drain.settle_backlog_batch`` 落账务）。
#: - **弃了什么**：弃「基建条目不进 plan」的旧边界（其成因是队列化前没有统一 FIFO 面，
#:   基建/测量只能另走 backlog；M4 队列化落地后该边界成了残留）。
#: - **为什么**：大任务的计划里**显式含基建改进环节**（plan-first 要求环节全走计划），旧边界
#:   让这类环节无处派发（PSEP_2 备忘点名的缺口）；队列化后双通道同队列并存是合理形态。
QUEUE_MEMBERS: tuple[str, ...] = (
    "sim",
    "figure",
    "writing",
    "theory",
    "lit",
    "drawing",
    "model3d",
    "deputy",
    "reviewer",
    "devops",
)

#: 宿主 job 的完成态读数。
OPEN = "open"
AWAITING_REVIEW = "awaiting_review"
COMPLETE = "complete"


def queue_path(member: str, *, root: Path | None = None) -> Path:
    """队列文件路径。``root`` 给 **state 根**（其下 ``queues/``）；缺省取 :data:`QUEUES_DIR`。"""
    base = (root / "queues") if root is not None else QUEUES_DIR
    return base / f"{member}.json"


def enqueue(
    member: str,
    *,
    text: str = "",
    task_file: str = "",
    dirs: list[str] | None = None,
    model_tier: str = "",
    effort: str = "",
    plan: str = "",
    step: str = "",
    session: str = "latest",
    review: bool = False,
    kind: str = "produce",
    parent: str = "",
    parent_member: str = "",
    slug: str = "job",
    item_id: str | None = None,
    max_turns: int | None = None,
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """入队一条 job（FIFO 尾插），返回条目。

    ``review=True`` 表示该宿主 job 交付后需 reviewer 复审（worker 自动投子 job）。
    ``kind="review"`` + ``parent``/``parent_member`` 是 review 子 job 的形态。
    ``max_turns``：单跳轮次上限（``plan run --max-turns`` 经此透传；None=成员默认）。
    ``extra``：附加字段并入条目（如 devops 批次的 ``backlog`` 元数据）——固定字段不覆盖。
    """
    qp = queue_path(member, root=root)
    item: dict[str, Any] = {
        "id": item_id or workq.new_item_id(qp, member, slug),
        "member": member,
        "kind": kind,
        "task": text,
        "task_file": task_file,
        "dirs": list(dirs or []),
        "model": model_tier,
        "effort": effort,
        "plan": plan,
        "step": step,
        "session": session or "latest",
        "review": bool(review),
        "parent": parent,
        "parent_member": parent_member,
        "status": workq.STATUS_PENDING,
        "created": workq._now(),
        "max_turns": int(max_turns) if max_turns else None,
        "result": None,
    }
    for key, value in dict(extra or {}).items():
        if key not in item:
            item[key] = value
    return workq.append(qp, item)


def items(member: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    """该队全部条目（入队序）。"""
    return workq.load(queue_path(member, root=root))


def counts(member: str, *, root: Path | None = None) -> dict[str, int]:
    """按状态计数（status 队列段读数）。"""
    return workq.counts(queue_path(member, root=root))


def _now_s(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def all_items(*, root: Path | None = None) -> list[dict[str, Any]]:
    """全部成员队列的条目（跨队聚合读数用）。"""
    out: list[dict[str, Any]] = []
    for m in all_members(root=root):
        out.extend(items(m, root=root))
    return out


def take_ready(
    member: str, *, now: float | None = None, root: Path | None = None
) -> dict[str, Any] | None:
    """**出队单点**（m4 预留、§3.3 在此接延期检查）：先过 backpressure 延期判据再取。

    未到期的条目不被取（``defer_until`` 在未来的 pending 跳过）——带宽类失败回队挂延期后，
    调度侧睡到到期再查即自然重取。缺省 ``now`` 取系统时钟（worker 主循环）；测试注入假时钟。
    """
    now_s = _now_s(now)
    return workq.take_next(
        queue_path(member, root=root),
        ready=lambda it: not backpressure.item_deferred(it, now_s),
    )


def defer(
    member: str,
    item_id: str,
    *,
    until: float,
    count: int,
    elapsed_s: float,
    note: str = "",
    root: Path | None = None,
) -> dict[str, Any] | None:
    """成员队列条目的延期回队（带宽类失败主处置；语义同 workflow.backlog_defer）。"""
    return workq.defer_item(
        queue_path(member, root=root),
        item_id,
        until=until,
        count=count,
        elapsed_s=elapsed_s,
        note=note,
    )


def deferred_count(
    member: str | None = None, *, now: float | None = None, root: Path | None = None
) -> int:
    """延期窗内的条目数；``member`` 给定则只数该队（worker 用），否则跨队聚合（观察面用）。"""
    now_s = _now_s(now)
    pool = items(member, root=root) if member is not None else all_items(root=root)
    return backpressure.deferred_count(pool, now=now_s)


def next_defer_s(
    member: str | None = None, *, now: float | None = None, root: Path | None = None
) -> float:
    """到最近一个延期条目到期的秒数（worker 睡到到期用）；``member`` 语义同上。"""
    now_s = _now_s(now)
    pool = items(member, root=root) if member is not None else all_items(root=root)
    return backpressure.next_defer_s(pool, now=now_s)


def water_lines(*, root: Path | None = None) -> list[str]:
    """观察面：成员队列水位（``orch status`` 队列段用）。

    **每个成员必出一行**（含全零计数，行尾「（空）」明示——修复：旧实现 ``if not c:
    continue`` 把计数为空的成员整行跳过，9 队只见少数几行）；在途明细行**带成员名**
    （形如 ``· [m] 在途 <id>（plan·step）起于 …``，归属一眼可辨）。

    **「皆空」的唯一合法场景 = pending 与 in_progress 双零**（用户 2026-10-14：在途条目
    曾被一句「各成员队列皆空」盖住——`plan run` 还在直派、队列照却是空的）。非双零时
    逐队列计数，并把**在途条目显名 + 起跑时刻**点出来。
    """
    members = all_members(root=root)
    out: list[str] = []
    inflight: list[str] = []
    pending_total = 0
    active_total = 0
    for m in members:
        c = counts(m, root=root)
        pending = c.get(workq.STATUS_PENDING, 0)
        gip = c.get(workq.STATUS_IN_PROGRESS, 0)
        pending_total += pending
        active_total += gip
        done = c.get(workq.STATUS_DONE, 0)
        failed = c.get(workq.STATUS_FAILED, 0)
        empty = not (pending or gip or done or failed)
        out.append(
            f"  [{m}] 待={pending} 在做={gip} 完成={done} 失败={failed}"
            + ("  ◀ 有积压" if pending else "")
            + ("  （空）" if empty else "")
        )
        if gip:
            for it in items(m, root=root):
                if str(it.get("status")) != workq.STATUS_IN_PROGRESS:
                    continue
                plan = str(it.get("plan") or "")
                step = str(it.get("step") or "")
                where = f"（{plan}·{step}）" if plan or step else ""
                inflight.append(
                    f"    · [{m}] 在途 {it.get('id')}{where}"
                    f" 起于 {it.get('started_at') or '?'}"
                )
    header = (
        f"  在途合计：pending={pending_total} in_progress={active_total}"
        f"（{len(members)} 队）"
    )
    if pending_total == 0 and active_total == 0:
        # 「皆空」只在这里出现：双零才敢这么写（其余情况必须点名在途，见上）。
        return [
            header,
            *out,
            f"  （{len(members)} 队皆空：pending 与 in_progress 双零"
            "——done/failed 历史不在本行；支撑席存量另报）",
        ]
    return [header, *out, *inflight]


def all_members(*, root: Path | None = None) -> tuple[str, ...]:
    """有队列文件的成员（含 QUEUE_MEMBERS 之外的遗留文件；只读不崩）。"""
    base = root or QUEUES_DIR
    if not base.exists():
        return QUEUE_MEMBERS
    found = tuple(sorted(p.stem for p in base.glob("*.json")))
    return tuple(dict.fromkeys([*QUEUE_MEMBERS, *found]))


def review_children(
    host_id: str, *, host_member: str = "", root: Path | None = None
) -> list[dict[str, Any]]:
    """宿主 job 的 review 子 job（在 reviewer 队里按 ``parent`` 反查）。"""
    return [
        it
        for it in items("reviewer", root=root)
        if str(it.get("parent")) == str(host_id)
        and (not host_member or str(it.get("parent_member")) == str(host_member))
    ]


def completion(
    host_id: str, *, host_member: str, root: Path | None = None
) -> tuple[str, str]:
    """宿主 job 的**完成语义**：``(状态, 说明)``——宿主+其 review 都收尾才算 ``complete``。

    - 宿主未终态 → ``open``；
    - 宿主终态但有 review 子 job 未终态 → ``awaiting_review``；
    - 宿主终态且（无 review 子 job 或子 job 终态）→ ``complete``。
    """
    host = next(
        (i for i in items(host_member, root=root) if str(i.get("id")) == str(host_id)),
        None,
    )
    if host is None:
        return OPEN, "宿主 job 不存在"
    if host.get("status") not in workq.TERMINAL_STATUSES:
        return OPEN, f"宿主 {host.get('status')}"
    kids = review_children(host_id, host_member=host_member, root=root)
    if not kids:
        return COMPLETE, "宿主收尾（无 review 附件）"
    pending = [k for k in kids if k.get("status") not in workq.TERMINAL_STATUSES]
    if pending:
        return AWAITING_REVIEW, f"待 review 子 job {len(pending)} 条"
    return COMPLETE, "宿主 + review 均收尾"


def spawn_review(
    host: dict[str, Any], *, slug: str = "review", root: Path | None = None
) -> dict[str, Any] | None:
    """宿主交付后**自动**向 reviewer 队投附属子 job（父子 id 关联）；``review`` 未置 → None。"""
    if not host.get("review"):
        return None
    host_id = str(host.get("id"))
    member = str(host.get("member"))
    brief = (
        f"审查交付：job {host_id}（成员 {member}）\n"
        f"原标题/正文：{str(host.get('task') or host.get('task_file') or '')[:200]}\n"
        "按 §5.3 审查链给 VERDICT；FAIL 的回派由组长侧处置。"
    )
    return enqueue(
        "reviewer",
        text=brief,
        kind="review",
        parent=host_id,
        parent_member=member,
        slug=slug,
        root=root,
    )


def backfill_host(
    review_item: dict[str, Any], result: dict[str, Any], *, root: Path | None = None
) -> bool:
    """review 子 job 收尾时**尽力**回填宿主记录（``review_result``）。

    完成判定**不依赖**本回填（:func:`completion` 是派生的）；跨队写有并发丢更新风险，
    故失败只返回 False，不抛、不掩盖。返回是否写成功。
    """
    parent = str(review_item.get("parent") or "")
    pm = str(review_item.get("parent_member") or "")
    if not parent or not pm:
        return False
    try:
        return (
            workq.mutate(queue_path(pm, root=root), parent, review_result=result)
            is not None
        )
    except OSError:
        return False
