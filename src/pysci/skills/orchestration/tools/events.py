"""统一事件/唤醒层（README §12，M5）：全员事件**一条流**。

M4 之前事件散在三处：drain 收尾靠 watch 差分、渠道回退在 ``channel-events.jsonl``、
任务书唤醒是 watch 的临时差分（M3 最小扩展）。M5 把「完成→通知→组长推进」收成**一个
追加式事件流** ``state/events.jsonl``（channel-events 同族），生产者直接落盘，消费者只有
一个：组长经 ``orch events drain`` 一次排空积压（**合并唤醒**，防唤醒风暴）。

设计边界（§纪律「同一能力不留两份实现」）：
- 本模块只放**流形状**（写入/读取/水位线/类型词表/唤醒策略），不放任何业务判定；
- **生产者直发**的新类型（worker 消化处）：:data:`JOB_DONE` / :data:`STEP_REVIEW_DONE`
  （:mod:`queueworker`）、:data:`QUEUE_EMPTY`（:mod:`drain`/:mod:`benchd`）、
  :data:`TASKBOOK_NEW`（:mod:`taskbooks`）；
- **沿现制迁入**的旧类型（CRASH/QUOTA-STOP/CHANNEL-STOP/NEEDS-LEADER/SALVAGE）：唤醒仍由
  :mod:`watch` 的基线差分拍板（既有语义零改动），:func:`watch.detect` 命中时把事件**镜像**
  入本流（记录面），故流里既有生产者直发、也有差分镜像，二者不重叠。

**边沿触发**：唤醒只看「新事件」（水位线之上）；多 worker 同时完工在流里是多条、醒来只折合
一个回合（组长一次 :func:`drain` 排空全部）。:func:`drain` 幂等——重复调用第二次返回空。

事件流**只追加不改写**（§纪律）；写失败一律 fail-open（观察面塌不了生产链，§0.6）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

#: 统一事件流（gitignored：每跳都可能新增，落跟踪面会让 main 每跳必 dirty）。
EVENTS_LOG: Path = ORCH_STATE_ROOT / "events.jsonl"
#: 消费者水位线（``orch events drain`` 排空到哪一条）；纯运行态。
SEEN_PATH: Path = ORCH_STATE_ROOT / "events-seen.json"

# ---------------------------------------------------------------------------
# 类型词表与唤醒策略（§3.2 表）
# ---------------------------------------------------------------------------
JOB_DONE = "JOB-DONE"
STEP_REVIEW_DONE = "STEP-REVIEW-DONE"
QUEUE_EMPTY = "QUEUE-EMPTY"
TASKBOOK_NEW = "TASKBOOK-NEW"
NEEDS_LEADER = "NEEDS-LEADER"
QUOTA_STOP = "QUOTA-STOP"
CHANNEL_STOP = "CHANNEL-STOP"
CRASH = "CRASH"
SALVAGE = "SALVAGE"
QUEUE_COMPLETE = (
    "QUEUE-COMPLETE"  # watch 差分的 devops 收尾名（legacy；流里用 QUEUE-EMPTY）
)

#: 事件 → 唤醒策略（人读；§3.2 表逐行落这里）。
WAKE: dict[str, str] = {
    JOB_DONE: "每件唤醒组长（7 专职+deputy，及时接续下一份工作）",
    STEP_REVIEW_DONE: "环节+附属 review 合并完成，唤醒一次（一个通知单元）",
    QUEUE_EMPTY: "devops / bencher 队列消化完，队列空唤醒一次",
    TASKBOOK_NEW: "用户投递任务书，立即唤醒组长",
    NEEDS_LEADER: "沿现制：drain 判定须组长处理",
    QUOTA_STOP: "沿现制：额度类全局停止",
    CHANNEL_STOP: "沿现制：渠道级故障全局停止",
    CRASH: "沿现制：worker 被中途杀死（无 .done）",
    SALVAGE: "沿现制：抢救知会（进度已保住，不必动手）",
}

#: 词表全集（顺序即 §3.2 表的呈现序）。
TYPES: tuple[str, ...] = tuple(WAKE)

#: **生产者直发**、由 watch 从流里读取来唤醒的类型。其余类型唤醒仍走 watch 差分（native），
#: 不经流回读——否则同一事件会被「直发」与「差分」各报一次（双触发）。
STREAM_TYPES: tuple[str, ...] = (
    JOB_DONE,
    STEP_REVIEW_DONE,
    QUEUE_EMPTY,
    TASKBOOK_NEW,
)

#: watch 差分命中后**镜像**入流（记录面）的类型——唤醒已由差分送出，这里只补流账。
#: QUEUE-COMPLETE 不镜（与生产者的 QUEUE-EMPTY 是同一收尾的两副名，镜了会重）。
MIRROR_TYPES: tuple[str, ...] = (
    CRASH,
    QUOTA_STOP,
    CHANNEL_STOP,
    NEEDS_LEADER,
    SALVAGE,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def wake_label(event: str) -> str:
    """该事件的唤醒策略文本（读不到给空串——新类型在读数面可见、不被静默）。"""
    return WAKE.get(str(event), "")


#: ``.done`` 的 ``stopped`` 字段 → 收尾方式的人读（devops/bencher worker 共用）。
_HOW_DONE: dict[str, str] = {
    "cooperative": "协作停止",
    "quota_exhausted": "额度耗尽全局停止",
    "channel_error": "渠道级故障全局停止",
}


def note_queue_empty(
    who: str,
    *,
    subject: str,
    count: int,
    elapsed_s: Any = "?",
    stopped: str = "",
) -> dict[str, Any] | None:
    """队列消化完事件（``QUEUE-EMPTY``）：devops / bencher worker 收尾的**统一入口**。

    两处 worker 的收尾形状相同（谁、多少项、多久、怎么收的），措辞收在这里，别各写各的
    （否则同一事实两副面孔，§纪律）。写失败 fail-open（返回 ``None``）。
    """
    how = _HOW_DONE.get(str(stopped or ""), "自然跑完")
    return emit(
        QUEUE_EMPTY,
        subject=str(subject),
        detail=f"{who} 队列消化完：处理 {int(count)} 项 / {elapsed_s}s，{how}",
        action="orch events drain 一次排空积压；orch status 看交付明细与台账",
    )


# ---------------------------------------------------------------------------
# 写入（生产者直发 / 差分镜像共用；fail-open）
# ---------------------------------------------------------------------------
def _read_lines() -> list[dict[str, Any]]:
    """读流全部条目（坏行跳过；文件缺失 → 空表）。"""
    try:
        raw = EVENTS_LOG.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _seq_of(rec: dict[str, Any]) -> int:
    try:
        return int(rec.get("seq", 0) or 0)
    except (TypeError, ValueError):
        return 0


def latest_seq() -> int:
    """当前流里最大 ``seq``（水位线口径 = 已写出的最后一条）。"""
    return max((_seq_of(e) for e in _read_lines()), default=0)


def emit(
    event: str,
    subject: str = "",
    detail: str = "",
    action: str = "",
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """追加一条事件（``seq`` 自增、``ts`` 落当前时刻），返回条目；失败返回 ``None``。

    写失败不抛（观察面写失败不炸主流程）；跨进程并发写同一秒极小概率拿到相同 ``seq``，
    对水位线只是「多读一条/少读一条」，不影响正确性（本流无唯一性约束）。
    """
    rec: dict[str, Any] = {
        "ts": _now(),
        "seq": latest_seq() + 1,
        "event": str(event),
        "subject": str(subject),
        "detail": str(detail),
        "action": str(action),
    }
    if extra:
        rec.update(extra)
    try:
        EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return rec


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def read_all(*, limit: int | None = None) -> list[dict[str, Any]]:
    """读流全部条目（append 序）；``limit`` 取尾部 N 条。"""
    ev = _read_lines()
    return ev[-limit:] if limit is not None else ev


def read_after(seq: int) -> list[dict[str, Any]]:
    """读 ``seq`` 之后的新事件（严格大于；边沿触发的「新」判据）。"""
    floor = int(seq or 0)
    return [e for e in _read_lines() if _seq_of(e) > floor]


def counts() -> dict[str, int]:
    """按事件类型计数（status 一行读数）。"""
    out: dict[str, int] = {}
    for e in _read_lines():
        k = str(e.get("event", "?"))
        out[k] = out.get(k, 0) + 1
    return out


def format_event(ev: dict[str, Any]) -> str:
    """事件单行呈现（与 :func:`watch.format_event` 同形，供 status/events 面复用）。"""
    return (
        f"{str(ev.get('event', '?'))} {str(ev.get('subject', '') or '')} —— "
        f"{str(ev.get('detail', '') or '')}｜建议处置：{str(ev.get('action', '') or '')}"
    )


# ---------------------------------------------------------------------------
# 水位线 / 排空（``orch events drain``，幂等）
# ---------------------------------------------------------------------------
def _load_seen() -> dict[str, Any]:
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_seen(data: dict[str, Any]) -> None:
    try:
        SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        SEEN_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def ack_seq() -> int:
    """已排空到的 ``seq``（水位线；缺省 0=从未排空）。"""
    try:
        return int(_load_seen().get("ack_seq", 0) or 0)
    except (TypeError, ValueError):
        return 0


def unacked() -> list[dict[str, Any]]:
    """水位线之上、尚未被 ``orch events drain`` 排空的积压事件。"""
    return read_after(ack_seq())


def drain() -> list[dict[str, Any]]:
    """**一次排空全部积压事件**并推进水位线，返回事件列表。

    幂等：紧接的第二次调用返回空（水位线已到最后一条）——合并唤醒的消费者面
    （组长醒来排空一次即清算全部，多 worker 同时完工折合一个回合）。
    """
    ev = unacked()
    if ev:
        _save_seen({"ack_seq": max(_seq_of(e) for e in ev), "ack_at": _now()})
    return ev


def status_lines() -> list[str]:
    """``orch status`` 事件段：水位 + 最近事件（观察面）。"""
    total = latest_seq()
    ack = ack_seq()
    pending = len(unacked())
    out = [f"  事件水位 seq={total}（已排空至 {ack}，待排空 {pending}）"]
    recent = read_all(limit=5)
    if not recent:
        out.append("  （尚无事件；生产者直发 + watch 差分镜像入 state/events.jsonl）")
        return out
    for e in recent:
        out.append(f"    · {str(e.get('ts', ''))[5:19]} {format_event(e)}")
    return out


__all__ = [
    "CHANNEL_STOP",
    "CRASH",
    "EVENTS_LOG",
    "JOB_DONE",
    "MIRROR_TYPES",
    "NEEDS_LEADER",
    "QUEUE_COMPLETE",
    "QUEUE_EMPTY",
    "QUOTA_STOP",
    "SALVAGE",
    "SEEN_PATH",
    "STEP_REVIEW_DONE",
    "STREAM_TYPES",
    "TASKBOOK_NEW",
    "TYPES",
    "WAKE",
    "ack_seq",
    "counts",
    "drain",
    "emit",
    "format_event",
    "latest_seq",
    "read_after",
    "read_all",
    "status_lines",
    "unacked",
    "wake_label",
]
