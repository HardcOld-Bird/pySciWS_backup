"""模型回退链的**策略面**：选席、粘滞跳过窗、回退事件留痕与告警文案。

用户裁决 2026-10-12 21:1x（backlog 20261012-211500-leader）：Mimo 主力双档 + 三级有序
回退 + **回退必告警**。本模块把这条策略收在一处，消费面（``dispatch`` 换席重试、``drain``
运行日志、``mirror status`` 概览、``orch status`` 判决包健康段）只调用不各自表述——
「同一事实三副面孔」是本项目反复付过学费的坑（见 ``preference-journal`` 与词表四面执法）。

配置事实（档位链、跳过窗时长、席→名反查表）在 :mod:`registry`；这里只管**决策与读数**。

fail-open 红线（README §4.3）：观察面（事件读取/文案）任何失败都不许挡住派发；
但**派发面**（选席）不能静默吞错——读不到窗就当作「无窗、全部照探」，宁可多烧一跳
探测，也不要把一条好链锁死。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from .registry import LEGACY_TIER_ALIASES, Registry, normalize_seats

#: 渠道回退事件流（gitignored：每跳都可能新增，落跟踪面会堵死一切碰 state 的单）。
EVENTS_PATH: Path = ORCH_STATE_ROOT / "channel-events.jsonl"

#: mirror/status 概览里「计数」口径的统计窗（天）。窗太短看不见「昨天那起故障」，太长失真。
COUNT_WINDOW_DAYS = 7


@dataclass
class SeatPlan:
    """一次派发开始前的选席结果。"""

    tier: str
    #: 本次允许依次探测的席位（首席在前）
    usable: list[str] = field(default_factory=list)
    #: 落在粘滞窗内、本跳**不探测**的席（告警面要逐席点名，故障可查性不能丢）
    skipped: list[dict[str, Any]] = field(default_factory=list)
    #: 全链都在窗内 → 强行探测最先到期的一席（宁可多烧一跳，不可无席可用）
    forced_probe: str = ""


def plan_seats(reg: Registry, tier: str, *, now: float | None = None) -> SeatPlan:
    """按链序选出本跳可探测的席，并把窗内的席列成告警。

    判据：某席在 ``model_fallback.skip_window_s`` 内有失败记录 ⇒ 跳过（省探测跳数）；
    窗口过期自动恢复探测——用户充值/修好后无需改代码即自愈。

    Args:
        reg: 已加载的注册表（读链与运行态故障窗）。
        tier: 角色键（``main``/``pro``/``minimum``）；旧档名 ``flash``/``max`` 归一到
            main/pro（:data:`~.registry.LEGACY_TIER_ALIASES`）。
        now: 注入的当前时刻（epoch 秒），单测用；缺省取系统时间。

    Returns:
        :class:`SeatPlan`。链里只剩一席时不会有 ``skipped``（无下一席可落，照跑）。
    """
    moment = time.time() if now is None else float(now)
    window = int(reg.fallback_policy.get("skip_window_s", 0) or 0)
    # 旧档名（flash/max）归一到角色：故障窗的读面已按角色归一，键不对齐会让窗全部失明。
    tier = LEGACY_TIER_ALIASES.get(tier, tier)
    chain = reg.model_chain(tier)
    fails = reg.seat_failures.get(tier, {}) if window else {}

    def age(seat: str) -> float:
        """该席距今的失败时长；无记录 = 无穷大（从未失败）。"""
        ts = fails.get(seat, {}).get("ts")
        try:
            return moment - float(ts)
        except (TypeError, ValueError):
            return float("inf")

    def in_window(seat: str) -> bool:
        return seat in fails and age(seat) < window

    usable = [seat for seat in chain if not in_window(seat)]
    skipped = [
        {
            "seat": seat,
            "label": reg.seat_label(seat),
            "kind": str(fails.get(seat, {}).get("kind", "")),
            "ts": fails.get(seat, {}).get("ts"),
            "member": str(fails.get(seat, {}).get("member", "")),
        }
        for seat in chain
        if in_window(seat)
    ]
    plan = SeatPlan(tier=tier, usable=usable, skipped=skipped)
    if not usable and chain:
        # 全链都在窗内：不能因此拒绝派发。取**失败最早**的那一席（=窗口剩得最短、最先到期）
        # 强行再探一次；挑刚坏的那席等于挑「最不可能已经修好」的一席，白烧。
        earliest = max(chain, key=age)
        plan.forced_probe = earliest
        plan.usable = [earliest]
        plan.skipped = [item for item in skipped if item["seat"] != earliest]
    return plan


def note_fallback(
    reg: Registry,
    *,
    tier: str,
    frm: str,
    to: str,
    reason: str,
    member: str = "",
    sid: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    """记一起降级回退：故障席进窗 + 事件流追加 + 返回事件体（调用者面文案用）。

    ``to`` 为空表示**全链耗尽**——这条也要落账，否则「停止」在观测面上是隐形的。
    """
    moment = time.time() if now is None else float(now)
    reg.note_seat_failure(tier, frm, kind=reason, member=member, now=moment)
    event = {
        "ts": datetime.fromtimestamp(moment).astimezone().isoformat(timespec="seconds"),
        "tier": tier,
        "from": frm,
        "from_label": reg.seat_label(frm),
        "to": to,
        "to_label": reg.seat_label(to) if to else "(全链耗尽)",
        "reason": reason,
        "member": member,
        "sid": sid,
    }
    append_event(event)
    return event


def note_seat_ok(reg: Registry, *, tier: str, seat: str) -> None:
    """席位跑通 → 销掉它的故障窗（充值/修好后第一次成功即彻底恢复正常优先级）。"""
    reg.note_seat_success(tier, seat)


def append_event(event: dict[str, Any], *, path: Path | None = None) -> None:
    """追加一行事件（观察面写失败不炸主流程——留痕尽力而为）。"""
    target = path or EVENTS_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_events(
    *,
    limit: int | None = None,
    since_days: float | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """读事件流（坏行跳过；文件缺失返回空表）。``limit`` 取尾部 N 条。"""
    target = path or EVENTS_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    if since_days is not None:
        cutoff = datetime.now().astimezone().timestamp() - since_days * 86400
        events = [e for e in events if _event_ts(e) >= cutoff]
    return events[-limit:] if limit is not None else events


def _event_ts(event: dict[str, Any]) -> float:
    """事件时间戳（epoch 秒）；读不出来按 0（=最旧，落进「窗外」而被过滤，不误计）。"""
    text = str(event.get("ts") or "")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def chain_text(value: Any, *, reg: Registry | None = None) -> str:
    """席位链（单值/列表兼容读）的一行呈现：``Mimo-V2.6-Pro → Qwen-3.8-Max → (用户级默认)``。

    呈现口径单一源：drain 运行日志与 ``orch refresh-models`` 打印共用，别各写各的
    （否则同一档值在两处排版不同，成了「同一事实两副面孔」）。
    """
    reader = reg or Registry.load()
    seats = normalize_seats(value)
    return " → ".join(reader.seat_label(seat) for seat in seats) if seats else "(空)"


def alert_line(event: dict[str, Any]) -> str:
    """调用者面的降级告警行（用户裁决 4a：dispatch/envelope 侧每次回退都要看见）。"""
    return (
        f"[!] 档位回退：{event.get('tier', '?')} 席{event.get('from_label', '?')}"
        f"不可用 → 已切换 {event.get('to_label', '?')}"
        f"（原因 {event.get('reason', '?')}）"
    )


def skip_line(info: dict[str, Any]) -> str:
    """窗内跳过的告警行（省跳数但不丢可查性：故障时刻与原因照样打出来）。"""
    when = ""
    try:
        when = datetime.fromtimestamp(float(info.get("ts"))).strftime("%H:%M")
    except (TypeError, ValueError):
        when = "近期"
    reason = info.get("kind") or "故障"
    return (
        f"[!] 席 {info.get('label', '?')} 跳过中（{when} 故障 {reason}，窗口内不重试）"
    )


def forced_line(info: SeatPlan) -> str:
    """全链都在窗内时的强探说明行（不停摆，但要把「这是被迫探测」讲清楚）。"""
    return (
        f"[!] {info.tier} 全部席位都在跳过窗内，按最先到期席再探一次："
        f"{info.usable[0] if info.usable else '?'}"
    )


def summary_head(*, tail: int | None = None, path: Path | None = None) -> str:
    """回退**统计行**（近 N 日 M 起｜累计 K 起）；无事件给「（尚无渠道回退事件）」。

    与 :func:`summary_line` 同源（tui-polish-c 反馈 #7：TUI 回退段要**列表式**渲染，
    故把头行与逐条项拆成两个取数口，而不是让 UI 去切一行长串）。
    """
    all_events = read_events(path=path)
    if not all_events:
        return "（尚无渠道回退事件）"
    recent = read_events(limit=tail, since_days=COUNT_WINDOW_DAYS, path=path)
    head = f"渠道回退：近 {COUNT_WINDOW_DAYS} 日 {len(recent)} 起｜累计 {len(all_events)} 起"
    if not recent:
        return head + "｜窗口内无新增"
    return head


def recent_event_lines(
    *, tail: int | None = None, path: Path | None = None
) -> list[str]:
    """最近几起回退事件的**逐条一行**（供列表式渲染；窗口内无新增 → 空表）。"""
    recent = read_events(limit=tail, since_days=COUNT_WINDOW_DAYS, path=path)
    if not recent:
        return []
    n = int(
        tail
        if tail is not None
        else Registry.load().fallback_policy.get("event_tail", 5)
    )
    shown = recent[-max(1, n) :]
    return [
        f"{str(e.get('ts', ''))[5:10]} {str(e.get('ts', ''))[11:16]} "
        f"{e.get('tier', '?')} {e.get('from_label', '?')}→{e.get('to_label', '?')}"
        f"（{e.get('reason', '?')}）"
        for e in shown
    ]


def summary_line(*, tail: int | None = None, path: Path | None = None) -> str:
    """一行总览（mirror 概览项与 ``orch status`` 判决包健康段共用同一句话）。

    无事件时给一行「（尚无渠道回退）」而不是空串——缺席与零事件在观测面上必须可辨，
    否则读的人分不清「没坏」和「没人记账」。
    """
    head = summary_head(tail=tail, path=path)
    items = recent_event_lines(tail=tail, path=path)
    if not items:
        return head
    return f"{head}\n      最近：{'；'.join(items)}"


__all__ = [
    "COUNT_WINDOW_DAYS",
    "EVENTS_PATH",
    "SeatPlan",
    "alert_line",
    "chain_text",
    "forced_line",
    "note_fallback",
    "note_seat_ok",
    "plan_seats",
    "read_events",
    "skip_line",
    "recent_event_lines",
    "summary_head",
    "summary_line",
]
