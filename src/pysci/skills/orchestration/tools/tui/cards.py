"""看板卡片块（s7）：队列/系统/待决三看板的**结构化块**（卡＝``card()``、分区＝``section_title()``）。

数据全走既有数据层（``queues``/``workflow``/``benchd``/``drain``/``inspectd``/``events``/
``backpressure``/``mirror``/``routing``/``taskbooks``）——本模块只定义「每张卡写什么」，
不重写任何读数（§2.5）。``tui/modules.py`` 的同名**文本面保留**（徽标指纹与静态读面用），
本模块是**UI 渲染面**：块 → 卡片栈（:meth:`PysciTUI._render_detail_cards`）。

块 schema：``{"kind": "section", "text": str}`` 或
``{"kind": "card", "lines": [str, …], "id": str | None, "highlight": bool}``。
"""

from __future__ import annotations

import time
from typing import Any

from .. import mirror
from .modules import _countdown_line, _hold_line, _safe_line, _safe_lines


def _countdown_card_text() -> str:
    """倒计时行文本（复用 ``tui.modules._countdown_line`` 单一实现；首帧与每秒 tick 共用）。"""
    from .. import routing as _rt

    reading = _rt.state()
    return _countdown_line(reading.next_at, reading.next_mode)


def countdown_text() -> str:
    """倒计时卡文本（供 ``_tick_detail`` 每秒只刷这一张卡）。"""
    return _countdown_card_text()


# ---------------------------------------------------------------------------
# 队列与运行（每成员一卡，含空队；在途条目带成员归属）
# ---------------------------------------------------------------------------
def queues_blocks() -> list[dict[str, Any]]:
    """成员队列（每成员恰一卡）＋ 支撑席存量卡 ＋ 事件水位卡。"""
    from .. import backpressure as _bp
    from .. import benchd as _bd
    from .. import drain as _dr
    from .. import events as _ev
    from .. import inspectd as _insp
    from .. import queues as _q
    from .. import workflow as _wf
    from .. import workq as _wq

    members = _q.all_members()
    blocks: list[dict[str, Any]] = [
        {
            "kind": "section",
            "text": f"成员队列（{len(members)} 队：7 专职 + deputy + reviewer + 计划步骤入队成员）",
        }
    ]
    for m in members:
        c = _q.counts(m)
        pending = c.get(_wq.STATUS_PENDING, 0)
        gip = c.get(_wq.STATUS_IN_PROGRESS, 0)
        done = c.get(_wq.STATUS_DONE, 0)
        failed = c.get(_wq.STATUS_FAILED, 0)
        empty = not (pending or gip or done or failed)
        # 在途明细带**成员归属**（s1 文本面语义照搬：`· [m] 在途 <id>（plan·step）起于 …`）
        lines = [
            f"[{m}] 待={pending} 在做={gip} 完成={done} 失败={failed}"
            + ("  ◀ 有积压" if pending else "")
            + ("  （空）" if empty else "")
        ]
        for it in _q.items(m):
            if str(it.get("status")) != _wq.STATUS_IN_PROGRESS:
                continue
            plan = str(it.get("plan") or "")
            step = str(it.get("step") or "")
            where = f"（{plan}·{step}）" if plan or step else ""
            lines.append(
                f"  · [{m}] 在途 {it.get('id')}{where}"
                f" 起于 {it.get('started_at') or '?'}"
            )
        blocks.append({"kind": "card", "lines": lines, "id": f"card-queue-{m}"})

    # 支撑席存量（与成员队列分列；「支撑席存量」字样保留——test_live_config_panel 钉住）
    sup: list[str] = [
        "支撑席存量（与上面分列，成员队列皆空 ≠ 支撑席也空；口径各报各的数）"
    ]
    try:
        member_pending = sum(_q.counts(m).get("pending", 0) for m in _q.all_members())
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞
        member_pending = "?"
    try:
        insp_pending = _wq.pending_count(_insp.QUEUE_PATH)
    except Exception:  # noqa: BLE001
        insp_pending = "?"
    sup.append(
        f"成员队列 pending={member_pending}｜bencher 队列 pending="
        f"{_bd.queue_counts().get('pending', 0)}｜inspector 队列 pending={insp_pending}"
    )
    sup.append(
        f"devops backlog：pending={len(_wf.backlog_pending())}"
        f"  needs_leader={len(_wf.backlog_by_status('needs_leader'))}"
        f"  锁={_dr.lock_status()['state']}"
    )
    sup.append(
        f"bencher 队列={_bd.queue_counts()}  锁={_dr.lock_status(_bd.BENCH_PATHS)['state']}"
    )
    sup.append(
        f"inspector 开关={_insp.enabled()}  锁={_dr.lock_status(_insp.INSPECT_PATHS)['state']}"
        f"  积压={_insp.depths()}"
    )
    if not _insp.enabled():
        sup.append(
            "  （开关关闭：不拉起巡检、其存量只报不告警——PENDING-WORK 不因它唤醒）"
        )
    stale = _safe_lines(lambda: _insp.stale_queue_items())
    sup.append(
        f"inspector 队列超时 = {len(stale)} 项（入队>{_insp.QUEUE_TIMEOUT_S}s 未处理）"
    )
    sup += [
        f"  · {it['id']} [{it.get('kind')}] 入队于 {it.get('enqueued_at')}"
        for it in stale[:3]
    ]
    sup.append(
        f"活跃会话（state/active-sessions.json）live={_bp.active_count(now=time.time())}"
    )
    sup.append(_safe_line(mirror._line_defer))
    blocks.append({"kind": "card", "lines": sup, "id": "card-support"})

    blocks.append({"kind": "section", "text": "事件水位（state/events.jsonl）"})
    blocks.append(
        {
            "kind": "card",
            "lines": _safe_lines(_ev.status_lines),
            "id": "card-events",
        }
    )
    return blocks


# ---------------------------------------------------------------------------
# 系统与路由（首卡＝倒计时，每秒只刷它）
# ---------------------------------------------------------------------------
def system_blocks() -> list[dict[str, Any]]:
    """① 倒计时卡（置顶、高亮）→ ② 开关/路由 → ③ 配置生效值 → ④ 仲裁/回退 → ⑤ 告警（有才出）。"""
    from .. import channels as _ch
    from .. import routing as _rt

    fv = _rt.face()
    blocks: list[dict[str, Any]] = [
        {
            "kind": "card",
            "lines": [countdown_text()],
            "id": "detail-countdown",
            "highlight": True,
        },
        {
            "kind": "card",
            "lines": [
                f"开关：{_safe_line(mirror._line_switches)}",
                f"路由：{_safe_line(mirror._line_routing)}",
            ],
            "id": "card-switches",
        },
        {
            "kind": "card",
            "lines": [
                "路由配置生效值",
                f"  入峰提前量 leadtime_s = {fv.leadtime_s}s（{fv.leadtime_s // 60} 分钟）"
                f"  〔真源：{fv.source('leadtime_s')}〕",
                f"  峰段 peak = {fv.windows_text()}  〔真源：{fv.source('peak')}〕",
                f"  谷时预设 valley_preset = {fv.valley_preset}"
                f"  〔真源：{fv.source('valley_preset')}〕",
                _hold_line(fv),
            ],
            "id": "card-config",
        },
        {
            "kind": "card",
            "lines": [
                f"仲裁：{_safe_line(mirror._line_jury)}",
                f"回退：{_safe_line(mirror._line_fallback)}",
                f"回退事件累计 = {len(_ch.read_events())} 起（state/channel-events.jsonl）",
            ],
            "id": "card-arbitration",
        },
    ]
    if fv.warnings:
        blocks.append(
            {
                "kind": "card",
                "lines": ["配置告警", *[f"  · {w}" for w in fv.warnings]],
                "id": "card-warnings",
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# 待决与验收（待决事项 / 待批复交付 / 偏差报告）
# ---------------------------------------------------------------------------
def pending_blocks() -> list[dict[str, Any]]:
    """三张卡：待决事项、待批复交付（delivered）、偏差报告（强制处理队列）。"""
    from .. import taskbooks as _tb

    items = _safe_lines(mirror.pending_items)
    blocks: list[dict[str, Any]] = [
        {
            "kind": "card",
            "lines": [
                f"待决事项（state/mirror-pending.json）= {len(items)}",
                *([f"  · {x}" for x in items] or ["  （无待决）"]),
            ],
            "id": "card-pending",
        }
    ]
    review: list[dict[str, Any]] = []
    try:
        review = _tb.sections()["review"]
    except Exception:  # noqa: BLE001 - 读数坏不塌看板
        review = []
    blocks.append(
        {
            "kind": "card",
            "lines": [
                f"待批复交付（taskbooks delivered）= {len(review)}",
                *[f"  · [{d.get('status')}] {d.get('title')}" for d in review[:5]],
                f"  {_safe_line(mirror._line_acceptance)}",
            ],
            "id": "card-review",
        }
    )
    blocks.append(
        {
            "kind": "card",
            "lines": [
                "偏差报告（强制处理队列）",
                *(_safe_lines(mirror.deviation_lines) or ["  （无）"]),
            ],
            "id": "card-deviations",
        }
    )
    return blocks
