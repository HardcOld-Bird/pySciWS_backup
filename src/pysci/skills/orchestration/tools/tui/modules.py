"""主菜单七模块（README §12 归并表）——结构与**只读详情**（f1 充实）。

数据层全部复用既有纯读数函数（§2.5：同一能力不留两份实现）——:mod:`..mirror` 的概览/全貌
读数、:mod:`..plans` 的计划态、:mod:`..queues`+`workq` 的队列水位、:mod:`..events` 的事件水位、
:mod:`..taskbooks` 的任务书分段、:mod:`..backpressure`/`drain`/`benchd`/`inspectd` 的 worker 读数。
本模块只定义「每模块读什么、更新怎么算」，**不重写任何读数**。

写入口不在这里（任务书面板在 :mod:`.app` 的 `t`/Enter；计划推进走 orch）——详情只读出
「真实现状 + 入口指向」；确未实现的面如实写「后续单」并指名真实归属，不写环节承诺。

**更新判据（§2.2，可解释）**：
- **有记录流**的模块（决策/队列/待决…）→ ``cursor`` = **条数**，``updates_since`` 返回条数差
  （徽标数字 = 详情里新增的条数）；
- **无记录流**的状态读数（系统与路由、计划与派发）→ ``cursor`` = 读数**指纹**，变了即报 1 条。

更新粒度是「**批量差**」（不是逐条 id）：够满足 §2.2「徽标数字与详情对得上」，且不需要每个
数据源都长出一条稳定的记录 id（§2.2 允许「指纹」这一路）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import mirror
from .watermark import fingerprint


@dataclass(frozen=True)
class Module:
    """主菜单一行：稳定 ``key`` + ``title`` + 读数/详情/水位四条钩子。"""

    key: str
    title: str
    summary: Callable[[], list[str]]
    detail: Callable[[], list[str]]
    cursor: Callable[[], str]
    updates_since: Callable[[str], int]


def _fingerprint_module(
    key: str,
    title: str,
    summary: Callable[[], list[str]],
    detail: Callable[[], list[str]],
) -> Module:
    """无记录流模块：水位=详情读数指纹，相对旧水位变了即报 1 条。"""

    def cursor() -> str:
        return fingerprint(detail())

    def updates_since(old: str) -> int:
        cur = cursor()
        return 0 if (not old or old == cur) else 1

    return Module(key, title, summary, detail, cursor, updates_since)


def _count_module(
    key: str,
    title: str,
    summary: Callable[[], list[str]],
    detail: Callable[[], list[str]],
    count: Callable[[], int],
) -> Module:
    """有记录流模块：水位=条数，相对旧水位的**条数差**即新增（首次未看过 → 全量）。"""

    def cursor() -> str:
        return str(count())

    def updates_since(old: str) -> int:
        try:
            base = int(old or 0)
        except ValueError:
            base = 0
        return max(0, int(count()) - base)

    return Module(key, title, summary, detail, cursor, updates_since)


# ---------------------------------------------------------------------------
# 通用：读数 fail-soft（观察面绝不阻塞，§0.6）
# ---------------------------------------------------------------------------
def _safe_line(fn: Callable[[], str]) -> str:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"[!] 读数失败：{type(exc).__name__}: {exc}"


def _safe_lines(fn: Callable[[], list[str]]) -> list[str]:
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001
        return [f"[!] 读数失败：{type(exc).__name__}: {exc}"]


# ---------------------------------------------------------------------------
# 任务书（§2.4 #1 / §3.4）——写入口在 TUI 面板（Enter/`t`），此处只读
# ---------------------------------------------------------------------------
def _taskbook_secs() -> dict:
    from .. import taskbooks as _tb  # 延迟导入：taskbooks 是叶子数据层

    return _tb.sections()


def _taskbook_summary() -> list[str]:
    secs = _taskbook_secs()
    return [
        f"现有大任务 = {len(secs['open'])}",
        f"待批复（delivered） = {len(secs['review'])}",
        f"历史（accepted/带批语） = {len(secs['history'])}",
    ]


def _taskbook_detail() -> list[str]:
    """列表三段（§2.4 #1 / §3.4）；写入口在 TUI 面板（`t`/Enter），此处只读。"""
    secs = _taskbook_secs()
    out: list[str] = []
    for name, key in (
        ("现有大任务（new/planning/active）", "open"),
        ("已完成待批复（delivered）", "review"),
        ("历史（accepted / 带驳回批语）", "history"),
    ):
        out.append(f"—— {name} ——")
        items = secs[key]
        if not items:
            out.append("  （无）")
            continue
        for d in items:
            rej = d.get("rejections") or []
            tail = ""
            if rej:
                tail = f"｜批语：{str(rej[-1].get('note', ''))[:40]}"
            elif d.get("plan_id"):
                tail = f"｜plan={d['plan_id']}"
            out.append(f"  [{d.get('status')}] {d.get('title')}{tail}")
    out += [
        "",
        "Enter / `t` 进面板块：`n` 新增 / `a` 接受选中 / `r` 驳回选中（批语输入框）",
    ]
    return out


def _taskbook_count() -> int:
    """更新徽标：待批复（delivered）+ 新投递（new）即「值得关注」（§3.4）。"""
    try:
        secs = _taskbook_secs()
        return len(secs["review"]) + sum(
            1 for d in secs["open"] if d.get("status") == "new"
        )
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞
        return 0


# ---------------------------------------------------------------------------
# 计划与派发（§2.4 #2）——复用 plans.list_plans / load_state
# ---------------------------------------------------------------------------
def _plan_mark(state: dict) -> str:
    if state.get("completed_at"):
        return "✔完成"
    if state.get("abandoned_at"):
        return "⊘中止"
    return "▶进行中"


def _plans_rows() -> list[tuple[str, str, int, int]]:
    from .. import plans as _plans  # 延迟导入：plans 引 dispatch/ledger

    return _plans.list_plans()


def _plan_state(pid: str) -> dict:
    from .. import plans as _plans

    return _plans.load_state(pid)


def _plans_summary() -> list[str]:
    try:
        rows = _plans_rows()
    except Exception as exc:  # noqa: BLE001
        return [f"[!] 计划读数失败：{type(exc).__name__}: {exc}"]
    if not rows:
        return ["计划 = （暂无；orch plan new 登记）"]
    n_active = sum(1 for pid, *_ in rows if _plan_mark(_plan_state(pid)) == "▶进行中")
    out = [f"计划数 = {len(rows)}（进行中 {n_active}）"]
    for pid, goal, done, total in rows[:5]:
        out.append(
            f"  {_plan_mark(_plan_state(pid))} {pid}  {done}/{total}  {goal[:28]}"
        )
    return out


def _plans_detail() -> list[str]:
    """active plan 清单 + 当前环节/进度 + 环节 session/task_file 关联 + 多 plan 并行读数。"""
    out = ["—— 计划清单（plans.list_plans / load_state）——"]
    try:
        rows = _plans_rows()
    except Exception as exc:  # noqa: BLE001
        return out + [f"[!] 读数失败：{type(exc).__name__}: {exc}"]
    if not rows:
        return out + ["  （暂无计划；orch plan new <yaml> 登记）"]
    for pid, goal, done, total in rows:
        st = _plan_state(pid)
        out.append(f"  {_plan_mark(st)} {pid}  {done}/{total}  {goal[:40]}")
        steps = st.get("steps") or []
        cur = next(
            (s for s in steps if s.get("status") not in ("done", "skipped")), None
        )
        if cur:
            sid = str(cur.get("session_id") or "-")[:8] or "-"
            tf = Path(str(cur.get("task_file") or "")).name or "-"
            out.append(
                f"      当前环节 {cur.get('id')}｜{cur.get('status')}"
                f"｜会话 {sid}｜任务书 {tf}"
            )
        for s in steps:
            mark = "✔" if s.get("status") in ("done", "skipped") else "·"
            review = (
                "  [review]" if s.get("review") not in (None, "pending", False) else ""
            )
            out.append(f"        {mark} {s.get('id')} [{s.get('status')}]{review}")
    out += [
        "",
        "推进：`orch plan advance <id>`（事件驱动，入队即返回）｜`orch plan run <id>`（兼容阻塞）",
        "多 plan 并行：同成员队列 FIFO 串行、异成员并行（上表逐 plan 列出即并行面）",
    ]
    return out


# ---------------------------------------------------------------------------
# 队列与运行（§2.4 #3）——复用 queues/events/workflow/drain/benchd/inspectd/backpressure
# ---------------------------------------------------------------------------
def _queues_summary() -> list[str]:
    return [mirror._line_queues(), mirror._line_defer()]


def _queues_detail() -> list[str]:
    """各成员队列水位 + 支撑席 worker 存活 + 延期到期 + 事件水位（全走既有读数）。"""
    from .. import backpressure as _bp
    from .. import benchd as _bd
    from .. import drain as _dr
    from .. import events as _ev
    from .. import inspectd as _insp
    from .. import queues as _q
    from .. import workflow as _wf
    from .. import workq as _wq

    out = ["—— 成员队列（9 成员：7 专职 + deputy + reviewer；state/queues/*.json）——"]
    out += _safe_lines(_q.water_lines)
    out.append(f"  {_safe_line(mirror._line_defer)}")
    out.append("—— 支撑席存量（与上面分列，**成员队列皆空 ≠ 支撑席也空**）——")
    # 口径拆分（用户 2026-10-14「真的为空吗」）：三面各报各的数，不再共用一句「皆空」。
    try:
        member_pending = sum(_q.counts(m).get("pending", 0) for m in _q.all_members())
    except Exception:  # noqa: BLE001
        member_pending = "?"
    try:
        insp_pending = _wq.pending_count(_insp.QUEUE_PATH)
    except Exception:  # noqa: BLE001
        insp_pending = "?"
    out.append(
        f"  口径：9 成员队列 pending={member_pending} ｜"
        f"bencher 队列 pending={_bd.queue_counts().get('pending', 0)}"
        f"｜inspector 队列 pending={insp_pending}"
    )
    out.append(
        "  devops backlog：pending="
        f"{len(_wf.backlog_pending())}"
        f"  needs_leader={len(_wf.backlog_by_status('needs_leader'))}"
        f"  锁={_dr.lock_status()['state']}"
    )
    out.append(
        f"  bencher 队列={_bd.queue_counts()}  锁={_dr.lock_status(_bd.BENCH_PATHS)['state']}"
    )
    out.append(
        f"  inspector 开关={_insp.enabled()}  锁={_dr.lock_status(_insp.INSPECT_PATHS)['state']}"
        f"  积压={_insp.depths()}"
    )
    if not _insp.enabled():
        out.append(
            "      （开关关闭：不拉起巡检、其存量**只报不告警**——PENDING-WORK 不因它唤醒）"
        )
    stale = _safe_lines(lambda: _insp.stale_queue_items())
    out.append(
        f"  inspector 队列超时 = {len(stale)} 项（入队>{_insp.QUEUE_TIMEOUT_S}s 未处理；"
        "已发 QUEUE-TIMEOUT 事件，不许静默积压）"
    )
    for it in stale[:3]:
        out.append(
            f"      · {it['id']} [{it.get('kind')}] 入队于 {it.get('enqueued_at')}"
        )
    out.append(
        f"  活跃会话（state/active-sessions.json）live={_bp.active_count(now=time.time())}"
    )
    out.append("—— 事件水位（state/events.jsonl）——")
    out += _safe_lines(_ev.status_lines)
    return out


def _queues_count() -> int:
    """队列水位：待消化条目 + 延期在等内容物 + **超时未处理项**（后者要「值得关注」）。"""
    from .. import inspectd as _insp  # 延迟导入：inspectd 顶层引整条 worker 链
    from .. import workflow as _wf  # 延迟导入：workflow 引 dispatch，顶层引会成环

    try:
        return (
            len(_wf.backlog_pending())
            + int(_wf.deferred_count())
            + len(_insp.stale_queue_items())
        )
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞（§0.6）
        return 0


# ---------------------------------------------------------------------------
# 盲审与审查（§2.4 #4）——twin 盲审 / reviewer 附属 / 宪章（组长钉：宪章归此）
# ---------------------------------------------------------------------------
def _review_summary() -> list[str]:
    return [mirror._line_review(), mirror._line_charter()]


def _review_detail() -> list[str]:
    from .. import queues as _q

    try:
        c = _q.counts("reviewer")
        ctext = (
            f"待={c.get('pending', 0)} 在做={c.get('in_progress', 0)}"
            f" 完成={c.get('done', 0)} 失败={c.get('failed', 0)}"
        )
    except Exception as exc:  # noqa: BLE001
        ctext = f"[!] 读数失败：{type(exc).__name__}: {exc}"
    from .. import inspectd as _insp

    stale = _safe_lines(lambda: _insp.stale_queue_items())
    return [
        "—— 盲审与审查 ——",
        f"  {_safe_line(mirror._line_review)}",
        f"  {_safe_line(mirror._line_charter)}",
        f"  reviewer 附属队列 = {ctext}（reviewer 队条目）",
        f"  审查面（inspector 队列）超时项 = {len(stale)}（入队>{_insp.QUEUE_TIMEOUT_S}s；"
        "已发 QUEUE-TIMEOUT 事件）",
        "  审查入口：`orch review <产物> --origin <member>`；宿主环节 `review:true` 时",
        "  由其交付自动投 reviewer 附属子 job（父子联动见 `orch queues list`）",
    ]


# ---------------------------------------------------------------------------
# 系统与路由（§2.4 #5）——开关/路由/仲裁/回退 多行（现 summary 行扩成详情）
# ---------------------------------------------------------------------------
#: 倒计时行标记（每秒都在变）：徽标指纹**不算**它——算进去徽标会常亮，违反 §2.2
#: 「徽标数字与详情对得上」。倒计时只由 :meth:`.app._tick_detail` 就地刷新。
_COUNTDOWN_MARK = "倒计时"


def _system_summary() -> list[str]:
    return [
        mirror._line_switches(),
        mirror._line_routing(),
        mirror._line_jury(),
        mirror._line_fallback(),
    ]


def _countdown_line(next_at: str, next_mode: str) -> str:
    """倒计时行：距下次切换 X 分 Y 秒（含时刻与目标模式；每秒由 TUI tick 重算）。"""
    from .. import routing as _rt

    left = _rt.countdown_s(next_at)
    if left is None:
        return "  倒计时：下次切换判不出（保守维持当前模式）"
    sign, left = ("", left) if left >= 0 else ("已过 ", -left)
    return (
        f"  倒计时：距下次切换还有{sign}{left // 60} 分 {left % 60} 秒"
        f"（{next_at} → {next_mode}）"
    )


def _hold_line(fv: Any) -> str:
    """边界避让读数：窗口秒数 + 「临近边界将延迟派发」状态 + 倒计时。"""
    from .. import routing as _rt

    hold = fv.boundary_hold_s
    if hold is None:
        return "  边界避让：toml 未配置 → 回退 registry 旧位（请迁进 data/pysci_settings.toml）"
    if hold <= 0:
        return "  边界避让：已关闭（0s，切换点前照常派发）"
    reading = _rt.state()
    left = _rt.countdown_s(reading.next_at)
    if left is None:
        return f"  边界避让：窗口 {hold}s（下次切换判不出 → 暂不延迟）"
    if 0 <= left < hold:
        return (
            f"  边界避让：窗口 {hold}s〔临近边界将延迟派发〕"
            f"倒计时：距边界 {left // 60} 分 {left % 60} 秒（{reading.next_mode}）"
        )
    return f"  边界避让：窗口 {hold}s（距边界 {max(left, 0) // 60} 分，窗口外不延迟）"


def _system_detail() -> list[str]:
    from .. import routing as _rt

    fv = _rt.face()
    reading = _rt.state()
    out = [
        "—— 系统与路由 ——",
        f"  开关：{_safe_line(mirror._line_switches)}",
        f"  路由：{_safe_line(mirror._line_routing)}",
        "  路由配置生效值 ——",
        f"    入峰提前量 leadtime_s = {fv.leadtime_s}s（{fv.leadtime_s // 60} 分钟）"
        f"  〔真源：{fv.source('leadtime_s')}〕",
        f"    峰段 peak = {fv.windows_text()}  〔真源：{fv.source('peak')}〕",
        f"    谷时预设 valley_preset = {fv.valley_preset}"
        f"  〔真源：{fv.source('valley_preset')}〕",
        _hold_line(fv),
        _countdown_line(reading.next_at, reading.next_mode),
        f"  仲裁：{_safe_line(mirror._line_jury)}",
        f"  回退：{_safe_line(mirror._line_fallback)}",
    ]
    from .. import channels as _ch

    out.append(
        f"  回退事件累计 = {len(_ch.read_events())} 起（state/channel-events.jsonl）"
    )
    if fv.warnings:
        out.append("  [!] 配置告警：")
        out.extend(f"      · {w}" for w in fv.warnings)
    return out


def _system_stable() -> list[str]:
    """系统与路由的**稳定**读数（去掉每秒在变的倒计时行）——徽标指纹用它，防常亮。"""
    return [ln for ln in _system_detail() if _COUNTDOWN_MARK not in ln]


# ---------------------------------------------------------------------------
# 决策审计（§2.4 #6）——现 cmd_decisions 全貌迁入（数据层复用，不复制）
# ---------------------------------------------------------------------------
def _decisions_summary() -> list[str]:
    return [mirror._line_inspector_decisions(), mirror._line_trend()]


# ---------------------------------------------------------------------------
# 待决与验收（§2.4 #7）——mirror-pending / 待批复交付 / 批复入口指向
# ---------------------------------------------------------------------------
def _pending_summary() -> list[str]:
    return [mirror._line_pending(), mirror._line_acceptance()]


def _pending_detail() -> list[str]:
    from .. import taskbooks as _tb

    out = ["—— 待决与验收 ——"]
    items = _safe_lines(mirror.pending_items)
    out.append(f"  待决事项（state/mirror-pending.json）= {len(items)}")
    out += [f"    · {x}" for x in items] if items else ["    （无待决）"]
    try:
        review = _tb.sections()["review"]
    except Exception as exc:  # noqa: BLE001
        review = []
        out.append(f"  [!] 任务书读数失败：{type(exc).__name__}: {exc}")
    out.append(f"  待批复交付（taskbooks delivered）= {len(review)}")
    for d in review[:5]:
        out.append(f"    · [{d.get('status')}] {d.get('title')}")
    out.append(f"  {_safe_line(mirror._line_acceptance)}")
    out.append("  批复入口：Enter「任务书」→ 选中 → `a` 接受 / `r` 驳回（批语输入框）")
    out.append("  口述动作：主菜单 `j`（偏好陈述 / 行动指令二分，法理同 journal）")
    return out


def _pending_count() -> int:
    """待决水位：待决事项 + 用户行动指令队列。"""
    try:
        return len(mirror.pending_items()) + len(mirror.pending_directives())
    except Exception:  # noqa: BLE001
        return 0


def modules() -> list[Module]:
    """主菜单七模块（§2.4 归并表顺序）；读数与详情全走既有数据层纯函数。"""
    return [
        _count_module(
            "taskbooks",
            "任务书",
            _taskbook_summary,
            _taskbook_detail,
            _taskbook_count,
        ),
        _fingerprint_module("plans", "计划与派发", _plans_summary, _plans_detail),
        _count_module(
            "queues", "队列与运行", _queues_summary, _queues_detail, _queues_count
        ),
        _count_module(
            "review",
            "盲审与审查",
            _review_summary,
            _review_detail,
            mirror.taste_cases,
        ),
        _fingerprint_module("system", "系统与路由", _system_summary, _system_stable),
        _count_module(
            "decisions",
            "决策审计",
            _decisions_summary,
            mirror.decisions_detail_lines,  # 现 cmd_decisions 全貌迁入（数据层复用，不复制）
            mirror.decisions_count,
        ),
        _count_module(
            "pending", "待决与验收", _pending_summary, _pending_detail, _pending_count
        ),
    ]
