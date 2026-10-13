"""主菜单七模块（REFACTOR.md §2.4 归并表）——结构与**只读详情**（m2）。

数据层全部复用 :mod:`..mirror` 的纯读数函数（§2.5：同一能力不留两份实现）；本模块只定义
「每模块读什么、更新怎么算」。详情里的**写入口**（任务书新增/批复、计划推进…）是 m3/m5 的
事，本环节如实标注「占位」。

**更新判据（§2.2，可解释）**：
- **有记录流**的模块（决策/队列/待决…）→ ``cursor`` = **条数**，``updates_since`` 返回条数差
  （徽标数字 = 详情里新增的条数）；
- **无记录流**的状态读数（系统与路由、占位模块）→ ``cursor`` = 读数**指纹**，变了即报 1 条。

更新粒度是「**批量差**」（不是逐条 id）：够满足 §2.2「徽标数字与详情对得上」，且不需要每个
数据源都长出一条稳定的记录 id（§2.2 允许「指纹」这一路）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def _queues_count() -> int:
    """队列水位：待消化条目 + 延期在等内容物。"""
    from .. import workflow as _wf  # 延迟导入：workflow 引 dispatch，顶层引会成环

    try:
        return len(_wf.backlog_pending()) + int(_wf.deferred_count())
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞（§0.6）
        return 0


def _pending_count() -> int:
    """待决水位：待决事项 + 用户行动指令队列。"""
    try:
        return len(mirror.pending_items()) + len(mirror.pending_directives())
    except Exception:  # noqa: BLE001
        return 0


def modules() -> list[Module]:
    """主菜单七模块（§2.4 归并表顺序）；读数与详情全走 mirror 纯函数。"""
    return [
        _fingerprint_module(
            "taskbooks",
            "任务书",
            lambda: ["（占位）m3 建大任务列表 / 新增多行输入 / 批复返工"],
            lambda: [
                "（占位）任务书模块",
                "m3 充实：大任务列表 / 新增多行输入 / 批复返工（REFACTOR §3.4）",
            ],
        ),
        _fingerprint_module(
            "plans",
            "计划与派发",
            lambda: ["（占位）m5 建 active plan 清单 / 环节进度 / 多 plan 并行"],
            lambda: [
                "（占位）计划与派发模块",
                "m5 充实：active plan 清单 / 环节进度 / 多 plan 并行面",
            ],
        ),
        _count_module(
            "queues",
            "队列与运行",
            lambda: [mirror._line_queues(), mirror._line_defer()],
            lambda: [
                mirror._line_queues(),
                mirror._line_defer(),
                "（细节 m3 补）各成员队列水位 / worker 状态 / 延期到期",
            ],
            _queues_count,
        ),
        _count_module(
            "review",
            "盲审与审查",
            lambda: [mirror._line_review(), mirror._line_charter()],
            lambda: [
                mirror._line_review(),
                mirror._line_charter(),
                "（细节 m3 补）twin 待盲审案件 / reviewer 环节附属面；宪章读数归此（组长钉）",
            ],
            mirror.taste_cases,
        ),
        _fingerprint_module(
            "system",
            "系统与路由",
            lambda: [
                mirror._line_switches(),
                mirror._line_routing(),
                mirror._line_jury(),
                mirror._line_fallback(),
            ],
            lambda: [
                mirror._line_switches(),
                mirror._line_routing(),
                mirror._line_jury(),
                mirror._line_fallback(),
                "（细节 m3 补）taste/inspector 开关 / 路由预设 / 席位健康 / 回退事件",
            ],
        ),
        _count_module(
            "decisions",
            "决策审计",
            lambda: [mirror._line_inspector_decisions(), mirror._line_trend()],
            mirror.decisions_detail_lines,  # 现 cmd_decisions 全貌迁入（数据层复用，不复制）
            mirror.decisions_count,
        ),
        _count_module(
            "pending",
            "待决与验收",
            lambda: [mirror._line_pending(), mirror._line_acceptance()],
            lambda: [
                mirror._line_pending(),
                mirror._line_acceptance(),
                "（细节 m3 补）mirror-pending 增删 / 待验收交付批复",
                "口述动作：主菜单按 `j`（偏好陈述 / 行动指令二分，法理同 journal）",
            ],
            _pending_count,
        ),
    ]
