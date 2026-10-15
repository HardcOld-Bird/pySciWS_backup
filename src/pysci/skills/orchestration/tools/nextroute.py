"""跨任务路由（PSEP_3 §3）：聚合全部活动 plan/任务书的**就绪动作** → 排序 → 任务卡片。

用户裁决⑤⑥：多大任务并行时按**优先级**决定推进顺序；一切引导输出必须携带**大任务名称/
简介/plan 内位置**（机械流程强推，不能只靠概览面板）。本模块是该路由的**数据层 + 渲染层**
唯一实现：``orch next`` 命令与四个机械推送点（advance/drain/watch/run 交接）共用它。

排序键（§3.2）：**（任务优先级 high>normal>low → 动作类别 blocked 决策>待验收>可派发 →
plan/step 序）**。无关联任务书的 plan 按 ``normal`` 处理。
"""

from __future__ import annotations

from typing import Any

#: 动作类别权重（越小越先，§3.2）。
CATEGORY_ORDER: dict[str, int] = {"blocked": 0, "review": 1, "dispatch": 2}
CATEGORY_LABEL: dict[str, str] = {
    "blocked": "blocked 决策",
    "review": "待验收",
    "dispatch": "可派发",
}
#: 优先级权重（越小越先）。
PRIORITY_ORDER: dict[str, int] = {"high": 0, "normal": 1, "low": 2}
#: 返工模式 → 中文标签（卡片展示）。
MODE_LABEL: dict[str, str] = {
    "full": "完整",
    "no_inspector": "无监察改进",
    "rework_only": "仅返工",
}


def _intro(tb_id: str) -> str:
    """任务书一行简介（正文首个非空非标题行，截 60 字）。"""
    from . import taskbooks as _tb

    for line in (_tb.read_body(tb_id) or "").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s[:60]
    return ""


def _plan_taskbook(plan_id: str, plan: dict) -> tuple[str, dict]:
    """plan ↔ taskbook 关联：``plan.taskbook`` 优先，回落 ``tb.plan_id == plan_id`` 反查。"""
    from . import taskbooks as _tb

    tb_id = str(plan.get("taskbook") or "")
    st = _tb.load_state(tb_id) if tb_id else None
    if st:
        return tb_id, st
    for st in _tb.list_all():
        if str(st.get("plan_id") or "") == plan_id:
            return str(st.get("id") or ""), st
    return "", {}


def _position(plan: dict, leaf_path: str) -> str:
    """plan 内位置（树路径，§3.2）：``A 综合项目改进 > A.a1 成因分析报告``。"""
    from .planflow import PlanTree

    by = {n.path: n for n in PlanTree(plan.get("steps") or []).nodes}
    parts: list[str] = []
    segs = leaf_path.split(".")
    for i in range(1, len(segs) + 1):
        node = by.get(".".join(segs[:i]))
        if node is None:
            continue
        brief = str(node.yaml.get("brief") or "")
        parts.append(f"{node.name} {brief}".strip())
    return " > ".join(parts)


def _plan_progress(plan: dict, state: dict) -> dict[str, int]:
    """plan 级进度读数（叶子 done/总、在途、待验收）——卡片「叶子进度 k/n」用。"""
    statuses = [str(s.get("status") or "pending") for s in state.get("steps", [])]
    return {
        "total": len(statuses),
        "done": sum(1 for s in statuses if s in ("done", "skipped")),
        "inflight": sum(1 for s in statuses if s in ("queued", "self_awaiting")),
        "need_leader": sum(
            1 for s in statuses if s in ("blocked", "failed", "run_failed")
        ),
    }


def _mode_tr(plan: dict, tb: dict) -> str:
    """模式标签：返工 → 中文三档 + 第 N 轮；非返工 →「单任务」。"""
    mode = str(plan.get("rework_mode") or tb.get("mode") or "")
    rnd = int(plan.get("rework_round") or tb.get("rework_round") or 1)
    if mode:
        return f"{MODE_LABEL.get(mode, mode)}／第 {rnd} 轮返工"
    return "单任务"


def collect_actions() -> list[dict[str, Any]]:
    """聚合全部活动 plan 的就绪动作 + 任务书待办（§3.2）。"""
    from . import plans as _plans
    from . import taskbooks as _tb

    actions: list[dict[str, Any]] = []
    linked_tb: set[str] = set()  # 已被 plan 关联（plan.taskbook）的任务书：不重复进待办
    for pid, _goal, _done, _total in _plans.list_plans():
        plan = _plans._load_yaml(pid)
        state = _plans._load_state(pid)
        if state.get("completed_at") or state.get("abandoned_at"):
            continue  # 非活动 plan
        tb_id, tb = _plan_taskbook(pid, plan)
        if tb_id:
            linked_tb.add(tb_id)
        prio = str(tb.get("priority") or "normal")
        prog = _plan_progress(plan, state)
        panes = _plans._worklist_panes(plan, state)
        base = {
            "priority": prio,
            "tb_id": tb_id,
            "tb_title": str(tb.get("title") or plan.get("goal") or pid),
            "tb_intro": _intro(tb_id) if tb_id else str(plan.get("goal") or "")[:60],
            "mode": _mode_tr(plan, tb),
            "plan_id": pid,
            "progress": prog,
        }
        for sid in panes["need_leader"]:
            actions.append(
                {
                    **base,
                    "category": "blocked",
                    "step_id": sid,
                    "position": _position(plan, sid),
                    "action": "blocked 决策：补充重派 / consult 副组长 / 上报用户",
                    "command": f'pysci-orch plan advance {pid} --text "<补充信息>"',
                }
            )
        disp = panes["dispatchable"]
        if disp:  # 一次引导一个（§3.2）
            sid = disp[0]
            actions.append(
                {
                    **base,
                    "category": "dispatch",
                    "step_id": sid,
                    "position": _position(plan, sid),
                    "action": "书写本环节任务书并派发",
                    "command": f'pysci-orch plan advance {pid} --text "<任务书>"',
                }
            )
    # 任务书待办（无 plan 关联者）：delivered → 待验收；new → 待拆解派发
    for st in _tb.list_all():
        if st.get("plan_id") or str(st.get("id")) in linked_tb:
            continue  # 已入 plan 的由 plan 面覆盖（不重复计数）
        tb_id = str(st.get("id") or "")
        status = str(st.get("status") or "")
        common = {
            "priority": str(st.get("priority") or "normal"),
            "tb_id": tb_id,
            "tb_title": str(st.get("title") or tb_id),
            "tb_intro": _intro(tb_id),
            "mode": _mode_tr({}, st),
            "plan_id": "",
            "progress": {"total": 0, "done": 0, "inflight": 0, "need_leader": 0},
            "step_id": "",
            "position": "",
        }
        if status == "delivered":
            actions.append(
                {
                    **common,
                    "category": "review",
                    "action": "用户验收大报告（accept / reject）",
                    "command": "pysci（TUI）→ 任务书 → a 接受 / r 驳回",
                }
            )
        elif status == "new":
            if st.get(
                "rework_of"
            ):  # 返工件：下一步是生成返工骨架，不是 plan new（引导缺口实锤）
                mode = str(st.get("mode") or "full")
                actions.append(
                    {
                        **common,
                        "category": "dispatch",
                        "action": "生成返工骨架并登记（plan rework）",
                        "command": (
                            f"pysci-orch plan rework {tb_id} --mode {mode}"
                            f" && pysci-orch taskbooks link {tb_id} --plan rework-{tb_id}"
                        ),
                    }
                )
            else:
                actions.append(
                    {
                        **common,
                        "category": "dispatch",
                        "action": "拆解新任务书（建 plan 并 link）",
                        "command": f"pysci-orch plan new <file> && pysci-orch taskbooks link {tb_id} --plan <pid>",
                    }
                )
    return actions


def sort_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """排序：**优先级 → 动作类别 → plan/step 序**（§3.2）。"""
    return sorted(
        actions,
        key=lambda a: (
            PRIORITY_ORDER.get(str(a.get("priority")), 1),
            CATEGORY_ORDER.get(str(a.get("category")), 9),
            str(a.get("plan_id") or ""),
            str(a.get("step_id") or ""),
        ),
    )


def render_card(a: dict[str, Any]) -> list[str]:
    """一张**任务卡片**（PSEP_3 §3.2 格式原文）。"""
    tb_id = str(a.get("tb_id") or "（无关联任务书）")
    head = (
        f"┌─【大任务 {tb_id}：{a.get('tb_title')}】"
        f"优先级={a.get('priority')} 模式={a.get('mode')}"
    )
    loc = f"plan {a.get('plan_id')}" if a.get("plan_id") else "（无 plan）"
    if a.get("position"):
        loc += f" @ {a['position']}"
    prog = a.get("progress") or {}
    if prog.get("total"):
        loc += (
            f"（叶子进度 {prog.get('done')}/{prog.get('total')}，"
            f"在途 {prog.get('inflight')}，待验收 {prog.get('need_leader')}）"
        )
    return [
        head,
        f"│ 简介：{a.get('tb_intro') or '（无）'}",
        f"│ 位置：{loc}",
        f"│ 动作：{a.get('action')}",
        f"└─ 命令模板：{a.get('command')}",
    ]


def next_lines(limit: int = 5) -> list[str]:
    """``orch next`` 全文：置顶一张任务卡片 + 其余就绪项简表（§3.2）。"""
    actions = sort_actions(collect_actions())
    head = f"== orch next：跨任务就绪路由（{len(actions)} 项；PSEP_3 §3.2）=="
    if not actions:
        return [head, "（无就绪动作：全部 plan 空闲、任务书待办为空）"]
    out = [head, *render_card(actions[0])]
    rest = actions[1:]
    if rest:
        out.append(f"—— 其余就绪项（{len(rest)}）——")
        shown = rest[: max(0, limit - 1)]
        for a in shown:
            where = a.get("position") or "（任务书级）"
            out.append(
                f"  · [{CATEGORY_LABEL.get(str(a['category']), a['category'])}]"
                f" {a['tb_title']}｜{where}"
            )
        if len(rest) > len(shown):
            out.append(
                f"  …（还有 {len(rest) - len(shown)} 项；orch next --limit N 看更多）"
            )
    return out


def card_lines(plan_id: str | None = None, *, limit: int = 3) -> list[str]:
    """**单个 plan** 的路由卡片（四个机械推送点用）；``plan_id=None`` 取全局首项。

    返回空表＝无就绪动作（调用方自行决定是否提示）。
    """
    actions = sort_actions(collect_actions())
    if plan_id is not None:
        actions = [a for a in actions if str(a.get("plan_id")) == plan_id]
    if not actions:
        return []
    out = ["[NEXT] 路由卡片：", *render_card(actions[0])]
    if plan_id is None and len(actions) > 1:
        out.append(f"  （其余 {len(actions) - 1} 项见 orch next）")
    del limit
    return out


def cmd_next(args: Any) -> int:
    """``orch next``：跨 plan 路由器（§3.2）——置顶任务卡片 + 其余简表。"""
    limit = int(getattr(args, "limit", 0) or 5)
    for line in next_lines(limit=limit):
        print(line)
    return 0
