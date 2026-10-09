"""plan-runner：计划驱动的任务流程状态机（README §4.2）。

计划 = ``state/plans/<id>.plan.yaml``（组长可直接编辑——amend 即编辑此文件，
run 时按环节 id 对账）；执行状态 = ``state/plans/<id>.state.json``（orch 专属）。
核心原则：**计划即状态机，聊天记录不是**——组长会话被压缩/重开后 plan run 仍从
断点继续；交接处暂停，精准 recall 下一环节粗描述原文。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from pysci.paths import ORCH_STATE_ROOT

from .dispatch import DispatchOutcome, do_dispatch
from .ledger import read_all, summarize

PLANS_DIR = ORCH_STATE_ROOT / "plans"

_DONE = ("done", "skipped")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _plan_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}.plan.yaml"


def _state_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}.state.json"


def _load_yaml(plan_id: str) -> dict:
    return yaml.safe_load(_plan_path(plan_id).read_text(encoding="utf-8")) or {}


def _load_state(plan_id: str) -> dict:
    p = _state_path(plan_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


#: 公开别名（门面 status 命令读取计划状态用）。
load_state = _load_state


def _save_state(plan_id: str, state: dict) -> None:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(plan_id).write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _reconcile(plan: dict, state: dict) -> dict:
    """按环节 id 对账 YAML（可能被组长 amend 过）与既有状态。

    保留已有环节状态；新环节置 pending；YAML 中已删除的环节从状态中移除。
    """
    steps_yaml = plan.get("steps") or []
    old = {s["id"]: s for s in state.get("steps", [])}
    steps: list[dict] = []
    for i, sy in enumerate(steps_yaml):
        sid = str(sy.get("id") or f"s{i + 1}")
        prev = old.get(sid)
        steps.append(
            {
                "id": sid,
                "status": prev["status"] if prev else "pending",
                "task_file": prev.get("task_file") if prev else None,
                "session_id": prev.get("session_id") if prev else None,
                "kind": prev.get("kind") if prev else None,
                "review": prev.get("review", "pending") if prev else "pending",
            }
        )
    state.update(
        {
            "plan_id": state.get("plan_id") or plan.get("id") or "",
            "goal": plan.get("goal", ""),
            "steps": steps,
            "completed_at": state.get("completed_at"),
        }
    )
    return state


def list_plans() -> list[tuple[str, str, int, int]]:
    """列出全部计划：(id, goal, 完成环节数, 总环节数)。"""
    out = []
    if not PLANS_DIR.exists():
        return out
    for p in sorted(PLANS_DIR.glob("*.plan.yaml")):
        pid = p.name[: -len(".plan.yaml")]
        plan = _load_yaml(pid)
        state = _load_state(pid)
        steps = state.get("steps", [])
        done = len([s for s in steps if s["status"] in _DONE])
        out.append((pid, str(plan.get("goal", ""))[:50], done, len(steps)))
    return out


def cmd_plan_new(args) -> int:
    """登记新计划（复制 YAML 入 state/plans/，初始化状态）。"""
    src = Path(args.file)
    if not src.is_absolute():
        from pysci.paths import PROJECT_ROOT

        src = PROJECT_ROOT / src
    if not src.exists():
        print(f"[!] 计划文件不存在：{src}")
        return 2
    plan = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if not plan.get("steps"):
        print("[!] 计划缺少 steps（格式见 README §4.2）")
        return 2
    plan_id = args.id or src.stem
    if _plan_path(plan_id).exists() and not args.force:
        print(f"[!] 计划 {plan_id} 已存在（覆盖用 --force；续跑用 plan run {plan_id}）")
        return 2
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan["id"] = plan_id
    _plan_path(plan_id).write_text(
        yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    state = _reconcile(plan, {"plan_id": plan_id, "created": _now()})
    _save_state(plan_id, state)
    print(f"[√] 计划已登记：{plan_id}（{len(state['steps'])} 环节）")
    print(f"    目标：{state['goal']}")
    for s_yaml, s_state in zip(plan["steps"], state["steps"]):
        print(
            f"    - {s_state['id']}: {s_yaml.get('member', '?')} | "
            f"{str(s_yaml.get('brief', ''))[:60]}"
            + ("  [review]" if s_yaml.get("review") else "")
        )
    first = state["steps"][0]
    fy = plan["steps"][0]
    _print_run_guidance(plan_id, first, fy)
    return 0


def _print_run_guidance(plan_id: str, step_state: dict, step_yaml: dict) -> None:
    print()
    print("[NEXT]")
    print(f"  当前环节 {step_state['id']}（{step_yaml.get('member')}）粗描述原文：")
    print(f"    {step_yaml.get('brief', '（无）')}")
    print(
        f"  白名单建议：{step_yaml.get('dirs', [])}；档位：{step_yaml.get('model', '(成员默认)')}"
    )
    print("  → 书写该环节任务书后推进：")
    print(f'    uv run pysci-orch plan run {plan_id} --text "<任务书全文>"')
    print(f"    或 uv run pysci-orch plan run {plan_id} --task <任务书文件>")
    print("  （长任务建议以**后台 Bash**运行本命令——完成时自动通知，见 README §4.5）")


def cmd_plan_run(args) -> int:
    """推进计划：执行当前环节（任务书延迟书写），交接处暂停呈报决策。"""
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    steps_yaml = {
        str(sy.get("id") or f"s{i + 1}"): sy
        for i, sy in enumerate(plan.get("steps") or [])
    }

    cur = next((s for s in state["steps"] if s["status"] not in _DONE), None)
    if cur is None:
        _report_plan_complete(plan_id, state)
        return 0

    # blocked 环节续作：需要组长决策后带 --text/--task 重入
    if cur["status"] in ("blocked", "failed", "run_failed") and not (
        args.text or args.task
    ):
        print(f"[!] 环节 {cur['id']} 上次状态为 {cur['status']}，需要决策：")
        _print_blocked_options(plan_id, cur, steps_yaml.get(cur["id"], {}))
        return 2

    sy = steps_yaml.get(cur["id"], {})
    if not (args.text or args.task) and not cur.get("task_file"):
        _print_run_guidance(plan_id, cur, sy)
        print("\n[!] 尚未提供本环节任务书（延迟书写：只强制当前环节）。")
        return 2

    slug = args.name or f"{plan_id}-{cur['id']}"
    # 环节曾有会话（blocked/failed 续作）→ 复用之；否则按组长指定或默认 latest
    sel = cur["session_id"][:8] if cur.get("session_id") else (args.session or "latest")
    outcome: DispatchOutcome = do_dispatch(
        str(sy.get("member", cur.get("member", ""))),
        text=args.text,
        task_file=args.task,
        slug=slug,
        session=sel,
        model_tier=args.model_tier or (str(sy["model"]) if sy.get("model") else None),
        max_turns=args.max_turns,
        dirs=_split_dirs(args.dirs.split(","))
        if args.dirs
        else _split_dirs(sy.get("dirs")),
        plan=plan_id,
        step=cur["id"],
        quiet=True,
    )
    cur["task_file"] = outcome.task_file or cur.get("task_file")
    cur["session_id"] = outcome.sid or cur.get("session_id")
    cur["kind"] = outcome.kind

    if outcome.code == 0:
        cur["status"] = "done"
        if sy.get("review"):
            cur["review"] = "not_wired"  # reviewer Phase 3 接入
    elif outcome.kind == "blocked":
        cur["status"] = "blocked"
    else:
        cur["status"] = "failed"
    if all(s["status"] in _DONE for s in state["steps"]):
        state["completed_at"] = _now()
    _save_state(plan_id, state)

    _report_handoff(plan_id, plan, state, cur, outcome)
    return outcome.code


def _split_dirs(dirs) -> list[str]:
    if not dirs:
        return []
    if isinstance(dirs, str):
        return [d.strip() for d in dirs.split(",") if d.strip()]
    return [str(d) for d in dirs]


def _print_blocked_options(plan_id: str, step: dict, sy: dict) -> None:
    sid8 = (step.get("session_id") or "")[:8]
    print()
    print("[NEXT] 可选：")
    print(
        f'  1) 补充信息续作本环节：uv run pysci-orch plan run {plan_id} --text "<补充>"'
    )
    print('  2) 咨询副组长：uv run pysci-orch consult "<问题>"')
    print(
        f"  3) 修改计划（编辑 {_plan_path(plan_id).name} 后重新 plan run；或跳过本环节：--skip）"
    )
    print("  4) 中止计划并上报用户")
    if sid8:
        print(f"  （本环节会话 {sid8}，续作自动复用）")


def _report_handoff(
    plan_id: str, plan: dict, state: dict, cur: dict, outcome: DispatchOutcome
) -> None:
    """交接处暂停报告：本环节摘要 + 下一环节原文 recall + 决策选项。"""
    print(f"== 计划 {plan_id}：环节 {cur['id']} → {cur['status']} ==")
    icon = {"result": "√", "blocked": "✗", "parse_error": "?", "run_failed": "✗"}.get(
        outcome.kind, "?"
    )
    env = outcome.envelope
    dur = env.duration_ms / 1000 if env else 0
    print(f"[{icon}] {outcome.member} 交付={outcome.kind}（{dur:.0f}s）")
    print(
        f"---- 交付正文（前 600 字）----\n{outcome.body[:600]}\n----------------------------"
    )
    for c in outcome.checks:
        print(f"  验收[{c['verdict']}] {c['check']}: {Path(c['path']).name}")
    if outcome.suggestion_id:
        print(f"  改进建议已登记：{outcome.suggestion_id}（orch approve/reject 处理）")

    if cur["status"] in ("blocked", "failed"):
        sy = {
            str(s.get("id") or f"s{i + 1}"): s
            for i, s in enumerate(plan.get("steps") or [])
        }.get(cur["id"], {})
        _print_blocked_options(plan_id, cur, sy)
        return
    if cur.get("review") == "not_wired":
        print(
            "  [i] 本环节标记 review:true，但 reviewer 尚未接入（Phase 3）——已记 not_wired 跳过。"
        )

    nxt = next((s for s in state["steps"] if s["status"] not in _DONE), None)
    if nxt is None:
        _report_plan_complete(plan_id, state)
        return
    steps_yaml = {
        str(sy.get("id") or f"s{i + 1}"): sy
        for i, sy in enumerate(plan.get("steps") or [])
    }
    ny = steps_yaml.get(nxt["id"], {})
    print()
    print(f"== 交接决策点：下一环节 {nxt['id']}（{ny.get('member', '?')}）==")
    print(f"  粗描述原文：{ny.get('brief', '（无）')}")
    print(f"  白名单建议：{ny.get('dirs', [])}；档位：{ny.get('model', '(成员默认)')}")
    if outcome.kind == "result":
        print("  交接物：上一环节交付正文中的产物路径（书写任务书时引用）。")
    print()
    print("[NEXT] 可选：")
    print(
        f'  1) 一键继续（书写下一环节任务书）：uv run pysci-orch plan run {plan_id} --text "..."'
    )
    print(
        f"  2) 修改后续计划：编辑 {_plan_path(plan_id)} 后重新 plan run（按环节 id 对账）"
    )
    print(f"  3) 中止：uv run pysci-orch plan drop {plan_id}")


def _report_plan_complete(plan_id: str, state: dict) -> None:
    print(f"== 计划 {plan_id} 全部环节完成（{state.get('completed_at') or _now()}）==")
    entries = [e for e in read_all() if e.plan == plan_id]
    s = summarize(entries)
    print(
        f"计划统计：{s['hops']} 跳，总耗时 {s['duration_ms'] / 1000:.0f}s，credits {s['credits']}"
    )
    for mid, row in sorted(s["by_member"].items()):
        print(
            f"  {mid:<10} 跳数={row['hops']}  耗时占比={row['duration_pct']}%  credits占比={row['credits_pct']}%"
        )
    print()
    print("[NEXT]")
    print(
        "  → 向用户交付汇总：成果路径 + 上述统计 + 未决事项（含待审批建议）+ 你自己的改进建议（可选）"
    )


def cmd_plan_amend(args) -> int:
    """amend = 直接编辑计划 YAML（组长文件工具可写 plans/*.yaml）；本命令做校验与状态对账。"""
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    _save_state(plan_id, state)
    print(
        f"[√] 计划 {plan_id} 已对账（编辑 {_plan_path(plan_id).relative_to(ORCH_STATE_ROOT.parent)} 后运行本命令校验）："
    )
    for sy, ss in zip(plan.get("steps") or [], state["steps"]):
        print(
            f"  - {ss['id']}: {sy.get('member', '?')} [{ss['status']}] {str(sy.get('brief', ''))[:50]}"
        )
    return 0


def cmd_plan_drop(args) -> int:
    """中止计划（标记 abandoned，保留状态供审计）。"""
    plan_id = args.id
    state = _load_state(plan_id)
    if not state:
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    state["abandoned_at"] = _now()
    _save_state(plan_id, state)
    print(f"[√] 计划 {plan_id} 已标记中止（状态文件保留供审计）。")
    return 0


def cmd_plan_list(args) -> int:
    """列出全部计划与进度。"""
    del args
    plans = list_plans()
    if not plans:
        print("（暂无计划；orch plan new <file> 登记）")
        return 0
    for pid, goal, done, total in plans:
        state = _load_state(pid)
        mark = (
            "✔完成"
            if state.get("completed_at")
            else ("⊘中止" if state.get("abandoned_at") else "▶进行中")
        )
        print(f"  {pid:<28} {mark}  {done}/{total}  {goal}")
    return 0


def cmd_plan_adhoc(args) -> int:
    """单步快捷计划：登记并立即执行（计划外事项的正确姿势，README §4.1）。"""
    plan_id = f"adhoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    plan = {
        "id": plan_id,
        "goal": f"adhoc：{str(args.text or args.task)[:60]}",
        "steps": [
            {
                "id": "s1",
                "member": args.member,
                "brief": str(args.text or "见任务书")[:120],
                "dirs": _split_dirs(args.dirs),
                "review": False,
            }
        ],
    }
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    _plan_path(plan_id).write_text(
        yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _save_state(plan_id, _reconcile(plan, {"plan_id": plan_id, "created": _now()}))
    print(f"[√] adhoc 计划已登记：{plan_id}")

    class _Args:  # 复用 cmd_plan_run 的参数面
        pass

    a = _Args()
    a.id, a.text, a.task, a.name, a.session = (
        plan_id,
        args.text,
        args.task,
        None,
        "latest",
    )
    a.model_tier, a.max_turns, a.dirs = args.model_tier, args.max_turns, args.dirs
    return cmd_plan_run(a)
