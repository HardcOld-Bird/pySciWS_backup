"""plan-runner：计划驱动的任务流程状态机（README §4.2）。

计划 = ``state/plans/<id>.plan.yaml``（组长可直接编辑——amend 即编辑此文件，
run 时按环节 id 对账）；执行状态 = ``state/plans/<id>.state.json``（orch 专属）。
核心原则：**计划即状态机，聊天记录不是**——组长会话被压缩/重开后 plan run 仍从
断点继续；交接处暂停，精准 recall 下一环节粗描述原文。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pysci.paths import ORCH_STATE_ROOT

from .dispatch import DispatchOutcome
from .ledger import format_credits, format_est_tokens, read_all, summarize

PLANS_DIR = ORCH_STATE_ROOT / "plans"

_DONE = ("done", "skipped")

#: ``plan run`` 的同步等待上限（秒）——与成员 registry ``timeout_s`` 默认（3h）同量级：
#: 直派时代本命令就一路阻塞在 ``do_dispatch`` 里，改为「入队+等收尾」后阻塞面不变。
PLAN_RUN_TIMEOUT_S: int = 3 * 3600

#: 等收尾的轮询间隔（秒）。等的是**队列条目状态**（worker 落盘），不是进程——进程死了
#: 条目会由孤儿回收/reset_orphans 收敛，本函数到点即退，把环节留 ``queued`` 交 advance 结账。
PLAN_RUN_POLL_S: float = 2.0


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
        f"  白名单建议：{step_yaml.get('dirs', [])}；"
        f"档位：{step_yaml.get('model', '(成员默认)')}；"
        f"effort：{step_yaml.get('effort', '(默认=中)')}"
    )
    print("  → 书写该环节任务书后推进：")
    print(f'    uv run --no-sync pysci-orch plan run {plan_id} --text "<任务书全文>"')
    print(f"    或 uv run --no-sync pysci-orch plan run {plan_id} --task <任务书文件>")
    print("  （长任务建议以**后台 Bash**运行本命令——完成时自动通知，见 README §4.5）")


def _wait_job(
    member: str,
    job_id: str,
    *,
    timeout_s: int = PLAN_RUN_TIMEOUT_S,
    sleep: Any = None,
    now: Any = None,
) -> dict | None:
    """等成员队列里该 job 收尾（``plan run`` 的同步语义＝**入队 + 等完成事件**）。

    条目状态一离开 ``pending/in_progress`` 即返回它；条目消失（裁剪/孤儿回收）或到点
    仍未收尾 → ``None``（调用方不改环节状态，留 ``queued`` 交 ``plan advance`` 结账）。
    ``sleep``/``now`` 是**测试注入口**（真等 3h 没法进套件）。
    """
    from . import queues, workq

    do_sleep = time.sleep if sleep is None else sleep
    do_now = time.time if now is None else now
    deadline = do_now() + float(timeout_s)
    while True:
        hit = [i for i in queues.items(member) if str(i.get("id")) == job_id]
        item = hit[-1] if hit else None
        if item is None:
            return None
        if str(item.get("status")) not in (
            workq.STATUS_PENDING,
            workq.STATUS_IN_PROGRESS,
        ):
            return item
        if do_now() >= deadline:
            return None
        do_sleep(PLAN_RUN_POLL_S)


def _outcome_from_job(member: str, item: dict) -> DispatchOutcome:
    """从成员队列条目**还原**一次派发的结构化结果（交接报告的输入面）。

    直派退役后 ``plan run`` 不再经手 ``DispatchOutcome``——worker 把分类/正文/验收/
    建议/时长落在 ``item.result``（见 ``queueworker._execute``），这里还原成交接报告
    原本就认识的形状，避免两套交接渲染。
    """
    from . import workq
    from .runner import Envelope

    res = item.get("result") or {}
    status = str(item.get("status") or "")
    code = res.get("code")
    code = int(code) if code is not None else (0 if status == workq.STATUS_DONE else 2)
    kind = str(res.get("kind") or "")
    if not kind:
        kind = "result" if status == workq.STATUS_DONE else "run_failed"
    return DispatchOutcome(
        code=code,
        kind=kind,
        member=str(res.get("member") or member),
        sid=str(res.get("sid") or ""),
        session_name=str(res.get("session_name") or ""),
        task_file=str(res.get("task_file") or item.get("task_file") or ""),
        body=str(res.get("body") or ""),
        checks=[dict(c) for c in (res.get("checks") or [])],
        suggestion_id=str(res.get("suggestion_id") or ""),
        envelope=Envelope(duration_ms=int(res.get("duration_ms") or 0)),
    )


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
    member = str(sy.get("member") or cur.get("member") or "")

    # ── 直派退役（用户裁定 2026-10-14）：环节执行一律经成员队列 ────────────────
    # 对外语义**不变**（仍是「推进到交接暂停」的同步命令），内部改为：入队 → 唤醒 worker
    # → 等它收尾 → 从条目还原 outcome 再报交接。于是队列面板在途计数 / active-sessions
    # 注册 / STALL 判据对**一切** plan 环节生效（此前 state/queues/ 长期空转、面板全盲）。
    # ``runner.run_headless``/``do_dispatch`` 的直调只剩**队列 worker 一个入口**。
    from . import queues, queueworker

    # 成员闸（全链一致性，task-20261014-devops-gap-d1）：与 plan advance 同款——非队列
    # 成员（打错字）在入口就拒，别入队到无人消费的野生队列（devops 双通道后 QUEUE_MEMBERS 即真集）。
    if member not in queues.QUEUE_MEMBERS:
        print(
            f"[!] 成员 {member} 非队列成员——plan 只覆盖 "
            f"{'、'.join(queues.QUEUE_MEMBERS)}（devops 双通道：plan 环节直派 + backlog 组包）"
        )
        return 2

    item = queues.enqueue(
        member,
        text=str(args.text or ""),
        task_file=str(args.task or ""),
        dirs=_split_dirs(args.dirs.split(","))
        if args.dirs
        else _split_dirs(sy.get("dirs")),
        model_tier=args.model_tier or (str(sy["model"]) if sy.get("model") else ""),
        effort=args.effort or (str(sy["effort"]) if sy.get("effort") else ""),
        plan=plan_id,
        step=str(cur["id"]),
        session=sel,
        review=bool(sy.get("review")),
        slug=slug,
        max_turns=args.max_turns,
    )
    job_id = str(item["id"])
    cur["status"] = "queued"
    cur["job_id"] = job_id
    if args.task:
        cur["task_file"] = args.task
    _save_state(plan_id, state)

    _, wake_msg = queueworker.wake_member(member)
    got = _wait_job(member, job_id)
    if got is None:
        # 不猜结果：留 queued 交 plan advance 结账（事件驱动语义优先于「同步要有个结论」）。
        print(
            f"[!] job {job_id} 未在 {member} 队列收尾（超时/条目失踪）——"
            f"环节 {cur['id']} 保持 queued，交 `orch plan advance {plan_id}` 结账（不重跑）。"
        )
        print(f"    {wake_msg}")
        return 2

    outcome = _outcome_from_job(member, got)
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


def _queue_item_for_step(plan_id: str, step_id: str, member: str) -> dict | None:
    """环节在成员队列里的**最新** job 条目（按其 ``plan``/``step`` 反查；无则 ``None``）。

    这是 advance 事件驱动闭环的接线点：``queueworker`` 消化时把结果写回条目，advance 每次
    被唤醒后回读该条目判「在途环节收尾没有」——不必让 worker 直接改 plan 状态（解耦）。
    """
    from . import queues

    if not member:
        return None
    hit = [
        i
        for i in queues.items(member)
        if str(i.get("plan")) == plan_id and str(i.get("step")) == step_id
    ]
    return hit[-1] if hit else None


def cmd_plan_advance(args) -> int:
    """**事件驱动推进**（§3.5，M5）：把下一环节入队 m4 成员队列**即返回**。

    与 ``plan run``（阻塞式，兼容形态）的区别：本命令不等待组员，入队后由 job 完成
    **事件**触发组长再敲一次 advance。任务书延迟书写语义保留——advance 时给**当前环节**
    任务书。多 plan 并行：各 active plan 各自 advance 互不阻塞；同一成员队列 FIFO 串行、
    不同成员即并行（用户模型）。既有语义保留：交接暂停 / 机械验收 FAIL 回派 / blocked
    由组长在 drain 后决策。

    闭环：advance → 入队（环节置 ``queued``）→ worker 消化（发 JOB-DONE/STEP-REVIEW-DONE）
    → 组长 ``orch events drain`` 醒来 → 再 advance：本命令先据队列条目结果把在途环节落终态，
    再入队下一环节。
    """
    from . import queues, queueworker, workq

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

    # ① 收尾在途环节：status=="queued" 的环节，回读它在成员队列里的 job 结果。
    queued = next((s for s in state["steps"] if s.get("status") == "queued"), None)
    if queued is not None:
        member = str(steps_yaml.get(queued["id"], {}).get("member") or "")
        item = _queue_item_for_step(plan_id, str(queued["id"]), member)
        if item is None:
            print(
                f"[i] 环节 {queued['id']} 在 {member or '?'} 队列中无对应条目"
                "（或被裁剪）——保守不推进；orch queues list 核实后重来。"
            )
            return 0
        st = str(item.get("status", ""))
        if st in (workq.STATUS_PENDING, workq.STATUS_IN_PROGRESS):
            print(
                f"[i] 环节 {queued['id']} 仍在 {member} 队列（{st}）——"
                "等它完成的 JOB-DONE 事件驱动下一次 advance。"
            )
            return 0
        kind = str((item.get("result") or {}).get("kind", ""))
        if st == workq.STATUS_DONE:
            queued["status"] = "done"
            if steps_yaml.get(queued["id"], {}).get("review"):
                queued["review"] = (
                    "review_done"  # 合并单元：宿主已交付（review 走附属）
                )
        elif kind == "blocked":
            queued["status"] = "blocked"
        else:
            queued["status"] = "failed"
        queued["job_id"] = str(item.get("id"))
        _save_state(plan_id, state)
        if queued["status"] in ("blocked", "failed"):
            print(
                f"[!] 环节 {queued['id']} 的 job 以 {queued['status']} 收尾，需要决策："
            )
            _print_blocked_options(plan_id, queued, steps_yaml.get(queued["id"], {}))
            return 2

    # ② 下一未完成环节
    cur = next((s for s in state["steps"] if s["status"] not in _DONE), None)
    if cur is None:
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
        return 0
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

    member = str(sy.get("member") or "")
    if member not in queues.QUEUE_MEMBERS:
        print(
            f"[!] 成员 {member} 非队列成员——plan advance 只覆盖 "
            f"{'、'.join(queues.QUEUE_MEMBERS)}（devops 双通道：plan 环节直派 + backlog 组包）"
        )
        return 2
    item = queues.enqueue(
        member,
        text=str(args.text or ""),
        task_file=str(args.task or ""),
        dirs=_split_dirs(args.dirs.split(","))
        if args.dirs
        else _split_dirs(sy.get("dirs")),
        model_tier=args.model_tier or (str(sy["model"]) if sy.get("model") else ""),
        effort=args.effort or (str(sy["effort"]) if sy.get("effort") else ""),
        plan=plan_id,
        step=str(cur["id"]),
        session=str(args.session or "latest"),
        review=bool(sy.get("review")),
        slug=f"{plan_id}-{cur['id']}",
    )
    cur["status"] = "queued"
    cur["job_id"] = str(item["id"])
    if args.task:
        cur["task_file"] = args.task
    _save_state(plan_id, state)

    _, msg = queueworker.wake_member(member, dry=bool(getattr(args, "dry", False)))
    print(
        f"[√] 计划 {plan_id} 环节 {cur['id']}（{member}）已入队 job {item['id']}"
        "——入队即返回，完成事件驱动下一次 advance"
    )
    print(f"    {msg}")
    print(
        f"[NEXT] 等 JOB-DONE/STEP-REVIEW-DONE 事件：orch events drain → "
        f'orch plan advance {plan_id} --text "<下一环节任务书>"'
    )
    return 0


def _print_blocked_options(plan_id: str, step: dict, sy: dict) -> None:
    sid8 = (step.get("session_id") or "")[:8]
    print()
    print("[NEXT] 可选：")
    print(
        f'  1) 补充信息续作本环节：uv run --no-sync pysci-orch plan run {plan_id} --text "<补充>"'
    )
    print('  2) 咨询副组长：uv run --no-sync pysci-orch consult "<问题>"')
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
    print(
        f"  白名单建议：{ny.get('dirs', [])}；档位：{ny.get('model', '(成员默认)')}；"
        f"effort：{ny.get('effort', '(默认=中)')}"
    )
    if outcome.kind == "result":
        print("  交接物：上一环节交付正文中的产物路径（书写任务书时引用）。")
    print()
    print("[NEXT] 可选：")
    print(
        f'  1) 一键继续（书写下一环节任务书）：uv run --no-sync pysci-orch plan run {plan_id} --text "..."'
    )
    print(
        f"  2) 修改后续计划：编辑 {_plan_path(plan_id)} 后重新 plan run（按环节 id 对账）"
    )
    print(f"  3) 中止：uv run --no-sync pysci-orch plan drop {plan_id}")


def _report_plan_complete(plan_id: str, state: dict) -> None:
    print(f"== 计划 {plan_id} 全部环节完成（{state.get('completed_at') or _now()}）==")
    entries = [e for e in read_all() if e.plan == plan_id]
    s = summarize(entries)
    print(
        f"计划统计：{s['hops']} 跳，总耗时 {s['duration_ms'] / 1000:.0f}s，"
        f"{format_est_tokens(s)}，{format_credits(s)}"
    )
    for mid, row in sorted(s["by_member"].items()):
        print(
            f"  {mid:<10} 跳数={row['hops']}  耗时占比={row['duration_pct']}%  "
            f"est占比={row['est_tokens_pct']}%  credits占比={row['credits_pct']}%"
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
    a.effort = args.effort
    return cmd_plan_run(a)
