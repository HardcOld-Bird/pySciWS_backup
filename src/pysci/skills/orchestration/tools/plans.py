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
from .planflow import (
    DONE_STATES,
    KIND_SELF,
    SELF_AWAITING,
    PlanTree,
    freeze_violations,
)

PLANS_DIR = ORCH_STATE_ROOT / "plans"

_DONE = DONE_STATES

#: 拆解指导真本（PSEP_3 §2.5 三级方案：必读文档 + plan new 打印 + read: 标记 lint）。
DECOMP_REF = "plan-decomposition.md"

#: ``plan new`` 输出**头部**固定打印的必读指引。
DECOMP_READ_HINT = f"※ 多环节大任务拆解前必读：orchestration/skills/orchestration/references/{DECOMP_REF}"


def _tree(plan: dict) -> PlanTree:
    """plan YAML → 编排树（PSEP_3 §2.1；纯解析无 I/O，每次现建——plan 小）。"""
    return PlanTree(plan.get("steps") or [])


def _leaf_map(plan: dict) -> dict[str, dict]:
    """叶子**点路径** → 叶子 YAML（superstep 嵌套下为全部叶子；扁平 plan 恒等原 id）。"""
    return _tree(plan).leaf_map()


def _status_of(state: dict):
    """状态读数闭包：叶子点路径 → status（缺省 pending）。"""
    st = {
        str(s.get("id")): str(s.get("status") or "pending")
        for s in state.get("steps", [])
    }
    return lambda path: st.get(str(path), "pending")


def _has_read_marker(plan: dict) -> bool:
    """plan YAML 是否含 ``read: plan-decomposition.md`` 标记行（PSEP_3 §2.5）。"""
    v = plan.get("read")
    if isinstance(v, (list, tuple)):
        return any(DECOMP_REF in str(x) for x in v)
    return DECOMP_REF in str(v or "")


def _print_plan_lints(plan: dict) -> None:
    """plan 结构 lint（**不阻断**，PSEP_3 §2.4/§2.5）：软提示 + 拆解必读标记行。"""
    tree = _tree(plan)
    lines = tree.soft_hints()
    n = len(tree.leaves)
    if (tree.has_superstep() or n >= 5) and not _has_read_marker(plan):
        lines.append(
            f"plan 含 superstep 或环节数≥{n}——建议在 YAML 加 `read: {DECOMP_REF}` 标记行"
            f"（拆解前必读 references/{DECOMP_REF}）"
        )
    for ln in lines:
        print(f"[lint] {ln}（软提示，不阻断）")


def _check_evidence(path: str, sy: dict) -> tuple[bool, str]:
    """校验 ``kind: self`` 环节的 evidence（PSEP_3 §2.1）：文件存在 + 模板要求的标题段。

    标题段判据：YAML ``evidence_sections: [...]`` 声明则逐个匹配 Markdown 标题行（``# …``
    含该标题）；未声明则要求至少一个 ATX 标题（文件是有结构的报告，非空）。
    """
    from pysci.paths import PROJECT_ROOT

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return False, f"evidence 文件不存在：{p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return False, f"evidence 文件为空：{p}"
    headings = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    sections = sy.get("evidence_sections") or []
    if sections:
        missing = [t for t in sections if not any(str(t) in h for h in headings)]
        if missing:
            return (
                False,
                f"evidence 缺标题段：{'、'.join(str(m) for m in missing)}（{p}）",
            )
        return True, f"evidence 通过（{len(sections)} 个标题段齐全）：{p}"
    if not headings:
        return False, f"evidence 无标题段（需至少一个 Markdown 标题行 `# …`）：{p}"
    return True, f"evidence 通过（含标题段）：{p}"


def _self_target(plan: dict, state: dict) -> dict | None:
    """待收尾的 ``kind:self`` 叶子：优先 ``self_awaiting``（已引导待证据），否则首个可派发 self。"""
    for s in state["steps"]:
        if str(s.get("status")) == SELF_AWAITING:
            return s
    lm = _leaf_map(plan)
    disp = _tree(plan).dispatchable(_status_of(state))
    for p in disp:
        if str((lm.get(p) or {}).get("kind") or "dispatch") == KIND_SELF:
            return next((s for s in state["steps"] if str(s["id"]) == p), None)
    return None


def _apply_evidence(plan_id: str, plan: dict, state: dict, evidence: str) -> int | None:
    """把 ``--evidence`` 落到当前 self 环节。返回 ``None``=已收尾（调用方继续推进）；否则错误码。"""
    tgt = _self_target(plan, state)
    if tgt is None:
        print(f"[!] 计划 {plan_id} 无可收尾的 kind:self 环节——--evidence 无处可施。")
        return 2
    sy = _leaf_map(plan).get(str(tgt["id"]), {})
    ok, msg = _check_evidence(evidence, sy)
    if not ok:
        tgt["status"] = SELF_AWAITING
        _save_state(plan_id, state)
        print(f"[!] 组长环节 {tgt['id']} evidence 校验未过：{msg}")
        return 2
    tgt["status"] = "done"
    tgt["evidence"] = str(evidence)
    _save_state(plan_id, state)
    print(f"[√] 组长环节 {tgt['id']} 收尾：{msg}")
    return None


def _report_after_evidence(plan_id: str, plan: dict, state: dict) -> None:
    """``--evidence`` 收尾 self 环节后报告：全完成则报完成，否则刷工作清单。"""
    if _plan_settled(plan, state):
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
    else:
        _print_worklist(plan_id, plan, state)


def _print_self_guidance(
    plan_id: str, step_state: dict, sy: dict, *, plan: dict | None = None
) -> None:
    """``kind: self`` 环节的引导块（evidence 规格 + 收尾命令）——plan new / advance 共用。

    ``expand_history`` 叶子（返工 a1，round≥2）在 [NEXT] 后**全文展开**历次改进报告（§4.3 #4）。
    """
    print()
    print("[NEXT]")
    print(f"  环节 {step_state['id']} 为组长自办环节（kind: self）——不入成员队列。")
    print(f"    粗描述原文：{sy.get('brief', '（无）')}")
    secs = sy.get("evidence_sections") or []
    if secs:
        print(f"  交付 evidence 须含标题段：{'、'.join(str(s) for s in secs)}")
    else:
        print("  交付 evidence 须为 Markdown 报告（含至少一个 `# …` 标题段）")
    print(
        f"  完成后提交证据收尾：uv run --no-sync pysci-orch plan advance "
        f"{plan_id} --evidence <路径>"
    )
    if sy.get("expand_history") and plan is not None:
        for ln in _rework_history_lines(plan):
            print(ln)


def _rework_history_lines(plan: dict) -> list[str]:
    """rework_round≥2 的**历次改进强制展示**（§4.3 #4）：前轮报告**全文**（不只给路径）。

    数据面：任务书 ``rework_of`` 链回溯；每轮取 ``data/taskbooks/<id>.report.md``（组装器产物）
    全文，缺则回落该轮任务书正文（简介）。
    """
    from . import taskbooks as _tb

    tb_id = str(plan.get("taskbook") or "")
    st = (_tb.load_state(tb_id) or {}) if tb_id else {}
    rnd = int(st.get("rework_round") or plan.get("rework_round") or 1)
    if rnd < 2:
        return []
    out = [f"  == 历次改进强制展示（第 {rnd} 轮；前轮报告全文，PSEP_3 §4.3 #4）=="]
    prev = str(st.get("rework_of") or "")
    n = rnd - 1
    while prev:
        pst = _tb.load_state(prev) or {}
        out.append(f"  —— 第 {n} 轮「{pst.get('title') or prev}」（{prev}）——")
        rp = _tb.report_path(prev)
        try:
            text = rp.read_text(encoding="utf-8").strip() if rp.exists() else ""
        except OSError:
            text = ""
        out.append(text or _tb.read_body(prev).strip() or "（无报告留痕）")
        out.append("")
        prev = str(pst.get("rework_of") or "")
        n -= 1
    return out


def _guide_self_step(plan_id: str, plan: dict, state: dict, cur: dict, sy: dict) -> int:
    """引导一个 ``kind:self`` 组长环节（不入成员队列）：打印 evidence 规格与收尾命令。"""
    if str(cur.get("status")) != SELF_AWAITING:
        cur["status"] = SELF_AWAITING
        _save_state(plan_id, state)
    _print_self_guidance(plan_id, cur, sy, plan=plan)
    _print_worklist(plan_id, plan, state)
    return 0


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
    """按**叶子点路径 id** 对账 YAML（可能被组长 amend 过）与既有状态。

    superstep 只在 YAML 里存在（递归结构），**state.json 只落叶子行**（PSEP_3 §2.1：
    聚合状态由叶子推导不落盘）；存量扁平 plan 的顶层叶子路径 == 原环节 id，行为完全不变
    （无 superstep 等价单元素串行）。保留已有叶子状态；新叶子置 pending；YAML 中已删除的
    叶子从状态中移除（增删节点天然兼容）。
    """
    tree = _tree(plan)
    old = {s["id"]: s for s in state.get("steps", [])}
    steps: list[dict] = []
    for path in tree.leaf_paths():
        prev = old.get(path)
        steps.append(
            {
                "id": path,
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
    print(DECOMP_READ_HINT)  # 头部固定打印必读指引（PSEP_3 §2.5 三级方案之二）
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
    tree = _tree(plan)
    print(f"[√] 计划已登记：{plan_id}（{len(state['steps'])} 环节）")
    print(f"    目标：{state['goal']}")
    for leaf in tree.leaves:
        sy = leaf.yaml
        indent = "  " * leaf.path.count(".")
        who = sy.get("member") or ("self（组长）" if leaf.kind == KIND_SELF else "?")
        print(
            f"    - {indent}{leaf.path}: {who} | "
            f"{str(sy.get('brief', ''))[:60]}"
            + ("  [review]" if sy.get("review") else "")
        )
    _print_plan_lints(plan)
    first = state["steps"][0]
    fy = next((n.yaml for n in tree.leaves if n.path == first["id"]), {})
    if str(fy.get("kind") or "dispatch") == KIND_SELF:
        _print_self_guidance(plan_id, first, fy, plan=plan)
    else:
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
        body_file=str(res.get("body_file") or ""),
        checks=[dict(c) for c in (res.get("checks") or [])],
        suggestion_id=str(res.get("suggestion_id") or ""),
        envelope=Envelope(duration_ms=int(res.get("duration_ms") or 0)),
    )


def _wait_all_inflight(
    plan_id: str,
    plan: dict,
    state: dict,
    *,
    timeout_s: int = PLAN_RUN_TIMEOUT_S,
    sleep: Any = None,
    now: Any = None,
) -> list[tuple[dict, DispatchOutcome]]:
    """等本计划**全部在途 job** 收尾（``plan run`` 的阻塞语义，PSEP_3 §2.3）。

    逐轮把已到齐的结果落终态（**乱序即到达序**）；到点仍未收尾 → 留 ``queued`` 交
    advance（与「超时不猜结果」旧语义一致）。无在途时也先收尾一轮（把已终态条目结掉）。
    ``sleep``/``now`` 是**测试注入口**。
    """
    do_sleep = time.sleep if sleep is None else sleep
    do_now = time.time if now is None else now
    deadline = do_now() + float(timeout_s)
    settled: list[tuple[dict, DispatchOutcome]] = []
    while True:
        settled.extend(_settle_terminal_jobs(plan_id, plan, state))
        pend = _inflight(state)
        if not pend:
            break
        # 条目失踪（被裁剪 / 入队未落盘 / 测试桩）：无法观察 → 不猜结果，停止等待留 queued
        # （与旧 `_wait_job`「条目消失 → 不改状态」语义一致；防下游按 3h 上限空转）。
        lm = _leaf_map(plan)
        if any(
            _queue_item_for_step(
                plan_id,
                str(s["id"]),
                str((lm.get(str(s["id"])) or {}).get("member") or ""),
            )
            is None
            for s in pend
        ):
            break
        if do_now() >= deadline:
            break
        do_sleep(PLAN_RUN_POLL_S)
    _save_state(plan_id, state)
    return settled


def cmd_plan_run(args) -> int:
    """推进计划：执行**就绪环节**（任务书延迟书写），**等全部在途 job 收尾**后交接。

    语义（PSEP_3 §2.3）：① 先把已终态队列条目全部结掉（多 job 并行时乱序到达，逐项交接）；
    ② 若仍有在途 job → 阻塞等它们全部收尾；③ 目标＝blocked/失败叶子（续作优先，--text
    即补充信息）或**首个可派发叶子**（递归就绪谓词）；④ 交接处打印三栏工作清单后暂停。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    tree = _tree(plan)
    steps_yaml = _leaf_map(plan)

    if tpl := template_violations(plan):
        print("[!] plan run 被拒：rework 模板必留节点缺失/乱序（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2

    # ①② 收尾全部已终态条目 + 等在途全部收尾（run 的阻塞语义）
    settled = _wait_all_inflight(plan_id, plan, state)
    for s_done, o_done in settled:
        _report_handoff(plan_id, plan, state, s_done, o_done)

    # --evidence：先把一个 kind:self 组长环节收尾（PSEP_3 §2.1）
    if getattr(args, "evidence", None):
        rc = _apply_evidence(plan_id, plan, state, args.evidence)
        if rc is not None:
            return rc
        _report_after_evidence(plan_id, plan, state)
        return 0

    status_of = _status_of(state)
    # blocked/失败叶子优先（续作语义：--text 即补充信息）；否则**首可派发**（就绪谓词）
    cur = next(
        (
            s
            for s in state["steps"]
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ),
        None,
    )
    if cur is None:
        disp = tree.dispatchable(status_of)
        cur = (
            next((s for s in state["steps"] if str(s["id"]) == disp[0]), None)
            if disp
            else None
        )
    if cur is None:
        if _plan_settled(plan, state):
            if not state.get("completed_at"):
                state["completed_at"] = _now()
                _save_state(plan_id, state)
            _report_plan_complete(plan_id, state)
            return 0
        if _inflight(state):
            print(
                f"[i] 计划 {plan_id} 无就绪环节；在途 {len(_inflight(state))} 项"
                "——交 `orch plan advance` 结账。"
            )
        else:
            _print_worklist(plan_id, plan, state)
        return 0

    # blocked 环节续作：需要组长决策后带 --text/--task 重入
    if cur["status"] in ("blocked", "failed", "run_failed") and not (
        args.text or args.task
    ):
        print(f"[!] 环节 {cur['id']} 上次状态为 {cur['status']}，需要决策：")
        _print_blocked_options(plan_id, cur, steps_yaml.get(cur["id"], {}))
        _print_worklist(plan_id, plan, state)
        return 2

    sy = steps_yaml.get(cur["id"], {})
    # 送审 gate（§4.3 #2/#3）：gate_jury 叶子须案卷 verdict=A 才派单
    if (gate := _gate_blocks(plan_id, plan, state, cur, sy)) is not None:
        return gate
    # kind:self 组长环节：不入成员队列，引导书写 + evidence 收尾
    if str(sy.get("kind") or "dispatch") == KIND_SELF:
        return _guide_self_step(plan_id, plan, state, cur, sy)
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
    outcomes = _wait_all_inflight(plan_id, plan, state)
    if not outcomes:
        # 不猜结果：留 queued 交 plan advance 结账（事件驱动语义优先于「同步要有个结论」）。
        print(
            f"[!] job {job_id} 未在 {member} 队列收尾（超时/条目失踪）——"
            f"环节 {cur['id']} 保持 queued，交 `orch plan advance {plan_id}` 结账（不重跑）。"
        )
        print(f"    {wake_msg}")
        return 2

    for s_done, o_done in outcomes:
        _report_handoff(plan_id, plan, state, s_done, o_done)
    if _plan_settled(plan, state):
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
    for line in _routing_card():
        print(line)
    return 0 if all(o.code == 0 for _, o in outcomes) else 2


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


def _settle_terminal_jobs(
    plan_id: str, plan: dict, state: dict
) -> list[tuple[dict, DispatchOutcome]]:
    """把**全部**已在成员队列收尾的 ``queued`` 叶子落终态（state 序＝叶子点路径序）。

    多 job 并行时结果**乱序到达**：一次把已到齐的都结掉（仍在途 / 条目失踪的保持
    ``queued`` 不动）；返回 ``[(叶子状态行, outcome)]`` 供调用方**逐项** ``_report_handoff``。
    本函数只改内存，调用方负责 ``_save_state``。
    """
    from . import workq

    lm = _leaf_map(plan)
    out: list[tuple[dict, DispatchOutcome]] = []
    for s in state["steps"]:
        if s.get("status") != "queued":
            continue
        sid = str(s["id"])
        member = str((lm.get(sid) or {}).get("member") or "")
        item = _queue_item_for_step(plan_id, sid, member)
        if item is None:
            continue  # 条目失踪：保守保持 queued（不猜结果）
        st = str(item.get("status") or "")
        if st in (workq.STATUS_PENDING, workq.STATUS_IN_PROGRESS):
            continue
        kind = str((item.get("result") or {}).get("kind") or "")
        if st == workq.STATUS_DONE:
            s["status"] = "done"
            if (lm.get(sid) or {}).get("review"):
                s["review"] = "review_done"  # 合并单元：宿主已交付（review 走附属）
        elif kind == "blocked":
            s["status"] = "blocked"
        else:
            s["status"] = "failed"
        s["job_id"] = str(item.get("id"))
        outcome = _outcome_from_job(member, item)
        # 会话/任务书/交付类别回填（续作复用与台账读数；run/advance 同口径）
        s["task_file"] = outcome.task_file or s.get("task_file")
        s["session_id"] = outcome.sid or s.get("session_id")
        s["kind"] = outcome.kind
        out.append((s, outcome))
    if state.get("steps") and all(x["status"] in _DONE for x in state["steps"]):
        state["completed_at"] = state.get("completed_at") or _now()
    return out


def _inflight(state: dict) -> list[dict]:
    """在途叶子（status=queued）。"""
    return [s for s in state["steps"] if s.get("status") == "queued"]


def _worklist_panes(plan: dict, state: dict) -> dict[str, list[str]]:
    """三栏工作清单的数据面（可派发/在途/待验收）——[NEXT] 等处共用同一判据。

    「在途」含成员队列 job（``queued``）与组长自办环节（``self_awaiting``）——两者都是
    「已在办、未收尾」，只是驱动者不同（worker vs 组长）。
    """
    tree = _tree(plan)
    status_of = _status_of(state)
    return {
        "dispatchable": tree.dispatchable(status_of),
        "inflight": [
            str(s["id"])
            for s in state["steps"]
            if str(s.get("status")) in ("queued", SELF_AWAITING)
        ],
        "need_leader": [
            str(s["id"])
            for s in state["steps"]
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ],
    }


def _plan_settled(plan: dict, state: dict) -> bool:
    """三栏全空（无可派发/无在途/无待验收）→ 计划可判完成。"""
    panes = _worklist_panes(plan, state)
    return not (panes["dispatchable"] or panes["inflight"] or panes["need_leader"])


def _print_worklist(plan_id: str, plan: dict, state: dict) -> None:
    """[NEXT] **三栏工作清单**（PSEP_3 §2.3）：可派发（按 id 序，一次引导一个）/在途/待验收。

    待验收栏含 blocked/失败决策点（可能多个）；可派发栏给出首个的成员与粗描述原文，
    其余逐一列名（本环节入队后可连续引导下一个）。
    """
    del plan_id
    panes = _worklist_panes(plan, state)
    lm = _leaf_map(plan)
    disp = panes["dispatchable"]
    print("[NEXT] 工作清单（可派发 / 在途 / 待验收）：")
    if disp:
        head = disp[0]
        sy = lm.get(head, {})
        print(
            f"  可派发（按 id 序，一次引导一个）：{head}（{sy.get('member', '?')}）"
            f" | {str(sy.get('brief', '（无）'))[:50]}"
        )
        if len(disp) > 1:
            print(f"    其余可派发：{', '.join(disp[1:])}（入队后可继续引导下一个）")
    else:
        print("  可派发：（无）")
    inflight_disp = [
        f"{sid}(self·待证据)"
        if str((lm.get(sid) or {}).get("kind") or "dispatch") == KIND_SELF
        else sid
        for sid in panes["inflight"]
    ]
    print(f"  在途：{', '.join(inflight_disp) if inflight_disp else '（无）'}")
    need = panes["need_leader"]
    print(
        f"  待验收：{', '.join(need) if need else '（无）'}"
        + ("（blocked/失败决策点见上方交接段；可能多个）" if need else "")
    )


def _routing_card() -> list[str]:
    """机械推送点用的**全局**路由卡片（PSEP_3 §3.3）：fail-soft（推送面绝不阻塞推进）。"""
    try:
        from . import nextroute

        return nextroute.card_lines(None)
    except Exception:  # noqa: BLE001 - 路由读数坏一项不塌推进
        return []


def cmd_plan_advance(args) -> int:
    """**事件驱动推进**（§3.5，M5）：把下一环节入队 m4 成员队列**即返回**。

    与 ``plan run``（阻塞式，兼容形态）的区别：本命令不等待组员，入队后由 job 完成
    **事件**触发组长再敲一次 advance。任务书延迟书写语义保留——advance 时给**当前环节**
    任务书。多 plan 并行：各 active plan 各自 advance 互不阻塞；同一成员队列 FIFO 串行、
    不同成员即并行（用户模型）。既有语义保留：交接暂停 / 机械验收 FAIL 回派 / blocked
    由组长在 drain 后决策。

    闭环：advance → 入队（环节置 ``queued``）→ worker 消化（发 JOB-DONE/STEP-REVIEW-DONE）
    → 组长 ``orch events drain`` 醒来 → 再 advance：本命令先收尾**全部**已终态条目（多 job
    并行时乱序到达，逐项交接），再按**递归就绪谓词**（PSEP_3 §2.2）入队**首个可派发**叶子，
    并以**三栏工作清单**（可派发/在途/待验收）收尾。
    """
    from . import queues, queueworker

    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    tree = _tree(plan)
    steps_yaml = _leaf_map(plan)

    if tpl := template_violations(plan):
        print("[!] advance 被拒：rework 模板必留节点缺失/乱序（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2

    # ① 收尾**全部**已终态队列条目（多 job 乱序到达：一次结清，逐项交接决策点）
    settled = _settle_terminal_jobs(plan_id, plan, state)
    if settled:
        _save_state(plan_id, state)
    for s_done, o_done in settled:
        _report_handoff(plan_id, plan, state, s_done, o_done)
    inflight = _inflight(state)
    if inflight:
        print(
            f"[i] 在途 {len(inflight)} 项仍在跑：{', '.join(str(s['id']) for s in inflight)}"
            "——等 JOB-DONE 事件驱动下一次 advance。"
        )

    # ② --evidence：先把一个 kind:self 组长环节收尾（PSEP_3 §2.1）
    if getattr(args, "evidence", None):
        rc = _apply_evidence(plan_id, plan, state, args.evidence)
        if rc is not None:
            return rc
        _report_after_evidence(plan_id, plan, state)
        return 0

    # ③ 目标环节：blocked/失败叶子（续作优先）；否则**首可派发**（递归就绪谓词）
    cur = next(
        (
            s
            for s in state["steps"]
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ),
        None,
    )
    if cur is None:
        disp = tree.dispatchable(_status_of(state))
        cur = (
            next((s for s in state["steps"] if str(s["id"]) == disp[0]), None)
            if disp
            else None
        )
    if cur is None:
        if _plan_settled(plan, state):
            if not state.get("completed_at"):
                state["completed_at"] = _now()
                _save_state(plan_id, state)
            _report_plan_complete(plan_id, state)
            return 0
        _print_worklist(plan_id, plan, state)
        return 0
    if cur["status"] in ("blocked", "failed", "run_failed") and not (
        args.text or args.task
    ):
        print(f"[!] 环节 {cur['id']} 上次状态为 {cur['status']}，需要决策：")
        _print_blocked_options(plan_id, cur, steps_yaml.get(cur["id"], {}))
        _print_worklist(plan_id, plan, state)
        return 2
    sy = steps_yaml.get(cur["id"], {})
    # 送审 gate（§4.3 #2/#3）：gate_jury 叶子须案卷 verdict=A 才派单
    if (gate := _gate_blocks(plan_id, plan, state, cur, sy)) is not None:
        return gate
    # kind:self 组长环节：不入成员队列，引导书写 + evidence 收尾
    if str(sy.get("kind") or "dispatch") == KIND_SELF:
        return _guide_self_step(plan_id, plan, state, cur, sy)
    if not (args.text or args.task) and not cur.get("task_file"):
        _print_run_guidance(plan_id, cur, sy)
        print("\n[!] 尚未提供本环节任务书（延迟书写：只强制当前环节）。")
        _print_worklist(plan_id, plan, state)
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
    _print_worklist(plan_id, plan, state)
    print(
        f"  事件驱动：orch events drain → "
        f'orch plan advance {plan_id} --text "<下一环节任务书>"'
    )
    for line in _routing_card():
        print(line)
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
    if getattr(outcome, "body_file", ""):
        print(f"  交付全文留档：{outcome.body_file}")
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

    panes = _worklist_panes(plan, state)
    nxt_id = next(iter(panes["dispatchable"]), "")
    if not nxt_id:
        # 完成判定交给调用方**收尾一次**（多 job 乱序同批到达时逐项交接，避免重复播报）。
        _print_worklist(plan_id, plan, state)
        return
    nxt = next((s for s in state["steps"] if str(s["id"]) == nxt_id), None)
    if nxt is None:  # 防御：理论上不可达（可派发项必有状态行）
        _print_worklist(plan_id, plan, state)
        return
    steps_yaml = _leaf_map(plan)
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
    _print_worklist(plan_id, plan, state)
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


# ---------------------------------------------------------------------------
# 返工流骨架（PSEP_3 §4.1/§4.2/§4.3）：模板三档 / 必留节点 lint / 送审 gate
# ---------------------------------------------------------------------------
MODE_FULL = "full"
MODE_NO_INSPECTOR = "no_inspector"
MODE_REWORK_ONLY = "rework_only"
REWORK_MODES: tuple[str, ...] = (MODE_FULL, MODE_NO_INSPECTOR, MODE_REWORK_ONLY)

#: 各档**必留叶子**（DFS 序＝骨架顺序；缺失/乱序 → advance 拒绝，§4.3 #1）。
REWORK_REQUIRED: dict[str, tuple[str, ...]] = {
    MODE_FULL: (
        "A.a1",
        "A.a2",
        "A.a3",
        "A.a4",
        "A.a5",
        "BC.B.b1",
        "BC.B.b2",
        "BC.B.b_last",
        "BC.C.c1",
        "BC.C.c2",
        "BC.C.c3",
        "BC.C.c4",
    ),
    MODE_NO_INSPECTOR: (
        "A.a1",
        "A.a2",
        "A.a3",
        "A.a4",
        "A.a5",
        "B.b1",
        "B.b2",
        "B.b_last",
    ),
    MODE_REWORK_ONLY: ("B.b1", "B.b2", "B.b_last"),
}


def rework_required_leaves(mode: str) -> tuple[str, ...]:
    return REWORK_REQUIRED.get(mode or MODE_FULL, REWORK_REQUIRED[MODE_FULL])


def template_violations(plan: dict) -> list[str]:
    """template=rework 计划的**必留节点 lint**（缺失/乱序即违规）；非模板计划恒空。"""
    if str(plan.get("template") or "") != "rework":
        return []
    required = list(rework_required_leaves(str(plan.get("rework_mode") or MODE_FULL)))
    actual = _tree(plan).leaf_paths()
    out: list[str] = []
    missing = [p for p in required if p not in actual]
    if missing:
        out.append("必留节点缺失：" + "、".join(missing))
    present = [p for p in actual if p in required]
    if present != required:
        out.append(
            "必留节点乱序（应 "
            + " → ".join(required)
            + "；实 "
            + " → ".join(present)
            + "）"
        )
    return out


def _rework_A(tb_id: str, rnd: int) -> dict:
    """A 综合项目改进（serial superstep）：a1 成因分析 → … → a5 改进实施。

    ``rnd>=2`` 时 a1 触发**历次改进强制展示**（``expand_history``）且 evidence 须含
    「前次改进复盘」标题段（§4.3 #4）。
    """
    a1_sections = ["成因分析", "结论"]
    a1: dict = {"id": "a1", "kind": "self", "brief": "成因分析报告", "role": "cause"}
    if rnd >= 2:
        a1_sections = ["前次改进复盘", "成因分析", "结论"]
        a1["expand_history"] = True
    a1["evidence_sections"] = a1_sections
    return {
        "id": "A",
        "mode": "serial",
        "brief": "综合项目改进",
        "children": [
            a1,
            {
                "id": "a2",
                "kind": "self",
                "brief": "解决方案报告",
                "role": "solution",
                "evidence_sections": ["解决方案", "结论"],
            },
            {
                "id": "a3",
                "kind": "self",
                "brief": f"送审 jury（改进后状态 vs 现状；案卷 {tb_id}-jury）",
                "role": "review",
                "evidence_sections": ["送审结论"],
            },
            {
                "id": "a4",
                "kind": "self",
                "brief": "拆解改进派单",
                "role": "improvement_plan",
                "gate_jury": f"{tb_id}-jury",
            },
            {"id": "a5", "member": "devops", "brief": "改进实施"},
        ],
    }


def _rework_B(node_id: str, tb_id: str) -> dict:
    """B 返工执行（serial superstep）：b1 重拆解 → b2 返工生产 → b_last 全流程报告。"""
    return {
        "id": node_id,
        "mode": "serial",
        "brief": "返工执行",
        "children": [
            {"id": "b1", "kind": "self", "brief": "重拆解返工", "role": "rework_plan"},
            {"id": "b2", "member": "devops", "brief": "返工生产环节"},
            {
                "id": "b_last",
                "kind": "self",
                "brief": "全流程报告组装",
                "role": "final_report",
            },
        ],
    }


def _rework_C(tb_id: str) -> dict:
    """C 监察改进（serial superstep）：可发现性送审 → 方案 → 派单 → 实施。"""
    return {
        "id": "C",
        "mode": "serial",
        "brief": "监察改进",
        "children": [
            {
                "id": "c1",
                "kind": "self",
                "brief": f"可发现性案卷送审（案卷 {tb_id}-discoverability）",
                "role": "discoverability",
                "evidence_sections": ["可发现性判断"],
            },
            {"id": "c2", "kind": "self", "brief": "改进方案（若判需改进）"},
            {
                "id": "c3",
                "kind": "self",
                "brief": "拆解派单",
                "gate_jury": f"{tb_id}-discoverability",
            },
            {"id": "c4", "member": "devops", "brief": "改进实施"},
        ],
    }


def build_rework_plan(
    tb_id: str, *, mode: str = MODE_FULL, goal: str = "", rework_round: int = 1
) -> dict:
    """生成返工 plan **骨架**（PSEP_3 §4.2 三档）。

    结构：A、BC 顶层 serial（A 全 done → B 起）；BC 并行组内 B id 靠前、C 靠后（B 派发后
    C 即可推进＝派发级）；B 是串行链（重拆解→返工生产→全流程报告）。``rework_round>=2``
    时 a1 触发历次改进强制展示 + 复盘段 lint（§4.3 #4）。

    Raises:
        ValueError: 未知模式。
    """
    if mode not in REWORK_MODES:
        raise ValueError(f"未知返工模式：{mode}（合法：{'、'.join(REWORK_MODES)}）")
    rnd = int(rework_round or 1)
    if mode == MODE_FULL:
        steps = [
            _rework_A(tb_id, rnd),
            {
                "id": "BC",
                "mode": "parallel",
                "brief": "返工与监察（并行）",
                "children": [_rework_B("B", tb_id), _rework_C(tb_id)],
            },
        ]
    elif mode == MODE_NO_INSPECTOR:
        steps = [_rework_A(tb_id, rnd), _rework_B("B", tb_id)]
    else:  # rework_only
        steps = [_rework_B("B", tb_id)]
    return {
        "goal": goal or f"返工：{tb_id}",
        "template": "rework",
        "rework_mode": mode,
        "rework_round": rnd,
        "taskbook": tb_id,
        "steps": steps,
    }


def plan_tree_lines(plan_id: str) -> list[str]:
    """plan 的 **superstep 进度树**（文本；组带聚合态、叶子带 ``kind=status``）——TUI 返工卡片用。"""
    if not _plan_path(plan_id).exists():
        return []
    plan = _load_yaml(plan_id)
    status_of = _status_of(_load_state(plan_id))
    tree = _tree(plan)
    icons = {
        "done": "✔",
        "queued": "▶",
        "self_awaiting": "✎",
        "blocked": "✗",
        "failed": "✗",
        "run_failed": "✗",
    }
    out: list[str] = []

    def walk(items: list, prefix: str, depth: int) -> None:
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("id") or f"s{i + 1}")
            path = f"{prefix}.{name}" if prefix else name
            indent = "  " * depth
            kids = raw.get("children")
            if kids:
                agg = tree.aggregate(path, status_of)
                mark = "✔" if agg["done"] else ("▶" if agg["started"] else "·")
                out.append(f"{indent}{mark} {name} [{raw.get('mode') or 'serial'}]")
                walk(kids, path, depth + 1)
            else:
                st = status_of(path)
                kind = str(raw.get("kind") or "dispatch")
                out.append(f"{indent}{icons.get(st, '·')} {name} ({kind}={st})")

    walk(plan.get("steps") or [], "", 0)
    return out


def _jury_case_ruling(case_id: str) -> tuple[str, str]:
    """读 jury 案卷留痕的**确定裁决**：返回 ``(verdict∈{"A","B",""}, 裁决书文本)``。"""
    from . import jury as _jury

    p = _jury.record_path(case_id)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    v = str(rec.get("final_verdict") or rec.get("verdict") or "")
    text = str((rec.get("verdict_summary") or {}).get("text") or "")
    return (v if v in ("A", "B") else ""), text


def _abandon_for_gate(
    plan_id: str, plan: dict, state: dict, leaf: dict, fails: list[dict]
) -> int:
    """两次送审不过 → 中止计划（abandoned）＋任务书 failed ＋附两轮裁决书（§4.3 #3）。"""
    state["abandoned_at"] = _now()
    state["abandon_reason"] = f"两次送审不过（{leaf['id']} gate）"
    state["gate_rulings"] = fails
    _save_state(plan_id, state)
    tb = str(plan.get("taskbook") or state.get("taskbook") or "")
    if tb:
        try:
            from . import taskbooks as _tb

            _tb.transition(
                tb, _tb.STATUS_FAILED, note="两次送审不过，已中止（附两轮裁决书）"
            )
        except Exception:  # noqa: BLE001 - 中止链绝不因台账写失败而回退
            pass
    print(f"== 计划 {plan_id} 已中止（abandoned）：两次送审不过（{leaf['id']}）==")
    for i, f in enumerate(fails, 1):
        print(f"  第 {i} 轮裁决书：{f.get('text') or f.get('case')}")
    if tb:
        print(f"  任务书 {tb} → failed（待用户补充指引；附两轮裁决书于 plan 状态）")
    return 2


def _gate_blocks(
    plan_id: str, plan: dict, state: dict, leaf: dict, sy: dict
) -> int | None:
    """送审 gate（§4.3 #2/#3）：``gate_jury`` 叶子派发前置校验——案卷 ``verdict=A`` 才放行。

    Returns ``None``＝放行；否则返回拒绝码（``2``）——两次 verdict=B 时中止计划并返回。
    """
    case_id = str(sy.get("gate_jury") or "")
    if not case_id:
        return None
    verdict, ruling = _jury_case_ruling(case_id)
    if verdict == "A":
        return None
    sid = str(leaf["id"])
    fails = state.setdefault("gate_fails", {}).setdefault(sid, [])
    if verdict == "B":
        fails.append({"ts": _now(), "case": case_id, "text": ruling})
        if len(fails) >= 2:
            return _abandon_for_gate(plan_id, plan, state, leaf, fails)
        _save_state(plan_id, state)
        print(
            f"[!] 送审 gate 未过（第 {len(fails)} 次 verdict=B，案卷 {case_id}）"
            f"——拒绝派发 {sid}"
        )
        print(f"    裁决书：{ruling or '（未读得裁决书）'}")
        _print_worklist(plan_id, plan, state)
        return 2
    print(
        f"[!] 送审 gate：案卷 {case_id} 未落/未裁决（verdict 非 A）"
        f"——先送审（orch jury）再派单 {sid}"
    )
    _print_worklist(plan_id, plan, state)
    return 2


def cmd_plan_rework(args) -> int:
    """``orch plan rework <tb-id> [--mode …]``：由返工任务书生成三档骨架并登记（§4.2）。"""
    tb_id = args.id
    mode = str(getattr(args, "mode", None) or MODE_FULL)
    if mode not in REWORK_MODES:
        print(f"[!] 未知返工模式：{mode}（合法：{'、'.join(REWORK_MODES)}）")
        return 2
    goal = str(getattr(args, "goal", "") or "")
    rnd = 1
    try:
        from . import taskbooks as _tb

        rnd = int((_tb.load_state(tb_id) or {}).get("rework_round") or 1)
    except Exception:  # noqa: BLE001 - 任务书缺席不拦骨架（round 回落 1）
        rnd = 1
    try:
        plan = build_rework_plan(tb_id, mode=mode, goal=goal, rework_round=rnd)
    except ValueError as exc:
        print(f"[!] {exc}")
        return 2
    plan_id = str(getattr(args, "plan_id", "") or f"rework-{tb_id}")
    if _plan_path(plan_id).exists() and not getattr(args, "force", False):
        print(f"[!] 计划 {plan_id} 已存在（覆盖用 --force）；续跑用 plan run {plan_id}")
        return 2
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan["id"] = plan_id
    _plan_path(plan_id).write_text(
        yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _save_state(plan_id, _reconcile(plan, {"plan_id": plan_id, "created": _now()}))
    print(
        f"[√] 返工计划已登记：{plan_id}（模式 {mode}，模板骨架 {len(plan['steps'])} 顶层节点）"
    )
    print(f"    任务书 {tb_id}；必留叶子：{'、'.join(rework_required_leaves(mode))}")
    tree = _tree(plan)
    for leaf in tree.leaves:
        indent = "  " * leaf.path.count(".")
        who = leaf.yaml.get("member") or (
            "self（组长）" if leaf.kind == KIND_SELF else "?"
        )
        gate = (
            f"  [gate:{leaf.yaml['gate_jury']}]" if leaf.yaml.get("gate_jury") else ""
        )
        print(
            f"    - {indent}{leaf.path}: {who} | {str(leaf.yaml.get('brief', ''))[:50]}{gate}"
        )
    return 0


def cmd_plan_amend(args) -> int:
    """amend = 直接编辑计划 YAML（组长文件工具可写 plans/*.yaml）；本命令做校验与状态对账。

    护栏（PSEP_3 §2.4）：**已 started 的 superstep 结构冻结**——删/挪已派发（或已 started
    子树内）节点即拒绝（rc=2，不改状态）；未派发部分放行。template=rework 计划另加**必留节点**
    lint（模板节点删除/乱序即拒，§4.3 #1）。附带打印结构 lint（软提示，不阻断）。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    old = _load_state(plan_id)
    tpl = template_violations(plan)
    if tpl:
        print("[!] amend 被拒：rework 模板必留节点不完整（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        print("  （返工模板节点不可删/不可乱序）")
        return 2
    violations = freeze_violations(
        {str(s["id"]): str(s.get("status") or "pending") for s in old.get("steps", [])},
        _tree(plan).leaf_paths(),
    )
    if violations:
        print("[!] amend 被拒：已 started 的 superstep 结构冻结（PSEP_3 §2.4）——")
        for v in violations:
            print(f"  - {v}")
        print("  （amend 只能改未派发部分；已派发/已 started 子树内节点不得删/挪）")
        return 2
    state = _reconcile(plan, old)
    _save_state(plan_id, state)
    rel: Any = _plan_path(plan_id)
    try:
        rel = rel.relative_to(ORCH_STATE_ROOT.parent)
    except ValueError:  # PLANS_DIR 被隔离到 tmp（测试面）：原样显示
        pass
    print(f"[√] 计划 {plan_id} 已对账（编辑 {rel} 后运行本命令校验）：")
    st = {str(s["id"]): s for s in state["steps"]}
    for leaf in _tree(plan).leaves:
        ss = st.get(leaf.path, {})
        print(
            f"  - {leaf.path}: {leaf.yaml.get('member', '?')} "
            f"[{ss.get('status', '?')}] {str(leaf.yaml.get('brief', ''))[:50]}"
        )
    _print_plan_lints(plan)
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
