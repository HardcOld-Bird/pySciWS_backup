"""pysci-orch 统一 CLI 入口——组长专属的编排工作通道（README §4）。

门面模式对齐其余 ``pysci-*`` CLI：argparse 子命令 + 薄编排，业务逻辑分布在同包模块：
registry / runner / delivery / checks / ledger / sessions / sync / dispatch / plans /
workflow。设计红线（README §4.3）：每条命令输出自含「下一步」（[NEXT] 段）；state 全为
可读格式；orch 故障可按 README 附录 A 降级手动。

命令面（plan-first：组长日常走 plan；dispatch/sessions 等为底层命令，降级与调试用）::

    uv run pysci-orch status
    ... orch plan new <file> | run <id> [--text/--task ...] | amend <id> | drop <id> | list
    ... orch plan adhoc <member> --text "..."        # 计划外事项的正确姿势
    ... orch approve|reject <suggestion-id> --note "..."
    ... orch suggestions | stats [--plan/--member/--days] | consult "<问题>"
    ... orch dispatch <member> (--task F | --text T | --from-backlog) [--session ...]
    ... orch sessions <member> [--all|--archive SID [--distill]|--prune SID]
    ... orch ledger [--member --days --stats] | sync [--check]

长任务纪律：dispatch / plan run 应以 Bash 工具的 run_in_background 启动（README §4.5
等待模型）；orch 自身同步阻塞等待组员进程属预期（进程级等待，零 token）。
"""

from __future__ import annotations

import argparse
import sys

from . import plans as _plans
from . import sessions as _sessions
from . import sync as _sync
from . import workflow as _workflow
from .dispatch import do_dispatch
from .ledger import LEDGER_PATH, read_all, summarize
from .registry import Registry, orchestration_rel


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    """全景：进行中计划、成员池摘要、待审批建议、backlog、台账近况。"""
    del args
    print("== 计划 ==")
    live = 0
    for pid, goal, done, total in _plans.list_plans():
        st = _plans.load_state(pid)
        if st.get("completed_at") or st.get("abandoned_at"):
            continue
        live += 1
        cur = next(
            (
                s["id"]
                for s in st.get("steps", [])
                if s["status"] not in ("done", "skipped")
            ),
            "-",
        )
        print(f"  ▶ {pid}  {done}/{total}（当前环节 {cur}）  {goal}")
    if not live:
        print("  （无进行中计划）")

    print("== 成员 ==")
    reg = Registry.load()
    members = reg.data.get("members", {})
    if not members:
        print("  （registry 尚无成员；dispatch 时自动注册）")
    for mid in sorted(members):
        m = reg.member(mid)
        stats = _sessions.pool_stats(m)
        latest = stats["latest"]
        latest_s = (
            f"最近={latest['name']}({latest['sid'][:8]}, 跳数{latest.get('hops', 0)})"
            if latest
            else "无活跃会话"
        )
        pod_mark = "" if m.pod.exists() else "  [pod 缺失!]"
        print(
            f"  {mid:<10} 活跃会话={stats['active']} 归档={stats['archived']}  "
            f"{latest_s}{pod_mark}"
        )

    sugg = _workflow.pending_suggestions()
    back = _workflow.backlog_pending()
    replies_n = (
        sum(1 for p in _workflow.REPLIES_DIR.glob("*/*.md"))
        if _workflow.REPLIES_DIR.exists()
        else 0
    )
    print(
        f"== 队列 == 待审批建议={len(sugg)}  backlog待办={len(back)}  "
        f"待附送回复={replies_n}"
    )
    for p in sugg:
        print(f"  [建议] {p.stem}")
    for b in back[:5]:
        print(f"  [backlog] {b['id']}: {str(b.get('summary', ''))[:60]}")

    print("== 台账（近 5 跳）==")
    for e in read_all()[-5:]:
        checks_s = ",".join(c["verdict"] for c in e.checks) or "-"
        print(
            f"  {e.ts}  {e.member:<9} {e.kind:<11} 验收[{checks_s}] "
            f"{e.duration_ms / 1000:.0f}s ctx={e.ctx_ratio:.0%}"
            + (f" 计划={e.plan}/{e.step}" if e.plan else "")
        )

    print()
    print("[NEXT]")
    if sugg:
        print("  有待审批建议：uv run pysci-orch suggestions 查看 → approve/reject")
    if live:
        print(
            '  推进进行中计划：uv run pysci-orch plan run <id> --text "<当前环节任务书>"'
        )
    else:
        print(
            "  新工作：uv run pysci-orch plan new <计划.yaml>（格式见 README §4.2）；"
        )
        print('  计划外单步：uv run pysci-orch plan adhoc <member> --text "..."')
    return 0


# ---------------------------------------------------------------------------
# dispatch（底层命令；日常建议走 plan）
# ---------------------------------------------------------------------------
def cmd_dispatch(args: argparse.Namespace) -> int:
    """派发一跳组员任务（底层命令；组长日常统一走 plan 系统）。"""
    text, task_file = args.text, args.task
    took_backlog_id = ""
    if args.from_backlog:
        item = _workflow.backlog_take_first()
        if item is None:
            print("[!] backlog 无 pending 条目")
            return 2
        took_backlog_id = item["id"]
        text = (
            f"基础设施改进任务（backlog FIFO 队首，id={item['id']}）：\n\n"
            f"摘要：{item.get('summary', '')}\n\n"
            f"证据：{item.get('evidence_file', '（见建议归档）')}\n\n"
            f"组长附注：{item.get('note', '（无）')}\n\n"
            "要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、"
            "commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，"
            f"并注明 backlog id={item['id']} 以便销账。"
        )
        if not args.name:
            args.name = item["id"]
    if not (text or task_file):
        print("[!] 需要 --task/--text/--from-backlog 之一")
        return 2
    outcome = do_dispatch(
        args.member,
        text=text,
        task_file=task_file,
        slug=args.name,
        session=args.session,
        model_tier=args.model_tier,
        max_turns=args.max_turns,
        timeout=args.timeout,
        dirs=[d.strip() for d in (args.dirs or "").split(",") if d.strip()],
        plan=args.plan or "",
        step=args.step or "",
        no_checks=args.no_checks,
        no_retry=args.no_retry,
    )
    if took_backlog_id and outcome.code == 0:
        _workflow.backlog_complete(took_backlog_id, note="orch 自动销账")
        print(f"[√] backlog 条目 {took_backlog_id} 已标记完成")
    return outcome.code


# ---------------------------------------------------------------------------
# sessions / ledger / sync
# ---------------------------------------------------------------------------
def cmd_sessions(args: argparse.Namespace) -> int:
    """会话池管理：清单 / 归档（可先蒸馏）/ 清理。"""
    reg = Registry.load()
    reg.ensure_member_defaults(args.member)
    m = reg.member(args.member)
    if args.prune:
        sess = m.find_session(args.prune)
        if sess is None:
            print(f"[!] 未找到会话 {args.prune}")
            return 2
        print(_sessions.prune(m, sess["sid"]))
        reg.write_member(m)
        reg.save()
        return 0
    if args.archive:
        sess = m.find_session(args.archive)
        if sess is None:
            print(f"[!] 未找到会话 {args.archive}")
            return 2
        sid = sess["sid"]
        if args.distill:
            from pysci.paths import PROJECT_ROOT

            from .dispatch import deployed_skills_for
            from .runner import Envelope, build_command, build_env, run_headless

            model = reg.models.get(m.model_tier, "")
            prompt = (
                "归档前蒸馏：把本会话中值得跨会话复用的经验、约定、教训写入你的 "
                "AGENTS.md（只记稳定事实，控制体量；临时状态不要写）。完成后以 "
                "<result>distilled</result> 收尾。"
            )
            cmd = build_command(
                m, prompt, session_id=sid, resume=True, model=model, max_turns=8
            )
            rc, out, _ = run_headless(
                cmd,
                cwd=PROJECT_ROOT,
                env=build_env(m, [], deployed_skills=deployed_skills_for(m.pod)),
                timeout_s=900,
            )
            env_json = Envelope.parse(out)
            print(f"蒸馏跳：rc={rc}，result 摘要={env_json.result[:120]}")
            if rc != 0 or env_json.is_error:
                print("[!] 蒸馏跳失败，未归档（可去掉 --distill 强制归档）。")
                return 2
        print(_sessions.archive(m, sid))
        reg.write_member(m)
        reg.save()
        return 0
    print(f"成员 {args.member}（pod: {orchestration_rel(m.pod)}）")
    print("\n".join(_sessions.format_pool(m, limit=99 if args.all else 5)))
    if args.all:
        for s in m.sessions:
            if s.get("status") in ("archived", "pruned"):
                print(f"  - [{s['status']}] {s['sid'][:8]}  {s.get('name', '')}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """台账查询与统计。"""
    entries = read_all()
    if args.member:
        entries = [e for e in entries if e.member == args.member]
    if args.days:
        from datetime import datetime, timedelta

        cutoff = (datetime.now().astimezone() - timedelta(days=args.days)).isoformat()
        entries = [e for e in entries if e.ts >= cutoff]
    if args.stats:
        s = summarize(entries)
        print(
            f"范围筛选后 {s['hops']} 跳；总耗时 {s['duration_ms'] / 1000:.0f}s；"
            f"credits {s['credits']}"
        )
        for mid, mrow in sorted(s["by_member"].items()):
            print(
                f"  {mid:<10} 跳数={mrow['hops']:<4} 耗时占比={mrow['duration_pct']}%  "
                f"credits占比={mrow['credits_pct']}%"
            )
        return 0
    if not entries:
        print(f"（台账为空：{orchestration_rel(LEDGER_PATH)}）")
        return 0
    for e in entries[-args.tail :]:
        checks_s = ",".join(f"{c['check']}:{c['verdict']}" for c in e.checks) or "-"
        print(
            f"{e.ts}  {e.member:<9} {e.kind:<11} sid={e.session_id[:8]} "
            f"{e.num_turns}轮 {e.duration_ms / 1000:.0f}s ctx={e.ctx_ratio:.0%} "
            f"验收[{checks_s}]"
            + (" 建议✓" if e.infra_suggestion else "")
            + (f" 计划={e.plan}/{e.step}" if e.plan else "")
        )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """技能真本 → 部署副本同步（--check 只报告）。"""
    report = _sync.sync(check_only=args.check)
    print("\n".join(report))
    if any(line.startswith("[!] manifest") for line in report):
        print()
        print("[NEXT] 尚无 manifest：先完成技能真本迁移（README §12）。")
        return 2
    if [line for line in report if line.startswith("[!]")] and args.check:
        print()
        print("[NEXT] 存在漂移：uv run pysci-orch sync 重新部署（真本→副本单向）。")
        return 2
    return 0


# ---------------------------------------------------------------------------
# plan 子命令路由
# ---------------------------------------------------------------------------
def _plan_router(action: str):
    """把 plan 子命令路由到 plans 模块对应处理器。"""
    handlers = {
        "new": _plans.cmd_plan_new,
        "run": _plans.cmd_plan_run,
        "amend": _plans.cmd_plan_amend,
        "drop": _plans.cmd_plan_drop,
        "list": _plans.cmd_plan_list,
        "adhoc": _plans.cmd_plan_adhoc,
    }

    def _run(args: argparse.Namespace) -> int:
        return handlers[action](args)

    return _run


# ---------------------------------------------------------------------------
# argparse 门面
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        prog="pysci-orch",
        description="pySci 多 Agent 编排 CLI（组长专属；设计见 orchestration/README.md）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="全景：计划/成员/队列/台账")
    p.set_defaults(func=cmd_status)

    # ---- plan（组长日常面）----
    p = sub.add_parser("plan", help="计划驱动（plan-first：日常工作统一走这里）")
    psub = p.add_subparsers(dest="plan_action", required=True)

    q = psub.add_parser("new", help="登记计划 YAML（格式见 README §4.2）")
    q.add_argument("file")
    q.add_argument("--id", help="计划 id（默认取文件名）")
    q.add_argument("--force", action="store_true", help="覆盖同 id 计划")
    q.set_defaults(func=_plan_router("new"))

    q = psub.add_parser("run", help="推进计划：执行当前环节（任务书延迟书写）")
    q.add_argument("id")
    q.add_argument("--text", help="当前环节任务书正文")
    q.add_argument("--task", help="当前环节任务书文件")
    q.add_argument("--name", help="会话 slug 覆盖（默认 <plan>-<step>）")
    q.add_argument("--session", help="latest(默认)|new|<sid前缀>")
    q.add_argument("--model-tier", choices=["max", "flash"])
    q.add_argument("--max-turns", type=int)
    q.add_argument("--dirs", help="覆盖计划环节的白名单（逗号分隔）")
    q.set_defaults(func=_plan_router("run"))

    q = psub.add_parser("amend", help="编辑计划 YAML 后校验对账（按环节 id 保留状态）")
    q.add_argument("id")
    q.set_defaults(func=_plan_router("amend"))

    q = psub.add_parser("drop", help="中止计划（标记 abandoned，状态保留审计）")
    q.add_argument("id")
    q.set_defaults(func=_plan_router("drop"))

    q = psub.add_parser("list", help="列出全部计划与进度")
    q.set_defaults(func=_plan_router("list"))

    q = psub.add_parser("adhoc", help="计划外单步任务：登记单环节计划并立即执行")
    q.add_argument("member")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--task")
    q.add_argument("--dirs")
    q.add_argument("--model-tier", choices=["max", "flash"])
    q.add_argument("--max-turns", type=int)
    q.set_defaults(func=_plan_router("adhoc"))

    # ---- 改进循环 ----
    p = sub.add_parser("suggestions", help="列出待审批改进建议")
    p.set_defaults(func=_workflow.cmd_suggestions)

    p = sub.add_parser("approve", help="采纳建议 → backlog + 审批回复附送")
    p.add_argument("suggestion_id")
    p.add_argument("--note", default="")
    p.set_defaults(func=_workflow.cmd_approve)

    p = sub.add_parser("reject", help="否决建议 → 仅回复附送")
    p.add_argument("suggestion_id")
    p.add_argument("--note", default="")
    p.set_defaults(func=_workflow.cmd_reject)

    p = sub.add_parser("consult", help="咨询副组长（平级协作：异议义务提醒自动注入）")
    p.add_argument("question")
    p.add_argument("--session", help="latest(默认)|new|<sid前缀>")
    p.set_defaults(func=_workflow.cmd_consult)

    p = sub.add_parser("stats", help="三级统计：--plan 计划级 / --member 成员级 / 全体")
    p.add_argument("--plan")
    p.add_argument("--member")
    p.add_argument("--days", type=int)
    p.set_defaults(func=_workflow.cmd_stats)

    # ---- 底层命令（降级/调试用；日常勿绕开 plan）----
    p = sub.add_parser("dispatch", help="[底层] 单点派发一跳")
    p.add_argument("member")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--task")
    g.add_argument("--text")
    p.add_argument(
        "--from-backlog",
        action="store_true",
        help="devops 专用：自动取 backlog 队首生成任务书，交付后自动销账",
    )
    p.add_argument("--session", default="latest")
    p.add_argument("--name")
    p.add_argument("--model-tier", choices=["max", "flash"])
    p.add_argument("--max-turns", type=int)
    p.add_argument("--timeout", type=int)
    p.add_argument("--dirs")
    p.add_argument("--plan", help="所属计划 id（仅记台账）")
    p.add_argument("--step")
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--no-retry", action="store_true")
    p.set_defaults(func=cmd_dispatch)

    p = sub.add_parser("sessions", help="[底层] 成员会话池管理")
    p.add_argument("member")
    p.add_argument("--all", action="store_true")
    p.add_argument("--archive", metavar="SID")
    p.add_argument("--distill", action="store_true")
    p.add_argument("--prune", metavar="SID")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("ledger", help="[底层] 台账查询")
    p.add_argument("--member")
    p.add_argument("--days", type=int)
    p.add_argument("--tail", type=int, default=15)
    p.add_argument("--stats", action="store_true")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("sync", help="[底层] 技能真本 → 部署副本同步")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
