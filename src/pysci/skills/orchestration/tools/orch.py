"""pysci-orch 统一 CLI 入口——组长专属的编排工作通道（README §4）。

门面模式对齐其余 ``pysci-*`` CLI：argparse 子命令 + 薄编排。设计红线（README §4.3）：
每条命令输出自含「下一步」（[NEXT] 段）；state 全为可读格式；orch 故障可降级手动。

子命令分组（Phase 1 范围；plan/approve/review/consult 于 Phase 2/3 接入）::

    uv run pysci-orch status
    ... orch dispatch <member> --task <file> | --text "<任务>" [--session latest|new|<sid>]
    ... orch sessions <member> [--all] [--archive <sid> [--distill]] [--prune <sid>]
    ... orch ledger [--member M] [--days N] [--stats]
    ... orch sync [--check]

长任务纪律：组长应经**后台 Bash**运行 dispatch（README §4.5 等待模型）；orch 自身
同步阻塞等待组员进程属预期行为（进程级等待，零 token）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import sessions as _sessions
from . import sync as _sync
from .checks import run_checks
from .delivery import parse_delivery
from .ledger import LEDGER_PATH, LedgerEntry, append, read_all, summarize
from .registry import Registry, orchestration_rel
from .runner import (
    PROTOCOL_REMINDER,
    Envelope,
    build_command,
    build_env,
    new_session_id,
    run_headless,
)

_BACKLOG = ORCH_STATE_ROOT / "backlog.json"
_REPLIES_DIR = ORCH_STATE_ROOT / "replies"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_next(lines: list[str]) -> None:
    """输出 [NEXT] 段——flash 级模型也能照做的下一步指引（README §4.3 红线 1）。"""
    print()
    print("[NEXT]")
    for line in lines:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def _collect_replies(member_id: str) -> tuple[str, list[Path]]:
    """收集待附送的审批回复（Phase 2 approve 写入；现在兼容手动投放）。"""
    d = _REPLIES_DIR / member_id
    if not d.exists():
        return "", []
    files = sorted(p for p in d.glob("*.md"))
    if not files:
        return "", []
    body = "\n\n".join(
        f"### 组长对你上次改进建议的回复（{p.stem}）\n{p.read_text(encoding='utf-8')}"
        for p in files
    )
    return body, files


def _deployed_skills_for(pod: Path) -> list[str]:
    """从 manifest 解析部署到指定 pod 的技能名（pod-guard 保护部署副本不被成员改写）。"""
    names: list[str] = []
    skills_dir = (pod / ".qoder" / "skills").resolve()
    try:
        for spec in _sync.load_manifest():
            if any(Path(t).resolve() == skills_dir for t in spec.targets):
                names.append(spec.name)
    except OSError:
        pass
    return names


def _write_taskbook(member_pod: Path, args: argparse.Namespace, slug: str) -> Path:
    """把 --task 文件复制或 --text 内容写成 inbox 任务书，返回任务书路径。"""
    inbox = member_pod / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    dest = inbox / f"task-{_ts()}-{slug}.md"
    header = f"# 任务书 {slug}\n\n（派发时间：{datetime.now().astimezone().isoformat(timespec='seconds')}）\n\n"
    if args.task:
        src = Path(args.task)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        dest.write_text(header + src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dest.write_text(header + args.text, encoding="utf-8")
    return dest


def cmd_dispatch(args: argparse.Namespace) -> int:
    """派发一跳组员任务：任务书落 inbox → headless 执行 → 解析交付 → 验收 → 台账。"""
    reg = Registry.load()
    reg.ensure_member_defaults(args.member)
    m = reg.member(args.member)
    if not m.pod.exists():
        print(f"[!] 成员 pod 不存在：{orchestration_rel(m.pod)}")
        print("    请先建 pod 骨架（Phase 1 清单），或检查 registry 成员 id。")
        return 2
    model = reg.models.get(args.model_tier or m.model_tier, "")
    slug = args.name or (Path(args.task).stem if args.task else f"adhoc-{_ts()}")

    # 会话选择（README §3.2：默认=继续最近活跃会话）
    pool_lines = _sessions.format_pool(m)
    if args.session == "new":
        sid, resume = new_session_id(), False
    elif args.session == "latest":
        active = m.active_sessions
        if active:
            sid, resume = active[0]["sid"], True
        else:
            sid, resume = new_session_id(), False
    else:
        found = m.find_session(args.session)
        if found is None:
            print(f"[!] 会话 '{args.session}' 未在池中找到。当前池：")
            print("\n".join(pool_lines))
            return 2
        sid, resume = found["sid"], True
    if not resume:
        _sessions.register_session(m, sid, slug)
    name = slug if not resume else (m.find_session(sid) or {}).get("name", slug)

    taskbook = _write_taskbook(m.pod, args, slug)
    replies_body, reply_files = _collect_replies(args.member)
    prompt_parts = [
        f"你的任务书（必读，位于你的 pod 内）：{taskbook.name}",
        f"绝对路径：{taskbook}",
    ]
    if replies_body:
        prompt_parts.append(f"\n{replies_body}")
    prompt_parts.append(f"\n{PROTOCOL_REMINDER}")
    prompt = "\n".join(prompt_parts)

    task_dirs = [d.strip() for d in (args.dirs or "").split(",") if d.strip()]
    cmd = build_command(
        m, prompt, session_id=sid, resume=resume, model=model, max_turns=args.max_turns
    )
    env = build_env(m, task_dirs, deployed_skills=_deployed_skills_for(m.pod))

    print(
        f"→ 派发 {args.member}（会话 {'续:' + sid[:8] if resume else '新:' + sid[:8]}"
        f"，模型 {model or '(默认)'}，任务书 {taskbook.name}）"
    )
    rc, out, err = run_headless(
        cmd, cwd=PROJECT_ROOT, env=env, timeout_s=args.timeout or m.timeout_s
    )
    env_json = Envelope.parse(out)

    # 失败自动重试一次（README §3.3：注入错误上下文，同会话续）
    if (rc != 0 or env_json.is_error) and not args.no_retry:
        print(f"[!] 首跑失败（rc={rc}, stop={env_json.stop_reason}），自动重试一次…")
        retry_prompt = (
            f"上一次运行异常中断（rc={rc}；stderr 摘要：{err.strip()[-400:]}）。"
            f"请继续完成任务书 {taskbook}，按交付协议收尾。\n\n{PROTOCOL_REMINDER}"
        )
        cmd2 = build_command(
            m,
            retry_prompt,
            session_id=sid,
            resume=True,
            model=model,
            max_turns=args.max_turns,
        )
        rc, out, err = run_headless(
            cmd2, cwd=PROJECT_ROOT, env=env, timeout_s=args.timeout or m.timeout_s
        )
        env_json = Envelope.parse(out)

    if rc != 0 or env_json.is_error:
        append(
            LedgerEntry(
                ts="",
                member=args.member,
                session_id=sid,
                session_name=name,
                task_file=orchestration_rel(taskbook),
                kind="run_failed",
                num_turns=env_json.num_turns,
                duration_ms=env_json.duration_ms,
                credits=env_json.total_credits,
                ctx_ratio=env_json.context_usage_ratio,
                permission_denials=len(env_json.permission_denials),
                error=(err.strip()[-400:] or env_json.stop_reason),
            )
        )
        _sessions.touch_session(m, sid)
        reg.write_member(m)
        reg.save()
        print(f"[✗] 运行失败（rc={rc}）。stderr 摘要：{err.strip()[-400:]}")
        print(f"    envelope.result 原文（前 600 字）：\n{env_json.result[:600]}")
        _print_next(
            [
                "运行级失败（非交付失败）。可选：",
                f'  1) 再试一跳：uv run pysci-orch dispatch {args.member} --session {sid[:8]} --text "继续任务书 {taskbook.name}"',
                "  2) 查会话原文定位卡点：见 registry 中该会话 jsonl 路径",
                "  3) 仍失败 → 上报用户（附本输出全文）",
            ]
        )
        return 2

    delivery = parse_delivery(env_json.result)
    check_results = (
        [] if args.no_checks else run_checks(delivery.artifacts, reg, member_pod=m.pod)
    )
    append(
        LedgerEntry(
            ts="",
            member=args.member,
            session_id=sid,
            session_name=name,
            task_file=orchestration_rel(taskbook),
            kind=delivery.kind,
            checks=check_results,
            infra_suggestion=bool(delivery.infra_suggestion),
            num_turns=env_json.num_turns,
            duration_ms=env_json.duration_ms,
            credits=env_json.total_credits,
            ctx_ratio=env_json.context_usage_ratio,
            permission_denials=len(env_json.permission_denials),
            plan=args.plan or "",
            step=args.step or "",
        )
    )
    # 审批回复已随本跳附送 → 移入 sent/（闭环）
    for p in reply_files:
        sent = p.parent / "sent"
        sent.mkdir(exist_ok=True)
        shutil.move(str(p), str(sent / p.name))

    _sessions.touch_session(m, sid)
    reg.write_member(m)
    reg.save()

    # ---- 汇总输出（面向组长的结构化报告 + [NEXT]）----
    icon = {"result": "√", "blocked": "✗", "parse_error": "?"}[delivery.kind]
    print(
        f"[{icon}] {args.member} 交付类型：{delivery.kind}"
        f"（{env_json.num_turns} 轮，{env_json.duration_ms / 1000:.0f}s，"
        f"credits={env_json.total_credits}，ctx={env_json.context_usage_ratio:.0%}）"
    )
    print(f"    会话：{sid[:8]}（{name}）")
    body_preview = delivery.body[:800]
    print(
        f"---- 交付正文（前 800 字）----\n{body_preview}\n----------------------------"
    )
    if check_results:
        print("机械验收：")
        for c in check_results:
            mark = {
                "pass": "√",
                "fail": "✗",
                "unrouted": "○ 未注册(放行)",
                "skipped": "– 成员判断不验收",
            }[c["verdict"]]
            line = f"  [{mark}] {c['check']}: {c['path']}"
            if c["verdict"] == "fail":
                line += f"\n      输出摘要：{c.get('output_tail', '')[-300:]}"
            if c.get("reason"):
                line += f"（理由：{c['reason']}）"
            print(line)
    if delivery.infra_suggestion:
        print(
            f"---- 改进建议 ----\n{delivery.infra_suggestion[:600]}\n------------------"
        )
    if env_json.permission_denials:
        print(
            f"[!] 权限拒绝 {len(env_json.permission_denials)} 次（配置体检信号）：{env_json.permission_denials[:3]}"
        )

    failed_checks = [c for c in check_results if c["verdict"] == "fail"]
    if delivery.kind == "result" and not failed_checks:
        nxt = ["本环节完成。"]
        if delivery.infra_suggestion:
            nxt.append(
                "改进建议待审批（Phase 2 前手动记录到 backlog.json 并回复组员）："
            )
            nxt.append(
                f"  编辑 {orchestration_rel(_BACKLOG)} 追加条目；审批回复写 {_REPLIES_DIR / args.member}/<id>.md"
            )
        nxt.append(
            "若属多环节工作：书写下一环节任务书后再次 dispatch（Phase 2 起由 plan 系统接管）。"
        )
        nxt.append(f"抽检产物：uv run pysci-orch ledger --member {args.member}")
        _print_next(nxt)
        return 0
    if failed_checks:
        detail = "; ".join(
            f"{c['check']}:{Path(c['path']).name}" for c in failed_checks
        )
        _print_next(
            [
                f"机械验收 FAIL（{detail}）→ 建议回派返工：",
                f'  uv run pysci-orch dispatch {args.member} --session {sid[:8]} --text "返工：机械验收未过（{detail}）。失败详情见你上次交付的验收输出，修正后重新交付。"',
                "同指纹连败 3 次 → 停止回派，升级决策（改任务书/咨询副组长/上报用户）。",
            ]
        )
        return 2
    if delivery.kind == "blocked":
        _print_next(
            [
                "组员被阻塞。可选：",
                f'  1) 补充信息重派：uv run pysci-orch dispatch {args.member} --session {sid[:8]} --text "<补充>"',
                "  2) 咨询副组长获取技术意见（Phase 2 起 orch consult；当前可直接 dispatch deputy）",
                "  3) 上报用户（附交付正文）",
            ]
        )
        return 2
    _print_next(
        [
            "交付格式解析失败（parse_error）。原回复见上方正文预览。",
            f'重派要求补格式：uv run pysci-orch dispatch {args.member} --session {sid[:8]} --text "你上次交付缺少 <result>/<blocked> 标签块，请按 charter 交付协议重新收尾（工作内容不必重做）。"',
        ]
    )
    return 2


# ---------------------------------------------------------------------------
# sessions
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
                env=build_env(m, [], deployed_skills=_deployed_skills_for(m.pod)),
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
    lines = _sessions.format_pool(m, limit=99 if args.all else 5)
    print(f"成员 {args.member}（pod: {orchestration_rel(m.pod)}）")
    print("\n".join(lines))
    if args.all:
        for s in m.sessions:
            if s.get("status") in ("archived", "pruned"):
                print(f"  - [{s['status']}] {s['sid'][:8]}  {s.get('name', '')}")
    return 0


# ---------------------------------------------------------------------------
# status / ledger / sync
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    """全景：成员池摘要、台账近况、backlog/待回复计数。"""
    del args
    reg = Registry.load()
    print("== 成员 ==")
    members = reg.data.get("members", {})
    if not members:
        print("  （registry 尚无成员；dispatch 时自动注册，或先完成 Phase 1 pod 骨架）")
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
            f"  {mid:<10} 活跃会话={stats['active']} 归档={stats['archived']}  {latest_s}{pod_mark}"
        )
    print("== 台账（近 5 跳）==")
    for e in read_all()[-5:]:
        checks_s = ",".join(c["verdict"] for c in e.checks) or "-"
        print(
            f"  {e.ts}  {e.member:<9} {e.kind:<11} 验收[{checks_s}] "
            f"{e.duration_ms / 1000:.0f}s cr={e.credits} ctx={e.ctx_ratio:.0%}"
        )
    backlog_n = 0
    if _BACKLOG.exists():
        import json

        backlog_n = len(
            json.loads(_BACKLOG.read_text(encoding="utf-8")).get("items", [])
        )
    replies_n = (
        sum(1 for p in _REPLIES_DIR.glob("*/*.md")) if _REPLIES_DIR.exists() else 0
    )
    print(f"== 队列 == backlog={backlog_n}  待附送审批回复={replies_n}")
    _print_next(
        [
            '派发：uv run pysci-orch dispatch <member> --task <file> | --text "..."（建议后台 Bash 运行）',
            "会话：uv run pysci-orch sessions <member>",
            "统计：uv run pysci-orch ledger --stats",
        ]
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """台账查询与统计。"""
    entries = read_all()
    if args.member:
        entries = [e for e in entries if e.member == args.member]
    if args.days:
        cutoff = (datetime.now().astimezone() - timedelta(days=args.days)).isoformat()
        entries = [e for e in entries if e.ts >= cutoff]
    if args.stats:
        s = summarize(entries)
        print(
            f"范围：{len(entries)} 跳中筛选后 {s['hops']} 跳；"
            f"总耗时 {s['duration_ms'] / 1000:.0f}s；总 credits {s['credits']}"
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
    n = args.tail
    for e in entries[-n:]:
        checks_s = ",".join(f"{c['check']}:{c['verdict']}" for c in e.checks) or "-"
        print(
            f"{e.ts}  {e.member:<9} {e.kind:<11} sid={e.session_id[:8]} "
            f"{e.num_turns}轮 {e.duration_ms / 1000:.0f}s cr={e.credits} "
            f"ctx={e.ctx_ratio:.0%} 验收[{checks_s}]"
            + (" 建议✓" if e.infra_suggestion else "")
            + (f" 计划={e.plan}/{e.step}" if e.plan else "")
        )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """技能真本 → 部署副本同步（--check 只报告）。"""
    report = _sync.sync(check_only=args.check)
    print("\n".join(report))
    if any(line.startswith("[!] manifest") for line in report):
        _print_next(["尚无 manifest：先完成技能真本迁移（Phase 1 清单 P1-4）。"])
        return 2
    drift = [line for line in report if line.startswith("[!]")]
    if drift and args.check:
        _print_next(
            ["存在漂移：运行 uv run pysci-orch sync 重新部署（真本→副本单向）。"]
        )
        return 2
    return 0


# ---------------------------------------------------------------------------
# argparse 门面
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        prog="pysci-orch",
        description="pySci 多 Agent 编排 CLI（组长专属工作通道；设计见 orchestration/README.md）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="全景：成员池/台账近况/队列计数")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "dispatch", help="派发一跳组员任务（长任务建议后台 Bash 运行本命令）"
    )
    p.add_argument("member", help="成员 id（registry/pods 目录名）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="任务书文件路径（复制进 pod inbox）")
    g.add_argument("--text", help="任务书正文（直接写入 pod inbox）")
    p.add_argument(
        "--session",
        default="latest",
        help="latest（默认，续最近活跃会话）| new | <sid前缀>",
    )
    p.add_argument("--name", help="会话/任务 slug（新会话命名用）")
    p.add_argument("--model-tier", choices=["max", "flash"], help="覆盖成员默认档位")
    p.add_argument("--max-turns", type=int, help="覆盖成员默认轮数上限")
    p.add_argument("--timeout", type=int, help="墙钟超时秒数（覆盖成员默认）")
    p.add_argument("--dirs", help="任务白名单目录（逗号分隔；注入 pod-guard）")
    p.add_argument("--plan", help="所属计划 id（Phase 2；当前仅记台账）")
    p.add_argument("--step", help="计划环节 id（同上）")
    p.add_argument(
        "--no-checks", action="store_true", help="跳过机械验收（抽检例外用）"
    )
    p.add_argument("--no-retry", action="store_true", help="禁用失败自动重试一次")
    p.set_defaults(func=cmd_dispatch)

    p = sub.add_parser("sessions", help="成员会话池：清单/归档/清理")
    p.add_argument("member")
    p.add_argument("--all", action="store_true", help="含归档/已清理会话")
    p.add_argument("--archive", metavar="SID", help="归档会话（可配 --distill）")
    p.add_argument(
        "--distill", action="store_true", help="归档前先派蒸馏跳（写 AGENTS.md）"
    )
    p.add_argument("--prune", metavar="SID", help="删除会话（按 sid 匹配原生序号删除）")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("ledger", help="台账查询与统计")
    p.add_argument("--member", help="按成员过滤")
    p.add_argument("--days", type=int, help="按最近 N 天过滤")
    p.add_argument("--tail", type=int, default=15, help="列出最近 N 条（默认 15）")
    p.add_argument("--stats", action="store_true", help="聚合统计（总量+成员占比）")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("sync", help="技能真本 → 部署副本同步")
    p.add_argument("--check", action="store_true", help="只检查漂移不写盘")
    p.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
