"""pysci-orch 统一 CLI 入口——组长专属的编排工作通道（README §4）。

门面模式对齐其余 ``pysci-*`` CLI：argparse 子命令 + 薄编排，业务逻辑分布在同包模块：
registry / runner / delivery / checks / ledger / sessions / sync / dispatch / plans /
workflow。设计红线（README §4.3）：每条命令输出自含「下一步」（[NEXT] 段）；state 全为
可读格式；orch 故障可按 README 附录 A 降级手动。

命令面（plan-first：组长日常走 plan；dispatch/sessions 等为底层命令，降级与调试用）::

    uv run --no-sync pysci-orch status
    ... orch plan new <file> | run <id> [--text/--task ...] | amend <id> | drop <id> | list
    ... orch plan adhoc <member> --text "..."        # 计划外事项的正确姿势
    ... orch approve <sid> (--jury | --bench <案id[,案id]> | --override "<理由>") [--note ...]
    ... orch reject <suggestion-id> --override "<依据>"
    ... orch suggestions | stats [--plan/--member/--days] | consult "<问题>"
    ... orch consult-twin <case-file>   # 咨询 twin（未启用则明确拒绝，§5.1）
    ... orch dispatch <member> (--task F | --text T | --from-backlog) [--session ...] [--effort …]
    ... orch watch [--timeout S=4h] [--interval S=60]   # 后台挂起等 drain 事件（§4.5）
    ... orch sessions <member> [--all|--archive SID [--distill]|--prune SID]
    ... orch ledger [--member --days --stats] | sync [--check]
    ... orch jury <case-file> [--dry] [--timeout S]   # Tier C 仲裁团（REFACTOR §4）
    ... orch taste <case-file> [--dry]        # twin 审美判断通道（宪章生效后可用，§9）
    ... orch bench-add | bench-up | bench-queue | bench-seen | bench-stop   # bencher 测量队列（§8）
    ... orch inspector-up [--dry|--force] | inspector-status | inspector-patrol |
        inspector-assign | inspector-on | inspector-off   # 闲时巡检引擎生命周期（§7）

长任务纪律：dispatch / plan run 应以 Bash 工具的 run_in_background 启动（README §4.5
等待模型）；orch 自身同步阻塞等待组员进程属预期（进程级等待，零 token）。
"""

from __future__ import annotations

import argparse
import sys

from . import benchd as _benchd
from . import channels as _channels
from . import drain as _drain
from . import eval as _eval
from . import inspectd as _inspectd
from . import jury as _jury
from . import mirror as _mirror
from . import plans as _plans
from . import presets as _presets
from . import sessions as _sessions
from . import sync as _sync
from . import taste as _taste
from . import verdict as _verdict
from . import watch as _watch
from . import workflow as _workflow
from .dispatch import do_dispatch
from .ledger import (
    LEDGER_PATH,
    format_by_effort,
    format_credits,
    format_est_tokens,
    read_all,
    summarize,
)
from .registry import (
    Registry,
    dirty_tracked_state,
    orchestration_rel,
    refresh_models,
)
from .runner import EFFORT_ALIASES, VALID_EFFORTS

#: ``--effort`` 可选值（原生 exe 档位 + off/disabled 别名）。**不传=跟随用户级默认（中）**；
#: 层级覆盖：旗标 > 计划环节/条目 effort > registry 成员 effort（README §3.5）。
EFFORT_CHOICES: list[str] = [*VALID_EFFORTS, *EFFORT_ALIASES]


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

    for line in _presets.status_lines(Registry.load()):
        print(line)

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

    # Tier A 阈值扫描（REFACTOR §2.4）在读取待办清单**之前**跑：本次新立的案要立刻
    # 出现在下面的 [建议] 列表里，否则「机械重算」与「呈现」之间差一拍，组长永远看的是
    # 上一轮的状态（status 是组长最常敲的命令，正是 §0 设计纪律 1 要的硬依赖入口）。
    for sid in _workflow.tier_a_sweep():
        print(f"[!] Tier A 阈值立案（自动）：{sid} —— orch suggestions 待审批")
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
    # 用户经 pysci-mirror 口述的**行动指令**：与偏差报告「纠正现有结果」同类，
    # 必须写任务书派 devops、不得 override（REFACTOR §5.4/§10）。不入本视野就等于
    # 没有队列——组长侧可见性是这条强制链的唯一执法点。
    pend = _mirror.pending_directives()
    if pend:
        print(f"  [!] mirror 行动指令待派发={len(pend)}（强制，不得 override）")
        for r in pend[:5]:
            print(f"    [指令] {r['ts']} {str(r.get('text', ''))[:60]}")
    for p in sugg:
        print(f"  [建议] {p.stem}")
    for b in back[:5]:
        eff = f" [effort={b['effort']}]" if b.get("effort") else ""
        print(f"  [backlog] {b['id']}: {str(b.get('summary', ''))[:60]}{eff}")

    print("== devops worker ==")
    lst = _drain.lock_status()
    if lst["state"] == "running":
        print(
            f"  ▶ running（pid {lst['pid']}，启动 {lst['started']}，"
            f"日志 {lst['run_log']}）"
        )
    elif lst["state"] == "stale":
        if lst.get("released"):
            print(
                f"  ⊘ 残留已标记（pid {lst['pid']} 的 run 已宣告收尾于 "
                f"{lst.get('released_at', '?')}，但锁文件删不掉〔Windows 共享句柄〕——"
                "released 墓碑不作互斥，下次 drain 就地接管，不占队列）"
            )
        elif lst.get("residue") == "done-exists":
            print(
                f"  ⊘ 残骸已收尾（该 run 已写出 .done，锁自称在跑纯属删不掉——连 released "
                f"墓碑都写不进的极端情形；pid {lst['pid']} 不做判活依据）——"
                "下次 approve/drain 就地接管，不占队列"
            )
        elif lst.get("crashed"):
            print(
                f"  ⊘ 崩溃（pid {lst['pid']} 已死且该 run 无 .done）——worker 被中途杀死；"
                "下次 approve/drain 自动接管并回收孤儿条目 + 清理残留 worktree"
            )
        else:
            print(
                f"  ⊘ stale 锁（pid {lst['pid']} 已死/超龄，无崩溃证据；"
                "该 run 的 .done 若声明了别的 run 则不按「已收尾」认定——"
                "下次 approve 或 drain 自动接管）"
            )
    else:
        print("  idle（无 drain worker 运行）")
    cancel = _drain.cancel_status()
    if cancel:
        print(
            f"  [s] 停止信号待消费（目标 pid {cancel.get('pid', '?')}，"
            f"请求于 {cancel.get('requested_at', '?')}）"
        )
    done = _drain.latest_done()
    if done:
        stopped = done.get("stopped")
        how = (
            "协作停止（drain-stop）"
            if stopped == "cooperative"
            # 两类系统性故障分列读数（backlog 20261012-194500-leader 要求 3）：处置动作
            # 不同——额度要充值/换渠道，渠道故障等恢复即可，混成一行就丢了这个信息
            else "额度耗尽全局停止（剩余条目保持 pending，待充值或换渠道）"
            if stopped == "quota_exhausted"
            else "渠道级故障全局停止（剩余条目保持 pending，待渠道恢复）"
            if stopped == "channel_error"
            else "异常中断"
            if done.get("error")
            else "自然跑完"
        )
        print(
            f"  最近 run：{done.get('name')}——处理 {done.get('count', 0)} 项，"
            f"耗时 {done.get('elapsed_s', '?')}s，停止方式={how}"
        )
        for it in done.get("items", [])[-5:]:
            extra = f" commit={it['commit']}" if it.get("commit") else ""
            print(f"    - {it.get('id')}: {it.get('status')}{extra}")
    needs_leader = _workflow.backlog_by_status("needs_leader")
    repeat_fails: dict[str, int] = {}
    if needs_leader:
        print(
            f"  [!] needs_leader 条目 {len(needs_leader)}（drain 跳过，待组长处理）："
        )
        repeat_fails = {
            k: v for k, v in _drain.failure_counts().items() if v >= 2
        }  # 阈值 2：一次失败是常态，两次才说明「原样重派」大概率再失败
        for b in needs_leader[:5]:
            n = repeat_fails.get(b["id"], 0)
            print(
                f"    - {b['id']}: {str(b.get('summary', ''))[:50]}"
                + (f"（累计失败 {n} 次，勿原样重派）" if n else "")
            )

    print("== 判决包健康（REFACTOR §5.1/§5.6：基建判断的通道用量）==")
    for line in _verdict.health_lines():
        print(line)

    bst = _benchd.bench_status()
    print("== bencher worker（harness 专测队列，REFACTOR §8）==")
    block = bst["lock"]
    if block["state"] == "running":
        print(f"  ▶ running（pid {block['pid']}，启动 {block['started']}）")
    elif block["state"] == "stale":
        print(
            f"  ⊘ stale 锁（pid {block['pid']}）——下次 bench-up 自动接管并回收孤儿条目"
        )
    else:
        print("  idle（worker 非常驻：队列非空才派生，消化完即自毁）")
    if bst["cancel"]:
        print(
            f"  [s] 停止信号待消费（请求于 {bst['cancel'].get('requested_at', '?')}）"
        )
    print(f"  队列计数：{bst['counts'] or '（空）'}")
    review = bst["review"]
    if review:
        print(
            f"  [待查看结果] {len(review)} 项（择机查看，不即时打断；"
            "查详情 `pysci-orch bench-queue`，看完 `bench-seen <id>`）"
        )
        for it in review[:5]:
            res = it.get("result") or {}
            line = res.get("headline") or str(res.get("error") or "")
            print(
                f"    - {it['id']}  [{it.get('status')}·{it.get('kind')}] "
                f"{str(line)[:70]}"
            )

    print("== inspector 巡检引擎（闲时自主巡逻，REFACTOR §7）==")
    for line in _inspectd.status_lines():
        print(line)
    # 内嵌拉起（§7「生命周期由代码保证」）：status 是组长最常敲的命令，把「记得去开
    # 巡检」这件事从人身上摘掉。放在读数之后——本轮显示的是拉起前的锁态，消息紧随其后，
    # 不会让人看到「idle」却不知道它刚被拉起。
    _inspectd.ensure_inspector(quiet=False)

    dirty = dirty_tracked_state()
    if dirty:
        print("== 跟踪 state 脏面（组员合并前检查拦的就是这些）==")
        for p in dirty:
            print(f"  [!] {p}")
        print(
            "  → 组员改 registry/backlog 的单会被合并前检查挡住：先提交或还原；"
            "若是运行时自愈改脏了跟踪面，那是缺陷（registry 只应留人写的覆盖）。"
        )

    print("== watch（drain 事件等待器）==")
    for line in _watch.status_lines():
        print(line)

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
        print(
            "  有待审批建议：uv run --no-sync pysci-orch suggestions 查看 → approve/reject"
        )
    if needs_leader:
        print(
            "  有 needs_leader 条目（drain 跳过）：查 state/backlog.json 后决定"
            "重派 / 改任务书 / 上报用户"
        )
    if repeat_fails:
        print(
            f"  [!] 重复失败 {len(repeat_fails)} 条（跨 run 累计 ≥2 次）："
            "上下文/轮数/任务书粒度失配的症状，改单再派，别原样重投"
        )
    if dirty:
        print(
            "  跟踪 state 有未提交改动（见上「脏面」段）：派发碰同文件的单前"
            "先提交或还原，否则组员只能交 <blocked>"
        )
    if live:
        print(
            '  推进进行中计划：uv run --no-sync pysci-orch plan run <id> --text "<当前环节任务书>"'
        )
    else:
        print(
            "  新工作：uv run --no-sync pysci-orch plan new <计划.yaml>（格式见 README §4.2）；"
        )
        print(
            '  计划外单步：uv run --no-sync pysci-orch plan adhoc <member> --text "..."'
        )
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
        text = _workflow.backlog_task_text(item)
        if not args.name:
            args.name = item["id"]
        # 条目自带 effort 标注（审批入队时标的）→ 作为本跳覆盖；显式 --effort 优先
        if not args.effort:
            args.effort = item.get("effort") or None
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
        effort=args.effort,
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
    if took_backlog_id:
        # 代码在「工作已交办」的当刻注入挂载指令，不靠组长记得去读文档（§4.5/§6）
        for line in _watch.mount_lines(indent="  ", header=False):
            print(line)
    # 巡检内嵌拉起之二（§7）：派完一跳，组长的手正空着读输出——inspector 的活路就在这
    # 个间隙里。让路/迟滞闸门由引擎自己判，这里只负责「想起来要试」。
    _inspectd.ensure_inspector()
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
        reg.write_sessions(m)
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

            model = reg.models.get(_presets.resolve_role(m.member_id), "")
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
        reg.write_sessions(m)
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
            f"{format_est_tokens(s)}；{format_credits(s)}"
        )
        for mid, mrow in sorted(s["by_member"].items()):
            print(
                f"  {mid:<10} 跳数={mrow['hops']:<4} 耗时占比={mrow['duration_pct']}%  "
                f"est占比={mrow['est_tokens_pct']}%  credits占比={mrow['credits_pct']}%"
            )
        if len(s["by_effort"]) > 1:
            # 单一档时 by_effort ≡ 全体，无 A/B 信号；>=2 档才打（backlog 20261010-153041-devops）
            print(
                "  按推理强度档位（轮均=num_turns 均值，返工率=kind≠result 跳占比）："
            )
            print(format_by_effort(s["by_effort"]))
        return 0
    if not entries:
        print(f"（台账为空：{orchestration_rel(LEDGER_PATH)}）")
        return 0
    for e in entries[-args.tail :]:
        checks_s = ",".join(f"{c['check']}:{c['verdict']}" for c in e.checks) or "-"
        est_s = f"est={e.est_tokens:,}" if e.est_tokens else "est=—"
        print(
            f"{e.ts}  {e.member:<9} {e.kind:<11} sid={e.session_id[:8]} "
            f"{e.num_turns}轮 {e.duration_ms / 1000:.0f}s ctx={e.ctx_ratio:.0%} "
            f"{est_s} "
            f"credits={e.credits:g}{f'[{e.model}]' if e.model else ''}"
            f"{f'[effort={e.effort}]' if e.effort else ''} "
            f"验收[{checks_s}]"
            + (
                " 建议✓"
                if e.infra_suggestion
                else " 建议✗未入库"
                if e.infra_kind
                else ""
            )
            + (f" 标签[{','.join(e.tags)}]" if e.tags else "")
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
        print(
            "[NEXT] 存在漂移：uv run --no-sync pysci-orch sync 重新部署（真本→副本单向）。"
        )
        return 2
    return 0


def cmd_drain_stop(args: argparse.Namespace) -> int:
    """请求后台 drain worker 协作式停止（写 state/devops.cancel）。

    分离进程无规范 kill 手段；本命令写停止信号，worker 在下一条目间隙干净退出。
    锁空闲时不写信号（避免残留信号危及下一个 worker），仅作提示。
    """
    del args
    ok, msg = _drain.request_stop()
    print(("[√] " if ok else "[i] ") + msg)
    return 0


def cmd_refresh_models(args: argparse.Namespace) -> int:
    """手动刷新模型映射的**运行态面**（--list-models 人类名→UUID；用户裁决 2026-10-10）。

    drain 启动时会自动执行同一刷新（自愈）；本命令供重配 BYOK 后立即校验/写回。解析值
    落 sessions-runtime.json（gitignored），跟踪 registry.json 的 models 是人写覆盖、
    读取时优先（三级回退，backlog 20261012-080410-devops）。未命中的档保持现值不清空
    （宁旧勿空——清空会静默改变计费渠道）。
    """
    del args
    try:
        res = refresh_models()
    except Exception as exc:
        print(f"[✗] 刷新失败：{type(exc).__name__}: {exc}")
        print(
            "[NEXT] 检查 qoderclicn exe（registry.exe / PYSCI_ORCH_EXE / 默认位置）与"
            "账户登录态；registry 现映射未动。"
        )
        return 2
    if res["changed"]:
        for tier, c in sorted(res["changed"].items()):
            print(f"  [{tier}] {c['old'] or '(空)'} → {_channels.chain_text(c['new'])}")
        print(
            f"[√] 已刷新 {len(res['changed'])} 个档位链"
            "（写 sessions-runtime.json 的 models_resolved，跟踪 registry.json 不动）。"
        )
    else:
        print(f"[√] 运行态面无变化（目录 {res['catalog_size']} 个模型），无需写回。")
    if res.get("shadowed"):
        for tier, s in sorted(res["shadowed"].items()):
            print(
                f"  [{tier}] 跟踪链 {_channels.chain_text(s['tracked'])} 遮蔽解析链 "
                f"{_channels.chain_text(s['resolved'])}（人写覆盖优先，未改）"
            )
        print(
            "    要换档：改 registry.json 的 models.<档>（整条链，护栏级经用户授权），"
            "或删掉该键让机器接管。"
        )
    if res.get("conflicts"):
        print("[!] 链模板与名→档规则冲突（串档，须人裁决）：")
        for item in res["conflicts"]:
            print(f"    {item}")
    if res["missing"]:
        print(f"[!] 未命中（该席跳过、该档保持现值）：{', '.join(res['missing'])}")
        print(
            "    模型改名 → 更新 registry.model_chains（或 registry.py 的默认链模板）。"
        )
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
    q.add_argument(
        "--model-tier",
        type=_presets.normalize_role,
        choices=list(_presets.ROLE_CHOICES),
    )
    q.add_argument(
        "--effort",
        choices=EFFORT_CHOICES,
        help="本环节推理强度（覆盖计划 YAML 的 steps[].effort；不传=跟随用户级默认=中）",
    )
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
    q.add_argument(
        "--model-tier",
        type=_presets.normalize_role,
        choices=list(_presets.ROLE_CHOICES),
    )
    q.add_argument(
        "--effort",
        choices=EFFORT_CHOICES,
        help="本跳推理强度按项标注（不传=跟随用户级默认=中）",
    )
    q.add_argument("--max-turns", type=int)
    q.set_defaults(func=_plan_router("adhoc"))

    # ---- 改进循环 ----
    p = sub.add_parser("suggestions", help="列出待审批改进建议")
    p.set_defaults(func=_workflow.cmd_suggestions)

    p = sub.add_parser("approve", help="采纳建议 → backlog + 审批回复附送")
    p.add_argument("suggestion_id")
    p.add_argument("--note", default="")
    p.add_argument(
        "--no-wake",
        action="store_true",
        help="入队后不自动唤醒 devops 后台消化（默认唤醒；锁忙时本就不重复唤醒）",
    )
    p.add_argument(
        "--effort",
        choices=EFFORT_CHOICES,
        help="按项标注该待办的推理强度（写入 backlog 条目，drain 消化时透传；"
        "不标=跟随用户级默认=中）",
    )
    p.add_argument(
        "--criteria",
        default="",
        help="补录成功判据（E0-B 判据 3：backlog 条目须带可证伪 success_criteria；"
        "建议正文缺 `success_criteria` 标记行时用此参数补，仍缺则拒绝入队）",
    )
    p.add_argument(
        "--jury",
        action="store_true",
        help="判决通道 J：自动组卷并阻塞仲裁（复用 orch jury 共享库）。裁决 A=入队、"
        "B=自动驳回（不计 override 率）、模棱两可=挂起呈组长",
    )
    p.add_argument(
        "--bench",
        default="",
        help="判决通道 M：捕获基线 commit 并排基线测量，合并销账时自动入 A/B 核验队列"
        "（§5.6 核验方式在审批时一次定死）。取案 id 列表，逗号分隔；空=全套",
    )
    p.add_argument(
        "--override",
        default="",
        help="判决通道 override：不选 J/M，凭组长自身判断定案（理由必填，"
        "留痕入决策日志并计入 override 率；不可与 --jury/--bench 同时给）",
    )
    p.set_defaults(func=_workflow.cmd_approve)

    p = sub.add_parser("reject", help="否决建议 → 须 --override 留痕（入决策日志）")
    p.add_argument("suggestion_id")
    p.add_argument("--note", default="", help="附送给成员的补充说明（可选）")
    p.add_argument(
        "--override",
        default="",
        help="否决判决依据（必填）：REFACTOR §5.1/E0 一切 reject 即 override，"
        "缺失则拒绝执行并非零退出；提供则写回复、移入 rejected/ 并追加 journal 行",
    )
    p.set_defaults(func=_workflow.cmd_reject)

    p = sub.add_parser("consult", help="咨询副组长（平级协作：异议义务提醒自动注入）")
    p.add_argument("question")
    p.add_argument("--session", help="latest(默认)|new|<sid前缀>")
    p.set_defaults(func=_workflow.cmd_consult)

    p = sub.add_parser(
        "consult-twin",
        help="咨询 twin 组员（高风险裁决第二意见；taste_enabled=false 时明确拒绝并提示待上线）",
    )
    p.add_argument("case_file", help="案卷路径（## 问题 / ## 选项 A / ## 选项 B）")
    p.add_argument("--session", help="latest(默认)|new|<sid前缀>")
    p.set_defaults(func=_verdict.cmd_consult_twin)

    p = sub.add_parser(
        "review", help="审查产物：VERDICT → FAIL 自动回派返工 → 复审一次 → 升级仲裁"
    )
    p.add_argument("artifact", help="产物路径（文件或目录）")
    p.add_argument(
        "--origin", required=True, help="生产该产物的成员 id（FAIL 回派对象）"
    )
    p.add_argument(
        "--rubric", default="generic", help="rubric 名（reviewer pod rubrics/ 下）"
    )
    p.add_argument(
        "--goal-from",
        dest="goal_from",
        metavar="<原生产任务书>",
        help="从原生产任务书摘录目标陈述嵌入审查任务书，供 G2 独立核验（勿依赖生产者 notes.md）",
    )
    p.add_argument(
        "--goal",
        metavar="<文本>",
        help="直接给定目标陈述文本（无单一任务书文件时用；优先于 --goal-from）",
    )
    p.set_defaults(func=_workflow.cmd_review)

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
    p.add_argument(
        "--model-tier",
        type=_presets.normalize_role,
        choices=list(_presets.ROLE_CHOICES),
    )
    p.add_argument(
        "--effort",
        choices=EFFORT_CHOICES,
        help="本跳推理强度（覆盖成员级 effort；--from-backlog 时默认取条目 effort）",
    )
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

    p = sub.add_parser(
        "drain-stop", help="请求后台 drain worker 协作式停止（写 devops.cancel）"
    )
    p.set_defaults(func=cmd_drain_stop)

    p = sub.add_parser(
        "refresh-models",
        help="按 --list-models 刷新 registry 模型映射（drain 启动亦自动执行）",
    )
    p.set_defaults(func=cmd_refresh_models)

    p = sub.add_parser(
        "watch",
        help="阻塞等待 drain 事件（CRASH/QUOTA-STOP/CHANNEL-STOP/NEEDS-LEADER/"
        "QUEUE-COMPLETE）；"
        "组长以 Bash run_in_background 启动，退出即唤醒本会话",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=_watch.DEFAULT_TIMEOUT_S,
        help=f"总等待秒数（默认 {_watch.DEFAULT_TIMEOUT_S}=4h；超时打印 TIMEOUT 退出）",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=_watch.DEFAULT_INTERVAL_S,
        help=f"轮询间隔秒数（钳制到 {_watch.MIN_INTERVAL_S}–{_watch.MAX_INTERVAL_S}，"
        f"默认 {_watch.DEFAULT_INTERVAL_S}）",
    )
    p.set_defaults(func=_watch.cmd_watch)

    p = sub.add_parser(
        "jury",
        help="Tier C 仲裁团（REFACTOR §4）：案卷送家族外评审团双序对局，阻塞出裁决",
    )
    p.add_argument(
        "case_file", help="案卷文件（## 问题 / ## 选项 A / ## 选项 B [+ ## 证据]）"
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="只组装并打印案卷（含自动附着材料与哈希），不调 API",
    )
    p.add_argument(
        "--timeout",
        type=int,
        help="整体等待上限秒数（覆盖 registry.jury.deadline_s，默认 300=5 min）",
    )
    p.set_defaults(func=_jury.cmd_jury)

    # ---- taste：twin 审美判断通道（REFACTOR §9 / E6-prep）----
    _taste._add_parser(sub)

    # ---- eval：Tier B harness 压测套件（REFACTOR §3 / E2）----
    p = sub.add_parser(
        "eval",
        help="Tier B harness 压测套件：最小成本档跑案卷 → 二元判决 + 基线对照",
    )
    ep = p.add_subparsers(dest="eval_action", required=True)
    pe = ep.add_parser(
        "run", help="跑一案/全套 → 通过|失败 + 指标行 + before/after 对照"
    )
    pg = pe.add_mutually_exclusive_group(required=True)
    pg.add_argument("--case", help="案卷 id（state/evals/harness/<id>.md 去后缀）")
    pg.add_argument("--all", action="store_true", help="按文件名序跑全部案卷")
    pe.add_argument(
        "--model-tier",
        type=_presets.normalize_role,
        choices=list(_presets.ROLE_CHOICES),
        help="覆盖案内档位（默认案卷的 tier:，缺省 minimum=压测最低成本档）",
    )
    pe.add_argument(
        "--label",
        help="本轮标签（如 before / after-<commit>）；入台账供 set-baseline 与分组对照",
    )
    pe.set_defaults(func=_eval.cmd_eval_run)
    pb = ep.add_parser("set-baseline", help="把最近一轮压测行回填为 baseline.json")
    pb.add_argument("--label", help="只取该标签的最近一轮（默认：每案最近一行）")
    pb.add_argument(
        "--runner",
        choices=["leader", "bencher"],
        help="只取该发起者的行（默认混合；E4 后专测行属 bencher）",
    )
    pb.set_defaults(func=_eval.cmd_eval_set_baseline)

    # ---- bench-*：bencher 专职测量员队列（REFACTOR §8 / E4）----
    p = sub.add_parser(
        "bench-add",
        help="向 bencher 队列注入一个测量任务（run/capture/ab/check）；--start 顺手拉起 worker",
    )
    p.add_argument("kind", choices=list(_benchd.ITEM_KINDS))
    p.add_argument("--cases", help="案卷 id，逗号分隔（run/capture/check 用）")
    p.add_argument("--label", help="本轮标签（进台账供对照/基线回填）")
    p.add_argument(
        "--tier",
        type=_presets.normalize_role,
        choices=list(_presets.ROLE_CHOICES),
        help="覆盖案内档位（缺省=案内 tier，再缺省 minimum）",
    )
    p.add_argument("--ref", help="capture：被测 ref（默认 HEAD）")
    p.add_argument("--ref-a", help="ab：基线侧 ref/commit")
    p.add_argument("--ref-b", help="ab：实施侧 ref/commit")
    p.add_argument("--brief", help="check：核查说明（派发正文）")
    p.add_argument("--requested-by", help="提请人（默认 leader）")
    p.add_argument("--summary", help="队列里展示的一句话（缺省按 kind 生成）")
    p.add_argument(
        "--start", action="store_true", help="入队后立即拉起 worker（默认等 bench-up）"
    )
    p.set_defaults(func=_benchd.cmd_bench_add)

    p = sub.add_parser(
        "bench-up",
        help="拉起 bencher worker 后台消化队列（非空即消化，空即自毁）",
    )
    p.add_argument(
        "--dry", action="store_true", help="干跑：不真派发，仅模拟销账（演练/测试）"
    )
    p.set_defaults(func=_benchd.cmd_bench_up)

    p = sub.add_parser("bench-queue", help="列 bencher 队列（默认只看在途与待查看）")
    p.add_argument("--all", action="store_true", help="含已查看的历史条目")
    p.set_defaults(func=_benchd.cmd_bench_queue)

    p = sub.add_parser(
        "bench-seen", help="标记某条目已查看（从 orch status 待查看面摘掉）"
    )
    p.add_argument("id", help="队列条目 id（bench-queue 可查）")
    p.set_defaults(func=_benchd.cmd_bench_seen)

    p = sub.add_parser(
        "bench-stop", help="请求 bencher worker 协作停止（条目间隙干净退出）"
    )
    p.set_defaults(func=_benchd.cmd_bench_stop)

    # ---- inspector-*：闲时巡检引擎的生命周期面（REFACTOR §7 / E5a）----
    # 组长侧只有「拉起/开关/读数/指派」四件事：巡检内容本身在 inspector pod 的会话里，
    # 发现的分流在 pysci-inspect（组员侧）。资源闸门（预算/让路/迟滞）没有 CLI 旋钮——
    # §7 要的是「代码保证」，可被随手关掉的成本控制等于没有。
    p = sub.add_parser(
        "inspector-up",
        help="幂等拉起 inspector 巡检 worker（开关关/在跑/冷却窗内均 no-op 并说明）",
    )
    p.add_argument(
        "--dry", action="store_true", help="干跑：派跳只演练不真派发（演练/测试）"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="跳过冷却窗（一次性自主巡检；不越开关与锁）",
    )
    p.set_defaults(func=_inspectd.cmd_inspector_up)

    p = sub.add_parser("inspector-status", help="巡检引擎面细读（闸门/阈值/队列/发现）")
    p.set_defaults(func=_inspectd.cmd_inspector_status)

    p = sub.add_parser(
        "inspector-patrol",
        help="手动排入一项巡检（--target 菜单 id，缺省随机抽；queue-first 的插队口）",
    )
    p.add_argument("--target", default="", help="巡检菜单项 id（空=随机抽取）")
    p.add_argument("--summary", default="", help="队列里展示的一句话")
    p.add_argument("--start", action="store_true", help="入队后立即拉起 worker")
    p.set_defaults(func=_inspectd.cmd_inspector_patrol)

    p = sub.add_parser(
        "inspector-assign",
        help="给 inspector 排一项指派核查（kind=assignment，正文走巡检队列）",
    )
    p.add_argument("--brief", required=True, help="指派任务书正文")
    p.add_argument("--summary", default="", help="队列里展示的一句话")
    p.add_argument("--start", action="store_true", help="入队后立即拉起 worker")
    p.set_defaults(func=_inspectd.cmd_inspector_assign)

    p = sub.add_parser("inspector-on", help="打开巡检全局开关（默认态）")
    p.set_defaults(func=_inspectd.cmd_inspector_on)

    p = sub.add_parser(
        "inspector-off",
        help="关闭巡检全局开关（成本控制；在跑的 worker 条目间隙自停）",
    )
    p.set_defaults(func=_inspectd.cmd_inspector_off)

    # ---- 内部命令（approve 自动唤醒派生；下划线前缀=内部，勿手调）----
    p = sub.add_parser(
        "_drain-devops",
        help="[内部] approve 自动唤醒派生的 backlog drain 循环（下划线前缀=内部命令）",
    )
    p.add_argument("--run-log", help="run 日志路径（唤醒时传入；默认自动生成）")
    p.add_argument(
        "--dry", action="store_true", help="干跑：不真派发，仅模拟销账（演练/测试）"
    )
    p.set_defaults(func=_drain.cmd_drain_devops)

    p = sub.add_parser(
        "_bench-worker",
        help="[内部] bencher 队列 drain 循环（下划线前缀=内部命令，勿手调）",
    )
    p.add_argument("--run-log", help="run 日志路径（唤醒时传入；默认自动生成）")
    p.add_argument(
        "--dry", action="store_true", help="干跑：不真派发，仅模拟销账（演练/测试）"
    )
    p.set_defaults(func=_benchd.cmd_bench_worker)

    p = sub.add_parser(
        "_inspect-worker",
        help="[内部] inspector 巡检激活循环（下划线前缀=内部命令，勿手调）",
    )
    p.add_argument("--run-log", help="run 日志路径（拉起时传入；默认自动生成）")
    p.add_argument(
        "--dry", action="store_true", help="干跑：派跳只演练，不真派发组员会话"
    )
    p.set_defaults(func=_inspectd.cmd_inspect_worker)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
