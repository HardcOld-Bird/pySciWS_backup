"""改进循环与统计命令：approve/reject/suggestions/consult/stats（README §4.4/§6）。

审批链：dispatch 把 infra_suggestion 持久化到 ``state/suggestions/<id>.md``（pending）
→ 组长 approve（入 backlog + 写审批回复）/ reject（写回复）→ 回复在下次派发该成员时
自动附送（闭环告知）。护栏级建议不得自行 approve（须转呈用户）——由组长规程软性约束。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT

from .dispatch import REPLIES_DIR, do_dispatch
from .ledger import read_all, summarize

SUGGESTIONS_DIR = ORCH_STATE_ROOT / "suggestions"
BACKLOG_PATH = ORCH_STATE_ROOT / "backlog.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pending_suggestions() -> list[Path]:
    """待审批建议文件列表（suggestions/ 顶层 = pending；子目录为已处理归档）。"""
    if not SUGGESTIONS_DIR.exists():
        return []
    return sorted(SUGGESTIONS_DIR.glob("*.md"))


def _find_suggestion(id_prefix: str) -> Path | None:
    hits = [p for p in pending_suggestions() if p.stem.startswith(id_prefix)]
    return hits[0] if len(hits) == 1 else None


def _member_of(sugg: Path) -> str:
    # 文件名约定：<ts>-<member>.md
    parts = sugg.stem.split("-", 2)
    return parts[2] if len(parts) >= 3 else "unknown"


def _backlog_append(entry: dict) -> int:
    data = {"version": 1, "items": []}
    if BACKLOG_PATH.exists():
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    items = data.setdefault("items", [])
    items.append(entry)
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(items)


def backlog_pending() -> list[dict]:
    """backlog 中 status=pending 的条目（FIFO 顺序）。"""
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return [i for i in data.get("items", []) if i.get("status") == "pending"]


def backlog_by_status(status: str) -> list[dict]:
    """backlog 中指定 status 的条目（FIFO 顺序）。"""
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return [i for i in data.get("items", []) if i.get("status") == status]


def backlog_take_first() -> dict | None:
    """取队首 pending 条目标记 in_progress 并返回（devops 派发用）。"""
    if not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("status") == "pending":
            item["status"] = "in_progress"
            item["started_at"] = _now()
            BACKLOG_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return item
    return None


def _backlog_set_status(
    item_id: str, status: str, note: str = "", *, stamp: str = ""
) -> None:
    """把 backlog 条目置为指定 status（可选写时间戳字段与附注）。"""
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id:
            item["status"] = status
            if stamp:
                item[stamp] = _now()
            if note:
                item["note"] = note
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def backlog_complete(item_id: str, note: str = "") -> None:
    """标记 backlog 条目完成（devops 交付后由组长/orch 调用）。"""
    _backlog_set_status(item_id, "done", note, stamp="done_at")


def backlog_needs_leader(item_id: str, note: str = "") -> None:
    """标记 backlog 条目需组长介入（drain 单项失败跳过时用；不再自动重试）。"""
    _backlog_set_status(item_id, "needs_leader", note, stamp="needs_leader_at")


def backlog_task_text(item: dict) -> str:
    """由 backlog 条目生成 devops 派发任务书正文。

    ``dispatch --from-backlog``（手动）与 ``_drain-devops``（自动）共用此模板，保证
    两条路径派发内容一致。末尾附**合并安全**提示：devops 合并 worktree 前须核对
    main 工作区，避免覆盖组长未提交改动（README §6 自动唤醒下的并发防护）。
    """
    return (
        f"基础设施改进任务（backlog FIFO 队首，id={item['id']}）：\n\n"
        f"摘要：{item.get('summary', '')}\n\n"
        f"证据：{item.get('evidence') or item.get('evidence_file', '（见建议归档）')}\n\n"
        f"组长附注：{item.get('note', '（无）')}\n\n"
        "要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、"
        "commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，"
        f"并注明 backlog id={item['id']} 以便销账。\n\n"
        "合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，"
        "交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。"
    )


def cmd_suggestions(args) -> int:
    """列出待审批改进建议。"""
    del args
    items = pending_suggestions()
    if not items:
        print("（无待审批建议）")
        return 0
    for p in items:
        body = p.read_text(encoding="utf-8")
        first = next((ln.strip() for ln in body.splitlines()[2:] if ln.strip()), "")
        print(f"  {p.stem}  [{_member_of(p)}]  {first[:70]}")
    print()
    print(
        '[NEXT] 审批：uv run pysci-orch approve <id> --note "..."；否决：reject 同构。'
    )
    return 0


def cmd_approve(args) -> int:
    """采纳建议：入 backlog FIFO + 写审批回复（下次派发自动附送）。"""
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'（orch suggestions 查看）")
        return 2
    member = _member_of(sugg)
    body = sugg.read_text(encoding="utf-8")
    first = next((ln.strip() for ln in body.splitlines()[2:] if ln.strip()), sugg.stem)
    n = _backlog_append(
        {
            "id": sugg.stem,
            "member": member,
            "summary": first[:200],
            "evidence_file": f"orchestration/state/suggestions/approved/{sugg.name}",
            "status": "pending",
            "proposed": _now(),
            "note": args.note or "",
        }
    )
    REPLIES_DIR.joinpath(member).mkdir(parents=True, exist_ok=True)
    (REPLIES_DIR / member / f"{sugg.stem}.md").write_text(
        f"# 组长审批：采纳（{_now()}）\n\n{args.note or '（无附注）'}\n\n"
        f"已入 backlog（第 {n} 位），devops 将按 FIFO 处理；完成后你会在后续任务中获知。\n",
        encoding="utf-8",
    )
    approved = SUGGESTIONS_DIR / "approved"
    approved.mkdir(exist_ok=True)
    sugg.rename(approved / sugg.name)
    print(
        f"[√] 已采纳 {sugg.stem} → backlog 第 {n} 位；审批回复将在下次派发 {member} 时附送。"
    )
    # 自动唤醒 devops 后台 FIFO 消化 backlog（README §6）；--no-wake 跳过。
    if not getattr(args, "no_wake", False):
        from .drain import (
            wake_devops,  # 延迟导入：workflow↔drain 环（drain 依赖 workflow）
        )

        _, msg = wake_devops()
        print(f"    {msg}")
    return 0


def cmd_reject(args) -> int:
    """否决建议：仅写回复（附理由），不入 backlog。"""
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'")
        return 2
    member = _member_of(sugg)
    REPLIES_DIR.joinpath(member).mkdir(parents=True, exist_ok=True)
    (REPLIES_DIR / member / f"{sugg.stem}.md").write_text(
        f"# 组长审批：暂不采纳（{_now()}）\n\n{args.note or '（未附理由——建议补充）'}\n",
        encoding="utf-8",
    )
    rejected = SUGGESTIONS_DIR / "rejected"
    rejected.mkdir(exist_ok=True)
    sugg.rename(rejected / sugg.name)
    print(f"[√] 已否决 {sugg.stem}；理由将在下次派发 {member} 时附送。")
    return 0


def cmd_consult(args) -> int:
    """快捷咨询副组长（平级协作条款：异议义务提醒注入任务文本）。"""
    text = (
        f"组长咨询（顾问模式，轻量回答即可）：\n\n{args.question}\n\n"
        "提醒（charter 平级协作条款）：你有责任补充不同视野——如认为组长的判断或前提"
        "有误，请直接指出并给出依据；鼓励以实测佐证。若你认为该咨询背后的任务派发本身"
        "不合理，你有权拒绝并说明理由。"
    )
    outcome = do_dispatch(
        "deputy",
        text=text,
        slug=f"consult-{datetime.now().strftime('%H%M%S')}",
        session=args.session or "latest",
    )
    if outcome.error == "pod 不存在":
        print("    （deputy pod 属 Phase 3；当前可先向对应专职组员或组长自行判断。）")
    return outcome.code


def cmd_stats(args) -> int:
    """三级统计：计划级（--plan）/ 成员级（--member/--days）/ 全体历史。"""
    entries = read_all()
    if args.plan:
        sub = [e for e in entries if e.plan == args.plan]
        title = f"计划 {args.plan}"
    elif args.member:
        sub = [e for e in entries if e.member == args.member]
        title = f"成员 {args.member}"
    else:
        sub = entries
        title = "全体历史"
    if args.days:
        cutoff = (datetime.now().astimezone() - timedelta(days=args.days)).isoformat()
        sub = [e for e in sub if e.ts >= cutoff]
        title += f"（近 {args.days} 天）"
    s = summarize(sub)
    if not s["hops"]:
        print(f"{title}：暂无台账数据")
        return 0
    print(f"== {title} ==")
    print(
        f"  跳数={s['hops']}  总耗时={s['duration_ms'] / 1000:.0f}s  credits={s['credits']}"
        f"（BYOK 模式下 credits 暂不上报，以耗时/轮数为准）"
    )
    for mid, row in sorted(s["by_member"].items()):
        print(
            f"  {mid:<10} 跳数={row['hops']:<4} 耗时占比={row['duration_pct']}%  "
            f"credits占比={row['credits_pct']}%"
        )
    kinds: dict[str, int] = {}
    for e in sub:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"  交付分布：{kinds}")
    return 0


# ---------------------------------------------------------------------------
# review（README §5.2：reviewer 派发 + VERDICT 解析 + FAIL 自动回派 + 复审一次）
# ---------------------------------------------------------------------------
import re  # noqa: E402

REVIEWS_DIR = ORCH_STATE_ROOT / "reviews"

_VERDICT_RE = re.compile(r"<verdict>\s*(PASS|FAIL)\s*</verdict>", re.IGNORECASE)


def parse_verdict(body: str) -> str | None:
    """从 reviewer 交付正文解析 VERDICT（PASS/FAIL/None）。"""
    m = _VERDICT_RE.search(body or "")
    return m.group(1).upper() if m else None


def _save_review(record: dict) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    p = REVIEWS_DIR / f"{record['id']}.json"
    p.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return p


def cmd_review(args) -> int:
    """审查产物：reviewer 出具 VERDICT；FAIL → 自动回派 origin 返工 → 复审一次；
    二次 FAIL → 升级组长仲裁（README §5.2）。长链路，建议后台 Bash 运行。"""
    from pysci.paths import PODS_ROOT

    review_id = f"rev-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rubric_rel = f"rubrics/{args.rubric}.md"
    rubric_abs = PODS_ROOT / "reviewer" / rubric_rel
    if not rubric_abs.exists():
        print(f"[!] rubric 不存在：{rubric_abs}")
        return 2
    record: dict = {
        "id": review_id,
        "artifact": args.artifact,
        "origin": args.origin,
        "rubric": args.rubric,
        "started": _now(),
        "rounds": [],
    }
    max_rounds = 2  # 初审 + 复审一次（用户裁决）
    for round_no in range(1, max_rounds + 1):
        task = (
            f"审查任务（第 {round_no} 轮，review id={review_id}）：\n\n"
            f"- 产物：{args.artifact}\n"
            f"- rubric（必读，按它逐项检查）：{rubric_abs}\n"
            f"- 生产组员：{args.origin}\n"
            + (
                "- 上一轮 FAIL 证据与返工说明见你的会话历史/任务书附件。\n"
                if round_no > 1
                else ""
            )
            + "\n要求：按 charter 的 VERDICT 交付格式出具 <result>，内含 "
            "<verdict>PASS|FAIL</verdict>、<scores>、<evidence>（FAIL 时附可执行返工指引）。"
        )
        outcome = do_dispatch(
            "reviewer", text=task, slug=f"{review_id}-r{round_no}", session="new"
        )
        verdict = parse_verdict(outcome.body) if outcome.kind == "result" else None
        record["rounds"].append(
            {
                "round": round_no,
                "reviewer_session": outcome.sid,
                "kind": outcome.kind,
                "verdict": verdict,
                "body_excerpt": outcome.body[:1200],
            }
        )
        if outcome.kind != "result" or verdict is None:
            _save_review(record)
            print(
                f"[?] 审查未完成（kind={outcome.kind}, verdict={verdict}）——见上方 reviewer 交付。"
            )
            print()
            print(
                "[NEXT] 人工判读 reviewer 输出：补派审查（dispatch reviewer --session "
                f"{outcome.sid[:8]}）或放弃本次审查。审查记录：{_rel(record)}"
            )
            return 2
        if verdict == "PASS":
            record["completed"] = _now()
            record["final"] = "PASS"
            p = _save_review(record)
            print(f"[√] 审查 PASS（第 {round_no} 轮）。记录：{p.name}")
            print()
            print("[NEXT] 该环节工作宣告完成；若有后续环节按计划推进，否则向用户交付。")
            return 0
        # FAIL
        print(f"[✗] 审查 FAIL（第 {round_no} 轮）。")
        if round_no >= max_rounds:
            record["completed"] = _now()
            record["final"] = "FAIL_ESCALATED"
            p = _save_review(record)
            print()
            print("[NEXT] 复审仍 FAIL → 组长仲裁（用户裁决的升级路径）：")
            print(
                '  1) 改派副组长攻坚：orch plan adhoc deputy --text "..."（附审查记录）'
            )
            print("  2) 修改方案/rubric 适用性后重审")
            print("  3) 呈报用户裁定。审查记录：" + p.name)
            return 2
        record_round = record["rounds"][-1]
        rework_text = (
            f"返工任务（审查 FAIL 自动回派，review id={review_id}）：\n\n"
            f"reviewer 对你的产物 {args.artifact} 出具 FAIL。审查证据与返工指引：\n\n"
            f"{outcome.body[:1500]}\n\n"
            "请针对证据逐条修正后重新交付（机械验收照旧声明）。"
        )
        rework = do_dispatch(
            args.origin, text=rework_text, slug=f"{review_id}-rework", session="latest"
        )
        record_round["rework_session"] = rework.sid
        record_round["rework_kind"] = rework.kind
        if rework.code != 0:
            _save_review(record)
            print(f"[!] 返工未成功（kind={rework.kind}）——中止复审，升级组长决策。")
            print()
            print(
                "[NEXT] 可选：补充信息再回派 / 咨询副组长 / 呈报用户（附审查记录 "
                f"{review_id}.json）"
            )
            return 2
        print("[√] 返工交付成功，进入复审…")
    return 2


def _rel(record: dict) -> str:
    return f"orchestration/state/reviews/{record['id']}.json"
