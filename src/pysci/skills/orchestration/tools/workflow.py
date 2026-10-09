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


def backlog_complete(item_id: str, note: str = "") -> None:
    """标记 backlog 条目完成（devops 交付后由组长/orch 调用）。"""
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id:
            item["status"] = "done"
            item["done_at"] = _now()
            if note:
                item["note"] = note
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
