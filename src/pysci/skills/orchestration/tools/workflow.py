"""改进循环与统计命令：approve/reject/suggestions/consult/stats（README §4.4/§6）。

审批链：dispatch 把 infra_suggestion 持久化到 ``state/suggestions/<id>.md``（pending）
→ 组长 approve（入 backlog + 写审批回复）/ reject（写回复）→ 回复在下次派发该成员时
自动附送（闭环告知）。护栏级建议不得自行 approve（须转呈用户）——由组长规程软性约束。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from .dispatch import REPLIES_DIR, do_dispatch
from .ledger import (
    format_by_effort,
    format_credits,
    format_est_tokens,
    read_all,
    summarize,
)
from .registry import orchestration_rel
from .runner import normalize_effort

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
    """取队首 pending 条目标记 in_progress 并返回（devops 派发用）。

    同时记入 ``worktree`` 字段（= 条目 id，与 dispatch slug 一致）：devops 按任务书
    约定用此名开 worktree，worker 中途死亡后 drain 孤儿回收据此定位并清理残留
    worktree/branch（见 :func:`drain.cleanup_orphan_worktrees`）。
    """
    if not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("status") == "pending":
            item["status"] = "in_progress"
            item["started_at"] = _now()
            item["worktree"] = str(item.get("id", ""))
            BACKLOG_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return item
    return None


def _effort_key(item: dict) -> str:
    """条目 effort 的**组包比较键**：规范档位名，空标注 → ""（=跟随用户级默认）。

    非法标注不抛（fail-soft）：原样作键，让它自己当种子派发时由 do_dispatch 硬失败并
    报出原因——组包阶段抛异常会拖垮整个 drain 循环。
    """
    try:
        return normalize_effort(item.get("effort")) or ""
    except ValueError:
        return str(item.get("effort") or "").strip().lower()


def backlog_take_batch(max_batch: int = 4) -> tuple[dict | None, list[dict]]:
    """取队首 pending 为**种子**（标记 in_progress），收集同 member（提请者）的 pending
    为**组包**（批量消化，backlog 20261010-batch-digest）。

    启发式：同提请者 ≈ 同模块/同视角（如 figure 的多项常同落 scientific_plotting），
    合并实施省重复上下文；leader 等较杂的组由 devops 的选取权兜底（可只吃种子）。

    Returns:
        ``(seed, group)``：无 pending 时 seed 为 None、group 空。group 为与种子同提请者
        的其余 pending 条目，**保持 pending**（不标 in_progress）——唯种子是本轮承诺单元，
        组包中未被 devops 选取的项留待下轮，worker 死亡也只种子成孤儿。
        ``len(group) <= max_batch - 1``（护栏：单批 ≤ max_batch 项，含种子）。
        member 为空的条目不组包（按单项处理），兼容旧数据与手动入队条目。
        effort 标注不同的条目也不组包（一批只有一跳，跳级参数不能一档多标）。
    """
    if not BACKLOG_PATH.exists():
        return None, []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    seed = next((it for it in items if it.get("status") == "pending"), None)
    if seed is None:
        return None, []
    seed["status"] = "in_progress"
    seed["started_at"] = _now()
    seed["worktree"] = str(seed.get("id", ""))
    member = seed.get("member")
    seed_effort = _effort_key(seed)
    group: list[dict] = []
    if member and max_batch > 1:
        for it in items:
            if it is seed or it.get("status") != "pending":
                continue
            # 组包只吃**同档位**：一批只有一跳，effort 是跳级参数——混档会让被标的项
            # 按种子档位跑完，静默污染台账 A/B 数据（backlog 20261010-orch-effort-per-item）
            if it.get("member") == member and _effort_key(it) == seed_effort:
                group.append(it)
                if len(group) >= max_batch - 1:
                    break
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return seed, group


def backlog_reclaim_orphans() -> list[str]:
    """把所有 in_progress 条目重置为 pending（孤儿回收），返回回收的 id 列表。

    in_progress 仅由 :func:`backlog_take_first` 标记，而 take_first 只在持锁的 drain
    循环内调用。因此**新 drain 成功获锁之时**，任何 in_progress 条目都必然是上一个
    worker 在 take_first 与 backlog_complete 之间被杀留下的孤儿（锁已判死/超龄才被
    接管）——不回收则永久卡死（take_first 只找 pending）。
    """
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    reclaimed = []
    for item in data.get("items", []):
        if item.get("status") == "in_progress":
            item["status"] = "pending"
            item["note"] = (
                f"孤儿条目回收（原 started_at={item.pop('started_at', '?')}）"
            )
            item.pop(
                "worktree", None
            )  # 复位为干净 pending（残留 worktree 由 drain 清理）
            reclaimed.append(str(item.get("id", "")))
    if reclaimed:
        BACKLOG_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return reclaimed


def _backlog_set_status(
    item_id: str,
    status: str,
    note: str = "",
    *,
    stamp: str = "",
    extra: dict | None = None,
) -> None:
    """把 backlog 条目置为指定 status（可选写时间戳字段、附注与额外字段）。"""
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
            if extra:
                item.update(extra)
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def backlog_complete(
    item_id: str,
    note: str = "",
    *,
    batch_size: int | None = None,
    batch_seed: str | None = None,
) -> None:
    """标记 backlog 条目完成（devops 交付后由组长/orch 调用）。

    batch_size/batch_seed：批量消化时记录本条目所属批次（同批一起 done 的条目数与
    种子 id），供 orch stats 批量大小分布与审计；单条消化时留空。
    """
    extra: dict = {}
    if batch_size is not None:
        extra["batch_size"] = batch_size
    if batch_seed is not None:
        extra["batch_seed"] = batch_seed
    _backlog_set_status(item_id, "done", note, stamp="done_at", extra=extra or None)


def backlog_needs_leader(item_id: str, note: str = "") -> None:
    """标记 backlog 条目需组长介入（drain 单项失败跳过时用；不再自动重试）。"""
    _backlog_set_status(item_id, "needs_leader", note, stamp="needs_leader_at")


def backlog_requeue(item_id: str, note: str = "") -> None:
    """把 in_progress 条目复位为 pending（清除 started_at）。

    用于**系统性故障**（如额度耗尽）导致的全局停止：条目本身没问题，应回队等待
    渠道恢复后续消化，而非标记 needs_leader 污染 backlog 语义。
    """
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id and item.get("status") == "in_progress":
            item["status"] = "pending"
            item.pop("started_at", None)
            if note:
                item["note"] = note
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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
        "要求：以 worktree 隔离实施，**worktree 名用本条目 id**——"
        f"`pysci-dev worktree add {item['id']}`（分支 wt-{item['id']}），以便 worker "
        "中途死亡时 drain 孤儿回收能据此定位并清理残留 worktree；完整测试后合并、"
        "commit（commit 后按 charter 的 push 规程执行：限时 best-effort、失败忽略）；"
        "交付中报告改动清单与验证证据，"
        f"并注明 backlog id={item['id']} 以便销账。\n\n"
        "合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，"
        "交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。"
    )


def backlog_batch_text(seed: dict, group: list[dict], *, max_batch: int = 4) -> str:
    """批量消化任务书：种子（backlog_task_text）+ 同提请者组包菜单 + 选取/销账指令。

    group 为空时退化为 :func:`backlog_task_text`（单项，向后兼容）。devops 自主选取
    ≥1 项（**必含种子**），可合并为同一 worktree 实施，交付以 ``backlog id=<id1>,<id2>,…``
    列出**本次完成的全部 id**，orch 逐 id 销账；未选项留 pending 下轮；单项失败按种子
    blocked 交付即可（组内其余不连坐）。
    """
    base = backlog_task_text(seed)
    if not group:
        return base
    member = seed.get("member", "?")
    seed_id = seed.get("id", "")
    menu = "\n".join(
        f"- `{g.get('id', '')}`：{str(g.get('summary', ''))[:120]}" for g in group
    )
    return (
        f"{base}\n\n---\n\n"
        f"## 批量菜单（同提请者 `{member}`，自主选题合并实施）\n\n"
        f"上面是**种子任务**（必做）。以下是同提请者的其余待办，你**可自主选取 ≥0 项**"
        f"与种子合并实施（含种子单批 ≤ {max_batch} 项）；同提请者≈同模块/同视角，"
        "合并可省重复上下文。可按需把选中项重写为合并方案/任务书，在**同一 worktree** 实施。\n\n"
        f"{menu}\n\n"
        "**选取与销账**：只吃种子也可以（组内其余自动留 pending 下轮）。交付时在 "
        f"<result> 中以 `backlog id=<id1>,<id2>,…` 列出**本次完成的全部 id**（必含种子 "
        f"`{seed_id}`），orch 据此逐 id 销账；未完成/未选的**不要**列入。若种子本身受阻，"
        "照常交付 <blocked>（组内其余不连坐，留 pending）。"
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
    entry = {
        "id": sugg.stem,
        "member": member,
        "summary": first[:200],
        "evidence_file": f"orchestration/state/suggestions/approved/{sugg.name}",
        "status": "pending",
        "proposed": _now(),
        "note": args.note or "",
    }
    # effort 是**入队时**的组长判断（推理密集项标 high，机械项不标=默认中）；drain 消化
    # 时透传给该跳（见 drain._drain_batch）。getattr 同 no_wake：程序化调用方常建 partial 参数。
    level = normalize_effort(getattr(args, "effort", None))
    if level:
        entry["effort"] = level
    n = _backlog_append(entry)
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


def _fmt_dur(seconds: float) -> str:
    """秒 → 人类可读时长（h/m/s）。"""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def backlog_stats() -> dict:
    """backlog 侧统计：done 条目**等待时长**（proposed→done_at）与**批量大小分布**。

    数据源为 backlog.json（非台账）——proposed/done_at/batch_size 均为 backlog 条目字段
    （batch_size 由批量消化销账时写入，缺省按 1）。等待时长反映建议从被采纳到落地的时延，
    批量分布反映 drain 组包消化的实际批大小（backlog 20261010-batch-digest）。
    """
    empty = {"done": 0, "pending": 0, "waits_s": [], "batch_dist": {}}
    if not BACKLOG_PATH.exists():
        return empty
    try:
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    waits: list[float] = []
    batch_dist: dict[int, int] = {}
    done = pending = 0
    for it in data.get("items", []):
        st = it.get("status")
        if st == "pending":
            pending += 1
        if st != "done":
            continue
        done += 1
        proposed, done_at = it.get("proposed"), it.get("done_at")
        if proposed and done_at:
            try:
                dt = (
                    datetime.fromisoformat(str(done_at))
                    - datetime.fromisoformat(str(proposed))
                ).total_seconds()
            except (ValueError, TypeError):
                dt = -1
            if dt >= 0:
                waits.append(dt)
        try:
            bs = max(1, int(it.get("batch_size", 1)))
        except (ValueError, TypeError):
            bs = 1
        batch_dist[bs] = batch_dist.get(bs, 0) + 1
    return {
        "done": done,
        "pending": pending,
        "waits_s": waits,
        "batch_dist": batch_dist,
    }


def _print_backlog_stats() -> None:
    """打印 backlog 等待时长 + 批量大小分布（cmd_stats 用；独立于台账数据）。"""
    bs = backlog_stats()
    if not bs["done"]:
        return
    print("== backlog 消化 ==")
    waits = sorted(bs["waits_s"])
    if waits:
        n = len(waits)
        median = waits[n // 2] if n % 2 else (waits[n // 2 - 1] + waits[n // 2]) / 2
        avg = sum(waits) / n
        print(
            f"  等待时长（proposed→done）：中位 {_fmt_dur(median)} / 平均 {_fmt_dur(avg)}"
            f" / 最长 {_fmt_dur(waits[-1])}（n={n}）"
        )
    dist = dict(sorted(bs["batch_dist"].items()))
    print(f"  批量大小分布（done 条目）：{dist}  当前待办={bs['pending']}")


def cmd_stats(args) -> int:
    """三级统计：计划级（--plan）/ 成员级（--member/--days）/ 全体历史 + backlog 消化。"""
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
    else:
        print(f"== {title} ==")
        print(
            f"  跳数={s['hops']}  总耗时={s['duration_ms'] / 1000:.0f}s  "
            f"{format_est_tokens(s)}  {format_credits(s)}"
        )
        for mid, row in sorted(s["by_member"].items()):
            print(
                f"  {mid:<10} 跳数={row['hops']:<4} 耗时占比={row['duration_pct']}%  "
                f"est占比={row['est_tokens_pct']}%  credits占比={row['credits_pct']}%"
            )
        if len(s["by_effort"]) > 1:
            # 单一档时 by_effort ≡ 全体，无 A/B 信号；>=2 档才打（backlog 20261010-153041-devops）
            print(
                "  按推理强度档位（轮均=num_turns 均值，返工率=kind≠result 跳占比）："
            )
            print(format_by_effort(s["by_effort"]))
        kinds: dict[str, int] = {}
        for e in sub:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        print(f"  交付分布：{kinds}")
    _print_backlog_stats()
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


def _extract_goal_excerpt(path: Path, limit: int = 1500) -> str:
    """从原生产任务书摘取目标陈述正文（供 review 任务书 G2 独立核验）。

    剥离 :func:`dispatch.write_taskbook` 自动加的头（``# 任务书 …`` 与 ``（派发时间：…）``）
    与前导空行，保留组长撰写的任务正文，限长 ``limit`` 字符（超出截断并标注）。

    Args:
        path: 原生产任务书路径（绝对或相对项目根）。
        limit: 摘录正文的最大字符数。

    Returns:
        目标陈述正文；文件不可读/为空时返回 ``""``（调用方据此告警，不中断审查）。
    """
    p = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    body: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        # 跳过自动头与前导空行，直到遇到第一行真实正文
        if not body and (
            s.startswith("# 任务书") or s.startswith("（派发时间") or not s
        ):
            continue
        body.append(ln)
    text = "\n".join(body).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …（截断）"
    return text


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
    # 目标陈述（G2 独立核验）：组长显式提供，避免 reviewer 依赖生产者 notes.md 自述。
    # --goal（直接文本）优先于 --goal-from（从原生产任务书摘录）；二者皆缺则不嵌入并告警。
    goal_text = ""
    goal_source = ""
    if getattr(args, "goal", None):
        goal_text = str(args.goal).strip()
        goal_source = "literal(--goal)"
    elif getattr(args, "goal_from", None):
        gp = Path(args.goal_from)
        goal_text = _extract_goal_excerpt(gp)
        if goal_text:
            goal_source = orchestration_rel(
                gp if gp.is_absolute() else PROJECT_ROOT / gp
            )
        else:
            print(
                f"[!] --goal-from 任务书不可读或为空：{gp}"
                "（本次审查 G2 将缺独立目标陈述）"
            )
    if not goal_text:
        print(
            "[!] 未提供目标陈述（--goal/--goal-from）——reviewer 的 G2「与任务书目标一致」"
            "将无独立依据，建议补派时带上原生产任务书。"
        )
    record: dict = {
        "id": review_id,
        "artifact": args.artifact,
        "origin": args.origin,
        "rubric": args.rubric,
        "goal_source": goal_source,
        "goal_excerpt": goal_text,
        "started": _now(),
        "rounds": [],
    }
    goal_block = (
        (
            f"- 目标陈述（来自原生产任务书：{goal_source}；G2「与任务书目标一致」"
            "**据此独立核验**，勿以生产者 notes.md/自述替代）：\n"
            f"{goal_text}\n"
        )
        if goal_text
        else ""
    )
    max_rounds = 2  # 初审 + 复审一次（用户裁决）
    for round_no in range(1, max_rounds + 1):
        task = (
            f"审查任务（第 {round_no} 轮，review id={review_id}）：\n\n"
            f"- 产物：{args.artifact}\n"
            f"- rubric（必读，按它逐项检查）：{rubric_abs}\n"
            f"- 生产组员：{args.origin}\n"
            + goal_block
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
