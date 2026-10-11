"""改进循环与统计命令：approve/reject/suggestions/consult/stats（README §4.4/§6）。

审批链：dispatch 把 infra_suggestion 持久化到 ``state/suggestions/<id>.md``（pending）
→ 组长 approve（入 backlog + 写审批回复）/ reject（**须 --override 留痕**，写回复 + 落决策日志）
→ 回复在下次派发该成员时自动附送（闭环告知）。护栏级建议不得自行 approve（须转呈用户）——由组长规程软性约束。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import journal
from .delivery import parse_suggestion_fields
from .dispatch import REPLIES_DIR, do_dispatch
from .failure_tags import TAGS_PATH, failure_counts, load_tags, within_window
from .ledger import (
    LEDGER_PATH,
    format_by_effort,
    format_by_failure,
    format_credits,
    format_est_tokens,
    read_all,
    summarize,
)
from .registry import Registry, orchestration_rel
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
        '[NEXT] 审批：uv run pysci-orch approve <id> --note "..."；'
        '否决：uv run pysci-orch reject <id> --override "<判决依据>"（必填，留痕入决策日志）。'
    )
    return 0


def cmd_approve(args) -> int:
    """采纳建议：入 backlog FIFO + 写审批回复（下次派发自动附送）。"""
    # Tier A 阈值扫描（REFACTOR §2.4）：每次审批入口机械重算，把越限的失败标签立成案。
    tier_a_sweep()
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'（orch suggestions 查看）")
        return 2
    member = _member_of(sugg)
    body = sugg.read_text(encoding="utf-8")
    # 摘要行：跳过标题+空行，并跳过 dispatch 头部注入的 `kind:` 标记行（否则会被误当摘要）。
    first = sugg.stem
    for ln in body.splitlines()[2:]:
        s = ln.strip()
        if not s or s.lower().startswith("kind:"):
            continue
        first = s
        break
    # E0-B（REFACTOR §12 判据 2/3）：backlog 条目须带结构化 kind 与可证伪 success_criteria。
    # 从建议正文解析（约定标记行），success_criteria 可由 --criteria 补录；缺任一 → 拒绝入队。
    kind, criteria = parse_suggestion_fields(body)
    cli_criteria = (getattr(args, "criteria", "") or "").strip()
    if cli_criteria:
        criteria = cli_criteria
    if not kind:
        print(
            f"[!] 建议 '{sugg.stem}' 缺 kind 分类，拒绝入队。请在建议正文补一行"
            "  `kind: defect`（缺陷，附复现证据）或 `kind: usability`（难用/体验类）。\n"
            '    （交付侧本应带 <infra_suggestion kind="...">；此处缺失说明来源早于 E0-B。）'
        )
        return 2
    if not criteria:
        print(
            f"[!] 建议 '{sugg.stem}'（kind={kind}）缺可证伪成功判据，拒绝入队。\n"
            "    补写其一：① 在建议正文加 `success_criteria` 标记行 + 其后要点；"
            '② 或本次 approve 追加 --criteria "<做完后如何机械验证算成功>"。'
            "（REFACTOR §12 判据 3：无判据的提案不可证伪，不进队。）"
        )
        return 2
    entry = {
        "id": sugg.stem,
        "member": member,
        "summary": first[:200],
        "kind": kind,
        "success_criteria": criteria,
        "evidence_file": f"orchestration/state/suggestions/approved/{sugg.name}",
        "status": "pending",
        "proposed": _now(),
        "note": args.note or "",
    }
    # 提案来源标记（Tier A 自动立案写 `source: tier-a`）随条目入队，让 drain 与人工
    # 复盘都能看出这条是「统计机制立的案」还是「组员自己提的建议」。
    source = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in body.splitlines()
            if ln.strip().lower().startswith("source:")
        ),
        "",
    )
    if source:
        entry["source"] = source
    # effort 是**入队时**的组长判断（推理密集项标 high，机械项不标=默认中）；drain 消化
    # 时透传给该跳（见 drain._drain_batch）。getattr 同 no_wake：程序化调用方常建 partial 参数。
    level = normalize_effort(getattr(args, "effort", None))
    if level:
        entry["effort"] = level
    n = _backlog_append(entry)
    # 审批回复的收件人：自动立案没有「提请成员」可回话（member=tier-a 不是真成员，
    # 回复永远不会被附送），而基建修复的承接方按角色边界就是 devops——归它，回执才落地。
    reply_member = "devops" if entry.get("source") == "tier-a" else member
    REPLIES_DIR.joinpath(reply_member).mkdir(parents=True, exist_ok=True)
    (REPLIES_DIR / reply_member / f"{sugg.stem}.md").write_text(
        f"# 组长审批：采纳（{_now()}）\n\n{args.note or '（无附注）'}\n\n"
        f"已入 backlog（第 {n} 位），devops 将按 FIFO 处理；完成后你会在后续任务中获知。\n",
        encoding="utf-8",
    )
    approved = SUGGESTIONS_DIR / "approved"
    approved.mkdir(exist_ok=True)
    sugg.rename(approved / sugg.name)
    print(
        f"[√] 已采纳 {sugg.stem}（kind={kind}）→ backlog 第 {n} 位；"
        f"审批回复将在下次派发 {reply_member} 时附送。"
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
    """否决建议：**必须附 ``--override`` 判决依据**（留痕入决策日志），否则非零退出。

    REFACTOR v1.0 §5.1/E0——组长对组员基建提案的否决 = override（有判决依据的 reject
    才算 override；无依据的否决不合规）。当前 approve 侧判决包（J/M/J+M）E5 才实装，
    故 E0 阶段一切 reject 均属 override，强制留痕。缺失 → 非零退出 + 可读提示；
    提供 → 正常否决（写回复 + 移入 rejected/）并自动追加 journal 行
    （``source=override``、``situation``=建议摘要、``reason``=override 理由）。
    """
    override = (getattr(args, "override", "") or "").strip()
    if not override:
        print(
            "[!] 否决必须附判决依据（override 留痕）：orch reject <id> --override "
            '"<详细理由>"\n'
            "    无 --override 的否决不合规（REFACTOR §5.1/E0：reject=override，须计入 "
            "override 率并落决策日志供审计回放）。"
        )
        return 2
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'")
        return 2
    member = _member_of(sugg)
    body = sugg.read_text(encoding="utf-8")
    first = next((ln.strip() for ln in body.splitlines()[2:] if ln.strip()), sugg.stem)
    # 先落审计原料（journal），再改建议状态——journal 写失败即中止，不留下「已否决却无留痕」。
    journal.append(
        {
            "ts": _now(),
            "situation": first[:200],
            "options": "",
            "choice": "reject",
            "reason": override,
            "source": "override",
            "kind": "override",
            "suggestion_id": sugg.stem,
            "member": member,
        }
    )
    note = (args.note or "").strip()
    reply = f"# 组长审批：暂不采纳（{_now()}）\n\n否决依据（override）：{override}\n"
    if note:
        reply += f"\n附注：{note}\n"
    REPLIES_DIR.joinpath(member).mkdir(parents=True, exist_ok=True)
    (REPLIES_DIR / member / f"{sugg.stem}.md").write_text(reply, encoding="utf-8")
    rejected = SUGGESTIONS_DIR / "rejected"
    rejected.mkdir(exist_ok=True)
    sugg.rename(rejected / sugg.name)
    print(
        f"[√] 已否决 {sugg.stem}（override 留痕入决策日志）；"
        f"理由将在下次派发 {member} 时附送。"
    )
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


#: 立案指纹台账（运行态，gitignore）：``{"version":1,"filed":{fp: 立案时刻}}``。
TIER_A_FILED_PATH = ORCH_STATE_ROOT / "tier-a-filed.json"
#: 同一立案指纹的重复抑制窗（REFACTOR §2.4「同指纹 24h 内不重复立案」）。
TIER_A_DEDUP_HOURS = 24


def _load_fingerprints(path: Path) -> dict[str, str]:
    """读立案指纹台账 ``{fp: 立案时刻 ISO}``；缺失/损坏 → 空（fail-open）。

    读不到只会失去一次去重（可能重复立案一条提案），不会丢数据——提案本体在
    ``suggestions/`` 里，台账只是「最近立过哪些」的记忆。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    filed = data.get("filed") if isinstance(data, dict) else None
    if not isinstance(filed, dict):
        return {}
    return {str(k): str(v) for k, v in filed.items() if isinstance(v, str)}


def _save_fingerprints(path: Path, fps: dict[str, str]) -> None:
    """写立案指纹台账（带 ``{fp: ts}`` 原始映射，供人工核对何时立过案）。

    只保留最近 500 个指纹（插入序淘汰）：指纹键含窗口桶，桶随时间单调推进，旧键永不再
    命中，留着只是让文件无界增长。
    """
    keep = dict(list(fps.items())[-500:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "filed": keep}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _recent_cutoff() -> datetime:
    """重复抑制窗的起点（现在 − :data:`TIER_A_DEDUP_HOURS` 小时）。"""
    return datetime.now().astimezone() - timedelta(hours=TIER_A_DEDUP_HOURS)


def tier_a_sweep(
    *,
    entries: list | None = None,
    tier: dict[str, int] | None = None,
    filed_path: Path | None = None,
) -> list[str]:
    """§2.4 阈值自动立案：同一 ``(标签, 成员)`` 在滑窗内达阈值 → 登记基建提案。

    在 ``orch status`` / ``orch approve`` 入口机械调用（REFACTOR §0 设计纪律 1：统计必须
    被流程硬性依赖，不靠「组长有空去看」）。产出的是**待审批提案**
    ``suggestions/<ts>-tier-a.md``（``kind: defect`` + 正文 ``kind:``/``success_criteria``
    标记行齐备，故直接走现行 approve 面），而不是直接写 backlog——§5.1 把「是否接受
    这份判决」设为强制选择点，代码负责把案立到人眼前，人负责点头或 override。

    Args:
        entries: 台账行（缺省现场 ``read_all()``；测试注入用）。
        tier: 窗口与阈值参数（缺省读 registry ``tier_a``）。
        filed_path: 立案指纹台账路径（测试注入用）。

    Returns:
        本次新立案的提案 id（文件名 stem）列表，按标签名排序；无新案 → ``[]``。
    """
    rows = read_all() if entries is None else entries
    params = tier if tier is not None else Registry.load().tier_a
    win = within_window(rows, days=params["window_days"], hops=params["window_hops"])
    counts = failure_counts(win)
    path = filed_path or TIER_A_FILED_PATH
    fps = _load_fingerprints(path)
    cutoff = _recent_cutoff().isoformat(timespec="seconds")
    vocab = load_tags()
    open_keys = _open_tier_a_keys()
    filed: list[str] = []
    for tag in sorted(counts):
        for member in sorted(counts[tag]):
            total = counts[tag][member]
            if total < params["threshold"]:
                continue
            # 窗口桶 = 窗内该 (标签, 成员) 最新一条台账行的日期：同一天内重跑必落同一桶
            # （幂等），跨天才可能换桶。
            hit = [r for r in win if _row_has_tag(r, tag, member)]
            newest = max((_row_ts(r) for r in hit), default="")
            fp = f"{tag}|{member}|{newest[:10]}"
            if fps.get(fp, "") >= cutoff:
                continue  # 同指纹 24h 内立过（§2.4 的去重窗）
            if f"{tag}|{member}" in open_keys:
                # 旧案仍在待审批面（既没 approve 也没 reject）→ 不重立。缺这一道，
                # 窗口桶每天前进就会把「同一个没修好的问题」每天再立一次。
                continue
            SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
            sid = f"{_filed_ts()}-tier-a"
            n = 0
            while (SUGGESTIONS_DIR / f"{sid}.md").exists():
                # 秒级 id 会同名（E4 引出的同类缺陷在 backlog 20261011-233248-devops）：
                # 一次扫描里立多个案子必须各占一文件，否则后写的把先写的静默覆盖。
                n += 1
                sid = f"{_filed_ts()}.{n}-tier-a"
            body = _tier_a_proposal(
                tag=tag,
                member=member,
                total=total,
                params=params,
                rows=hit,
                definition=vocab.get(tag, ""),
                fingerprint=fp,
            )
            (SUGGESTIONS_DIR / f"{sid}.md").write_text(body, encoding="utf-8")
            open_keys.add(f"{tag}|{member}")
            fps[fp] = _now()
            filed.append(sid)
    if filed:
        _save_fingerprints(path, fps)
    return filed


def _open_tier_a_keys() -> set[str]:
    """仍在待审批面的自动立案键 ``{"tag|member"}``（读提案正文的 ``fingerprint:`` 标记）。

    靠文件内容而非文件名识别来源：提案文件名里只有一个信息位（成员），留给了
    ``tier-a`` 这个来源标记，具体标签与被立案成员在正文里。
    """
    out: set[str] = set()
    for p in pending_suggestions():
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not any(ln.strip().lower() == "source: tier-a" for ln in lines):
            continue
        for ln in lines:
            s = ln.strip()
            if s.startswith("fingerprint:"):
                parts = s.split(":", 1)[1].strip().split("|")
                if len(parts) >= 2:
                    out.add(f"{parts[0]}|{parts[1]}")
    return out


def _filed_ts() -> str:
    """提案文件名时间戳（与 dispatch 的 ``_ts()`` 同法：``YYYYMMDD-HHMMSS``）。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _row_has_tag(row: Any, tag: str, member: str) -> bool:
    """台账行是否属该成员且带该标签（sweep 取证用，dataclass 与 dict 两种形态都认）。"""
    if _cell(row, "member", "") != member:
        return False
    return tag in (_cell(row, "tags", []) or [])


def _row_ts(row: Any) -> str:
    """台账行的 ``ts``（缺失 → 空串，字典序排最前，不抛）。"""
    return str(_cell(row, "ts", ""))


def _cell(row: Any, name: str, default: Any) -> Any:
    """属性或字典键两种形态都认（与 :func:`failure_tags._field` 同法，本地不引私有函数）。"""
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _tier_a_proposal(
    *,
    tag: str,
    member: str,
    total: int,
    params: dict[str, int],
    rows: list,
    definition: str,
    fingerprint: str,
) -> str:
    """自动生成提案正文（approve 面要求的标记行齐备：``kind:`` / ``success_criteria``）。

    行序有讲究：``cmd_approve`` 取「标题+空行+``kind:`` 行之后第一个非空行」当 backlog
    摘要，故摘要行必须紧跟在 ``kind:`` 行之后、其余 ``source:``/``fingerprint:`` 标记之前。

    证据取台账行引用（最多 8 行，附台账文件相对路径）——立案不靠模型回忆，
    靠可复核的时间戳与任务书。
    """
    refs = "\n".join(
        f"- {_cell(r, 'ts', '')} {_cell(r, 'kind', '')} "
        f"tags={','.join(_cell(r, 'tags', []) or [])} "
        f"任务书={_cell(r, 'task_file', '')}"
        for r in rows[-8:]
    )
    tail = "" if len(rows) <= 8 else f"\n- …（窗内共 {len(rows)} 行，仅列最近 8 行）"
    return (
        f"# Tier A 自动立案（{member} × {tag}，待审批）\n\n"
        f"kind: defect\n"
        f"**{member} 的 `{tag}` 类失败在 {params['window_days']} 天 / "
        f"{params['window_hops']} 跳滑窗内累计 {total} 次**"
        f"（阈值 {params['threshold']}，REFACTOR §2.4 自动立案）\n\n"
        f"source: tier-a\n"
        f"fingerprint: {fingerprint}\n\n"
        f"标签定义（词表 {orchestration_rel(TAGS_PATH)}）："
        f"{definition or '（词表无定义）'}\n\n"
        f"## 证据（台账行引用，台账 {orchestration_rel(LEDGER_PATH)}）\n\n"
        f"{refs}{tail}\n\n"
        f"## 待判定\n\n"
        f"- 设施缺陷（补能力/修工具）、流程缺口（改任务书写法或增补词表），"
        f"还是成员用法问题（回派纠正）？\n"
        f"- 复核：``orch ledger --member {member} --days {params['window_days']}``\n\n"
        f"success_criteria:\n"
        f"- 同一 `(标签={tag}, 成员={member})` 在下一个 {params['window_days']} 天 / "
        f"{params['window_hops']} 跳滑窗内的计数低于阈值 {params['threshold']}"
        f"（``orch stats`` 的 by_failure 段可复核）。\n"
    )


def format_failure_section(entries: list, *, tier: dict[str, int] | None = None) -> str:
    """Tier A 失败标签段（REFACTOR §2）：滑窗计数 + 阈值标注，渲染成可打印多行文本。

    窗口与阈值都取自 registry ``tier_a``（:meth:`Registry.tier_a`），与 :func:`tier_a_sweep`
    的立案判定同一把尺——统计上看着达标的标签，立案逻辑必须同样够得着，两侧不许各算各的。

    Args:
        entries: 台账行（已按 --plan/--member/--days 过滤过的那批）。
        tier: 覆盖窗口参数（测试注入用）；缺省现场读 registry。

    Returns:
        以标题行开头的多行文本（无末行换行）。
    """
    params = tier if tier is not None else Registry.load().tier_a
    win = within_window(entries, days=params["window_days"], hops=params["window_hops"])
    counts = failure_counts(win)
    head = (
        f"  失败标签 by_failure（窗口 {params['window_days']}天 或 最近"
        f"{params['window_hops']}跳，阈值 {params['threshold']}，! = 达阈值将自动立案）："
    )
    return "\n".join(
        [
            head,
            format_by_failure(counts, threshold=params["threshold"], vocab=load_tags()),
        ]
    )


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
        print(format_failure_section(sub))
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
