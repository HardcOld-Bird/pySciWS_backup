"""交付台账：每跳一行 JSON（credits/est_tokens/耗时/验收/上下文占用），支撑 §4.4 统计。

存储于 ``orchestration/state/ledger/deliveries.jsonl``——append-only、可读、
fail-open（损坏行跳过并告警，不阻塞派发）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from .failure_tags import failure_counts

LEDGER_PATH: Path = ORCH_STATE_ROOT / "ledger" / "deliveries.jsonl"

#: ``by_effort`` 桶里代表「未标注 → 跟随用户级默认（中）」的哨兵键。真实档位永不会
#: 叫这个名字（CLI 枚举 auto/none/low/medium/high，见 runner.VALID_EFFORTS），故不冲。
EFFORT_DEFAULT_KEY = "(默认)"


@dataclass
class LedgerEntry:
    """一次派发（一跳）的台账记录。"""

    ts: str
    member: str
    session_id: str
    session_name: str = ""
    task_file: str = ""
    kind: str = ""  # result | blocked | parse_error | run_failed
    checks: list[dict[str, Any]] = field(default_factory=list)
    infra_suggestion: bool = False
    # 本字段=**已入库**（审批队列里有对应条目），不是「交付文本里出现过那个块」；
    # 非法/缺失 kind 的块会被 dispatch._persist_suggestion 挡在队列外，此时本字段为假。
    # E0-B（判据 2）：本行建议的 kind 属性原值（defect/usability/非法串/""）——旧行无此
    # 字段→""。两字段合起来才读得全：kind 非空而未入库=被词表闸拒掉的那一跳，呈现面据此
    # 打「建议✗未入库」。供台账侧统计提案分布、也随交付落盘一份可追溯记录。
    infra_kind: str = ""
    # E1（README §5.4）：<blocked tags="a,b"> 解析出的失败标签（受控词表见
    # state/failure-tags.json）。空列表=未标注（合法缺省）；旧行无此字段→[]。
    # 本字段是 §2.4 阈值自动立案的唯一原料：标签只在这里落一次盘，统计与立案都读
    # 台账，不再各自解析交付文本（两处解析迟早漂移）。
    tags: list[str] = field(default_factory=list)
    num_turns: int = 0
    duration_ms: int = 0
    credits: float = 0.0
    # ADR（2026-10-10，backlog 20261009-byok-cost-accounting）：credits 并非「BYOK 恒 0」。
    # 实测 14 跳台账：**BYOK 渠道**（人类名 Qwen-3.8-Max，带连字符）正常上报 total_credits，
    # **内置渠道**（Qwen3.8-Flash，无连字符）恒 0——计量能力**依模型而定**，与档位名无关。
    # 故新增 model 字段：缺它则 credits=0 无法区分「未计量」与「真零」，成本列不可解释。
    # 渠道↔档位 2026-10-12 互换（backlog 20261012-173500-leader）后这条更是硬道理：flash 档
    # 改 BYOK 优先，档位名不再预示是否计量，判读一律看本字段（首跳上报行为待真派发取证）。
    # 用户裁决（2026-10-09）BYOK 为主力、完整计费留待未来，故此处**不做** token/账单核算，
    # 只补齐让既有部分数据可解释的最小维度。
    # model = 本跳实际派发的模型名（dispatch 解析 -m 后写入；旧台账无此字段→""）。
    model: str = ""
    # effort = 本跳实际所用的推理强度档位（""=跟随用户级默认=中）。用户裁决 2026-10-10
    # 的按项标注机制要靠台账做 A/B（同类项 high vs medium 的轮数/失败率/返工率），
    # 缺这个字段就分不清哪一跳是被标注的——旧行无此字段→""。
    effort: str = ""
    # 相对用量尺子（用户裁决 2026-10-10，backlog 20261010-ledger-est-tokens）：
    # envelope 与 jsonl 逐轮 usage 在 BYOK/内置**双渠道实测全为 0**、平台网关无
    # usage/balance 路由 → 精确 token 计量当前不可得；改为「本跳会话 jsonl 字符增量」
    # 估算。delta_chars 存**未换算的增量原值**（系数改动后可重算历史，无需重跑派发），
    # est_tokens = delta_chars × registry est_tokens_per_char（启发式，见
    # registry.DEFAULT_TOKENS_PER_CHAR）。要点是**相对排名**而非绝对精度：两渠道同法同尺
    # 才可比；不做缓存命中修正（本地不可得，对排名影响近似中性）。旧行无此二字段→0。
    delta_chars: int = 0
    est_tokens: int = 0
    ctx_ratio: float = 0.0
    permission_denials: int = 0
    plan: str = ""  # 所属计划 id（Phase 2 起填充）
    step: str = ""
    #: sid 在途闸换新会话的原因（``prev_alive``=上一跳进程仍活；""=没换）。
    #: 台账留痕：批次自动换 sid 与人工拒绝都必须在读数面可见（184000 修复面④）。
    sid_new_reason: str = ""
    error: str = ""
    # 同席重试读数（用户裁决 2026-10-13，backlog 20261013-143000-leader）：
    # ``same_seat_retries`` = 本跳在原席上 resume 重试了几次（0=没重试）；
    # ``same_seat_retry_ms`` = 这些重试累计跑了多久（ms，含退避）。
    # 落台账而不入「渠道回退」事件面（重试不是降级）——但抖动频率与预算触顶要有统计原料，
    # 「今天 channel_error 重试了 37 次」正是判断「渠道在抖 vs 渠道已死」的读数。
    same_seat_retries: int = 0
    same_seat_retry_ms: int = 0
    # 边界避让：本跳出发前等了多少秒对齐峰谷边界（0=没等；用户裁决 2026-10-13，
    # backlog 20261013-144500-leader）。台账留这个字段是为了统计「避让节省」——
    # 等待本身不产生成本（零 LLM 调用），但要知道它发生了、发生了多少次。
    hold_s: int = 0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append(entry: LedgerEntry) -> None:
    """追加一条台账记录（自动补时间戳）。"""
    if not entry.ts:
        entry.ts = _now()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def read_all() -> list[LedgerEntry]:
    """读取全部台账；损坏行跳过（fail-open）。"""
    if not LEDGER_PATH.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(LedgerEntry(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return entries


def summarize(entries: list[LedgerEntry]) -> dict[str, Any]:
    """聚合统计：总耗时/credits/est_tokens，按成员/推理强度分档（§4.4 三层统计的第一层）。

    Args:
        entries: 台账记录子集（调用方负责筛选范围）。

    Returns:
        ``{"hops", "duration_ms", "credits", "credits_metered_hops", "est_tokens",
        "est_tokens_metered_hops", "by_member": {id: {...占比}},
        "by_effort": {档: {...占比, avg_turns, fail_pct}},
        "by_failure": {tag: {member: 次数}}}``。
        ``credits_metered_hops`` 为 credits>0 的跳数——BYOK 下计量依模型而定（见
        Ledger.model 处 ADR），故须与 ``hops`` 并读才知成本覆盖率。
        ``est_tokens_metered_hops`` 同理是 est>0 的跳数：本字段上线前的旧行记 0，
        覆盖率不足时**占比只在已计量的跳之间有意义**。
        ``by_effort`` 依 ``Ledger.effort`` 分档；空标注（跟随用户级默认）归入哨兵键
        :data:`EFFORT_DEFAULT_KEY`。**服务档位数据的自然 A/B**（用户裁决 2026-10-10，
        backlog `20261010-153041-devops`）：high vs medium 同类项的 ``avg_turns``
        （轮均）与 ``fail_pct``（``kind != "result"`` 跳占比 = 返工率）就是档位策略
        应否调整的直接读数——不必再人肉 jsonl 逐行比对。
    """
    total_ms = sum(e.duration_ms for e in entries)
    total_cr = sum(e.credits for e in entries)
    total_est = sum(e.est_tokens for e in entries)
    metered_hops = sum(1 for e in entries if e.credits > 0)
    est_hops = sum(1 for e in entries if e.est_tokens > 0)
    by_member: dict[str, dict[str, Any]] = {}
    by_effort: dict[str, dict[str, Any]] = {}
    for e in entries:
        m = by_member.setdefault(
            e.member,
            {"hops": 0, "duration_ms": 0, "credits": 0.0, "est_tokens": 0},
        )
        m["hops"] += 1
        m["duration_ms"] += e.duration_ms
        m["credits"] += e.credits
        m["est_tokens"] += e.est_tokens
        key = e.effort or EFFORT_DEFAULT_KEY
        f = by_effort.setdefault(
            key,
            {
                "hops": 0,
                "duration_ms": 0,
                "credits": 0.0,
                "est_tokens": 0,
                "sum_turns": 0,
                "fail_hops": 0,
            },
        )
        f["hops"] += 1
        f["duration_ms"] += e.duration_ms
        f["credits"] += e.credits
        f["est_tokens"] += e.est_tokens
        f["sum_turns"] += e.num_turns
        if e.kind != "result":
            f["fail_hops"] += 1
    for m in by_member.values():
        m["duration_pct"] = (
            round(100 * m["duration_ms"] / total_ms, 1) if total_ms else 0.0
        )
        m["credits_pct"] = round(100 * m["credits"] / total_cr, 1) if total_cr else 0.0
        m["est_tokens_pct"] = (
            round(100 * m["est_tokens"] / total_est, 1) if total_est else 0.0
        )
    for f in by_effort.values():
        f["duration_pct"] = (
            round(100 * f["duration_ms"] / total_ms, 1) if total_ms else 0.0
        )
        f["credits_pct"] = round(100 * f["credits"] / total_cr, 1) if total_cr else 0.0
        f["est_tokens_pct"] = (
            round(100 * f["est_tokens"] / total_est, 1) if total_est else 0.0
        )
        f["avg_turns"] = round(f["sum_turns"] / f["hops"], 1) if f["hops"] else 0.0
        f["fail_pct"] = round(100 * f["fail_hops"] / f["hops"], 1) if f["hops"] else 0.0
    return {
        "hops": len(entries),
        "duration_ms": total_ms,
        "credits": round(total_cr, 3),
        "credits_metered_hops": metered_hops,
        "est_tokens": total_est,
        "est_tokens_metered_hops": est_hops,
        "by_member": by_member,
        "by_effort": by_effort,
        # E1（README §5.4）：标签×成员计数（只数带 tags 的行）。窗口由调用方决定——
        # stats 用 --days/tier_a 滑窗，立案扫描用 registry 的 tier_a 窗，同一函数两种窗。
        "by_failure": failure_counts(entries),
    }


def format_by_failure(
    by_failure: dict[str, dict[str, int]],
    *,
    threshold: int | None = None,
    vocab: dict[str, str] | None = None,
) -> str:
    """把 summarize 的 ``by_failure`` 渲染成对齐多行文本（Tier A 失败标签面）。

    每标签一行：合计 + 「成员×次数」明细（成员按次数降序、同次数按名字典序），
    标签之间按合计降序。两个可选标注让读数自解释，免得把「已经被人管过」的标签
    和「还在攒证据」的标签混着看：

    * ``threshold`` 给出时，达阈值的行首打 ``!``——那正是 §2.4 会自动立案的一类；
    * ``vocab`` 给出时，不在词表内的标签打 ``[词表外]``（历史行里残留的已删标签
      必须看得见，不能被统计静默吞掉）。

    Args:
        by_failure: ``summarize()["by_failure"]``。
        threshold: 立案阈值（registry ``tier_a.threshold``）；None → 不标注。
        vocab: 受控词表；None → 不校验标签归属。

    Returns:
        多行字符串（无末行换行）；空输入 → ``"（无失败标签数据）"``。
    """
    if not by_failure:
        return "（无失败标签数据）"
    rows = []
    for tag, members in by_failure.items():
        total = sum(members.values())
        detail = ", ".join(
            f"{m}×{n}"
            for m, n in sorted(members.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        mark = "!" if threshold and total >= threshold else " "
        off = "" if vocab is None or tag in vocab else "[词表外]"
        rows.append((-total, tag, f"  {mark} {tag:<18}{off} 合计={total:<4} {detail}"))
    rows.sort()
    return "\n".join(r[2] for r in rows)


def format_est_tokens(summary: dict[str, Any]) -> str:
    """把 summarize 的 est_tokens 维度渲染成一行（相对用量尺子，见 Ledger.est_tokens ADR）。

    credits 未计量的渠道（实测=内置 Qwen3.8-Flash）靠这一列补位——它是**唯一**双渠道都
    可比的用量指标，
    但只是估算：绝对值无意义（系数为启发式），可比的是同尺下的相对量，故文本附覆盖率。

    Args:
        summary: summarize 的返回值。

    Returns:
        ``est≈N tok（覆盖 M/N 跳）``；全覆盖时省略覆盖率；无数据 → ``est —``。
    """
    hops = summary.get("hops", 0)
    if not hops:
        return "est —"
    metered = summary.get("est_tokens_metered_hops", 0)
    tail = "" if metered == hops else f"（覆盖 {metered}/{hops} 跳）"
    return f"est≈{summary.get('est_tokens', 0):,} tok{tail}"


def format_credits(summary: dict[str, Any]) -> str:
    """把 summarize 的成本维度渲染成**诚实**的一行文本（供 stats/plan/workflow 复用）。

    BYOK 下 credits 依模型上报（内置渠道恒 0、BYOK 渠道计量），单看总额会误导：既可能把
    「未计量」当成「零成本」，也可能把部分覆盖当成全量。故文本显式标注覆盖跳数。
    **跨渠道比用量请用 est_tokens**（:func:`format_est_tokens`）——它双渠道同法同尺，
    而未计量的渠道根本无法用 credits 排名。

    Args:
        summary: summarize 的返回值。

    Returns:
        三种情形之一：全未计量 → ``credits 未计量（BYOK，0/N 跳上报）``；
        部分/全部计量 → ``credits X（覆盖 M/N 跳）``；无数据 → ``credits —``。
    """
    hops = summary.get("hops", 0)
    if not hops:
        return "credits —"
    metered = summary.get("credits_metered_hops", 0)
    if not metered:
        return f"credits 未计量（BYOK，0/{hops} 跳上报）"
    return f"credits {summary.get('credits', 0)}（覆盖 {metered}/{hops} 跳）"


def format_by_effort(by_effort: dict[str, dict[str, Any]]) -> str:
    """把 summarize 的 ``by_effort`` 渲染成对齐的多行文本（服务档位 A/B）。

    每档一行：跳数 / 轮均（num_turns 均值）/ 返工率（kind≠result 跳占比）/
    耗时·est·credits 占比。排序：真实档位名字典序在前，:data:`EFFORT_DEFAULT_KEY`
    哨兵恒置末（未标注≡跟随用户级默认，与显式档位是不同语义层，不该混着读）。

    Args:
        by_effort: ``summarize()["by_effort"]``。

    Returns:
        多行字符串（行间以 ``\\n`` 分隔，无末行换行）；空输入 → ``"（无 by_effort 数据）"``。
    """
    if not by_effort:
        return "（无 by_effort 数据）"
    keys = sorted(k for k in by_effort if k != EFFORT_DEFAULT_KEY)
    if EFFORT_DEFAULT_KEY in by_effort:
        keys.append(EFFORT_DEFAULT_KEY)
    lines = []
    for k in keys:
        r = by_effort[k]
        lines.append(
            f"  {k:<8} 跳数={r['hops']:<4} 轮均={r['avg_turns']:<5} "
            f"返工率={r['fail_pct']}%  耗时占比={r['duration_pct']}%  "
            f"est占比={r['est_tokens_pct']}%  credits占比={r['credits_pct']}%"
        )
    return "\n".join(lines)
