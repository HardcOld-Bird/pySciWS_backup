"""交付台账：每跳一行 JSON（credits/耗时/验收/上下文占用），支撑 §4.4 统计。

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

LEDGER_PATH: Path = ORCH_STATE_ROOT / "ledger" / "deliveries.jsonl"


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
    num_turns: int = 0
    duration_ms: int = 0
    credits: float = 0.0
    # ADR（2026-10-10，backlog 20261009-byok-cost-accounting）：credits 并非「BYOK 恒 0」。
    # 实测 14 跳台账：max 档 BYOK 模型（Qwen3.8-Max）正常上报 total_credits，flash 档
    # （Qwen3.8-Flash）恒 0——计量能力**依模型而定**。故新增 model 字段：缺它则 credits=0
    # 无法区分「未计量档位」与「真零」，成本列不可解释。用户裁决（2026-10-09）BYOK 为主力、
    # 完整计费留待未来，故此处**不做** token/账单核算，只补齐让既有部分数据可解释的最小维度。
    # model = 本跳实际派发的模型名（dispatch 解析 -m 后写入；旧台账无此字段→""）。
    model: str = ""
    # effort = 本跳实际所用的推理强度档位（""=跟随用户级默认=中）。用户裁决 2026-10-10
    # 的按项标注机制要靠台账做 A/B（同类项 high vs medium 的轮数/失败率/返工率），
    # 缺这个字段就分不清哪一跳是被标注的——旧行无此字段→""。
    effort: str = ""
    ctx_ratio: float = 0.0
    permission_denials: int = 0
    plan: str = ""  # 所属计划 id（Phase 2 起填充）
    step: str = ""
    error: str = ""


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
    """聚合统计：总耗时/credits，按成员占比（§4.4 三层统计的第一层）。

    Args:
        entries: 台账记录子集（调用方负责筛选范围）。

    Returns:
        ``{"hops", "duration_ms", "credits", "credits_metered_hops",
        "by_member": {id: {...占比}}}``。``credits_metered_hops`` 为 credits>0 的跳数——
        BYOK 下计量依模型而定（见 Ledger.model 处 ADR），故须与 ``hops`` 并读才知成本覆盖率。
    """
    total_ms = sum(e.duration_ms for e in entries)
    total_cr = sum(e.credits for e in entries)
    metered_hops = sum(1 for e in entries if e.credits > 0)
    by_member: dict[str, dict[str, Any]] = {}
    for e in entries:
        m = by_member.setdefault(
            e.member, {"hops": 0, "duration_ms": 0, "credits": 0.0}
        )
        m["hops"] += 1
        m["duration_ms"] += e.duration_ms
        m["credits"] += e.credits
    for m in by_member.values():
        m["duration_pct"] = (
            round(100 * m["duration_ms"] / total_ms, 1) if total_ms else 0.0
        )
        m["credits_pct"] = round(100 * m["credits"] / total_cr, 1) if total_cr else 0.0
    return {
        "hops": len(entries),
        "duration_ms": total_ms,
        "credits": round(total_cr, 3),
        "credits_metered_hops": metered_hops,
        "by_member": by_member,
    }


def format_credits(summary: dict[str, Any]) -> str:
    """把 summarize 的成本维度渲染成**诚实**的一行文本（供 stats/plan/workflow 复用）。

    BYOK 下 credits 依模型上报（flash 档恒 0、max 档计量），单看总额会误导：既可能把
    「未计量」当成「零成本」，也可能把部分覆盖当成全量。故文本显式标注覆盖跳数。

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
