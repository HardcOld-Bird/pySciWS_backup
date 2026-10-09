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
        ``{"hops", "duration_ms", "credits", "by_member": {id: {...占比}}}``。
    """
    total_ms = sum(e.duration_ms for e in entries)
    total_cr = sum(e.credits for e in entries)
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
        "by_member": by_member,
    }
