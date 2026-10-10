"""台账成本计量回归（backlog 20261009-byok-cost-accounting）。

锁定三件事：
1. ``model`` 字段写入/读回round-trip，且旧台账（无 model 字段）向后兼容读为 ""；
2. ``summarize`` 报告 ``credits_metered_hops``（credits>0 的跳数）——BYOK 下计量依模型而定，
   须与 hops 并读才知成本覆盖率；
3. ``format_credits`` 三分支诚实渲染（无数据/全未计量/部分或全部计量）。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import ledger
from pysci.skills.orchestration.tools.ledger import (
    LedgerEntry,
    append,
    format_credits,
    read_all,
    summarize,
)


@pytest.fixture
def iso_ledger(tmp_path, monkeypatch):
    """把 LEDGER_PATH 重定向到 tmp，隔离真实台账。"""
    path = tmp_path / "deliveries.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    return path


def _entry(member: str, credits: float, model: str = "") -> LedgerEntry:
    return LedgerEntry(
        ts="2026-10-10T00:00:00+08:00",
        member=member,
        session_id="sid-" + member,
        kind="result",
        credits=credits,
        model=model,
        duration_ms=1000,
        num_turns=5,
    )


# --- 1. model round-trip + 向后兼容 -------------------------------------


def test_model_roundtrip(iso_ledger):
    append(_entry("devops", 142.5, model="Qwen3.8-Max"))
    got = read_all()
    assert len(got) == 1
    assert got[0].model == "Qwen3.8-Max"
    assert got[0].credits == 142.5


def test_read_legacy_entry_without_model(iso_ledger):
    """旧台账行没有 model 键 → 读为默认 ""，不抛异常（fail-open 向后兼容）。"""
    legacy = {
        "ts": "2026-10-09T20:09:31+08:00",
        "member": "figure",
        "session_id": "abc",
        "kind": "result",
        "credits": 0.0,
        "ctx_ratio": 0.15,
        "num_turns": 7,
        "duration_ms": 48000,
    }
    iso_ledger.write_text(
        json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    got = read_all()
    assert len(got) == 1
    assert got[0].model == ""
    assert got[0].member == "figure"


# --- 2. summarize credits_metered_hops ----------------------------------


def test_summarize_counts_metered_hops():
    entries = [
        _entry("figure", 0.0, "Qwen3.8-Flash"),
        _entry("lit", 0.0, "Qwen3.8-Flash"),
        _entry("devops", 142.5, "Qwen3.8-Max"),
        _entry("reviewer", 13.0, "Qwen3.8-Max"),
    ]
    s = summarize(entries)
    assert s["hops"] == 4
    assert s["credits_metered_hops"] == 2
    assert s["credits"] == round(0.0 + 0.0 + 142.5 + 13.0, 3)


def test_summarize_all_unmetered():
    entries = [_entry("figure", 0.0, "Qwen3.8-Flash"), _entry("lit", 0.0)]
    s = summarize(entries)
    assert s["hops"] == 2
    assert s["credits_metered_hops"] == 0


def test_summarize_empty():
    s = summarize([])
    assert s["hops"] == 0
    assert s["credits_metered_hops"] == 0
    assert s["credits"] == 0


# --- 3. format_credits 三分支 -------------------------------------------


def test_format_credits_no_data():
    assert format_credits(summarize([])) == "credits —"


def test_format_credits_all_unmetered():
    s = summarize([_entry("figure", 0.0), _entry("lit", 0.0), _entry("figure", 0.0)])
    text = format_credits(s)
    assert "未计量" in text
    assert "0/3" in text


def test_format_credits_partial_coverage():
    s = summarize(
        [
            _entry("figure", 0.0, "Qwen3.8-Flash"),
            _entry("devops", 100.0, "Qwen3.8-Max"),
        ]
    )
    text = format_credits(s)
    assert "覆盖 1/2 跳" in text
    assert "100.0" in text or "100" in text


def test_format_credits_full_coverage():
    s = summarize(
        [_entry("devops", 50.0, "Qwen3.8-Max"), _entry("reviewer", 25.0, "Qwen3.8-Max")]
    )
    text = format_credits(s)
    assert "覆盖 2/2 跳" in text
    assert "未计量" not in text
