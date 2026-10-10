"""台账 by_effort 聚合（backlog 20261010-153041-devops）。

按项标注机制的价值全押在「台账自然 A/B」——本组锁定 summarize 的 by_effort
桶与 format_by_effort 呈现：空 effort 归哨兵键、非 result 类算返工、单档不打印
A/B 段（无对比信号）、多档按真实名在前哨兵末位排序、cmd_stats/orch --stats 两处
输出面都覆盖。
"""

from __future__ import annotations

from argparse import Namespace

from pysci.skills.orchestration.tools import orch, workflow
from pysci.skills.orchestration.tools.ledger import (
    EFFORT_DEFAULT_KEY,
    LedgerEntry,
    format_by_effort,
    summarize,
)


def _e(
    member: str,
    *,
    effort: str = "",
    kind: str = "result",
    turns: int = 5,
    dur_ms: int = 1000,
    est: int = 100,
    credits: float = 0.0,
) -> LedgerEntry:
    return LedgerEntry(
        ts="2026-10-10T00:00:00+08:00",
        member=member,
        session_id=f"sid-{member}-{effort}-{kind}",
        kind=kind,
        effort=effort,
        num_turns=turns,
        duration_ms=dur_ms,
        est_tokens=est,
        credits=credits,
    )


# --- 1. summarize.by_effort 桶基本语义 -------------------------------------


def test_summarize_returns_by_effort_key():
    s = summarize([_e("devops", effort="high"), _e("lit")])
    assert set(s["by_effort"]) == {"high", EFFORT_DEFAULT_KEY}


def test_empty_effort_bucketed_under_default_sentinel():
    """ ""（跟随用户级默认）不进 medium/high 任何真实档，独立哨兵桶。"""
    s = summarize([_e("a"), _e("b"), _e("c")])
    assert list(s["by_effort"]) == [EFFORT_DEFAULT_KEY]
    assert s["by_effort"][EFFORT_DEFAULT_KEY]["hops"] == 3


def test_bucket_arithmetic_avg_turns_and_fail_pct():
    # high: 3 跳 turns=4/6/8 均 6.0；其中 blocked 1 跳 → 返工率 33.3%
    # medium: 2 跳 turns=3/5 均 4.0；全 result → 0.0%
    # 时长各 3000ms → 总 15000，high=9000 占 60.0%；est 各 100 → 高 300/500=60.0%
    entries = [
        _e("a", effort="high", turns=4, dur_ms=3000, est=100),
        _e("b", effort="high", turns=6, dur_ms=3000, est=100, kind="blocked"),
        _e("c", effort="high", turns=8, dur_ms=3000, est=100),
        _e("d", effort="medium", turns=3, dur_ms=3000, est=100),
        _e("e", effort="medium", turns=5, dur_ms=3000, est=100),
    ]
    s = summarize(entries)
    high = s["by_effort"]["high"]
    med = s["by_effort"]["medium"]
    assert high["hops"] == 3
    assert high["sum_turns"] == 18 and high["avg_turns"] == 6.0
    assert high["fail_hops"] == 1 and high["fail_pct"] == 33.3
    assert high["duration_ms"] == 9000 and high["duration_pct"] == 60.0
    assert high["est_tokens_pct"] == 60.0
    assert med["avg_turns"] == 4.0 and med["fail_pct"] == 0.0
    assert med["duration_pct"] == 40.0
    assert med["est_tokens_pct"] == 40.0


def test_all_nonresult_yields_100_fail_pct():
    entries = [_e("a", effort="high", kind="run_failed", turns=2)]
    f = summarize(entries)["by_effort"]["high"]
    assert f["fail_hops"] == 1 and f["fail_pct"] == 100.0
    # parse_error 也算返工（未交付）
    f2 = summarize([_e("b", effort="high", kind="parse_error")])["by_effort"]["high"]
    assert f2["fail_pct"] == 100.0


def test_summarize_empty_yields_no_effort_buckets():
    s = summarize([])
    assert s["by_effort"] == {}


# --- 2. format_by_effort 呈现 ----------------------------------------------


def test_format_by_effort_real_tiers_before_sentinel():
    s = summarize(
        [
            _e("a", effort="high"),
            _e("b", effort="low"),
            _e("c", effort="medium"),
            _e("d"),  # 空 → 哨兵
        ]
    )
    text = format_by_effort(s["by_effort"])
    order = [ln.strip().split()[0] for ln in text.splitlines()]
    assert order == ["high", "low", "medium", EFFORT_DEFAULT_KEY]


def test_format_by_effort_carries_all_key_fields():
    s = summarize([_e("a", effort="high", turns=7, kind="blocked")])
    text = format_by_effort(s["by_effort"])
    for token in ("跳数=1", "轮均=7.0", "返工率=100.0%"):
        assert token in text, f"缺 {token}：{text}"


def test_format_by_effort_empty_bucket_placeholder():
    assert "无 by_effort" in format_by_effort({})


# --- 3. cmd_stats / orch ledger --stats 打印门控 ----------------------------


def test_workflow_stats_skips_by_effort_when_single_tier(monkeypatch, capsys):
    """全体同档（未标注 ≡ 单一哨兵桶）时 A/B 段无信号可看，不打印。"""
    entries = [_e("a"), _e("b")]  # 都空
    monkeypatch.setattr(workflow, "read_all", lambda: entries)
    workflow.cmd_stats(Namespace(plan=None, member=None, days=None))
    out = capsys.readouterr().out
    assert "按推理强度档位" not in out


def test_workflow_stats_prints_by_effort_when_multi_tier(monkeypatch, capsys):
    entries = [_e("a", effort="high"), _e("b", effort="medium")]
    monkeypatch.setattr(workflow, "read_all", lambda: entries)
    workflow.cmd_stats(Namespace(plan=None, member=None, days=None))
    out = capsys.readouterr().out
    assert "按推理强度档位" in out
    assert "high" in out and "medium" in out
    # 每档行包含轮均/返工率读数（A/B 主指标）
    assert "轮均=" in out and "返工率=" in out


def test_orch_ledger_stats_skips_by_effort_when_single_tier(monkeypatch, capsys):
    entries = [_e("a", effort="high"), _e("b", effort="high")]
    monkeypatch.setattr(orch, "read_all", lambda: entries)
    orch.cmd_ledger(Namespace(member=None, days=0, stats=True, tail=10))
    out = capsys.readouterr().out
    assert "按推理强度档位" not in out, "单一档不打印 A/B 段"


def test_orch_ledger_stats_prints_by_effort_when_multi_tier(monkeypatch, capsys):
    entries = [_e("a", effort="high"), _e("b")]  # high + 哨兵 = 2 档
    monkeypatch.setattr(orch, "read_all", lambda: entries)
    orch.cmd_ledger(Namespace(member=None, days=0, stats=True, tail=10))
    out = capsys.readouterr().out
    assert "按推理强度档位" in out
    assert "high" in out and EFFORT_DEFAULT_KEY in out
