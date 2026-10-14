"""Tier A 失败标签全链（E1，backlog 20261011-233001-leader）：解析→台账→统计。

覆盖 README §5.4 的数据面（词表读取、``<blocked tags=…>`` 解析、台账持久化、
滑窗计数与呈现）；§2.4 的**立案**后果见 ``test_tier_a_filing.py``，guard 侧
校验见 ``test_delivery_gate_tags.py``。

要锁住的核心性质：**标签只归属失败**（result/parse_error 不记）、**旧台账行不炸**
（新增字段带默认值，追加式文件的历史行必须继续可读）、**统计与 guard 读同一份词表**。
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import failure_tags, ledger, orch, workflow
from pysci.skills.orchestration.tools.delivery import parse_delivery
from pysci.skills.orchestration.tools.ledger import (
    LedgerEntry,
    format_by_failure,
    summarize,
)

NOW = datetime.now().astimezone()


def _ts(days_ago: int = 0) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _e(
    member: str,
    *,
    tags: list[str] | None = None,
    kind: str = "blocked",
    ts: str = "",
) -> LedgerEntry:
    return LedgerEntry(
        ts=ts or _ts(),
        member=member,
        session_id=f"sid-{member}",
        kind=kind,
        tags=tags or [],
        num_turns=3,
        duration_ms=1000,
    )


# --- 1. 词表读取 ------------------------------------------------------------


def test_load_tags_reads_shipped_file():
    """真文件优先：随附词表的受控标签都要在（与 guard 侧读同一文件）。"""
    vocab = failure_tags.load_tags()
    assert "cli_missing" in vocab and "whitelist_conflict" in vocab


def test_shipped_vocab_disambiguates_whitelist_conflict():
    """mis-tag 防回退（backlog 20261014-113347-tier-a）：whitelist_conflict 的定义须点明
    它**不是** merge_dirty_face / concurrent_writer 的兜底，且新标签有非空定义。"""
    vocab = failure_tags.load_tags()
    assert "concurrent_writer" in vocab
    wl = vocab["whitelist_conflict"]
    assert "merge_dirty_face" in wl and "concurrent_writer" in wl
    assert all(d.strip() for d in vocab.values())


def test_load_tags_missing_file_falls_back_to_defaults(iso_state):
    """词表文件没部署（新克隆/隔离态）→ 兜底集合顶上，统计不崩。"""
    assert set(failure_tags.load_tags()) == set(failure_tags.DEFAULT_TAGS)


def test_load_tags_file_wins_including_deletions(iso_state):
    """文件形状即协议：文件里的键**整体覆盖**兜底集，删掉的键不再认。"""
    import json

    (iso_state / "failure-tags.json").write_text(
        json.dumps(
            {"version": 1, "tags": {"only_one": "唯一标签"}}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    assert failure_tags.load_tags() == {"only_one": "唯一标签"}


def test_load_tags_broken_json_falls_back(iso_state):
    (iso_state / "failure-tags.json").write_text("{坏", encoding="utf-8")
    assert set(failure_tags.load_tags()) == set(failure_tags.DEFAULT_TAGS)


# --- 2. tags 属性解析 -------------------------------------------------------


def test_parse_tags_normalizes_case_dedupe_and_separators():
    got = failure_tags.parse_tags(" CLI_Missing ,，cli_missing； tool_bug\tunknown ")
    assert got == ["cli_missing", "tool_bug", "unknown"]


def test_parse_tags_empty_is_legal_empty_list():
    """缺省（不标注）= 空列表，是协议里的合法选择，不是错误。"""
    assert failure_tags.parse_tags(None) == []
    assert failure_tags.parse_tags("  ") == []


def test_unknown_tags_against_vocab():
    assert failure_tags.unknown_tags(["quota", "nonsense"]) == ["nonsense"]


# --- 3. 交付解析：标签只归属失败 -------------------------------------------


def test_blocked_with_tags_parsed():
    d = parse_delivery(
        '<blocked tags="cli_missing,tool_bug">pysci-fig 无 sub 命令</blocked>'
    )
    assert d.kind == "blocked"
    assert d.tags == ["cli_missing", "tool_bug"]


def test_blocked_without_tags_yields_empty_list():
    d = parse_delivery("<blocked>缺信息</blocked>")
    assert d.kind == "blocked" and d.tags == []


def test_result_never_carries_tags():
    """标签是失败分类面；成功没有可分类的对象（协议里也不给 result 写 tags）。"""
    d = parse_delivery('<result tags="quota">做完了</result>')
    assert d.kind == "result" and d.tags == []


def test_parse_error_never_carries_tags():
    d = parse_delivery("既没标签块又有两个块 <result>a</result><blocked>b</blocked>")
    assert d.kind == "parse_error" and d.tags == []


# --- 4. 台账持久化与向后兼容 ------------------------------------------------


def test_ledger_roundtrip_keeps_tags(iso_state):
    """tags 入 deliveries.jsonl 后读回必须一致（success_criteria 之一）。"""
    ledger.append(_e("drawing", tags=["cli_missing"]))
    got = ledger.read_all()
    assert got[0].tags == ["cli_missing"]


def test_old_line_without_tags_still_reads(iso_state):
    """追加式文件的历史行没有 tags 字段：取默认空列表，不得抛（旧台账全废就大事了）。"""
    iso_state.joinpath("ledger").mkdir()
    (iso_state / "ledger" / "deliveries.jsonl").write_text(
        '{"ts": "2026-10-01T00:00:00+08:00", "member": "figure", "session_id": "s",'
        ' "kind": "blocked", "num_turns": 2, "duration_ms": 1}\n',
        encoding="utf-8",
    )
    assert ledger.read_all()[0].tags == []


def test_summarize_carries_by_failure():
    s = summarize(
        [_e("a", tags=["quota"]), _e("a", tags=["quota"]), _e("b", tags=["tool_bug"])]
    )
    assert s["by_failure"] == {"quota": {"a": 2}, "tool_bug": {"b": 1}}


def test_summarize_ignores_untagged_rows():
    assert summarize([_e("a")])["by_failure"] == {}


# --- 5. 滑窗与呈现 ----------------------------------------------------------


def test_within_window_union_of_days_and_hops():
    """§2.4 的「14 天或 20 跳」取并集：陈旧但属于最近 N 跳的行也要入窗。"""
    old = _e("a", tags=["quota"], ts=_ts(days_ago=400))
    recent = _e("a", tags=["tool_bug"], ts=_ts(days_ago=1))
    got = failure_tags.within_window([old, recent], days=14, hops=2)
    assert {e.tags[0] for e in got} == {"quota", "tool_bug"}


def test_within_window_day_filter_alone():
    old, recent = _e("a", ts=_ts(days_ago=400)), _e("b", ts=_ts(days_ago=1))
    got = failure_tags.within_window([old, recent], days=14, hops=None)
    assert [e.member for e in got] == ["b"]


def test_format_by_failure_marks_threshold_and_off_vocab():
    counts = {"cli_missing": {"drawing": 3}, "legacy_tag": {"figure": 1}}
    text = format_by_failure(counts, threshold=3, vocab={"cli_missing": "d"})
    lines = text.splitlines()
    assert lines[0].strip().startswith("!") and "cli_missing" in lines[0]
    assert "drawing×3" in lines[0]
    assert "[词表外]" in lines[1] and not lines[1].strip().startswith("!")


def test_format_by_failure_orders_by_total_desc():
    text = format_by_failure({"a": {"m": 1}, "b": {"m": 5, "n": 2}})
    assert text.index("b ") < text.index("a ")  # 合计 7 排在合计 1 之前
    assert "m×5" in text.splitlines()[0]


def test_format_by_failure_empty_placeholder():
    assert "无失败标签" in format_by_failure({})


# --- 6. 命令面：stats by_failure 段 与 ledger 行渲染 -----------------------


def test_stats_prints_by_failure_section(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(
        workflow,
        "read_all",
        lambda: [
            _e("a", tags=["quota"]),
            _e("a", tags=["quota"]),
            _e("a", tags=["quota"]),
        ],
    )
    workflow.cmd_stats(Namespace(plan=None, member=None, days=None))
    out = capsys.readouterr().out
    assert "by_failure" in out
    assert "quota" in out and "a×3" in out
    assert "!" in out  # 达阈值行要打标（与自动立案同尺）


def test_stats_section_placeholder_when_no_tags(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(workflow, "read_all", lambda: [_e("a", kind="result", tags=[])])
    workflow.cmd_stats(Namespace(plan=None, member=None, days=None))
    out = capsys.readouterr().out
    assert "by_failure" in out and "无失败标签数据" in out


def test_ledger_row_renders_tags(iso_state, monkeypatch, capsys):
    """``orch ledger`` 行渲染显示 tags（success_criteria：台账可见）。"""
    monkeypatch.setattr(orch, "read_all", lambda: [_e("drawing", tags=["cli_missing"])])
    orch.cmd_ledger(Namespace(member=None, days=0, stats=False, tail=10))
    out = capsys.readouterr().out
    assert "标签[cli_missing]" in out


def test_ledger_row_omits_marker_when_untagged(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(orch, "read_all", lambda: [_e("drawing")])
    orch.cmd_ledger(Namespace(member=None, days=0, stats=False, tail=10))
    assert "标签[" not in capsys.readouterr().out


# --- 7. registry 参数面（tier_a 三个数） ------------------------------------


def test_registry_tier_a_defaults_when_absent(iso_registry):
    reg_file, _, _ = iso_registry
    reg_file.write_text('{"version":1,"members":{}}', encoding="utf-8")
    from pysci.skills.orchestration.tools.registry import Registry

    assert Registry.load().tier_a == {
        "window_days": 14,
        "window_hops": 20,
        "threshold": 3,
    }


def test_registry_tier_a_per_key_merge_and_fail_open(iso_registry):
    """逐键覆盖 + 非法键回落：手改坏一个数不该让统计与立案停摆。"""
    reg_file, _, _ = iso_registry
    reg_file.write_text(
        '{"version":1,"members":{},"tier_a":'
        '{"window_days": 30, "threshold": 0, "window_hops": "x"}}',
        encoding="utf-8",
    )
    from pysci.skills.orchestration.tools.registry import Registry

    assert Registry.load().tier_a == {
        "window_days": 30,
        "window_hops": 20,
        "threshold": 3,
    }
