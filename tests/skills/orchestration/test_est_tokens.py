"""台账 est_tokens 相对用量估算（backlog 20261010-ledger-est-tokens）。

用户裁决 2026-10-10：精确 token 计量不可得——调研实锤 Qoder envelope 与 jsonl 逐轮
usage 在 BYOK/内置**双渠道全为 0**、平台网关无 usage/balance 路由、网页仪表盘仅总览
（平台产品策略，不对抗）。故采纳「会话 jsonl 字符增量 × 启发式系数」做**相对排名**、
放弃绝对精度。本组覆盖：系数可配与 fail-open、测量面（中文按字符不按字节）、双端测量
与跨跳基准、记账外增长的守恒归账、旧台账行兼容、stats/ledger/计划报告呈现面。
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    ledger,
    orch,
    plans,
    workflow,
)
from pysci.skills.orchestration.tools import sessions as session_tools
from pysci.skills.orchestration.tools.ledger import LedgerEntry
from pysci.skills.orchestration.tools.registry import DEFAULT_TOKENS_PER_CHAR, Registry

from .conftest import (
    enable_registry_save,
    envelope,
    fake_headless,
    grow_session,
    read_registry,
)

OK_BODY = "<result>出图完成，commit aaa1111 已合并。</result>"
SID = "sess-0001"


def _write_registry(state, data):
    (state / "registry.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def seed_session(state, sid=SID, **extra):
    """预置一个活跃会话（使 do_dispatch 走 resume 路径、sid 与文件名可预期）。"""
    data = read_registry(state)
    entry = {
        "sid": sid,
        "name": "任务A",
        "hops": 0,
        "last_active": "2026-10-10T00:00:00+08:00",
        "status": "active",
    }
    entry.update(extra)
    data["members"]["quotamember"]["sessions"] = [entry]
    _write_registry(state, data)


def set_coefficient(state, value):
    """写 registry 顶层 est_tokens_per_char（value=None 表示键不存在=走默认）。"""
    data = read_registry(state)
    if value is None:
        data.pop("est_tokens_per_char", None)
    else:
        data["est_tokens_per_char"] = value
    _write_registry(state, data)


def ledger_rows(state):
    """读回隔离台账（不用 ledger.read_all：它绑的是模块级常量路径）。"""
    p = state / "ledger.jsonl"
    if not p.exists():
        return []
    return [
        LedgerEntry(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ok_after_growth(state, sid=SID, chunk=""):
    """fake_headless 响应器：本「跳」会话文件长 chunk，然后成功交付。"""

    def responder(n, cmd):
        grow_session(state, sid, chunk)
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    return responder


def dispatch_one(state, monkeypatch, chunk, **kwargs):
    """一跳成功派发（会话已预置，期间 jsonl 增长 chunk）。"""
    seed_session(state)
    fake_headless(monkeypatch, ok_after_growth(state, chunk=chunk))
    return dispatch.do_dispatch("quotamember", text="出图", quiet=True, **kwargs)


# ---------------------------------------------------------------------------
# 换算系数：默认值 / 可配 / fail-open
# ---------------------------------------------------------------------------
def test_default_coefficient_is_adjudicated_heuristic():
    """0.6 token/字符 = 用户裁决 2026-10-10 的混合中英启发式（非测量值）。"""
    assert DEFAULT_TOKENS_PER_CHAR == pytest.approx(0.6)


@pytest.mark.parametrize("value", ["0.25", 0.25])
def test_coefficient_configurable_from_registry(iso_dispatch, value):
    set_coefficient(iso_dispatch, value)
    assert Registry.load().est_tokens_per_char == pytest.approx(0.25)


@pytest.mark.parametrize("value", [None, 0, -1, "", "abc", {}, []])
def test_coefficient_fail_open_falls_back_to_default(iso_dispatch, value):
    """缺失/非数/非正 → 默认。系数失准只影响估算，绝不该拖垮派发。"""
    set_coefficient(iso_dispatch, value)
    assert Registry.load().est_tokens_per_char == pytest.approx(DEFAULT_TOKENS_PER_CHAR)


# ---------------------------------------------------------------------------
# 测量面：字符数（非字节数），fail-open
# ---------------------------------------------------------------------------
def test_session_chars_counts_characters_not_bytes(iso_dispatch):
    """中文按字符计（实测依据：jsonl 里中文以原生 UTF-8 落盘，不被 \\uXXXX 转义膨胀）。"""
    m = Registry.load().member("quotamember")
    grow_session(iso_dispatch, SID, "中" * 3 + "abc")  # 6 字符 / 12 字节
    assert session_tools.session_chars(m, SID) == 6


def test_session_chars_missing_file_is_zero(iso_dispatch):
    """文件不存在（新会话首跳前）→ 0 且不抛（fail-open 红线）。"""
    m = Registry.load().member("quotamember")
    assert session_tools.session_chars(m, "never-created") == 0


# ---------------------------------------------------------------------------
# do_dispatch：双端测量 → 台账 + outcome + 会话水位
# ---------------------------------------------------------------------------
def test_hop_records_delta_and_est(iso_dispatch, monkeypatch):
    o = dispatch_one(iso_dispatch, monkeypatch, "x" * 100)
    assert o.kind == "result"
    e = ledger_rows(iso_dispatch)[-1]
    assert e.delta_chars == 100, "落盘的是**未换算原值**（改系数可重算历史）"
    assert e.est_tokens == 60
    assert (o.delta_chars, o.est_tokens) == (100, 60)


def test_est_follows_configured_coefficient(iso_dispatch, monkeypatch):
    set_coefficient(iso_dispatch, 1.0)  # 系数=1 → est 与字符增量同值，断言最直白
    o = dispatch_one(iso_dispatch, monkeypatch, "y" * 40)
    assert ledger_rows(iso_dispatch)[-1].est_tokens == 40 == o.est_tokens


def test_preexisting_session_does_not_inherit_history(iso_dispatch, monkeypatch):
    """功能上线前已存在的会话：无记账水位 → 以本跳起点实测为准，不背历史包袱。"""
    seed_session(iso_dispatch)
    grow_session(iso_dispatch, SID, "h" * 400)  # 旧内容（派发前已存在）
    fake_headless(monkeypatch, ok_after_growth(iso_dispatch, chunk="n" * 10))
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    e = ledger_rows(iso_dispatch)[-1]
    assert (e.delta_chars, e.est_tokens) == (10, 6)


def test_offset_continuity_across_two_hops(iso_dispatch, monkeypatch):
    """第二跳只计本跳增量；水位与跳数随派发前进（跨进程基准必须落 registry）。"""
    enable_registry_save(monkeypatch, iso_dispatch)
    dispatch_one(iso_dispatch, monkeypatch, "a" * 100)
    sess = read_registry(iso_dispatch)["members"]["quotamember"]["sessions"][0]
    assert sess["chars_offset"] == 100
    fake_headless(monkeypatch, ok_after_growth(iso_dispatch, chunk="b" * 50))
    dispatch.do_dispatch("quotamember", text="继续", quiet=True)
    e = ledger_rows(iso_dispatch)[-1]
    assert e.delta_chars == 50, "不重复计入上一跳的 100"
    sess = read_registry(iso_dispatch)["members"]["quotamember"]["sessions"][0]
    assert sess["chars_offset"] == 150 and sess["hops"] == 2


def test_unaccounted_growth_charged_to_next_hop(iso_dispatch, monkeypatch):
    """崩溃跳/手动 resume 的「记账外增长」计入同会话下一跳——总量守恒才不扭曲排名。"""
    seed_session(iso_dispatch, chars_offset=40)  # 上一跳结束水位 40
    grow_session(iso_dispatch, SID, "c" * 60)  # 其中 20 字符从未被 orch 记过账
    fake_headless(monkeypatch, ok_after_growth(iso_dispatch, chunk=""))
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert ledger_rows(iso_dispatch)[-1].delta_chars == 20


def test_rewritten_session_file_clamps_not_negative(iso_dispatch, monkeypatch):
    """水位高于实测起点=文件被重写（压缩/清理）→ 以实测起点为准，增量不为负。"""
    seed_session(iso_dispatch, chars_offset=5000)
    grow_session(iso_dispatch, SID, "d" * 100)  # 实际文件只有 100 字符
    fake_headless(monkeypatch, ok_after_growth(iso_dispatch, chunk="e" * 20))
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    e = ledger_rows(iso_dispatch)[-1]
    assert (e.delta_chars, e.est_tokens) == (20, 12)


def test_failed_hop_still_measured(iso_dispatch, monkeypatch):
    """运行失败跳同样计量（它真烧了 token），水位照记。"""
    enable_registry_save(monkeypatch, iso_dispatch)
    seed_session(iso_dispatch)

    def responder(n, cmd):
        grow_session(iso_dispatch, SID, "f" * 30)
        return envelope("炸了")

    fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_retry=True)
    assert o.kind == "run_failed"
    e = ledger_rows(iso_dispatch)[-1]
    assert (e.kind, e.delta_chars, e.est_tokens) == ("run_failed", 30, 18)
    sess = read_registry(iso_dispatch)["members"]["quotamember"]["sessions"][0]
    assert sess["chars_offset"] == 30


def test_internal_retry_growth_counted_in_one_row(iso_dispatch, monkeypatch):
    """一次派发内部的重试只写一行台账 → 两跳增长合并计入该行。"""
    seed_session(iso_dispatch)
    calls = []

    def run(cmd, cwd=None, env=None, timeout_s=None):
        calls.append(cmd)
        grow_session(iso_dispatch, SID, "g" * 25)
        if len(calls) == 1:
            return envelope("首跑异常")
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    monkeypatch.setattr(dispatch, "run_headless", run)
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    rows = ledger_rows(iso_dispatch)
    assert len(rows) == 1 and len(calls) == 2
    assert rows[0].delta_chars == 50, "基准取进派发时的水位，合并本派发内全部增长"


def test_no_growth_means_zero_est(iso_dispatch, monkeypatch):
    """假 headless 不落会话文件 → 0/0（既有测试与旧行为语义不变）。"""
    seed_session(iso_dispatch)
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert (o.delta_chars, o.est_tokens) == (0, 0)


def test_success_report_shows_est(iso_dispatch, monkeypatch, capsys):
    """组长侧直接看见本跳用量（credits 恒 0 的档位只有这把尺子可比）。"""
    seed_session(iso_dispatch)
    fake_headless(monkeypatch, ok_after_growth(iso_dispatch, chunk="i" * 100))
    dispatch.do_dispatch("quotamember", text="出图", quiet=False)
    assert "est≈60 tok" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 旧台账行兼容 + 聚合
# ---------------------------------------------------------------------------
def test_old_lines_without_est_fields(tmp_path, monkeypatch):
    """上线前的行（无这两个键）→ 0，聚合不得炸（fail-open）。"""
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        json.dumps({"ts": "t", "member": "figure", "session_id": "s", "credits": 5})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ledger, "LEDGER_PATH", p)
    entries = ledger.read_all()
    assert (entries[0].delta_chars, entries[0].est_tokens) == (0, 0)
    s = ledger.summarize(entries)
    assert s["est_tokens"] == 0 and s["by_member"]["figure"]["est_tokens_pct"] == 0.0


def _entries(*pairs):
    """构造台账行：pairs = (member, est_tokens)。"""
    return [
        LedgerEntry(
            ts="2026-10-10T00:00:00+08:00",
            member=member,
            session_id=f"s-{i}",
            est_tokens=est,
        )
        for i, (member, est) in enumerate(pairs)
    ]


def test_summarize_aggregates_est_and_coverage():
    s = ledger.summarize(_entries(("figure", 60), ("figure", 40), ("devops", 100)))
    assert s["est_tokens"] == 200
    assert s["est_tokens_metered_hops"] == 3
    assert s["by_member"]["figure"]["est_tokens"] == 100
    assert s["by_member"]["figure"]["est_tokens_pct"] == 50.0
    assert s["by_member"]["devops"]["est_tokens_pct"] == 50.0


def test_summarize_marks_unmeasured_hops_in_coverage():
    """混合行（旧行 est=0）→ 覆盖率如实缩小：占比只在已计量的跳之间有意义。"""
    s = ledger.summarize(_entries(("figure", 0), ("devops", 100)))
    assert (s["hops"], s["est_tokens_metered_hops"]) == (2, 1)
    assert ledger.format_est_tokens(s) == "est≈100 tok（覆盖 1/2 跳）"


def test_format_est_tokens_text_shapes():
    assert ledger.format_est_tokens({"hops": 0}) == "est —"
    full = ledger.summarize(_entries(("devops", 1200)))
    assert ledger.format_est_tokens(full) == "est≈1,200 tok", "全覆盖时不挂覆盖率噪声"


# ---------------------------------------------------------------------------
# 呈现面：orch ledger / orch stats / 计划完成报告
# ---------------------------------------------------------------------------
def test_orch_ledger_stats_prints_est_share(monkeypatch, capsys):
    entries = _entries(("figure", 60), ("devops", 40))
    monkeypatch.setattr(orch, "read_all", lambda: entries)
    rc = orch.cmd_ledger(Namespace(member=None, days=0, stats=True, tail=10))
    out = capsys.readouterr().out
    assert rc == 0 and "est≈100 tok" in out and "est占比=60.0%" in out


def test_orch_ledger_tail_shows_est_per_hop(monkeypatch, capsys):
    monkeypatch.setattr(
        orch, "read_all", lambda: _entries(("figure", 60), ("devops", 0))
    )
    orch.cmd_ledger(Namespace(member=None, days=0, stats=False, tail=10))
    out = capsys.readouterr().out
    assert "est=60" in out
    assert "est=—" in out, "旧行如实显示未记录，不伪装成零消耗"


def test_workflow_stats_prints_est_share(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(
        workflow, "read_all", lambda: _entries(("devops", 100), ("figure", 0))
    )
    workflow.cmd_stats(Namespace(plan="", member="", days=0))
    out = capsys.readouterr().out
    assert "est≈100 tok（覆盖 1/2 跳）" in out
    assert "devops" in out and "est占比=100.0%" in out


def test_plan_complete_report_prints_est(monkeypatch, capsys):
    entries = _entries(("figure", 100))
    for e in entries:
        e.plan = "p1"
    monkeypatch.setattr(plans, "read_all", lambda: entries)
    plans._report_plan_complete("p1", {})
    out = capsys.readouterr().out
    assert "计划统计" in out and "est≈100 tok" in out and "est占比=100.0%" in out
