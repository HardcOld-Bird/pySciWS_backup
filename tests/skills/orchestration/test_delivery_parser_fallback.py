"""交付协议的**解析器版本错位**兜底（backlog 20261012-002857-devops）。

威胁面：交付协议的属性扩展（E0-B 的 ``kind=``、E1 的 ``tags=``）合进 HEAD 之后，**已经
启动**的派发/审批进程里 import 的还是旧解析器——旧正则只认字面 ``<blocked>``，带属性的
交付整块看不见 → ``kind="parse_error"`` → ``dispatch`` code=2 → drain「仅种子 needs_leader」，
组员白跑一跳，失败原因还显示成「交付格式解析失败」而非真问题。

锁住的核心性质：
① 严格面全缺时宽松面**救回**（kind/tags/产物如实解析）并附一行「解析器早于协议」告警；
② 现协议形态**一律不走兜底**（``warning`` 恒空——兜底静默通过才是事故）；
③ 现有 ``parse_error`` 语义不变：两者皆缺、两块并存都照旧（并存**不**救，救它等于把
   协议违规洗成告警）；
④ 告警一路传到组长按得见的三面：``Delivery.warning`` → ``DispatchOutcome.warning``
   → 派发报告打印 + drain run 日志。
"""

from __future__ import annotations

import re

from pysci.skills.orchestration.tools import delivery, dispatch, drain
from pysci.skills.orchestration.tools.delivery import (
    PROTOCOL_LAG_HINT,
    parse_delivery,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import seed_backlog

#: **旧版**协议的严格面（E1 合并前的字面形态，建议书正文 ``git show 636c4a8`` 引的原文）：
#: 只认裸标签、没有 ``attrs`` 组——用它替换 ``_STRICT_BLOCKS`` 即复现「解析器早于协议」。
LEGACY_STRICT = {
    "result": re.compile(r"<result>(?P<body>.*?)</result>", re.DOTALL),
    "blocked": re.compile(r"<blocked>(?P<body>.*?)</blocked>", re.DOTALL),
}


def _legacy_parser(monkeypatch):
    monkeypatch.setattr(delivery, "_STRICT_BLOCKS", dict(LEGACY_STRICT))


# --- ① 兜底救回 -------------------------------------------------------------


def test_attributed_blocked_survives_legacy_parser(monkeypatch):
    """success_criteria 主案：旧严格面看不见的 ``tags=`` 交付仍判 blocked 且标签在。"""
    _legacy_parser(monkeypatch)
    d = parse_delivery('<blocked tags="quota">额度见底</blocked>')
    assert d.kind == "blocked", d
    assert d.tags == ["quota"]
    assert PROTOCOL_LAG_HINT in d.warning


def test_attributed_result_survives_legacy_parser_with_artifacts(monkeypatch):
    """兜底不只救 kind，还要把 result 正文里的产物声明一并抽出（否则机械验收空跑）。"""
    _legacy_parser(monkeypatch)
    d = parse_delivery(
        '<result verdict="pass">正文 '
        '<artifact check="orch-tests" reason="测试">tests/x.py</artifact>'
        "</result>"
    )
    assert d.kind == "result" and d.body.startswith("正文")
    assert [(a.path, a.check) for a in d.artifacts] == [("tests/x.py", "orch-tests")]
    assert PROTOCOL_LAG_HINT in d.warning


def test_single_quoted_and_bare_attrs_rescued(monkeypatch):
    """宽松属性面：单引号与裸值都取值（协议未来换引号形态也不当次报废）。"""
    _legacy_parser(monkeypatch)
    single = parse_delivery("<blocked tags='quota'>x</blocked>")
    bare = parse_delivery("<blocked tags=quota>x</blocked>")
    assert single.tags == ["quota"] and single.kind == "blocked"
    assert bare.tags == ["quota"] and bare.kind == "blocked"


def test_rescue_is_causal_not_incidental(monkeypatch):
    """因果对照：关掉宽松面后同一输入必须退回 ``parse_error``——救回的是兜底，不是运气。

    旧版正则（``636c4a8`` 的 ``<tag>`` 字面形态）对带属性交付整块看不见，这正是事故本体；
    宽松面一撤，交付就应当再次报废。断言反过来（撤掉兜底仍 pass）说明测试脱靶。
    """
    _legacy_parser(monkeypatch)
    monkeypatch.setattr(
        delivery,
        "_LENIENT_BLOCK_RE",
        re.compile(r"<never\b(?P<attrs>[^>]*)>(?P<body>.*?)</never>", re.DOTALL),
    )
    d = parse_delivery('<blocked tags="quota">额度见底</blocked>')
    assert d.kind == "parse_error" and d.tags == [] and d.warning == ""


# --- ② 现协议形态绝不走兜底（告警不许污染正常路径）--------------------------


def test_current_parser_emits_no_warning():
    for text in (
        "<blocked>缺信息</blocked>",
        '<blocked tags="cli_missing,tool_bug">无 sub 命令</blocked>',
        "<result>完成</result>",
        '<result>完成 <artifact check="none" reason="示意">p.png</artifact></result>',
    ):
        d = parse_delivery(text)
        assert d.warning == "", (text, d.warning)


def test_legacy_parser_still_reads_bare_blocks_without_warning(monkeypatch):
    """旧解析器读旧形态：本来就是正常路径，兜底不该插手、也不该告警。"""
    _legacy_parser(monkeypatch)
    d = parse_delivery("<blocked>裸形态</blocked>")
    assert d.kind == "blocked" and d.warning == ""


# --- ③ parse_error 语义不变 --------------------------------------------------


def test_both_blocks_stay_parse_error():
    d = parse_delivery("两个块都在 <result>a</result><blocked>b</blocked>")
    assert d.kind == "parse_error" and d.tags == [] and d.warning == ""


def test_missing_blocks_stay_parse_error():
    d = parse_delivery("什么块都没写")
    assert d.kind == "parse_error" and d.body == "" and d.warning == ""


def test_legacy_parser_does_not_rescue_two_attributed_blocks(monkeypatch):
    """并存是协议违规：宽松面扫到两块也**不救**，照旧 parse_error（不加告警、不定 kind）。"""
    _legacy_parser(monkeypatch)
    d = parse_delivery(
        '<result verdict="pass">a</result><blocked tags="quota">b</blocked>'
    )
    assert d.kind == "parse_error" and d.warning == ""


# --- ④ 告警传到组长看得见的那几面 -------------------------------------------


def test_dispatch_report_prints_warning(capsys):
    o = DispatchOutcome(
        code=0,
        kind="result",
        member="devops",
        body="done",
        warning=PROTOCOL_LAG_HINT,
    )
    dispatch._report_success(o, [])
    out = capsys.readouterr().out
    assert "解析器早于协议" in out


def test_drain_logs_warning_to_run_log(iso_state, monkeypatch):
    """run 日志是 drain 唯一留痕面：兜底救回的每一跳都要在那里看得见（不改变判读结果）。"""

    def fake_dispatch(member, **kw):
        return DispatchOutcome(
            code=0,
            kind="result",
            body="完成，commit abc1234def",
            warning=PROTOCOL_LAG_HINT,
        )

    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    text = run_log.read_text(encoding="utf-8")
    assert summary["items"][0]["status"] == "done"  # 判读不受影响
    assert "解析器早于协议" in text and "[!]" in text


def test_stale_parser_note_survives_into_outcome_field():
    """``Delivery.warning`` 是解析器唯一的对外出口：文案改动会同时打断派发与 drain 断言。"""
    assert "解析器早于协议" in PROTOCOL_LAG_HINT
