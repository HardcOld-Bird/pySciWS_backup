"""续跑/换档跳的 envelope 读数累计（E2 验收实测缺陷，backlog 20261012-075200-leader）。

实测：`orch bench eval` 的 fig-basic 行记 turns=2，而该会话 transcript 里躺着 50+ 条
assistant 消息——首跳崩了（rc=1），``do_dispatch`` 用 ``resume=True`` 自动重试，而
``-o json`` 的 envelope 只报**本次进程调用**的读数。拿末跳 envelope 记账/判决会把
轮数、耗时、credits 与 pod-guard 拒绝次数一起做低。本组测试钉住整任务读数。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch
from pysci.skills.orchestration.tools.runner import Envelope


def hop(*, result, turns, dur=1000, credits=0.0, denials=0, is_error=False, rc=0):
    """构造一次 run_headless 的 (rc, stdout, stderr)（stdout 为 envelope JSON）。"""
    return (
        rc,
        json.dumps(
            {
                "result": result,
                "is_error": is_error,
                "stop_reason": "stop_sequence",
                "num_turns": turns,
                "duration_ms": dur,
                "total_credits": credits,
                "usage": {"context_usage_ratio": 0.2},
                "permission_denials": [{"tool": "Write"}] * denials,
                "session_id": "aaaa1111",
            }
        ),
        "",
    )


def test_accumulate_hop_sums_counters():
    prev = Envelope.parse(
        hop(result="崩", turns=9, dur=4000, denials=2, is_error=True)[1]
    )
    last = Envelope.parse(hop(result="<result> done </result>", turns=2, dur=1000)[1])
    agg = dispatch._accumulate_hop(last, prev)
    assert agg.num_turns == 11 and agg.duration_ms == 5000
    assert len(agg.permission_denials) == 2, (
        "首跳被 pod-guard 拦的拒绝不得被续跑清零——否则案卷「权限拒绝上限」假通过"
    )
    assert agg.result == last.result and agg.is_error is False, (
        "分类字段取末跳：交付正文与成败只由最后一次调用决定"
    )


def test_accumulate_hop_takes_max_ctx_ratio():
    prev = Envelope(num_turns=1, context_usage_ratio=0.31)
    last = Envelope(num_turns=1, context_usage_ratio=0.28)
    assert dispatch._accumulate_hop(last, prev).context_usage_ratio == 0.31, (
        "ctx_ratio 是会话态（累计占比），不是每跳增量——求和会算出 >1 的荒谬值"
    )


def test_retry_outcome_reports_total_turns(iso_dispatch, monkeypatch):
    """集成面：首跳崩 + 续跑成功交付，outcome/台账都记整任务读数。"""
    calls = []

    def responder(n, cmd):
        calls.append(n)
        if n == 1:
            return hop(
                result="internal boom",
                turns=9,
                dur=40000,
                denials=1,
                is_error=True,
                rc=1,
            )
        return hop(result="<result>已完成，产物在盘。</result>", turns=2, dur=5000)

    monkeypatch.setattr(
        dispatch, "run_headless", lambda cmd, **kw: responder(len(calls) + 1, cmd)
    )
    o = dispatch.do_dispatch("quotamember", text="做某事", quiet=True, no_checks=True)
    assert o.kind == "result" and len(calls) == 2
    assert o.envelope.num_turns == 11, "台账轮数必须是整任务读数，不是续跑那两跳"
    assert o.envelope.duration_ms == 45000
    assert len(o.envelope.permission_denials) == 1
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["num_turns"] == 11, "生产台账同样记整任务读数"
