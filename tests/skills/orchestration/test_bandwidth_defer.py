"""带宽类延期不停链 + 兜底回升 + 浅回退 + 预防面（backlog 20261013-193500-leader）。

原 #115 拆单 2/2 的**语义半**：组长裁决把 `channel_error` 的全局熔断授权**仅对 bandwidth
类**反转，并附兜底闭环。三支必须同时成立才叫「反转了熔断」而不是「拆了熔断」：

- `bandwidth` → **延期续跑**（跑通即止，不换席、不停链）；
- **兜底**：延期预算（次数 / 壁钟）触顶仍不愈 → **回升为全局停止**（`channel_saturated`）——
  封死「分类器误判/文案漂移导致逐项烧跳」（2026-10-12 连烧四单的形态）；
- `outage` → **维持原全局停止**（一字不动）。

对应 success_criteria 把 `test_drain_channel_error_stops_globally` 拆成三例。
"""

from __future__ import annotations

import json
import time

from pysci.skills.orchestration.tools import (
    backpressure,
    dispatch,
    drain,
    mirror,
    workflow,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog

MIMO_TEXT = "Custom model service error, please retry later or switch to another model."
OUTAGE_TEXT = "401 Unauthorized: invalid api key"


def _outcome(kind: str, *, error_text: str) -> DispatchOutcome:
    return DispatchOutcome(
        code=2,
        kind=kind,
        member="devops",
        body=error_text,
        error=error_text,
        error_text=error_text,
    )


def _two_items():
    return [
        {"id": "b1", "status": "pending", "summary": "s1", "member": "devops"},
        {"id": "b2", "status": "pending", "summary": "s2", "member": "devops"},
    ]


def _run(iso_state, monkeypatch, outcome, items=None):
    seed_backlog(iso_state, items or _two_items())
    calls: list[str] = []

    def fake_dispatch(member, **kw):
        calls.append(kw.get("slug"))
        return outcome

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    monkeypatch.setattr(drain, "_sleep", lambda s: None)
    # 「全队皆在延期」那一支会睡到最近到期再查；睡被垫成 no-op 后时钟不动，
    # 直接跑就成了死循环。这里让它在**第二次**查询时报「已到期」，等价于时间推进。
    seen = {"n": 0}

    def _wait() -> float:
        seen["n"] += 1
        return 300.0 if seen["n"] == 1 else 0.0

    monkeypatch.setattr(workflow, "next_defer_s", _wait)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    return summary, calls, run_log


# ---------------------------------------------------------------------------
# ① 三支分流
# ---------------------------------------------------------------------------
def test_bandwidth_defers_and_keeps_going(iso_state, monkeypatch):
    """bandwidth → 延期续跑：**不停链**，继续吃下一项。"""
    summary, calls, run_log = _run(
        iso_state, monkeypatch, _outcome("channel_error", error_text=MIMO_TEXT)
    )

    assert not summary.get("stopped"), "带宽类不许停链（熔断授权已反转）"
    assert calls == ["b1", "b2"], "延期后要接着吃下一项（不占槽）"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["b1"]["status"] == "pending" and by_id["b1"]["defer_count"] == 1
    assert by_id["b1"]["defer_until"] > 0, "回队时要挂上到期时刻"
    text = run_log.read_text(encoding="utf-8")
    assert "带宽延期" in text and "不停链" in text


def test_outage_still_stops_globally(iso_state, monkeypatch):
    """outage → **维持原全局停止**（原断言保号：种子复位、剩余不烧）。"""
    summary, calls, run_log = _run(
        iso_state, monkeypatch, _outcome("channel_error", error_text=OUTAGE_TEXT)
    )

    assert summary["stopped"] == "channel_error"
    assert "渠道级故障" in summary["error"]
    assert calls == ["b1"], "渠道真不可用 → 剩余条目不许再烧"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["b1"]["status"] == "pending" and "defer_until" not in by_id["b1"]
    assert "[C] 全局停止" in run_log.read_text(encoding="utf-8")


def test_budget_exhausted_promotes_to_channel_saturated(iso_state, monkeypatch):
    """**兜底**：延期预算触顶仍不愈 → 回升为全局停止（channel_saturated）。"""
    items = _two_items()
    items[0]["defer_count"] = backpressure.DEFER_MAX_ATTEMPTS  # 次数闸已满
    summary, calls, run_log = _run(
        iso_state, monkeypatch, _outcome("channel_error", error_text=MIMO_TEXT), items
    )

    assert summary["stopped"] == "channel_saturated", "延期是让步，不是放弃熔断"
    assert calls == ["b1"], "触顶后立刻停链，不许再吃下一项"
    text = run_log.read_text(encoding="utf-8")
    assert "回升全局停止" in text and "[C]" in text
    assert "带宽类延期" in summary["error"], (
        "报告面要能一眼看出是带宽触顶，而不是笼统的渠道不可用"
    )


def test_defer_budget_accumulates_across_hops(iso_state, monkeypatch):
    """第二次延期把**两次等待**都记进 `defer_elapsed_s`（壁钟闸据此判顶）。"""
    items = _two_items()
    items[0]["defer_count"] = 1
    items[0]["defer_elapsed_s"] = 120.0
    _run(iso_state, monkeypatch, _outcome("channel_error", error_text=MIMO_TEXT), items)

    item = {i["id"]: i for i in read_backlog(iso_state)}["b1"]
    assert item["defer_count"] == 2
    assert item["defer_elapsed_s"] == 120.0 + 240.0, "累计＝上次已耗 + 本次退避"


def test_unknown_takes_outage_branch_when_nothing_higher_is_running(
    iso_state, monkeypatch
):
    """unknown 且**无更高优先级活会话** → 按 outage（保守：停链）。"""
    summary, calls, _ = _run(
        iso_state,
        monkeypatch,
        _outcome("channel_error", error_text="some unrecognised provider trouble"),
    )
    assert summary["stopped"] == "channel_error" and calls == ["b1"]


def test_quota_is_untouched(iso_state, monkeypatch):
    """额度类不在本单分流面内：仍走原全局停止。"""
    summary, calls, _ = _run(
        iso_state,
        monkeypatch,
        _outcome(
            "quota_exhausted", error_text="You've reached your credit usage limit"
        ),
    )
    assert summary["stopped"] == "quota_exhausted" and calls == ["b1"]


# ---------------------------------------------------------------------------
# ② 观察面：mirror 看得到「有条目在延期」
# ---------------------------------------------------------------------------
def test_mirror_line_reports_deferred_items(iso_state):
    seed_backlog(
        iso_state,
        [
            {
                "id": "d1",
                "status": "pending",
                "summary": "s",
                "member": "devops",
                "defer_until": time.time() + 300,
                "defer_count": 2,
            }
        ],
    )
    line = mirror._line_defer()
    assert "带宽延期" in line and "1" in line


def _no_same_seat_retry(state) -> None:
    """关掉同席重试，让「浅回退」成为首败后的第一动作（重试路径另有专测）。"""
    reg = state / "registry.json"
    data = json.loads(reg.read_text(encoding="utf-8"))
    data.setdefault("model_fallback", {})["same_seat_retry"] = {
        "channel_error": 0,
        "backoff_s": 0,
        "budget_s": 0,
    }
    reg.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# ③ 浅回退（隐性 DeepSeek 态下的带宽失败先回预设主力席）
# ---------------------------------------------------------------------------
DS_UUID = "1a54cc57-e614-4521-80cc-447de31dc896"
DS_BUILTIN = "DeepSeek-Flash"
MIMO_MAIN = "seat-mimo-main"
BUILTIN = "Qwen3.8-Flash"


def test_shallow_fallback_returns_to_preset_main_seat(iso_dispatch, monkeypatch):
    """隐性态带宽失败 → **先浅回退一拍**到本预设主力席（不是直接换席）。"""
    from .conftest import (
        VALLEY_MOMENT,
        cmd_model,
        envelope,
        fake_headless,
        set_models,
        set_presets,
        set_routing_clock,
    )

    set_presets(
        iso_dispatch,
        mimo={"main": [MIMO_MAIN, BUILTIN], "pro": ["seat-mimo-pro", BUILTIN]},
    )
    set_models(iso_dispatch, flash=[MIMO_MAIN])
    _no_same_seat_retry(iso_dispatch)
    set_routing_clock(monkeypatch, VALLEY_MOMENT)  # 谷时：main 链被隐性覆盖成 DeepSeek

    def responder(n, cmd):
        if n == 1:
            # 首跳落在隐性 DeepSeek 席，报带宽/渠道故障
            from pysci.skills.orchestration.tools import sessions

            from .conftest import append_turn, sid_from_argv

            append_turn(
                iso_dispatch,
                sid_from_argv(cmd),
                model=sessions.SYNTHETIC_MODEL,
                text=MIMO_TEXT,
            )
            return envelope(MIMO_TEXT)
        return envelope("<result>完工</result>", is_error=False, stop="end_turn", rc=0)

    calls = fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result", o.error
    assert cmd_model(calls[0]) == DS_UUID, "首跳应在隐性 DeepSeek 席上"
    assert cmd_model(calls[1]) == MIMO_MAIN, (
        "浅回退落点＝本预设主力席（不是链上下一席）"
    )


def test_shallow_fallback_then_defer_when_preset_seat_also_fails(
    iso_dispatch, monkeypatch
):
    """**双爆**：浅回退席也爆 → 仍归 channel_error，交由 drain 判成带宽类走延期。"""
    from pysci.skills.orchestration.tools import sessions

    from .conftest import (
        VALLEY_MOMENT,
        append_turn,
        cmd_model,
        envelope,
        fake_headless,
        set_models,
        set_presets,
        set_routing_clock,
        sid_from_argv,
    )

    set_presets(
        iso_dispatch,
        mimo={"main": [MIMO_MAIN, BUILTIN], "pro": ["seat-mimo-pro", BUILTIN]},
    )
    set_models(iso_dispatch, flash=[MIMO_MAIN])
    _no_same_seat_retry(iso_dispatch)
    set_routing_clock(monkeypatch, VALLEY_MOMENT)

    def responder(n, cmd):
        append_turn(
            iso_dispatch,
            sid_from_argv(cmd),
            model=sessions.SYNTHETIC_MODEL,
            text=MIMO_TEXT,
        )
        return envelope(MIMO_TEXT)

    calls = fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "channel_error"
    assert cmd_model(calls[0]) == DS_UUID and cmd_model(calls[1]) == MIMO_MAIN
    assert o.error_text and "带宽" not in o.error_text, "原文照实落，分类交由分类器"
