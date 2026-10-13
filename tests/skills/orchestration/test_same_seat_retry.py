"""channel_error 同席重试（种子 backlog 20261013-143000-leader，用户裁决 2026-10-13）

旧策略（backlog 20261012-194500-leader）把「席位不可用一律不重试」写死在
``test_channel_error.py``：渠道故障首败即换席。实测推翻了它——Mimo 平台侧算力不足造成
**间歇** ``channel_error``（2026-10-13 一日 4 起），这类中断是瞬态的，同会话 resume
续跑大概率即愈且零工作损失（会话逐回合持久化）；而换席落到的是 DeepSeek 付费席。
故付费席降为后手：先原席自救，双闸都过才换席。

本组钉住规格四条：

1. 首败 → 同席 resume 重试（**不动席位**），重试成功即续跑交付；
2. ``quota_exhausted`` 不进重试循环——确定性耗尽重试只是白烧；
3. 重试**不写** ``note_fallback``（重试不是降级），但次数/耗时进台账
   ``same_seat_retries``/``same_seat_retry_ms`` 与 run 日志，抖动频率才可统计；
4. 双闸（``channel_error`` 次数 + ``budget_s`` 壁钟预算）耗尽 → 换席，次数闸挡不住
   「真下线」场景（每次失败耗时数分钟，10 次拖住串行队列近小时）。

退避与预算走 :func:`dispatch._sleep` / :func:`dispatch._clock` 两个可替换落点：等待
45s/300s 不能在用例里真发生，但「等了多久、等了几次」必须被断言。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch, drain, registry

from .conftest import (
    CHANNEL_TEXT,
    QUOTA_TEXT,
    append_turn,
    cmd_model,
    envelope,
    fake_headless,
    read_runtime,
    seed_backlog,
    set_models,
    sid_from_argv,
)

A = "seat-a"
B = "seat-b"
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"


def _broken(state):
    """渠道级故障响应器：写末轮合成故障形状 + 返回该文案 envelope（#77 判据支）。"""

    def responder(n, cmd):
        append_turn(
            state,
            sid_from_argv(cmd),
            model="<synthetic>",
            text=CHANNEL_TEXT,
            stop_reason="stop_sequence",
            stop_sequence="",
        )
        return envelope(CHANNEL_TEXT)

    return responder


def _last_ledger_row(state):
    lines = (state / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def _set_retry(state, **over):
    """写 ``model_fallback.same_seat_retry``（只写被测键，其余走出厂缺省）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data.setdefault("model_fallback", {})["same_seat_retry"] = over
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _no_backoff(monkeypatch):
    """退避改成 no-op 并记录等待秒数——用例不真睡，但等了多久要能断言。"""
    slept: list[float] = []
    monkeypatch.setattr(dispatch, "_sleep", lambda s: slept.append(s))
    return slept


def _fake_clock(monkeypatch, step: float):
    """单调时钟替身：每读一次前进 ``step`` 秒（预算闸要能被造出来触顶）。"""
    box = {"t": 0.0}

    def tick() -> float:
        box["t"] += step
        return box["t"]

    monkeypatch.setattr(dispatch, "_clock", tick)
    return box


# ---------------------------------------------------------------------------
# ① 首败 → 同席重试成功（不动席位、不进回退事件面、不进粘滞窗）
# ---------------------------------------------------------------------------
def test_channel_error_retries_same_seat_then_recovers(
    iso_state, iso_dispatch, monkeypatch
):
    slept = _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=3, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])

    def responder(n, cmd):
        if n == 1:
            return _broken(iso_dispatch)(n, cmd)
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    calls = fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result" and o.code == 0, "重试跑通即正常交付，不是带伤交付"
    assert len(calls) == 2, "首败 + 一次同席重试"
    assert all(cmd_model(c) == A for c in calls), "重试必须停在原席，不许偷换席位"
    assert o.same_seat_retries == 1, "次数进 outcome 供 run 日志读数"
    assert slept == [], "backoff_s=0 时不产生等待"
    assert read_runtime(iso_dispatch).get(registry.RUNTIME_SEAT_FAIL_KEY, {}) == {}, (
        "重试不是降级：不许把原席写进粘滞窗"
    )


def test_backoff_is_honoured_between_retries(iso_state, iso_dispatch, monkeypatch):
    """``backoff_s`` 真的会被等（退避不是装饰）：每次重试前等一次，等的秒数对得上。"""
    slept = _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=2, backoff_s=45, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _broken(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "channel_error"
    assert o.same_seat_retries == 2
    assert slept == [45, 45], "两次重试各退避一次"


# ---------------------------------------------------------------------------
# ② quota 不重试 / 重试耗尽后换席 / 预算闸先于次数闸
# ---------------------------------------------------------------------------
def test_quota_exhaustion_never_enters_retry_loop(iso_state, iso_dispatch, monkeypatch):
    """额度是确定性耗尽：重试只是再烧一跳，直接走换席通路。"""
    _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=10, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "quota_exhausted"
    assert len(calls) == 2, "1 首跳 + 1 换席；重试计数必须是 0"
    assert cmd_model(calls[1]) == B
    assert o.same_seat_retries == 0, "额度耗尽不得计入同席重试读数"


def test_retries_exhausted_then_falls_to_next_seat(
    iso_state, iso_dispatch, monkeypatch
):
    """次数闸耗尽（渠道一直坏）→ 才换席；重试次数与回退事件两账分开。"""
    _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=2, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _broken(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "channel_error"
    assert len(calls) == 4, "首跳 + 2 次同席重试 + 1 次换席"
    assert [cmd_model(c) for c in calls] == [A, A, A, B], "前 3 跳都必须停在原席"
    assert o.same_seat_retries == 2
    # ``to=""`` 是既有语义（末席仍坏 = 全链耗尽，#77 的落账），与本单无关；
    # 这里只断言**没有因重试而多出**换席事件——重试次数 2 已由上面的 argv 序列证明。
    assert [e["to"] for e in o.fallbacks] == [B, ""], "重试不写事件，只有真实换席写"


def test_wall_clock_budget_stops_retries_before_count_limit(
    iso_state, iso_dispatch, monkeypatch
):
    """预算闸先于次数闸触顶：「真下线」时每次失败都耗时数分钟，不能拖住串行队列。"""
    _no_backoff(monkeypatch)
    _fake_clock(monkeypatch, step=200.0)
    _set_retry(iso_dispatch, channel_error=10, backoff_s=0, budget_s=250)
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _broken(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.same_seat_retries == 1, "预算 250s、每跳 200s → 只够一次重试"
    assert [cmd_model(c) for c in calls] == [A, A, B], "触顶后立即换席，不再原席死等"
    assert len(calls) == 3


def test_zero_budget_disables_the_wall_clock_gate(iso_state, iso_dispatch, monkeypatch):
    """``budget_s=0`` = 不设限（与 ``skip_window_s=0`` 同口径）：次数闸独自把关。"""
    _no_backoff(monkeypatch)
    _fake_clock(monkeypatch, step=9999.0)
    _set_retry(iso_dispatch, channel_error=2, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _broken(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.same_seat_retries == 2, "关掉预算闸后必须跑满次数闸"
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# ③ 观察面：台账字段 + run 日志
# ---------------------------------------------------------------------------
def test_retry_counts_land_in_ledger(iso_state, iso_dispatch, monkeypatch):
    _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=2, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    fake_headless(monkeypatch, _broken(iso_dispatch))
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    row = _last_ledger_row(iso_dispatch)
    assert row["same_seat_retries"] == 2, "重试次数必须落台账，抖动频率才有统计原料"
    assert row["same_seat_retry_ms"] >= 0
    assert row["hold_s"] == 0, "没避让就记 0，不许留空串或缺字段"


def test_quota_path_records_zero_retries_in_ledger(
    iso_state, iso_dispatch, monkeypatch
):
    """额度路径的台账行也必须带这几个字段（旧行缺字段会让统计面误读成 0 覆盖）。"""
    _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=10, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    row = _last_ledger_row(iso_dispatch)
    assert row["kind"] == "quota_exhausted"
    assert row["same_seat_retries"] == 0, "额度路径不得虚报重试"


def test_drain_run_log_names_same_seat_retry(iso_state, iso_dispatch, monkeypatch):
    """run 日志要看得见「重试过几次」——否则抖动与真死在批次账上分不开。"""
    _no_backoff(monkeypatch)
    _set_retry(iso_dispatch, channel_error=2, backoff_s=0, budget_s=0)
    set_models(iso_dispatch, flash=[A, B])
    seed_backlog(
        iso_state,
        [{"id": "r1", "status": "pending", "summary": "s1", "member": "quotamember"}],
    )
    fake_headless(monkeypatch, _broken(iso_dispatch))
    run_log = drain.new_run_log()
    drain.drain_backlog(run_log)

    text = run_log.read_text(encoding="utf-8")
    assert "同席重试" in text, "run 日志必须点名重试，否则批次账上看不出渠道在抖"
