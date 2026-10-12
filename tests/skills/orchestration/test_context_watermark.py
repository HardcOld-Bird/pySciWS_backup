"""会话上下文水位闸（backlog insp-20261012-224853-mechanical / 020358-mechanical-002）。

根因（两案同因）：``session="latest"`` 无条件续上一跳的会话，而 ctx 只在台账里事后可见，
运行态会话池从未记过它。于是会话一路爬到压缩悬崖（devops 实测 81%→87%）后接 run_failed /
rc=1 腰斩，inspector 再把这读数捡回成案。修复：``dispatch.touch_session`` 把末跳
``context_usage_ratio`` 落进会话池，``session="latest"`` 据 ``FRESH_CTX_RATIO`` 水位决定
续/新——过崖即起新会话（记忆已按 charter 蒸馏进只读层与台账，续作靠任务书不靠会话记忆）。

量的是**真实读路**：env 的 ctx 嵌在 ``usage`` 块（runner.Envelope.parse 唯一取处），
``fake_headless`` 桩只管派生、不碰判据。``iso_dispatch`` 夹具 no-op 了 ``Registry.save``
（防写真实 registry.json），故跨跳的水位不靠「跑一跳自然累计」——直接写
``sessions-runtime.json``（``write_runtime``）预置，闸读的是磁盘面，不被 no-op 骗过。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import dispatch

from .conftest import (
    enable_registry_save,
    envelope,
    fake_headless,
    read_runtime,
    set_models,
    write_runtime,
)

OK = "<result>完工，commit abc1234 已合并。</result>"
BUILTIN = "Qwen3.8-Flash"
MEMBER = "quotamember"


def _seed_session(state, sid: str, *, ctx: float | None, last_active: str) -> None:
    """往运行态会话池预置一具会话（``ctx=None`` 时不写该键，模拟崩溃跳无记录）。"""
    rt = read_runtime(state)
    sess = rt["members"][MEMBER]["sessions"]
    sess[:] = [{"sid": sid, "status": "active", "last_active": last_active, "hops": 1}]
    if ctx is not None:
        sess[0]["ctx_ratio"] = ctx
    write_runtime(state, rt)


def _ok(ctx: float):
    return lambda n, cmd: envelope(OK, is_error=False, rc=0, ctx=ctx)


# ---------------------------------------------------------------------------
# ① 末跳 ctx 落池：success / failure 两支都记
# ---------------------------------------------------------------------------
def test_touch_session_records_ctx_ratio_on_success(iso_dispatch, monkeypatch):
    # 量的是「真落盘的池」——默认夹具 no-op save，这里还原真实 save 才能读回 ctx 水位
    enable_registry_save(monkeypatch)
    fake_headless(monkeypatch, _ok(0.42))
    o = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    assert o.code == 0 and o.kind == "result"
    rt = read_runtime(iso_dispatch)["members"][MEMBER]["sessions"]
    assert rt and rt[0]["ctx_ratio"] == 0.42, "成功跳也要落 ctx 水位"


def test_touch_session_records_ctx_ratio_on_failure(iso_dispatch, monkeypatch):
    """失败跳（run_failed/rc=1 腰斩）同样记 ctx——正是高水位那几跳最容易腰斩。"""
    enable_registry_save(monkeypatch)
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(
            "炸了", is_error=True, stop="max_turns", rc=1, ctx=0.87
        ),
    )
    o = dispatch.do_dispatch(MEMBER, text="出图", quiet=True, no_checks=True)
    assert o.code == 2
    rt = read_runtime(iso_dispatch)["members"][MEMBER]["sessions"]
    assert rt and rt[0]["ctx_ratio"] == 0.87, "失败跳也要落 ctx 水位"


# ---------------------------------------------------------------------------
# ② 水位闸：低于阈值续、过崖起新会话
# ---------------------------------------------------------------------------
def test_latest_resumes_below_watermark(iso_dispatch, monkeypatch):
    """末跳 ctx 未过 FRESH_CTX_RATIO → 照旧续同一会话（不改变健康路径语义）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.2))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    # 夹具 no-op save → 水位不自然累计，直接预置一具低水位会话（闸读磁盘面）
    _seed_session(iso_dispatch, o1.sid, ctx=0.2, last_active="9999")
    fake_headless(monkeypatch, _ok(0.3))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid == o1.sid, "低于水位应续旧会话"


def test_latest_starts_fresh_when_past_watermark(iso_dispatch, monkeypatch):
    """末跳 ctx 已过 FRESH_CTX_RATIO → 不再续，起新会话（记忆已蒸馏进只读层与台账）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.87))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, o1.sid, ctx=0.87, last_active="9999")
    fake_headless(monkeypatch, _ok(0.1))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid != o1.sid, "过崖的会话不该被续"


def test_session_new_ignores_watermark(iso_dispatch, monkeypatch):
    """显式 session="new" 不受闸约束（用户/组长显式要新会话，水位不拦）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.9))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, o1.sid, ctx=0.9, last_active="9999")
    fake_headless(monkeypatch, _ok(0.9))
    o2 = dispatch.do_dispatch(
        MEMBER, text="显式新", quiet=True, no_checks=True, session="new"
    )
    assert o2.sid != o1.sid, "session=new 每次都新起，与水位无关"


def test_no_record_session_still_resumes(iso_dispatch, monkeypatch):
    """无 ctx 记录（首次/崩溃跳）不拦：fail-open 续，别把「没读数」当「过崖」。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.0))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, o1.sid, ctx=None, last_active="9999")
    fake_headless(monkeypatch, _ok(0.0))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid == o1.sid, "无读数=fail-open 续，不误判成过崖"


def test_fresh_gate_is_visible_in_report(iso_dispatch, monkeypatch, capsys):
    """闸触发时报告面要说清「为何起新会话」——静默换会话会让组长以为记忆丢了。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.87))
    first = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, first.sid, ctx=0.87, last_active="9999")
    fake_headless(monkeypatch, _ok(0.1))
    dispatch.do_dispatch(MEMBER, text="续跑", quiet=False, no_checks=True)
    out = capsys.readouterr().out
    assert "起新会话" in out, f"闸要可见，实际输出未提新会话：{out[-400:]}"
