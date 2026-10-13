"""会话 ctx 观测面 + fe68a40 水位闸回退（用户 2026-10-13 05:5x 审查裁决，backlog 20261013-060000-leader）。

fe68a40 曾给 ``session="latest"`` 加过 60% 自动换会话闸——经授权链审计确认它绕过了全部
判决通道（策略变更属 J/M/twin 管辖），且压缩防溢出使闸的宣称收益无效，用户异议=回退授权。
**保留**其良性附带：``dispatch.touch_session`` 把末跳 ``context_usage_ratio`` 落进会话池
（纯观测，有单测钉住）；**回退**主体：``session="latest"`` 恢复「续最近活跃会话」，哪怕
末跳 ctx 已过 60% 也不断续（重启类似机制须经判决包 + 真实账单/缓存读数）。

量的是**真实读路**：env 的 ctx 嵌在 ``usage`` 块（runner.Envelope.parse 唯一取处），
``fake_headless`` 桩只管派生、不碰判据。``iso_dispatch`` 夹具 no-op 了 ``Registry.save``
（防写真实 registry.json），故跨跳的水位不靠「跑一跳自然累计」——直接写
``sessions-runtime.json``（``write_runtime``）预置，读的是磁盘面，不被 no-op 骗过。
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
# ② 回退实测：水位高低都不拦 latest 续（fe68a40 主体已撤销）
# ---------------------------------------------------------------------------
def test_latest_resumes_below_watermark(iso_dispatch, monkeypatch):
    """末跳 ctx 低 → 续旧会话（健康路径语义不变）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.2))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    # 夹具 no-op save → 水位不自然累计，直接预置一具低水位会话（读磁盘面）
    _seed_session(iso_dispatch, o1.sid, ctx=0.2, last_active="9999")
    fake_headless(monkeypatch, _ok(0.3))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid == o1.sid, "低水位应续旧会话"


def test_latest_resumes_even_past_60pct(iso_dispatch, monkeypatch):
    """回退实测（success_criteria）：末跳 ctx 65%（含 87% 高位）→ **仍续**原会话。

    fe68a40 的 60% 自动换会话属策略变更、绕过了判决通道，用户审查裁决回退。
    """
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.65))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, o1.sid, ctx=0.65, last_active="9999")
    fake_headless(monkeypatch, _ok(0.7))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid == o1.sid, "65% 水位不得断续原会话（闸已回退）"

    _seed_session(iso_dispatch, o1.sid, ctx=0.87, last_active="9999")
    o3 = dispatch.do_dispatch(MEMBER, text="再续", quiet=True, no_checks=True)
    assert o3.sid == o1.sid, "87% 同样续（回退=恢复无条件续最近活跃会话）"


def test_session_new_still_starts_fresh(iso_dispatch, monkeypatch):
    """显式 session="new" 每次新起（与水位无关，语义不因回退而变）。"""
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
    """无 ctx 记录（首次/崩溃跳）照续：观测面缺失不该影响选择。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.0))
    o1 = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, o1.sid, ctx=None, last_active="9999")
    fake_headless(monkeypatch, _ok(0.0))
    o2 = dispatch.do_dispatch(MEMBER, text="续跑", quiet=True, no_checks=True)
    assert o2.sid == o1.sid


def test_no_gate_message_on_past_watermark(iso_dispatch, monkeypatch, capsys):
    """回退后不再有「起新会话」闸播报——静默/显式切换都该与 fe68a40 前一致。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, _ok(0.87))
    first = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="new"
    )
    _seed_session(iso_dispatch, first.sid, ctx=0.87, last_active="9999")
    fake_headless(monkeypatch, _ok(0.1))
    dispatch.do_dispatch(MEMBER, text="续跑", quiet=False, no_checks=True)
    out = capsys.readouterr().out
    assert "压缩悬崖前兆" not in out and "起新会话" not in out, (
        f"水位闸播报已随闸回退，实际仍出现：{out[-400:]}"
    )
