"""额度类失败的**换席**通路（backlog 20261010-flash-fallback → 20261012-211500-leader 链化）。

机制与档位无关：当前席不可用时沿该档的**有序回退链**一席一跳换下去。本组用例写的是链化
**之后**的实现（``dispatch`` 不再读 ``<tier>_fallback`` 单值，改读 ``model_chain(tier)``），
但断言面刻意保持**旧写形态**（``flash`` 单值 + ``flash_fallback`` 键）——那是兼容读的判据：
老 registry 不改一个字节，链化后照样能换席。真链形态的用例见 ``test_model_fallback_chain.py``。

全链耗尽 → 仍归类原系统性 kind（quota_exhausted/channel_error），drain 全局停止语义不变
（#77）。告警文案单一口径 ``[!] 档位回退：<tier> 席<名>不可用 → 已切换 <新席>（原因 …）``
由 :mod:`channels` 出，观察面（事件流）同源。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import dispatch, registry

from .conftest import (
    QUOTA_TEXT,
    cmd_model,
    envelope,
    fake_headless,
    set_models,
)

BUILTIN = "byok-flash-main"  # 主席样本（BYOK 形态——内置席已被执行面硬过滤，样本不得再用内置名）
FB = "byok-flash-id"  # 备用渠道（BYOK modelID 占位）
OK_BODY = "<result>出图完成，commit eee5555 已合并。</result>"


def test_default_models_new_channel_strategy():
    """出厂默认（三级回退的最后一级）：main/pro 空串（席位随预设，真值在跟踪 presets 块
    与代码出厂表，出厂层不猜账号级 UUID/人名）、minimum=内置免费。**出厂值保守取内置名是
    有意为之**——BYOK modelID 是账号级 UUID，不写进代码。"""
    assert registry.DEFAULT_MODELS["main"] == ""
    assert registry.DEFAULT_MODELS["pro"] == ""
    assert registry.DEFAULT_MODELS["minimum"] == "Qwen3.8-Flash"
    assert set(registry.DEFAULT_MODELS) == {"main", "pro", "minimum"}, "档位键已退役"


def test_quota_fallback_switches_channel_and_succeeds(iso_dispatch, monkeypatch):
    set_models(iso_dispatch, max="", flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n == 1
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)
    assert o.kind == "result" and o.code == 0
    assert len(calls) == 2
    assert cmd_model(calls[0]) == BUILTIN
    assert cmd_model(calls[1]) == FB, "第二跳必须换到备用渠道"
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert '"result"' in line and FB in line, "台账记实际所用渠道"


def test_quota_fallback_also_quota(iso_dispatch, monkeypatch):
    """备用渠道也耗尽 → quota_exhausted（drain 全局停止语义不变），不再三跳。"""
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "quota_exhausted"
    assert len(calls) == 2
    assert cmd_model(calls[1]) == FB


def test_quota_without_fallback_single_call(iso_dispatch, monkeypatch):
    """fallback 为空（渠道未开通）→ 保持单跳 quota_exhausted，不浪费重试。"""
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback="")
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "quota_exhausted" and len(calls) == 1


def test_quota_fallback_null_tolerated(iso_dispatch, monkeypatch):
    """registry 里 flash_fallback 为 JSON null（任务书『现值 null』）→ 容忍不回退。"""
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=None)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "quota_exhausted" and len(calls) == 1


def test_fallback_same_model_no_wasted_retry(iso_dispatch, monkeypatch):
    """fallback 与当前渠道同型号 → 换档无意义，不重试。"""
    set_models(iso_dispatch, flash=FB, flash_fallback=FB)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "quota_exhausted" and len(calls) == 1


def test_no_retry_suppresses_fallback(iso_dispatch, monkeypatch):
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_retry=True)
    assert o.kind == "quota_exhausted" and len(calls) == 1


def test_generic_failure_never_uses_fallback(iso_dispatch, monkeypatch):
    """非额度失败仍走同渠道重试一次，绝不换档。"""
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope("internal boom"))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "run_failed" and len(calls) == 2
    assert all(cmd_model(c) == BUILTIN for c in calls)


def test_tier_without_fallback_key(iso_dispatch, monkeypatch):
    """回退按 <tier>_fallback 通用命名：max 档未配置 max_fallback → 不回退。"""
    set_models(iso_dispatch, max="max-id", flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="重活", quiet=True, model_tier="max")
    assert o.kind == "quota_exhausted" and len(calls) == 1


def test_fallback_switch_printed_and_final_report(iso_dispatch, monkeypatch, capsys):
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    dispatch.do_dispatch("quotamember", text="出图", quiet=False)
    out = capsys.readouterr().out
    assert "[!] 档位回退：main 席" in out and "已切换" in out, (
        "换席必须在调用者面留一行告警（用户裁决 2026-10-12 21:1x 规格 4a）"
    )
    assert FB in out and BUILTIN in out, "告警要点名原席与新席"
    assert "额度类失败（quota_exhausted）" in out, "备用渠道也耗尽→最终仍报额度面"
