"""flash 档额度回退（backlog 20261010-flash-fallback）。

用户裁决 2026-10-10：flash 档走内置免费额度；额度耗尽且 registry models 配置了
``<tier>_fallback``（BYOK modelID，账号相关、渠道待用户开通）时，do_dispatch 自动
换档重试一次；备用渠道也耗尽 → 仍归类 quota_exhausted（drain 全局停止语义不变）。
代码就绪即可，BYOK flash 渠道开通前无法实测线上分支，本组单测覆盖逻辑。
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

BUILTIN = "Qwen3.8-Flash"  # 内置免费档
FB = "byok-flash-id"  # 备用渠道（BYOK modelID 占位）
OK_BODY = "<result>出图完成，commit eee5555 已合并。</result>"


def test_default_models_new_channel_strategy():
    """出厂默认对齐新渠道策略：max=''（走用户级默认 BYOK，禁内置烧 credit）、
    flash=内置免费、flash_fallback=''（无默认回退，实际值在 registry.json 配置）。"""
    assert registry.DEFAULT_MODELS["max"] == ""
    assert registry.DEFAULT_MODELS["flash"] == "Qwen3.8-Flash"
    assert registry.DEFAULT_MODELS["flash_fallback"] == ""


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
    assert "切换备用渠道重试一次" in out
    assert "额度类失败（quota_exhausted）" in out, "备用渠道也耗尽→最终仍报额度面"
