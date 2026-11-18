"""出图护栏判定的机械断言（watermark / seed 兼容档 / 成本上限 / 取图方式）。"""

from __future__ import annotations

import pytest

from pysci.skills.image_generation import guardrails


def test_watermark_explicitly_false_by_default():
    """方舟默认加水印 → 护栏恒显式 watermark=False。"""
    payload: dict = {}
    guardrails.apply_watermark(payload)
    assert payload["watermark"] is False
    guardrails.apply_watermark(payload, allow_watermark=True)
    assert payload["watermark"] is True


def test_seed_range_per_provider():
    """seed 兼容档：方舟 [0,65535]、硅基 [0,9999999999]；None 放行。"""
    assert guardrails.check_seed(None, "ark") is None
    assert guardrails.check_seed(777, "ark") == 777
    assert guardrails.check_seed(777, "siliconflow") == 777
    assert guardrails.check_seed(9999999999, "siliconflow") == 9999999999


def test_seed_out_of_range_raises():
    """越界 seed 直接报错，不静默截断。"""
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_seed(65536, "ark")
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_seed(10_000_000_000, "siliconflow")
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_seed(-1, "siliconflow")
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_seed(1.5, "ark")  # type: ignore[arg-type]


def test_unknown_provider_rejected():
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_seed(1, "comfyui")


def test_clamp_count_cost_cap(monkeypatch):
    """成本护栏：默认上限 4，环境变量 IMAGEGEN_MAX_IMAGES 可覆盖。"""
    monkeypatch.delenv("IMAGEGEN_MAX_IMAGES", raising=False)
    assert guardrails.max_images_cap() == 4
    assert guardrails.clamp_count(10) == 4
    assert guardrails.clamp_count(0) == 1
    monkeypatch.setenv("IMAGEGEN_MAX_IMAGES", "2")
    assert guardrails.clamp_count(10) == 2
    monkeypatch.setenv("IMAGEGEN_MAX_IMAGES", "not-a-number")
    assert guardrails.max_images_cap() == 4  # 非法值回退默认


def test_images_budget_ark_sequential_hard_limit():
    """方舟组图硬上限：参考+产出 ≤ 15（超出报错；未超时不受成本上限 clamp）。"""
    assert (
        guardrails.check_images_budget(3, provider="ark", n_refs=2, sequential=True) == 3
    )
    with pytest.raises(guardrails.GuardrailError):
        guardrails.check_images_budget(15, provider="ark", n_refs=1, sequential=True)
    # 非组图走成本上限
    assert guardrails.check_images_budget(10, provider="ark") == 4


def test_response_format_preference_and_url_ttl():
    """b64 规避 URL 失效：方舟默认 b64_json；硅基仅 url（1h，必须立即下载）。"""
    assert guardrails.preferred_response_format("ark") == "b64_json"
    assert guardrails.preferred_response_format("siliconflow") == "url"
    assert "1h" in guardrails.url_ttl_note("siliconflow")
    assert "24h" in guardrails.url_ttl_note("ark")
