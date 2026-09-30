"""providers 单测：模型/尺寸解析、provider 注册表。纯离线、无联网。"""

from __future__ import annotations

import pytest

from pysci.skills.ai_drawing.tools import providers


def test_get_provider_default():
    p = providers.get_provider(None)
    assert p.name == providers.DEFAULT_PROVIDER == "jimeng-ark"
    assert p.wired is True
    assert p.client_node == "JimengAPIClient"
    assert p.node_class == "JimengSeedream4"


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError, match="未知 provider"):
        providers.get_provider("does-not-exist")


def test_list_providers_wired_only():
    all_p = providers.list_providers()
    wired = providers.list_providers(wired_only=True)
    assert providers.get_provider("jimeng-ark") in wired
    # 预留 provider（dashscope / s3 / s5）不接线
    assert all(p.wired for p in wired)
    assert len(wired) < len(all_p)
    assert providers.get_provider("dashscope") not in wired


def test_default_model_is_seedream4():
    p = providers.get_provider("jimeng-ark")
    m = p.default_model
    assert m.ui_version == "doubao-seedream-4.0"
    assert m.api_id == "doubao-seedream-4-0-250828"
    assert m.node_class == "JimengSeedream4"


@pytest.mark.parametrize(
    "query",
    [
        "doubao-seedream-4.0",       # 完整 UI 版本
        "doubao-seedream-4-0-250828",  # API ID
        "4.0",                        # 简写
        "seedream-4.0",               # 带前缀简写
    ],
)
def test_resolve_model_accepts_variants(query):
    p = providers.get_provider("jimeng-ark")
    assert p.resolve_model(query).ui_version == "doubao-seedream-4.0"


def test_resolve_model_45():
    p = providers.get_provider("jimeng-ark")
    m = p.resolve_model("doubao-seedream-4.5")
    assert m.api_id == "doubao-seedream-4-5-251128"


def test_resolve_model_none_is_default():
    p = providers.get_provider("jimeng-ark")
    assert p.resolve_model(None) is p.default_model
    assert p.resolve_model("") is p.default_model


def test_resolve_model_unknown_raises():
    p = providers.get_provider("jimeng-ark")
    with pytest.raises(KeyError, match="无模型"):
        p.resolve_model("seedream-9.9")


def test_normalize_size_default_and_exact():
    p = providers.get_provider("jimeng-ark")
    assert p.normalize_size(None) == p.default_size == "2K (adaptive)"
    assert p.normalize_size("2K (adaptive)") == "2K (adaptive)"
    assert p.normalize_size("2048x2048 (1:1)") == "2048x2048 (1:1)"


def test_normalize_size_prefix_match():
    p = providers.get_provider("jimeng-ark")
    # "2K" → "2K (adaptive)"；"2048x2048" → "2048x2048 (1:1)"
    assert p.normalize_size("2K") == "2K (adaptive)"
    assert p.normalize_size("2048x2048") == "2048x2048 (1:1)"


def test_normalize_size_case_insensitive():
    p = providers.get_provider("jimeng-ark")
    assert p.normalize_size("custom") == "Custom"


def test_normalize_size_unknown_passthrough():
    p = providers.get_provider("jimeng-ark")
    # 无法匹配 → 原样返回（交服务器校验）
    assert p.normalize_size("999x999") == "999x999"


def test_i2i_support_flags():
    assert providers.get_provider("jimeng-ark").supports_i2i is True
    # Seedream 3.0 t2i provider 不支持 i2i
    assert providers.get_provider("jimeng-ark-s3").supports_i2i is False


def test_resolve_model_helper():
    p, m = providers.resolve_model(None, None)
    assert p.name == "jimeng-ark"
    assert m.ui_version == "doubao-seedream-4.0"


def test_model_versions_listed():
    p = providers.get_provider("jimeng-ark")
    versions = p.model_versions()
    assert "doubao-seedream-4.0" in versions
    assert "doubao-seedream-4.5" in versions
