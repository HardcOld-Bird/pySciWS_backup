"""config 单测：Settings 目录 / _get_env* / images_endpoint / summary 脱敏。

不联网、不校验密钥有效性——只验证 .env 读取与派生属性的确定性行为。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from pysci.skills.ai_drawing.tools import config


def test_get_env_blank_is_unset(monkeypatch):
    """空串/纯空白视为未设置（避免 .env 里写个空值把默认项顶掉）。"""
    monkeypatch.setenv("ARK_BASE_URL", "   ")
    assert config._get_env("ARK_BASE_URL") is None
    monkeypatch.setenv("ARK_BASE_URL", "http://x:1")
    assert config._get_env("ARK_BASE_URL") == "http://x:1"
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    assert config._get_env("ARK_BASE_URL", "dflt") == "dflt"


def test_get_env_int(monkeypatch):
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "8")
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 8
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "not-an-int")
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 4  # 非整数 → 回退默认
    monkeypatch.delenv("AI_DRAWING_MAX_IMAGES", raising=False)
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 4  # 未设置 → 默认


def test_settings_dirs_exist():
    """build_settings 会自建全部技能数据区子目录（import 即可用，无需手工 mkdir）。"""
    s = config.settings
    for d in (
        s.module_dir,
        s.assets_dir,
        s.gallery_dir,
        s.recipes_dir,
        s.prompts_dir,
        s.cache_dir,
        s.runs_dir,
    ):
        assert d.is_dir(), d
    assert s.module_dir == config.MODULE_DIR
    assert s.max_images >= 1
    assert s.timeout > 0


def test_images_endpoint_derives_from_base_url():
    """images_endpoint 拼 base + /images/generations，且容忍 base 尾斜杠。"""
    s = config.settings
    assert s.images_endpoint == f"{s.ark_base_url.rstrip('/')}/images/generations"
    trailing = dataclasses.replace(s, ark_base_url="https://example.com/api/v3/")
    assert trailing.images_endpoint == "https://example.com/api/v3/images/generations"


def test_defaults_match_module_constants():
    """未设 env 时的默认值就是模块常量（无隐藏的第三套默认）。"""
    s = config.build_settings()
    assert s.ark_base_url == config.DEFAULT_ARK_BASE_URL or s.ark_base_url
    assert config.DEFAULT_MODEL.startswith("doubao-seedream-")
    assert config.DEFAULT_SIZE == "2K"
    assert config.DEFAULT_TIMEOUT == 180.0


def _no_dotenv(monkeypatch):
    """屏蔽真实 .env 加载，使 build_settings 只反映本测试注入的 env（hermetic）。"""
    monkeypatch.setattr(config, "_load_dotenv_if_available", lambda: None)


def test_build_settings_reads_env(monkeypatch):
    """build_settings 反映 .env 键（用 env 覆盖，不触碰真实 .env）。"""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("AI_DRAWING_BACKEND", "ImageGen")
    monkeypatch.setenv("AI_DRAWING_MODEL", "custom-model-x")
    monkeypatch.setenv("AI_DRAWING_DEFAULT_SIZE", "4K")
    monkeypatch.setenv("ARK_API_KEY", "sk-secret-abcdef123456")
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.example.com/api/v3")
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "3")
    monkeypatch.setenv("AI_DRAWING_TIMEOUT", "42.5")
    s = config.build_settings()
    assert s.default_backend == "imagegen"  # 归一化小写
    assert s.default_model == "custom-model-x"
    assert s.default_size == "4K"
    assert s.max_images == 3
    assert s.timeout == 42.5
    assert s.ark_api_key == "sk-secret-abcdef123456"
    assert s.ark_ready is True
    assert s.images_endpoint == "https://ark.example.com/api/v3/images/generations"


def test_build_settings_bad_timeout_falls_back(monkeypatch, capsys):
    """AI_DRAWING_TIMEOUT 非数字 → 告警并回退默认（不让技能因配置笔误而 import 失败）。"""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("AI_DRAWING_TIMEOUT", "soon")
    s = config.build_settings()
    assert s.timeout == config.DEFAULT_TIMEOUT
    assert "AI_DRAWING_TIMEOUT" in capsys.readouterr().err


def test_raw_env_only_collects_own_prefixes(monkeypatch):
    """_raw_env 只快照本技能相关前缀，不把整个环境（含其它技能的密钥）拖进来。"""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "sk-ark")
    monkeypatch.setenv("AI_DRAWING_MODEL", "m")
    monkeypatch.setenv("ELSEVIER_API_KEY", "sk-elsevier")
    raw = config.build_settings()._raw_env
    assert "ARK_API_KEY" in raw and "AI_DRAWING_MODEL" in raw
    assert "ELSEVIER_API_KEY" not in raw


def test_summary_masks_api_key(monkeypatch):
    """summary() 绝不整串泄露 ARK_API_KEY。"""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "sk-super-secret-token-0123456789")
    s = config.build_settings()
    txt = s.summary()
    assert "ark_api_key" in txt
    assert s.ark_api_key not in txt  # 脱敏：完整 key 不出现
    assert "***" in txt


def test_summary_when_key_unset(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    s = config.build_settings()
    assert s.ark_ready is False
    assert "(unset)" in s.summary()


def test_module_dir_is_ai_drawing_root():
    assert config.MODULE_DIR.name == "ai_drawing"
    assert isinstance(config.MODULE_DIR, Path)
