"""config 单测：Settings 目录 / discover_comfy / _get_env* / summary 脱敏。

不启服务、不联网——只验证 .env 读取与 ComfyUI 发现的确定性行为。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.ai_drawing.tools import config


def test_get_env_blank_is_unset(monkeypatch):
    monkeypatch.setenv("COMFY_SERVER_URL", "   ")
    assert config._get_env("COMFY_SERVER_URL") is None
    monkeypatch.setenv("COMFY_SERVER_URL", "http://x:1")
    assert config._get_env("COMFY_SERVER_URL") == "http://x:1"
    monkeypatch.delenv("COMFY_SERVER_URL", raising=False)
    assert config._get_env("COMFY_SERVER_URL", "dflt") == "dflt"


def test_get_env_int(monkeypatch):
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "8")
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 8
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "not-an-int")
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 4  # 非整数 → 回退默认
    monkeypatch.delenv("AI_DRAWING_MAX_IMAGES", raising=False)
    assert config._get_env_int("AI_DRAWING_MAX_IMAGES", 4) == 4  # 未设置 → 默认


def test_discover_comfy_default(monkeypatch):
    """无 COMFY_ROOT / COMFY_SERVER_URL → 默认 url、found=False、source=default。"""
    monkeypatch.delenv("COMFY_ROOT", raising=False)
    monkeypatch.delenv("COMFY_SERVER_URL", raising=False)
    inst = config.discover_comfy()
    assert inst.found is False
    assert inst.root is None
    assert inst.server_url == config.DEFAULT_COMFY_URL
    assert inst.source == "default"


def test_discover_comfy_root_env(monkeypatch, tmp_path):
    """COMFY_ROOT 指向存在的目录 → found=True、source=env。"""
    monkeypatch.setenv("COMFY_ROOT", str(tmp_path))
    monkeypatch.delenv("COMFY_SERVER_URL", raising=False)
    inst = config.discover_comfy()
    assert inst.found is True
    assert inst.source == "env"
    assert inst.root == tmp_path


def test_discover_comfy_root_missing_falls_back(monkeypatch, tmp_path, capsys):
    """COMFY_ROOT 指向不存在的目录 → 告警并回退（found=False）。"""
    monkeypatch.setenv("COMFY_ROOT", str(tmp_path / "nope"))
    monkeypatch.delenv("COMFY_SERVER_URL", raising=False)
    inst = config.discover_comfy()
    assert inst.found is False
    assert "COMFY_ROOT" in capsys.readouterr().err


def test_discover_comfy_url_only(monkeypatch):
    """只配 COMFY_SERVER_URL（远程/云端）→ root=None 但 url 生效、source=env。"""
    monkeypatch.delenv("COMFY_ROOT", raising=False)
    monkeypatch.setenv("COMFY_SERVER_URL", "http://10.0.0.5:8188")
    inst = config.discover_comfy()
    assert inst.found is False
    assert inst.root is None
    assert inst.server_url == "http://10.0.0.5:8188"
    assert inst.source == "env"


def test_settings_dirs_exist():
    s = config.settings
    for d in (
        s.module_dir,
        s.assets_dir,
        s.gallery_dir,
        s.workflows_dir,
        s.prompts_dir,
        s.cache_dir,
        s.runs_dir,
    ):
        assert d.is_dir(), d
    assert s.module_dir == config.MODULE_DIR
    assert s.max_images >= 1
    assert s.comfy_server_url == s.comfy.server_url


def _no_dotenv(monkeypatch):
    """屏蔽真实 .env 加载，使 build_settings 只反映本测试注入的 env（hermetic）。"""
    monkeypatch.setattr(config, "_load_dotenv_if_available", lambda: None)


def test_build_settings_reads_env(monkeypatch):
    """build_settings 反映 .env 键（用 env 覆盖，不触碰真实 .env）。"""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("AI_DRAWING_BACKEND", "ImageGen")
    monkeypatch.setenv("COMFY_DEFAULT_MODEL", "custom-model-x")
    monkeypatch.setenv("AI_DRAWING_DEFAULT_SIZE", "4K")
    monkeypatch.setenv("ARK_API_KEY", "sk-secret-abcdef123456")
    monkeypatch.setenv("AI_DRAWING_MAX_IMAGES", "3")
    s = config.build_settings()
    assert s.default_backend == "imagegen"  # 归一化小写
    assert s.default_model == "custom-model-x"
    assert s.default_size == "4K"
    assert s.max_images == 3
    assert s.ark_ready is True


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
