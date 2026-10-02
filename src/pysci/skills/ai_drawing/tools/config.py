"""统一配置加载 + 云端 provider 设置 + 资源/成本护栏。

从项目根 ``.env`` 读取设置，并暴露：
- 技能数据区各子目录（assets/gallery/recipes/prompts/cache/runs，自动创建）
- 云端图像 provider（火山方舟 / 即梦 Seedream）的端点、密钥与默认模型
- 成本护栏（单次生成默认最大张数）

设计原则：**import 廉价、离线可用**。构造 :class:`Settings` 时只读环境变量，
绝不联网、绝不校验密钥有效性；密钥是否可用由 :mod:`ark_client` 在真实调用时
以 HTTP 状态码反馈（``doctor`` 子命令也只做存在性探测与脱敏展示）。

可选 ``.env`` 键（除 ``ARK_API_KEY`` 外均有默认值，不加也能工作）：
- ``ARK_API_KEY``             — 火山方舟 API Key（**云端出图必需**；仅存本地 .env，不进版本库）
- ``ARK_BASE_URL``            — 方舟推理端点（默认 ``https://ark.cn-beijing.volces.com/api/v3``）
- ``AI_DRAWING_BACKEND``      — 默认后端（``ark`` | ``imagegen``；默认 ``ark``）
- ``AI_DRAWING_MODEL``        — 默认图像模型 ID（默认 ``doubao-seedream-5-0-flash-260915``）
- ``AI_DRAWING_DEFAULT_SIZE`` — 默认出图尺寸（默认 ``2K``；方舟支持 1K/2K/4K/3K 或 WxH）
- ``AI_DRAWING_MAX_IMAGES``   — 单次生成默认最大张数（成本护栏，默认 4）

模型 ID 以方舟控制台「模型列表」为唯一真值源——本模块**不维护本地 provider 注册表**，
``model`` 一律字符串直传，避免注册表随方舟上新而腐烂。

用法::

    from pysci.skills.ai_drawing.tools.config import settings

    settings.ark_api_key         # None 表示未配置
    settings.default_model
    settings.module_dir          # data/skills/ai_drawing/
    settings.assets_dir
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import AI_DRAWING_ROOT, PROJECT_ROOT

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
# 路径统一由 pysci.paths 收口（标记法查找项目根）。MODULE_DIR 指向 AI 绘图数据区
# data/skills/ai_drawing/（assets/gallery/recipes/prompts/cache/runs 的父目录）。
MODULE_DIR: Path = AI_DRAWING_ROOT

#: 方舟推理端点（OpenAI 兼容协议的 base，图像生成走 ``{base}/images/generations``）。
DEFAULT_ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"

#: 默认图像模型（Seedream 5.0-flash：本账号已开通且 2026-10 实测出图；
#: 4.0 已下架、5.0-lite 关闭订阅即将下架，本账号均不可开通）。
DEFAULT_MODEL: str = "doubao-seedream-5-0-flash-260915"

#: 默认出图尺寸（方舟 size 字段：1K/2K/4K，或形如 1024x1024）。
DEFAULT_SIZE: str = "2K"

#: 单次请求默认超时（秒）。云端出图同步返回，2K 图通常 30–60 s。
DEFAULT_TIMEOUT: float = 180.0


# ---------------------------------------------------------------------------
# .env 加载（与 comsol_simulation / scientific_plotting 同款：dotenv 优先，手写解析兜底）
# ---------------------------------------------------------------------------
def _load_dotenv_if_available() -> None:
    """尝试用 python-dotenv 加载项目根 .env；失败则手写解析（不硬依赖 dotenv）。"""
    env_path = PROJECT_ROOT / ".env"
    override = os.environ.get("PYSCIWS_ENV_FILE")
    if override:
        env_path = Path(override)
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            os.environ.setdefault(key.strip(), value)


def _get_env(key: str, default: str | None = None) -> str | None:
    """从环境变量读取，空串视为未设置。"""
    v = os.environ.get(key)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _get_env_int(key: str, default: int) -> int:
    v = _get_env(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        print(
            f"[ai_drawing.config] WARNING: {key}={v!r} 不是整数，回退默认 {default}",
            file=sys.stderr,
        )
        return default


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 ai_drawing 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    assets_dir: Path  # 生成图入库（云端出图 / Agent-ImageGen 产物）
    gallery_dir: Path  # 精选审美范本
    recipes_dir: Path  # 流水线配方知识卡片（Markdown，与其余技能同构）
    prompts_dir: Path  # prompt 配方库
    cache_dir: Path  # 临时预览/中间产物（如图层拆分的分层输出）
    runs_dir: Path  # 云端响应原始 JSON 快照 + 日志（排障用）

    # --- 后端 ---
    default_backend: str  # "ark"（云端出图）| "imagegen"（Qoder 内置，Agent 直接调）

    # --- provider（火山方舟 / 即梦 Seedream）---
    ark_base_url: str
    ark_api_key: str | None  # 云端出图必需；summary() 中脱敏展示
    default_model: str
    default_size: str
    timeout: float

    # --- 成本护栏 ---
    max_images: int

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 便捷判断 ----------
    @property
    def ark_ready(self) -> bool:
        """火山方舟 API Key 是否已配置（存在性，不代表有效）。"""
        return bool(self.ark_api_key)

    @property
    def images_endpoint(self) -> str:
        """图片生成端点完整 URL。"""
        return f"{self.ark_base_url.rstrip('/')}/images/generations"

    def summary(self) -> str:
        """人类可读的配置摘要（敏感字段脱敏）。"""

        def mask(v: str | None) -> str:
            if not v:
                return "(unset)"
            return "***" if len(v) <= 6 else f"{v[:3]}***{v[-3:]}"

        lines = [
            "=== pySciWS ai_drawing configuration ===",
            f"project_root     : {self.project_root}",
            f"module_dir       : {self.module_dir}",
            f"assets_dir       : {self.assets_dir}",
            f"gallery_dir      : {self.gallery_dir}",
            f"recipes_dir      : {self.recipes_dir}",
            f"prompts_dir      : {self.prompts_dir}",
            f"cache_dir        : {self.cache_dir}",
            f"runs_dir         : {self.runs_dir}",
            "",
            f"default_backend  : {self.default_backend}",
            "",
            f"ark_base_url     : {self.ark_base_url}",
            f"ark_api_key      : {mask(self.ark_api_key)}",
            f"ark_ready        : {self.ark_ready}",
            f"default_model    : {self.default_model}",
            f"default_size     : {self.default_size}",
            f"timeout          : {self.timeout}s",
            f"max_images       : {self.max_images}",
            "========================================",
        ]
        return "\n".join(lines)


def build_settings() -> Settings:
    """加载 .env、构造 Settings。技能数据目录会自动创建。"""
    _load_dotenv_if_available()

    assets_dir = MODULE_DIR / "assets"
    gallery_dir = MODULE_DIR / "gallery"
    recipes_dir = MODULE_DIR / "recipes"
    prompts_dir = MODULE_DIR / "prompts"
    cache_dir = MODULE_DIR / "cache"
    runs_dir = MODULE_DIR / "runs"
    for d in (assets_dir, gallery_dir, recipes_dir, prompts_dir, cache_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    timeout_raw = _get_env("AI_DRAWING_TIMEOUT")
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT
    except ValueError:
        print(
            f"[ai_drawing.config] WARNING: AI_DRAWING_TIMEOUT={timeout_raw!r} 不是数字，"
            f"回退默认 {DEFAULT_TIMEOUT}",
            file=sys.stderr,
        )
        timeout = DEFAULT_TIMEOUT

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        assets_dir=assets_dir,
        gallery_dir=gallery_dir,
        recipes_dir=recipes_dir,
        prompts_dir=prompts_dir,
        cache_dir=cache_dir,
        runs_dir=runs_dir,
        default_backend=(_get_env("AI_DRAWING_BACKEND", "ark") or "ark").lower(),
        ark_base_url=_get_env("ARK_BASE_URL", DEFAULT_ARK_BASE_URL)
        or DEFAULT_ARK_BASE_URL,
        ark_api_key=_get_env("ARK_API_KEY"),
        default_model=_get_env("AI_DRAWING_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        default_size=_get_env("AI_DRAWING_DEFAULT_SIZE", DEFAULT_SIZE) or DEFAULT_SIZE,
        timeout=timeout,
        max_images=_get_env_int("AI_DRAWING_MAX_IMAGES", 4),
        _raw_env={
            k: v for k, v in os.environ.items() if k.startswith(("AI_DRAWING_", "ARK_"))
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


# ---------------------------------------------------------------------------
# CLI: 打印当前配置摘要（等价于 imagine.py doctor 的配置部分）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(settings.summary())
