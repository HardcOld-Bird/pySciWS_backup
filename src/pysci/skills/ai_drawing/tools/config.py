"""统一配置加载 + ComfyUI 发现 + 资源/成本护栏。

从项目根 ``.env`` 读取设置，并暴露：
- 技能数据区各子目录（assets/gallery/workflows/prompts/cache/runs，自动创建）
- ComfyUI 编排器发现（root / server_url），**廉价**：只读 env，不启服务、不联网
- provider（火山方舟 / 即梦）密钥存在性探测与默认模型/尺寸
- 成本护栏（单次生成默认最大张数）

设计原则（对齐 comsol_simulation.config）：ComfyUI 本体 + torch + 自定义节点由 comfy-cli
装在**独立环境**，绝不进 pysci 依赖；本技能只经 HTTP 与之通信，故这里发现的只是
"如何连上/如何拉起"它，而非把它 import 进来。

新增可选 ``.env`` 键（均有默认值，不加也能工作）：
- ``AI_DRAWING_BACKEND``    — 默认后端（``comfyui`` | ``imagegen``；默认 ``comfyui``）
- ``COMFY_ROOT``            — 外部 ComfyUI workspace 根（comfy-cli 安装处；供 launch 用）
- ``COMFY_SERVER_URL``      — ComfyUI 服务器地址（默认 ``http://127.0.0.1:8188``）
- ``COMFY_CLI``             — ``comfy`` 可执行文件路径（不在 PATH 时供 server start/stop 用）
- ``COMFY_JIMENG_KEY_NAME`` — gen/i2i 默认 ``JimengAPIClient.key_name``（匹配 api_keys.json 的 customName）
- ``COMFY_DEFAULT_MODEL``   — 默认图像模型 ID（默认 ``doubao-seedream-4-0-250828``）
- ``AI_DRAWING_DEFAULT_SIZE`` — 默认出图尺寸（默认 ``2K``；方舟支持 1K/2K/4K 或 WxH）
- ``ARK_API_KEY``           — 火山方舟 API Key（**仅探测存在性**；实际密钥存节点 api_keys.json）
- ``AI_DRAWING_MAX_IMAGES`` — 单次生成默认最大张数（成本护栏，默认 4）

用法::

    from pysci.skills.ai_drawing.tools.config import settings

    settings.comfy_server_url
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
# data/skills/ai_drawing/（assets/gallery/workflows/prompts/cache/runs 的父目录）。
MODULE_DIR: Path = AI_DRAWING_ROOT

#: 默认 ComfyUI 服务器地址（comfy-cli / 便携版默认端口 8188）。
DEFAULT_COMFY_URL: str = "http://127.0.0.1:8188"

#: 默认图像模型（火山方舟即梦 Seedream 4.0 首推版本）。
DEFAULT_MODEL: str = "doubao-seedream-4-0-250828"

#: 默认出图尺寸（方舟 size 字段：1K/2K/4K，或形如 1024x1024）。
DEFAULT_SIZE: str = "2K"


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
# ComfyUI 编排器发现（廉价：只读 env，不启服务、不联网）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComfyInstall:
    """ComfyUI 编排器的连接/拉起信息。``found=False`` 表示未配置外部安装路径。

    注意：这里的"发现"仅指**本机是否配置了 ComfyUI workspace 路径**（供 launch 用）；
    服务器**是否真的在运行**由 :mod:`comfy_session` / :mod:`comfy_client` 做实时探活，
    不在构造 Settings 时联网（保持 import 廉价、离线可用）。
    """

    found: bool = False          # 是否配置了 COMFY_ROOT（外部 workspace 路径）
    root: Path | None = None     # ComfyUI workspace 根（comfy-cli 安装处）
    server_url: str = DEFAULT_COMFY_URL
    source: str = "none"         # "env" | "default" | "none"


def discover_comfy() -> ComfyInstall:
    """发现本机 ComfyUI 编排器配置（廉价：读 env，不启服务、不联网）。

    优先级：
    1. ``COMFY_ROOT`` 指向的外部 workspace（存在则 ``found=True``）；
    2. 仅配置了 ``COMFY_SERVER_URL``（可连远程/云端 ComfyUI，则 root 为 None 但 url 生效）；
    3. 都没有 → 默认 ``http://127.0.0.1:8188``（``found=False``，待用户安装/配置）。
    """
    url = _get_env("COMFY_SERVER_URL", DEFAULT_COMFY_URL) or DEFAULT_COMFY_URL
    env_root = _get_env("COMFY_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if root.exists():
            return ComfyInstall(found=True, root=root, server_url=url, source="env")
        print(
            f"[ai_drawing.config] WARNING: COMFY_ROOT={root} 不存在，忽略（仍可连 COMFY_SERVER_URL）",
            file=sys.stderr,
        )
    # 无 COMFY_ROOT：若显式配了 URL，视为连接远程/云端编排器
    if _get_env("COMFY_SERVER_URL"):
        return ComfyInstall(found=False, root=None, server_url=url, source="env")
    return ComfyInstall(found=False, root=None, server_url=url, source="default")


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 ai_drawing 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    assets_dir: Path        # 生成图入库（Agent-ImageGen 产物 / ComfyUI 出图）
    gallery_dir: Path       # 精选审美范本
    workflows_dir: Path     # 保存的 ComfyUI API 格式工作流配方
    prompts_dir: Path       # prompt 配方库
    cache_dir: Path         # 临时预览/中间产物
    runs_dir: Path          # 服务器状态文件（comfy_server.json）+ 日志

    # --- 后端与 ComfyUI 编排器 ---
    default_backend: str    # "comfyui" | "imagegen"
    comfy: ComfyInstall

    # --- provider（火山方舟 / 即梦）---
    default_model: str
    default_size: str
    ark_api_key: str | None  # 仅用于探测存在性/脱敏展示，实际密钥存节点 api_keys.json

    # --- 成本护栏 ---
    max_images: int

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 便捷判断 ----------
    @property
    def comfy_server_url(self) -> str:
        """ComfyUI 服务器地址。"""
        return self.comfy.server_url

    @property
    def ark_ready(self) -> bool:
        """火山方舟 API Key 是否已配置（存在性）。"""
        return bool(self.ark_api_key)

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
            f"workflows_dir    : {self.workflows_dir}",
            f"prompts_dir      : {self.prompts_dir}",
            f"cache_dir        : {self.cache_dir}",
            f"runs_dir         : {self.runs_dir}",
            "",
            f"default_backend  : {self.default_backend}",
            f"comfy_found      : {self.comfy.found} (source={self.comfy.source})",
            f"comfy_root       : {self.comfy.root or '(none)'}",
            f"comfy_server_url : {self.comfy.server_url}",
            "",
            f"default_model    : {self.default_model}",
            f"default_size     : {self.default_size}",
            f"ark_api_key      : {mask(self.ark_api_key)}",
            f"ark_ready        : {self.ark_ready}",
            f"max_images       : {self.max_images}",
            "========================================",
        ]
        return "\n".join(lines)


def build_settings() -> Settings:
    """加载 .env、发现 ComfyUI、构造 Settings。技能数据目录会自动创建。"""
    _load_dotenv_if_available()

    assets_dir = MODULE_DIR / "assets"
    gallery_dir = MODULE_DIR / "gallery"
    workflows_dir = MODULE_DIR / "workflows"
    prompts_dir = MODULE_DIR / "prompts"
    cache_dir = MODULE_DIR / "cache"
    runs_dir = MODULE_DIR / "runs"
    for d in (assets_dir, gallery_dir, workflows_dir, prompts_dir, cache_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        assets_dir=assets_dir,
        gallery_dir=gallery_dir,
        workflows_dir=workflows_dir,
        prompts_dir=prompts_dir,
        cache_dir=cache_dir,
        runs_dir=runs_dir,
        default_backend=(_get_env("AI_DRAWING_BACKEND", "comfyui") or "comfyui").lower(),
        comfy=discover_comfy(),
        default_model=_get_env("COMFY_DEFAULT_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        default_size=_get_env("AI_DRAWING_DEFAULT_SIZE", DEFAULT_SIZE) or DEFAULT_SIZE,
        ark_api_key=_get_env("ARK_API_KEY"),
        max_images=_get_env_int("AI_DRAWING_MAX_IMAGES", 4),
        _raw_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("AI_DRAWING_", "COMFY_", "ARK_"))
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
