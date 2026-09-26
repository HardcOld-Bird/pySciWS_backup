"""统一配置加载 + 路径锚点 + 技能数据目录自动创建。

从项目根 ``.env`` 读取可选设置，并暴露科研绘图技能的数据区路径与默认参数。

新增可选 ``.env`` 键（均有默认值，不加也能工作）：
- ``SCIPLOT_DEFAULT_STYLE``   — 默认期刊风格预设（默认 ``aps``；可选 ``nature``）
- ``SCIPLOT_DEFAULT_FORMATS`` — 默认导出格式，逗号分隔（默认 ``eps,pdf,svg,png``）
- ``SCIPLOT_PREVIEW_DPI``     — PNG 预览分辨率（默认 200）
- ``SCIPLOT_SAVE_DPI``        — 位图/预览之外的栅格 dpi（默认 600，用于 png 交付件）

用法::

    from pysci.skills.scientific_plotting.tools.config import settings

    settings.default_style
    settings.module_dir          # data/skills/scientific_plotting/
    settings.templates_dir
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import PLOTTING_ROOT, PROJECT_ROOT

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
# 路径统一由 pysci.paths 收口（标记法查找项目根）。MODULE_DIR 指向科研绘图数据区
# data/skills/scientific_plotting/（templates/cache/recipes 的父目录）。
MODULE_DIR: Path = PLOTTING_ROOT

#: 默认导出格式（矢量交付件 + 位图交付件）。PNG 预览始终另外产出，供 Agent 视觉校验。
DEFAULT_FORMATS: tuple[str, ...] = ("eps", "pdf", "svg", "png")


# ---------------------------------------------------------------------------
# .env 加载（与 literature_research / comsol_simulation 同款：dotenv 优先，手写解析兜底）
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
            f"[sciplot.config] WARNING: {key}={v!r} 不是整数，回退默认 {default}",
            file=sys.stderr,
        )
        return default


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 scientific_plotting 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    templates_dir: Path     # 脚手架模板（新建图管线时复制）
    cache_dir: Path         # 预览/临时栅格缓存
    recipes_dir: Path       # 固化下来的可复用绘图配方画廊

    # --- 默认绘图参数 ---
    default_style: str
    default_formats: tuple[str, ...]
    preview_dpi: int
    save_dpi: int

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        """人类可读的配置摘要。"""
        lines = [
            "=== pySciWS scientific_plotting configuration ===",
            f"project_root     : {self.project_root}",
            f"module_dir       : {self.module_dir}",
            f"templates_dir    : {self.templates_dir}",
            f"cache_dir        : {self.cache_dir}",
            f"recipes_dir      : {self.recipes_dir}",
            "",
            f"default_style    : {self.default_style}",
            f"default_formats  : {','.join(self.default_formats)}",
            f"preview_dpi      : {self.preview_dpi}",
            f"save_dpi         : {self.save_dpi}",
            "=================================================",
        ]
        return "\n".join(lines)


def _parse_formats(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_FORMATS
    items = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return items or DEFAULT_FORMATS


def build_settings() -> Settings:
    """加载 .env、构造 Settings。技能数据目录会自动创建。"""
    _load_dotenv_if_available()

    templates_dir = MODULE_DIR / "templates"
    cache_dir = MODULE_DIR / "cache"
    recipes_dir = MODULE_DIR / "recipes"
    for d in (templates_dir, cache_dir, recipes_dir):
        d.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        templates_dir=templates_dir,
        cache_dir=cache_dir,
        recipes_dir=recipes_dir,
        default_style=(_get_env("SCIPLOT_DEFAULT_STYLE", "aps") or "aps").lower(),
        default_formats=_parse_formats(_get_env("SCIPLOT_DEFAULT_FORMATS")),
        preview_dpi=_get_env_int("SCIPLOT_PREVIEW_DPI", 200),
        save_dpi=_get_env_int("SCIPLOT_SAVE_DPI", 600),
        _raw_env={
            k: v for k, v in os.environ.items() if k.startswith("SCIPLOT_")
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


# ---------------------------------------------------------------------------
# CLI: 打印当前配置摘要（等价于 figures.py doctor 的配置部分）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(settings.summary())
