"""统一配置加载 + 路径锚点 + 技能数据目录自动创建。

从项目根 ``.env`` 读取可选设置，并暴露理论计算技能的数据区路径与默认参数。

新增可选 ``.env`` 键（均有默认值，不加也能工作）：
- ``THEORY_DEFAULT_GRID_RESOLUTION`` — 参数空间默认网格分辨率（默认 80）
- ``THEORY_DEFAULT_PLOT_DPI``        — 探索图 PNG 导出分辨率（默认 150）
- ``THEORY_PLOT_BACKEND``            — matplotlib 后端（默认 ``Agg``，无头模式）
- ``THEORY_PYVISTA_OFF_SCREEN``      — pyvista 是否离屏渲染（默认 ``1``）

用法::

    from pysci.skills.theoretical_computation.tools.config import settings

    settings.module_dir          # data/skills/theoretical_computation/
    settings.templates_dir
    settings.research_theory_data_dir("gain_ep")  # data/research/1_gain_ep/theory/
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import PROJECT_ROOT, THEORY_ROOT, research_asset_dir

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
MODULE_DIR: Path = THEORY_ROOT


# ---------------------------------------------------------------------------
# .env 加载（与其他 skill 同款：dotenv 优先，手写解析兜底）
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
            f"[theory.config] WARNING: {key}={v!r} 不是整数，回退默认 {default}",
            file=sys.stderr,
        )
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    v = _get_env(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 theoretical_computation 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    templates_dir: Path     # 脚手架模板（new 命令使用）
    cache_dir: Path         # 符号计算缓存
    recipes_dir: Path       # 可复用的计算配方

    # --- 默认计算参数 ---
    default_grid_resolution: int
    default_plot_dpi: int
    plot_backend: str
    pyvista_off_screen: bool

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    def research_theory_data_dir(self, research_name: str) -> Path:
        """解析某研究线的理论计算数据目录。

        Args:
            research_name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。

        Returns:
            ``data/research/<n>_<name>/theory/`` 路径。
        """
        return research_asset_dir(research_name) / "theory"

    def research_theory_code_dir(self, research_name: str) -> Path:
        """解析某研究线的理论计算代码目录。

        Args:
            research_name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。

        Returns:
            ``src/pysci/research/<name>/theory/`` 路径。
        """
        return (
            self.project_root / "src" / "pysci" / "research" / research_name / "theory"
        )

    def summary(self) -> str:
        """人类可读的配置摘要。"""
        lines = [
            "=== pySciWS theoretical_computation configuration ===",
            f"project_root             : {self.project_root}",
            f"module_dir               : {self.module_dir}",
            f"templates_dir            : {self.templates_dir}",
            f"cache_dir                : {self.cache_dir}",
            f"recipes_dir              : {self.recipes_dir}",
            "",
            f"default_grid_resolution  : {self.default_grid_resolution}",
            f"default_plot_dpi         : {self.default_plot_dpi}",
            f"plot_backend             : {self.plot_backend}",
            f"pyvista_off_screen       : {self.pyvista_off_screen}",
            "======================================================",
        ]
        return "\n".join(lines)


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
        default_grid_resolution=_get_env_int("THEORY_DEFAULT_GRID_RESOLUTION", 80),
        default_plot_dpi=_get_env_int("THEORY_DEFAULT_PLOT_DPI", 150),
        plot_backend=_get_env("THEORY_PLOT_BACKEND", "Agg") or "Agg",
        pyvista_off_screen=_get_env_bool("THEORY_PYVISTA_OFF_SCREEN", True),
        _raw_env={
            k: v for k, v in os.environ.items() if k.startswith("THEORY_")
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


# ---------------------------------------------------------------------------
# CLI: 打印当前配置摘要
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(settings.summary())
