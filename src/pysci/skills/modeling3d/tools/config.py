"""modeling3d 统一配置加载：路径、Blender 可执行文件探测、渲染默认值、Tripo 凭据。

从项目根 ``.env`` 读取设置（与 ai_drawing / theoretical_computation 同款：dotenv 优先、
手写解析兜底），暴露 :class:`Settings` 单例。

设计原则：**import 廉价、离线可用**——构造时只读环境变量与探测文件系统，绝不联网；
Tripo 密钥有效性由 :mod:`tripo_client` 在真实调用时以 HTTP 状态码反馈。

可选 ``.env`` 键（除 ``TRIPO_API_KEY`` 外均有默认值/自动探测，不加也能工作）：
- ``MODELING3D_BLENDER``      — blender.exe 绝对路径（未设时按常见安装位置自动探测）
- ``MODELING3D_RENDER_ENGINE``— 默认渲染引擎（``EEVEE`` | ``CYCLES``；默认 EEVEE，
  实际引擎 ID 随 Blender 版本浮动，渲染脚本内有回退链）
- ``MODELING3D_RENDER_SAMPLES`` — Cycles 采样数（默认 128）
- ``MODELING3D_RENDER_SIZE``  — 默认出图尺寸 ``宽x高``（默认 ``1920x1440``）
- ``TRIPO_API_KEY``           — Tripo 开放平台 API Key（**图生 3D 必需**；platform.tripo3d.ai）
- ``TRIPO_BASE_URL``          — Tripo API base（默认 ``https://api.tripo3d.ai``）

用法::

    from pysci.skills.modeling3d.tools.config import settings

    settings.blender_exe       # None 表示未找到
    settings.tripo_ready
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import MODELING3D_ROOT, PROJECT_ROOT

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
MODULE_DIR: Path = MODELING3D_ROOT

#: Tripo 开放平台 API base（Bearer 鉴权；契约详见 tripo_client 模块 docstring）。
DEFAULT_TRIPO_BASE_URL: str = "https://api.tripo3d.ai"

#: Tripo 任务轮询间隔与默认超时（秒）。图生 3D 通常 1–3 分钟。
TRIPO_POLL_INTERVAL: float = 5.0
DEFAULT_TRIPO_TIMEOUT: float = 600.0

#: Blender 可执行文件的常见安装位置（glob，取版本最高者）。
_BLENDER_PROBE_GLOBS: tuple[str, ...] = (
    r"D:\XiGPrograms\blender\base\Blender *\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender *\ *\blender.exe",
)


def probe_blender_exe() -> Path | None:
    """探测本机 Blender 可执行文件。

    依次尝试：PATH 中的 ``blender`` → :data:`_BLENDER_PROBE_GLOBS` 各安装位置
    （同盘多版本时按目录名排序取最高）。

    Returns:
        blender.exe 路径；未找到返回 None。
    """
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    candidates: list[Path] = []
    for pattern in _BLENDER_PROBE_GLOBS:
        candidates.extend(Path(p) for p in glob.glob(pattern))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def probe_mcp_addon() -> Path | None:
    """探测 blender-mcp addon 是否已装入某版本 Blender 的用户脚本目录。"""
    base = Path(os.environ.get("APPDATA", "")) / "Blender Foundation" / "Blender"
    if not base.is_dir():
        return None
    hits = sorted(base.glob("*/scripts/addons*/blendermcp_addon.py"))
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# .env 加载（与其他技能同款：dotenv 优先，手写解析兜底）
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


def _parse_size(raw: str, default: tuple[int, int]) -> tuple[int, int]:
    """解析 ``宽x高`` 尺寸串；非法时告警并回退默认。"""
    try:
        w, _, h = raw.partition("x")
        return int(w), int(h)
    except ValueError:
        print(
            f"[modeling3d.config] WARNING: 尺寸 {raw!r} 非法（应为 宽x高），"
            f"回退默认 {default[0]}x{default[1]}",
            file=sys.stderr,
        )
        return default


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 modeling3d 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    templates_dir: Path  # 脚手架模板（print_part / scene）
    recipes_dir: Path  # 可复用渲染/建模配方（Markdown）
    cache_dir: Path  # 中间产物缓存

    # --- Blender ---
    blender_exe: Path | None  # None 表示未探测到；doctor 会给出设置指引
    mcp_addon_path: Path | None  # blender-mcp addon（交互通道，非 CLI 必需）
    render_engine: str  # "EEVEE" | "CYCLES"（渲染脚本内有引擎 ID 回退链）
    render_samples: int
    render_size: tuple[int, int]

    # --- Tripo（图生 3D）---
    tripo_base_url: str
    tripo_api_key: str | None  # summary() 中脱敏展示
    tripo_timeout: float

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 便捷判断 ----------
    @property
    def blender_ready(self) -> bool:
        """是否找到 Blender 可执行文件。"""
        return self.blender_exe is not None and self.blender_exe.exists()

    @property
    def tripo_ready(self) -> bool:
        """Tripo API Key 是否已配置（存在性，不代表有效）。"""
        return bool(self.tripo_api_key)

    # ---------- 研究线目录 ----------
    def research_model_code_dir(self, research: str) -> Path:
        """研究线的 CAD/场景**代码**目录：``src/pysci/research/<name>/models/``。"""
        return self.project_root / "src" / "pysci" / "research" / research / "models"

    def summary(self) -> str:
        """人类可读的配置摘要（敏感字段脱敏）。"""

        def mask(v: str | None) -> str:
            if not v:
                return "(unset)"
            return "***" if len(v) <= 6 else f"{v[:3]}***{v[-3:]}"

        lines = [
            "=== pySciWS modeling3d configuration ===",
            f"project_root     : {self.project_root}",
            f"module_dir       : {self.module_dir}",
            f"templates_dir    : {self.templates_dir}",
            f"recipes_dir      : {self.recipes_dir}",
            f"cache_dir        : {self.cache_dir}",
            "",
            f"blender_exe      : {self.blender_exe or '(NOT FOUND — 设 MODELING3D_BLENDER)'}",
            f"mcp_addon        : {self.mcp_addon_path or '(未安装)'}",
            f"render_engine    : {self.render_engine}",
            f"render_samples   : {self.render_samples}",
            f"render_size      : {self.render_size[0]}x{self.render_size[1]}",
            "",
            f"tripo_base_url   : {self.tripo_base_url}",
            f"tripo_api_key    : {mask(self.tripo_api_key)}",
            f"tripo_ready      : {self.tripo_ready}",
            f"tripo_timeout    : {self.tripo_timeout}s",
            "========================================",
        ]
        return "\n".join(lines)


def build_settings() -> Settings:
    """加载 .env、构造 Settings。技能数据目录会自动创建。"""
    _load_dotenv_if_available()

    templates_dir = MODULE_DIR / "templates"
    recipes_dir = MODULE_DIR / "recipes"
    cache_dir = MODULE_DIR / "cache"
    for d in (templates_dir, recipes_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    blender_raw = _get_env("MODELING3D_BLENDER")
    blender_exe = Path(blender_raw) if blender_raw else probe_blender_exe()

    timeout_raw = _get_env("MODELING3D_TIMEOUT")
    try:
        tripo_timeout = float(timeout_raw) if timeout_raw else DEFAULT_TRIPO_TIMEOUT
    except ValueError:
        print(
            f"[modeling3d.config] WARNING: MODELING3D_TIMEOUT={timeout_raw!r} 不是数字，"
            f"回退默认 {DEFAULT_TRIPO_TIMEOUT}",
            file=sys.stderr,
        )
        tripo_timeout = DEFAULT_TRIPO_TIMEOUT

    samples_raw = _get_env("MODELING3D_RENDER_SAMPLES")
    try:
        samples = int(samples_raw) if samples_raw else 128
    except ValueError:
        print(
            f"[modeling3d.config] WARNING: MODELING3D_RENDER_SAMPLES={samples_raw!r} "
            "不是整数，回退默认 128",
            file=sys.stderr,
        )
        samples = 128

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        templates_dir=templates_dir,
        recipes_dir=recipes_dir,
        cache_dir=cache_dir,
        blender_exe=blender_exe,
        mcp_addon_path=probe_mcp_addon(),
        render_engine=(
            _get_env("MODELING3D_RENDER_ENGINE", "EEVEE") or "EEVEE"
        ).upper(),
        render_samples=samples,
        render_size=_parse_size(
            _get_env("MODELING3D_RENDER_SIZE", "1920x1440") or "1920x1440",
            (1920, 1440),
        ),
        tripo_base_url=_get_env("TRIPO_BASE_URL", DEFAULT_TRIPO_BASE_URL)
        or DEFAULT_TRIPO_BASE_URL,
        tripo_api_key=_get_env("TRIPO_API_KEY"),
        tripo_timeout=tripo_timeout,
        _raw_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MODELING3D_", "TRIPO_"))
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


if __name__ == "__main__":
    print(settings.summary())
