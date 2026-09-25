"""document_writing 配置加载与外部工具链探测。

从项目根 `.env` 读取少量可选覆盖项，并暴露：
- 各数据目录路径（templates / projects / assets / cache，自动创建）
- 外部工具探测：TeX 发行版（pdflatex/xelatex/lualatex/latexmk/chktex/bibtex/biber）、
  LibreOffice（转换后端）、以及 Python 库可用性（markitdown / python-pptx / pymupdf）。

用法::

    from pysci.skills.document_writing.tools.config import settings
    settings.projects_dir
    settings.find_tex_tools()      # {'latexmk': 'C:/texlive/2025/bin/windows/latexmk.exe', ...}
    settings.tex_ready             # bool：latexmk + 至少一个引擎可用
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import DOCWRITING_ROOT, PROJECT_ROOT

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
# 路径统一由 pysci.paths 收口（标记法查找项目根），不依赖本文件所在的脆弱层级数学。
# MODULE_DIR 指向文档写作数据区 data/skills/document_writing/。
MODULE_DIR: Path = DOCWRITING_ROOT
CACHE_DIR_ENV_DEFAULT: str = "data/skills/document_writing/cache"


def _resolve_dir(raw: str | None, default: str) -> Path:
    """把 .env 中的（可能相对的）目录解析为绝对路径。"""
    p = Path(raw) if raw else Path(default)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------------------------------------------------------------------
# .env 加载（python-dotenv 是核心依赖，恒可用；仍保留最小回退以防万一）
# ---------------------------------------------------------------------------
def _load_dotenv_if_available() -> None:
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
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


# ---------------------------------------------------------------------------
# 外部工具探测
# ---------------------------------------------------------------------------
def _registry_path_dirs() -> list[Path]:
    """读注册表中持久化的用户/系统 PATH（即“重开终端后”的真实视图）。

    当前 shell 的 PATH 可能是安装前的旧快照；注册表 PATH 才反映新装工具。
    据此兜底可覆盖任意自定义安装盘符/嵌套目录（如 D:/XiGPrograms/tex/...）。
    """
    if sys.platform != "win32":
        return []
    import winreg  # Windows-only stdlib

    out: list[Path] = []
    targets = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for hkey, subkey in targets:
        try:
            with winreg.OpenKey(hkey, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if not isinstance(value, str):
            continue
        for part in os.path.expandvars(value).split(os.pathsep):
            part = part.strip().strip('"')
            if part:
                out.append(Path(part))
    return [p for p in out if p.is_dir()]


def _candidate_bin_dirs() -> list[Path]:
    """常见 TeX / LibreOffice 安装位置的 bin 目录（PATH 未刷新时的兜底探测）。

    刚装完 TeX Live 常未重启终端，`shutil.which` 找不到；这里先看注册表持久化
    PATH（新终端的真实视图），再 glob 常见路径，使 doctor 在装好但未重开 shell
    时仍能给出准确结论。
    """
    dirs: list[Path] = list(_registry_path_dirs())
    # TeX Live（年度目录）
    for root in (Path("C:/texlive"), Path("D:/texlive")):
        if root.exists():
            dirs.extend(sorted(root.glob("*/bin/windows"), reverse=True))
            dirs.extend(sorted(root.glob("*/bin/win32"), reverse=True))
    local = os.environ.get("LOCALAPPDATA")
    appdata = os.environ.get("APPDATA")
    # TinyTeX
    for base in (appdata, local):
        if base:
            dirs.append(Path(base) / "TinyTeX" / "bin" / "windows")
    # MiKTeX
    if local:
        dirs.append(Path(local) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64")
    dirs.append(Path("C:/Program Files/MiKTeX/miktex/bin/x64"))
    # LibreOffice
    dirs.append(Path("C:/Program Files/LibreOffice/program"))
    dirs.append(Path("C:/Program Files (x86)/LibreOffice/program"))
    dirs.append(Path("D:/Program Files/LibreOffice/program"))
    return [d for d in dirs if d.is_dir()]


def which(name: str) -> str | None:
    """在 PATH 中查找可执行文件；找不到再扫常见安装目录。返回绝对路径或 None。"""
    found = shutil.which(name)
    if found:
        return found
    exe = name if name.endswith(".exe") else f"{name}.exe"
    for d in _candidate_bin_dirs():
        cand = d / exe
        if cand.exists():
            return str(cand)
        cand_noext = d / name
        if cand_noext.exists():
            return str(cand_noext)
    return None


def lib_available(name: str) -> bool:
    """检查某 Python 库是否可导入（不实际导入）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# TeX 工具集：探测这些可执行文件
TEX_TOOLS = (
    "latexmk",
    "pdflatex",
    "xelatex",
    "lualatex",
    "chktex",
    "bibtex",
    "biber",
    "kpsewhich",
)
ENGINES = ("pdflatex", "xelatex", "lualatex")


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 document_writing 的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    templates_dir: Path
    projects_dir: Path
    assets_dir: Path
    cache_dir: Path
    cache_extracted: Path  # pptx/docx/pdf → markdown 的提取缓存
    cache_renders: Path  # PDF → PNG 的渲染缓存（供 LLM 看图）

    # --- LaTeX ---
    default_latex_engine: str  # "auto" 表示交给 latexmk / 项目 latexmkrc 决定

    # --- 原始 env 快照（便于调试） ---
    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 外部工具探测 ----------
    def find_tex_tools(self) -> dict[str, str | None]:
        """探测 TeX 工具集，返回 {工具名: 绝对路径 | None}。"""
        return {t: which(t) for t in TEX_TOOLS}

    def find_engines(self) -> dict[str, str | None]:
        """探测可用引擎（pdflatex/xelatex/lualatex）。"""
        return {e: which(e) for e in ENGINES}

    def augmented_env(self) -> dict[str, str]:
        """子进程环境：把 TeX bin 目录注入 PATH 前端。

        latexmk 内部按名字调用 pdflatex/bibtex 等；若当前 shell PATH 是旧快照，
        必须显式注入 bin 目录，否则子进程找不到引擎。
        """
        env = dict(os.environ)
        extra: list[str] = []
        for tool in ("latexmk", *ENGINES):
            p = which(tool)
            if p:
                d = str(Path(p).parent)
                if d not in extra:
                    extra.append(d)
        if extra:
            env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
        return env

    def find_libreoffice(self) -> str | None:
        """探测 LibreOffice（soffice），用于格式转换 / 整页渲染。

        优先尊重 .env 的 DOCWRITING_SOFFICE（自定义安装路径，如 D 盘非标位置），
        再走 PATH / 注册表 / 常见目录兜底探测。
        """
        override = _get_env("DOCWRITING_SOFFICE")
        if override:
            p = Path(override)
            if p.is_file():
                return str(p)
        for name in ("soffice", "libreoffice"):
            p = which(name)
            if p:
                return p
        return None

    # ---------- 便捷判断 ----------
    @property
    def tex_ready(self) -> bool:
        """latexmk + 至少一个引擎可用即为就绪。"""
        tools = self.find_tex_tools()
        return bool(tools.get("latexmk")) and any(
            tools.get(e) for e in ENGINES
        )

    @property
    def python_libs(self) -> dict[str, bool]:
        """写作相关 Python 库可用性。"""
        return {
            "markitdown": lib_available("markitdown"),
            "pptx": lib_available("pptx"),
            "docx": lib_available("docx"),
            "pymupdf": lib_available("pymupdf") or lib_available("fitz"),
        }

    def summary(self) -> str:
        """人类可读的配置 + 工具链摘要，供 doctor 使用。"""
        tex = self.find_tex_tools()
        libs = self.python_libs
        lo = self.find_libreoffice()

        def tool(name: str) -> str:
            return tex.get(name) or "✗ 未找到"

        lines = [
            "=== pySciWS document_writing configuration ===",
            f"project_root      : {self.project_root}",
            f"module_dir        : {self.module_dir}",
            f"templates_dir     : {self.templates_dir}",
            f"projects_dir      : {self.projects_dir}",
            f"cache_dir         : {self.cache_dir}",
            "",
            "【TeX 工具链】",
            f"  tex_ready       : {self.tex_ready}",
            f"  latexmk         : {tool('latexmk')}",
            f"  pdflatex        : {tool('pdflatex')}",
            f"  xelatex         : {tool('xelatex')}",
            f"  lualatex        : {tool('lualatex')}",
            f"  chktex          : {tool('chktex')}",
            f"  bibtex / biber  : {tex.get('bibtex') or '✗'} / {tex.get('biber') or '✗'}",
            f"  default_engine  : {self.default_latex_engine}",
            "",
            "【转换后端 LibreOffice】",
            f"  soffice         : {lo or '✗ 未找到（Phase 2 转换需要）'}",
            "",
            "【Python 库】",
            f"  markitdown      : {'✓' if libs['markitdown'] else '✗（uv sync --extra writing）'}",
            f"  python-pptx     : {'✓' if libs['pptx'] else '✗（uv sync --extra writing）'}",
            f"  python-docx     : {'✓' if libs['docx'] else '✗（Phase 2）'}",
            f"  pymupdf         : {'✓' if libs['pymupdf'] else '✗'}",
            "=========================================",
        ]
        return "\n".join(lines)


def build_settings() -> Settings:
    """加载 .env 并构造 Settings。数据目录会自动创建。"""
    _load_dotenv_if_available()

    templates_dir = MODULE_DIR / "templates"
    projects_dir = MODULE_DIR / "projects"
    assets_dir = MODULE_DIR / "assets"
    cache_dir = _resolve_dir(_get_env("DOCWRITING_CACHE_DIR"), CACHE_DIR_ENV_DEFAULT)
    cache_extracted = cache_dir / "extracted"
    cache_renders = cache_dir / "renders"
    for d in (
        templates_dir,
        projects_dir,
        assets_dir,
        cache_dir,
        cache_extracted,
        cache_renders,
    ):
        d.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        templates_dir=templates_dir,
        projects_dir=projects_dir,
        assets_dir=assets_dir,
        cache_dir=cache_dir,
        cache_extracted=cache_extracted,
        cache_renders=cache_renders,
        default_latex_engine=(_get_env("DOCWRITING_LATEX_ENGINE", "auto") or "auto").lower(),
        _raw_env={
            k: v for k, v in os.environ.items() if k.startswith("DOCWRITING_")
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


if __name__ == "__main__":
    print(settings.summary())
    sys.exit(0 if settings.tex_ready else 1)
