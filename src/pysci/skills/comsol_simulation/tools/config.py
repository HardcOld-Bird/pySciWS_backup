"""统一配置加载 + COMSOL 安装发现 + 资源护栏。

从项目根 ``.env`` 读取设置，并暴露：
- COMSOL 安装信息（root / jvm / server / batch / doc-pdf 目录），经 ``mph.discovery`` 自动发现
- 技能数据区各子目录（docs / cache / recipes / templates / knowledge / runs，自动创建）
- 资源护栏（求解线程上限、磁盘 tempdir），适配单机 license + 有限空闲内存
- 复用的 PDF 抽取设置（MinerU token / 后端），供 docs.py 转换手册

新增可选 ``.env`` 键（均有默认值，不加也能工作）：
- ``COMSOL_INSTALL_DIR`` — 覆盖自动发现的 COMSOL 安装根（默认自动发现）
- ``COMSOL_MAX_CORES``   — 求解线程上限（默认 4；单机 12 核 + ~5GB 空闲内存下的保守值）

用法::

    from pysci.skills.comsol_simulation.tools.config import settings, discover_comsol

    settings.comsol_max_cores
    settings.install.root            # COMSOL 安装根（或 None）
    settings.manual_path("COMSOL_Multiphysics/COMSOL_ProgrammingReferenceManual.pdf")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import COMSOL_ROOT, PROJECT_ROOT

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
# 路径统一由 pysci.paths 收口（标记法查找项目根）。MODULE_DIR 指向 COMSOL 仿真数据区
# data/skills/comsol_simulation/（docs/cache/recipes/templates/knowledge/runs 的父目录）。
MODULE_DIR: Path = COMSOL_ROOT

#: 核心手册优先转换集（相对于 ``<COMSOL_INSTALL>/doc/pdf``）。
#: 波动声学 + API 命令参考 + 后处理 + 全集参考，覆盖自动建模闭环的高频查阅需求。
PRIORITY_MANUALS: tuple[str, ...] = (
    "COMSOL_Multiphysics/COMSOL_ProgrammingReferenceManual.pdf",
    "COMSOL_Multiphysics/ApplicationProgrammingGuide.pdf",
    "Acoustics_Module/AcousticsModuleUsersGuide.pdf",
    "COMSOL_Multiphysics/COMSOL_PostprocessingAndVisualization.pdf",
    "COMSOL_Multiphysics/COMSOL_ReferenceManual.pdf",
)

#: Windows 下 COMSOL 可执行文件相对安装根的子路径。
_BINDIR_RELPATH: str = "bin/win64"


# ---------------------------------------------------------------------------
# .env 加载（与 literature_research.config 同款：dotenv 优先，手写解析兜底）
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
            f"[comsol.config] WARNING: {key}={v!r} 不是整数，回退默认 {default}",
            file=sys.stderr,
        )
        return default


# ---------------------------------------------------------------------------
# COMSOL 安装发现
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComsolInstall:
    """COMSOL 安装的关键路径与版本。``found=False`` 时表示未发现可用安装。"""

    found: bool = False
    version: str | None = None
    root: Path | None = None
    jvm: Path | None = None
    server_exe: Path | None = None
    batch_exe: Path | None = None
    doc_pdf_dir: Path | None = None
    source: str = "none"  # "env" | "mph-discovery" | "none"

    def manual_path(self, rel: str) -> Path | None:
        """解析某本手册的绝对路径（rel 相对 doc/pdf）；未安装或文件缺失返回 None。"""
        if self.doc_pdf_dir is None:
            return None
        p = self.doc_pdf_dir / rel
        return p if p.exists() else None

    def priority_manuals(self) -> list[Path]:
        """返回优先转换集中本机实际存在的手册路径。"""
        out: list[Path] = []
        for rel in PRIORITY_MANUALS:
            p = self.manual_path(rel)
            if p is not None:
                out.append(p)
        return out


def _derive_from_root(root: Path, *, source: str) -> ComsolInstall:
    """由安装根推导 server/batch/doc 路径与版本（不启动 JVM）。"""
    bindir = root / _BINDIR_RELPATH
    server = bindir / "comsolmphserver.exe"
    batch = bindir / "comsolbatch.exe"
    doc_pdf = root / "doc" / "pdf"
    jvm = root / "java" / "win64" / "jre" / "bin" / "server" / "jvm.dll"
    # 版本：从形如 .../comsol/6.4/base 的路径尽力推断
    version = None
    for part in root.parts:
        if "." in part and part.replace(".", "").isdigit():
            version = part
            break
    return ComsolInstall(
        found=root.exists(),
        version=version,
        root=root if root.exists() else None,
        jvm=jvm if jvm.exists() else None,
        server_exe=server if server.exists() else None,
        batch_exe=batch if batch.exists() else None,
        doc_pdf_dir=doc_pdf if doc_pdf.exists() else None,
        source=source,
    )


def discover_comsol(install_dir: str | None = None) -> ComsolInstall:
    """发现本机 COMSOL 安装（廉价：读注册表/路径，不启动 JVM、不占 license）。

    优先级：
    1. 显式 ``install_dir`` 或 ``.env`` 的 ``COMSOL_INSTALL_DIR``；
    2. ``mph.discovery.backend()``（读 Windows 注册表 + 磁盘探测）；
    3. 都没有 → ``ComsolInstall(found=False)``。
    """
    env_dir = install_dir or _get_env("COMSOL_INSTALL_DIR")
    if env_dir:
        root = Path(env_dir).expanduser()
        if root.exists():
            return _derive_from_root(root, source="env")
        print(
            f"[comsol.config] WARNING: COMSOL_INSTALL_DIR={root} 不存在，回退自动发现",
            file=sys.stderr,
        )

    try:
        from mph import discovery as _disc  # 延迟导入：避免无 mph 时阻断本模块
    except ImportError:
        return ComsolInstall(found=False, source="none")

    try:
        backend = _disc.backend()
    except Exception as e:  # noqa: BLE001 - 发现失败不应中断导入
        print(f"[comsol.config] WARNING: mph 自动发现失败：{e!r}", file=sys.stderr)
        return ComsolInstall(found=False, source="none")

    root_raw = backend.get("root") if isinstance(backend, dict) else None
    if not root_raw:
        return ComsolInstall(found=False, source="none")

    root = Path(root_raw)
    inst = _derive_from_root(root, source="mph-discovery")
    # 用 mph 报告的精确版本/jvm/server 覆盖推导值（更权威）
    major, minor, patch = backend.get("major"), backend.get("minor"), backend.get("patch")
    version = (
        f"{major}.{minor}.{patch}"
        if None not in (major, minor, patch)
        else (inst.version or None)
    )
    jvm_raw = backend.get("jvm")
    server_raw = backend.get("server")
    server_exe = Path(server_raw[0]) if isinstance(server_raw, (list, tuple)) and server_raw else inst.server_exe
    return ComsolInstall(
        found=True,
        version=version,
        root=root,
        jvm=Path(jvm_raw) if jvm_raw else inst.jvm,
        server_exe=server_exe,
        batch_exe=inst.batch_exe,
        doc_pdf_dir=inst.doc_pdf_dir,
        source="mph-discovery",
    )


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理 comsol_simulation 技能的配置项。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    docs_dir: Path          # MinerU 转换后的手册 Markdown
    cache_dir: Path         # 缓存根
    doc_index_db: Path      # SQLite FTS5 索引 cache/doc_index.db
    recipes_dir: Path       # 固化的建模配方
    templates_dir: Path     # 种子 .mph 模板
    knowledge_dir: Path     # Java→Python 对照、踩坑笔记
    runs_dir: Path          # 运行日志/导出默认落盘
    tempdir: Path           # COMSOL 求解磁盘临时目录

    # --- COMSOL 安装与资源护栏 ---
    install: ComsolInstall
    comsol_max_cores: int

    # --- 复用的 PDF 抽取设置 ---
    pdf_extract_backend: str
    mineru_token: str | None

    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 便捷判断 ----------
    @property
    def comsol_ready(self) -> bool:
        """是否发现可用的 COMSOL 安装（root 存在）。"""
        return self.install.found and self.install.root is not None

    @property
    def mineru_ready(self) -> bool:
        """MinerU 云端转换是否可用（配置了 token）。"""
        return bool(self.mineru_token)

    def manual_path(self, rel: str) -> Path | None:
        """透传到 install.manual_path。"""
        return self.install.manual_path(rel)

    def summary(self) -> str:
        """人类可读的配置摘要（敏感字段脱敏）。"""

        def mask(v: str | None) -> str:
            if not v:
                return "(unset)"
            return "***" if len(v) <= 6 else f"{v[:3]}***{v[-3:]}"

        inst = self.install
        lines = [
            "=== pySciWS comsol_simulation configuration ===",
            f"project_root        : {self.project_root}",
            f"module_dir          : {self.module_dir}",
            f"docs_dir            : {self.docs_dir}",
            f"doc_index_db        : {self.doc_index_db}",
            f"runs_dir            : {self.runs_dir}",
            f"tempdir             : {self.tempdir}",
            "",
            f"comsol_found        : {inst.found} (source={inst.source})",
            f"comsol_version      : {inst.version or '(unknown)'}",
            f"comsol_root         : {inst.root or '(none)'}",
            f"comsol_server_exe   : {inst.server_exe or '(none)'}",
            f"comsol_batch_exe    : {inst.batch_exe or '(none)'}",
            f"comsol_doc_pdf_dir  : {inst.doc_pdf_dir or '(none)'}",
            f"comsol_max_cores    : {self.comsol_max_cores}",
            "",
            f"pdf_extract_backend : {self.pdf_extract_backend}",
            f"mineru_token        : {mask(self.mineru_token)}",
            f"mineru_ready        : {self.mineru_ready}",
            "===============================================",
        ]
        return "\n".join(lines)


def build_settings() -> Settings:
    """加载 .env、发现 COMSOL、构造 Settings。技能数据目录会自动创建。"""
    _load_dotenv_if_available()

    docs_dir = MODULE_DIR / "docs"
    cache_dir = MODULE_DIR / "cache"
    recipes_dir = MODULE_DIR / "recipes"
    templates_dir = MODULE_DIR / "templates"
    knowledge_dir = MODULE_DIR / "knowledge"
    runs_dir = MODULE_DIR / "runs"
    tempdir = runs_dir / "tmp"
    for d in (docs_dir, cache_dir, recipes_dir, templates_dir, knowledge_dir, runs_dir, tempdir):
        d.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        docs_dir=docs_dir,
        cache_dir=cache_dir,
        doc_index_db=cache_dir / "doc_index.db",
        recipes_dir=recipes_dir,
        templates_dir=templates_dir,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
        tempdir=tempdir,
        install=discover_comsol(),
        comsol_max_cores=_get_env_int("COMSOL_MAX_CORES", 4),
        pdf_extract_backend=(_get_env("PDF_EXTRACT_BACKEND", "auto") or "auto").lower(),
        mineru_token=_get_env("MINERU_TOKEN"),
        _raw_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("COMSOL_", "MINERU_", "PDF_"))
        },
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


# ---------------------------------------------------------------------------
# CLI: 打印当前配置摘要（等价于 simulation.py doctor 的配置部分）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(settings.summary())
