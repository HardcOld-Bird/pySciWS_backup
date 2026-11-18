"""Blender 自检——路径 / Cycles CPU / 内存预算 / 单实例（⑥域）。

硬件红线（本机：仅 CPU、无独显、16GB 内存可用约 5GB）：

- **EEVEE 需 GPU，本机不可用**——渲染引擎只有 Cycles CPU 一条路；
- **内存预算 ~5GB**：单帧低采样、**禁并行多实例**（:func:`ensure_single_instance`）；
- Blender 由基础设施层 vendor 至 ``deps_management/deps/blender/<版本>/``（路径常量先行，
  :data:`BLENDER_VENDOR_DIR`），也可用 ``BLENDER_EXE`` 环境变量覆盖（指向 ``blender.exe``
  或其所在目录；环境契约见 :data:`BLENDER_ENV`）。

``blender --version`` / ``tasklist`` 的输出解析为纯函数（可机械断言）；真实进程探测
只在 Windows 上进行，其余平台返回空值（fail-open，doctor 报告而非硬失败）。

坑注记（T2 实测，2026-11-17）：mcp 客户端 ``stdio_client`` 不显式传 ``env`` 时会剥掉
自定义环境变量（只留 get_default_environment 安全子集）——胶水/测试里起 MCP 或其他
子进程需要自定义 env 时**显式传 env**，别依赖继承。
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pysci.toolchain import DEPS_ROOT

#: Blender 可执行覆盖变量（环境变量优先于 vendor 目录）。环境契约（T2 基础设施
#: 约定，2026-11-17）：``BLENDER_EXE`` 指向 ``blender.exe``，薄胶水集中读取；
#: blender-mcp 本体不读它（它用 ``BLENDER_HOST``/``BLENDER_PORT``=9876 连 addon）。
BLENDER_ENV = "BLENDER_EXE"

#: Blender vendor 根（并行代理落盘处；含 ``blender.exe`` 或单层版本子目录）。
BLENDER_VENDOR_DIR: Path = DEPS_ROOT / "blender"

#: 内存预算（GB）：单实例 Cycles CPU 渲染可用上限（16GB 机型留系统/venv 余量）。
MEMORY_BUDGET_GB = 5.0

_VERSION_RE = re.compile(r"^Blender\s+(?P<ver>\S+)", re.MULTILINE)
_TASKLIST_RE = re.compile(r'^"(?P<image>[^"]+)","(?P<pid>\d+)"')


def find_blender(
    *,
    env: str | None = None,
    vendor_dir: Path | str | None = None,
) -> Path | None:
    """定位 ``blender.exe``：环境变量 → vendor 目录（浅层优先）。

    Args:
        env: 覆盖变量值（测试注入用）；缺省读 :data:`BLENDER_ENV`。
        vendor_dir: 覆盖 vendor 根（测试注入用）；缺省 :data:`BLENDER_VENDOR_DIR`。

    Returns:
        ``blender.exe`` 路径；找不到返回 None。
    """
    env_val = env if env is not None else os.environ.get(BLENDER_ENV)
    if env_val:
        p = Path(env_val)
        if p.is_file():
            return p
        if p.is_dir():
            for cand in sorted(p.rglob("blender.exe")):
                return cand
    root = Path(vendor_dir) if vendor_dir is not None else BLENDER_VENDOR_DIR
    if root.is_dir():
        direct = root / "blender.exe"
        if direct.is_file():
            return direct
        hits = sorted(root.glob("*/blender.exe")) + sorted(root.glob("*/*/blender.exe"))
        if hits:
            return hits[0]
    return None


def parse_blender_version(output: str) -> str | None:
    """从 ``blender --version`` 输出解析版本号（纯函数）。"""
    m = _VERSION_RE.search(output)
    return m.group("ver") if m else None


def blender_version(exe: Path | str) -> str | None:
    """运行 ``<exe> --version`` 探测版本号；失败返回 None（fail-open）。"""
    try:
        proc = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_blender_version(proc.stdout or proc.stderr)


def parse_tasklist_pids(output: str) -> list[int]:
    """从 ``tasklist /FO CSV /NH`` 输出解析在跑的 blender.exe PID（纯函数）。"""
    pids = []
    for line in output.splitlines():
        m = _TASKLIST_RE.match(line.strip())
        if m and m.group("image").lower() == "blender.exe":
            pids.append(int(m.group("pid")))
    return pids


def blender_processes() -> list[int]:
    """列出在跑的 Blender 进程 PID（仅 Windows；其余平台/失败返回 []）。"""
    if os.name != "nt":
        return []
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq blender.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_tasklist_pids(proc.stdout or "")


def ensure_single_instance() -> None:
    """单实例红线：已有 Blender 在跑则拒绝启动（内存预算禁并行）。

    Raises:
        RuntimeError: 存在其他 Blender 进程。
    """
    pids = blender_processes()
    if pids:
        raise RuntimeError(
            f"已有 Blender 实例在跑（PID {pids}）：内存预算 {MEMORY_BUDGET_GB} GB "
            "禁并行多实例，等待其结束或先关闭"
        )


def available_memory_gb() -> float | None:
    """当前可用物理内存（GB，Windows ctypes；探测失败返回 None）。"""
    if os.name != "nt":
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
    except (AttributeError, OSError):
        return None
    return stat.ullAvailPhys / (1024**3)


@dataclass
class DoctorReport:
    """自检结果（逐项状态 + 建议）。"""

    ok: bool
    blender_path: Path | None
    blender_version: str | None
    memory_budget_gb: float
    available_memory_gb: float | None
    running_pids: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def format(self) -> str:
        """人类可读自检报告（中文）。"""
        lines = [
            "=== modeling-3d doctor ===",
            f"  blender        : {self.blender_path or 'NOT FOUND'}",
            f"  version        : {self.blender_version or 'N/A'}",
            "  render engine  : CYCLES device=CPU（EEVEE 需 GPU，本机不可用）",
            f"  memory budget  : {self.memory_budget_gb:.1f} GB"
            + (
                f"（可用 {self.available_memory_gb:.1f} GB）"
                if self.available_memory_gb is not None
                else "（可用量未知）"
            ),
            f"  instances      : {self.running_pids or '无其他 Blender 实例'}",
            "  RESULT         : OK ✅" if self.ok else "  RESULT         : 有问题 ❌",
        ]
        lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def doctor(*, probe_version: bool = True) -> DoctorReport:
    """跑全量自检：Blender 路径/版本、渲染引擎口径、内存预算、单实例。

    Args:
        probe_version: 是否真实执行 ``blender --version``（纯函数测试传 False）。

    Returns:
        :class:`DoctorReport`。``ok=False`` 当 Blender 缺失或存在并行实例。
    """
    notes: list[str] = []
    exe = find_blender()
    if exe is None:
        notes.append(
            f"Blender 未找到：等待 vendor 落盘 {BLENDER_VENDOR_DIR}，"
            f"或设 {BLENDER_ENV} 环境变量"
        )
    version = blender_version(exe) if (probe_version and exe) else None
    if exe is not None and version is None and probe_version:
        notes.append("版本探测失败：确认 blender.exe 可执行（--version 应输出 'Blender x.y.z'）")
    avail = available_memory_gb()
    if avail is not None and avail < MEMORY_BUDGET_GB:
        notes.append(
            f"可用内存 {avail:.1f} GB 已低于预算 {MEMORY_BUDGET_GB} GB：暂缓渲染，先释放内存"
        )
    pids = blender_processes()
    if pids:
        notes.append(f"存在其他 Blender 实例（PID {pids}）：禁并行，先收尾其一")
    return DoctorReport(
        ok=exe is not None and not pids,
        blender_path=exe,
        blender_version=version,
        memory_budget_gb=MEMORY_BUDGET_GB,
        available_memory_gb=avail,
        running_pids=pids,
        notes=notes,
    )
