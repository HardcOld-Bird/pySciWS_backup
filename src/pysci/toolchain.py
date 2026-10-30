"""vendored 工具链解析（自包含依赖层，``sub_root/dependencies/``）。

按与 mySimsWS2 同范式，工具链以项目级自包含方式部署在
``sub_root/dependencies/``：``<tool>/<version>/`` 版本目录 + ``<tool>/current.txt``
当前版本指针 + ``toolchain.json`` 唯一版本真源（url/sha256 可复现下载）。
本模块把各工具的可执行文件解析为**绝对路径**，供 subprocess 调用——Windows 上
Python ``subprocess`` 找不到 PATH 里的 ``.cmd`` shim（PATHEXT 不适用），必须给真
``.exe``；交互 shell 场景才走 ``D:\\XXXIIIGGG\\common\\bin`` 的共享 shim 路由层。

解析规则（fail-open 降级）：vendored 目录缺失时回退裸命令名（``"git"``/``"uv"``…，
交 PATH 解析）——冷启动/换机尚未跑 ``init-deps`` 时不废掉调用方；缺失状态由
``pysci-dev doctor`` 巡检暴露，不在解析层硬失败。

用法::

    from pysci.toolchain import git_exe, uv_exe

    subprocess.run([str(git_exe()), "status"], cwd=PROJECT_ROOT)
"""

from __future__ import annotations

import os
from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 自包含依赖根（与 mySimsWS2 布局同相对路径，共享 shim 层据此向上路由）。
DEPS_ROOT: Path = PROJECT_ROOT / "sub_root" / "dependencies"

#: 项目内 uv tool 安装区（UV_TOOL_DIR/UV_TOOL_BIN_DIR 指向此处）。
UV_TOOL_DIR: Path = DEPS_ROOT / "uv-tools"

#: 项目内 uv tool 可执行目录（zotero-mcp / zotero-cli 等 console script 落这里）。
UV_TOOL_BIN_DIR: Path = UV_TOOL_DIR / "bin"

#: 项目内 uv 缓存（UV_CACHE_DIR 指向此处）。
UV_CACHE_DIR: Path = DEPS_ROOT / "cache" / "packages" / "uv"


def _current_version(tool: str) -> str | None:
    """读 ``<tool>/current.txt`` 版本指针。

    Args:
        tool: 工具名（``git`` / ``node`` / ``uv`` / ``python``）。

    Returns:
        已落盘版本号字符串；指针文件缺失或为空返回 ``None``。
    """
    pointer = DEPS_ROOT / tool / "current.txt"
    try:
        version = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return version or None


def _tool_dir(tool: str) -> Path | None:
    """解析 ``<tool>/<version>/`` 安装目录。

    Args:
        tool: 工具名。

    Returns:
        版本目录路径；未落盘返回 ``None``。
    """
    version = _current_version(tool)
    if version is None:
        return None
    install = DEPS_ROOT / tool / version
    return install if install.is_dir() else None


def _resolve(tool: str, *rel: str, fallback: str) -> str:
    """解析工具可执行文件：vendored 绝对路径优先，缺失回退裸命令名。

    Args:
        tool: 工具名（定位 ``<tool>/<version>/``）。
        *rel: 可执行文件相对版本目录的路径段（如 ``"cmd"``, ``"git.exe"``）。
        fallback: vendored 缺失时的裸命令名（交 PATH 解析）。

    Returns:
        可执行文件路径字符串。
    """
    install = _tool_dir(tool)
    if install is not None:
        exe = install.joinpath(*rel)
        if exe.is_file():
            return str(exe)
    return fallback


def git_exe() -> str:
    """vendored git 可执行文件（``git/<ver>/cmd/git.exe``），缺失回退 ``"git"``。"""
    return _resolve("git", "cmd", "git.exe", fallback="git")


def bash_exe() -> str:
    """vendored Git Bash（``git/<ver>/usr/bin/bash.exe``），缺失回退 ``"bash"``。

    hook/guard 等须以**绝对路径**调用——按名解析 ``bash`` 在系统 Git 移除后会落到
    WindowsApps 的 WSL 别名（2026-10-25 实案：每次 hook 弹 wsl.exe + conhost）。
    """
    return _resolve("git", "usr", "bin", "bash.exe", fallback="bash")


def uv_exe() -> str:
    """vendored uv 可执行文件（``uv/<ver>/uv.exe``），缺失回退 ``"uv"``。"""
    return _resolve("uv", "uv.exe", fallback="uv")


def uvx_exe() -> str:
    """vendored uvx 可执行文件（``uv/<ver>/uvx.exe``），缺失回退 ``"uvx"``。"""
    return _resolve("uv", "uvx.exe", fallback="uvx")


def node_exe() -> str:
    """vendored node 可执行文件（``node/<ver>/node.exe``），缺失回退 ``"node"``。"""
    return _resolve("node", "node.exe", fallback="node")


def base_python_exe() -> str:
    """vendored 基座解释器（``python/<ver>/python.exe``），缺失回退 ``"python"``。

    这是 uv 管理的 standalone CPython（venv 的 ``pyvenv.cfg`` home 指向它）；
    项目运行时解释器一律用 ``.venv/Scripts/python.exe``，本函数用于需要**基座**
    解释器的场景（如测试夹具、venv 重建）。
    """
    install = _tool_dir("python")
    if install is not None:
        exe = install / "python.exe"
        if exe.is_file():
            return str(exe)
    return "python"


def venv_scripts_dir() -> Path:
    """项目 venv 的 Scripts 目录（``<root>/.venv/Scripts``）。"""
    return PROJECT_ROOT / ".venv" / "Scripts"


def venv_python_exe() -> Path:
    """项目 venv 解释器（``<root>/.venv/Scripts/python.exe``）。"""
    return venv_scripts_dir() / "python.exe"


def venv_pythonw_exe() -> Path:
    """项目 venv GUI 解释器（``<root>/.venv/Scripts/pythonw.exe``，无控制台）。"""
    return venv_scripts_dir() / "pythonw.exe"


def uv_tool_exe(name: str) -> str:
    """项目内 uv tool 的 console script（``uv-tools/bin/<name>.exe``）。

    Args:
        name: console script 名（如 ``zotero-cli`` / ``zotero-mcp``）。

    Returns:
        绝对路径字符串；未安装时返回裸名字（交 PATH 解析，兼容过渡期）。
    """
    exe = UV_TOOL_BIN_DIR / f"{name}.exe"
    return str(exe) if exe.is_file() else name


def resolve_command(name: str) -> str:
    """把裸命令名解析为 vendored 绝对路径（注册表命令模板等数据驱动场景用）。

    Args:
        name: 裸命令名（``git`` / ``uv`` / ``uvx`` / ``node`` / ``bash``）。

    Returns:
        vendored 绝对路径；未纳入工具链的名字原样返回（交 PATH 解析）。
    """
    resolvers = {
        "git": git_exe,
        "uv": uv_exe,
        "uvx": uvx_exe,
        "node": node_exe,
        "bash": bash_exe,
    }
    resolve = resolvers.get(name)
    return resolve() if resolve is not None else name


def toolchain_path_dirs() -> list[Path]:
    """vendored 工具的可执行目录清单（前插 PATH 用）。

    Returns:
        依次为 git ``cmd``、git ``usr/bin``（bash/coreutils）、node、uv、uv-tools/bin
        中**已落盘**的目录。
    """
    dirs: list[Path] = []
    git = _tool_dir("git")
    if git is not None:
        dirs.extend((git / "cmd", git / "usr" / "bin"))
    node = _tool_dir("node")
    if node is not None:
        dirs.append(node)
    uv = _tool_dir("uv")
    if uv is not None:
        dirs.append(uv)
    dirs.append(UV_TOOL_BIN_DIR)
    return [d for d in dirs if d.is_dir()]


def prepend_toolchain_path(path: str | None = None) -> str:
    """把 vendored 工具目录前插进 PATH 字符串。

    Windows 上 ``subprocess`` 找不到 PATH 里的 ``.cmd`` shim，第三方工具（pre-commit）
    与测试里的裸 ``node``/``git``/``bash`` 调用都需要**真 .exe** 在 PATH 上——本函数把
    ``sub_root/dependencies`` 下的真实工具目录前插，一处修复全量生效。

    Args:
        path: 原 PATH 字符串；缺省取 ``os.environ["PATH"]``。

    Returns:
        前插后的 PATH 字符串。
    """
    current = path if path is not None else os.environ.get("PATH", "")
    parts = [str(d) for d in toolchain_path_dirs()]
    return os.pathsep.join([*parts, current]) if parts else current
