"""Blender headless 驱动：``blender -b -P script.py`` 子进程封装。

渲染管线唯一的 Blender 入口。交互探索请用 blender-mcp（addon + MCP server，
Qoder 会话内直接可用，见 ``.qoder/skills/modeling3d/references/mcp.md``）；
但**交付物必须出自本模块驱动的 headless 脚本**（可复现、可进版本库）。

用法::

    from pysci.skills.modeling3d.tools.blender import run_blender_script, blender_version

    print(blender_version())
    rc = run_blender_script(scene_py, extra_args=["--mesh", stl, "--out", png])
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import settings

#: 包内随附的功能性 Blender 脚本（studio 灯光/相机/材质模板）。
#: 与 data/skills/modeling3d/templates/ 的脚手架模板不同：本目录脚本是 CLI 的
#: 组成部分，由 ``model3d render --mesh`` 直接调用，不应手工修改。
BLENDER_SCRIPTS_DIR: Path = Path(__file__).resolve().parent.parent / "blender_scripts"

#: 快速渲染入口脚本（studio 布光 + 自动取景 + 引擎回退链）。
RENDER_STUDIO_SCRIPT: Path = BLENDER_SCRIPTS_DIR / "render_studio.py"


class BlenderNotFound(RuntimeError):
    """未探测到 Blender 可执行文件。"""

    def __init__(self) -> None:
        super().__init__(
            "未找到 Blender：在项目根 .env 设 MODELING3D_BLENDER=<blender.exe 绝对路径>"
        )


def blender_exe() -> Path:
    """返回 Blender 可执行文件路径。

    Raises:
        BlenderNotFound: 未配置且探测失败。
    """
    exe = settings.blender_exe
    if exe is None or not exe.exists():
        raise BlenderNotFound()
    return exe


def blender_version() -> str:
    """``blender --version`` 的首行（如 ``Blender 5.1.1``）；不可用时返回错误描述。"""
    try:
        out = subprocess.run(
            [str(blender_exe()), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"UNAVAILABLE ({e!r})"


def run_blender_script(
    script: str | Path,
    *,
    extra_args: list[str] | None = None,
    stream: bool = False,
    timeout: float | None = 1800.0,
) -> int:
    """以 headless 模式执行 Blender Python 脚本。

    命令形态：``blender -b -P <script> -- <extra_args...>``；脚本内用
    ``sys.argv[sys.argv.index("--") + 1:]`` 取参。

    Args:
        script: Blender 内执行的 .py 脚本（跑在 Blender 自带解释器中，非项目 .venv）。
        extra_args: 透传给脚本的参数（置于 ``--`` 之后）。
        stream: True → 实时透传 stdout/stderr（长渲染推荐）；False → 捕获后转发。
        timeout: 子进程超时秒数（默认 30 分钟）。

    Returns:
        Blender 进程退出码（0 = 成功）。
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        print(f"[model3d] Blender 脚本不存在: {script_path}", file=sys.stderr)
        return 2

    cmd = [str(blender_exe()), "-b", "-P", str(script_path)]
    if extra_args:
        cmd.append("--")
        cmd.extend(str(a) for a in extra_args)

    print(f"[model3d] 执行: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=not stream,
        text=True,
        timeout=timeout,
        cwd=str(settings.project_root),
    )
    if not stream:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Blender 把大量正常信息写 stderr（如版本横幅），只原样转发不判错。
            print(result.stderr, file=sys.stderr)
    return result.returncode
