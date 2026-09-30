"""ComfyUI 编排器服务器生命周期（镜像 comsol_simulation.session 的常驻 server 模式）。

ComfyUI 由 **comfy-cli** 装在独立环境（不进 pysci 依赖）。本模块负责：探活、（best-effort）
经 ``comfy`` CLI 拉起/停止无头服务器、跨进程状态文件（``runs/comfy_server.json``）以便多条
CLI 命令复用同一服务器（省冷启动），以及端口探活 + taskkill 兜底。

三种现实形态：
1. **外部已起**（推荐，Phase 0 手动 ``comfy launch --background -- --cpu``）：本模块只探活/记录。
2. **本模块拉起**：``comfy server start`` shell out 到 ``comfy`` CLI（需在 PATH 或设 ``COMFY_CLI``）。
3. **远程/云端**：``COMFY_SERVER_URL`` 指向别处，本模块只探活，不管理其生命周期。

.. note::
   与 comsol 不同，ComfyUI 无 license 抢占问题；但无头 ``--cpu`` 服务器仍占内存，用完
   ``comfy server stop`` 释放。GUI 可同端口 ``:8188`` 并存，供用户实时协作查看。

用法::

    from pysci.skills.ai_drawing.tools import comfy_session

    comfy_session.status()
    comfy_session.launch(cpu=True)   # best-effort 经 comfy CLI
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import settings


class SessionError(RuntimeError):
    """服务器拉起/停止/探测失败。"""


_SERVER_STATE_NAME = "comfy_server.json"


def _state_path() -> Path:
    return settings.runs_dir / _SERVER_STATE_NAME


def _url_host_port(url: str) -> tuple[str, int]:
    """从 server_url 解析 (host, port)。"""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8188)
    return host, port


def _port_alive(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_comfy_cli() -> Path | None:
    """定位 ``comfy`` CLI 可执行文件：``COMFY_CLI`` env 优先，其次 PATH。"""
    import os

    env = os.environ.get("COMFY_CLI")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    found = shutil.which("comfy")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# 可用性探测（不启服务、不联网之外的重活）
# ---------------------------------------------------------------------------
def check_available() -> dict[str, Any]:
    """返回 ComfyUI 编排器可用性诊断，供 doctor 与测试守卫使用（廉价）。"""
    host, port = _url_host_port(settings.comfy_server_url)
    cli = _find_comfy_cli()
    return {
        "comfy_cli": str(cli) if cli else None,
        "comfy_root": str(settings.comfy.root) if settings.comfy.root else None,
        "comfy_root_configured": settings.comfy.found,
        "server_url": settings.comfy_server_url,
        "host": host,
        "port": port,
        "port_alive": _port_alive(host, port),
    }


def is_available() -> bool:
    """服务器当前是否可达（端口存活）。这是 Tier 1 命令的前置门槛。"""
    host, port = _url_host_port(settings.comfy_server_url)
    return _port_alive(host, port)


def require_available() -> None:
    """不可达则抛 :class:`SessionError`（附设置指引）。"""
    if not is_available():
        raise SessionError(
            f"ComfyUI 服务器不可达：{settings.comfy_server_url}\n"
            "  请先启动它（任选其一）：\n"
            "    1) 外部：在 comfy-cli 环境跑 `comfy launch --background -- --cpu`\n"
            "    2) 本 CLI：`pysci-imagine comfy server start --cpu`（需 comfy 在 PATH 或设 COMFY_CLI）\n"
            "  并把 COMFY_ROOT / COMFY_SERVER_URL 写入项目根 .env。详见 references/comfyui.md。"
        )


# ---------------------------------------------------------------------------
# 状态文件
# ---------------------------------------------------------------------------
def _read_state() -> dict[str, Any] | None:
    sp = _state_path()
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(info: dict[str, Any]) -> None:
    sp = _state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_state() -> None:
    _state_path().unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# launch / stop / status
# ---------------------------------------------------------------------------
def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_alive(host, port, timeout=1.0):
            return True
        time.sleep(1.0)
    return False


def launch(
    *,
    cpu: bool = True,
    port: int | None = None,
    startup_timeout: float = 120.0,
) -> dict[str, Any]:
    """（best-effort）经 ``comfy`` CLI 拉起无头 ComfyUI 并等待端口就绪，记录状态文件。

    幂等：若服务器已可达，直接记录/复用，不重复拉起。

    Args:
        cpu: 传 ``--cpu``（本机无 CUDA，默认 True）。
        port: 覆盖端口（默认取 ``COMFY_SERVER_URL`` 的端口）。
        startup_timeout: 等待端口就绪的超时（秒）。

    Raises:
        SessionError: 找不到 ``comfy`` CLI，或拉起后端口未就绪。
    """
    host, cur_port = _url_host_port(settings.comfy_server_url)
    target_port = int(port) if port else cur_port

    # 幂等：已在跑就直接记录
    if _port_alive(host, target_port):
        info = {
            "host": host,
            "port": target_port,
            "url": settings.comfy_server_url,
            "pid": None,
            "launched_via": "already-running",
            "started_ts": time.time(),
        }
        _write_state(info)
        return info

    cli = _find_comfy_cli()
    if not cli:
        raise SessionError(
            "找不到 comfy CLI（comfy-cli）。请：\n"
            "  - 在 comfy-cli 所在环境把 `comfy` 加入 PATH，或在 .env 设 COMFY_CLI=<comfy 可执行路径>；\n"
            "  - 或直接在 comfy-cli 环境手动跑 `comfy launch --background -- --cpu`，"
            "本 CLI 随后即可探活复用。"
        )

    args = [str(cli), "launch", "--background"]
    passthrough = []
    if cpu:
        passthrough.append("--cpu")
    if port:
        passthrough += ["--port", str(target_port)]
    if passthrough:
        args += ["--", *passthrough]

    cwd = str(settings.comfy.root) if (settings.comfy.root and settings.comfy.root.exists()) else None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(  # noqa: S603 - comfy CLI 路径来自受信发现/用户配置
            args,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError as e:
        raise SessionError(f"启动 comfy CLI 失败：{e!r}") from e

    if not _wait_for_port(host, target_port, startup_timeout):
        raise SessionError(
            f"ComfyUI 在 {startup_timeout}s 内未在 {host}:{target_port} 就绪。"
            "请检查 comfy-cli 安装/ workspace（`comfy which` / `comfy env`），或手动 `comfy launch`。"
        )

    info = {
        "host": host,
        "port": target_port,
        "url": f"http://{host}:{target_port}",
        "pid": proc.pid,
        "launched_via": f"comfy-cli:{cli}",
        "started_ts": time.time(),
    }
    _write_state(info)
    return info


def _kill_pid(pid: int) -> None:
    """跨平台 best-effort 终止进程（Windows taskkill，POSIX SIGTERM）。"""
    try:
        if sys.platform == "win32":
            subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            import os
            import signal

            os.kill(pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass


def stop() -> dict[str, Any]:
    """停止由本模块拉起的服务器：优先 ``comfy stop``，其次 taskkill 记录的 pid。

    Returns:
        ``{"stopped": bool, "method": str}``。无状态文件且端口本就不可达 → stopped=False。
    """
    state = _read_state()
    cli = _find_comfy_cli()
    method = "none"
    stopped = False

    if cli and (state is None or str(state.get("launched_via", "")).startswith(("comfy-cli", "already"))):
        try:
            subprocess.run(  # noqa: S603
                [str(cli), "stop"],
                capture_output=True,
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            method = "comfy stop"
            stopped = True
        except (OSError, subprocess.SubprocessError):
            method = "comfy stop failed"

    if not stopped and state and state.get("pid"):
        _kill_pid(int(state["pid"]))
        method = f"taskkill {state['pid']}"
        stopped = True

    host, port = _url_host_port(settings.comfy_server_url)
    if not stopped:
        # 最后兜底：若端口仍活着但无 pid/CLI，提示用户手动停
        if _port_alive(host, port):
            return {"stopped": False, "method": "unmanaged (请手动 `comfy stop`)"}
        return {"stopped": False, "method": "not-running"}

    _clear_state()
    return {"stopped": True, "method": method}


def status() -> dict[str, Any]:
    """返回服务器状态（state file + 端口存活 + CLI 可用性），供 ``comfy server status``。"""
    state = _read_state()
    host, port = _url_host_port(settings.comfy_server_url)
    alive = _port_alive(host, port)
    info: dict[str, Any] = {
        "server_url": settings.comfy_server_url,
        "host": host,
        "port": port,
        "running": alive,
        "comfy_cli": str(_find_comfy_cli() or "") or None,
        "comfy_root": str(settings.comfy.root) if settings.comfy.root else None,
        "state_file": str(_state_path()),
        "launched_via": (state or {}).get("launched_via"),
        "pid": (state or {}).get("pid"),
    }
    return info


if __name__ == "__main__":
    for k, v in check_available().items():
        print(f"{k:22}: {v}")
