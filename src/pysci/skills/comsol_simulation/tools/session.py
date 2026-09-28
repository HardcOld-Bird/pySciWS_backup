"""mph 客户端生命周期管理。

三种使用形态：
1. **standalone**（默认）：``mph.start()`` 在进程内启动一个独立 COMSOL 会话（经 JPype
   拉起 COMSOL 自带 JRE）。用完即 ``disconnect()`` 释放内存——适配单机 license + 有限空闲内存。
2. **server**：先拉起常驻 ``comsolmphserver.exe``，再 ``mph.connect()`` 连上。省去每次
   10–30s 冷启动，适合密集迭代；会话结束显式关闭 server 进程，避免僵尸常驻吃内存。
3. **connect**：连到一个已由外部启动的 server（不管理其生命周期）。

会话上下文管理器 :func:`session` 负责异常安全的 teardown。

.. warning::
   单 license 约束：常驻 server（``mode="server"`` / :func:`start_persistent_server`）
   会**占用 license**，切勿与交互式 COMSOL GUI 同时运行，否则二者争抢唯一 license。

用法::

    from pysci.skills.comsol_simulation.tools.session import session

    with session(mode="standalone", cores=4) as client:
        model = client.load("path/to/model.mph")
        model.solve()
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings


class SessionError(RuntimeError):
    """会话建立/连接/teardown 失败。"""


# ---------------------------------------------------------------------------
# 可用性探测（不启动 JVM、不占 license）
# ---------------------------------------------------------------------------
def check_available() -> dict[str, Any]:
    """返回 COMSOL/mph 可用性诊断，供 doctor 与测试守卫使用。"""
    info: dict[str, Any] = {
        "mph_importable": False,
        "mph_version": None,
        "comsol_found": settings.comsol_ready,
        "comsol_version": settings.install.version,
        "server_exe": str(settings.install.server_exe or ""),
    }
    try:
        import mph

        info["mph_importable"] = True
        info["mph_version"] = getattr(mph, "__version__", None)
    except ImportError:
        pass
    return info


def is_available() -> bool:
    """能否尝试建立会话：mph 可导入且发现了 COMSOL 安装。"""
    d = check_available()
    return bool(d["mph_importable"] and d["comsol_found"])


def _require_mph() -> Any:
    try:
        import mph
    except ImportError as e:  # pragma: no cover
        raise SessionError("mph 不可用（COMSOL 仿真需要）：uv sync 安装依赖") from e
    return mph


def _resolve_cores(cores: int | None) -> int:
    """把请求线程数夹到 [1, comsol_max_cores]，None 则用配置默认。"""
    cap = max(1, settings.comsol_max_cores)
    if cores is None:
        return cap
    return max(1, min(int(cores), cap))


# ---------------------------------------------------------------------------
# standalone
# ---------------------------------------------------------------------------
def start_standalone(cores: int | None = None, *, version: str | None = None) -> Any:
    """启动一个独立 COMSOL 会话，返回 ``mph.Client``。

    Args:
        cores: 求解线程数（夹到 config 上限内）。None → 用 ``COMSOL_MAX_CORES``。
        version: 指定 COMSOL 版本（多版本共存时）；None → mph 自动选择。
    """
    mph = _require_mph()
    if not settings.comsol_ready:
        raise SessionError(
            "未发现 COMSOL 安装。请确认已安装，或在 .env 设置 COMSOL_INSTALL_DIR。"
        )
    n = _resolve_cores(cores)
    kwargs: dict[str, Any] = {"cores": n}
    if version:
        kwargs["version"] = version
    try:
        return mph.start(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise SessionError(f"启动 standalone COMSOL 会话失败：{e!r}") from e


# ---------------------------------------------------------------------------
# server 生命周期
# ---------------------------------------------------------------------------
def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """轮询直到 TCP 端口可连接（server 就绪）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@dataclass
class ServerHandle:
    """常驻 comsolmphserver 进程句柄。"""

    proc: subprocess.Popen[bytes]
    host: str
    port: int

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        """优雅终止 server 进程（先 terminate，超时再 kill）。"""
        if not self.alive():
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass


def launch_server(
    port: int = 2036, cores: int | None = None, *, startup_timeout: float = 90.0
) -> ServerHandle:
    """拉起常驻 ``comsolmphserver.exe``，等待端口就绪后返回句柄。"""
    exe = settings.install.server_exe
    if not exe or not Path(exe).exists():
        raise SessionError(f"未找到 comsolmphserver.exe（发现结果：{exe!r}）")
    n = _resolve_cores(cores)
    args = [str(exe), "-port", str(port), "-np", str(n)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(  # noqa: S603 - 路径来自受信的 COMSOL 安装发现
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError as e:  # pragma: no cover
        raise SessionError(f"启动 comsolmphserver 失败：{e!r}") from e

    if not _wait_for_port("127.0.0.1", port, timeout=startup_timeout):
        proc.kill()
        raise SessionError(
            f"comsolmphserver 在 {startup_timeout}s 内未在端口 {port} 就绪（可能 license 占用）"
        )
    return ServerHandle(proc=proc, host="127.0.0.1", port=port)


def connect(host: str = "127.0.0.1", port: int = 2036) -> Any:
    """连接到一个已运行的 COMSOL server，返回 ``mph.Client``。"""
    mph = _require_mph()
    try:
        return mph.connect(host=host, port=port)
    except Exception as e:  # noqa: BLE001
        raise SessionError(f"连接 COMSOL server {host}:{port} 失败：{e!r}") from e


def stop_server(handle: ServerHandle) -> None:
    """停止由 :func:`launch_server` 启动的 server。"""
    handle.stop()


# ---------------------------------------------------------------------------
# 跨进程常驻 server（state file）——多步 CLI 复用同一 JVM
# ---------------------------------------------------------------------------
_SERVER_STATE_NAME = "comsol_server.json"


def _server_state_path() -> Path:
    return settings.runs_dir / _SERVER_STATE_NAME


def _port_alive(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def start_persistent_server(port: int = 2036, cores: int | None = None) -> dict[str, Any]:
    """拉起**跨进程常驻** server 并记录 pid/port 状态文件（供后续 connect / stop）。

    与 ``session(mode="server")`` 的区别：本函数**不**在退出时关闭 server，使多条 CLI
    命令可复用同一 JVM（省每次 ~30s 冷启动与 license 抖动）。若状态文件显示 server 已在
    运行则直接复用（幂等）。

    .. warning:: 常驻 server 占用 license，**不要**与交互式 COMSOL GUI 同时运行。
    """
    state = _server_state_path()
    if state.exists():
        try:
            info = json.loads(state.read_text(encoding="utf-8"))
            if _port_alive(str(info.get("host", "127.0.0.1")), int(info.get("port", 0) or 0)):
                return info
        except Exception:  # noqa: BLE001
            pass
    handle = launch_server(port=port, cores=cores)
    info = {"pid": handle.proc.pid, "host": handle.host, "port": handle.port}
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(info), encoding="utf-8")
    return info


def server_status(port: int | None = None) -> dict[str, Any]:
    """返回常驻 server 状态（state file 内容 + 端口是否存活）。"""
    state = _server_state_path()
    info: dict[str, Any] = {
        "state_file": str(state),
        "running": False,
        "pid": None,
        "host": "127.0.0.1",
        "port": port,
    }
    if state.exists():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            info.update(pid=data.get("pid"), port=data.get("port"), host=data.get("host", "127.0.0.1"))
            info["running"] = _port_alive(info["host"], int(info["port"] or 0))
        except Exception:  # noqa: BLE001
            pass
    elif port:
        info["running"] = _port_alive("127.0.0.1", port)
    return info


def _kill_pid(pid: int) -> None:
    """跨平台 best-effort 终止进程（Windows 用 taskkill，POSIX 用 SIGTERM）。"""
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


def stop_persistent_server() -> bool:
    """按 state file 记录的 pid 终止常驻 server 并清理状态文件。无状态文件返回 False。"""
    state = _server_state_path()
    if not state.exists():
        return False
    try:
        pid = int(json.loads(state.read_text(encoding="utf-8")).get("pid", 0))
    except Exception:  # noqa: BLE001
        pid = 0
    if pid:
        _kill_pid(pid)
    state.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# 统一上下文管理器
# ---------------------------------------------------------------------------
def _disconnect(client: Any) -> None:
    try:
        client.disconnect()
    except Exception as e:  # noqa: BLE001
        print(f"[comsol.session] WARNING: 断开会话时出现警告：{e!r}", file=sys.stderr)


@contextmanager
def session(
    mode: str = "standalone",
    cores: int | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 2036,
    manage_server: bool = True,
    version: str | None = None,
) -> Iterator[Any]:
    """建立一个 COMSOL 会话并在退出时安全 teardown。

    Args:
        mode: ``"standalone"``（默认，进程内独立会话）| ``"server"``（自拉起常驻 server 并连接）
            | ``"connect"``（连接外部已运行的 server，不管理其生命周期）。
        cores: 求解线程数（夹到 config 上限）。
        host/port: server / connect 模式的连接地址。
        manage_server: server 模式下，退出时是否关闭自拉起的 server 进程。
        version: standalone 模式指定 COMSOL 版本。

    Yields:
        ``mph.Client`` 实例。
    """
    mode = (mode or "standalone").lower()
    server_handle: ServerHandle | None = None
    client: Any = None
    try:
        if mode == "standalone":
            client = start_standalone(cores=cores, version=version)
        elif mode == "server":
            server_handle = launch_server(port=port, cores=cores)
            client = connect(host=server_handle.host, port=server_handle.port)
        elif mode == "connect":
            client = connect(host=host, port=port)
        else:
            raise SessionError(f"未知会话模式：{mode!r}（standalone/server/connect）")
        yield client
    finally:
        if client is not None:
            _disconnect(client)
        if server_handle is not None and manage_server:
            stop_server(server_handle)


if __name__ == "__main__":
    # 廉价自检：仅打印可用性，不真正启动会话
    for k, v in check_available().items():
        print(f"{k:18}: {v}")
