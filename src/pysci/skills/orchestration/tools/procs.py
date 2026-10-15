"""进程面原语：判活 / 身份核对 / 终止进程树（backlog 20261012-184000-leader）。

编排体系的孤儿回收原本只看**三面**（锁、backlog 条目、worktree 名），漏了第四面
**进程**：worker 被杀时它派生的 headless 会话进程不随之退出（独立进程树），仍在原
worktree 里写文件；下一次 drain 回收条目、删同名 worktree、重派同单 → 新旧两个会话
并发双写同一个隔离面（2026-10-12 实测，见 state/suggestions/approved/ 本单证据）。

本模块只回答三个问题，且都不查业务状态（drain 保持只编排）：

* :func:`pid_alive` —— 这个 pid 还在吗；
* :func:`pid_is_session` —— 它还**是**当初那个会话吗（pid 会被复用，核对 argv 里的
  sid 才敢动手：判不准宁可不杀，调用方退化为「异名隔离重建」）；
* :func:`terminate_tree` —— 让它和它的后代一起退出。

Windows 侧查询一律走 argv 列表（``subprocess`` 不经 shell），免疫 Git Bash 的 MSYS
路径转换（basic.md §3 记的 ``tasklist /FI`` 假阴性坑）；POSIX 读 ``/proc`` 不依赖外部
命令。任何查询失败都返回「不知道」，由调用方按保守方向处置。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

#: 单条查询/终止命令的墙钟上限（秒）——回收发生在 drain 启动路径上，挂起会阻塞消化。
QUERY_TIMEOUT_S: int = 20


def _hidden_creationflags(base: int = 0) -> int:
    """win32 上叠加 ``CREATE_NO_WINDOW``；POSIX 恒 0（无此概念）。"""
    if sys.platform != "win32":
        return base
    return int(base) | int(subprocess.CREATE_NO_WINDOW)


def run_hidden(
    args: list[str] | str, *, creationflags: int = 0, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    """**唯一**的子进程 spawn 口径：跑一条命令并**隐藏窗口**（win32 恒带 ``CREATE_NO_WINDOW``）。

    为什么必须有单一落点（用户 2026-10-15 加急实案）：``pythonw.exe``/GUI 子系统进程或
    分离进程（``_spawn_detached``）**没有控制台**，它们派生 console 子系统程序（``git``/``sh``/
    ``pre-commit``…）时若不带该旗标，Windows 会**新开一个可见控制台**；命令秒退 → 窗口闪现
    夺焦点。看门狗一次 tick 连发约 28 个 git ⇒ 用户每分钟被 ~28 个空终端打断（30 分钟+）。

    调用方一律走本函数（「同一能力不留两份口径」）：``runner``/``drain`` 原先各自手写旗标，
    现并入此处；POSIX 侧**不加**任何旗标、行为完全不变。
    """
    if sys.platform == "win32" and not kwargs.get("shell"):
        kwargs["creationflags"] = _hidden_creationflags(creationflags)
    return subprocess.run(args, **kwargs)  # type: ignore[arg-type]


def popen_hidden(
    argv: list[str] | str, *, creationflags: int = 0, **kwargs: object
) -> subprocess.Popen:
    """``Popen`` 版的同一口径（分离/长驻子进程用；win32 恒带 ``CREATE_NO_WINDOW``）。"""
    if sys.platform == "win32" and not kwargs.get("shell"):
        kwargs["creationflags"] = _hidden_creationflags(creationflags)
    return subprocess.Popen(argv, **kwargs)  # type: ignore[arg-type]


def _cmd(args: list[str], timeout_s: int = QUERY_TIMEOUT_S) -> str:
    """跑一条查询命令，返回合并输出；任何异常（缺命令/超时）→ 空串（= 不知道）。

    走 :func:`run_hidden`：本模块的探活（``tasklist``）与终止（``taskkill``）常在**无控制台**
    的巡更/回收路径上被调，裸 spawn 会闪窗（2026-10-15 加急「空终端风暴」）。
    """
    try:
        proc = run_hidden(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return ""
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def pid_alive(pid: int) -> bool:
    """pid 是否仍存活；**判不出来时保守返回 True**（宁可不动手，不可误判成死）。

    Windows 用 ``tasklist`` 按 pid 过滤（``/NH`` 去表头，argv 直传免 MSYS 转换）；
    POSIX 用 0 信号探活。
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        out = _cmd(["tasklist", "/FI", f"PID eq {pid}", "/NH"], timeout_s=15)
        return str(pid) in out if out else True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def process_cmdline(pid: int) -> str:
    """进程完整命令行；查不到/无权限 → 空串（= 不知道）。"""
    if pid <= 0:
        return ""
    if sys.platform == "win32":
        ps = (
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
            "if ($p) { $p.CommandLine }"
        )
        return _cmd(["powershell", "-NoProfile", "-Command", ps])
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace").replace("\x00", " ").strip()


def pid_is_session(pid: int, sid: str) -> bool:
    """pid 是否仍是那个会话进程（命令行里带着该 sid 才算数）。

    为什么不能只凭 pid：孤儿回收读到的 pid 属于**上一个已死的 worker** 的子进程，而
    系统会在一段时间后把同一个数字发给毫不相干的进程；对着它 ``taskkill /F`` 就是误杀
    别人的作业。headless 会话的 argv 必含 ``--session-id <sid>`` 或 ``--resume <sid>``
    （见 runner.build_command），故按 sid 核对身份。

    Returns:
        True 仅当命令行读得到且含该 sid；读不到或不含 → False（宁可不杀）。
    """
    if not sid or pid <= 0:
        return False
    return sid in process_cmdline(pid)


def terminate_tree(pid: int) -> tuple[bool, str]:
    """终止 pid 及其后代。

    Windows 走 ``taskkill /T /F``：``/T`` 是硬要求——headless CLI 自己还会派生子进程，
    只杀父进程等于留下更难发现的孙进程继续写盘。POSIX 先 SIGTERM（给对手机会收尾）。

    Returns:
        ``(已执行, 回显)``。**不代表进程已死**——存活与否由调用方复查
        :func:`pid_alive`（判活与终止分离，测试才好各钉一例）。
    """
    if pid <= 0:
        return False, "pid 非法"
    if sys.platform == "win32":
        out = _cmd(["taskkill", "/PID", str(pid), "/T", "/F"])
        return True, out
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "SIGTERM"
