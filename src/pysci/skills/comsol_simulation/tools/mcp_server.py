"""社区 comsol MCP 服务端的生命周期托管（path C：HTTP transport + 惰性 COMSOL）。

为什么需要这个模块
------------------
上游 ``src/server.py`` 的 ``main()`` 只在 ``COMSOL_MCP_TRANSPORT=stdio`` 时 pre-start COMSOL；
其余取值只设 host/port 便 ``mcp.run(transport)``，COMSOL 推迟到首次工具调用才启动。在单机单
license 场景下这带来两个决定性好处：

1. **空闲不占 license**——服务端常驻但没有 ``jvm.dll``，CLI 与 GUI 可自由使用唯一 license；
2. **握手秒级**——不再有 ~30s 的 COMSOL 冷启动拖慢 MCP 握手，而那正是 Qoder 在启动期重写
   ``mcp.json`` 触发重载、把旧实例变成孤儿、最终两个 JVM 抢一个 license 的根因。

代价：Qoder 不再替你拉起服务端（URL 型注册只负责连接），必须有人保证它在跑。本模块把这件事
收口成一个**幂等**命令::

    uv run pysci-simulation mcp ensure

不在跑就脱离终端派生一个，在跑就直接复用并回报状态。SKILL.md 把它固化为"调用任何 comsol MCP
工具之前的第 0 步"，于是 LLM 无需理解 transport / license / 派生细节即可安全使用。

关键实现约束（每条都对应一次实测踩坑）
--------------------------------------
- **必须经 WMI 派生**。Agent 驱动的终端里，普通子进程会随终端拆除被静默回收——无 traceback、
  无 Windows Error Reporting 记录（2026-10-01 实测：一次 27 分钟的 RAG 构建就这样无声消失）。
  同一条命令里的 A/B：``Start-Process`` 与 WMI ``Win32_Process.Create`` 各派生一个 ~15 分钟的
  ``ping`` 标记，下一条命令时前者已消失、后者仍在——WMI 进程的父是 ``WmiPrvSE.exe``，根本不进
  终端的 job。``CREATE_BREAKAWAY_FROM_JOB`` 在这里无效（Qoder 的 job 未设
  ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``），``schtasks`` 需要管理员权限（实测拒绝访问）。
  判定“是否脱离”只能看祖先链里有没有 ``WmiPrvSE.exe``：``IsProcessInJob`` 会误导——WMI 进程与
  Qoder 自己的 stdio 实例都报 in_job=True 却都能存活。
- **就绪判据是"收到任何 HTTP 响应"**，而非 TCP 连通。``transport=sse`` 时 ``GET /sse`` 返回
  200 + ``text/event-stream``；``transport=streamable-http`` 时 ``GET /mcp`` 返回 **400**（缺
  session id）——两者都证明 uvicorn 已就绪。返回 **404** 说明 URL 路径与实际 transport 不匹配
  （``/sse`` ↔ sse、``/mcp`` ↔ streamable-http，必须成对）。
- **license 判定看 ``jvm.dll``，绝不看 ``_jpype.pyd``**。后者在 ``import mph`` 时就加载，存在于
  每个 comsol-mcp 进程中，与 COMSOL 是否启动无关；只有 ``startJVM()`` 才会拉进 COMSOL JRE 的
  ``jvm.dll``，也只有它消耗 license。实测：空闲的 HTTP 服务端只有 ``_jpype.pyd``（74MB），已
  pre-start 的 stdio 实例两者都有（~330MB）。用 ``jvm|jpype`` 模糊匹配会把空闲服务端误报成
  "正在占用 license"。
- 只用标准库（``urllib`` + ``ctypes``）：本模块是编排层，不应为此引入 psutil/httpx。
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import settings

#: 状态文件名（落在 ``settings.runs_dir``，与 ``comsol_server.json`` 并列）。
_STATE_NAME = "comsol_mcp_server.json"

#: 服务端日志名（每次启动截断；崩溃现场会保留到下次启动前）。
_LOG_NAME = "comsol_mcp_server.log"

#: FastMCP 各 transport 服务的端点路径（``mcp.server.fastmcp.server.Settings`` 的默认值）。
_ENDPOINT_PATHS = {"sse": "/sse", "streamable-http": "/mcp"}

#: 允许托管的 transport。``stdio`` 由 Qoder 自己派生，不归本模块管。
MANAGED_TRANSPORTS = ("sse", "streamable-http")

#: 可能持有 COMSOL license 的进程映像名（小写）。
_LICENSE_CANDIDATE_EXES = frozenset({
    "python.exe",
    "pythonw.exe",
    "comsol-mcp.exe",
    "comsol.exe",
    "comsolmphserver.exe",
    "comsolbatch.exe",
    "comsolparallel.exe",
    "java.exe",
    "javaw.exe",
})

# Windows 进程/模块枚举常量
_TH32CS_SNAPPROCESS = 0x00000002

#: 祖先链里出现这些 exe 即视为“已脱离终端”——它们的生命周期与本会话无关。
_DETACHED_ANCESTORS = frozenset({"wmiprvse.exe", "services.exe", "wininit.exe"})
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_LIST_MODULES_ALL = 0x03
_STILL_ACTIVE = 259

# CreateProcess flags
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000

# GetExtendedTcpTable 常量
_AF_INET = 2
_AF_INET6 = 23
_TCP_TABLE_OWNER_PID_LISTENER = 3
_MIB_TCP_STATE_LISTEN = 2


class _MIB_TCPROW_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("dwState", ctypes.c_uint32),
        ("dwLocalAddr", ctypes.c_uint32),
        ("dwLocalPort", ctypes.c_uint32),
        ("dwRemoteAddr", ctypes.c_uint32),
        ("dwRemotePort", ctypes.c_uint32),
        ("dwOwningPid", ctypes.c_uint32),
    ]


class _MIB_TCPTABLE_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("dwNumEntries", ctypes.c_uint32),
        ("table", _MIB_TCPROW_OWNER_PID * 1),
    ]


class McpServerError(RuntimeError):
    """MCP 服务端托管失败（配置缺失、派生失败、就绪超时）。"""


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
def endpoint_path(transport: str) -> str:
    """transport → FastMCP 端点路径。未知 transport 直接报错，避免拼出 404 的 URL。"""
    try:
        return _ENDPOINT_PATHS[transport]
    except KeyError:
        raise McpServerError(
            f"未知 transport {transport!r}；本模块只托管 {'/'.join(MANAGED_TRANSPORTS)}"
            "（stdio 由 Qoder 自行派生，不在此列）"
        ) from None


def mcp_url(transport: str | None = None, host: str | None = None, port: int | None = None) -> str:
    """拼出应写进 Qoder ``mcp.json`` 的 ``url`` 字段。"""
    t = transport or settings.mcp_transport
    h = host or settings.mcp_host
    p = port or settings.mcp_port
    return f"http://{h}:{p}{endpoint_path(t)}"


def mcp_registration_json(transport: str | None = None, host: str | None = None, port: int | None = None) -> str:
    """生成可直接粘贴进 ``mcp.json`` 的 comsol 条目（URL 型注册）。"""
    url = mcp_url(transport, host, port)
    block = {"mcpServers": {"comsol": {"type": "sse", "url": url}}}
    return json.dumps(block, indent=2, ensure_ascii=False)


def _state_path() -> Path:
    return settings.runs_dir / _STATE_NAME


def _log_path() -> Path:
    return settings.runs_dir / _LOG_NAME


def _require_exe() -> Path:
    """返回社区 comsol-mcp 可执行文件；缺失时给出可操作的修复指引。"""
    exe = settings.mcp_exe
    if exe is None or not Path(exe).exists():
        raise McpServerError(
            "未找到社区 comsol-mcp 可执行文件。请先运行 "
            "scripts\\comsol_mcp\\setup_comsol_mcp.ps1，或在 .env 设置 COMSOL_MCP_REPO / COMSOL_MCP_EXE。"
            f"（当前解析结果：repo={settings.mcp_repo_dir!r} exe={exe!r}）"
        )
    return Path(exe)


def _server_env() -> dict[str, str]:
    """派生服务端进程的环境变量。

    URL 型注册的 ``mcp.json`` 没有 ``env`` 字段，所以 ``COMSOL_MCP_*`` 只能由这里注入——包括
    ``COMSOL_MCP_VERSION``（缺了它上游可能挑错 COMSOL 版本）。
    """
    env = dict(os.environ)
    env["COMSOL_MCP_TRANSPORT"] = settings.mcp_transport
    env["COMSOL_MCP_HOST"] = settings.mcp_host
    env["COMSOL_MCP_PORT"] = str(settings.mcp_port)
    version = settings.mcp_version or (settings.install.version or "")
    if version:
        env["COMSOL_MCP_VERSION"] = version
    return env


# ---------------------------------------------------------------------------
# 进程 / license 探测（ctypes，无第三方依赖）
# ---------------------------------------------------------------------------
class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


# --- Win32 绑定：显式声明 argtypes / restype ------------------------------
# 不声明会踩两个坑：(1) 64 位 HANDLE / HMODULE 被按 C int 转换而报
# OverflowError: int too long to convert；(2) restype 默认 c_int，会把句柄截断。
# 这里用本模块私有的 WinDLL 实例而不是 ctypes.windll 缓存的那份：后者是进程全局共享的，
# 在上面改 argtypes 会影响其它库对同一函数的调用约定。
_HANDLE = ctypes.c_void_p
_DWORD = ctypes.c_uint32
_BOOL = ctypes.c_int
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32")
    _psapi = ctypes.WinDLL("psapi")
    _iphlpapi = ctypes.WinDLL("iphlpapi")

    _k32.OpenProcess.argtypes = [_DWORD, _BOOL, _DWORD]
    _k32.OpenProcess.restype = _HANDLE
    _k32.CloseHandle.argtypes = [_HANDLE]
    _k32.CloseHandle.restype = _BOOL
    _k32.GetExitCodeProcess.argtypes = [_HANDLE, ctypes.POINTER(_DWORD)]
    _k32.GetExitCodeProcess.restype = _BOOL
    _k32.CreateToolhelp32Snapshot.argtypes = [_DWORD, _DWORD]
    _k32.CreateToolhelp32Snapshot.restype = _HANDLE
    _k32.Process32FirstW.argtypes = [_HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.Process32FirstW.restype = _BOOL
    _k32.Process32NextW.argtypes = [_HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.Process32NextW.restype = _BOOL
    _k32.IsProcessInJob.argtypes = [_HANDLE, _HANDLE, ctypes.POINTER(_BOOL)]
    _k32.IsProcessInJob.restype = _BOOL
    _psapi.EnumProcessModulesEx.argtypes = [
        _HANDLE,
        ctypes.POINTER(ctypes.c_void_p),
        _DWORD,
        ctypes.POINTER(_DWORD),
        _DWORD,
    ]
    _psapi.EnumProcessModulesEx.restype = _BOOL
    _psapi.GetModuleFileNameExW.argtypes = [_HANDLE, ctypes.c_void_p, ctypes.c_wchar_p, _DWORD]
    _psapi.GetModuleFileNameExW.restype = _DWORD
    _iphlpapi.GetExtendedTcpTable.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DWORD),
        _BOOL,
        _DWORD,
        _DWORD,
        _DWORD,
    ]
    _iphlpapi.GetExtendedTcpTable.restype = _DWORD
else:  # pragma: no cover - 非 Windows 下所有探测函数直接返回空值
    _k32 = None
    _psapi = None
    _iphlpapi = None


def _snapshot_full() -> list[tuple[int, int, str]]:
    """一次 Toolhelp 快照拿到 (pid, 父 pid, exe 名)，避免对每个进程都 OpenProcess。"""
    if _k32 is None:
        return []
    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID_HANDLE_VALUE:
        return []
    out: list[tuple[int, int, str]] = []
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        if not _k32.Process32FirstW(snap, ctypes.byref(entry)):
            return []
        while True:
            out.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile or ""))
            if not _k32.Process32NextW(snap, ctypes.byref(entry)):
                break
    finally:
        _k32.CloseHandle(snap)
    return out


def _snapshot_processes() -> list[tuple[int, str]]:
    return [(pid, exe) for pid, _ppid, exe in _snapshot_full()]


def _ancestor_exes(pid: int, limit: int = 16) -> list[str]:
    """从 pid 往上走到根，返回沿途的 exe 名（含起点自身）。

    走整条链而不是只看父进程：WMI 派生出来的层级是
    ``WmiPrvSE.exe → cmd.exe → comsol-mcp.exe → python.exe``，而我们要判定的是链尾那个真正
    承载 uvicorn（以及将来 jvm.dll）的解释器，它的直接父进程是启动器壳，看不出脱离与否。
    """
    table = {p: (pp, exe) for p, pp, exe in _snapshot_full()}
    chain: list[str] = []
    seen: set[int] = set()
    cur = int(pid)
    for _ in range(limit):
        if cur in seen:  # pid 复用可能造出环，见到重复立刻停
            break
        seen.add(cur)
        entry = table.get(cur)
        if entry is None:
            break
        chain.append(entry[1])
        parent = entry[0]
        if not parent or parent == cur:
            break
        cur = parent
    return chain


def spawn_is_detached(pid: int) -> bool | None:
    """该进程是否由“不会随 agent 终端一起消失”的系统服务派生；无法判定时返回 None。

    这是“派生是否真的脱离了终端”的**直接证据**。刻意不用 ``IsProcessInJob``：它回答的是
    “是否在**任意** job 里”，而 WMI 派生的进程和 Qoder 自己长期存活的 stdio 实例都返回
    True——job 成员身份并不等于“终端拆除时会被回收”。
    """
    chain = _ancestor_exes(pid)
    if not chain:
        return None
    return any(exe.lower() in _DETACHED_ANCESTORS for exe in chain)


def _pid_modules(pid: int) -> list[str]:
    """列出某进程已加载模块的完整路径（打不开或无权限时返回空表）。"""
    if _k32 is None or _psapi is None:
        return []
    handle = _k32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, int(pid))
    if not handle:
        return []
    try:
        needed = _DWORD(0)
        arr = (ctypes.c_void_p * 2048)()
        if not _psapi.EnumProcessModulesEx(
            handle, arr, _DWORD(ctypes.sizeof(arr)), ctypes.byref(needed), _DWORD(_LIST_MODULES_ALL)
        ):
            return []
        count = min(needed.value // ctypes.sizeof(ctypes.c_void_p), len(arr))
        buf = ctypes.create_unicode_buffer(1024)
        paths: list[str] = []
        for i in range(count):
            if _psapi.GetModuleFileNameExW(handle, arr[i], buf, _DWORD(1024)):
                paths.append(buf.value)
        return paths
    finally:
        _k32.CloseHandle(handle)


def pid_alive(pid: int | None) -> bool:
    """进程是否仍在（Windows 用 OpenProcess + 退出码，POSIX 用 signal 0）。"""
    if not pid or int(pid) <= 0:
        return False
    if _k32 is None:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return False
        return True
    handle = _k32.OpenProcess(_PROCESS_QUERY_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = _DWORD(0)
        if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _k32.CloseHandle(handle)


def pid_holds_license(pid: int) -> bool:
    """该进程是否加载了 ``jvm.dll``——即是否真的占用了唯一 license。

    只认 ``jvm.dll``，不认 ``_jpype.pyd``（见模块 docstring 的第三条约束）。
    """
    return any(Path(p).name.lower() == "jvm.dll" for p in _pid_modules(pid))


def pid_in_job(pid: int) -> bool | None:
    """该进程是否仍属于某个 job object；无法判定时返回 None。

    这是“派生是否真的脱离了终端”的**直接证据**。Agent 驱动的终端会把子进程纳入自己的
    job，终端拆除时整棵树被静默回收（无 traceback、无 WER 记录）；只有 ``IsProcessInJob``
    返回 False 才能确定服务端会在会话结束后继续活着。无法打开进程时不猜，返回 None。
    """
    if _k32 is None:
        return None
    handle = _k32.OpenProcess(_PROCESS_QUERY_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        result = _BOOL(0)
        # hJob 传 NULL 即“是否属于任意 job”。
        if not _k32.IsProcessInJob(handle, None, ctypes.byref(result)):
            return None
        return bool(result.value)
    finally:
        _k32.CloseHandle(handle)


def license_holders() -> list[dict[str, Any]]:
    """扫描本机当前占用 COMSOL license 的进程。

    覆盖三种形态：MPh standalone（``jvm.dll`` 在某个 python.exe 内）、常驻
    ``comsolmphserver.exe``、交互式 GUI ``comsol.exe``。返回空表即"license 空闲"。
    """
    out: list[dict[str, Any]] = []
    for pid, exe in _snapshot_processes():
        if exe.lower() not in _LICENSE_CANDIDATE_EXES:
            continue
        jvm = [p for p in _pid_modules(pid) if Path(p).name.lower() == "jvm.dll"]
        if jvm:
            out.append({"pid": pid, "exe": exe, "jvm": jvm[0]})
    return out


def _pid_listening_on(port: int) -> int | None:
    """找出监听某端口的 pid；找不到返回 None。

    走 ``GetExtendedTcpTable``（iphlpapi）而不是 ``netstat -ano``：后者在 zh-CN Windows 上
    输出 GBK 字节（``text=True`` 会抛 UnicodeDecodeError）且状态列可能本地化，而本函数的
    结果直接决定 ``holds_license`` 的真假，不能靠解析本地化文本。

    为什么必须找端口归属者：派生的是 ``comsol-mcp.exe`` 启动器壳，它 re-exec 出 venv 的
    ``python.exe``，uvicorn 又跑在**那个**解释器进程内。壳永远不会加载 ``jvm.dll``，所以拿
    壳的 pid 判 license 会恒为 False（假阴性）；监听端口的进程才是真正会持有 JVM 的那个。
    """
    if sys.platform != "win32" or _iphlpapi is None:
        return None
    for family in (_AF_INET, _AF_INET6):
        pid = _listener_pid_in_family(port, family)
        if pid:
            return pid
    return None


def _listener_pid_in_family(port: int, family: int) -> int | None:
    size = _DWORD(0)
    # 第一次调用只为拿到所需缓冲区大小（必然返回 ERROR_INSUFFICIENT_BUFFER）。
    _iphlpapi.GetExtendedTcpTable(
        None, ctypes.byref(size), False, _DWORD(family), _DWORD(_TCP_TABLE_OWNER_PID_LISTENER), _DWORD(0)
    )
    if size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    ret = _iphlpapi.GetExtendedTcpTable(
        buf,
        ctypes.byref(size),
        False,
        _DWORD(family),
        _DWORD(_TCP_TABLE_OWNER_PID_LISTENER),
        _DWORD(0),
    )
    if ret != 0:
        return None
    table = ctypes.cast(buf, ctypes.POINTER(_MIB_TCPTABLE_OWNER_PID)).contents
    n = int(table.dwNumEntries)
    if n <= 0:
        return None
    # 结构体里的数组只能声明为 `* 1`（长度运行时才知道），所以按 dwNumEntries 重建一份：
    # 直接索引 table.table[i] 在 i>0 时会抛 IndexError。
    rows = (_MIB_TCPROW_OWNER_PID * n).from_buffer_copy(buf, _MIB_TCPTABLE_OWNER_PID.table.offset)
    for row in rows:
        if row.dwState == _MIB_TCP_STATE_LISTEN and socket.ntohs(row.dwLocalPort) == port:
            return int(row.dwOwningPid)
    return None


# ---------------------------------------------------------------------------
# HTTP 就绪探测
# ---------------------------------------------------------------------------
def _read_error_body(err: urllib.error.HTTPError, limit: int = 160) -> str:
    try:
        raw = err.read(limit).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""
    return " ".join(raw.split())


def probe_http(url: str, timeout: float = 3.0) -> dict[str, Any]:
    """GET 端点一次，把"能否收到 HTTP 响应"与状态码分开报告。

    不读响应体：SSE 端点的 body 是永不结束的事件流，读了就会阻塞到超时。
    """
    req = urllib.request.Request(url, method="GET", headers={"Accept": "text/event-stream"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return {"reached": True, "status": int(resp.status), "note": ""}
    except urllib.error.HTTPError as e:
        return {"reached": True, "status": int(e.code), "note": _read_error_body(e)}
    except Exception as e:  # noqa: BLE001
        return {"reached": False, "status": None, "note": f"{type(e).__name__}: {e}"}


def interpret_probe(status: int | None, transport: str) -> tuple[bool, str]:
    """把 HTTP 状态码翻译成 (是否就绪, 人类可读结论)。"""
    if status is None:
        return False, "端点无响应（服务端未启动或端口不通）"
    if status == 404:
        return False, (
            f"404：URL 路径与 transport 不匹配。transport={transport} 应配 "
            f"{endpoint_path(transport)}；请核对 mcp.json 的 url 与服务端环境变量。"
        )
    if status == 403:
        return False, (
            "403 Invalid Origin header：被 FastMCP 的 DNS-rebinding 防护拒绝。"
            "本探测不发 Origin，若仍 403 说明防护规则被改；若只有 Qoder 连不上，"
            "则是 Qoder 发了不被允许的 Origin（如 vscode-file://vscode-app），"
            "见 UPSTREAM.lock.json 的 origin_guard 条目。"
        )
    if transport == "streamable-http" and status == 400:
        return True, "400（缺 session id）属正常：证明 /mcp 上的 streamable-http 已就绪"
    if 200 <= status < 300:
        return True, f"{status} 就绪"
    return False, f"意外状态码 {status}"


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 状态文件
# ---------------------------------------------------------------------------
def read_state() -> dict[str, Any]:
    """读取状态文件；不存在或损坏时返回空 dict（不抛）。"""
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(info: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_state() -> None:
    _state_path().unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 派生
# ---------------------------------------------------------------------------
#: 启动器脚本名（落在 ``settings.runs_dir``，可人工查看 / 双击排查）。
_LAUNCHER_NAME = "start_comsol_mcp.cmd"

#: 需要注入服务端的 ``COMSOL_MCP_*`` 变量。URL 型 ``mcp.json`` 没有 ``env`` 字段，只能写在这里。
_LAUNCHER_ENV_KEYS = ("COMSOL_MCP_TRANSPORT", "COMSOL_MCP_HOST", "COMSOL_MCP_PORT", "COMSOL_MCP_VERSION")


def _launcher_path() -> Path:
    return settings.runs_dir / _LAUNCHER_NAME


def _write_launcher(exe: Path) -> Path:
    """生成一个自包含的 ``.cmd`` 启动器：设好环境变量，输出重定向到日志，同时截断日志并盖时间戳。

    为什么要绕这一层：WMI 的 ``Win32_Process.Create`` 不接受环境变量，而 transport/host/port/
    version 必须有人注入。写成一个可被人直接查看、双击复现的 ``.cmd``，比在 PowerShell
    命令行里拼多层引号可靠得多（``cmd.exe /c`` 与 PowerShell 单引号字面量各有一套转义规则）。
    """
    launcher = _launcher_path()
    launcher.parent.mkdir(parents=True, exist_ok=True)
    log = _log_path()
    env = _server_env()
    lines = [
        "@echo off",
        "rem Auto-generated by pysci.skills.comsol_simulation.tools.mcp_server -- do not edit.",
        "rem Runs the community comsol-mcp HTTP server detached from any terminal.",
        "rem It is invoked through WMI so that it survives the teardown of the spawning shell.",
    ]
    lines += [f"set {key}={env[key]}" for key in _LAUNCHER_ENV_KEYS if env.get(key)]
    lines.append(f'"{exe}" >> "{log}" 2>&1')
    # .cmd 由 cmd.exe 按 ANSI 解析；路径含非 ASCII 字符时 errors="replace" 至少不会在
    # 派生这个最坏的时刻抛 UnicodeEncodeError。
    launcher.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", errors="replace")
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(
            f"# comsol-mcp {settings.mcp_transport} on {settings.mcp_host}:{settings.mcp_port}\n"
            f"# started {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"# exe {exe}\n"
        )
    return launcher


def _ps_literal(text: str) -> str:
    """把字符串安全地放进 PowerShell 单引号字面量（内部的 ``'`` 需要翻倍）。"""
    return "'" + text.replace("'", "''") + "'"


def _spawn_via_wmi(cmdline: str, cwd: Path) -> int | None:
    """用 WMI ``Win32_Process.Create`` 派生，返回新进程 pid；不可用时返回 None。

    这是本模块**唯一被实测证明**能让服务端活过 agent 终端拆除的机制（证据见模块 docstring）。
    返回的 pid 属于 ``cmd.exe``，真正的服务端的它的子孙；就绪与否一律以 HTTP 探测为准，
    不依赖这个 pid。调用 PowerShell 而不是 ``wmic.exe``：后者已废弃且在较新的 Windows 11
    上被移除，PowerShell 则一定存在。
    """
    script = (
        "$s=([wmiclass]'\\\\.\\root\\cimv2:Win32_ProcessStartup').CreateInstance();"
        "$s.ShowWindow=$false;"
        f"$r=([wmiclass]'\\\\.\\root\\cimv2:Win32_Process').Create("
        f"{_ps_literal(cmdline)},{_ps_literal(str(cwd))},$s);"
        "Write-Output $r.ReturnValue;Write-Output $r.ProcessId"
    )
    try:
        done = subprocess.run(  # noqa: S603
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
    # 期望两行：ReturnValue（必须为 0）与 ProcessId。WMI 不可用时这里就返回 None 降级。
    if len(out) < 2 or out[0] != "0":
        return None
    try:
        return int(out[1])
    except ValueError:
        return None


def _spawn_detached(exe: Path) -> tuple[int, str, Path]:
    """派生服务端，返回 ``(pid, 派生方式, 日志路径)``。

    优先 WMI；WMI 不可用时降级到 CreateProcess 的各 flag 组合。降级派生**会**随终端拆除被
    回收，所以把派生方式回报出去，由 :func:`mcp_server_status` 如实告知 ``detached`` 真假，
    而不是默默假装成功。
    """
    log = _log_path()
    launcher = _write_launcher(exe)
    pid = _spawn_via_wmi(f'cmd.exe /c "{launcher}"', exe.parent)
    if pid:
        return pid, "wmi", log

    env = _server_env()
    base_flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    variants: list[tuple[int, str]] = (
        [(base_flags | _CREATE_BREAKAWAY_FROM_JOB, "popen-breakaway"), (base_flags, "popen")]
        if sys.platform == "win32"
        else [(0, "posix")]
    )
    last_err: OSError | None = None
    for flags, method in variants:
        try:
            # 追加而不截断：日志头已由 _write_launcher 写好，这里保留它。
            with log.open("a", encoding="utf-8", errors="replace") as fh:
                proc = subprocess.Popen(  # noqa: S603 - 路径来自受信的 MCP 安装发现
                    [str(exe)],
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(exe.parent),
                    creationflags=flags,
                    close_fds=True,
                )
            return proc.pid, method, log
        except OSError as e:
            last_err = e
    raise McpServerError(f"派生 comsol-mcp 失败（WMI 与 CreateProcess 均不可用）：{last_err!r}") from last_err


def _wait_ready(url: str, timeout: float) -> dict[str, Any]:
    """轮询直到端点给出可解释为就绪的 HTTP 响应。"""
    deadline = time.time() + timeout
    last: dict[str, Any] = {"reached": False, "status": None, "note": "超时前未探测"}
    while time.time() < deadline:
        last = probe_http(url, timeout=2.0)
        ready, why = interpret_probe(last["status"], settings.mcp_transport)
        if ready:
            return {"ready": True, "why": why, **last}
        if last["reached"]:
            # 端口已有 HTTP 服务但语义不对（404/403）：再等也没用，立刻返回。
            return {"ready": False, "why": why, **last}
        time.sleep(0.5)
    return {"ready": False, "why": interpret_probe(last["status"], settings.mcp_transport)[1], **last}


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------
def mcp_server_status() -> dict[str, Any]:
    """汇总 MCP 服务端状态：状态文件、端口、探测结果、license 占用。

    返回字典的关键字段：

    - ``running``：端点是否给出就绪响应（这是 Qoder 能否连上的判据）
    - ``pid`` / ``pid_alive``：**端口归属者**的 pid（承载 uvicorn 与将来 JVM 的真实进程）
    - ``detached``：该进程的祖先链里是否有系统服务——False 则终端拆除时它会被静默回收
    - ``holds_license``：该 pid 是否已加载 ``jvm.dll``（即 COMSOL 是否已被惰性启动）
    - ``license_holders``：全机 license 占用者（含 GUI / comsolmphserver / 其它 python）
    - ``url`` / ``registration_json``：应写进 Qoder ``mcp.json`` 的内容
    """
    state = read_state()
    transport = settings.mcp_transport
    url = mcp_url()
    recorded_pid = state.get("pid")
    port_open = _port_open(settings.mcp_host, settings.mcp_port)
    probe = probe_http(url) if port_open else {"reached": False, "status": None, "note": "端口未监听"}
    ready, why = interpret_probe(probe.get("status"), transport)

    # 端口归属者优先：它才是承载 uvicorn（以及将来 jvm.dll）的真实进程；状态文件里的
    # recorded_pid 是 comsol-mcp.exe 启动器壳，端口不通时才有参考价值。
    owner = _pid_listening_on(settings.mcp_port) if port_open else None
    if owner is None and pid_alive(recorded_pid):
        owner = recorded_pid
    # “本模块派生”的判据是启动器壳仍在（壳与真实解释器是两个 pid，二者不同属正常）。
    ours = bool(recorded_pid) and pid_alive(recorded_pid)

    info: dict[str, Any] = {
        "state_file": str(_state_path()),
        "log_file": str(_log_path()),
        "exe": str(settings.mcp_exe or ""),
        "launcher": str(_launcher_path()),
        "transport": transport,
        "host": settings.mcp_host,
        "port": settings.mcp_port,
        "url": url,
        "endpoint_path": _ENDPOINT_PATHS.get(transport),
        "port_open": port_open,
        "running": bool(ready),
        "probe": probe,
        "probe_verdict": why,
        "pid": owner,
        "pid_alive": pid_alive(owner),
        "launcher_pid": recorded_pid if recorded_pid != owner else None,
        "started_at": state.get("started_at"),
        "spawn_method": state.get("spawn_method"),
        "externally_started": bool(owner) and not ours,
        #: 脱离终端的直接证据：True = 祖先链里有 WmiPrvSE.exe 之类生命周期与本会话无关的
        #: 系统服务；False = 由终端派生，终端拆除时会被静默回收；None = 无法判定。
        "detached": (None if not owner else spawn_is_detached(int(owner))),
        "ancestor_chain": _ancestor_exes(int(owner)) if owner else [],
        "holds_license": pid_holds_license(int(owner)) if owner else False,
        "license_holders": license_holders(),
        "registration_json": mcp_registration_json(),
    }
    return info


def ensure_mcp_server(*, timeout: float = 90.0, restart: bool = False) -> dict[str, Any]:
    """幂等地保证 MCP 服务端在跑，返回 :func:`mcp_server_status` 的结果。

    流程：已就绪 → 直接复用；端口被外部进程占着 → 认领并复用；否则脱离终端派生一个并等到
    就绪。``restart=True`` 时先停掉已有实例（改了 transport/port 之后需要）。

    Raises:
        McpServerError: 找不到可执行文件、派生失败或就绪超时。
    """
    endpoint_path(settings.mcp_transport)  # 提前校验 transport，避免起了个 404 的服务端

    if restart:
        stop_mcp_server()

    url = mcp_url()
    if not restart:
        probe = probe_http(url, timeout=2.0) if _port_open(settings.mcp_host, settings.mcp_port) else None
        if probe is not None:
            ready, why = interpret_probe(probe["status"], settings.mcp_transport)
            if ready:
                info = mcp_server_status()
                info["action"] = "reused"
                info["probe_verdict"] = why
                return info
            if probe["reached"]:
                raise McpServerError(f"端口 {settings.mcp_port} 上有服务但不是预期的 MCP 端点：{why}")
        # 端口不通但状态文件说在跑 → 记录已失效，清掉再派生。
        if read_state():
            _clear_state()

    exe = _require_exe()
    pid, method, log = _spawn_detached(exe)
    waited = _wait_ready(url, timeout)
    if not waited["ready"]:
        _kill_tree(pid)
        raise McpServerError(
            f"comsol-mcp 在 {timeout}s 内未就绪（{waited['why']}）。日志：{log}\n"
            f"错误信息：{waited.get('note', '')}"
        )

    _write_state({
        "pid": pid,
        "spawn_method": method,
        "host": settings.mcp_host,
        "port": settings.mcp_port,
        "transport": settings.mcp_transport,
        "url": url,
        "exe": str(exe),
        "launcher": str(_launcher_path()),
        "log_file": str(log),
        "started_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })
    info = mcp_server_status()
    info["action"] = "started"
    # 派生出来的是 cmd.exe 壳（WMI 路径）或 comsol-mcp.exe 启动器壳（降级路径）；真正承载
    # uvicorn 与将来 JVM 的是链尾那个解释器，故一律以端口归属者为准。
    if info.get("pid") and info["pid"] != pid:
        info["launcher_pid"] = pid
    return info


def stop_mcp_server() -> bool:
    """终止本模块（或端口归属者）派生的服务端并清理状态文件。

    返回是否真的杀掉了什么。注意：若该进程已惰性启动过 COMSOL，杀它同时释放 license。
    """
    state = read_state()
    pid = state.get("pid")
    if not pid_alive(pid):
        pid = _pid_listening_on(settings.mcp_port) if _port_open(settings.mcp_host, settings.mcp_port) else None
    _clear_state()
    if not pid:
        return False
    _kill_tree(int(pid))
    deadline = time.time() + 10.0
    while time.time() < deadline and _port_open(settings.mcp_host, settings.mcp_port):
        time.sleep(0.3)
    return True


def _kill_tree(pid: int) -> None:
    """跨平台 best-effort 终止进程树（Windows 用 taskkill /T，POSIX 用 SIGTERM）。"""
    try:
        if sys.platform == "win32":
            subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            import signal

            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        pass


if __name__ == "__main__":
    # 廉价自检：只读状态，不派生任何进程
    for k, v in mcp_server_status().items():
        if k == "registration_json":
            continue
        print(f"{k:20}: {v}")
