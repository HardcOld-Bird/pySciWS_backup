"""存活会话登记表与「主会话」创建时序判据（wake-adopt-20261018 s1）。

用户 2026-10-18 直接裁决（高于 twin A）：唤醒权归属必须**程序化硬保证**、零人工命令。
判据（用户给定不变量：子会话存在时主会话一定存在；主会话存在时子会话不一定存在）：

- 候选集 = **存活 ∧ cwd==项目根** 的会话；
- **主会话 = 候选集中创建更早者**；候选集仅一个时即它；
- owner 记录指向者仍存活 → 保持/恢复（watchdog 重启后按记录恢复）；
- owner 缺席/死亡 → 最早创建的存活项目根会话**自动补位**；
- 晚创建者在早者存活期间永不夺主；pod cwd 会话永不入候选。

本模块只做三件事（归属判定在 watch.hook_decision 单点）：

1. **登记表**（``state/sessions-live.json``，gitignored 运行态）：钩子自登记
   ``{session_id, cwd, pid, created_at, first_seen, last_seen}``；SessionEnd 注销；
   候选求值/看门狗巡更顺手清理死会话。
2. **创建时刻锚定**：沿进程祖先链找**会话 CLI**（exe 名含 ``qoderclicn``），取该进程的
   OS 创建时刻（psutil ``create_time()``；缺 psutil 时 Windows 走 ``GetProcessTimes``、
   POSIX 走 ``/proc``）。锚不到 → 回退 ``first_seen``（首个钩子时刻 ≈ 会话创建）。
   平局按 ``session_id`` 字典序（确定性）。
3. **存活性**：优先 **PID 探活**（:func:`procs.pid_alive`，argv 直传 tasklist 免 MSYS
   路径转换）；仅当 PID 缺失才回退 ``last_seen`` 心跳新鲜度。**心跳过期而 PID 存活 =
   存活**（长回合保护——旧 stale-30min 判死正是在这里误伤主会话）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import procs

#: 会话登记表（gitignored 运行态）；``PYSCI_SESSIONS_LIVE_PATH`` 可覆盖（钩子子进程
#: 测试 / E4 副本隔离）。
SESSIONS_PATH: Path = ORCH_STATE_ROOT / "sessions-live.json"

#: **PID 缺失时**的心跳回退阈值（秒）。有 PID 时一律以进程存活为准，心跳只作读数。
REGISTRY_STALE_S: int = 30 * 60

#: 祖先链最大深度（python → bash → harness → CLI；8 层足够且防环）。
_MAX_ANCESTOR_DEPTH: int = 8


def _path() -> Path:
    """登记表路径（环境变量覆盖优先，供测试/副本隔离）。"""
    override = os.environ.get("PYSCI_SESSIONS_LIVE_PATH", "").strip()
    return Path(override) if override else SESSIONS_PATH


def _now_iso(now: float | None = None) -> str:
    """epoch 秒 → 本地 ISO 串（带时区）。用 UTC 构造再转本地：Windows 上早于本地时区
    偏移的极小 epoch 直接 ``fromtimestamp`` 会抛 ``OSError(Errno 22)``（实测创建时序
    回退值 100/200 即命中）。"""
    moment = time.time() if now is None else float(now)
    return (
        datetime.fromtimestamp(moment, tz=UTC)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _epoch(value: Any) -> float | None:
    """ISO 串 / epoch 数字 → epoch 秒；不可解析 → None。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 路径判据（单点：watch.cwd_is_project_root 也委托到这里）
# ---------------------------------------------------------------------------
def is_project_root_cwd(raw: Any) -> bool | None:
    """``raw`` 是否项目根：True/False；空或不可解析 → ``None``（判不了，不拦）。

    兼容 MSYS 形态 ``/d/repo``（钩子脚本 ``pwd`` 的输出）与 Windows 形态；
    大小写按 ``normcase`` 归一。
    """
    import re

    text = str(raw or "").strip()
    if not text:
        return None
    match = re.match(r"^/([A-Za-z])/(.*)$", text)
    if match:
        text = f"{match.group(1).upper()}:/{match.group(2)}"
    try:
        actual = os.path.normcase(str(Path(text).resolve()))
        root = os.path.normcase(str(PROJECT_ROOT.resolve()))
    except (OSError, ValueError):
        return None
    return actual == root


# ---------------------------------------------------------------------------
# 进程快照：pid → (ppid, name, create_time)
# ---------------------------------------------------------------------------
def _psutil_module():
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - 可选加速器；缺了走 ctypes//proc
        return None
    return psutil


def _process_table() -> dict[int, dict[str, Any]]:
    """一次进程快照：``{pid: {ppid, name, create_time}}``；失败 → ``{}``（判不了）。"""
    psutil = _psutil_module()
    if psutil is not None:
        table: dict[int, dict[str, Any]] = {}
        try:
            for proc in psutil.process_iter(["pid", "ppid", "name", "create_time"]):
                try:
                    info = proc.info
                    table[int(info["pid"])] = {
                        "ppid": int(info.get("ppid") or 0),
                        "name": str(info.get("name") or ""),
                        "create_time": info.get("create_time"),
                    }
                except (psutil.Error, OSError, ValueError):
                    continue
        except Exception:  # noqa: BLE001 - 快照失败按「判不了」处理
            return {}
        return table
    try:
        if sys.platform == "win32":
            return _win_process_table()
        return _posix_process_table()
    except Exception:  # noqa: BLE001
        return {}


def _win_process_table() -> dict[int, dict[str, Any]]:
    """Windows 无 psutil 兜底：Toolhelp32 快照（ppid + exe 名；create_time 另取）。"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    table: dict[int, dict[str, Any]] = {}
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return table
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            table[int(entry.th32ProcessID)] = {
                "ppid": int(entry.th32ParentProcessID),
                "name": str(entry.szExeFile),
                "create_time": None,
            }
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def _posix_process_table() -> dict[int, dict[str, Any]]:
    """POSIX 兜底：读 ``/proc/<pid>/stat``（ppid、comm、starttime）+ ``/proc/stat`` btime。"""
    try:
        hz = float(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        hz = 100.0
    btime = None
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                btime = float(line.split()[1])
                break
    except OSError:
        pass
    table: dict[int, dict[str, Any]] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        try:
            text = (proc_dir / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        rparen = text.rfind(")")
        if rparen < 0:
            continue
        comm = text[text.find("(") + 1 : rparen]
        fields = text[rparen + 2 :].split()
        try:
            ppid = int(fields[1])
            start_ticks = int(fields[19])
        except (IndexError, ValueError):
            continue
        created = (btime + start_ticks / hz) if btime is not None else None
        table[pid] = {"ppid": ppid, "name": comm, "create_time": created}
    return table


def _win_create_time(pid: int) -> float | None:
    """Windows 无 psutil 兜底：``GetProcessTimes`` → epoch 秒。"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return ticks / 1e7 - 11644473600.0  # FILETIME → Unix epoch
    finally:
        kernel32.CloseHandle(handle)


def process_create_time(
    pid: int, *, table: dict[int, dict[str, Any]] | None = None
) -> float | None:
    """进程 OS 创建时刻（epoch 秒）；锚不到 → ``None``。"""
    if pid <= 0:
        return None
    psutil = _psutil_module()
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except (psutil.Error, OSError, ValueError):
            return None
    snapshot = _process_table() if table is None else table
    created = (snapshot.get(pid) or {}).get("create_time")
    if created is not None:
        return float(created)
    if sys.platform == "win32":
        return _win_create_time(pid)
    return None


def anchor_session_pid(
    start_pid: int | None = None,
    *,
    table: dict[int, dict[str, Any]] | None = None,
    is_cli=None,
) -> int | None:
    """沿祖先链找**会话 CLI** 进程（exe 名含 ``qoderclicn``）；找不到 → ``None``。

    钩子进程树：``orch``（python）→ ``bash``（watch-arm）→ … → 会话 CLI。取最近一个
    CLI 祖先作为「会话进程」——它的 OS 创建时刻即会话创建时刻（``create_time`` 实测见
    交付）。锚不到（钩子由脱链进程执行 / 进程表读不到）由调用方回退 ``first_seen``。
    """
    snapshot = _process_table() if table is None else table
    if not snapshot:
        return None
    match = is_cli or (lambda name: "qoderclicn" in str(name).lower())
    pid = os.getpid() if start_pid is None else int(start_pid)
    seen: set[int] = set()
    for _ in range(_MAX_ANCESTOR_DEPTH):
        if pid in seen:
            return None
        seen.add(pid)
        parent = int((snapshot.get(pid) or {}).get("ppid") or 0)
        if parent <= 0:
            return None
        info = snapshot.get(parent) or {}
        if match(str(info.get("name") or "")):
            return parent
        pid = parent
    return None


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------
def load_registry() -> dict[str, dict[str, Any]]:
    """读登记表；缺失/损坏 → ``{}``。只保留形状正确的条目。"""
    path = _path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, dict):
        return {}
    return {
        str(sid): dict(entry)
        for sid, entry in sessions.items()
        if isinstance(entry, dict)
    }


def save_registry(sessions: dict[str, dict[str, Any]]) -> None:
    """落登记表（tmp + replace，避免并发读到半写）；失败不抛（观察面不塌）。"""
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": 1, "updated_at": _now_iso(), "sessions": sessions}
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except OSError:
        pass


def register_session(
    sid: str,
    cwd: str,
    *,
    pid: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """登记/刷新一个会话；返回条目。

    ``created_at``（OS 创建时刻）与 ``first_seen`` 一经写入即**不重写**——续作会话
    （同 sid 新进程）沿用原始创建时序，保证「早者恒早」。``pid`` 每跳可随锚定结果刷新
    （会话进程重启后指向新进程）。
    """
    sid = str(sid or "").strip()
    if not sid:
        return {}
    sessions = load_registry()
    entry = dict(sessions.get(sid) or {})
    existing_pid = int(entry.get("pid") or 0)
    if pid is not None:
        anchored: int | None = int(pid)
    elif existing_pid > 0 and procs.pid_alive(existing_pid):
        # 既有会话进程仍活：不重新锚定（同会话每跳登记稳定；测试显式 pid 也不被覆盖）
        anchored = existing_pid
    else:
        # 进程重启（同 sid 新 CLI）或首次登记：沿祖先链锚定；锚不到则清空 pid，
        # 由 last_seen 心跳回退判活（fail-open，不误判死）。
        anchored = anchor_session_pid()
    stamp = _now_iso(now)
    entry.setdefault("first_seen", stamp)
    if entry.get("created_at") is None and anchored is not None:
        created = process_create_time(int(anchored))
        if created is not None:
            entry["created_at"] = float(created)
    entry.update(
        {
            "session_id": sid,
            "cwd": str(cwd or entry.get("cwd") or ""),
            "pid": int(anchored) if anchored else None,
            "last_seen": stamp,
        }
    )
    sessions[sid] = entry
    save_registry(sessions)
    return entry


def unregister_session(sid: str, *, now: float | None = None) -> None:
    """SessionEnd 注销（幂等）；顺带刷新 ``updated_at``。"""
    sid = str(sid or "").strip()
    if not sid:
        return
    sessions = load_registry()
    if sessions.pop(sid, None) is not None:
        save_registry(sessions)


def session_alive(entry: dict[str, Any], *, now: float | None = None) -> bool:
    """会话是否存活：**PID 优先**；PID 缺失才看 ``last_seen`` 心跳新鲜度。

    **心跳过期而 PID 存活 = 存活**（长回合保护）：主会话跑大任务时单回合可远超
    30 min，Stop 钩子不触发、心跳停摆，但进程明明活着——旧 stale 判死正是在这里误伤。
    """
    pid = int(entry.get("pid") or 0)
    if pid > 0:
        return procs.pid_alive(pid)
    last = _epoch(entry.get("last_seen") or entry.get("first_seen"))
    if last is None:
        return False
    moment = time.time() if now is None else float(now)
    return (moment - last) <= REGISTRY_STALE_S


def creation_epoch(entry: dict[str, Any]) -> float:
    """创建时序键：OS ``created_at`` 优先，回退 ``first_seen``；都缺 → ``inf``。"""
    created = _epoch(entry.get("created_at"))
    if created is not None:
        return created
    seen = _epoch(entry.get("first_seen"))
    return seen if seen is not None else float("inf")


def creation_key(entry: dict[str, Any]) -> tuple[float, str]:
    """排序键：创建时刻升序；平局按 session_id 字典序（确定性）。"""
    return (creation_epoch(entry), str(entry.get("session_id") or ""))


def ordered_candidates(
    *, now: float | None = None, prune: bool = True
) -> list[dict[str, Any]]:
    """存活 ∧ cwd==项目根 的会话，按创建时序升序（首位即主会话）。"""
    sessions = load_registry()
    kept: dict[str, dict[str, Any]] = {}
    alive_candidates: list[dict[str, Any]] = []
    for sid, entry in sessions.items():
        if not session_alive(entry, now=now):
            continue  # 死会话：顺手清理
        kept[sid] = entry
        if is_project_root_cwd(entry.get("cwd")) is True:
            alive_candidates.append(entry)
    if prune and len(kept) != len(sessions):
        save_registry(kept)
    return sorted(alive_candidates, key=creation_key)


def prune_dead(*, now: float | None = None) -> list[str]:
    """清理死会话登记（供看门狗巡更/状态读数调用）；返回被清掉的 sid。"""
    sessions = load_registry()
    dead = [sid for sid, e in sessions.items() if not session_alive(e, now=now)]
    if dead:
        for sid in dead:
            sessions.pop(sid, None)
        save_registry(sessions)
    return dead


def session_lines(entry: dict[str, Any], *, now: float | None = None) -> str:
    """status 用单行读数：sid / 创建时序 / pid / 存活。"""
    created = creation_epoch(entry)
    created_txt = _now_iso(created) if created != float("inf") else "未知"
    pid = entry.get("pid") or "-"
    alive = "存活" if session_alive(entry, now=now) else "已死"
    return f"{entry.get('session_id')}｜创建 {created_txt}｜pid {pid}｜{alive}"
