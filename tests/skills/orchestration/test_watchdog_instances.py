"""看门狗**逻辑实例数**读数（backlog 20261015-182745-devops）。

现场（2026-10-15）：验收口径「wmic 过滤 ``_watchdog-loop`` 实例数恒 ≤1」在本项目**永远
读不到 1**——``.venv/Scripts/python.exe`` 是**转发垫片**，一个逻辑看门狗恒产生两个同 argv
进程（父=垫片、子=uv 托管的真解释器，锁由子进程持有）。实测同刻：

    pid 16832  …\\.venv\\Scripts\\python.exe -m …orch _watchdog-loop --run-log …\\20261015-194744.log …
    pid 22412  D:\\XiGPrograms\\uv\\python\\cpython-3.13.15…\\python.exe  （同 argv）

⇒ 2 进程 / 1 个 ``--run-log``。故口径改为 **distinct ``--run-log`` 数 = 逻辑实例数**，并把
它固化进 ``orch watchdog status``（工具面），而不是留在口头。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import procs, watchdog

SHIM = r"D:\repo\.venv\Scripts\python.exe"
REAL = r"D:\XiGPrograms\uv\python\cpython-3.13.15-windows-x86_64-none\python.exe"
LOG_A = r"D:\repo\orchestration\state\watchdog-runs\20261015-194744.log"
LOG_B = r"D:\repo\orchestration\state\watchdog-runs\20261015-195000.log"


def _loop_cmd(log: str) -> str:
    return (
        f"{SHIM} -m pysci.skills.orchestration.tools.orch _watchdog-loop "
        f"--run-log {log} --interval 60 --stale 300"
    )


# ---------------------------------------------------------------------------
# 纯函数：watchdog_instances
# ---------------------------------------------------------------------------
def test_shim_pair_collapses_to_one_instance():
    """垫片 + 托管解释器（同 argv 同 run-log）→ 2 进程 = **1** 逻辑实例。"""
    rows = [(16832, _loop_cmd(LOG_A)), (22412, _loop_cmd(LOG_A))]
    got = watchdog.watchdog_instances(rows)
    assert got is not None
    assert got["loop_processes"] == 2
    assert got["logical"] == 1
    assert got["run_logs"] == [LOG_A]


def test_distinct_run_logs_are_distinct_instances():
    """两个不同 ``--run-log`` = 两个逻辑实例（真双生，仍须能被数出来）。"""
    rows = [
        (1, _loop_cmd(LOG_A)),
        (2, _loop_cmd(LOG_B)),
        (3, _loop_cmd(LOG_A)),  # LOG_A 的第二进程=垫片，不再增实例
    ]
    got = watchdog.watchdog_instances(rows)
    assert got is not None
    assert got["logical"] == 2 and got["loop_processes"] == 3


def test_run_log_path_is_case_insensitive():
    """Windows 路径大小写不同 = 同一个 run-log（不得数成两个实例）。"""
    rows = [(1, _loop_cmd(LOG_A)), (2, _loop_cmd(LOG_A.upper()))]
    got = watchdog.watchdog_instances(rows)
    assert got is not None
    assert got["logical"] == 1


def test_run_log_equals_form_and_quoted_path():
    """``--run-log=PATH`` 与带引号（含空格）的路径都认得。"""
    spaced = r"D:\repo dir\watchdog-runs\a b.log"
    rows = [
        (1, f"{SHIM} _watchdog-loop --run-log={LOG_A}"),
        (2, f'{SHIM} _watchdog-loop --run-log "{spaced}" --interval 60'),
    ]
    got = watchdog.watchdog_instances(rows)
    assert got is not None
    assert got["logical"] == 2
    assert spaced in got["run_logs"]


def test_mention_only_processes_are_not_instances():
    """仅「提到标记」的进程（grep / PowerShell 探针，无 --run-log）不算实例——防假阳性。"""
    rows = [
        (7, "bash -c grep -n _watchdog-loop /tmp/x"),
        (8, "powershell -NoProfile -Command Get-CimInstance ... _watchdog-loop ..."),
        (9, _loop_cmd(LOG_A)),
    ]
    got = watchdog.watchdog_instances(rows)
    assert got is not None
    assert got["logical"] == 1 and got["loop_processes"] == 1
    assert got["mentioned"] == 2


def test_unreadable_process_face_is_none_not_zero():
    """进程面读不到 → ``None``（**不是** 0）：折算成 0 会让验收假绿。"""
    assert watchdog.watchdog_instances(None) is None


def test_empty_process_list_is_zero():
    got = watchdog.watchdog_instances([])
    assert got is not None
    assert got["logical"] == 0 and got["loop_processes"] == 0 and got["mentioned"] == 0


# ---------------------------------------------------------------------------
# 工具面：status_lines 的读数行
# ---------------------------------------------------------------------------
def test_status_lines_reports_logical_instance_count():
    """`orch watchdog status` 的第四行给出逻辑实例数与口径（垫片对 → 1）。"""
    rows = [(16832, _loop_cmd(LOG_A)), (22412, _loop_cmd(LOG_A))]
    lines = watchdog.status_lines(scan=lambda marker: rows)

    joined = "\n".join(lines)
    assert "逻辑实例数 = 1" in joined
    assert "loop 进程 2" in joined
    assert "distinct --run-log" in joined  # 口径写进输出本身，不留在口头


def test_status_lines_marks_unknown_when_scan_unavailable():
    """进程面读不到 → 显式 ``?``，绝不显示 0（防假绿）。"""
    joined = "\n".join(watchdog.status_lines(scan=lambda marker: None))
    assert "逻辑实例数 = ?" in joined


def test_status_lines_cache_line_report_keeps_legacy_lines():
    """保留既有三行读数（看门狗态/心跳年龄/listener）与停止纪元段。"""
    lines = watchdog.status_lines(scan=lambda marker: [])
    joined = "\n".join(lines)
    assert "看门狗 =" in joined and "watch 心跳 age =" in joined
    assert "listener =" in joined
    assert "逻辑实例数 = 0" in joined


def test_status_lines_instances_can_be_disabled():
    """``instances=False`` 退回旧行为（不扫进程）。"""
    called: list[str] = []

    def _scan(marker: str):
        called.append(marker)
        return []

    lines = watchdog.status_lines(instances=False, scan=_scan)
    assert called == []
    assert "逻辑实例数" not in "\n".join(lines)


# ---------------------------------------------------------------------------
# 扫描原语契约（真实调用；只钉「返回行表或 None，绝不抛」）
# ---------------------------------------------------------------------------
def test_list_processes_matching_contract():
    """扫描器要么返回 ``[(pid, cmd), …]``，要么 ``None``（读不到）——**绝不抛**。"""
    rows = procs.list_processes_matching("_watchdog-loop")
    assert rows is None or isinstance(rows, list)
    if rows:
        assert all(isinstance(pid, int) and isinstance(cmd, str) for pid, cmd in rows)
