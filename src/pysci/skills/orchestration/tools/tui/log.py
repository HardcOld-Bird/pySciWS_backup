"""TUI 运行日志（backlog 20261017-120500）：生命周期 + 未捕获异常 traceback 落盘。

为什么需要：本仓此前**无 TUI 运行日志**，04:48 的 MouseDown 崩溃只能靠 traceback 机理推演。
日志面独立于 orch 的 watch/events——它是「终端里发生了什么」的唯一留痕，落 ``data/others/logs/``
（gitignored，运行态）。
"""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 日志目录（测试可 monkeypatch）。
LOG_DIR: Path = PROJECT_ROOT / "data" / "others" / "logs"


def log_path() -> Path:
    """当日日志文件：``data/others/logs/tui-YYYYMMDD.log``。"""
    return LOG_DIR / f"tui-{datetime.now().strftime('%Y%m%d')}.log"


def log(msg: str) -> None:
    """追加一行「时间戳 + 消息」；**绝不抛**（日志面不该反噬 TUI）。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_path().open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def log_exception(exc: BaseException, where: str = "uncaught") -> None:
    """把异常类型/消息 + 完整 traceback 落盘（``where`` 标出事发面）。"""
    log(f"[{where}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
