"""理论计算会话脚手架与持久化杂件（④域"杂项"胶水）。

ADR（自研理由，2026-11-17）：theory CLI 全集已按用户裁决砍除（调用形态＝纯
``uv run python`` + 模板），但会话目录约定（``results/``、``plots/``、``notes.md``）
仍需**唯一实现**，防止各计算脚本自造目录结构漂移；本模块只承载该约定 + 数值结果/日志
落盘，不做 run/list 编排（shell 直跑即可）。路径一律走 :mod:`pysci.paths`
（``research_theory_dir`` / ``assert_within_data`` 护栏，无 ``parents[N]``）：
会话固定落在 ``data/research/<n>_<name>/theory/<slug>/``，data/ 外写入即报错。

机械断言：``tests/skills/theory_calculation/test_session.py``。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pysci.paths import assert_within_data, research_theory_dir

_NOTES_HEADER = "# 理论计算日志（时间线：每步操作与发现）\n\n"


def ensure_session(name: str, slug: str) -> Path:
    """建立（或复用）一条理论计算会话的目录骨架。

    骨架：``results/``（数值 .npz/.csv）、``plots/``（探索 PNG）、``notes.md``
    （计算日志，首次写入表头、之后不覆盖）。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"example"``。
        slug: 计算会话 slug，如 ``"ep_scan"``。

    Returns:
        会话目录绝对路径。
    """
    session = assert_within_data(
        research_theory_dir(name, slug=slug), what="理论计算会话"
    )
    (session / "results").mkdir(parents=True, exist_ok=True)
    (session / "plots").mkdir(parents=True, exist_ok=True)
    notes = session / "notes.md"
    if not notes.exists():
        notes.write_text(_NOTES_HEADER, encoding="utf-8")
    return session


def save_results(
    session_dir: Path | str, data: Mapping[str, Any], name: str = "results"
) -> Path:
    """把数值结果数组落盘为会话 ``results/<name>.npz``（``np.savez`` 语义）。

    Args:
        session_dir: :func:`ensure_session` 返回的会话目录。
        data: 数组字典（键=变量名，值=array-like）。
        name: 文件名前缀（不含扩展名）。

    Returns:
        生成的 ``.npz`` 路径。
    """
    session = assert_within_data(Path(session_dir), what="理论计算结果")
    results_dir = session / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.npz"
    np.savez(out, **data)
    return out


def append_note(session_dir: Path | str, entry: str) -> Path:
    """向会话 ``notes.md`` 追加一条带时间戳的日志（``## <ISO 时间>`` 小节）。

    Args:
        session_dir: :func:`ensure_session` 返回的会话目录。
        entry: 日志正文（Markdown）。

    Returns:
        ``notes.md`` 路径。
    """
    session = assert_within_data(Path(session_dir), what="理论计算日志")
    notes = session / "notes.md"
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with notes.open("a", encoding="utf-8") as f:
        f.write(f"## {stamp}\n{entry.rstrip()}\n\n")
    return notes
