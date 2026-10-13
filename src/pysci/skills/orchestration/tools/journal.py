"""决策日志（preference-journal）——评价驱动自改进体系的原料层（README §6，E0 启动）。

单文件 JSONL：`orchestration/state/preference-journal.jsonl`，**每行一条判决记录**，
schema 固定为 ``{ts, situation, options, choice, reason, source, kind}``。它是 twin 训练
语料与 override 审计的共同来源，故必须先于其他 E 阶段落地。

来源（``source`` 取值）：``backfill-readme13``（README §13 存量回灌）、``override``
（组长无判决依据的否决，§5.1）、``jury``/``bencher``（E3/E4 判决）、``mirror``（E6 盲审/
口述）、``verify``（§5.6 交付后核验）、``user``（用户 prompt 语料 / E7 上线宣告）。

``kind`` 粗分记录性质：``ruling``（裁决）、``override``（留痕否决）、``verdict``（判决）等。

路径取模块级 ``JOURNAL_PATH`` 且**在函数内读取全局**（非默认参数绑定）——测试 monkeypatch
``journal.JOURNAL_PATH`` 即隔离落盘面（同 registry 运行态的教训：dataclass/默认参数在定义时
已绑真实路径，无视 monkeypatch）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT

#: 决策日志文件（git 跟踪——长期资产，供 twin 训练与 override 审计回放）。
JOURNAL_PATH: Path = ORCH_STATE_ROOT / "preference-journal.jsonl"

#: 固定 schema（README §6 原料层）；记录缺 ts 时 append 自动补，缺其余键写入空串保持行结构齐整。
SCHEMA: tuple[str, ...] = (
    "ts",
    "situation",
    "options",
    "choice",
    "reason",
    "source",
    "kind",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize(record: dict) -> dict:
    """补全 schema：按 ``SCHEMA`` 键序，缺键置空串，缺 ``ts`` 填当前时间。

    多余键原样保留在尾部（不破坏固定前缀键序，向后兼容 mirror/核验可能附带的字段）。
    """
    out: dict = {}
    for key in SCHEMA:
        if key == "ts":
            out[key] = record.get("ts") or _now()
        else:
            out[key] = record.get(key, "")
    for key, value in record.items():
        if key not in out:
            out[key] = value
    return out


def append(record: dict, *, path: Path | None = None) -> dict:
    """追加一条判决记录（一行 JSON，UTF-8，不换行编码）。

    Args:
        record: 至少含 ``source``/``choice``；缺 ``ts`` 自动补，缺其余 schema 键置空串。
        path: 目标文件；默认模块级 ``JOURNAL_PATH``（读全局，可被 monkeypatch 隔离）。

    Returns:
        规范化后实际写入的记录 dict（供调用方回显/断言）。
    """
    target = path if path is not None else JOURNAL_PATH
    normalized = normalize(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(normalized, ensure_ascii=False)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return normalized


def read_records(path: Path | None = None) -> list[dict]:
    """读回全部记录（跳过空行；文件缺失返回空表）。坏行按读侧 fail-soft 跳过。"""
    target = path if path is not None else JOURNAL_PATH
    if not target.exists():
        return []
    records: list[dict] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return records


def line_count(path: Path | None = None) -> int:
    """非空行数（E0 判据 3 的核验尺子：应 ≥ README §13 裁决条目数）。"""
    return len(read_records(path))
