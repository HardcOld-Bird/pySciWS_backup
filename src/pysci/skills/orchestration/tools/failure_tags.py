"""Tier A 失败标签：受控词表的读取、解析与滑窗计数（README §5.4）。

词表落在 ``orchestration/state/failure-tags.json``（git 跟踪）。**文件形状即协议**：
``{"version": 1, "discipline": "...", "tags": {<tag>: <定义>}, "infra_kinds": {…}}``——
Node 侧（``orchestration/guards/delivery-gate.mjs``）直接读同一文件校验组员交付的
``<blocked tags="a,b">`` 与 ``<infra_suggestion kind="…">``，Python 侧唯一读取口是本模块。
两侧口径必须一致，所以增补标签/分类只改文件（须附定义），不改代码。

本模块只懂「词表与标签」，不懂台账与 backlog：计数交给 :func:`failure_counts` 这种
纯函数形态，由 ledger（统计面）与 workflow（立案面）各自决定窗口与后果，避免
failure_tags 反过来依赖台账/审批链（那会让 guard 侧的等价实现更难对齐）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

TAGS_PATH: Path = ORCH_STATE_ROOT / "failure-tags.json"

#: 词表缺席/损坏时的兜底集合（只保**键**，定义空串——真定义在文件里）。存在的意义是
#: 让统计与立案在文件丢失时仍能跑（fail-open，不阻塞派发），而不是给组员一张空白名单。
#: 键集合与随附文件的一致性由 `test_failure_tags.py` 钉住（改一边必改另一边）。
DEFAULT_TAGS: dict[str, str] = {
    "whitelist_conflict": "",
    "cli_missing": "",
    "doc_stale": "",
    "tool_bug": "",
    "harness_gap": "",
    "model_capability": "",
    "quota": "",
    "env_broken": "",
    "merge_dirty_face": "",
    "design_smell": "",
    "unknown": "",
}

#: 提案分类（``<infra_suggestion kind="…">``）词表的兜底键集，口径同 :data:`DEFAULT_TAGS`。
#: 三面共读这一份词表：gate 拦（Node 侧默认清单由 test_infra_kind_vocab 钉住与文件一致）、
#: dispatch 落盘前复核（不合法不入库）、approve 入队前复核（backlog 20261012-120001-leader：
#: 非法 kind 的占位空壳曾绕过 gate 直达审批队列）。
DEFAULT_INFRA_KINDS: dict[str, str] = {"defect": "", "usability": ""}

#: ``<blocked reason="…">`` 成因档位的兜底键集（backlog 20261013-191213-devops）。``reason``
#: 把语义过载的 ``<blocked>`` 分档：``needs_decision``（待裁决，重派无效须停链等）与
#: ``capacity``（执行端余量不足，补足可重派）的最优处置相反，却都长成同一个 ``<blocked>``。
#: 键集与随附文件的一致性由 `test_failure_reason.py` 钉住（改一边必改另一边）。
DEFAULT_REASONS: dict[str, str] = {
    "needs_decision": "",
    "capacity": "",
    "blocked_by_env": "",
    "blocked_by_upstream": "",
}

#: tags 属性的分隔符：半/全角逗号分号与空白都收（组员在中文语境里手打，别逼他改符号）。
#: 与 guard 侧 ``splitTags`` 的字符集必须一致，否则「gate 放行、统计另算」会出现。
_SPLIT_RE = re.compile(r"[,，;；\s]+")


def _read_section(
    key: str, default: dict[str, str], path: Path | None
) -> dict[str, str]:
    """从词表文件取一个段落（``tags`` / ``infra_kinds``）为 ``{键: 定义}``。

    读不到（文件缺失/坏 JSON/段落缺席或为空）一律回落 ``default``：词表故障不该让派发、
    统计或审批崩掉，最多失去一次校验——但绝不回落成空白名单（那会把「一切都不合法」
    伪装成「一切都合法」的反面，同样不可用）。
    """
    target = path if path is not None else TAGS_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    section = data.get(key) if isinstance(data, dict) else None
    if not isinstance(section, dict) or not section:
        return dict(default)
    return {str(k): str(v) for k, v in section.items()}


def load_tags(path: Path | None = None) -> dict[str, str]:
    """读取受控词表：``{tag: 定义}``。

    Args:
        path: 覆盖读取路径（测试注入用）；缺省取模块级 :data:`TAGS_PATH`。

    Returns:
        文件可解析且含非空 ``tags`` 时返回文件内容（**整体覆盖**默认，含删减语义）；
        否则回落 :data:`DEFAULT_TAGS`。异常一律吞掉（fail-open）：词表读取失败不该
        让派发或统计崩掉，最多失去一次标签校验。
    """
    return _read_section("tags", DEFAULT_TAGS, path)


def load_infra_kinds(path: Path | None = None) -> dict[str, str]:
    """读取提案分类词表：``{kind: 定义}``（读取与回落口径同 :func:`load_tags`）。"""
    return _read_section("infra_kinds", DEFAULT_INFRA_KINDS, path)


def load_reasons(path: Path | None = None) -> dict[str, str]:
    """读取 blocked 成因档位词表：``{reason: 定义}``（读取与回落口径同 :func:`load_tags`）。"""
    return _read_section("reasons", DEFAULT_REASONS, path)


def parse_tags(value: str | None) -> list[str]:
    """把 ``tags="a, b"`` 属性值解析成标签列表：小写化、去空、按首次出现去重。

    Args:
        value: 属性原文（``None``/空串 → 空列表 = 未标注，协议里合法）。

    Returns:
        规范化后的标签列表（**不**校验词表——校验是 gate 与统计两侧各自的职责，
        解析器保持无状态才能被双方复用）。
    """
    out: list[str] = []
    for part in _SPLIT_RE.split((value or "").strip()):
        tag = part.strip().strip("\"'").lower()
        if tag and tag not in out:
            out.append(tag)
    return out


def unknown_tags(tags: list[str], vocab: dict[str, str] | None = None) -> list[str]:
    """返回不在词表内的标签（保持输入顺序）；``vocab`` 缺省现场读取。"""
    table = load_tags() if vocab is None else vocab
    return [t for t in tags if t not in table]


def parse_reason(value: str | None) -> str:
    """``reason="…"`` 属性值 → 单个成因键（小写化、去引号与首尾空白）。

    reason 是**单值档位**（不是 :func:`parse_tags` 那样的列表）：整串即一个键。出现分隔符
    或第二个词即判非法（落到词表外，由 gate 退回）——**不切分**，否则
    ``reason="capacity, foo"`` 会静默丢掉 ``foo`` 又放行 ``capacity``，把「标错」伪装成
    「标对」。缺省空串=未标注（合法）。
    """
    return (value or "").strip().strip("\"'").lower()


def failure_counts(entries: list[Any]) -> dict[str, dict[str, int]]:
    """标签 × 成员计数：``{tag: {member: 次数}}``（只数**带标签**的台账行）。

    Args:
        entries: 带 ``tags`` 与 ``member`` 属性的对象列表（:class:`LedgerEntry`，
            或同形状的字典/轻量替身——纯函数，不挑类型）。

    Returns:
        嵌套计数字典；无任何带标签行时为 ``{}``。滑窗由**调用方**负责（见
        :func:`within_window`），本函数只数拿到手的那批。
    """
    counts: dict[str, dict[str, int]] = {}
    for e in entries:
        member = str(_field(e, "member", ""))
        for tag in _field(e, "tags", []) or []:
            row = counts.setdefault(str(tag), {})
            row[member] = row.get(member, 0) + 1
    return counts


def within_window(
    entries: list[Any],
    *,
    days: int | None = None,
    hops: int | None = None,
) -> list[Any]:
    """取滑窗内的台账行：**天数窗与跳数窗取并集**（任一命中即入窗）。

    Args:
        entries: 台账行列表（按 ``ts`` 追加序，末行最新）。
        days: 最近 N 天（``None``/非正 → 该支不生效）。
        hops: 最近 N 跳（``None``/非正 → 该支不生效）。

    Returns:
        入窗行（保持原顺序）。并集而非交集是刻意的：14 天窗口在新体系头几天根本
        攒不出历史，而 20 跳窗口在长停摆后又太窄——README §5.4 写「14 天或 20 跳」
        正是这个意思。两侧都关（非法参数）则返回全集（不做窗口约束）。
    """
    from datetime import datetime, timedelta

    dated = days and days > 0
    counted = hops and hops > 0
    if not dated and not counted:
        return list(entries)
    keep: set[int] = set()
    if dated:
        cutoff = (datetime.now().astimezone() - timedelta(days=days)).isoformat()
        for i, e in enumerate(entries):
            if str(_field(e, "ts", "")) >= cutoff:
                keep.add(i)
    if counted:
        keep.update(range(max(0, len(entries) - hops), len(entries)))
    return [e for i, e in enumerate(entries) if i in keep]


def _field(obj: Any, name: str, default: Any) -> Any:
    """属性或字典键两种形态都认（纯函数要能吃的了 LedgerEntry，也吃测试替身）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def summarize_counts(counts: dict[str, dict[str, int]]) -> int:
    """计数表的总次数（渲染与阈值判定的公共口径，免得两侧各写一遍求和）。"""
    return sum(n for row in counts.values() for n in row.values())
