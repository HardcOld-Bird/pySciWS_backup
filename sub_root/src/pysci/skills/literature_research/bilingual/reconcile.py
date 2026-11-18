"""work.md 对账三桶与占位假译判定（status / submit / TM 回填共用的纯函数层）。

判据（旧实现平移，2026-10-23 事故防复发语义不变）：

- **未译**：缺 ID、内容与英文原文相同、或为**占位假译**（夹具混入交付物事故口径）；
- **已译**：含 CJK 字符（且非占位）；
- **存疑**：其余（无 CJK 的异文，交译者自查）。

占位假译双判据：固定前缀族（``假译``/``旧译-``）**或**「仅差固定前缀」等值
（``译文 == 前缀 + 英文原文``）。占位**不计已译**、**拒绝出稿**、TM **不回填**。
"""

from __future__ import annotations

import re
from typing import Any

#: CJK 统一表意文字基本区（中文判据）。
_CJK_RE = re.compile(r"[一-鿿]")

#: 占位译文的固定前缀族（测试夹具/冒烟假译；backlog 20261023-162903-lit）。
#: 「假译」「旧译-」两支是实锤夹具族（2026-10-23 事故文本「假译：旧译-S0001-0」等）。
PLACEHOLDER_PREFIXES: tuple[str, ...] = ("假译", "旧译-")

#: 「仅差固定前缀」判定用的更宽前缀族：译文 == 前缀 + 英文原文时即占位（精确判据，
#: 不会误伤以「中译/旧译」等字起头的真实译文——那要求与英文原文逐字只差前缀）。
_PLACEHOLDER_EQ_PREFIXES: tuple[str, ...] = (
    "假译：", "假译:", "假译", "旧译-", "旧译 ", "旧译", "中译 ", "中译",
)


def has_cjk(text: str) -> bool:
    """文本是否含 CJK 字符（「已译」判据）。"""
    return bool(_CJK_RE.search(str(text)))


def is_placeholder_translation(zh: str, en: str = "") -> bool:
    """占位译文判定（固定前缀启发式 + 「仅差固定前缀」等值判据）。

    测试夹具/冒烟假译（``假译：…``、``旧译-…``）与「前缀+英文原文」形态都算占位——
    它们含 CJK 会被「含 CJK=已译」口径误判，从而被 status 计数、被 submit 当交付物
    出稿（2026-10-23 实锤事故）。
    """
    z = str(zh).strip()
    if not z:
        return False
    if any(z.startswith(p) for p in PLACEHOLDER_PREFIXES):
        return True
    e = str(en).strip()
    return bool(e) and any(z == f"{p}{e}" for p in _PLACEHOLDER_EQ_PREFIXES)


def classify_translation(en: str, zh: str | None) -> str:
    """单行译文状态：``untranslated``/``done``/``doubtful``。

    缺 ID 或内容与 en 相同记「未译」，**占位译文（假译类）也记「未译」**（不计已译），
    含 CJK 字符记「已译」，其余记「存疑」。
    """
    if zh is None or zh.strip() == en.strip():
        return "untranslated"
    if is_placeholder_translation(zh, en):
        return "untranslated"
    return "done" if has_cjk(zh) else "doubtful"


def placeholder_ids(
    manifest: dict[str, Any], translations: dict[str, str]
) -> list[str]:
    """manifest 单元中译文为占位（假译类）的 ID 清单（status/submit 提示用）。"""
    return [
        str(u["id"])
        for u in manifest["units"]
        if is_placeholder_translation(
            translations.get(u["id"], ""), str(u.get("en", ""))
        )
    ]


def reconcile(
    manifest: dict[str, Any], translations: dict[str, str]
) -> dict[str, list[str]]:
    """按 ID 对账，返回 ``{"untranslated": [...], "doubtful": [...], "done": [...]}``。"""
    buckets: dict[str, list[str]] = {"untranslated": [], "doubtful": [], "done": []}
    for unit in manifest["units"]:
        st = classify_translation(unit["en"], translations.get(unit["id"]))
        buckets[st].append(unit["id"])
    return buckets
