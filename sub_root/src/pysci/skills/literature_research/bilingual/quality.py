"""质量层（新增，2026-11-17）：回译校验 + 术语一致性核对——机械闸门，无 LLM 评审。

旧实现只有机械闸门（对账/占位/闭合/自检），本模块补两个可离线核验的缺口：

1. **回译校验**（:func:`backtranslation_findings`）：把中文译文经**可插拔回译器**
   （``zh -> en'`` 可调用对象；deepl-python / argostranslate 由依赖代理安装，本层不绑
   引擎）翻回英文，与 manifest 英文原文做归一化相似度比对（difflib），低于阈值报
   「存疑」级发现——抓漏译/截断/张冠李戴。
2. **术语一致性核对**（:func:`term_consistency_findings`）：术语表
   （``{英文术语: 中文译名}``，JSON）逐单元核对——英文出现术语而中文缺规范译名即报，
   防同一篇内同词多译。

发现只报告不删改（人工裁决是设计的一部分）；提交前的硬门仍是 :mod:`.reconcile`
（未译/占位拒出稿）与 :mod:`.manifest.check_work_structure`。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .reconcile import classify_translation


@dataclass(frozen=True)
class QualityFinding:
    """一条质量发现。

    Attributes:
        kind: ``backtranslation`` / ``term``。
        unit_id: 相关单元 ID。
        detail: 人类可读说明（含依据数值）。
        severity: ``warn``（存疑，交人工）/ ``fail``（术语缺规范译名）。
    """

    kind: str
    unit_id: str
    detail: str
    severity: str


def _norm(text: str) -> str:
    """归一化空白并小写（相似度比对口径）。"""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def backtranslation_findings(
    manifest: dict[str, Any],
    translations: dict[str, str],
    back_translator: Callable[[str], str],
    *,
    min_ratio: float = 0.5,
) -> list[QualityFinding]:
    """回译校验：已译单元回译成英文后与原文比对，相似度过低即报。

    Args:
        manifest: 工作区 manifest。
        translations: ``{ID: 译文}``。
        back_translator: 回译器 ``zh -> en'``（引擎可插拔；抛错由调用方处理）。
        min_ratio: 归一化相似度下限（0-1，difflib.SequenceMatcher.ratio）。

    Returns:
        发现清单（仅「已译」单元参与；未译/存疑由对账层负责）。
    """
    out: list[QualityFinding] = []
    for unit in manifest["units"]:
        uid = str(unit["id"])
        zh = translations.get(uid)
        if classify_translation(str(unit.get("en", "")), zh) != "done":
            continue
        back = back_translator(zh)
        ratio = SequenceMatcher(None, _norm(unit["en"]), _norm(back)).ratio()
        if ratio < min_ratio:
            out.append(
                QualityFinding(
                    kind="backtranslation",
                    unit_id=uid,
                    detail=(
                        f"回译相似度 {ratio:.2f} < {min_ratio}："
                        f"原文 {unit['en'][:40]!r} → 回译 {back[:40]!r}"
                    ),
                    severity="warn",
                )
            )
    return out


def term_consistency_findings(
    manifest: dict[str, Any],
    translations: dict[str, str],
    glossary: dict[str, str],
) -> list[QualityFinding]:
    """术语一致性核对：英文出现术语而中文缺规范译名即报（同词多译防复发）。

    Args:
        manifest: 工作区 manifest。
        translations: ``{ID: 译文}``。
        glossary: ``{英文术语: 中文译名}``（见 :func:`load_glossary`）。

    Returns:
        发现清单（``kind="term"``、``severity="fail"``——出稿前应修正或裁决）。
    """
    out: list[QualityFinding] = []
    for unit in manifest["units"]:
        uid = str(unit["id"])
        zh = translations.get(uid)
        if classify_translation(str(unit.get("en", "")), zh) != "done":
            continue
        for en_term, zh_term in glossary.items():
            if re.search(rf"\b{re.escape(en_term)}\b", str(unit["en"]), re.I) and (
                zh_term not in str(zh)
            ):
                out.append(
                    QualityFinding(
                        kind="term",
                        unit_id=uid,
                        detail=(
                            f"术语 {en_term!r} 应译 {zh_term!r}，该行未出现"
                            f"（行译文：{str(zh)[:40]!r}）"
                        ),
                        severity="fail",
                    )
                )
    return out


def load_glossary(path: Path) -> dict[str, str]:
    """载入术语表 JSON（``{"英文术语": "中文译名"}``）。

    Raises:
        ValueError: JSON 顶层不是字符串到字符串的映射。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"术语表须为 {{英文术语: 中文译名}} JSON 映射：{path}")
    return data


def make_back_translator() -> Callable[[str], str] | None:
    """构造回译器（``zh -> en``）：deepl-python（.env 的 ``DEEPL_AUTH_KEY``）。

    优雅降级（K2 2026-11-17 约定）：无 key / deepl 未装 → 返回 ``None``，调用方
    **跳过回译校验并提示**——主通道（译者逐句改 work.md）不受影响，不阻塞出稿。
    """
    key = os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY") or ""
    if not key:
        return None
    try:
        import deepl
    except ImportError:  # pragma: no cover - 依赖代理安装后消失
        return None
    translator = deepl.Translator(key)

    def back(zh: str) -> str:
        return str(
            translator.translate_text(zh, source_lang="ZH", target_lang="EN-US").text
        )

    return back
