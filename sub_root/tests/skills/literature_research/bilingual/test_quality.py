"""质量层回归：回译校验 + 术语一致性核对（纯函数，注入回译器，离线确定）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysci.skills.literature_research.bilingual import quality

MANIFEST = {
    "units": [
        {"id": "S0001", "en": "The metamaterial works well."},
        {"id": "S0002", "en": "We measured the band gap."},
        {"id": "S0003", "en": "Not translated yet."},
    ]
}
TRANSLATIONS = {
    "S0001": "超材料工作良好。",
    "S0002": "我们测量了带隙。",
    "S0003": "Not translated yet.",
}


# ---------------------------------------------------------------------------
# 回译校验
# ---------------------------------------------------------------------------
def test_backtranslation_flags_low_similarity() -> None:
    """回译与原文相似度过低（漏译/截断/张冠李戴）报 warn 发现；未译单元不参与。"""

    def back_translator(zh: str) -> str:
        return "Completely unrelated output."

    findings = quality.backtranslation_findings(
        MANIFEST, TRANSLATIONS, back_translator, min_ratio=0.5
    )
    assert {f.unit_id for f in findings} == {"S0001", "S0002"}
    assert all(f.kind == "backtranslation" and f.severity == "warn" for f in findings)
    assert "回译相似度" in findings[0].detail


def test_backtranslation_passes_faithful_translation() -> None:
    """回译高度贴合原文时不报（min_ratio 之下才是发现）。"""

    def back_translator(_zh: str) -> str:
        return "The metamaterial works well."

    findings = quality.backtranslation_findings(
        MANIFEST, {"S0001": "超材料工作良好。"}, back_translator, min_ratio=0.5
    )
    assert findings == []


def test_backtranslation_min_ratio_boundary() -> None:
    """阈值可调：min_ratio=1.0 时任何回译差异都报。"""

    def back_translator(_zh: str) -> str:
        return "The metamaterial works well!"

    findings = quality.backtranslation_findings(
        MANIFEST, {"S0001": "超材料工作良好。"}, back_translator, min_ratio=1.0
    )
    assert [f.unit_id for f in findings] == ["S0001"]


# ---------------------------------------------------------------------------
# 术语一致性
# ---------------------------------------------------------------------------
def test_term_consistency_flags_missing_glossary_term() -> None:
    """英文出现术语而中文缺规范译名 → fail 发现（同词多译防复发）。"""
    findings = quality.term_consistency_findings(
        MANIFEST, TRANSLATIONS, {"metamaterial": "超材料", "band gap": "禁带"}
    )
    assert [(f.unit_id, f.severity) for f in findings] == [("S0002", "fail")]
    assert "band gap" in findings[0].detail and "禁带" in findings[0].detail


def test_term_consistency_passes_with_canonical_terms() -> None:
    """规范译名齐备时不报。"""
    findings = quality.term_consistency_findings(
        MANIFEST, TRANSLATIONS, {"metamaterial": "超材料", "band gap": "带隙"}
    )
    assert findings == []


def test_term_consistency_word_boundary_matching() -> None:
    """术语按词边界匹配（``gap`` 不命中 ``bandgap`` 的英文侧误报口径）。"""
    findings = quality.term_consistency_findings(
        {"units": [{"id": "S0001", "en": "The bandgap closes."}]},
        {"S0001": "带隙闭合。"},
        {"gap": "缝隙"},
    )
    assert findings == []


def test_term_consistency_skips_untranslated_units() -> None:
    """未译单元不参与术语核对（由对账层负责）。"""
    findings = quality.term_consistency_findings(
        {"units": [{"id": "S0003", "en": "The metamaterial works."}]},
        {"S0003": "The metamaterial works."},
        {"metamaterial": "超材料"},
    )
    assert findings == []


# ---------------------------------------------------------------------------
# 术语表载入
# ---------------------------------------------------------------------------
def test_load_glossary_json(tmp_path: Path) -> None:
    """JSON 映射载入；非映射形态报 ValueError。"""
    p = tmp_path / "glossary.json"
    p.write_text(json.dumps({"metamaterial": "超材料"}, ensure_ascii=False), encoding="utf-8")
    assert quality.load_glossary(p) == {"metamaterial": "超材料"}
    p.write_text(json.dumps(["not", "a", "map"]), encoding="utf-8")
    with pytest.raises(ValueError, match="术语表"):
        quality.load_glossary(p)


# ---------------------------------------------------------------------------
# 回译引擎优雅降级（K2 约定：无 DEEPL_AUTH_KEY 跳过并提示，不阻塞主通道）
# ---------------------------------------------------------------------------
def test_make_back_translator_degrades_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无 DEEPL_AUTH_KEY：返回 None（调用方跳过回译校验），不抛错。"""
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    assert quality.make_back_translator() is None
