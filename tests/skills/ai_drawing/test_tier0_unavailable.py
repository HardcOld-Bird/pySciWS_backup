"""Tier 0（Qoder 内置 ImageGen）当前不可用 → 文档改指 Tier 1（backlog 20261015-205821-drawing）。

现场（drawing 成员报 2 次 + devops 复现 1 次）：

    Error: Image generation API error: 400
      {"code":"invalid_model_error","message":"Unsupported model \"1a54cc57-e614-4521-80cc-447de31dc896\""}

Tier 0 是「零安装/零成本兜底」路径，但它**当前在平台侧不可用**（模型 ID 不被支持），三次调用
（含 devops 复现）签名一致。修复不在本仓（平台侧），故按 success_criteria 的第二支做：
**技能文档注明 Tier 0 当前不可用、改指 Tier 1**，并把该口径同时固化到 `imagine doctor`（工具面）。

本文件钉三件事：① 三处文档都点明不可用 + 改指 Tier 1；② doctor 会打这条告警；③ 三册仍在 8KB 预算内。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.ai_drawing.tools import imagine

REFS = ORCHESTRATION_ROOT / "skills" / "ai-drawing" / "references"
SKILL = ORCHESTRATION_ROOT / "skills" / "ai-drawing" / "SKILL.md"
BACKENDS = REFS / "backends-01-overview.md"
WORKFLOWS = REFS / "workflows.md"


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_skill_marks_tier0_unavailable_and_points_to_tier1():
    md = _text(SKILL)
    at = md.index("Tier 0 — Qoder built-in")
    bullet = md[at : at + 420]
    assert "Currently unavailable" in bullet
    assert "invalid_model_error" in bullet
    assert "Tier 1" in bullet, "未给出回退路径"


def test_backends_records_the_error_signature_and_fallback():
    md = _text(BACKENDS)
    assert "Tier 0 is currently unavailable" in md
    assert "invalid_model_error" in md
    assert "fall back" in md and "Tier 1" in md
    assert "20261015-205821-drawing" in md  # 溯源到本单


def test_workflows_checklist_does_not_silently_send_to_tier0():
    md = _text(WORKFLOWS)
    assert "平台侧当前不可用" in md and "Tier 1" in md


def test_doctor_prints_the_tier0_warning(capsys):
    """工具面：`imagine doctor` 也点明（成员排障时先看它）。"""
    assert imagine.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "tier 0" in out.lower()
    assert "当前不可用" in out and "invalid_model_error" in out
    assert "Tier 1" in out


def test_docs_within_budget():
    for p in (SKILL, BACKENDS, WORKFLOWS):
        size = p.stat().st_size
        assert size <= 8192, f"{p.name} 超预算：{size}B"
