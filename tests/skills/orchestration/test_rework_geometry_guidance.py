"""F3/F4 教义分册：索引可见、预算内、行序案例已裁决（tb-3 R4 L.l3）。"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT

REF = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "rework-geometry-and-final-list.md"
)
TPL = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "taskbook-template-02-rework-fallback-list.md"
)


def test_reference_indexed_and_within_budget():
    text = REF.read_text(encoding="utf-8")
    assert "既定几何" in text and "现行定版清单" in text
    assert REF.name in TPL.read_text(encoding="utf-8")
    assert REF.stat().st_size <= 8192
    assert TPL.stat().st_size <= 8192


def test_f4_row_order_case_is_adjudicated():
    text = REF.read_text(encoding="utf-8")
    assert "场图行 1–2、雷达行 3" in text
    assert "2026-10-17 第 2 轮批语 3" in text
    assert "覆盖来源" in text


def test_f4_scan_geometry_case_is_adjudicated():
    text = REF.read_text(encoding="utf-8")
    assert "0.60（径向）×0.31（横向）" in text
    assert "近边 0.25 m" in text
    assert "R6 批语 2" in text and "twin 裁决 A" in text
    assert "R1 批语 5" in text  # 旧口径只作覆盖来源


def test_f4_colorbar_position_case_is_adjudicated():
    text = REF.read_text(encoding="utf-8")
    assert "每列紧邻右侧" in text
    assert "第 5 轮批语 1" in text
    assert "共同的右侧" in text  # 旧值仅作覆盖来源留档
