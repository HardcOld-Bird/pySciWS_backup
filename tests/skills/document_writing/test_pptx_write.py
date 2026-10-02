"""pptx_io 写入能力测试：build_pptx / add_slide_to 回读闭环。

markdown→pptx 写转换已改由 Pandoc 承担，测试见 test_pandoc_convert.py。
"""

from __future__ import annotations

import pytest

pptx = pytest.importorskip("pptx", reason="需要 python-pptx")

from pysci.skills.document_writing.tools import pptx_io  # noqa: E402


def test_build_pptx_roundtrip(tmp_path):
    out = tmp_path / "deck.pptx"
    specs = [
        pptx_io.SlideSpec(
            title="Key Results",
            bullets=["Observed EP of order 4", "  Gain contrast 12 dB"],
            notes="Stress robustness.",
        ),
    ]
    pptx_io.build_pptx(specs, out, deck_title="Gain-EP", deck_subtitle="Report 2026")

    slides = pptx_io.read_pptx(out)
    assert len(slides) == 2  # 标题页 + 内容页
    assert slides[0].title == "Gain-EP"
    assert "Report 2026" in slides[0].paragraphs  # 副标题经 read 归入 paragraphs
    assert slides[1].title == "Key Results"
    # 层级保留：第二项为 level-1（带缩进前缀）
    assert any(
        "Gain contrast" in p and p.startswith("  ") for p in slides[1].paragraphs
    )
    assert slides[1].notes == "Stress robustness."


def test_add_slide_to_appends(tmp_path):
    out = tmp_path / "deck.pptx"
    pptx_io.build_pptx([], out, deck_title="T")
    assert len(pptx_io.read_pptx(out)) == 1
    pptx_io.add_slide_to(out, pptx_io.SlideSpec(title="Second", bullets=["a"]))
    slides = pptx_io.read_pptx(out)
    assert len(slides) == 2
    assert slides[1].title == "Second"
