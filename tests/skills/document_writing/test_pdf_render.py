"""pdf_render 测试：用 pymupdf 现场造一个多页 PDF，验证渲染成 PNG。

无需 TeX —— 直接构造 PDF，覆盖「编译后看图校对」闭环的渲染环节。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="需要 pymupdf（随 pymupdf4llm 提供）")

from pysci.skills.document_writing.tools import pdf_render  # noqa: E402


def _make_pdf(path: Path, n_pages: int = 3) -> None:
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()  # 默认 A4
        page.insert_text((72, 100), f"Page {i + 1} — layout proof", fontsize=18)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    _make_pdf(p, 3)
    return p


def test_page_count(sample_pdf: Path):
    assert pdf_render.page_count(sample_pdf) == 3


def test_render_all_pages(sample_pdf: Path, tmp_path: Path):
    out = tmp_path / "renders"
    pngs = pdf_render.render_pdf_pages(sample_pdf, out_dir=out, dpi=100)
    assert len(pngs) == 3
    for p in pngs:
        assert p.exists() and p.suffix == ".png"
        assert p.stat().st_size > 0


def test_render_selected_pages(sample_pdf: Path, tmp_path: Path):
    pngs = pdf_render.render_pdf_pages(
        sample_pdf, out_dir=tmp_path / "r2", pages=[1, 3]
    )
    assert len(pngs) == 2
    assert pngs[0].name.endswith("001.png")
    assert pngs[1].name.endswith("003.png")


def test_render_max_pages(sample_pdf: Path, tmp_path: Path):
    pngs = pdf_render.render_pdf_pages(
        sample_pdf, out_dir=tmp_path / "r3", max_pages=2
    )
    assert len(pngs) == 2


def test_render_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pdf_render.render_pdf_pages(tmp_path / "nope.pdf")
