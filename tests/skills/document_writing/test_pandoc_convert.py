"""pandoc_convert 测试：Markdown→docx/pptx（Pandoc）回读闭环 + 降级分支。

写转换依赖 pandoc 二进制（pypandoc-binary 随 uv sync 装入）；不可用时相关用例跳过。
回读复用 docx_io.read_docx / pptx_io.read_pptx（python-docx / python-pptx）。
"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing.tools import pandoc_convert
from pysci.skills.document_writing.tools.config import settings

requires_pandoc = pytest.mark.skipif(
    not settings.pandoc_ready,
    reason="需要 pandoc（uv sync 应随 pypandoc-binary 装入）",
)

docx_io = pytest.importorskip(
    "pysci.skills.document_writing.tools.docx_io", reason="需要 python-docx"
)
pptx_io = pytest.importorskip(
    "pysci.skills.document_writing.tools.pptx_io", reason="需要 python-pptx"
)


@requires_pandoc
def test_md_to_docx_roundtrip(tmp_path):
    md = tmp_path / "in.md"
    md.write_text(
        "# Title\n\n## Section\n\n- bullet\n- nested\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\nPlain para.\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.docx"
    got_path = pandoc_convert.md_to_docx(md, out)
    assert got_path == out and out.exists()

    blocks = docx_io.read_docx(out)
    texts = [b.text for b in blocks]
    assert "Title" in texts
    assert "Section" in texts
    assert "Plain para." in texts
    assert any(b.kind == "table" for b in blocks), "管道表应转为 Word 表格"


@requires_pandoc
def test_md_to_pptx_notes_and_content(tmp_path):
    md = tmp_path / "in.md"
    md.write_text(
        "# Deck Title\n\n## Slide One\n\n- bullet one\n\n"
        "::: notes\nspoken words here\n:::\n\n"
        "## Slide Two\n\n- second point\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.pptx"
    pandoc_convert.md_to_pptx(md, out, slide_level=2)
    assert out.exists()

    slides = pptx_io.read_pptx(out)
    assert len(slides) >= 2
    content = " ".join((s.title or "") + " " + " ".join(s.paragraphs) for s in slides)
    assert "Slide One" in content
    notes = " ".join((s.notes or "") for s in slides)
    assert "spoken words" in notes, "`::: notes` 应落入演讲者备注"


def test_md_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pandoc_convert.md_to_docx(tmp_path / "nope.md", tmp_path / "o.docx")


@requires_pandoc
def test_reference_doc_missing_raises(tmp_path):
    md = tmp_path / "in.md"
    md.write_text("# T\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        pandoc_convert.md_to_docx(
            md, tmp_path / "o.docx", reference_doc=tmp_path / "nope.docx"
        )


def test_pandoc_not_available(monkeypatch, tmp_path):
    """find_pandoc 返回 None 时应抛 PandocNotAvailable（供 compose 降级为 exit 3）。"""

    class _NoPandoc:
        def find_pandoc(self):
            return None

    monkeypatch.setattr(pandoc_convert, "settings", _NoPandoc())
    md = tmp_path / "in.md"
    md.write_text("# T\n", encoding="utf-8")
    with pytest.raises(pandoc_convert.PandocNotAvailable):
        pandoc_convert.md_to_docx(md, tmp_path / "o.docx")
