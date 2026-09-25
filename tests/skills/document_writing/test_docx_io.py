"""docx_io 测试：结构化读取 / 构建 / markdown 闭环（无需外部工具）。"""

from __future__ import annotations

import pytest

docx = pytest.importorskip("docx", reason="需要 python-docx：uv sync --extra writing")

from pysci.skills.document_writing.tools import docx_io  # noqa: E402


def test_build_and_read_roundtrip(tmp_path):
    out = tmp_path / "doc.docx"
    blocks = [
        docx_io.DocxBlock(kind="heading", text="Intro", level=1),
        docx_io.DocxBlock(kind="paragraph", text="Body sentence."),
        docx_io.DocxBlock(kind="bullet", text="point one"),
        docx_io.DocxBlock(kind="bullet", text="sub point", level=1),
        docx_io.DocxBlock(kind="table", rows=[["g", "Q"], ["0.5", "820"]]),
    ]
    docx_io.build_docx(blocks, out, title="Report")

    got = docx_io.read_docx(out)
    kinds = [b.kind for b in got]
    assert kinds[0] == "heading" and got[0].text == "Report"
    assert "heading" in kinds and "paragraph" in kinds and "bullet" in kinds and "table" in kinds
    tbl = next(b for b in got if b.kind == "table")
    assert tbl.rows[0][:2] == ["g", "Q"]
    sub = next(b for b in got if b.kind == "bullet" and b.text == "sub point")
    assert sub.level == 1


def test_markdown_to_docx_roundtrip(tmp_path):
    md = "# Title\n\n## Section\n- bullet\n  - nested\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nPlain para.\n"
    out = tmp_path / "from_md.docx"
    docx_io.markdown_to_docx(md, out)

    got = docx_io.read_docx(out)
    assert got[0].kind == "heading" and got[0].text == "Title"
    sec = next(b for b in got if b.text == "Section")
    assert sec.kind == "heading" and sec.level == 2
    nested = next(b for b in got if b.text == "nested")
    assert nested.kind == "bullet" and nested.level == 1
    assert any(b.kind == "table" and b.rows[1][:2] == ["1", "2"] for b in got)
    assert any(b.kind == "paragraph" and b.text == "Plain para." for b in got)


def test_add_block_appends(tmp_path):
    out = tmp_path / "doc.docx"
    docx_io.build_docx([docx_io.DocxBlock(kind="paragraph", text="first")], out)
    assert len(docx_io.read_docx(out)) == 1
    docx_io.add_block_to(out, docx_io.DocxBlock(kind="heading", text="Appended", level=2))
    got = docx_io.read_docx(out)
    assert len(got) == 2
    assert got[1].kind == "heading" and got[1].text == "Appended"


def test_markdown_to_docx_strips_bom(tmp_path):
    out = tmp_path / "bom.docx"
    docx_io.markdown_to_docx("\ufeff# Titled\n\nbody\n", out)
    got = docx_io.read_docx(out)
    assert got[0].text == "Titled"  # BOM 不应混入标题
