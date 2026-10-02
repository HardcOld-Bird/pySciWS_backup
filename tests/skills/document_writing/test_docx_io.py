"""docx_io 测试：结构化读取 / 构建 / 增量追加（无需外部工具）。

markdown→docx 写转换已改由 Pandoc 承担，测试见 test_pandoc_convert.py。
"""

from __future__ import annotations

import pytest

docx = pytest.importorskip("docx", reason="需要 python-docx")

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


def test_add_block_appends(tmp_path):
    out = tmp_path / "doc.docx"
    docx_io.build_docx([docx_io.DocxBlock(kind="paragraph", text="first")], out)
    assert len(docx_io.read_docx(out)) == 1
    docx_io.add_block_to(out, docx_io.DocxBlock(kind="heading", text="Appended", level=2))
    got = docx_io.read_docx(out)
    assert len(got) == 2
    assert got[1].kind == "heading" and got[1].text == "Appended"
