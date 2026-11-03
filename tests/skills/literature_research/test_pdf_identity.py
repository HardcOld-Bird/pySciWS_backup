"""PDF 内容身份探针与题名一致性（backlog 20261103-234914-lit）。

实案：任务书声明 `cache/pdfs/10-1103-physrevapplied-12-034021.pdf`（已按 DOI 命名），
实际内容是同一作者组**另一版手稿**（元数据 `Microsoft Word - Manuscript_…docx`），
文件名 DOI ≠ 内容题名而无任何环节比对——静默版本错配。本文件钉住：

- `cache_manager.pdf_content_probe`：首页题名（前 3 非空行）+ 首行 + 元数据题名；
- `cache_manager.title_match`：题名粗判等价（折行/大小写/标点不敏感，fail-open）；
- `cache_manager.title_consistency_note`：PDF/抽取件/DOI 元数据三方题名比对告警；
- `research._backfill_note_local_pdf`：papers/ 台账 `local_pdf_path` 自动回填（方案③）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import cache_manager, research


def _mk_pdf(path: Path, lines: list[str], *, meta_title: str = "") -> Path:
    import pymupdf

    doc = pymupdf.open()
    pg = doc.new_page(width=300, height=200)
    y = 40
    for ln in lines:
        pg.insert_text((30, y), ln, fontsize=11)
        y += 16
    if meta_title:
        doc.set_metadata({"title": meta_title})
    doc.save(str(path))
    doc.close()
    return path


MANUSCRIPT_TITLE = "Highly efficient acoustic refractive metasurfaces by harnessing near field coupling"
PUBLISHED_TITLE = (
    "Highly efficient acoustic metagrating for enhanced sound transmission"
)


def test_probe_reads_title_first_line_and_meta(tmp_path: Path) -> None:
    pdf = _mk_pdf(
        tmp_path / "a.pdf",
        [
            "Highly efficient acoustic refractive",
            "metasurfaces by harnessing",
            "Authors",
        ],
        meta_title="Microsoft Word - Manuscript_PRAppl Letter.docx",
    )
    probe = cache_manager.pdf_content_probe(pdf)
    assert probe["title"].startswith(
        "Highly efficient acoustic refractive metasurfaces"
    )
    assert probe["first_line"] == "Highly efficient acoustic refractive"
    assert probe["meta_title"] == "Microsoft Word - Manuscript_PRAppl Letter.docx"


def test_probe_fail_open_on_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4")
    probe = cache_manager.pdf_content_probe(bad)
    assert probe == {"title": "", "first_line": "", "meta_title": ""}


def test_title_match_tolerant() -> None:
    assert cache_manager.title_match(PUBLISHED_TITLE, PUBLISHED_TITLE.upper())
    # 折行连字号：meta-surfaces 与 metasurfaces 归一后同 token
    assert cache_manager.title_match("A metasurfaces study", "A meta-surfaces study")
    assert cache_manager.title_match("钙超胞声子晶体", "钙超胞声子晶体研究")
    assert not cache_manager.title_match(MANUSCRIPT_TITLE, PUBLISHED_TITLE)
    assert cache_manager.title_match("", PUBLISHED_TITLE)  # 无可比性 → 放行


def test_consistency_note_flags_each_mismatched_pair(tmp_path: Path) -> None:
    pdf = _mk_pdf(
        tmp_path / "b.pdf",
        [
            "Highly efficient acoustic refractive",
            "metasurfaces by harnessing near field",
            "coupling",
        ],
    )
    lines = cache_manager.title_consistency_note(
        pdf=pdf, md_title=MANUSCRIPT_TITLE, ref_title=PUBLISHED_TITLE, where="t"
    )
    # PDF/抽取件同为手稿题名 → 只有 DOI 元数据相关的两对告警
    assert len(lines) == 2, lines
    assert all("[!] 题名不一致（t）" in ln for ln in lines)
    assert any("DOI 元数据" in ln and "PDF 内容题名" in ln for ln in lines)
    assert any("DOI 元数据" in ln and "抽取件题名" in ln for ln in lines)


def test_consistency_note_silent_when_all_match(tmp_path: Path) -> None:
    pdf = _mk_pdf(tmp_path / "c.pdf", [PUBLISHED_TITLE])
    assert (
        cache_manager.title_consistency_note(
            pdf=pdf, md_title=PUBLISHED_TITLE, ref_title=PUBLISHED_TITLE, where="t"
        )
        == []
    )


def test_backfill_note_local_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(research, "PAPERS_DIR", papers)
    (papers / "2019_hou.md").write_text(
        "---\ndoi: 10.1103/PhysRevApplied.12.034021\nlocal_pdf_path: ''\n---\n\nbody\n",
        encoding="utf-8",
    )
    pdf = _mk_pdf(tmp_path / "p.pdf", ["x"])
    msg = research._backfill_note_local_pdf("10.1103/PhysRevApplied.12.034021", pdf)
    assert "local_pdf_path" in msg
    saved = (papers / "2019_hou.md").read_text(encoding="utf-8")
    assert str(pdf) in saved
    # 幂等：已有值不再回填
    assert (
        research._backfill_note_local_pdf("10.1103/PhysRevApplied.12.034021", pdf) == ""
    )


def test_backfill_skips_without_note_or_filled(tmp_path: Path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(research, "PAPERS_DIR", papers)
    pdf = _mk_pdf(tmp_path / "p.pdf", ["x"])
    assert research._backfill_note_local_pdf("10.1/x", pdf) == ""  # 无笔记
    (papers / "a.md").write_text(
        "---\ndoi: 10.1/y\nlocal_pdf_path: /old/path.pdf\n---\n\nb\n", encoding="utf-8"
    )
    assert research._backfill_note_local_pdf("10.1/y", pdf) == ""  # 已有值
