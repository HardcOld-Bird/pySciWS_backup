"""统一缓存 PDF 查找（cache find；20261018-221735-lit）。

实案：Nature 490 的 PDF 落在 ``cache/html_fulltext/10.1038_490472a/…pdf``，而按
``cache/pdfs/<doi>.pdf`` 扫描的批处理漏掉它，需人工复制。本文件钉住：

- `cache_manager.find_cached_pdf` 同时扫 ``pdfs/``（含 theme 子目录）与
  ``html_fulltext/<目录>/*.pdf``（目录名按 DOI 归一匹配）；
- `pysci-research cache find <DOI>` 给出同一入口的 CLI 面。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import cache_manager, research


@pytest.fixture
def cache_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pdfs = tmp_path / "cache" / "pdfs"
    html = tmp_path / "cache" / "html_fulltext"
    pdfs.mkdir(parents=True)
    html.mkdir(parents=True)
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(
        cache_manager,
        "settings",
        replace(cache_manager.settings, cache_pdfs=pdfs, cache_html_fulltext=html),
    )
    # cache find 现做内容身份检查（backlog 20261103-234914-lit）：封掉 OpenAlex 与
    # papers/ 写面，保测试封闭（个别用例再按需覆盖 get_work 返回值）。
    monkeypatch.setattr(research, "PAPERS_DIR", papers)
    monkeypatch.setattr(research.openalex_client, "get_work", lambda **kw: None)
    return pdfs, html


def test_find_in_pdfs_root_and_theme_subdir(cache_dirs):
    pdfs, _html = cache_dirs
    root_pdf = pdfs / "10-1103-physrevlett-117-034301.pdf"
    root_pdf.write_bytes(b"%PDF-1.4")
    theme_pdf = pdfs / "group_pubs" / "10-1126-science-1210713.pdf"
    theme_pdf.parent.mkdir()
    theme_pdf.write_bytes(b"%PDF-1.4")

    assert cache_manager.find_cached_pdf("10.1103/PhysRevLett.117.034301") == root_pdf
    assert cache_manager.find_cached_pdf("10.1126/science.1210713") == theme_pdf


def test_find_in_html_fulltext_bundle(cache_dirs):
    _pdfs, html = cache_dirs
    bundle = html / "10.1038_490472a"
    bundle.mkdir()
    pdf = bundle / "10.1038_490472a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert cache_manager.find_cached_pdf("10.1038/490472a") == pdf, (
        "html_fulltext 的 DOI 目录（下划线写法）必须被拾取"
    )
    assert (
        cache_manager.find_cached_pdf("10.1038/490472a", slug="10.1038_490472a") == pdf
    )


def test_find_returns_none_for_unknown_or_empty(cache_dirs):
    assert cache_manager.find_cached_pdf("10.9999/nope") is None
    assert cache_manager.find_cached_pdf("") is None


def test_cache_find_cli(cache_dirs, capsys):
    _pdfs, html = cache_dirs
    bundle = html / "10.1038_490472a"
    bundle.mkdir()
    pdf = bundle / "10.1038_490472a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert research.main(["cache", "find", "10.1038/490472a"]) == 0
    out = capsys.readouterr().out
    assert str(pdf) in out and "→" in out

    assert research.main(["cache", "find", "10.9999/nope"]) == 1
    assert "未找到已缓存 PDF" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 内容身份（backlog 20261103-234914-lit）：输出首页题名/元数据题名，文件名 DOI ≠
# 内容题名时告警——不再仅回路径
# ---------------------------------------------------------------------------
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


def test_cache_find_prints_content_title_and_warns_on_mismatch(
    cache_dirs, monkeypatch, capsys
):
    pdfs, _html = cache_dirs
    pdf = pdfs / "10-1103-physrevapplied-12-034021.pdf"
    _mk_pdf(
        pdf,
        [
            "Highly efficient acoustic refractive",
            "metasurfaces by harnessing near field",
        ],
        meta_title="Microsoft Word - Manuscript_PRAppl Letter.docx",
    )
    monkeypatch.setattr(
        research.openalex_client,
        "get_work",
        lambda **kw: {
            "title": "Highly efficient acoustic metagrating for enhanced sound transmission"
        },
    )
    assert research.main(["cache", "find", "10.1103/PhysRevApplied.12.034021"]) == 0
    out = capsys.readouterr().out
    assert "内容题名" in out and "Highly efficient acoustic refractive" in out
    assert "元数据题名：Microsoft Word - Manuscript" in out
    assert "[!] 题名不一致" in out


def test_cache_find_match_case_no_warning(cache_dirs, monkeypatch, capsys):
    pdfs, _html = cache_dirs
    title = "Highly efficient acoustic metagrating for enhanced sound transmission"
    pdf = pdfs / "10-1103-xyz.pdf"
    _mk_pdf(pdf, [title])
    monkeypatch.setattr(
        research.openalex_client, "get_work", lambda **kw: {"title": title}
    )
    assert research.main(["cache", "find", "10.1103/xyz"]) == 0
    out = capsys.readouterr().out
    assert "题名比对：与 DOI 元数据一致" in out
    assert "[!] 题名不一致" not in out
