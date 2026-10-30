"""docs FTS5 知识索引纯 Python 单测：切章（含 MinerU 分块页码跟踪）+ 建索引 + 检索 + 按节读取。

用 tmp_path 造一本小 Markdown 手册，monkeypatch ``docs.settings`` 指到临时目录，
既不触碰真实 ``cache/doc_index.db``，也不需要 COMSOL 或 MinerU。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pysci.skills.comsol_simulation.tools import docs

SAMPLE_MD = """\
<!-- ==================== pages 1-50 ==================== -->
# Introduction
Pressure acoustics solves for the acoustic pressure field.

## Perfectly Matched Layer
The perfectly matched layer (PML) absorbs outgoing waves without reflection.

<!-- ==================== pages 51-100 ==================== -->
# API Reference
Call model.study to run a study, and model.result to export plots.
"""


@pytest.fixture()
def indexed(tmp_path, monkeypatch):
    """建一个只含 FakeManual.md 的临时文档区并索引好，返回诊断信息。"""
    docs_dir = tmp_path / "docs"
    cache_dir = tmp_path / "cache"
    docs_dir.mkdir()
    cache_dir.mkdir()
    (docs_dir / "FakeManual.md").write_text(SAMPLE_MD, encoding="utf-8")
    monkeypatch.setattr(
        docs,
        "settings",
        SimpleNamespace(
            docs_dir=docs_dir,
            cache_dir=cache_dir,
            doc_index_db=cache_dir / "doc_index.db",
        ),
    )
    n = docs.build_index()
    return SimpleNamespace(n=n, docs_dir=docs_dir, db=cache_dir / "doc_index.db")


def test_split_sections_tracks_pages():
    secs = docs.split_sections(SAMPLE_MD, "FakeManual")
    heads = [(s.heading, s.level, s.pages) for s in secs]
    assert heads == [
        ("Introduction", 1, "1-50"),
        ("Perfectly Matched Layer", 2, "1-50"),
        ("API Reference", 1, "51-100"),
    ]
    assert "absorbs outgoing waves" in secs[1].body
    assert all(s.doc == "FakeManual" for s in secs)


def test_split_sections_no_headings():
    assert docs.split_sections("just plain text\nno headings here", "D") == []


def test_build_index_counts_and_db(indexed):
    assert indexed.n == 3
    assert indexed.db.exists()


def test_list_docs(indexed):
    lst = docs.list_docs()
    assert len(lst) == 1
    assert lst[0]["doc"] == "FakeManual"
    assert lst[0]["n_sections"] == 3
    assert lst[0]["built_at"]


def test_search_hits(indexed):
    hits = docs.search("perfectly matched layer")
    assert hits, "应命中 PML 章节"
    assert all(isinstance(h, docs.Hit) for h in hits)
    assert any(h.heading == "Perfectly Matched Layer" for h in hits)
    top = hits[0]
    assert top.doc == "FakeManual"
    assert top.snippet  # snippet 非空
    assert "FakeManual" in top.report()


def test_search_doc_filter_and_limit(indexed):
    hits = docs.search("study", doc="FakeManual", limit=5)
    assert hits and all(h.doc == "FakeManual" for h in hits)
    assert docs.search("study", doc="NoSuchManual") == []


def test_search_no_hits(indexed):
    assert docs.search("zzz nonexistent term zzz") == []


def test_search_bad_syntax_raises(indexed):
    with pytest.raises(ValueError):
        docs.search('"unbalanced')


def test_read_section(indexed):
    body = docs.read("FakeManual", heading="API Reference")
    assert "model.study" in body


def test_read_whole_doc(indexed):
    txt = docs.read("FakeManual")
    assert txt.startswith("<!--")
    assert "Introduction" in txt


def test_read_missing_section_raises(indexed):
    with pytest.raises(KeyError):
        docs.read("FakeManual", heading="No Such Section")


def test_read_missing_doc_raises(indexed):
    with pytest.raises(FileNotFoundError):
        docs.read("NoSuchManual")


def test_build_index_rebuild_idempotent(indexed):
    # 再次 build（rebuild=True）应先删旧条目再插，不重复累积
    assert docs.build_index() == 3
    assert docs.list_docs()[0]["n_sections"] == 3
    assert len(docs.search("perfectly matched layer")) >= 1
