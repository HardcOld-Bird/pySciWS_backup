"""rag（PaperQA2 语义检索基础设施）契约离线快照测试（**绝不触网**）。

锁定阶段 5「RAG 层」的核心契约，全部通过 tmp_path 造文件 + monkeypatch 拦截后端
（paperqa / litellm / _import_backend / _ask_once / search），不发任何真实请求：

- **元数据派生纯函数**：``_clean_inline`` / ``_derive_meta_from_md``（期刊论文得
  「Xia et al. (2025)」；手册/无作者退化为标题；空文件/无 H1 用文件名兜底）；
- **路径与 glob**：``_docname_for``（相对数据区、分隔符折 ``__``）/ ``_candidate_md_files``
  （默认扫 extracted、显式路径、目录递归、去重）；
- **索引持久化**：``_save_meta``/``_load_meta`` 往返、``index_status``（空/已建）；
- **配置形状**：``_router_cfg``（LiteLLM model_list）、``_import_backend`` 未就绪即抛；
- **build_index**：全量重建 / 增量跳过未变 / 变更标 stale / 空目录报错 / 落盘 pickle+meta；
- **search**：Text→RagChunk 映射（rank/docname/citation/source_path）、无索引抛错；
- **ask 三态**：免费成功 / 免费失败回退付费 / 全失败静默降级为 search（永不抛）；
- **渲染**：``render_search`` / ``render_ask``（含降级态）纯字符串输出；
- **数据结构**：``to_dict`` JSON 可序列化。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pysci.skills.literature_research.tools import rag

# ---------------------------------------------------------------------------
# 假后端（paperqa / Docs / Text / Doc）——形状对齐真实 paperqa，但绝不触网
# ---------------------------------------------------------------------------
FAKE_PQA_VERSION = "2026.8.12"


class _FakeDoc:
    def __init__(self, docname: str, citation: str) -> None:
        self.docname = docname
        self.citation = citation
        self.dockey = docname


class _FakeText:
    def __init__(self, text: str, name: str, doc: _FakeDoc) -> None:
        self.text = text
        self.name = name
        self.doc = doc


class _FakeDocs:
    def __init__(self) -> None:
        self.docs: dict[str, _FakeDoc] = {}
        self.texts: list[_FakeText] = []
        self.added: list[dict] = []

    async def aadd(
        self,
        path,
        docname=None,
        citation=None,
        title=None,
        doi=None,
        settings=None,
        **kw,
    ):
        self.added.append(
            {
                "path": str(path),
                "docname": docname,
                "citation": citation,
                "title": title,
                "doi": doi,
            }
        )
        d = _FakeDoc(docname, citation or "")
        self.docs[docname] = d
        self.texts.append(
            _FakeText(f"chunk body of {docname}", f"{docname} lines 0-10", d)
        )
        return docname

    async def retrieve_texts(self, query, k, settings=None, **kw):
        return self.texts[:k]


class _FakePaperQA:
    __version__ = FAKE_PQA_VERSION
    Docs = _FakeDocs

    @staticmethod
    def Settings(**kw):
        return SimpleNamespace(**kw)


def _fake_settings(tmp_path, *, ready: bool = True) -> SimpleNamespace:
    """替换 rag.settings 的最小替身（frozen 真 Settings 不可 setattr，故整体换掉）。"""
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "rag"
    home.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=tmp_path,
        cache_extracted=extracted,
        pqa_home=home,
        siliconflow_api_key="sk-test-abcdef" if ready else None,
        siliconflow_base_url="https://api.siliconflow.cn/v1",
        pqa_embedding="openai/BAAI/bge-m3",
        pqa_llm="openai/Qwen/Qwen2.5-7B-Instruct",
        pqa_llm_fallback="openai/Qwen/Qwen2.5-32B-Instruct",
        openalex_email="tester@example.com",
        pqa_ready=ready,
    )


def _write_md(dirpath, rel: str, body: str):
    p = dirpath / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


PRL_MD = """# Observation of Coherent Perfect Acoustic Absorption at an Exceptional Point

Yi-Fei Xia, $^{1,*}$ Zi-Xiang Xu, $^{1,*}$ and Johan Christensen $^{2}$

$^{1}$ Institute of Acoustics, Nanjing University

(Received 29 November 2024; published 6 August 2025)

We demonstrate CPA EP in a two-channel acoustic waveguide.

DOI: 10.1103/slhy-f76q
"""

MANUAL_MD = """# Acoustics Module User's Guide

www.comsol.com  Acoustics Module, COMSOL AB.

This guide describes the acoustics interfaces. No publication date line here.
"""


# ===========================================================================
#  纯函数：_clean_inline / _derive_meta_from_md
# ===========================================================================
def test_clean_inline_strips_latex_and_markers():
    assert (
        rag._clean_inline("Yi-Fei Xia, $^{1,*}$ Zi-Xiang Xu")
        == "Yi-Fei Xia, Zi-Xiang Xu"
    )
    assert rag._clean_inline("A $\\alpha$ B") == "A B"
    assert rag._clean_inline("  spaced   out ") == "spaced out"


def test_derive_meta_prl_paper(tmp_path):
    p = _write_md(tmp_path, "cpa_ep/paper_prl.md", PRL_MD)
    d = rag._derive_meta_from_md(p)
    assert d["title"].startswith("Observation of Coherent Perfect Acoustic")
    assert d["year"] == 2025  # 取 published 年，而非 received 的 2024
    assert d["doi"] == "10.1103/slhy-f76q"
    assert d["first_author"] == "Xia"
    assert d["citation"] == "Xia et al. (2025)"


def test_derive_meta_manual_degrades_to_title(tmp_path):
    """手册：作者行含 www/.com 应被拒；无 published/received → 无年份 → 引文=标题。"""
    p = _write_md(tmp_path, "manual.md", MANUAL_MD)
    d = rag._derive_meta_from_md(p)
    assert d["title"] == "Acoustics Module User's Guide"
    assert d["first_author"] == ""  # 不误抓 www.comsol.com
    assert d["citation"] == "Acoustics Module User's Guide"


def test_derive_meta_and_separated_authors(tmp_path):
    body = "# Some Title\n\nJohn Smith and Jane Doe\n\n(published 1 January 2021)\n"
    d = rag._derive_meta_from_md(_write_md(tmp_path, "x.md", body))
    assert d["first_author"] == "Smith"
    assert d["year"] == 2021
    assert d["citation"] == "Smith et al. (2021)"


def test_derive_meta_copyright_year(tmp_path):
    body = "# Report Title\n\nSome body without published line.\n\n© 2020 Acme Corp\n"
    d = rag._derive_meta_from_md(_write_md(tmp_path, "r.md", body))
    assert d["year"] == 2020
    assert d["citation"] == "Report Title (2020)"  # 有年无姓 → 标题(年)


def test_derive_meta_affiliation_line_skipped(tmp_path):
    """单位行（含 University）不应被当作者行。"""
    body = "# Title Here\n\nDepartment of Physics, Nanjing University\n\nLi Wang, Mei Zhang\n\n(published 5 May 2019)\n"
    d = rag._derive_meta_from_md(_write_md(tmp_path, "a.md", body))
    assert d["first_author"] == "Wang"  # 跳过单位行，取到 "Li Wang"


def test_derive_meta_no_h1_uses_stem(tmp_path):
    body = "Just plain text without a markdown heading.\n"
    d = rag._derive_meta_from_md(_write_md(tmp_path, "my_file_name.md", body))
    assert d["title"] == "my file name"  # stem 下划线转空格
    assert d["citation"] == "my file name"


def test_derive_meta_empty_file(tmp_path):
    p = _write_md(tmp_path, "empty_doc.md", "")
    d = rag._derive_meta_from_md(p)
    assert d["title"] == "empty doc"
    assert d["year"] is None
    assert d["citation"] == "empty doc"


def test_derive_meta_missing_file_no_raise(tmp_path):
    d = rag._derive_meta_from_md(tmp_path / "does_not_exist.md")
    assert d["citation"] == "does not exist"  # 兜底 stem，不抛


def test_derive_meta_year_only_from_keywords(tmp_path):
    """正文里出现的无关四位数不应被当年份（只认 published/accepted/received/©）。"""
    body = "# T\n\nThe sample was 1234 units and reference 1999 was cited.\n"
    d = rag._derive_meta_from_md(_write_md(tmp_path, "n.md", body))
    assert d["year"] is None


# ===========================================================================
#  路径 / glob / docname
# ===========================================================================
def test_docname_for_relative_to_extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    p = _write_md(rag.settings.cache_extracted, "cpa_ep/paper.md", "# T\n")
    assert rag._docname_for(p) == "cpa_ep__paper"


def test_docname_for_toplevel_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    p = _write_md(rag.settings.cache_extracted, "solo.md", "# T\n")
    assert rag._docname_for(p) == "solo"


def test_candidate_md_default_scans_extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    ex = rag.settings.cache_extracted
    _write_md(ex, "a/one.md", "# 1\n")
    _write_md(ex, "b/two.md", "# 2\n")
    _write_md(ex, "b/notmd.txt", "x")
    files = rag._candidate_md_files(None)
    names = sorted(f.name for f in files)
    assert names == ["one.md", "two.md"]  # 递归、仅 .md


def test_candidate_md_explicit_paths_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    ex = rag.settings.cache_extracted
    p1 = _write_md(ex, "a/one.md", "# 1\n")
    _write_md(ex, "b/two.md", "# 2\n")
    # 显式：一个文件 + 一个目录 + 重复同一文件 → 去重
    files = rag._candidate_md_files([str(p1), str(ex / "b"), str(p1)])
    names = sorted(f.name for f in files)
    assert names == ["one.md", "two.md"]


# ===========================================================================
#  索引持久化 / 状态
# ===========================================================================
def test_meta_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    assert rag._load_meta() == {}  # 无文件 → 空 dict
    rag._save_meta({"n_docs": 3, "files": {"x": {"path": "/x"}}})
    m = rag._load_meta()
    assert m["n_docs"] == 3
    assert m["files"]["x"]["path"] == "/x"


def test_meta_corrupt_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    (rag.settings.pqa_home / rag.META_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    assert rag._load_meta() == {}


def test_index_status_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    st = rag.index_status()
    assert st["ready"] is True
    assert st["exists"] is False
    assert st["n_docs"] == 0


def test_index_status_after_save(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    (rag.settings.pqa_home / rag.INDEX_FILENAME).write_bytes(b"fake pickle")
    rag._save_meta(
        {"n_docs": 5, "n_chunks": 40, "embedding_model": "openai/BAAI/bge-m3"}
    )
    st = rag.index_status()
    assert st["exists"] is True
    assert st["n_docs"] == 5 and st["n_chunks"] == 40


def test_router_cfg_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    cfg = rag._router_cfg("openai/BAAI/bge-m3")
    ml = cfg["model_list"][0]
    assert ml["model_name"] == "openai/BAAI/bge-m3"
    assert ml["litellm_params"]["model"] == "openai/BAAI/bge-m3"
    assert ml["litellm_params"]["api_base"] == "https://api.siliconflow.cn/v1"
    assert ml["litellm_params"]["api_key"] == "sk-test-abcdef"


# ===========================================================================
#  后端守卫
# ===========================================================================
def test_import_backend_raises_when_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path, ready=False))
    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        rag._import_backend(models=[("openai/BAAI/bge-m3", "embedding")])


# ===========================================================================
#  build_index
# ===========================================================================
def _patch_backend(monkeypatch):
    monkeypatch.setattr(rag, "_import_backend", lambda *, models: _FakePaperQA())
    monkeypatch.setattr(rag, "_build_pqa_settings", lambda **kw: None)
    monkeypatch.setattr(rag, "_map_mailto_env", lambda: None)


def test_build_index_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    _patch_backend(monkeypatch)
    ex = rag.settings.cache_extracted
    _write_md(ex, "cpa_ep/p1.md", PRL_MD)
    _write_md(
        ex,
        "ep/p2.md",
        "# Another Paper\n\nJane Roe, John Doe\n\n(published 3 March 2020)\n",
    )

    rep = rag.build_index(rebuild=True, verbose=False)
    assert rep.n_added == 2 and rep.n_docs == 2 and rep.n_chunks == 2
    assert rep.errors == [] and rep.rebuilt is True
    # 落盘
    assert (rag.settings.pqa_home / rag.INDEX_FILENAME).exists()
    meta = rag._load_meta()
    assert meta["n_docs"] == 2
    assert meta["paperqa_version"] == FAKE_PQA_VERSION
    assert "cpa_ep__p1" in meta["files"]
    assert meta["files"]["cpa_ep__p1"]["citation"] == "Xia et al. (2025)"


def test_build_index_incremental_skips_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    _patch_backend(monkeypatch)
    ex = rag.settings.cache_extracted
    _write_md(ex, "a/p1.md", PRL_MD)
    rag.build_index(rebuild=True, verbose=False)

    # 第二次（非 rebuild）：同一文件未变 → 跳过；新增一个 → 只加新的
    _write_md(ex, "b/p2.md", "# Second\n\nAnn Author\n\n(published 1 January 2022)\n")
    rep = rag.build_index(rebuild=False, verbose=False)
    assert rep.n_skipped == 1
    assert rep.n_added == 1
    assert rep.n_docs == 2


def test_build_index_stale_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    _patch_backend(monkeypatch)
    ex = rag.settings.cache_extracted
    p1 = _write_md(ex, "a/p1.md", PRL_MD)
    rag.build_index(rebuild=True, verbose=False)

    # 改动内容 + 强制 mtime 变化 → 标 stale（增量不覆盖同名 doc）
    p1.write_text(PRL_MD + "\nExtra line.\n", encoding="utf-8")
    import os

    st = p1.stat()
    os.utime(p1, (st.st_atime + 100, st.st_mtime + 100))
    rep = rag.build_index(rebuild=False, verbose=False)
    assert rep.n_stale == 1
    assert rep.n_added == 0


def test_build_index_empty_dir_reports_error(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    _patch_backend(monkeypatch)
    rep = rag.build_index(rebuild=True, verbose=False)
    assert rep.n_docs == 0
    assert rep.errors and "未找到可索引" in rep.errors[0]


# ===========================================================================
#  search
# ===========================================================================
def _patch_search_backend(monkeypatch, docs, files_map):
    monkeypatch.setattr(rag, "_import_backend", lambda *, models: _FakePaperQA())
    monkeypatch.setattr(rag, "_build_pqa_settings", lambda **kw: None)
    monkeypatch.setattr(
        rag, "_load_index_or_raise", lambda paperqa: (docs, {"files": files_map})
    )


def test_search_maps_texts_to_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    docs = _FakeDocs()
    d1 = _FakeDoc("cpa_ep__p1", "Xia et al. (2025)")
    docs.docs["cpa_ep__p1"] = d1
    docs.texts = [
        _FakeText("first chunk about EP", "cpa_ep__p1 lines 0-10", d1),
        _FakeText("second chunk about CPA", "cpa_ep__p1 lines 11-20", d1),
    ]
    files_map = {"cpa_ep__p1": {"path": "/abs/cpa_ep/p1.md"}}
    _patch_search_backend(monkeypatch, docs, files_map)

    res = rag.search("exceptional point absorption", k=2)
    assert res.query == "exceptional point absorption"
    assert len(res.chunks) == 2
    c0 = res.chunks[0]
    assert c0.rank == 1
    assert c0.text == "first chunk about EP"
    assert c0.docname == "cpa_ep__p1"
    assert c0.citation == "Xia et al. (2025)"
    assert c0.source_path == "/abs/cpa_ep/p1.md"
    assert res.chunks[1].rank == 2


def test_search_k_limits_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    docs = _FakeDocs()
    d = _FakeDoc("doc", "Cite (2020)")
    docs.docs["doc"] = d
    docs.texts = [_FakeText(f"c{i}", f"doc:{i}", d) for i in range(5)]
    _patch_search_backend(monkeypatch, docs, {"doc": {"path": "/p.md"}})
    assert len(rag.search("q", k=3).chunks) == 3


def test_search_no_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    monkeypatch.setattr(rag, "_import_backend", lambda *, models: _FakePaperQA())
    # 不造 index.pkl → _load_index_or_raise 应抛
    with pytest.raises(RuntimeError, match="research rag index"):
        rag.search("q", k=3)


# ===========================================================================
#  ask 三态：免费 / 回退 / 降级
# ===========================================================================
def _patch_ask_common(monkeypatch, tmp_path, search_result=None):
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))
    monkeypatch.setattr(rag, "_import_backend", lambda *, models: _FakePaperQA())
    monkeypatch.setattr(
        rag, "_load_index_or_raise", lambda paperqa: (_FakeDocs(), {"files": {}})
    )
    monkeypatch.setattr(
        rag,
        "search",
        lambda q, k=8: (
            search_result
            or rag.RagSearchResult(
                query=q, chunks=[rag.RagChunk(1, "t", "d", "C (2020)")]
            )
        ),
    )


def test_ask_free_success(tmp_path, monkeypatch):
    _patch_ask_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        rag,
        "_ask_once",
        lambda docs, q, model: ("免费答案", 0.0, ["Xia et al. (2025)"]),
    )
    res = rag.ask("q")
    assert res.backend == "free"
    assert res.model == "openai/Qwen/Qwen2.5-7B-Instruct"
    assert res.answer == "免费答案"
    assert res.citations == ["Xia et al. (2025)"]
    assert res.degraded is False
    assert res.search is None


def test_ask_fallback_on_free_failure(tmp_path, monkeypatch):
    _patch_ask_common(monkeypatch, tmp_path)
    calls = []

    def _fake_ask(docs, q, model):
        calls.append(model)
        if model == "openai/Qwen/Qwen2.5-7B-Instruct":
            raise RuntimeError("free tier 503")
        return ("付费答案", 0.02, ["C (2020)"])

    monkeypatch.setattr(rag, "_ask_once", _fake_ask)
    res = rag.ask("q")
    assert res.backend == "fallback"
    assert res.model == "openai/Qwen/Qwen2.5-32B-Instruct"
    assert res.answer == "付费答案"
    assert res.degraded is False
    assert calls == [
        "openai/Qwen/Qwen2.5-7B-Instruct",
        "openai/Qwen/Qwen2.5-32B-Instruct",
    ]


def test_ask_degrades_when_all_llm_fail(tmp_path, monkeypatch):
    _patch_ask_common(monkeypatch, tmp_path)

    def _always_fail(docs, q, model):
        raise RuntimeError("boom")

    monkeypatch.setattr(rag, "_ask_once", _always_fail)
    res = rag.ask("q", k=4)
    assert res.backend == "none"
    assert res.degraded is True
    assert res.answer is None
    assert res.search is not None  # 降级为检索结果
    assert res.search.chunks  # 且检索有内容
    assert "boom" in (res.error or "")


def test_ask_degrades_when_backend_unavailable(tmp_path, monkeypatch):
    """_import_backend / 索引不可用 → 不抛，降级为 search。"""
    monkeypatch.setattr(rag, "settings", _fake_settings(tmp_path))

    def _raise(*, models):
        raise RuntimeError("no key")

    monkeypatch.setattr(rag, "_import_backend", _raise)
    monkeypatch.setattr(
        rag,
        "search",
        lambda q, k=8: rag.RagSearchResult(
            query=q, chunks=[rag.RagChunk(1, "t", "d", "C")]
        ),
    )
    res = rag.ask("q")
    assert res.degraded is True
    assert res.backend == "none"
    assert res.search is not None


def test_ask_no_fallback_configured(tmp_path, monkeypatch):
    """未配付费回退 → 免费失败后直接降级（不尝试回退）。"""
    st = _fake_settings(tmp_path)
    st.pqa_llm_fallback = None
    monkeypatch.setattr(rag, "settings", st)
    monkeypatch.setattr(rag, "_import_backend", lambda *, models: _FakePaperQA())
    monkeypatch.setattr(
        rag, "_load_index_or_raise", lambda paperqa: (_FakeDocs(), {"files": {}})
    )
    monkeypatch.setattr(
        rag, "search", lambda q, k=8: rag.RagSearchResult(query=q, chunks=[])
    )

    def _fail(docs, q, model):
        raise RuntimeError("free down")

    monkeypatch.setattr(rag, "_ask_once", _fail)
    res = rag.ask("q")
    assert res.backend == "none" and res.degraded is True


# ===========================================================================
#  渲染 + 数据结构
# ===========================================================================
def test_render_search_human():
    res = rag.RagSearchResult(
        query="ep",
        chunks=[
            rag.RagChunk(
                1,
                "some text body",
                "d1",
                "Xia et al. (2025)",
                "/p/p1.md",
                "d1 lines 0-9",
            )
        ],
        n_docs=1,
        n_chunks=1,
        embedding_model="openai/BAAI/bge-m3",
    )
    out = rag.render_search(res)
    assert "Xia et al. (2025)" in out
    assert "/p/p1.md" in out
    assert "some text body" in out


def test_render_search_empty():
    out = rag.render_search(rag.RagSearchResult(query="q"))
    assert "无命中" in out


def test_render_search_truncates():
    long = "x" * 1000
    res = rag.RagSearchResult(query="q", chunks=[rag.RagChunk(1, long, "d", "C")])
    out = rag.render_search(res, chars=100)
    assert "…" in out


def test_render_ask_normal():
    res = rag.RagAskResult(
        query="q",
        answer="综述内容",
        backend="free",
        model="m",
        cost=0.0,
        citations=["A (2020)"],
    )
    out = rag.render_ask(res)
    assert "综述内容" in out
    assert "backend=free" in out
    assert "A (2020)" in out


def test_render_ask_degraded():
    sr = rag.RagSearchResult(
        query="q", chunks=[rag.RagChunk(1, "body", "d", "C (2020)")]
    )
    res = rag.RagAskResult(
        query="q", degraded=True, backend="none", error="llm down", search=sr
    )
    out = rag.render_ask(res)
    assert "降级为检索结果" in out
    assert "body" in out


def test_to_dict_json_serializable():
    sr = rag.RagSearchResult(
        query="q", chunks=[rag.RagChunk(1, "t", "d", "c", "/p", "n")]
    )
    json.dumps(sr.to_dict(), ensure_ascii=False)  # 不抛即通过
    ar = rag.RagAskResult(query="q", answer="a", backend="free", search=sr)
    d = ar.to_dict()
    json.dumps(d, ensure_ascii=False)
    assert d["search"]["query"] == "q"  # 嵌套 dataclass 亦转 dict
    ir = rag.IndexReport(n_added=1, n_docs=1)
    json.dumps(ir.to_dict(), ensure_ascii=False)
