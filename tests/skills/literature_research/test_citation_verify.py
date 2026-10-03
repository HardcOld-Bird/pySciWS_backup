"""citation_verify 三源交叉核验契约离线快照测试（不打真实 API / 不触网）。

锁定阶段 6「引用完整性门」的核心契约，全部通过 monkeypatch 拦截三源 fetcher /
底层 HTTP / 各客户端，绝不触网：

- **归一化纯函数**：``normalize_title`` / ``title_similarity`` / ``normalize_last_name``
  / ``normalize_doi`` / ``extract_year``（跨源比对的公共前提）；
- **比对器严重度**：标题/DOI/首作者冲突=hard；年份相差 1=soft、≥2=hard；期刊冲突=soft；
- **三源解析**：Crossref ``message`` / OpenAlex work-dict / arXiv entry → SourceRecord；
- **裁决逻辑**：全一致=PASS；任一 hard 冲突=FAIL；仅 soft=WARN；三源查无=NOT_FOUND；
  单源不可达只降级不误判；claim（笔记声称值）与源冲突=FAIL（检出错配 DOI / 虚构引用）；
- **优雅降级**：只有 FAIL 阻断（``passed()`` 为 False），WARN/NOT_FOUND 均放行；
- **Crossref HTTP**：缓存命中跳过网络、mailto 进 polite pool、404→未命中、异常→不可达。
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pytest

from pysci.skills.literature_research.tools import citation_verify as cv
from pysci.skills.literature_research.tools import openalex_client as oa
from pysci.skills.literature_research.tools import research

# ---------------------------------------------------------------------------
#  一致基准：PRL 121, 124501 (2018) —— 三源对同一篇的归一化记录
# ---------------------------------------------------------------------------
TITLE = "Simultaneous observation of topological exceptional points"
DOI = "10.1103/physrevlett.121.124501"


def _rec(
    source: str,
    *,
    title: str = "",
    authors: list[str] | None = None,
    first: str = "",
    year: int | None = None,
    journal: str = "",
    doi: str = "",
    arxiv_id: str = "",
    reachable: bool = True,
    found: bool = True,
    error: str = "",
) -> cv.SourceRecord:
    return cv.SourceRecord(
        source=source,
        reachable=reachable,
        found=found,
        title=title,
        authors=authors or [],
        first_author_last_name=first,
        year=year,
        journal=journal,
        doi=doi,
        arxiv_id=arxiv_id,
        error=error,
    )


def _consistent_openalex() -> cv.SourceRecord:
    return _rec(
        "openalex",
        title=TITLE,
        first="zhu",
        year=2018,
        journal="Physical Review Letters",
        doi=DOI,
    )


def _consistent_crossref() -> cv.SourceRecord:
    return _rec(
        "crossref",
        title=TITLE,
        first="zhu",
        year=2018,
        journal="Physical Review Letters",
        doi=DOI,
    )


def _consistent_arxiv() -> cv.SourceRecord:
    return _rec("arxiv", title=TITLE, first="zhu", year=2018, arxiv_id="1803.04110")


CITE = {
    "doi": "10.1103/PhysRevLett.121.124501",
    "title": TITLE,
    "first_author_last_name": "Zhu",
    "year": 2018,
    "journal": "Physical Review Letters",
}


def _patch_fetchers(monkeypatch, *, oa=None, cr=None, ax=None) -> None:
    """把三源 fetcher 替换为返回固定 SourceRecord 的桩（绝不触网）。"""
    if oa is not None:
        monkeypatch.setattr(cv, "fetch_openalex", lambda **kw: oa)
    if cr is not None:
        monkeypatch.setattr(cv, "fetch_crossref", lambda **kw: cr)
    if ax is not None:
        monkeypatch.setattr(cv, "fetch_arxiv", lambda **kw: ax)


# ===========================================================================
#  归一化纯函数
# ===========================================================================
def test_normalize_title_lowercases_and_strips_punct():
    assert cv.normalize_title("Simultaneous Observation!") == "simultaneous observation"
    assert cv.normalize_title("  Gain   &  Loss ") == "gain loss"
    assert cv.normalize_title("") == ""
    assert cv.normalize_title(None) == ""


def test_normalize_title_strips_accents():
    assert cv.normalize_title("Café Résumé") == "cafe resume"


def test_title_similarity():
    assert cv.title_similarity(TITLE, TITLE) == pytest.approx(1.0)
    assert cv.title_similarity("", "x") == 0.0
    assert (
        cv.title_similarity("acoustic exceptional point", "acoustic exceptional points")
        > 0.9
    )
    assert (
        cv.title_similarity("unicorns and rainbows", "topological exceptional point")
        < 0.4
    )


def test_normalize_last_name_forms():
    assert cv.normalize_last_name("Zhu, Zheng") == "zhu"  # 逗号前为姓
    assert cv.normalize_last_name("Zheng Zhu") == "zhu"  # 末词为姓
    assert cv.normalize_last_name("Zheng") == "zheng"  # 单词名
    assert cv.normalize_last_name("Ludwig van Beethoven") == "beethoven"
    assert cv.normalize_last_name("José García") == "garcia"  # 去声调
    assert cv.normalize_last_name("") == ""
    assert cv.normalize_last_name(None) == ""


def test_normalize_doi():
    assert cv.normalize_doi("https://doi.org/10.1103/PhysRevLett.121.124501") == DOI
    assert cv.normalize_doi("http://dx.doi.org/10.1/X") == "10.1/x"
    assert cv.normalize_doi("doi:10.1/X") == "10.1/x"
    assert cv.normalize_doi("  10.1/X  ") == "10.1/x"
    assert cv.normalize_doi("") == ""
    assert cv.normalize_doi(None) == ""


def test_extract_year_variants():
    assert cv.extract_year(2018) == 2018
    assert cv.extract_year("2018") == 2018
    assert cv.extract_year("2018-09-21") == 2018
    assert cv.extract_year([[2018, 9, 21]]) == 2018  # Crossref date-parts
    assert cv.extract_year([2018, 9]) == 2018
    assert cv.extract_year("") is None
    assert cv.extract_year(None) is None
    assert cv.extract_year("no year here") is None
    assert cv.extract_year(True) is None  # bool 不当年份
    assert cv.extract_year(42) is None  # 非四位年


# ===========================================================================
#  比对器严重度
# ===========================================================================
def test_cmp_title():
    assert cv._cmp_title(cv.normalize_title(TITLE), cv.normalize_title(TITLE)) is None
    assert cv._cmp_title("acoustic wave", "completely unrelated subject") == "hard"
    assert cv._cmp_title("", "anything") is None  # 空值不判冲突


def test_cmp_exact():
    assert cv._cmp_exact(DOI, DOI) is None
    assert cv._cmp_exact(DOI, "10.9999/other") == "hard"
    assert cv._cmp_exact("", "x") is None


def test_cmp_year_severity():
    assert cv._cmp_year(2018, 2018) is None
    assert cv._cmp_year(2018, 2019) == "soft"  # 相差 1（online-first vs 见刊）
    assert cv._cmp_year(2018, 2020) == "hard"  # 相差 ≥2
    assert cv._cmp_year(None, 2018) is None


def test_cmp_name_is_soft():
    assert cv._cmp_name("zhu", "zhu") is None
    assert cv._cmp_name("zhu", "wang") == "soft"  # 作者姓冲突只记 soft
    assert cv._cmp_name("", "zhu") is None


def test_cmp_journal_soft_and_containment():
    assert cv._cmp_journal("physical review letters", "physical review letters") is None
    assert (
        cv._cmp_journal("physical review", "physical review letters") is None
    )  # 包含 → 不判冲突
    assert cv._cmp_journal("physical review letters", "nature communications") == "soft"
    assert cv._cmp_journal("", "nature") is None


def test_field_check_worst_priority():
    chk = cv.FieldCheck(field="x", conflicts=[("a", "b", "soft"), ("c", "d", "hard")])
    assert chk.worst == "hard"
    assert cv.FieldCheck(field="x", conflicts=[("a", "b", "soft")]).worst == "soft"
    assert cv.FieldCheck(field="x", conflicts=[]).worst is None


# ===========================================================================
#  三源解析 → SourceRecord
# ===========================================================================
CROSSREF_MSG = {
    "DOI": "10.1103/PhysRevLett.121.124501",
    "type": "journal-article",
    "title": [TITLE],
    "container-title": ["Physical Review Letters"],
    "publisher": "American Physical Society (APS)",
    "volume": "121",
    "issue": "12",
    "page": "124501",
    "issued": {"date-parts": [[2018, 9, 21]]},
    "author": [
        {"given": "Zheng", "family": "Zhu"},
        {"given": "Xiangang", "family": "Wan"},
    ],
}


def test_record_from_crossref():
    r = cv._record_from_crossref(CROSSREF_MSG)
    assert r.source == "crossref" and r.reachable and r.found
    assert r.title == TITLE
    assert r.journal == "Physical Review Letters"
    assert r.doi == DOI  # 归一化小写
    assert r.year == 2018
    assert r.authors == ["Zheng Zhu", "Xiangang Wan"]
    assert r.first_author_last_name == "zhu"


def test_record_from_crossref_missing_fields_safe():
    r = cv._record_from_crossref({"DOI": "10.1/x"})
    assert r.title == "" and r.journal == "" and r.year is None
    assert r.authors == [] and r.first_author_last_name == ""


OPENALEX_WORK = {
    "title": TITLE,
    "publication_year": 2018,
    "doi": "10.1103/PhysRevLett.121.124501",
    "journal": "Physical Review Letters",
    "arxiv_id": "1803.04110",
    "first_author_last_name": "zhu",
    "authors": [{"name": "Zheng Zhu"}, {"name": "Xiangang Wan"}],
}


def test_record_from_openalex():
    r = cv._record_from_openalex(OPENALEX_WORK)
    assert r.source == "openalex" and r.found
    assert r.title == TITLE and r.year == 2018
    assert r.doi == DOI and r.arxiv_id == "1803.04110"
    assert r.authors == ["Zheng Zhu", "Xiangang Wan"]
    assert r.first_author_last_name == "zhu"


ARXIV_ENTRY = {
    "title": TITLE,
    "published": "2018-03-12T00:00:00Z",
    "doi": "",
    "journal_ref": "",
    "arxiv_id": "1803.04110",
    "first_author_last_name": "zhu",
    "authors": [{"name": "Zheng Zhu"}],
}


def test_record_from_arxiv():
    r = cv._record_from_arxiv(ARXIV_ENTRY)
    assert r.source == "arxiv" and r.found
    assert r.title == TITLE and r.year == 2018 and r.arxiv_id == "1803.04110"
    assert r.first_author_last_name == "zhu"


def test_openalex_and_crossref_transliterate_author_consistently():
    """回归护栏（阶段6 实测修复）：首作者姓均从完整名转写声调（Büttner→buttner），
    而非用 openalex 上游预处理的 'bttner'（删声调）——避免两源假冲突。"""
    w = {
        "title": TITLE,
        "publication_year": 2016,
        "doi": "10.1038/nphys1914",
        "journal": "Nature Physics",
        "first_author_last_name": "bttner",  # 上游 openalex 删声调的值（不可信）
        "authors": [{"name": "Henning Büttner"}],
    }
    r_oa = cv._record_from_openalex(w)
    r_cr = cv._record_from_crossref(
        {
            "DOI": "10.1038/nphys1914",
            "title": [TITLE],
            "container-title": ["Nature Physics"],
            "issued": {"date-parts": [[2016]]},
            "author": [{"given": "Henning", "family": "Büttner"}],
        }
    )
    assert r_oa.first_author_last_name == "buttner"  # 从完整名转写，而非 'bttner'
    assert r_cr.first_author_last_name == "buttner"
    assert r_oa.first_author_last_name == r_cr.first_author_last_name


# ===========================================================================
#  fetch_openalex / fetch_arxiv（monkeypatch 底层客户端）
# ===========================================================================
def test_fetch_openalex_by_doi(monkeypatch):
    monkeypatch.setattr(oa, "get_work", lambda **kw: OPENALEX_WORK)
    r = cv.fetch_openalex(doi="10.1103/PhysRevLett.121.124501")
    assert r.found and r.doi == DOI and r.year == 2018


def test_fetch_openalex_not_found(monkeypatch):
    monkeypatch.setattr(oa, "get_work", lambda **kw: None)
    r = cv.fetch_openalex(doi="10.1/x")
    assert r.reachable is True and r.found is False


def test_fetch_openalex_network_error_is_unreachable(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(oa, "get_work", _boom)
    r = cv.fetch_openalex(doi="10.1/x")
    assert r.reachable is False and r.found is False
    assert "RuntimeError" in r.error


def test_fetch_openalex_by_title_picks_best(monkeypatch):
    monkeypatch.setattr(
        oa,
        "search_works",
        lambda **kw: {
            "results": [
                {"title": "unrelated noise", "publication_year": 2001},
                OPENALEX_WORK,
            ]
        },
    )
    r = cv.fetch_openalex(title=TITLE)
    assert r.found and r.year == 2018  # 选中标题最相似者


def test_fetch_arxiv_by_id(monkeypatch):
    from pysci.skills.literature_research.tools import arxiv_client

    monkeypatch.setattr(arxiv_client, "get_paper", lambda *a, **k: ARXIV_ENTRY)
    r = cv.fetch_arxiv(arxiv_id="1803.04110")
    assert r.found and r.arxiv_id == "1803.04110" and r.year == 2018


def test_fetch_arxiv_not_found(monkeypatch):
    from pysci.skills.literature_research.tools import arxiv_client

    monkeypatch.setattr(arxiv_client, "get_paper", lambda *a, **k: None)
    r = cv.fetch_arxiv(arxiv_id="9999.99999")
    assert r.reachable is True and r.found is False


def test_fetch_arxiv_error_is_unreachable(monkeypatch):
    from pysci.skills.literature_research.tools import arxiv_client

    def _boom(*a, **k):
        raise ValueError("bad id")

    monkeypatch.setattr(arxiv_client, "get_paper", _boom)
    r = cv.fetch_arxiv(arxiv_id="x")
    assert r.reachable is False and "ValueError" in r.error


# ===========================================================================
#  Crossref HTTP 层：缓存 / mailto / 404 / 异常
# ===========================================================================
class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, resp, captured):
        self._resp = resp
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, timeout=None):
        self._captured.update(url=url, params=dict(params or {}), timeout=timeout)
        return self._resp


def _patch_settings(monkeypatch, **kw):
    monkeypatch.setattr(cv, "settings", dataclasses.replace(cv.settings, **kw))


def test_crossref_cache_path_prefixed():
    p = cv._crossref_cache_path("https://api.crossref.org/works/10.1/x", {})
    assert p.name.startswith("crossref_") and p.name.endswith(".json")


def test_crossref_message_cache_hit_skips_http(monkeypatch):
    monkeypatch.setattr(
        oa, "_read_cache", lambda path, **kw: {"message": {"DOI": "10.1/x"}}
    )

    def _no_http(*a, **k):
        raise AssertionError("cache hit must not touch HTTP")

    monkeypatch.setattr(cv, "http_session", _no_http)
    msg = cv._crossref_message("works/10.1/x")
    assert msg == {"DOI": "10.1/x"}


def test_crossref_message_http_success_writes_cache(monkeypatch):
    captured: dict = {}
    written: dict = {}
    monkeypatch.setattr(oa, "_read_cache", lambda path, **kw: None)
    monkeypatch.setattr(
        oa, "_write_cache", lambda path, data: written.update(data=data)
    )
    _patch_settings(monkeypatch, openalex_email="me@example.org", http_timeout=7)
    payload = {"status": "ok", "message": {"DOI": "10.1/x", "title": ["T"]}}
    monkeypatch.setattr(
        cv, "http_session", lambda *a, **k: _FakeSession(_FakeResp(payload), captured)
    )
    msg = cv._crossref_message("works/10.1/x")
    assert msg == {"DOI": "10.1/x", "title": ["T"]}
    assert captured["params"]["mailto"] == "me@example.org"  # polite pool
    assert captured["timeout"] == 7
    assert written["data"] == payload  # 全量落缓存


def test_crossref_message_no_mailto_when_unset(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(oa, "_read_cache", lambda path, **kw: None)
    monkeypatch.setattr(oa, "_write_cache", lambda path, data: None)
    _patch_settings(monkeypatch, openalex_email=None)
    monkeypatch.setattr(
        cv,
        "http_session",
        lambda *a, **k: _FakeSession(_FakeResp({"message": {}}), captured),
    )
    cv._crossref_message("works/10.1/x")
    assert "mailto" not in captured["params"]


def test_fetch_crossref_by_doi(monkeypatch):
    monkeypatch.setattr(cv, "_crossref_message", lambda *a, **k: CROSSREF_MSG)
    r = cv.fetch_crossref(doi="10.1103/PhysRevLett.121.124501")
    assert r.found and r.doi == DOI and r.year == 2018


def test_crossref_message_404_returns_none(monkeypatch):
    """真实 Crossref 对未知 DOI 返回 404 → _crossref_message 归为 None（查无），不抛异常。"""
    monkeypatch.setattr(oa, "_read_cache", lambda path, **kw: None)
    wrote = {"called": False}
    monkeypatch.setattr(
        oa, "_write_cache", lambda path, data: wrote.__setitem__("called", True)
    )
    monkeypatch.setattr(
        cv,
        "http_session",
        lambda *a, **k: _FakeSession(_FakeResp(None, status=404), {}),
    )
    assert cv._crossref_message("works/10.1/unknown") is None
    assert wrote["called"] is False  # 404 不落缓存


def test_fetch_crossref_404_via_http_is_not_found(monkeypatch):
    """端到端：Crossref HTTP 404 → reachable=True, found=False（非「不可达」）。"""
    monkeypatch.setattr(oa, "_read_cache", lambda path, **kw: None)
    monkeypatch.setattr(oa, "_write_cache", lambda path, data: None)
    monkeypatch.setattr(
        cv,
        "http_session",
        lambda *a, **k: _FakeSession(_FakeResp(None, status=404), {}),
    )
    r = cv.fetch_crossref(doi="10.1/unknown")
    assert r.reachable is True and r.found is False


def test_fetch_crossref_doi_404_is_not_found(monkeypatch):
    monkeypatch.setattr(cv, "_crossref_message", lambda *a, **k: None)
    r = cv.fetch_crossref(doi="10.1/unknown")
    assert r.reachable is True and r.found is False


def test_fetch_crossref_http_error_is_unreachable(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(cv, "_crossref_message", _boom)
    r = cv.fetch_crossref(doi="10.1/x")
    assert r.reachable is False and "RuntimeError" in r.error


def test_fetch_crossref_by_title_bibliographic(monkeypatch):
    captured: dict = {}

    def _fake(path, params=None, **kw):
        captured.update(path=path, params=dict(params or {}))
        return {"items": [CROSSREF_MSG]}

    monkeypatch.setattr(cv, "_crossref_message", _fake)
    r = cv.fetch_crossref(title=TITLE, first_author="Zhu")
    assert r.found and r.year == 2018
    assert captured["path"] == "works"
    assert "query.bibliographic" in captured["params"]
    assert captured["params"]["rows"] == 1
    assert TITLE in captured["params"]["query.bibliographic"]


def test_fetch_crossref_no_identifier_is_not_found():
    r = cv.fetch_crossref()
    assert r.reachable is True and r.found is False


# ===========================================================================
#  verify_citation —— 裁决逻辑（三源桩，离线）
# ===========================================================================
def test_verify_all_agree_pass(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.PASS and v.passed() is True
    assert v.kind == "doi"


def test_verify_title_mismatch_fail(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec(
            "crossref",
            title="Totally unrelated paper about chess",
            first="zhu",
            year=2018,
            doi=DOI,
        ),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.FAIL and v.passed() is False
    assert any("title" in r for r in v.reasons)


def test_verify_doi_mismatch_fail(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec("crossref", title=TITLE, first="zhu", year=2018, doi="10.9999/wrong"),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.FAIL
    assert any("doi" in r for r in v.reasons)


def test_verify_first_author_mismatch_warn(monkeypatch):
    """首作者姓跨库差异只记 WARN（soft），不硬阻断——题名/DOI/年份才是硬信号。"""
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_rec("arxiv", title=TITLE, first="wang", year=2018, arxiv_id="1803.04110"),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.WARN and v.passed() is True
    assert any("first_author" in r for r in v.reasons)


def test_verify_year_off_by_one_warn(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec(
            "crossref",
            title=TITLE,
            first="zhu",
            year=2019,
            journal="Physical Review Letters",
            doi=DOI,
        ),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.WARN and v.passed() is True


def test_verify_year_off_by_two_fail(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec(
            "crossref",
            title=TITLE,
            first="zhu",
            year=2020,
            journal="Physical Review Letters",
            doi=DOI,
        ),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.FAIL


def test_verify_journal_only_mismatch_warn(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec(
            "crossref",
            title=TITLE,
            first="zhu",
            year=2018,
            journal="Nature Communications",
            doi=DOI,
        ),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.WARN and v.passed() is True


def test_verify_claim_vs_source_mismatch_fail(monkeypatch):
    """笔记声称标题与三源解析出的真实标题不符 → 检出错配 DOI / 虚构引用。"""
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    bad = dict(CITE, title="A completely different and wrong claimed title")
    v = cv.verify_citation(bad)
    assert v.status == cv.FAIL
    assert any("title" in r for r in v.reasons)


def test_verify_one_source_unreachable_not_fail(monkeypatch):
    """某源网络不可达只降级，绝不误判为 FAIL。"""
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec("crossref", reachable=False, found=False, error="Timeout"),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.PASS and v.passed() is True
    assert any("不可达" in r for r in v.reasons)


def test_verify_none_found_is_not_found(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_rec("openalex", found=False),
        cr=_rec("crossref", found=False),
        ax=_rec("arxiv", found=False),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.NOT_FOUND and v.passed() is True  # 可疑但不阻断
    assert any("查无此条" in r for r in v.reasons)


def test_verify_all_unreachable_is_not_found(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_rec("openalex", reachable=False, found=False, error="E"),
        cr=_rec("crossref", reachable=False, found=False, error="E"),
        ax=_rec("arxiv", reachable=False, found=False, error="E"),
    )
    v = cv.verify_citation(CITE)
    assert v.status == cv.NOT_FOUND
    assert any("不可达" in r for r in v.reasons)


def test_verify_empty_citation(monkeypatch):
    _patch_fetchers(monkeypatch, oa=_consistent_openalex())
    v = cv.verify_citation({})
    assert v.status == cv.NOT_FOUND and v.kind == "empty"
    assert v.passed() is True


def test_verify_doi_only_skips_arxiv_fetch(monkeypatch):
    """无标题/无 arXiv id 时不打 arXiv（省一次请求），仅 OpenAlex×Crossref 互证。"""
    _patch_fetchers(monkeypatch, oa=_consistent_openalex(), cr=_consistent_crossref())

    def _no_arxiv(**kw):
        raise AssertionError("arXiv must not be fetched for a DOI-only citation")

    monkeypatch.setattr(cv, "fetch_arxiv", _no_arxiv)
    v = cv.verify_citation({"doi": "10.1103/PhysRevLett.121.124501"})
    assert v.status == cv.PASS
    assert v.records["arxiv"].found is False


def test_verify_respects_sources_subset(monkeypatch):
    _patch_fetchers(monkeypatch, oa=_consistent_openalex(), cr=_consistent_crossref())
    v = cv.verify_citation(CITE, sources=("openalex", "crossref"))
    assert "arxiv" not in v.records
    assert v.status == cv.PASS


# ===========================================================================
#  _classify_citation / citation_from_frontmatter / verdict 属性
# ===========================================================================
def test_classify_citation_priority():
    assert cv._classify_citation({"doi": "10.1/x", "arxiv_id": "2301.1"}) == "doi"
    assert cv._classify_citation({"arxiv_id": "2301.00001"}) == "arxiv"
    assert cv._classify_citation({"openalex_id": "W1"}) == "openalex"
    assert cv._classify_citation({"title": "T"}) == "title"
    assert cv._classify_citation({}) == "empty"


def test_citation_from_frontmatter():
    fm = {
        "doi": "10.1/X",
        "arxiv_id": "2301.00001",
        "openalex_id": "W1",
        "title": "T",
        "first_author_last_name": "Zhu",
        "authors": ["Zheng Zhu"],
        "year": 2018,
        "journal": "PRL",
        "zotero_key": "IGNORED",
    }
    cite = cv.citation_from_frontmatter(fm)
    assert set(cite) == {
        "doi",
        "arxiv_id",
        "openalex_id",
        "title",
        "first_author_last_name",
        "authors",
        "year",
        "journal",
    }
    assert cite["doi"] == "10.1/X" and cite["year"] == 2018


def test_verdict_label_priority():
    v = cv.CitationVerdict(input={"doi": "10.1/x", "title": "T"}, kind="doi")
    assert v.label == "10.1/x"
    v2 = cv.CitationVerdict(input={"title": "T" * 80}, kind="title")
    assert len(v2.label) == 60  # 标题截断
    assert cv.CitationVerdict(input={}, kind="empty").label == "(空引用)"


def test_verdict_passed_only_fail_blocks():
    for status, expected in [
        (cv.PASS, True),
        (cv.WARN, True),
        (cv.NOT_FOUND, True),
        (cv.FAIL, False),
    ]:
        assert (
            cv.CitationVerdict(input={}, kind="doi", status=status).passed() is expected
        )


# ===========================================================================
#  verify_frontmatter / verify_note_file / _parse_frontmatter_scalars
# ===========================================================================
def test_verify_frontmatter(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_frontmatter(CITE)
    assert v.status == cv.PASS


NOTE_MD = f"""---
title: {TITLE}
short_title: Simultaneous observation topological
doi: 10.1103/PhysRevLett.121.124501
year: 2018
first_author_last_name: Zhu
journal: Physical Review Letters
authors:
  - Zheng Zhu
  - Xiangang Wan
status: unread
---

# body
"""


def test_parse_frontmatter_scalars():
    fm = cv._parse_frontmatter_scalars(NOTE_MD)
    assert fm["title"] == TITLE
    assert fm["doi"] == "10.1103/PhysRevLett.121.124501"
    assert fm["year"] == 2018  # 纯数字 → int
    assert fm["first_author_last_name"] == "Zhu"
    # 列表项（以 - 开头）被跳过，不进标量结果
    assert "  - Zheng Zhu" not in fm


def test_parse_frontmatter_scalars_no_frontmatter():
    assert cv._parse_frontmatter_scalars("just body text") == {}


def test_verify_note_file(monkeypatch, tmp_path):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    p = tmp_path / "2018_zhu_note.md"
    p.write_text(NOTE_MD, encoding="utf-8")
    v = cv.verify_note_file(p)
    assert v.status == cv.PASS and v.kind == "doi"


def test_verify_note_file_detects_bad_doi(monkeypatch, tmp_path):
    """笔记 DOI 与标题错配（标题声称 A，DOI 解析出 B）→ FAIL。"""
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    bad = NOTE_MD.replace(
        f"title: {TITLE}", "title: Wrong title that does not match at all"
    )
    p = tmp_path / "bad.md"
    p.write_text(bad, encoding="utf-8")
    v = cv.verify_note_file(p)
    assert v.status == cv.FAIL


# ===========================================================================
#  渲染
# ===========================================================================
def _pass_verdict(monkeypatch) -> cv.CitationVerdict:
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_consistent_crossref(),
        ax=_consistent_arxiv(),
    )
    return cv.verify_citation(CITE)


def test_render_verdict_pass(monkeypatch):
    txt = cv.render_verdict(_pass_verdict(monkeypatch))
    assert "[✓ PASS]" in txt
    assert "openalex=命中" in txt and "crossref=命中" in txt and "arxiv=命中" in txt


def test_render_verdict_fail_marks_and_color(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec(
            "crossref", title="unrelated chess paper", first="zhu", year=2018, doi=DOI
        ),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    plain = cv.render_verdict(v)
    assert "[✗ FAIL]" in plain
    colored = cv.render_verdict(v, color=True)
    assert "\033[31m" in colored  # FAIL 标红


def test_render_verdict_shows_unreachable(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec("crossref", reachable=False, found=False, error="Timeout"),
        ax=_consistent_arxiv(),
    )
    txt = cv.render_verdict(cv.verify_citation(CITE))
    assert "crossref=不可达" in txt


def test_render_report_summary(monkeypatch):
    v1 = _pass_verdict(monkeypatch)
    txt = cv.render_report([v1, v1])
    assert "汇总：共 2 条" in txt
    assert "✓PASS 2" in txt


def test_render_report_empty():
    assert cv.render_report([]) == "（无引用可核验）"


def test_render_report_flags_fail(monkeypatch):
    _patch_fetchers(
        monkeypatch,
        oa=_consistent_openalex(),
        cr=_rec("crossref", title="unrelated", first="zhu", year=2018, doi=DOI),
        ax=_consistent_arxiv(),
    )
    v = cv.verify_citation(CITE)
    txt = cv.render_report([v])
    assert "✗FAIL 1" in txt
    assert "存在 FAIL" in txt


# ===========================================================================
#  research.py 集成：写入前 gate + citecheck 子命令（离线，monkeypatch 核验器）
# ===========================================================================
def _fake_verdict(status: str, reasons: list[str] | None = None) -> cv.CitationVerdict:
    return cv.CitationVerdict(
        input={"doi": "10.1/x"}, kind="doi", status=status, reasons=reasons or []
    )


def test_citation_gate_blocks_on_fail(monkeypatch):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_frontmatter",
        lambda fm, **kw: _fake_verdict(cv.FAIL, ["title 冲突"]),
    )
    assert research._citation_gate({"doi": "10.1/x"}) is False


def test_citation_gate_passes_on_pass(monkeypatch):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_frontmatter",
        lambda fm, **kw: _fake_verdict(cv.PASS),
    )
    assert research._citation_gate({"doi": "10.1/x"}) is True


def test_citation_gate_warn_not_blocking(monkeypatch):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_frontmatter",
        lambda fm, **kw: _fake_verdict(cv.WARN),
    )
    assert research._citation_gate({"doi": "10.1/x"}) is True


def test_citation_gate_force_overrides_fail(monkeypatch):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_frontmatter",
        lambda fm, **kw: _fake_verdict(cv.FAIL),
    )
    assert research._citation_gate({"doi": "10.1/x"}, force=True) is True


def test_citation_gate_swallows_verify_error(monkeypatch):
    """核验本身抛异常（网络全断）→ 优雅降级放行，绝不卡住主流程。"""

    def _boom(fm, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(research.citation_verify, "verify_frontmatter", _boom)
    assert research._citation_gate({"doi": "10.1/x"}) is True


def test_cite_from_identifier_classification():
    assert research._cite_from_identifier("10.1103/PhysRevLett.121.124501") == {
        "doi": "10.1103/PhysRevLett.121.124501"
    }
    assert research._cite_from_identifier("2301.09876") == {"arxiv_id": "2301.09876"}
    assert research._cite_from_identifier("W2789790776") == {
        "openalex_id": "W2789790776"
    }
    assert research._cite_from_identifier("some paper title") == {
        "title": "some paper title"
    }


def test_verdict_to_json_shape():
    v = _fake_verdict(cv.PASS)
    v.records = {"openalex": _consistent_openalex()}
    d = research._verdict_to_json(v)
    assert d["status"] == "PASS" and d["passed"] is True
    assert d["sources"]["openalex"]["found"] is True
    assert d["sources"]["openalex"]["year"] == 2018
    assert d["conflicts"] == []


def _citecheck_args(**kw) -> argparse.Namespace:
    base = dict(targets=[], note=None, all=False, json=False, no_color=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_citecheck_pass_returns_0(monkeypatch, capsys):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_citation",
        lambda cite, **kw: _fake_verdict(cv.PASS),
    )
    rc = research.cmd_citecheck(_citecheck_args(targets=["10.1/x"]))
    assert rc == 0
    assert "汇总" in capsys.readouterr().out


def test_cmd_citecheck_fail_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_citation",
        lambda cite, **kw: _fake_verdict(cv.FAIL, ["doi 冲突"]),
    )
    rc = research.cmd_citecheck(_citecheck_args(targets=["10.1/x"]))
    assert rc == 1


def test_cmd_citecheck_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_citation",
        lambda cite, **kw: _fake_verdict(cv.PASS),
    )
    rc = research.cmd_citecheck(_citecheck_args(targets=["10.1/x"], json=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and data[0]["status"] == "PASS"


def test_cmd_citecheck_no_input_returns_2(monkeypatch, capsys):
    # all=False 且无 targets/note → 无可核验（不扫 papers/，离线安全）
    rc = research.cmd_citecheck(_citecheck_args())
    assert rc == 2
    assert "无引用可核验" in capsys.readouterr().err


def test_cmd_citecheck_note_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        research.citation_verify,
        "verify_note_file",
        lambda p, **kw: _fake_verdict(cv.PASS),
    )
    p = tmp_path / "n.md"
    p.write_text(NOTE_MD, encoding="utf-8")
    rc = research.cmd_citecheck(_citecheck_args(note=[str(p)]))
    assert rc == 0


def test_cmd_add_gate_blocks_write(monkeypatch, capsys):
    """端到端护栏：add 解析出的元数据三源冲突 → 阻止入库/写笔记（返回 1）。"""
    monkeypatch.setattr(
        research, "_resolve_work", lambda raw: ("openalex", OPENALEX_WORK)
    )
    monkeypatch.setattr(research, "_enrich_work", lambda work, **kw: work)
    monkeypatch.setattr(
        research.citation_verify,
        "verify_frontmatter",
        lambda fm, **kw: _fake_verdict(cv.FAIL, ["x"]),
    )
    called = {"zotero": False, "write": False}
    monkeypatch.setattr(research.zotero_cli, "available", lambda: False)
    monkeypatch.setattr(
        research,
        "_write_note",
        lambda fm, **kw: (
            called.__setitem__("write", True) or research.PAPERS_DIR / "x.md"
        ),
    )
    args = argparse.Namespace(
        doi="10.1/x", tags=None, overwrite=False, verify=True, force=False
    )
    rc = research.cmd_add(args)
    assert rc == 1
    assert called["write"] is False  # FAIL → 未写笔记
