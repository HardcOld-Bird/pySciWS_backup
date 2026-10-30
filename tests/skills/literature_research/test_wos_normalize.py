"""wos_client 规范化契约离线快照测试（不打真实 API）。

锁定 Web of Science **Starter API** → 本项目统一 work-dict 的映射，以及阶段 1b
修正的关键行为：
- ``_normalize_hit`` / ``_normalize_journal`` 的字段映射与「诚实性」边界
  （绝不伪造 JIF / JCR 分区 / ESI —— Starter API 无此数据）；
- Times Cited 从 ``citations`` 列表里取 ``db=WOS`` 的 count；
- ``search`` 用 **camelCase 线参数** ``sortField``（而非官方 Python 客户端 kwarg
  ``sort_field``）—— 这是手写 HTTP 的回归护栏，用错名会被 API 以 400 拒绝；
- ``enrich_openalex_work`` 只补 ``wos_id``、静默降级、不覆盖既有字段。

所有测试通过 monkeypatch 拦截 HTTP / settings，绝不触网。
"""

from __future__ import annotations

import dataclasses

import pytest

from pysci.skills.literature_research.tools import wos_client as wos

# 一份贴近真实 WoS Starter API /documents 命中记录的 JSON（camelCase 嵌套结构）
RAW_DOC = {
    "uid": "WOS:000732948100001",
    "title": "Observation of an exceptional point in a coupled acoustic cavity system",
    "types": ["Journal", "Early Access"],
    "source_types": ["Journal"],
    "source": {
        "sourceTitle": "COMMUNICATIONS PHYSICS",
        "publishYear": 2021,
        "publishMonth": "DEC",
        "volume": "4",
        "issue": "1",
        "articleNumber": "190",
        "pages": {"range": "1-9", "begin": "1", "end": "9", "count": 9},
    },
    "names": {
        "authors": [
            {
                "displayName": "Zhang, San",
                "wosStandard": "Zhang, S",
                "researcherId": "R-1234",
            },
            {"displayName": "Li, Si", "wosStandard": "Li, S"},
        ]
    },
    "links": {
        "record": "https://gateway.webofknowledge.com/gateway/FullRecord.do?Product=WOS&SearchMode=GeneralSearch&qid=1&DestApp=pysciws-wos-lit-review&SrcApp=pysciws-wos-lit-review&SID=A1B2C3D&search_mode=GeneralSearch&product=WOS&cacheurl=1&utm_source=SearchAction&formValue=TS%3D%28acoustic%20exceptional%20point%29&clearQueryFirst=Yes&returnQuery=0&returnQueryDesc=0&returnSort=0&returnLimit=0&returnOffset=0&returnProduct=0&returnDB=0&returnService=0&returnQueryDescLang=0&returnDocDelivery=0&returnEditions=0&returnEditionLang=0&returnViewType=0&returnViewLang=0&returnEditionLang2=0&returnSortLang=0&returnDocDelLang=0&returnSort2=0&returnLimit2=0&returnOffset2=0&returnProduct2=0&returnDB2=0&returnService2=0&returnQuery2=0&returnQueryLang2=0&returnView2=0&returnView2Lang=0&returnSort2Lang=0&returnDocDel2Lang=0&returnQuery3=0&returnQueryLang3=0&returnView3=0&returnView3Lang=0&returnSort3=0&returnDocDel3Lang=0&returnQuery4=0&returnQueryLang4=0&returnView4=0&returnView4Lang=0&returnSort4=0&returnDocDel4Lang=0&returnQuery5=0&returnQueryLang5=0&returnView5=0&returnView5Lang=0&returnSort5=0&returnDocDel5Lang=0&returnQuery6=0&returnQueryLang6=0&returnView6=0&returnView6Lang=0&returnSort6=0&returnDocDel6Lang=0&returnQuery7=0&returnQueryLang7=0&returnView7=0&returnView7Lang=0&returnSort7=0&returnDocDel7Lang=0&returnQuery8=0&returnQueryLang8=0&returnView8=0&returnView8Lang=0&returnSort8=0&returnDocDel8Lang=0&returnQuery9=0&returnQueryLang9=0&returnView9=0&returnView9Lang=0&returnSort9=0&returnDocDel9Lang=0&returnQuery10=0&returnQueryLang10=0&returnView10=0&returnView10Lang=0&returnSort10=0&returnDocDel10Lang=0",
        "citingArticles": "https://gateway.webofknowledge.com/citing",
        "references": "https://gateway.webofknowledge.com/refs",
        "related": "https://gateway.webofknowledge.com/related",
    },
    "citations": [{"db": "WOS", "count": 38}, {"db": "MEDLINE", "count": 5}],
    "identifiers": {
        "doi": "10.1038/s42005-021-00779-x",
        "issn": "2399-3642",
        "eissn": "2399-3642",
        "pmid": "12345678",
    },
    "keywords": {
        "keywords": ["exceptional point", "acoustics"],
        "keywordsPlus": ["NON-HERMITIAN SYSTEMS", "COUPLED RESONATORS"],
    },
}

RAW_JOURNAL = {
    "id": "PHYS REV LETT-2025",
    "name": "PHYSICAL REVIEW LETTERS",
    "jcrTitle": "PHYSICAL REVIEW LETTERS",
    "isoTitle": "Phys. Rev. Lett.",
    "issn": "0031-9007",
    "eIssn": "1079-7114",
    "previousIssn": ["0031-9006"],
    "links": [
        {"type": "JCR", "url": "https://jcr.clarivate.com/jcr-jp/?issn=0031-9007"},
        {"type": "WOS", "url": "https://gateway.webofknowledge.com/..."},
    ],
}


# ---------------------------------------------------------------------------
# _normalize_hit —— Document → work-dict
# ---------------------------------------------------------------------------
def test_normalize_hit_identity():
    h = wos._normalize_hit(RAW_DOC)
    assert h["uid"] == "WOS:000732948100001"
    assert h["wos_id"] == "WOS:000732948100001"  # 与 uid 同源
    assert h["doi"] == "10.1038/s42005-021-00779-x"
    assert h["title"].startswith("Observation of an exceptional point")


def test_normalize_hit_times_cited_prefers_wos_db():
    h = wos._normalize_hit(RAW_DOC)
    assert h["cited_by_count"] == 38  # 取 db=WOS，而非 MEDLINE 的 5


def test_normalize_hit_source_and_pages():
    h = wos._normalize_hit(RAW_DOC)
    assert h["journal"] == "COMMUNICATIONS PHYSICS"
    assert h["source"] == "COMMUNICATIONS PHYSICS"  # 兼容别名
    assert h["publication_year"] == 2021
    assert h["published_year"] == 2021  # 兼容别名
    assert h["publication_date"] == "2021"  # str(year)
    assert (h["volume"], h["issue"], h["article_number"]) == ("4", "1", "190")
    assert h["pages"] == "1-9"
    assert h["first_page"] == "1"
    assert h["last_page"] == "9"
    assert h["page_count"] == 9


def test_normalize_hit_authors():
    h = wos._normalize_hit(RAW_DOC)
    assert h["authors"] == ["Zhang, San", "Li, Si"]
    assert h["first_author_last_name"] == "Zhang"  # 逗号前为姓


def test_normalize_hit_identifiers_and_keywords():
    h = wos._normalize_hit(RAW_DOC)
    assert h["issn"] == "2399-3642"
    assert h["eissn"] == "2399-3642"
    assert h["pmid"] == "12345678"
    assert h["isbn"] == ""  # 缺失 → 空串
    assert h["keywords"] == ["exceptional point", "acoustics"]
    assert h["keywords_plus"] == ["NON-HERMITIAN SYSTEMS", "COUPLED RESONATORS"]


def test_normalize_hit_links_and_types():
    h = wos._normalize_hit(RAW_DOC)
    assert h["record_url"].startswith("https://gateway.webofknowledge.com/")
    assert h["citing_articles_url"].endswith("/citing")
    assert h["cited_references_url"].endswith("/refs")
    assert h["related_records_url"].endswith("/related")
    assert h["doc_type"] == "Journal"  # types[0]
    assert h["types"] == ["Journal", "Early Access"]
    assert h["source_types"] == ["Journal"]


def test_normalize_hit_never_fabricates_jif_or_esi():
    """诚实性护栏：规范化结果绝不携带 JIF / 分区 / ESI 字段（Starter API 无此数据）。"""
    h = wos._normalize_hit(RAW_DOC)
    for forbidden in (
        "jif",
        "jif_5yr",
        "jcr_quartile",
        "esi_highly_cited",
        "esi_hot_paper",
    ):
        assert forbidden not in h


def test_normalize_hit_empty_doc_is_safe():
    h = wos._normalize_hit({})
    assert h["uid"] == "" and h["wos_id"] == ""
    assert h["cited_by_count"] == 0
    assert h["authors"] == [] and h["first_author_last_name"] == ""
    assert h["journal"] == "" and h["publication_year"] is None
    assert h["record_url"] == "" and h["keywords"] == []


# ---------------------------------------------------------------------------
# 纯辅助函数
# ---------------------------------------------------------------------------
def test_times_cited_fallback_and_empty():
    assert wos._times_cited({"citations": [{"db": "MEDLINE", "count": 7}]}) == 7
    assert wos._times_cited({"citations": []}) == 0
    assert wos._times_cited({}) == 0
    assert wos._times_cited({"citations": [{"db": "WOS", "count": 3}]}) == 3


def test_as_str_list_variants():
    assert wos._as_str_list("a; b;c") == ["a", "b", "c"]  # 分号分隔串
    assert wos._as_str_list(["x", "y"]) == ["x", "y"]  # 已是列表
    assert wos._as_str_list([{"keyword": "k1"}, {"value": "k2"}, {"name": "k3"}]) == [
        "k1",
        "k2",
        "k3",
    ]
    assert wos._as_str_list(None) == []
    assert wos._as_str_list("") == []
    assert wos._as_str_list(123) == []  # 非 str/list → 空


def test_first_author_last_name():
    assert wos._first_author_last_name(["Zhang, San"]) == "Zhang"  # 逗号前
    assert wos._first_author_last_name(["San Zhang"]) == "Zhang"  # 无逗号取末词
    assert wos._first_author_last_name([]) == ""


# ---------------------------------------------------------------------------
# _normalize_journal —— Journal → 身份 + JCR URL（无 JIF/分区）
# ---------------------------------------------------------------------------
def test_normalize_journal_identity_and_urls():
    j = wos._normalize_journal(RAW_JOURNAL)
    assert j["id"] == "PHYS REV LETT-2025"
    assert j["name"] == "PHYSICAL REVIEW LETTERS"
    assert j["jcr_title"] == "PHYSICAL REVIEW LETTERS"
    assert j["iso_title"] == "Phys. Rev. Lett."
    assert j["issn"] == "0031-9007"
    assert j["eissn"] == "1079-7114"
    assert j["previous_issn"] == ["0031-9006"]
    assert j["jcr_url"].startswith("https://jcr.clarivate.com/")  # 从 links 里挑出
    assert j["wos_url"].startswith("https://gateway.webofknowledge.com/")


def test_normalize_journal_has_no_jif_fields():
    j = wos._normalize_journal(RAW_JOURNAL)
    for forbidden in ("jif", "impact_factor", "quartile", "jcr_quartile"):
        assert forbidden not in j


def test_normalize_journal_without_links():
    j = wos._normalize_journal({"id": "X", "name": "N"})
    assert j["jcr_url"] == "" and j["wos_url"] == ""
    assert j["links"] == []


# ---------------------------------------------------------------------------
# WOSClient._compose_query —— 年份/类型并入 query（Starter API 无独立参数）
# ---------------------------------------------------------------------------
def test_compose_query_year_range_and_doctype():
    q = wos.WOSClient._compose_query(
        "TS=(acoustic EP)", year_from=2020, year_to=2026, doc_type="Article"
    )
    assert q == "(TS=(acoustic EP)) AND (PY=2020-2026 AND DT=Article)"


def test_compose_query_open_ended_year():
    assert "PY>=2020" in wos.WOSClient._compose_query("TS=(x)", year_from=2020)
    assert "PY<=2026" in wos.WOSClient._compose_query("TS=(x)", year_to=2026)


def test_compose_query_no_duplicate_when_present():
    q = wos.WOSClient._compose_query("TS=(x) AND PY=2021", year_from=2020)
    assert q.count("PY=") == 1  # 已含 PY= → 不再追加


def test_compose_query_no_filter_passthrough():
    assert wos.WOSClient._compose_query("TS=(plain)") == "TS=(plain)"


# ---------------------------------------------------------------------------
# WOSClient.search —— camelCase 线参数护栏（阶段 1b 修正的核心 BUG）
# ---------------------------------------------------------------------------
def _client_with_captured_get(monkeypatch, canned):
    """构造一个跳过 key 检查、且 _get 被替换为捕获器的 WOSClient。"""
    captured: dict = {}
    monkeypatch.setattr(wos.WOSClient, "__init__", lambda self: None)

    def fake_get(self, endpoint, params=None):
        captured["endpoint"] = endpoint
        captured["params"] = dict(params or {})
        return canned

    monkeypatch.setattr(wos.WOSClient, "_get", fake_get)
    return wos.WOSClient(), captured


def test_search_uses_camelcase_sortfield_not_snake_case(monkeypatch):
    canned = {"metadata": {"total": 295, "page": 1, "limit": 2}, "hits": [RAW_DOC]}
    client, captured = _client_with_captured_get(monkeypatch, canned)
    client.search("TS=(acoustic exceptional point)", sort="citations", limit=2)
    params = captured["params"]
    assert captured["endpoint"] == wos.EP_DOCUMENTS
    # 关键：线参数名是 camelCase sortField；用 snake_case 会被 API 以 400 拒绝
    assert params["sortField"] == "TC+D"
    assert "sort_field" not in params
    assert params["q"] == "TS=(acoustic exceptional point)"
    assert params["db"] == "WOS"
    assert params["limit"] == 2
    assert params["page"] == 1


def test_search_sort_name_mapping(monkeypatch):
    canned = {"metadata": {}, "hits": []}
    client, captured = _client_with_captured_get(monkeypatch, canned)
    for friendly, expected in {
        "relevance": "RS+D",
        "date": "LD+D",
        "citations": "TC+D",
        "year": "PY+D",
    }.items():
        client.search("TS=(x)", sort=friendly)
        assert captured["params"]["sortField"] == expected


def test_search_explicit_sort_field_overrides(monkeypatch):
    canned = {"metadata": {}, "hits": []}
    client, captured = _client_with_captured_get(monkeypatch, canned)
    client.search("TS=(x)", sort_field="PY+A")
    assert captured["params"]["sortField"] == "PY+A"


def test_search_clamps_limit_and_page(monkeypatch):
    canned = {"metadata": {}, "hits": []}
    client, captured = _client_with_captured_get(monkeypatch, canned)
    client.search("TS=(x)", limit=100, page=0)
    assert captured["params"]["limit"] == wos.MAX_LIMIT  # 100 → 50
    assert captured["params"]["page"] == 1  # 0 → 1


def test_search_detail_only_when_given(monkeypatch):
    canned = {"metadata": {}, "hits": []}
    client, captured = _client_with_captured_get(monkeypatch, canned)
    client.search("TS=(x)")
    assert "detail" not in captured["params"]
    client.search("TS=(x)", detail="short")
    assert captured["params"]["detail"] == "short"


def test_search_returns_normalized_shape(monkeypatch):
    canned = {"metadata": {"total": 295, "page": 1, "limit": 2}, "hits": [RAW_DOC]}
    client, _ = _client_with_captured_get(monkeypatch, canned)
    res = client.search("TS=(x)", limit=2)
    assert res["total"] == 295 and res["page"] == 1 and res["limit"] == 2
    assert len(res["hits"]) == 1
    assert res["hits"][0]["uid"] == "WOS:000732948100001"  # 已规范化
    assert res["_raw"] is canned


# ---------------------------------------------------------------------------
# enrich_openalex_work —— 只补 wos_id、静默降级、不覆盖既有字段
# ---------------------------------------------------------------------------
def test_enrich_skips_when_not_ready(monkeypatch):
    monkeypatch.setattr(
        wos, "settings", dataclasses.replace(wos.settings, wos_api_key=None)
    )

    def _boom():  # 未就绪时绝不应构造客户端
        raise AssertionError("WOSClient must not be constructed when not ready")

    monkeypatch.setattr(wos, "WOSClient", _boom)
    w = {"doi": "10.1/x", "title": "T"}
    assert wos.enrich_openalex_work(w) is w


def test_enrich_adds_only_wos_id(monkeypatch):
    monkeypatch.setattr(
        wos, "settings", dataclasses.replace(wos.settings, wos_api_key="test-key")
    )

    class FakeClient:
        def __init__(self) -> None:
            pass

        def get_document(self, uid=None, doi=None):
            assert doi == "10.1/x"
            return {"uid": "WOS:0009999999"}

    monkeypatch.setattr(wos, "WOSClient", FakeClient)
    w = {"doi": "10.1/x", "title": "T", "jif": 8.6, "jcr_quartile": "Q1"}
    out = wos.enrich_openalex_work(w)
    assert out["wos_id"] == "WOS:0009999999"
    # 既有字段原样保留，不被覆盖（Starter API 无 JIF/分区，不碰它们）
    assert out["jif"] == 8.6 and out["jcr_quartile"] == "Q1"
    assert out["doi"] == "10.1/x" and out["title"] == "T"


def test_enrich_no_doi_returns_unchanged(monkeypatch):
    monkeypatch.setattr(
        wos, "settings", dataclasses.replace(wos.settings, wos_api_key="test-key")
    )

    class NoCallClient:
        def __init__(self) -> None:
            pass

        def get_document(self, uid=None, doi=None):
            raise AssertionError("get_document must not be called without a DOI")

    monkeypatch.setattr(wos, "WOSClient", NoCallClient)
    w = {"title": "no doi"}
    assert wos.enrich_openalex_work(w) is w


def test_enrich_swallows_api_error(monkeypatch, capsys):
    monkeypatch.setattr(
        wos, "settings", dataclasses.replace(wos.settings, wos_api_key="test-key")
    )

    class BoomClient:
        def __init__(self) -> None:
            pass

        def get_document(self, uid=None, doi=None):
            raise wos.WOSAPIError("500 boom")

    monkeypatch.setattr(wos, "WOSClient", BoomClient)
    w = {"doi": "10.1/x"}
    assert wos.enrich_openalex_work(w) == w
    assert "[wos] enrichment skipped" in capsys.readouterr().out


def test_client_raises_when_key_missing(monkeypatch):
    monkeypatch.setattr(
        wos, "settings", dataclasses.replace(wos.settings, wos_api_key=None)
    )
    with pytest.raises(wos.WOSNotConfigured):
        wos.WOSClient()
