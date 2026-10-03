"""openalex_client 规范化契约离线快照测试（不打真实 API）。

锁定各源→本项目 frontmatter schema 的规范化层：
- ``_extract_work_summary`` 把原始 OpenAlex /works JSON 收敛为统一 work-dict；
- ``work_to_note_frontmatter`` 把 work-dict 转为 paper_note frontmatter；
- 纯辅助函数 ``reconstruct_abstract`` / ``_guess_last_name`` / ``_make_short_title`` /
  ``_format_pages``。

替换检索/HTTP 层（阶段 2 起）时，这些字段映射必须逐字段保持不变——本测试即护栏。
"""

from __future__ import annotations

from pysci.skills.literature_research.tools import openalex_client as oa

# 一份贴近真实 OpenAlex /works 记录的原始 JSON（含倒排摘要、arXiv location、concepts）
RAW_WORK = {
    "id": "https://openalex.org/W2789790776",
    "doi": "https://doi.org/10.1103/PhysRevLett.121.124501",
    "display_name": (
        "Simultaneous Observation of a Topological Edge State and Exceptional Point"
    ),
    "title": "Simultaneous Observation of a Topological Edge State and Exceptional Point",
    "publication_year": 2018,
    "publication_date": "2018-09-21",
    "type": "article",
    "cited_by_count": 230,
    "cited_by_percentile_year": {"min": 98.5},
    "open_access": {"is_oa": True, "oa_status": "green"},
    "best_oa_location": {
        "pdf_url": "https://arxiv.org/pdf/1803.04110",
        "landing_page_url": "https://arxiv.org/abs/1803.04110",
    },
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S137773608",
            "display_name": "Physical Review Letters",
            "issn_l": "0031-9007",
            "host_organization_name": "American Physical Society (APS)",
        }
    },
    "locations": [
        {
            "source": {"display_name": "arXiv (Cornell University)"},
            "landing_page_url": "https://arxiv.org/abs/1803.04110",
            "pdf_url": "https://arxiv.org/pdf/1803.04110",
        }
    ],
    "biblio": {"volume": "121", "issue": "12", "first_page": "124501", "last_page": ""},
    "authorships": [
        {
            "author": {
                "display_name": "Zheng Zhu",
                "orcid": "https://orcid.org/0000-0001",
                "id": "https://openalex.org/A5012345",
            },
            "institutions": [{"display_name": "Nanjing University"}],
            "is_corresponding": True,
            "raw_affiliation_string": "Nanjing University",
        },
        {
            "author": {"display_name": "Xiangang Wan", "orcid": "", "id": ""},
            "institutions": [],
            "is_corresponding": False,
            "raw_affiliation_string": "",
        },
    ],
    "abstract_inverted_index": {
        "Simultaneous": [0],
        "observation": [1],
        "of": [2],
        "topological": [3],
    },
    "concepts": [{"display_name": "Exceptional point", "id": "C1", "score": 0.9}],
    "topics": [{"display_name": "Non-Hermitian physics", "id": "T1", "score": 0.8}],
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
    "related_works": ["https://openalex.org/W3"],
    "counts_by_year": [{"year": 2020, "cited_by_count": 10}],
}


# ---------------------------------------------------------------------------
# _extract_work_summary —— 统一 work-dict 契约
# ---------------------------------------------------------------------------
def test_extract_identity_fields():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["openalex_id"] == "W2789790776"  # 前缀被剥离
    assert s["doi"] == "10.1103/PhysRevLett.121.124501"  # https://doi.org/ 被剥离
    assert s["title"].startswith("Simultaneous Observation")
    assert s["publication_year"] == 2018
    assert s["publication_date"] == "2018-09-21"
    assert s["type"] == "article"


def test_extract_arxiv_id_from_locations():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["arxiv_id"] == "1803.04110"


def test_extract_citations_and_oa():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["cited_by_count"] == 230
    assert s["cited_by_percentile_year"] == 98.5  # 取 {"min": ...}
    assert s["is_oa"] is True
    assert s["oa_status"] == "green"
    assert s["oa_url"] == "https://arxiv.org/pdf/1803.04110"  # 优先 pdf_url


def test_extract_journal_and_publisher():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["journal"] == "Physical Review Letters"
    assert s["journal_issn_l"] == "0031-9007"
    assert s["journal_openalex_id"] == "S137773608"
    assert s["publisher"] == "American Physical Society (APS)"
    assert (s["volume"], s["issue"], s["first_page"], s["last_page"]) == (
        "121",
        "12",
        "124501",
        "",
    )


def test_extract_authors_shape():
    s = oa._extract_work_summary(RAW_WORK)
    a0 = s["authors"][0]
    assert a0["name"] == "Zheng Zhu"
    assert a0["institutions"] == ["Nanjing University"]
    assert a0["is_corresponding"] is True
    assert a0["raw_affiliation"] == "Nanjing University"
    assert set(a0) == {
        "name",
        "orcid",
        "openalex_id",
        "institutions",
        "is_corresponding",
        "raw_affiliation",
    }
    assert s["first_author_last_name"] == "zhu"


def test_extract_abstract_reconstructed():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["abstract"] == "Simultaneous observation of topological"


def test_extract_referenced_and_related_stripped():
    s = oa._extract_work_summary(RAW_WORK)
    assert s["referenced_works_count"] == 2
    assert s["referenced_works"] == ["W1", "W2"]
    assert s["related_works"] == ["W3"]
    assert s["counts_by_year"] == [{"year": 2020, "cited_by_count": 10}]
    assert s["_raw"] is RAW_WORK


def test_extract_concepts_and_topics_truncated():
    work = dict(RAW_WORK)
    work["concepts"] = [
        {"display_name": f"c{i}", "id": f"C{i}", "score": 0.1} for i in range(15)
    ]
    work["topics"] = [
        {"display_name": f"t{i}", "id": f"T{i}", "score": 0.1} for i in range(8)
    ]
    s = oa._extract_work_summary(work)
    assert len(s["concepts"]) == 10  # 硬截断到 10
    assert len(s["topics"]) == 5  # 硬截断到 5


def test_extract_missing_fields_are_safe():
    s = oa._extract_work_summary({})
    assert s["openalex_id"] == ""
    assert s["doi"] == ""
    assert s["arxiv_id"] == ""
    assert s["title"] == ""
    assert s["cited_by_count"] == 0
    assert s["authors"] == []
    assert s["first_author_last_name"] == ""
    assert s["abstract"] == ""


# ---------------------------------------------------------------------------
# 纯辅助函数
# ---------------------------------------------------------------------------
def test_reconstruct_abstract_orders_by_position():
    inv = {"world": [1], "hello": [0], "again": [2]}
    assert oa.reconstruct_abstract(inv) == "hello world again"


def test_reconstruct_abstract_empty():
    assert oa.reconstruct_abstract(None) == ""
    assert oa.reconstruct_abstract({}) == ""


def test_guess_last_name():
    assert oa._guess_last_name("Zheng Zhu") == "zhu"
    assert oa._guess_last_name("Plato") == "plato"  # 单段
    assert oa._guess_last_name("Jean-Luc Picard") == "picard"
    assert oa._guess_last_name("") == ""


def test_make_short_title_strips_stopwords():
    assert oa._make_short_title("On the Theory of Acoustic Waves") == (
        "Theory Acoustic Waves"
    )
    assert oa._make_short_title("") == ""
    # 全是停用词 → 退回原标题截断
    assert oa._make_short_title("The a of") == "The a of"


def test_format_pages():
    assert oa._format_pages("124501", "") == "124501"
    assert oa._format_pages("100", "110") == "100-110"
    assert oa._format_pages("", "110") == "110"
    assert oa._format_pages("", "") == ""


# ---------------------------------------------------------------------------
# work_to_note_frontmatter —— frontmatter 契约（get_source 被 mock，全程离线）
# ---------------------------------------------------------------------------
def test_frontmatter_jif_rounding_and_h_index(monkeypatch):
    monkeypatch.setattr(
        oa, "get_source", lambda **kw: {"2yr_mean_citedness": 8.6789, "h_index": 100}
    )
    fm = oa.work_to_note_frontmatter(oa._extract_work_summary(RAW_WORK))
    assert fm["jif"] == 8.68  # round(8.6789, 2)
    assert fm["journal_h_index"] == 100


def test_frontmatter_contract_when_source_unavailable(monkeypatch):
    monkeypatch.setattr(oa, "get_source", lambda **kw: None)
    fm = oa.work_to_note_frontmatter(oa._extract_work_summary(RAW_WORK))
    # 身份 / 出处
    assert fm["first_author_last_name"] == "zhu"
    assert fm["corresponding_author"] == "Zheng Zhu"
    assert fm["authors"] == ["Zheng Zhu", "Xiangang Wan"]
    assert fm["year"] == 2018
    assert fm["journal"] == "Physical Review Letters"
    assert fm["pages"] == "124501"
    assert fm["doi"] == "10.1103/PhysRevLett.121.124501"
    assert fm["arxiv_id"] == "1803.04110"
    assert fm["openalex_id"] == "W2789790776"
    assert fm["cited_by_count"] == 230
    assert fm["cited_by_count_normalized"] == 98.5
    assert fm["oa_status"] == "green"
    assert fm["keywords_auto"] == ["exceptional point"]
    # WoS 独家字段：OpenAlex 转换时留空，由 wos_client.enrich_openalex_work 补齐
    assert fm["wos_id"] == ""
    # Starter API / OpenAlex 都无法给出的字段：诚实留空而非伪造
    assert fm["jif"] is None
    assert fm["jif_5yr"] is None
    assert fm["jcr_quartile"] == ""
    assert fm["scimago_quartile"] == ""
    assert fm["citescore"] is None
    assert fm["esi_highly_cited"] is False
    assert fm["esi_hot_paper"] is False
    # 笔记工作流状态字段
    assert fm["status"] == "unread"
    assert fm["my_rating"] is None
    assert fm["review_count"] == 0
    assert fm["related_to_my_work"] is None
    assert fm["topics"] == [] and fm["methods"] == [] and fm["systems"] == []
