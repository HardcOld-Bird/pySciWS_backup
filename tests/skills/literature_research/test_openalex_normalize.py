"""openalex_client 规范化契约离线快照测试（不打真实 API）。

锁定各源→本项目 frontmatter schema 的规范化层：
- ``_extract_work_summary`` 把原始 OpenAlex /works JSON 收敛为统一 work-dict；
- ``work_to_note_frontmatter`` 把 work-dict 转为 paper_note frontmatter；
- 纯辅助函数 ``reconstruct_abstract`` / ``normalize_last_name``（从 :mod:`notes` re-export）/
  ``_make_short_title`` / ``_format_pages``。

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


def test_normalize_last_name():
    """原 ``_guess_last_name`` 已删除，姓氏归一化统一到 :func:`notes.normalize_last_name`。"""
    assert oa.normalize_last_name("Zheng Zhu") == "zhu"
    assert oa.normalize_last_name("Plato") == "plato"  # 单段
    assert oa.normalize_last_name("Jean-Luc Picard") == "picard"
    assert oa.normalize_last_name("") == ""


def test_normalize_last_name_transliterates_accents():
    """R4 语义修正：声调是**转写**而非删除。

    旧 ``_guess_last_name`` 用 ``re.sub(r"[^a-zA-Z]", "", ...)`` 直接剔除非 ASCII 字母，
    ``Büttner`` 会变成 ``bttner``；而 ``citation_verify`` 那份实现用 ``unicodedata`` 转写
    得到 ``buttner``。同一位作者在两个源派生出不同姓氏，现已统一到转写语义。
    """
    assert oa.normalize_last_name("Kai Büttner") == "buttner"
    assert oa.normalize_last_name("José García") == "garcia"


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


def test_format_pages_does_not_double_a_single_article_number():
    """``first == last`` 只写一次：电子刊用文章号，OpenAlex 会把它填进两个字段。

    实测 10.1103/PhysRevLett.121.124501 的 ``biblio`` 正是
    ``first_page == last_page == "124501"``，旧实现因此产出 ``124501-124501``。
    那不是无害的冗余：``refs_bridge`` 把 ``pages`` 原样映射成 BibTeX 的 ``pages``，
    于是参考文献里会出现一个并不存在的页码区间。
    """
    assert oa._format_pages("124501", "124501") == "124501"
    # 非数字的文章号同样适用（见 test_journal_ref 里那个 ``eabn7905`` 用例）
    assert oa._format_pages("eabn7905", "eabn7905") == "eabn7905"


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
    assert fm["publication_date"] == "2018-09-21"  # date-only，与 arXiv 源归一
    assert fm["journal"] == "Physical Review Letters"
    assert fm["journal_ref"] == ""  # OpenAlex 不给引文串；本键只为与 arXiv 源契约对齐
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
    # Starter API / OpenAlex 都无法给出的字段：诚实留空而非伪造。
    # 注意 esi_* 必须是 None（= 未知）而不是 False（= 已确认不是）：写 False 对真正的
    # 高被引论文是数据里的谎言，而 ESI 名单只能由 WoS Journals API 给出。
    assert fm["jif"] is None
    assert fm["jif_5yr"] is None
    assert fm["jcr_quartile"] == ""
    assert fm["scimago_quartile"] == ""
    assert fm["citescore"] is None
    assert fm["esi_highly_cited"] is None
    assert fm["esi_hot_paper"] is None
    # 期刊档次三件套：没有 source 就无从判断，诚实留空而非伪造（arXiv 源同样留空，
    # 之后由 research._attach_journal_metrics 拿到 source 再回填）。
    assert fm["journal_tier"] == ""
    assert fm["journal_tier_basis"] == []
    assert fm["listed_in"] == []
    # 笔记工作流状态字段
    assert fm["status"] == "unread"
    assert fm["my_rating"] is None
    assert fm["review_count"] == 0
    assert fm["related_to_my_work"] is None
    assert fm["topics"] == [] and fm["methods"] == [] and fm["systems"] == []


def test_frontmatter_truncates_datetime_publication_date(monkeypatch):
    """``publication_date`` 一律截断为 YYYY-MM-DD。

    OpenAlex 本身给 date-only，截断是防御性的；真正的动机是与 arXiv 归一——后者给
    ``"2018-03-12T04:08:19Z"``，不截断则同一篇论文在两个源下写出两种值，merge 时无法对齐。
    """
    monkeypatch.setattr(oa, "get_source", lambda **kw: None)
    w = oa._extract_work_summary(RAW_WORK)
    w["publication_date"] = "2018-09-21T13:45:00Z"
    assert oa.work_to_note_frontmatter(w)["publication_date"] == "2018-09-21"
    # 缺失时不报错，得空串
    w["publication_date"] = None
    assert oa.work_to_note_frontmatter(w)["publication_date"] == ""


# ---------------------------------------------------------------------------
# WP-E：``_extract_source`` 不再丢弃 ``listed_in``
# ---------------------------------------------------------------------------
#: 实测的 PRL source 响应（只留相关字段）。``listed_in`` 与 ``summary_stats`` 同居一份
#: 响应，因此保留它**零额外请求**。
RAW_SOURCE = {
    "id": "https://openalex.org/S85682845",
    "display_name": "Physical Review Letters",
    "issn": ["0031-9007", "1079-7114"],
    "summary_stats": {"h_index": 1005, "2yr_mean_citedness": 8.97},
    "listed_in": ["cwts-core", "jufo-3", "ki-jl-2", "medline", "norway-2"],
}


def test_extract_source_keeps_listed_in():
    """P0 回归：旧实现只取 ``summary_stats``，把 ``listed_in`` 整个丢了。

    丢掉它的后果是期刊档次只剩引用类指标可用——对声学这类**低引用密度**领域严重失真
    （``J. Acoust. Soc. Am.`` 的 ``2yr_mean_citedness`` 只有 0.82，但 JUFO 专家小组判它
    最高档）。档次派生本身的用例见 ``test_journal_metrics.py``。
    """
    src = oa._extract_source(RAW_SOURCE)
    assert src["listed_in"] == RAW_SOURCE["listed_in"]


def test_extract_source_listed_in_defaults_to_empty_list():
    """缺字段时给 ``[]`` 而不是 ``None``。

    下游 ``derive_journal_tier`` 与 frontmatter 的 list 字段都假定可迭代；``None`` 还会让
    ``merge_frontmatter`` 把它当「空键」反复回填，每次 ``index --fix`` 都报变更。
    """
    src = oa._extract_source({"id": "https://openalex.org/S1"})
    assert src["listed_in"] == []
    assert src["listed_in"] is not None
