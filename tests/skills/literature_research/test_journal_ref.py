"""WP-D 的离线回归测试：``parse_journal_ref`` + arXiv frontmatter 字段语义 + 期刊指标补齐。

护栏对象是三个实测缺陷：

1. **``journal_ref`` 自由文本污染**（D1）：旧实现把 arXiv 的整串引文
   ``"Phys. Rev. Lett. 121, 124501 (2018)"`` 直接塞进 ``journal`` 键，于是 ``INDEX.md``
   的 Journal 列、笔记的 Journal-tier 段落都变成一串引文，所有按刊名做的期刊指标查询
   全部落空。仓库里的 ``2018_zhu`` 笔记正是这个样子。
2. **arXiv 笔记的期刊指标缺失**（D2）：``2018_zhu``（PRL）的 ``jif: null`` 不是因为 PRL
   无数据（实测 OpenAlex ``2yr_mean_citedness`` = 8.97），而是 OpenAlex 转换器只在
   ``journal_openalex_id`` 存在时才查 source，而 arXiv entry 没有这个字段。
3. **``esi_*`` 假阴性**（D3）：``False`` 断言的是「这篇不是 ESI 高被引」，而真实状态是
   「未知」——对真正的高被引论文，这是数据里的谎言。

全部测试离线：OpenAlex 的 ``get_work`` / ``get_source`` 一律 monkeypatch。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from pysci.skills.literature_research.tools import (
    arxiv_client,
    notes,
    openalex_client,
    research,
)

# ---------------------------------------------------------------------------
# 夹具：一篇真实的 arXiv entry（1803.04110，即仓库里的 2018_zhu）
# ---------------------------------------------------------------------------
ARXIV_ENTRY: dict[str, Any] = {
    "arxiv_id": "1803.04110",
    "version": "v2",
    "title": (
        "Simultaneous Observation of Topological Edge State and Exceptional "
        "Point in an Open and Non-Hermitian System"
    ),
    "abstract": "...",
    "authors": [
        {"name": "Weiwei Zhu", "affiliation": ""},
        {"name": "Yong Li", "affiliation": ""},
    ],
    "first_author_last_name": "zhu",
    # arXiv 给的是带时分秒的 ISO-8601；OpenAlex 给 date-only —— 两者必须归一
    "published": "2018-03-12T04:08:19Z",
    "updated": "2018-09-25T02:11:33Z",
    "primary_category": "cond-mat.mes-hall",
    "categories": ["cond-mat.mes-hall", "physics.app-ph", "physics.class-ph"],
    "journal_ref": "Phys. Rev. Lett. 121, 124501 (2018)",
    "doi": "10.1103/PhysRevLett.121.124501",
    "comment": "",
    "abs_url": "https://arxiv.org/abs/1803.04110v2",
    "pdf_url": "https://arxiv.org/pdf/1803.04110v2",
    "src_url": "https://arxiv.org/e-print/1803.04110v2",
}

#: OpenAlex ``sources`` 端点对 PRL 的实测值（``2yr_mean_citedness`` ≈ JIF）。
PRL_SOURCE: dict[str, Any] = {
    "openalex_id": "S85682845",
    "display_name": "Physical Review Letters",
    "issn": ["0031-9007", "1079-7114"],
    "h_index": 1005,
    "2yr_mean_citedness": 8.97,
}


@pytest.fixture
def entry() -> dict[str, Any]:
    """返回一份**副本**，避免测试之间互相污染模块级夹具。"""
    return dict(ARXIV_ENTRY)


def _mock_source_lookup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: dict[str, Any] | None = PRL_SOURCE,
    work: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """把 OpenAlex 的两级查找都 mock 掉，并记录调用参数供断言。"""
    calls: dict[str, list[Any]] = {"get_work": [], "get_source": []}
    work = (
        work
        if work is not None
        else {"journal_openalex_id": "S85682845", "title": ARXIV_ENTRY["title"]}
    )

    def _get_work(**kw: Any) -> dict[str, Any] | None:
        calls["get_work"].append(kw)
        return work

    def _get_source(**kw: Any) -> dict[str, Any] | None:
        calls["get_source"].append(kw)
        # openalex_id 精确路径命中；name 兜底路径也返回同一份，便于两种分支共用夹具
        return source

    monkeypatch.setattr(research.openalex_client, "get_work", _get_work)
    monkeypatch.setattr(research.openalex_client, "get_source", _get_source)
    return calls


# ===========================================================================
#  D1 —— parse_journal_ref
# ===========================================================================
def test_parses_full_citation():
    """方案点名的真实字符串：刊名 / 卷 / 页 / 年 各归其位。"""
    got = arxiv_client.parse_journal_ref("Phys. Rev. Lett. 121, 124501 (2018)")
    assert got["journal"] == "Phys. Rev. Lett."
    assert got["volume"] == "121"
    assert got["pages"] == "124501"
    assert got["year"] == 2018


def test_parses_without_year():
    """退化模式：只有「刊名 卷, 页」，year 为 None 而不是编一个出来。"""
    got = arxiv_client.parse_journal_ref("Phys. Rev. B 96, 085117")
    assert got["journal"] == "Phys. Rev. B"
    assert got["volume"] == "96"
    assert got["pages"] == "085117"
    assert got["year"] is None


def test_parses_journal_name_only():
    got = arxiv_client.parse_journal_ref("Nature Physics")
    assert got["journal"] == "Nature Physics"
    assert got["volume"] is None
    assert got["pages"] is None
    assert got["year"] is None


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_empty_input_yields_all_blank(raw):
    got = arxiv_client.parse_journal_ref(raw)
    assert got == {
        "journal": "",
        "volume": None,
        "pages": None,
        "year": None,
        "journal_ref": "",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "Phys. Rev. Lett. 121, 124501 (2018)",
        "Phys. Rev. B 96, 085117",
        "Phys. Rev. Applied 19, 054003 (2023)",
        "Sci. Adv. 8, eabn7905 (2022)",
        "Nature 605, 44-48 (2022)",
        "J. Acoust. Soc. Am. 143, 1245 (2018)",
    ],
)
def test_journal_field_no_longer_contains_volume_pages_year(raw):
    """核心回归：``journal`` 里**不得**再出现卷号、页码或年份括号。"""
    journal = arxiv_client.parse_journal_ref(raw)["journal"]
    assert "(" not in journal and ")" not in journal
    assert not any(ch.isdigit() for ch in journal), journal
    assert "," not in journal


def test_handles_missing_space_before_year():
    """``page`` 的字符类含 ``()``，缺空格时靠回溯仍能正确拆开。"""
    got = arxiv_client.parse_journal_ref("Phys. Rev. B 96, 085117(2018)")
    assert got["journal"] == "Phys. Rev. B"
    assert got["volume"] == "96"
    assert got["pages"] == "085117"
    assert got["year"] == 2018


def test_handles_non_numeric_pages():
    """页码可以是文章号（``eabn7905``），所以 ``pages`` 保持字符串而不数值化。"""
    got = arxiv_client.parse_journal_ref("Sci. Adv. 8, eabn7905 (2022)")
    assert got["pages"] == "eabn7905"
    assert got["year"] == 2022


def test_handles_page_range():
    got = arxiv_client.parse_journal_ref("Nature 605, 44-48 (2022)")
    assert got["journal"] == "Nature"
    assert got["pages"] == "44-48"


def test_journal_name_may_contain_digits():
    """惰性 ``.+?`` 保证刊名里的数字不被误当卷号。"""
    got = arxiv_client.parse_journal_ref("2D Materials 5, 031001 (2018)")
    assert got["journal"] == "2D Materials"
    assert got["volume"] == "5"


def test_collapses_whitespace_and_newlines():
    """arXiv 的 ``journal_ref`` 偶尔带换行；折叠后仍可解析，原串也按折叠形态留档。"""
    got = arxiv_client.parse_journal_ref("Phys. Rev. Lett.  121,\n  124501  (2018)")
    assert got["journal"] == "Phys. Rev. Lett."
    assert got["pages"] == "124501"
    assert got["journal_ref"] == "Phys. Rev. Lett. 121, 124501 (2018)"


def test_raw_string_is_preserved_for_audit():
    """拆解是**有损**的：原串必须留在 ``journal_ref`` 里，否则无从核对解析是否正确。"""
    raw = "Phys. Rev. Lett. 121, 124501 (2018)"
    assert arxiv_client.parse_journal_ref(raw)["journal_ref"] == raw


def test_unparsable_string_falls_back_to_raw():
    """解析不出结构时 ``journal`` 回落为原串——宁可粗糙也不丢信息。"""
    raw = "submitted to Nature Communications"
    got = arxiv_client.parse_journal_ref(raw)
    assert got["journal"] == raw
    assert got["volume"] is None and got["year"] is None


def test_ignores_trailing_junk_after_year():
    """主模式不锚定行尾，故尾随的 DOI / 备注不影响解析。"""
    got = arxiv_client.parse_journal_ref(
        "Phys. Rev. Lett. 121, 124501 (2018). doi:10.1103/PhysRevLett.121.124501"
    )
    assert got["journal"] == "Phys. Rev. Lett."
    assert got["year"] == 2018


# ===========================================================================
#  D1 + D3 + D5 —— arxiv_to_note_frontmatter 的字段语义
# ===========================================================================
def test_frontmatter_journal_is_clean(entry):
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["journal"] == "Phys. Rev. Lett."
    assert fm["journal_ref"] == "Phys. Rev. Lett. 121, 124501 (2018)"
    assert fm["volume"] == "121"
    assert fm["pages"] == "124501"


def test_frontmatter_publication_date_is_date_only(entry):
    """D5：arXiv 的 ``2018-03-12T04:08:19Z`` 必须截断成与 OpenAlex 同形态。"""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["publication_date"] == "2018-03-12"


def test_frontmatter_year_prefers_arxiv_published(entry):
    """``year`` 决定文件名 ``{year}_{last}_{slug}.md``，改动会破坏既有 wiki 链接。

    故跨年发表（2017-12 投稿、2018 见刊）时**仍以 arXiv 的 submitted 年为准**，
    只在 ``published`` 缺失时才回落 ``journal_ref`` 里的出版年。
    """
    entry["published"] = "2017-12-30T00:00:00Z"
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["year"] == 2017


def test_frontmatter_year_falls_back_to_journal_ref(entry):
    entry["published"] = ""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["year"] == 2018


@pytest.mark.parametrize("field", ["esi_highly_cited", "esi_hot_paper"])
def test_esi_fields_are_none_not_false(entry, field):
    """D3：``None`` = 未知，``False`` = 已确认不是。后者对高被引论文是谎言。"""
    assert arxiv_client.arxiv_to_note_frontmatter(entry)[field] is None


def test_topics_and_keywords_auto_no_longer_duplicate(entry):
    """D5：旧实现两键都等于 arXiv categories，AI 无处填自己的研究主题判断。"""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["topics"] == []
    assert fm["keywords_auto"] == [
        "cond-mat.mes-hall",
        "physics.app-ph",
        "physics.class-ph",
    ]


def test_keywords_auto_capped_at_five(entry):
    entry["categories"] = [f"c{i}" for i in range(9)]
    assert len(arxiv_client.arxiv_to_note_frontmatter(entry)["keywords_auto"]) == 5


def test_unpublished_preprint_uses_placeholder(entry):
    entry["journal_ref"] = ""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["journal"] == arxiv_client.ARXIV_PREPRINT_JOURNAL
    assert fm["journal_ref"] == ""
    assert fm["volume"] == "" and fm["pages"] == ""


def test_frontmatter_keys_are_a_subset_of_field_order(entry):
    """转换器产出的每个键都必须在 ``notes.FIELD_ORDER`` 里，否则 ``index --fix`` 会把它
    当成未知字段排到末尾，规范化 diff 里就会出现纯粹的字段搬家噪声。"""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    unknown = [k for k in fm if k not in notes.FIELD_ORDER]
    assert unknown == []


def test_both_converters_produce_the_same_key_set(entry):
    """两个源的 frontmatter 契约必须一致，否则 merge 时会出现单源独有的键。

    不给 ``journal_openalex_id``，避开 ``work_to_note_frontmatter`` 内部的 get_source 调用。
    """
    arxiv_keys = set(arxiv_client.arxiv_to_note_frontmatter(entry))
    openalex_keys = set(openalex_client.work_to_note_frontmatter({"title": "T"}))
    assert arxiv_keys == openalex_keys
    # 两边都必须包含 WP-D 新增/修正的键
    for k in ("journal_ref", "esi_highly_cited", "esi_hot_paper", "publication_date"):
        assert k in arxiv_keys and k in openalex_keys


# ===========================================================================
#  D2 —— _attach_journal_metrics
# ===========================================================================
def test_attaches_jif_and_h_index_for_arxiv_note(entry, monkeypatch, capsys):
    """P0 回归：arXiv 来源的笔记此前 ``jif`` 恒为 null，尽管 PRL 明明有数据。"""
    _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["jif"] is None

    got = research._attach_journal_metrics(fm)

    assert got["jif"] == 8.97
    assert got["journal_h_index"] == 1005
    out = capsys.readouterr().out
    assert "Phys. Rev. Lett." in out and "jif" in out
    # 输出必须点明 JIF 是估算值，否则读者会把它当成官方 JCR
    assert "非官方 JCR" in out


def test_uses_doi_path_first(entry, monkeypatch):
    """DOI → work → journal_openalex_id 是**唯一确定**的路径，优先于模糊的刊名搜索。"""
    calls = _mock_source_lookup(monkeypatch)
    research._attach_journal_metrics(arxiv_client.arxiv_to_note_frontmatter(entry))
    assert calls["get_work"] == [{"doi": "10.1103/PhysRevLett.121.124501"}]
    assert calls["get_source"] == [{"openalex_id": "S85682845"}]


def test_falls_back_to_name_lookup_without_doi(entry, monkeypatch):
    calls = _mock_source_lookup(monkeypatch, work={"journal_openalex_id": ""})
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    fm["doi"] = ""

    research._attach_journal_metrics(fm)

    assert calls["get_work"] == []  # 无 DOI 就不必浪费一次请求
    assert calls["get_source"] == [{"name": "Phys. Rev. Lett."}]


def test_skips_when_jif_already_present(entry, monkeypatch):
    """已有值说明上游查过；重查既浪费配额，又可能把人工校订的值顶掉。"""
    calls = _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    fm["jif"] = 4.08

    assert research._attach_journal_metrics(fm)["jif"] == 4.08
    assert calls["get_work"] == [] and calls["get_source"] == []


def test_zero_jif_counts_as_a_real_value(entry, monkeypatch):
    """``jif == 0`` 是合法取值（零引用密度刊），不得被当成「未查过」。"""
    calls = _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    fm["jif"] = 0

    assert research._attach_journal_metrics(fm)["jif"] == 0
    assert calls["get_source"] == []


def test_skips_arxiv_preprint_placeholder(entry, monkeypatch):
    """``"arXiv preprint"`` 不是刊名，拿它去搜期刊会得到完全无关的命中。"""
    calls = _mock_source_lookup(monkeypatch)
    entry["journal_ref"] = ""
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert fm["journal"] == arxiv_client.ARXIV_PREPRINT_JOURNAL

    assert research._attach_journal_metrics(fm)["jif"] is None
    assert calls["get_work"] == [] and calls["get_source"] == []


def test_skips_blank_journal(entry, monkeypatch):
    calls = _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    fm["journal"] = "   "

    assert research._attach_journal_metrics(fm)["jif"] is None
    assert calls["get_source"] == []


def test_does_not_overwrite_existing_h_index(entry, monkeypatch):
    calls = _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    fm["journal_h_index"] = 42

    got = research._attach_journal_metrics(fm)

    assert got["journal_h_index"] == 42  # 非空值不动
    assert got["jif"] == 8.97  # 空的照补
    assert calls["get_source"]


def test_does_not_rewrite_journal_name(entry, monkeypatch):
    """补齐指标**不得**顺手把刊名改成 OpenAlex 的规范全名——那是覆盖数据源给出的值。"""
    _mock_source_lookup(monkeypatch)
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)

    assert research._attach_journal_metrics(fm)["journal"] == "Phys. Rev. Lett."


@pytest.mark.parametrize("bad", ["not_found", "raises"])
def test_silent_degradation(entry, monkeypatch, capsys, bad):
    """不变量 1：查不到 / 抛异常都不报错、不阻塞，只留一行提示。"""
    if bad == "not_found":
        _mock_source_lookup(monkeypatch, source=None)
    else:

        def _boom(**kw: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr(research.openalex_client, "get_work", _boom)
        monkeypatch.setattr(research.openalex_client, "get_source", _boom)

    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    got = research._attach_journal_metrics(fm)

    assert got["jif"] is None and got["journal_h_index"] is None
    err = capsys.readouterr().err
    assert "未取到期刊指标" in err
    assert "RuntimeError" not in err  # 客户端内部的 traceback 噪声被拦下了


def test_source_without_metrics_leaves_fields_empty(entry, monkeypatch, capsys):
    """拿到 source 但缺 ``2yr_mean_citedness`` 时不编造数值，也不打印「补齐」。"""
    _mock_source_lookup(monkeypatch, source={"display_name": "Physical Review Letters"})
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)

    got = research._attach_journal_metrics(fm)

    assert got["jif"] is None and got["journal_h_index"] is None
    captured = capsys.readouterr()
    assert "补齐" not in captured.out


def test_jif_is_rounded_to_two_decimals(entry, monkeypatch):
    _mock_source_lookup(
        monkeypatch, source={**PRL_SOURCE, "2yr_mean_citedness": 8.971234}
    )
    fm = arxiv_client.arxiv_to_note_frontmatter(entry)
    assert research._attach_journal_metrics(fm)["jif"] == 8.97


# ===========================================================================
#  D2 —— _build_frontmatter 漏斗
# ===========================================================================
def test_build_frontmatter_attaches_metrics_for_arxiv(entry, monkeypatch):
    """三个命令都走这个漏斗，所以期刊指标补齐不会被任何一处遗漏。"""
    _mock_source_lookup(monkeypatch)

    fm = research._build_frontmatter("arxiv", entry)

    assert fm["journal"] == "Phys. Rev. Lett."
    assert fm["jif"] == 8.97
    assert fm["journal_h_index"] == 1005
    assert fm["arxiv_id"] == "1803.04110"


def test_build_frontmatter_overlays_wos_id(entry, monkeypatch):
    """WoS 叠加必须在漏斗里仍然生效（``_enrich_work`` 把 wos_id 写进 work）。"""
    _mock_source_lookup(monkeypatch)
    entry["wos_id"] = "WOS:000443812300013"

    assert research._build_frontmatter("arxiv", entry)["wos_id"] == entry["wos_id"]


def test_all_frontmatter_call_sites_use_the_funnel():
    """回归护栏：``cmd_read`` / ``cmd_get`` / ``cmd_add`` 不得再各自拼三步。

    期刊指标补齐是容易忘的一步：只要三个命令都走漏斗，就不会出现「read 有指标、
    add 没指标」这种难查的不一致。本测试故意做成源码文本层面的结构护栏。
    """
    src = inspect.getsource(research)
    # 三步组装只允许在漏斗内部出现一次
    assert src.count("_overlay_enrichment(_frontmatter_from_work(") == 1
    assert src.count("_build_frontmatter(src, work)") == 3


# ===========================================================================
#  D1 —— 展示层同样不得被污染
# ===========================================================================
def test_row_journal_column_is_not_polluted():
    """``_row`` 曾直接把 ``journal_ref`` 当刊名展示，与刚修好的 frontmatter 不一致。"""
    r = research._row("arxiv", ARXIV_ENTRY)
    assert r["journal"] == "Phys. Rev. Lett."
    assert "124501" not in r["journal"] and "(2018)" not in r["journal"]


def test_row_prefers_explicit_journal_over_journal_ref():
    w = {**ARXIV_ENTRY, "journal": "Physical Review Letters"}
    assert research._row("arxiv", w)["journal"] == "Physical Review Letters"


def test_row_falls_back_to_arxiv_label():
    w = {**ARXIV_ENTRY, "journal_ref": ""}
    assert research._row("arxiv", w)["journal"] == "arXiv"
    # 非 arXiv 源没有这个占位约定
    assert research._row("openalex", w)["journal"] == ""
