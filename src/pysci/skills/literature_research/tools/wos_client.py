"""Web of Science (Clarivate) **Starter API** 客户端。

状态：**已批准并启用**——但它是 OpenAlex 主源之上的**可选增强**，不是主力：
它只补 Times Cited 与收录号，而期刊质量指标（JIF / 分区）一律不经过本模块。

Starter API 是 Clarivate 的轻量元数据/检索接口，鉴权只需 **API Key**
（``X-ApiKey`` 头，无需 client secret / OAuth2）。Base URL::

    https://api.clarivate.com/apis/wos-starter/v1

**真实能力边界（务必据实描述，勿夸大）**
Starter API 提供文献/期刊的基础元数据 + 检索，其独有价值在于：

1. **Times Cited**（Web of Science 核心合集的官方被引次数，在 ``citations`` 里）；
2. **权威 UID**（WoS 收录号）与 WoS 记录 URL / 引用该文的 URL；
3. 严格的期刊收录过滤（自动排除未收录来源）；
4. **JCR URL**（``/journals`` 的 links 中指向 Journal Citation Reports 的链接，供人工查阅）。

**Starter API 不提供**（切勿声称本模块能给出）：官方 JIF 数值、JCR 分区（Q1–Q4）、
ESI Highly Cited / Hot Paper 标签、Citation Report（h-index / 总被引汇总等）。

本项目 frontmatter 里那几个字段的**真实来源**（WP-D/WP-E 后的现状）：

- ``jif`` —— OpenAlex ``summary_stats.2yr_mean_citedness`` 的**估算值**，非官方 JIF；
  中段（3-9）较准，顶刊与低引用密度刊可低估 2-3 倍；
- ``jcr_quartile`` —— **恒为空**，直到下述 Journals API 接入；
- ``scimago_quartile`` —— SCImago SJR 本地索引（:mod:`.journal_metrics`，按 ISSN）；
- ``journal_tier`` —— OpenAlex ``listed_in`` 里 JUFO / Norway / KI-JL 的专家评议分级；
- ``esi_highly_cited`` / ``esi_hot_paper`` —— **恒为 ``None``**（“未知”）。这里曾是
  硬编码的 ``False``，而 ``False`` 断言的是“这篇不是 ESI 高被引”——对真正的高被引
  论文那是数据里的谎言，故改为未知。

**升级路径（官方 JIF / JIF 分区 / JCI / ESI 的程序化来源）**

要拿到上述四个指标，需的是 **Web of Science Journals API**::

    https://api.clarivate.com/apis/wos-journals/v1

同样用 ``X-ApiKey`` 鉴权（无 OAuth2），故接入时可直接复用本模块的 ``_headers`` /
``_get`` 写法；官方 Python 客户端（OpenAPI 生成）：
https://github.com/clarivate/wosjournals-python-client

**易混淆点（已查证，记下免得重新研究一遍）**：WoS API **Expanded** 名字里带
“Web of Science”，但它增加的是作者 / 机构 / 标识符 / 基金等**文献级**字段，
**不含 JIF**。JIF 从来不在 WoS 系 API 里，而在单独的 Journals / JCR 产品线上。
该申请尚未落地；在此之前上述两个**免费**替代层（``journal_tier`` /
``scimago_quartile``）已接进笔记生成链路，三者语义不同、并存不冲突。

端点（均为 GET）::

    /documents         检索文献（q 必填；db / limit≤50 / page / sort_field / detail）
    /documents/{uid}   按 UID 取单篇（返回单个 Document，非列表）
    /journals          按 ISSN 查期刊（返回 JournalsList）
    /journals/{id}     按期刊 ID 取单条（返回单个 Journal）

官方文档：

- 门户: https://developer.clarivate.com/
- Starter API: https://developer.clarivate.com/apis/wos-starter
- 官方 Python 客户端（OpenAPI 生成；本模块据其模型手写薄封装，不引入该重依赖）:
  https://github.com/clarivate/wosstarter_python_client

配额（Starter API，学术申请）：约 25,000 records/year，软限速 ~1 req/s。
"""

from __future__ import annotations

from typing import Any

from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
WOS_BASE = settings.wos_api_base_url.rstrip("/")

# 端点（Starter API 仅有 documents / journals 两组）
EP_DOCUMENTS = "/documents"  # 检索文献；/{uid} 取单篇
EP_JOURNALS = "/journals"  # 查期刊；/{id} 取单条

# /documents 分页与排序的真实约束
MAX_LIMIT = 50  # limit ∈ [1, 50]，默认 10
DEFAULT_LIMIT = 10
# sort_field 支持的字段：LD(载入日) / PY(出版年) / RS(相关性) / TC(被引)。
# 形如 "<FIELD>+<A|D>"（A 升序 / D 降序），多字段逗号分隔。
SORT_FIELDS = frozenset({"LD", "PY", "RS", "TC"})

# 友好排序名 → WoS sort_field（供 research CLI 的 --sort 复用）
_SORT_MAP = {
    "relevance": "RS+D",
    "date": "LD+D",  # 载入日降序（最新收录）
    "citations": "TC+D",  # 被引降序
    "year": "PY+D",  # 出版年降序
}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class WOSNotConfigured(RuntimeError):
    """WoS 凭据未配置时抛出，附带申请指引。"""

    def __init__(self) -> None:
        super().__init__(
            "WoS Starter API credentials are not configured.\n"
            "Please:\n"
            "  1. Register an app at https://developer.clarivate.com/ and subscribe\n"
            "     to the Web of Science Starter API\n"
            "  2. Copy the API key into project-root .env as WOS_API_KEY\n"
            "     (Starter API authenticates with the X-ApiKey header only; no secret)\n"
            "  3. Retry the request"
        )


class WOSAPIError(RuntimeError):
    """API 返回错误状态码时抛出。"""


# ---------------------------------------------------------------------------
# 主客户端
# ---------------------------------------------------------------------------
class WOSClient:
    """WoS Starter API 客户端（X-ApiKey 鉴权，无 OAuth2）。

    用法::

        client = WOSClient()
        res = client.search("TS=(exceptional point AND acoustic)", limit=10)
        for h in res["hits"]:
            print(h["title"], h["uid"], h["cited_by_count"])

        doc = client.get_document(uid="WOS:000267144200002")
        jr = client.get_journal(issn="0031-9007")  # PRL → 含 JCR URL（无 JIF 数值）
    """

    def __init__(self) -> None:
        if not settings.wos_api_key:
            raise WOSNotConfigured()

    def _headers(self) -> dict[str, str]:
        return {"X-ApiKey": settings.wos_api_key or ""}

    def _get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET 一个端点。传输层重试由 http_session 处理；非 200 抛 WOSAPIError。

        X-ApiKey 无效会返回 401（其响应体为 ``{error, error_description}``，与常规
        错误体不同）；无令牌可刷新，故不做 401 重试。
        """
        url = f"{WOS_BASE}{endpoint}"
        with http_session() as s:
            r = s.get(
                url,
                params=params or {},
                headers=self._headers(),
                timeout=settings.http_timeout,
            )
            if r.status_code != 200:
                raise WOSAPIError(
                    f"WoS API {endpoint} failed ({r.status_code}): {r.text[:500]}"
                )
            return r.json()

    # -----------------------------------------------------------------------
    # 检索
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        page: int = 1,
        db: str = "WOS",
        year_from: int | None = None,
        year_to: int | None = None,
        doc_type: str | None = None,
        sort: str = "relevance",
        sort_field: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """按 query 检索 WoS（默认核心合集 ``db=WOS``）。

        query 用 WoS 高级检索语法；Starter API 仅支持部分字段标签：
        ``TI``(标题) ``TS``(主题=标题/摘要/作者关键词/Keywords Plus) ``AU``(作者)
        ``AI``(作者标识) ``SO``(来源出版物) ``IS``(ISSN/ISBN) ``VL``(卷) ``CS``(期)
        ``PG``(页) ``PY``(出版年) ``UT``(收录号) ``DO``(DOI) ``DT``(文献类型)
        ``PMID`` ``OG``(机构) ``SUR``(数据源 URL)。

        分页用 ``page``（从 1 起，**不是 offset**）；``limit`` ∈ [1, 50]。排序用
        ``sort_field``（``LD``/``PY``/``RS``/``TC``，形如 ``"PY+D"``；A 升 D 降），或
        传友好名 ``sort``（relevance/date/citations/year）由本函数映射。``db`` 可选
        ``WOS``(核心合集，默认) / ``MEDLINE`` / ``PPRN``(预印本) / ``WOK``(全库) 等。

        ``doc_type`` 不是独立请求参数——Starter API 通过查询里的 ``DT=`` 过滤，故本
        函数把它并入 query（默认不过滤，避免静默缩小结果集）。
        """
        q = self._compose_query(
            query, year_from=year_from, year_to=year_to, doc_type=doc_type
        )
        sf = sort_field or _SORT_MAP.get(sort, "RS+D")
        # 注意：HTTP 线参数名为 camelCase（sortField），与官方 Python 客户端的
        # snake_case kwarg（sort_field）不同；本模块手写 HTTP，必须用线名。
        # 合法线参数：db, detail, edition, limit, modifiedTimeSpan, page,
        #             publishTimeSpan, q, sortField, tcModifiedTimeSpan。
        params: dict[str, Any] = {
            "q": q,
            "db": db,
            "limit": max(1, min(int(limit), MAX_LIMIT)),
            "page": max(1, int(page)),
            "sortField": sf,
        }
        if detail:
            params["detail"] = detail
        data = self._get(EP_DOCUMENTS, params=params)
        meta = data.get("metadata") or {}
        return {
            "total": meta.get("total", 0),
            "page": meta.get("page", params["page"]),
            "limit": meta.get("limit", params["limit"]),
            "hits": [_normalize_hit(h) for h in (data.get("hits") or [])],
            "_raw": data,
        }

    @staticmethod
    def _compose_query(
        query: str,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        doc_type: str | None = None,
    ) -> str:
        """把年份 / 文献类型过滤并入 WoS 查询式（Starter API 无独立 year/docType 参数）。

        若 query 已自带 ``PY=`` / ``DT=``，则不重复追加，避免与之冲突。
        """
        q = (query or "").strip()
        qu = q.upper()
        extra: list[str] = []
        if (year_from or year_to) and "PY=" not in qu:
            if year_from and year_to:
                extra.append(f"PY={year_from}-{year_to}")
            elif year_from:
                extra.append(f"PY>={year_from}")
            else:
                extra.append(f"PY<={year_to}")
        if doc_type and "DT=" not in qu:
            extra.append(f"DT={doc_type}")
        if extra:
            joined = " AND ".join(extra)
            q = f"({q}) AND ({joined})" if q else joined
        return q

    def get_document(
        self, uid: str | None = None, doi: str | None = None
    ) -> dict[str, Any] | None:
        """按 UID（WoS 收录号）取单篇，或按 DOI 检索首条命中。

        ``/documents/{uid}`` 直接返回**单个 Document**（非列表）；DOI 路径退回
        ``/documents?q=DO=<doi>`` 取第一条。
        """
        if uid:
            data = self._get(f"{EP_DOCUMENTS}/{uid}")
            return _normalize_hit(data) if data else None
        if doi:
            r = self.search(f"DO={doi}", limit=1)
            return r["hits"][0] if r["hits"] else None
        raise ValueError("Must provide uid or doi")

    # -----------------------------------------------------------------------
    # 期刊（身份 + JCR URL；Starter API 无 JIF/分区数值）
    # -----------------------------------------------------------------------
    def get_journal(
        self, issn: str | None = None, journal_id: str | None = None
    ) -> dict[str, Any] | None:
        """按 ISSN 或期刊 ID 获取期刊元数据。

        **Starter API 的 Journal 不含 JIF / JCR 分区数值**，只有身份信息 + 指向各
        WoS 产品（含 Journal Citation Reports）的链接。返回里的 ``jcr_url`` 供人工
        查阅官方 JIF/分区。
        """
        if journal_id:
            data = self._get(f"{EP_JOURNALS}/{journal_id}")
            return _normalize_journal(data) if data else None
        if issn:
            data = self._get(EP_JOURNALS, params={"issn": issn})
        else:
            raise ValueError("Must provide issn or journal_id")
        # JournalsList：{metadata?, journals:[...]}（防御式兼容 hits / 直接列表）
        if isinstance(data, list):
            items = data
        else:
            items = data.get("journals") or data.get("hits") or []
        if not items:
            return None
        return _normalize_journal(items[0])


# ---------------------------------------------------------------------------
# 数据规范化（Document / Journal → 本项目统一 work-dict 形状）
# ---------------------------------------------------------------------------
def _as_str_list(v: Any) -> list[str]:
    """把 keywords 等字段规范为字符串列表（兼容 str / list[str] / list[dict]）。"""
    if not v:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(";") if s.strip()]
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if isinstance(x, dict):
                s = x.get("keyword") or x.get("value") or x.get("name") or ""
            else:
                s = str(x)
            if s:
                out.append(s)
        return out
    return []


def _first_author_last_name(authors: list[str]) -> str:
    """从首个作者 displayName 推姓氏。WoS 形如 "Zhang, San"（逗号前为姓）；
    无逗号时（"San Zhang"）取末词为姓。"""
    if not authors:
        return ""
    a0 = authors[0]
    if "," in a0:
        return a0.split(",")[0].strip()
    parts = a0.split()
    return parts[-1] if parts else ""


def _times_cited(h: dict[str, Any]) -> int:
    """Times Cited 在 ``citations`` 列表里：优先取 db=WOS（核心合集）的 count。"""
    fallback = 0
    for c in h.get("citations") or []:
        if not isinstance(c, dict):
            continue
        cnt = c.get("count") or 0
        if (c.get("db") or "").upper() == "WOS":
            return int(cnt)
        if not fallback:
            fallback = int(cnt)
    return fallback


def _normalize_hit(h: dict[str, Any]) -> dict[str, Any]:
    """把 WoS Starter API 的 Document 规范化为本项目统一 work-dict 形状。

    Document 是嵌套结构（JSON 键为 camelCase）::

        uid, title, types[], source_types[],
        source{sourceTitle, publishYear, publishMonth, volume, issue,
               articleNumber, pages{range, begin, end, count}},
        names{authors[{displayName, wosStandard, researcherId}], ...},
        links{record, citingArticles, references, related},
        citations[{db, count}],          # ← Times Cited 在此
        identifiers{doi, issn, eissn, isbn, eisbn, pmid},
        keywords{keywords[], keywordsPlus[]}

    输出键与 openalex_client._extract_work_summary 的 work-dict 契约对齐
    （publication_year / journal / cited_by_count / first_author_last_name 等），
    以便 research.py 的 _row / _frontmatter 直接消费。
    """
    src = h.get("source") or {}
    pages = src.get("pages") or {}
    names = h.get("names") or {}
    idn = h.get("identifiers") or {}
    links = h.get("links") or {}
    kw = h.get("keywords") or {}

    authors: list[str] = []
    for a in names.get("authors") or []:
        if isinstance(a, dict):
            nm = a.get("displayName") or a.get("wosStandard") or ""
        else:
            nm = str(a)
        if nm:
            authors.append(nm)

    types = h.get("types") or []
    uid = h.get("uid") or ""
    year = src.get("publishYear")

    return {
        "uid": uid,
        "wos_id": uid,  # 供 research.py 的 _overlay_enrichment 叠加
        "doi": idn.get("doi") or "",
        "title": h.get("title") or "",
        "authors": authors,
        "first_author_last_name": _first_author_last_name(authors),
        # 来源 / 期刊
        "journal": src.get("sourceTitle") or "",
        "source": src.get("sourceTitle") or "",  # 兼容别名
        # 年份 / 卷期页
        "publication_year": year,
        "published_year": year,  # 兼容别名
        "publication_date": str(year) if year else "",
        "volume": src.get("volume") or "",
        "issue": src.get("issue") or "",
        "pages": pages.get("range") or "",
        "first_page": pages.get("begin") or "",
        "last_page": pages.get("end") or "",
        "page_count": pages.get("count"),
        "article_number": src.get("articleNumber") or "",
        # 被引（WoS 官方 Times Cited）
        "cited_by_count": _times_cited(h),
        # 类型 / 标识符
        "doc_type": (types[0] if types else "") or "",
        "types": types,
        "source_types": h.get("source_types") or [],
        "issn": idn.get("issn") or "",
        "eissn": idn.get("eissn") or "",
        "isbn": idn.get("isbn") or "",
        "pmid": idn.get("pmid") or "",
        # 关键词
        "keywords": _as_str_list(kw.get("keywords")),
        "keywords_plus": _as_str_list(kw.get("keywordsPlus")),
        # 链接（WoS 产品 URL）
        "record_url": links.get("record") or "",
        "citing_articles_url": links.get("citingArticles") or "",
        "cited_references_url": links.get("references") or "",
        "related_records_url": links.get("related") or "",
        "_raw": h,
    }


def _normalize_journal(j: dict[str, Any]) -> dict[str, Any]:
    """把 Journal 规范化：身份信息 + 从 links 里挑出 JCR / WoS URL。

    links 为 ``[{type, url}]``；``type`` 描述产品/页面（如 Journal Citation Reports）。
    **不含 JIF / 分区数值**——Starter API 无此数据。
    """
    links = j.get("links") or []
    jcr_url = ""
    wos_url = ""
    for ln in links:
        if not isinstance(ln, dict):
            continue
        t = (ln.get("type") or "").upper()
        u = ln.get("url") or ""
        if not u:
            continue
        if "JCR" in t or "JOURNAL CITATION" in t:
            jcr_url = u
        elif "WOS" in t or "WEB OF SCIENCE" in t:
            wos_url = u
    return {
        "id": j.get("id") or "",
        "name": j.get("name") or "",
        "jcr_title": j.get("jcrTitle") or "",
        "iso_title": j.get("isoTitle") or "",
        "issn": j.get("issn") or "",
        "eissn": j.get("eIssn") or "",
        "previous_issn": j.get("previousIssn") or [],
        "jcr_url": jcr_url,
        "wos_url": wos_url,
        "links": links,
        "_raw": j,
    }


# ---------------------------------------------------------------------------
# 与 openalex_client 的对接：为 OpenAlex 结果补充 WoS 独家字段
# ---------------------------------------------------------------------------
def enrich_openalex_work(w: dict[str, Any]) -> dict[str, Any]:
    """对 openalex_client.get_work() 返回的 dict 补充 WoS 独家字段。

    Starter API 能提供的独家字段仅有 **wos_id**（WoS 收录号）——它进而可拼出 WoS
    记录 URL。**不含** JIF / JCR 分区 / ESI（Starter API 无此数据；这些 frontmatter
    字段仍由 OpenAlex 估算值填充，不被本函数覆盖）。

    若 WoS 未配置或查询失败，静默返回原 dict（不阻塞主流程）。
    """
    if not settings.wos_ready:
        return w
    try:
        client = WOSClient()
        doc = client.get_document(doi=w["doi"]) if w.get("doi") else None
        if not doc or not doc.get("uid"):
            return w
        enriched = dict(w)
        enriched["wos_id"] = doc["uid"]
        return enriched
    except (WOSNotConfigured, WOSAPIError) as e:
        print(f"[wos] enrichment skipped: {e}")
        return w


# ---------------------------------------------------------------------------
# CLI: 凭据检查 / 检索 / 期刊
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    print(
        "[wos_client] 调试后门——凭据与可达性看 `research doctor` 的【检索源】段，"
        "检索走 `research search '<query>' --source wos`。"
        "注意本入口的 journal 动作（按 ISSN 查 WoS 期刊记录 + JCR URL）在 research CLI "
        "里没有等价物：`research journal lookup` 查的是 OpenAlex listed_in 与 SCImago "
        "SJR，数据源不同，两者互补而非替代。",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="WoS Starter API CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("check", help="检查凭据是否配置正确")
    p_search = sub.add_parser("search", help="检索测试")
    p_search.add_argument(
        "query", nargs="?", default="TS=(exceptional point AND acoustic)"
    )
    p_search.add_argument("--limit", type=int, default=5)
    p_journal = sub.add_parser("journal", help="按 ISSN 查期刊（含 JCR URL）")
    p_journal.add_argument("issn")

    args = parser.parse_args()

    if args.cmd in (None, "check"):
        print(f"wos_ready       : {settings.wos_ready}")
        print(f"wos_api_key     : {'(set)' if settings.wos_api_key else '(unset)'}")
        print(f"wos_api_base_url: {settings.wos_api_base_url}")
        print(
            "\n[i] Starter API 仅提供基础元数据 + Times Cited + JCR URL；"
            "\n    不含官方 JIF / 分区 / ESI（那些需 WoS API Expanded / JCR API）。"
        )
        if settings.wos_ready:
            print(
                "\n[OK] API Key present. Try: "
                "python -m pysci.skills.literature_research.tools.wos_client search"
            )
        else:
            print("\n[!] API Key missing. Please fill WOS_API_KEY in .env")

    elif args.cmd == "search":
        try:
            client = WOSClient()
            r = client.search(args.query, limit=args.limit)
            print(f"Total hits: {r['total']} (page {r['page']}, limit {r['limit']})")
            for i, h in enumerate(r["hits"], 1):
                print(f"\n#{i} [{h['publication_year']}] {h['title']}")
                print(f"   UID        : {h['uid']}")
                print(f"   Source     : {h['journal']}")
                print(f"   DOI        : {h['doi']}")
                print(f"   TimesCited : {h['cited_by_count']}")
                print(f"   Authors    : {'; '.join(h['authors'][:5])}")
                print(f"   WoS URL    : {h['record_url']}")
        except WOSNotConfigured as e:
            print(e)

    elif args.cmd == "journal":
        try:
            client = WOSClient()
            j = client.get_journal(issn=args.issn)
            if not j:
                print(f"No journal found for ISSN {args.issn}")
            else:
                print(f"name     : {j['name']}")
                print(f"id       : {j['id']}")
                print(f"issn     : {j['issn']}   eissn: {j['eissn']}")
                print(f"jcrTitle : {j['jcr_title']}")
                print(f"JCR URL  : {j['jcr_url'] or '(none)'}")
        except WOSNotConfigured as e:
            print(e)
