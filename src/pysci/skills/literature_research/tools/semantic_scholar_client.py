"""Semantic Scholar 学术检索客户端（OpenAlex 的补充源）。

Semantic Scholar (S2) 由 Allen Institute for AI 维护，提供 OpenAlex 无法覆盖的独特能力：
1. **TLDR**：AI 生成的一句话论文摘要（快速筛选利器）
2. **influentialCitationCount**：实质性引用数（排除仅在 background 段提及的引用）
3. **citation intent 标签**：background / method / result（理解论文如何被使用）
4. **SPECTER embeddings**：语义相似度（推荐相关论文）

API 文档：https://api.semanticscholar.org/api-docs/graph
- 免费 API key：https://www.semanticscholar.org/product/api#api-key
- 有 key：100 req/5min；无 key：1 req/s（不稳定）
- 无需 OAuth，只需在 header 中传 `x-api-key`

本模块提供：
- :func:`search_papers`       : 关键词检索（含 TLDR、引用数）
- :func:`get_paper`           : 按 DOI/arXiv ID/S2 ID 获取单篇详情
- :func:`get_citations`       : 获取施引文献（含 citation intent）
- :func:`get_references`      : 获取参考文献
- :func:`get_recommendations` : 基于正/负例推荐相关论文
- :func:`paper_to_note_frontmatter` : 转换为 paper_note.md 的 YAML 字段

用法示例::

    from pysci.skills.literature_research.tools.semantic_scholar_client import search_papers, get_paper

    results = search_papers("exceptional point acoustic metasurface", year_range="2024-2026")
    for p in results["data"]:
        print(p["title"], p["tldr"])

    detail = get_paper("DOI:10.1103/PhysRevLett.131.066601")
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Iterable

from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"

# 指数退避参数（满足 S2 API 使用条款中对 backoff 的承诺）
# 申请 key 时需勾选 "I will apply exponential backoff ..."，本模块在客户端落实该策略。
S2_MAX_RETRIES = 5          # 429/5xx 最多重试次数
S2_BASE_DELAY = 1.0         # 首次退避基数（秒），按 2^attempt 递增
S2_MAX_DELAY = 60.0         # 单次退避上限（秒）
S2_RETRY_STATUS = (429, 500, 502, 503, 504)

# 默认返回字段（平衡信息量与响应大小）
DEFAULT_FIELDS = ",".join([
    "paperId", "externalIds", "url", "title", "abstract",
    "venue", "publicationVenue", "year", "publicationDate",
    "citationCount", "influentialCitationCount", "referenceCount",
    "authors", "tldr", "openAccessPdf", "isOpenAccess", "fieldsOfStudy",
    "s2FieldsOfStudy", "journal", "citationStyles",
])

CITATION_FIELDS = ",".join([
    "paperId", "title", "year", "venue", "citationCount",
    "authors", "tldr", "contexts", "intents", "isInfluential",
])

RECOMMENDATION_FIELDS = ",".join([
    "paperId", "title", "year", "venue", "citationCount",
    "authors", "tldr", "openAccessPdf", "externalIds",
])


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class S2NotConfigured(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Semantic Scholar API key is not configured.\n"
            "Please:\n"
            "  1. Request a free key at https://www.semanticscholar.org/product/api#api-key\n"
            "  2. Fill SEMANTIC_SCHOLAR_API_KEY in project-root .env\n"
            "Note: The API works without a key but is rate-limited to 1 req/s and unstable."
        )


class S2APIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 底层请求
# ---------------------------------------------------------------------------
def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        h["x-api-key"] = settings.semantic_scholar_api_key
    return h


def _backoff_delay(attempt: int, retry_after: str | None) -> float:
    """计算第 attempt 次重试前的等待秒数。

    优先采用服务端 Retry-After；否则用指数退避 + 拖动（jitter），上限 S2_MAX_DELAY。
    """
    if retry_after:
        try:
            return min(float(retry_after), S2_MAX_DELAY)
        except ValueError:
            pass
    delay = S2_BASE_DELAY * (2 ** attempt)
    delay += random.uniform(0, delay * 0.1)   # 拖动，避免多客户端同步重试
    return min(delay, S2_MAX_DELAY)


def _request_with_backoff(method: str, url: str, **kwargs: Any) -> Any:
    """统一的带指数退避请求（GET/POST 均适用）。

    - 传输层重试关闭（retries=0, retry_on_status=False），429/5xx 作为普通响应返回，
      退避完全由本函数控制，避免双重退避并能读取 Retry-After；
    - 对 429/5xx 及网络异常重试，尊重 Retry-After；
    - 重试耗尽后抛出 S2APIError。
    """
    kwargs.setdefault("timeout", settings.http_timeout)
    last_status: int | None = None
    last_text = ""
    for attempt in range(S2_MAX_RETRIES + 1):
        try:
            with http_session(retries=0, retry_on_status=False) as s:
                r = s.request(method, url, **kwargs)
        except Exception as e:   # 网络异常也退避重试
            delay = _backoff_delay(attempt, None)
            print(f"[s2] {method} network error ({e}); retry {attempt + 1}/{S2_MAX_RETRIES} in {delay:.1f}s")
            if attempt < S2_MAX_RETRIES:
                time.sleep(delay)
                continue
            raise S2APIError(f"S2 API {method} {url} failed after {S2_MAX_RETRIES} retries: {e}") from e

        if r.status_code == 200:
            return r
        if r.status_code in S2_RETRY_STATUS and attempt < S2_MAX_RETRIES:
            last_status, last_text = r.status_code, r.text[:200]
            delay = _backoff_delay(attempt, r.headers.get("Retry-After"))
            print(f"[s2] HTTP {r.status_code}; retry {attempt + 1}/{S2_MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)
            continue
        # 不可重试的状态码或重试耗尽：返回响应由调用方处理
        return r
    # 理论不会到达（循环内已 return/raise）
    raise S2APIError(f"S2 API {method} {url} failed ({last_status}): {last_text}")


def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET 请求，自动附加 API key、指数退避处理限速。"""
    url = f"{S2_BASE}{endpoint}"
    r = _request_with_backoff("GET", url, params=params or {}, headers=_headers())
    if r.status_code != 200:
        raise S2APIError(f"S2 API {endpoint} failed ({r.status_code}): {r.text[:500]}")
    return r.json()


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST 请求（用于推荐等端点），同样带指数退避。"""
    url = f"{S2_BASE}{endpoint}"
    r = _request_with_backoff("POST", url, json=payload, headers=_headers())
    if r.status_code != 200:
        raise S2APIError(f"S2 API POST {endpoint} failed ({r.status_code}): {r.text[:500]}")
    return r.json()


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def search_papers(
    query: str,
    *,
    year_range: str | None = None,      # 例如 "2024-2026" 或 "2024-"
    venue: str | None = None,
    fields_of_study: str | None = None,  # 例如 "Physics"
    min_citation_count: int | None = None,
    open_access_only: bool = False,
    fields: str | None = None,
    limit: int = 20,                     # S2 最大 100
    offset: int = 0,
    sort: str | None = None,             # 例如 "citationCount:desc"
) -> dict[str, Any]:
    """关键词检索论文。

    返回::

        {
            "total": int,
            "offset": int,
            "data": [paper_dict, ...],
        }
    """
    params: dict[str, Any] = {
        "query": query,
        "fields": fields or DEFAULT_FIELDS,
        "limit": min(limit, 100),
        "offset": offset,
    }
    if year_range:
        params["year"] = year_range
    if venue:
        params["venue"] = venue
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    if min_citation_count is not None:
        params["minCitationCount"] = min_citation_count
    if open_access_only:
        params["openAccessPdf"] = ""
    if sort:
        params["sort"] = sort

    data = _get("/paper/search", params=params)
    return {
        "total": data.get("total", 0),
        "offset": data.get("offset", 0),
        "data": [_normalize_paper(p) for p in (data.get("data") or [])],
    }


def search_papers_bulk(
    query: str,
    *,
    year_range: str | None = None,
    fields: str | None = None,
    sort: str = "citationCount:desc",
    limit: int = 100,
) -> dict[str, Any]:
    """批量检索端点（/paper/search/bulk），单次最多 1000 条，适合大规模综述。

    注意：bulk 端点不支持 offset 分页，而是用 token 翻页。
    """
    params: dict[str, Any] = {
        "query": query,
        "fields": fields or DEFAULT_FIELDS,
        "sort": sort,
    }
    if year_range:
        params["year"] = year_range

    all_papers: list[dict] = []
    token: str | None = None
    while len(all_papers) < limit:
        if token:
            params["token"] = token
        data = _get("/paper/search/bulk", params=params)
        batch = data.get("data") or []
        all_papers.extend(batch)
        token = data.get("token")
        if not token or not batch:
            break

    return {
        "total": data.get("total", len(all_papers)),
        "data": [_normalize_paper(p) for p in all_papers[:limit]],
    }


# ---------------------------------------------------------------------------
# 单篇详情
# ---------------------------------------------------------------------------
def get_paper(
    paper_id: str,
    *,
    fields: str | None = None,
) -> dict[str, Any] | None:
    """按 ID 获取单篇论文详情。

    paper_id 支持多种格式：
    - DOI:       "DOI:10.1103/PhysRevLett.131.066601"
    - arXiv:     "ARXIV:2401.12345"
    - S2 ID:     "649def34f8be52c8b66281af98ae884c09aef38b"
    - Corpus ID: "CorpusId:123456789"
    - URL:       "URL:https://arxiv.org/abs/2401.12345"
    """
    params = {"fields": fields or DEFAULT_FIELDS}
    try:
        data = _get(f"/paper/{paper_id}", params=params)
    except S2APIError as e:
        print(f"[s2] get_paper({paper_id}) failed: {e}")
        return None
    return _normalize_paper(data)


def get_paper_by_doi(doi: str, **kwargs: Any) -> dict[str, Any] | None:
    """便捷方法：按 DOI 获取。"""
    return get_paper(f"DOI:{doi}", **kwargs)


def get_paper_by_arxiv(arxiv_id: str, **kwargs: Any) -> dict[str, Any] | None:
    """便捷方法：按 arXiv ID 获取。"""
    # 去掉版本号（S2 不认 v1/v2）
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    return get_paper(f"ARXIV:{clean_id}", **kwargs)


# ---------------------------------------------------------------------------
# 引用与被引
# ---------------------------------------------------------------------------
def get_citations(
    paper_id: str,
    *,
    fields: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_intents: bool = True,
) -> dict[str, Any]:
    """获取引用该论文的文献（含 citation intent 标签）。

    返回的每条 citation 包含：
    - contexts: 引用上下文句子
    - intents: ["background", "method", "result"] 的子集
    - isInfluential: 是否为实质性引用
    """
    params: dict[str, Any] = {
        "fields": fields or CITATION_FIELDS,
        "limit": min(limit, 1000),
        "offset": offset,
    }
    data = _get(f"/paper/{paper_id}/citations", params=params)
    return {
        "offset": data.get("offset", 0),
        "next": data.get("next"),
        "data": [
            {
                "citing_paper": _normalize_paper(c.get("citingPaper") or {}),
                "contexts": c.get("contexts") or [],
                "intents": c.get("intents") or [],
                "isInfluential": c.get("isInfluential", False),
            }
            for c in (data.get("data") or [])
        ],
    }


def get_references(
    paper_id: str,
    *,
    fields: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """获取该论文的参考文献列表。"""
    params: dict[str, Any] = {
        "fields": fields or CITATION_FIELDS,
        "limit": min(limit, 1000),
        "offset": offset,
    }
    data = _get(f"/paper/{paper_id}/references", params=params)
    return {
        "offset": data.get("offset", 0),
        "next": data.get("next"),
        "data": [
            {
                "cited_paper": _normalize_paper(r.get("citedPaper") or {}),
                "contexts": r.get("contexts") or [],
                "intents": r.get("intents") or [],
                "isInfluential": r.get("isInfluential", False),
            }
            for r in (data.get("data") or [])
        ],
    }


# ---------------------------------------------------------------------------
# 推荐
# ---------------------------------------------------------------------------
def get_recommendations(
    positive_ids: list[str],
    negative_ids: list[str] | None = None,
    *,
    fields: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """基于正/负例论文推荐相关文献（SPECTER2 语义相似度）。

    positive_ids: 你喜欢的论文 ID 列表（DOI:xxx 或 S2 paperId）
    negative_ids: 你不喜欢的论文 ID 列表（可选）
    """
    payload: dict[str, Any] = {
        "positivePaperIds": positive_ids,
        "negativePaperIds": negative_ids or [],
        "fields": fields or RECOMMENDATION_FIELDS,
        "limit": min(limit, 500),
    }
    data = _post("/recommendations/v1/papers/", payload)
    return [_normalize_paper(p) for p in (data.get("recommendedPapers") or [])]


# ---------------------------------------------------------------------------
# 数据规范化
# ---------------------------------------------------------------------------
def _normalize_paper(p: dict[str, Any]) -> dict[str, Any]:
    """把 S2 原始 JSON 精简为本项目统一格式。"""
    ext_ids = p.get("externalIds") or {}
    authors = [
        {"name": a.get("name", ""), "authorId": a.get("authorId", "")}
        for a in (p.get("authors") or [])
    ]
    tldr_obj = p.get("tldr") or {}
    oa_pdf = p.get("openAccessPdf") or {}
    journal = p.get("journal") or {}
    pub_venue = p.get("publicationVenue") or {}

    return {
        "s2_id": p.get("paperId", ""),
        "doi": ext_ids.get("DOI", ""),
        "arxiv_id": ext_ids.get("ArXiv", ""),
        "corpus_id": ext_ids.get("CorpusId"),
        "title": p.get("title", ""),
        "abstract": p.get("abstract") or "",
        "tldr": tldr_obj.get("text", "") if isinstance(tldr_obj, dict) else "",
        "year": p.get("year"),
        "publication_date": p.get("publicationDate", ""),
        "venue": p.get("venue", "") or pub_venue.get("name", ""),
        "journal_name": journal.get("name", "") or p.get("venue", ""),
        "journal_volume": journal.get("volume", ""),
        "journal_pages": journal.get("pages", ""),
        "authors": authors,
        "first_author_last_name": _guess_last_name(authors[0]["name"]) if authors else "",
        "citation_count": p.get("citationCount", 0),
        "influential_citation_count": p.get("influentialCitationCount", 0),
        "reference_count": p.get("referenceCount", 0),
        "is_open_access": p.get("isOpenAccess", False),
        "oa_pdf_url": oa_pdf.get("url", "") if isinstance(oa_pdf, dict) else "",
        "fields_of_study": p.get("fieldsOfStudy") or [],
        "s2_fields_of_study": [
            {"category": f.get("category", ""), "source": f.get("source", "")}
            for f in (p.get("s2FieldsOfStudy") or [])
        ],
        "url": p.get("url", ""),
        "citation_styles": p.get("citationStyles") or {},
        "_raw": p,
    }


def _guess_last_name(full_name: str) -> str:
    if not full_name:
        return ""
    parts = re.split(r"\s+", full_name.strip())
    last = parts[-1] if parts else ""
    return re.sub(r"[^a-zA-Z]", "", last).lower() or (parts[-1].lower() if parts else "")


# ---------------------------------------------------------------------------
# 与 paper_note.md 模板对接
# ---------------------------------------------------------------------------
def paper_to_note_frontmatter(p: dict[str, Any]) -> dict[str, Any]:
    """把 S2 paper dict 转换为 paper_note.md 的 YAML frontmatter 字段。

    注意：S2 不提供 JIF/JCR 分区，这些字段留空由 OpenAlex 或 WoS 补齐。
    """
    authors_names = [a["name"] for a in (p.get("authors") or [])]
    return {
        "title": p.get("title", ""),
        "short_title": " ".join(re.split(r"\W+", p.get("title", ""))[:6]),
        "authors": authors_names,
        "first_author_last_name": p.get("first_author_last_name", ""),
        "corresponding_author": "",
        "year": p.get("year"),
        "publication_date": p.get("publication_date", ""),
        "journal": p.get("journal_name", "") or p.get("venue", ""),
        "publisher": "",
        "volume": p.get("journal_volume", ""),
        "issue": "",
        "pages": p.get("journal_pages", ""),
        "doi": p.get("doi", ""),
        "arxiv_id": p.get("arxiv_id", ""),
        "openalex_id": "",
        "wos_id": "",
        "zotero_key": "",
        "zotero_uri": "",
        "local_pdf_path": "",
        "oa_url": p.get("oa_pdf_url", ""),
        "oa_status": "gold" if p.get("is_open_access") else "closed",

        "cited_by_count": p.get("citation_count"),
        "cited_by_count_normalized": None,
        "jif": None,                    # S2 不提供，由 OpenAlex/WoS 补齐
        "jif_5yr": None,
        "jcr_quartile": "",
        "scimago_quartile": "",
        "citescore": None,
        "esi_highly_cited": False,
        "esi_hot_paper": False,
        "journal_h_index": None,

        # S2 独有字段（附加到 frontmatter）
        "influential_citation_count": p.get("influential_citation_count", 0),
        "tldr": p.get("tldr", ""),
        "s2_id": p.get("s2_id", ""),

        "topics": [f["category"] for f in (p.get("s2_fields_of_study") or [])],
        "methods": [],
        "systems": [],
        "related_to_my_work": None,
        "related_to_my_work_reason": "",

        "status": "unread",
        "my_rating": None,
        "added_date": "",
        "last_reviewed": "",
        "review_count": 0,

        "keywords_auto": p.get("fields_of_study", [])[:5],
    }


# ---------------------------------------------------------------------------
# 便捷：多源融合（S2 + OpenAlex）
# ---------------------------------------------------------------------------
def enrich_from_s2(openalex_work: dict[str, Any]) -> dict[str, Any]:
    """对 openalex_client 返回的 work dict 补充 S2 独有字段。

    补充：tldr, influential_citation_count, s2_id
    若 S2 未配置或查询失败，静默返回原 dict。
    """
    doi = openalex_work.get("doi", "")
    arxiv_id = openalex_work.get("arxiv_id", "")
    if not doi and not arxiv_id:
        return openalex_work

    paper_id = f"DOI:{doi}" if doi else f"ARXIV:{arxiv_id}"
    try:
        s2_paper = get_paper(paper_id, fields="paperId,tldr,influentialCitationCount,citationCount")
    except (S2APIError, S2NotConfigured):
        return openalex_work
    except Exception:
        return openalex_work

    if not s2_paper:
        return openalex_work

    enriched = dict(openalex_work)
    enriched["s2_id"] = s2_paper.get("s2_id", "")
    enriched["tldr"] = s2_paper.get("tldr", "")
    enriched["influential_citation_count"] = s2_paper.get("influential_citation_count", 0)
    return enriched


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Scholar CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search", help="检索论文")
    p_search.add_argument("query", nargs="?", default="exceptional point acoustic metasurface")
    p_search.add_argument("--year", default="2024-2026")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--sort", default=None)

    p_get = sub.add_parser("get", help="获取单篇详情")
    p_get.add_argument("paper_id", help="DOI:xxx / ARXIV:xxx / S2 paperId")

    p_check = sub.add_parser("check", help="检查 API key 配置")

    args = parser.parse_args()

    if args.cmd == "check" or args.cmd is None:
        has_key = bool(settings.semantic_scholar_api_key)
        print(f"semantic_scholar_api_key: {'(set)' if has_key else '(unset)'}")
        print(f"Rate limit: {'100 req/5min (with key)' if has_key else '1 req/s (no key, unstable)'}")
        if not has_key:
            print("\n[!] Request a free key at: https://www.semanticscholar.org/product/api#api-key")

    elif args.cmd == "search":
        res = search_papers(args.query, year_range=args.year, limit=args.limit, sort=args.sort)
        print(f"[s2] total: {res['total']}   returned: {len(res['data'])}")
        for i, p in enumerate(res["data"], 1):
            print(f"\n#{i} [{p['year']}] {p['title']}")
            print(f"   Venue   : {p['venue']}")
            print(f"   DOI     : {p['doi']}   arXiv: {p['arxiv_id']}")
            print(f"   Citations: {p['citation_count']} (influential: {p['influential_citation_count']})")
            print(f"   TLDR    : {p['tldr'][:120] or '(none)'}")
            print(f"   Authors : {', '.join(a['name'] for a in p['authors'][:4])}")

    elif args.cmd == "get":
        p = get_paper(args.paper_id)
        if p:
            print(json.dumps({k: v for k, v in p.items() if k != "_raw"}, indent=2, ensure_ascii=False)[:3000])
        else:
            print(f"Paper not found: {args.paper_id}")
