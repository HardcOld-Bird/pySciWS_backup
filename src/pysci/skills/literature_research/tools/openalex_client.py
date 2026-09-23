"""OpenAlex 学术元数据检索客户端。

OpenAlex 是完全免费、CC0 协议的开放学术数据库（https://openalex.org），
覆盖 2.5 亿+ 学术记录，包含引用数、期刊指标、OA 全文链接、参考文献图等。

本模块提供以下能力：
- :func:`search_works`      : 关键词/主题检索，返回结构化 dict 列表
- :func:`get_work`          : 按 DOI 或 OpenAlex ID 获取单篇详情
- :func:`get_source`        : 按 ISSN 或名称获取期刊/来源信息（含 JIF 等）
- :func:`reconstruct_abstract`: 从倒排索引重建摘要文本
- :func:`work_to_note_frontmatter`: 将 work dict 转换为 paper_note.md 的 YAML frontmatter

所有 GET 响应会缓存到 ``literature/cache/api_responses/``，避免重复请求。

用法示例::

    from pysci.skills.literature_research.tools.openalex_client import search_works, get_work

    results = search_works(
        query="exceptional point acoustic active gain",
        year_from=2024, year_to=2026,
        min_citations=5,
        per_page=25,
    )
    for w in results:
        print(w["display_name"], w["cited_by_count"], w["journal"])
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .cache_manager import bump_mtime
from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
OPENALEX_BASE = "https://api.openalex.org"

# 用户领域的常见 OpenAlex concept ID（可扩充）
# 参考: https://api.openalex.org/concepts?search=...
KNOWN_CONCEPTS: dict[str, str] = {
    # 非厄米 / EP
    "non-hermitian": "C121616955",  # 需要核实，占位
    "exceptional-point": "",  # OpenAlex 未直接给出，用 keyword 检索
    # 声学 / 超构
    "acoustic-metamaterial": "",
    "metasurface": "",
    "phononic-crystal": "",
    # 拓扑
    "topological-insulator": "",
    "topological-photonics": "",
    # BIC / CPA
    "bound-state-in-continuum": "",
    "coherent-perfect-absorption": "",
}

# 用户领域常见期刊 ISSN（用于 venue 过滤）
KNOWN_VENUES_ISSN: dict[str, str] = {
    "PRL": "0031-9007",
    "PRX": "2160-3308",
    "PRB": "2469-9950",
    "PRApplied": "2331-7019",
    "Nature Physics": "1745-2473",
    "Nature Communications": "2041-1723",
    "Light: Science & Applications": "2047-7538",
    "Optica": "2334-2536",
    "ACS Photonics": "2330-4022",
    "Photonics Research": "2327-9125",
    "JASA": "0001-4966",
    "Ultrasonics": "0041-624X",
    "Applied Physics Letters": "0003-6951",
    "Advanced Science": "2198-3844",
    "Science Advances": "2375-2548",
}


# ---------------------------------------------------------------------------
# 缓存工具
# ---------------------------------------------------------------------------
def _cache_key(url: str, params: dict[str, Any] | None = None) -> Path:
    """根据 URL + params 生成稳定的缓存文件名。"""
    raw = url + "|" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    # 从 URL 提取一个可读的 slug
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("?")[0])[-60:]
    return settings.cache_api_responses / f"openalex_{slug}_{h}.json"


def _read_cache(path: Path, max_age_seconds: int = 86400 * 7) -> Any | None:
    """读取缓存；若不存在或过期则返回 None。默认 7 天有效期。命中会 touch 更新 mtime（供 LRU）。"""
    if not path.exists():
        return None
    import time

    age = time.time() - path.stat().st_mtime
    if age > max_age_seconds:
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    bump_mtime(path)
    return data


def _write_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 摘要重建（OpenAlex 存的是倒排索引）
# ---------------------------------------------------------------------------
def reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """OpenAlex 的 abstract_inverted_index 是 {word: [positions]} 结构。

    本函数重建为正常英文摘要字符串。
    """
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


# ---------------------------------------------------------------------------
# 核心 API 调用
# ---------------------------------------------------------------------------
def _get(
    url: str, params: dict[str, Any] | None = None, use_cache: bool = True
) -> dict[str, Any]:
    """底层 GET，自动附加 polite pool 邮箱、自动缓存。"""
    params = dict(params or {})
    if settings.openalex_email and "mailto" not in params:
        params["mailto"] = settings.openalex_email

    cache_path = _cache_key(url, params)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    with http_session() as s:
        r = s.get(url, params=params, timeout=settings.http_timeout)
        r.raise_for_status()
        data = r.json()

    _write_cache(cache_path, data)
    return data


# ---------------------------------------------------------------------------
# Work（论文）相关
# ---------------------------------------------------------------------------
def _extract_work_summary(w: dict[str, Any]) -> dict[str, Any]:
    """把 OpenAlex 原始 work JSON 精简为本项目使用的统一结构。"""
    primary_loc = w.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    best_oa = w.get("best_oa_location") or {}

    authors = []
    for a in w.get("authorships", []) or []:
        au = a.get("author") or {}
        insts = [i.get("display_name", "") for i in (a.get("institutions") or [])]
        authors.append(
            {
                "name": au.get("display_name", ""),
                "orcid": au.get("orcid", ""),
                "openalex_id": au.get("id", ""),
                "institutions": insts,
                "is_corresponding": bool(a.get("is_corresponding")),
                "raw_affiliation": a.get("raw_affiliation_string", ""),
            }
        )

    # 抽取 arXiv ID（若存在）
    arxiv_id = ""
    for loc in w.get("locations", []) or []:
        src = loc.get("source") or {}
        if "arxiv" in (src.get("display_name") or "").lower():
            landing = loc.get("landing_page_url") or ""
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", landing)
            if m:
                arxiv_id = m.group(1)
                break
            pdf_url = loc.get("pdf_url") or ""
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", pdf_url)
            if m:
                arxiv_id = m.group(1)
                break

    doi_url = w.get("doi") or ""
    doi = doi_url.replace("https://doi.org/", "") if doi_url else ""

    return {
        "openalex_id": w.get("id", "").replace("https://openalex.org/", ""),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "title": w.get("display_name") or w.get("title") or "",
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date", ""),
        "type": w.get("type", ""),
        "cited_by_count": w.get("cited_by_count", 0),
        "cited_by_percentile_year": (
            (w.get("cited_by_percentile_year") or {}).get("min")
        ),
        "is_oa": (w.get("open_access") or {}).get("is_oa", False),
        "oa_status": (w.get("open_access") or {}).get("oa_status", ""),
        "oa_url": best_oa.get("pdf_url") or best_oa.get("landing_page_url") or "",
        "journal": source.get("display_name", ""),
        "journal_issn_l": source.get("issn_l", ""),
        "journal_openalex_id": (source.get("id") or "").replace(
            "https://openalex.org/", ""
        ),
        "publisher": source.get("host_organization_name", "")
        or (w.get("primary_location") or {}).get("source", {}).get("publisher", ""),
        "volume": (w.get("biblio") or {}).get("volume", ""),
        "issue": (w.get("biblio") or {}).get("issue", ""),
        "first_page": (w.get("biblio") or {}).get("first_page", ""),
        "last_page": (w.get("biblio") or {}).get("last_page", ""),
        "authors": authors,
        "first_author_last_name": _guess_last_name(authors[0]["name"])
        if authors
        else "",
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "concepts": [
            {
                "name": c.get("display_name", ""),
                "id": c.get("id", ""),
                "score": c.get("score", 0.0),
            }
            for c in (w.get("concepts") or [])[:10]
        ],
        "topics": [
            {
                "name": t.get("display_name", ""),
                "id": t.get("id", ""),
                "score": t.get("score", 0.0),
            }
            for t in (w.get("topics") or [])[:5]
        ],
        "referenced_works_count": len(w.get("referenced_works") or []),
        "referenced_works": [
            x.replace("https://openalex.org/", "")
            for x in (w.get("referenced_works") or [])[:20]
        ],
        "related_works": [
            x.replace("https://openalex.org/", "")
            for x in (w.get("related_works") or [])[:10]
        ],
        "counts_by_year": w.get("counts_by_year", []),
        "_raw": w,  # 保留原始 JSON，供特殊需求
    }


def _guess_last_name(full_name: str) -> str:
    """从 'First M. Last' 形式猜测姓氏（末段），做基础清理。"""
    if not full_name:
        return ""
    parts = re.split(r"\s+", full_name.strip())
    if len(parts) == 1:
        return parts[0].lower()
    last = parts[-1]
    # 去声调、去非字母
    last = re.sub(r"[^a-zA-Z]", "", last)
    return last.lower() or parts[-1].lower()


def search_works(
    query: str | None = None,
    *,
    filter_expr: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    min_citations: int | None = None,
    venues: Iterable[str] | None = None,
    concepts: Iterable[str] | None = None,
    oa_only: bool = False,
    type_: str = "article",
    sort: str = "relevance_score:desc",
    per_page: int = 25,
    page: int | None = None,
    cursor: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """按关键词/过滤条件检索论文。

    参数：
        query: 自由文本检索词（会命中 title / abstract / fulltext）
        filter_expr: 手写 OpenAlex filter 表达式（与下面的便捷参数二选一或叠加）
        year_from, year_to: 发表年范围
        min_citations: 最小被引数
        venues: 期刊名列表（会转成 ISSN 过滤；未知名会被忽略）
        concepts: OpenAlex concept ID 列表
        oa_only: 仅 OA
        type_: 文献类型，默认 'article'（可设 'preprint'、'book-chapter' 等）
        sort: 'relevance_score:desc' | 'cited_by_count:desc' | 'publication_date:desc'
        per_page: 每页数量（≤200）
        page: 页码（≤10000 条时可用）；cursor 用于深分页
        cursor: '*' 表示开始 cursor 分页

    返回：
        {"results": [work_summary, ...], "meta": {...}, "next_cursor": str | None}
    """
    filters: list[str] = []
    if filter_expr:
        filters.append(filter_expr)
    if year_from or year_to:
        yr_parts = []
        if year_from:
            yr_parts.append(f"publication_year:>{year_from - 1}")
        if year_to:
            yr_parts.append(f"publication_year:<{year_to + 1}")
        filters.extend(yr_parts)
    if min_citations is not None:
        filters.append(f"cited_by_count:>{min_citations - 1}")
    if venues:
        issns = [KNOWN_VENUES_ISSN[v] for v in venues if v in KNOWN_VENUES_ISSN]
        if issns:
            filters.append("primary_location.source.issn_l:" + "|".join(issns))
    if concepts:
        filters.append("concepts.id:" + "|".join(concepts))
    if oa_only:
        filters.append("is_oa:true")
    if type_:
        filters.append(f"type:{type_}")

    params: dict[str, Any] = {
        "per-page": min(max(per_page, 1), 200),
        "sort": sort,
    }
    if filters:
        params["filter"] = ",".join(filters)
    if query:
        params["search"] = query
    if cursor:
        params["cursor"] = cursor
    elif page:
        params["page"] = page

    data = _get(f"{OPENALEX_BASE}/works", params=params, use_cache=use_cache)

    results = [_extract_work_summary(w) for w in (data.get("results") or [])]
    meta = data.get("meta") or {}
    return {
        "results": results,
        "meta": {
            "count": meta.get("count", 0),
            "db_response_time_ms": meta.get("db_response_time_ms"),
            "page": meta.get("page"),
            "per_page": meta.get("per_page"),
            "next_cursor": meta.get("next_cursor"),
        },
        "next_cursor": meta.get("next_cursor"),
    }


def get_work(
    doi: str | None = None, openalex_id: str | None = None, use_cache: bool = True
) -> dict[str, Any] | None:
    """按 DOI 或 OpenAlex ID 获取单篇论文详情。

    示例::

        get_work(doi="10.1103/PhysRevLett.131.066601")
        get_work(openalex_id="W4388143908")
    """
    if doi:
        url = f"{OPENALEX_BASE}/works/doi:{quote(doi, safe='')}"
    elif openalex_id:
        oid = openalex_id.replace("https://openalex.org/", "").replace("W", "")
        url = f"{OPENALEX_BASE}/works/W{oid}"
    else:
        raise ValueError("Must provide either doi or openalex_id")

    try:
        data = _get(url, use_cache=use_cache)
    except Exception as e:  # 404 等
        print(f"[openalex] get_work failed for {doi or openalex_id}: {e}")
        return None
    return _extract_work_summary(data)


# ---------------------------------------------------------------------------
# Source（期刊）相关
# ---------------------------------------------------------------------------
def get_source(
    issn: str | None = None,
    name: str | None = None,
    openalex_id: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """按 ISSN、名称或 OpenAlex ID 获取期刊元数据（含 JIF 近似值、h-index、OA 政策等）。"""
    if issn:
        url = f"{OPENALEX_BASE}/sources/issn:{issn}"
    elif openalex_id:
        sid = openalex_id.replace("https://openalex.org/", "").replace("S", "")
        url = f"{OPENALEX_BASE}/sources/S{sid}"
    elif name:
        # 名称检索：走 sources?search=...
        data = _get(
            f"{OPENALEX_BASE}/sources",
            params={"search": name, "per-page": 5},
            use_cache=use_cache,
        )
        results = data.get("results") or []
        if not results:
            return None
        return _extract_source(results[0])
    else:
        raise ValueError("Must provide issn, name, or openalex_id")

    try:
        data = _get(url, use_cache=use_cache)
    except Exception as e:
        print(f"[openalex] get_source failed: {e}")
        return None
    return _extract_source(data)


def _extract_source(s: dict[str, Any]) -> dict[str, Any]:
    stats = s.get("summary_stats") or {}
    return {
        "openalex_id": (s.get("id") or "").replace("https://openalex.org/", ""),
        "display_name": s.get("display_name", ""),
        "alternate_titles": s.get("alternate_titles", []),
        "issn_l": s.get("issn_l", ""),
        "issn": s.get("issn", []),
        "publisher": s.get("host_organization_name", ""),
        "type": s.get("type", ""),
        "is_oa": s.get("is_oa", False),
        "is_in_doaj": s.get("is_in_doaj", False),
        "apc_usd": (s.get("apc_usd") or 0),
        "country_code": s.get("country_code", ""),
        # 关键指标
        "h_index": stats.get("h_index"),
        "works_count": stats.get("works_count"),
        "cited_by_count": stats.get("cited_by_count"),
        "2yr_mean_citedness": stats.get("2yr_mean_citedness"),  # ≈ JIF
        "_raw": s,
    }


# ---------------------------------------------------------------------------
# 与 paper_note.md 模板对接
# ---------------------------------------------------------------------------
def work_to_note_frontmatter(w: dict[str, Any]) -> dict[str, Any]:
    """把 work_summary 转换为可直接写入 markdown YAML frontmatter 的 dict。

    上层（未来的 skill 或脚本）可以此为基础，加入 AI 生成的评价字段。
    """
    source_stats = {}
    if w.get("journal_openalex_id"):
        src = get_source(openalex_id=w["journal_openalex_id"])
        if src:
            source_stats = src

    authors_names = [a["name"] for a in w.get("authors", [])]
    corresponding = next(
        (a["name"] for a in w.get("authors", []) if a.get("is_corresponding")), ""
    )

    jif = source_stats.get("2yr_mean_citedness")
    # OpenAlex 不提供官方 JCR 分区，此处留空由 wos_client 补齐
    return {
        "title": w.get("title", ""),
        "short_title": _make_short_title(w.get("title", "")),
        "authors": authors_names,
        "first_author_last_name": w.get("first_author_last_name", ""),
        "corresponding_author": corresponding,
        "year": w.get("publication_year"),
        "publication_date": w.get("publication_date", ""),
        "journal": w.get("journal", ""),
        "publisher": w.get("publisher", ""),
        "volume": w.get("volume", ""),
        "issue": w.get("issue", ""),
        "pages": _format_pages(w.get("first_page", ""), w.get("last_page", "")),
        "doi": w.get("doi", ""),
        "arxiv_id": w.get("arxiv_id", ""),
        "openalex_id": w.get("openalex_id", ""),
        "wos_id": "",
        "zotero_key": "",
        "zotero_uri": "",
        "local_pdf_path": "",
        "oa_url": w.get("oa_url", ""),
        "oa_status": w.get("oa_status", ""),
        "cited_by_count": w.get("cited_by_count"),
        "cited_by_count_normalized": w.get("cited_by_percentile_year"),
        "jif": round(jif, 2) if isinstance(jif, (int, float)) else None,
        "jif_5yr": None,
        "jcr_quartile": "",
        "scimago_quartile": "",
        "citescore": None,
        "esi_highly_cited": False,
        "esi_hot_paper": False,
        "journal_h_index": source_stats.get("h_index"),
        "topics": [],
        "methods": [],
        "systems": [],
        "related_to_my_work": None,
        "related_to_my_work_reason": "",
        "status": "unread",
        "my_rating": None,
        "added_date": "",
        "last_reviewed": "",
        "review_count": 0,
        "keywords_auto": [
            c["name"].lower() for c in (w.get("concepts") or [])[:5] if c.get("name")
        ],
    }


def _format_pages(first: str, last: str) -> str:
    if first and last:
        return f"{first}-{last}"
    return first or last or ""


def _make_short_title(title: str, max_words: int = 6) -> str:
    """从长标题里截取前若干实词，用于文件命名与目录显示。"""
    if not title:
        return ""
    stop = {
        "on",
        "of",
        "the",
        "in",
        "a",
        "an",
        "for",
        "and",
        "to",
        "with",
        "via",
        "from",
    }
    words = [w for w in re.split(r"\W+", title) if w and w.lower() not in stop]
    return " ".join(words[:max_words]) if words else title[:40]


# ---------------------------------------------------------------------------
# CLI: 简单检索测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAlex search CLI")
    parser.add_argument(
        "query", nargs="?", default="exceptional point acoustic active gain"
    )
    parser.add_argument("--year-from", type=int, default=2023)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--min-citations", type=int, default=None)
    parser.add_argument("--per-page", type=int, default=10)
    parser.add_argument("--sort", default="relevance_score:desc")
    args = parser.parse_args()

    res = search_works(
        query=args.query,
        year_from=args.year_from,
        year_to=args.year_to,
        min_citations=args.min_citations,
        per_page=args.per_page,
        sort=args.sort,
    )
    print(f"[openalex] total matches: {res['meta']['count']}")
    for i, w in enumerate(res["results"], 1):
        print(f"\n#{i} [{w['publication_year']}] {w['title']}")
        print(f"   Journal : {w['journal']}  |  OA: {w['oa_status']}")
        print(f"   DOI     : {w['doi']}  |  arXiv: {w['arxiv_id']}")
        print(f"   Cited by: {w['cited_by_count']}")
        print(f"   Authors : {', '.join(a['name'] for a in w['authors'][:5])}")
