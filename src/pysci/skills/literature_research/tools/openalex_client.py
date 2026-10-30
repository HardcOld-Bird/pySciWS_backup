"""OpenAlex 学术元数据检索客户端。

OpenAlex 是完全免费、CC0 协议的开放学术数据库（https://openalex.org），
覆盖 2.5 亿+ 学术记录，包含引用数、期刊指标、OA 全文链接、参考文献图等。

本模块提供以下能力：
- :func:`search_works`      : 关键词/主题检索，返回结构化 dict 列表
- :func:`get_work`          : 按 DOI 或 OpenAlex ID 获取单篇详情
- :func:`get_source`        : 按 ISSN 或名称获取期刊/来源信息（含 JIF 等）
- :func:`reconstruct_abstract`: 从倒排索引重建摘要文本
- :func:`work_to_note_frontmatter`: 将 work dict 转换为 paper_note.md 的 YAML frontmatter

所有 GET 响应会缓存到 ``data/skills/literature_research/cache/api_responses/``，避免重复请求。

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

import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import journal_metrics
from .cache_manager import (
    API_CACHE_MAX_AGE_SECONDS,
    cache_key,
    read_cache,
    write_cache,
)
from .config import http_session, settings
from .notes import derive_journal_tier, normalize_last_name

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
OPENALEX_BASE = "https://api.openalex.org"

# 用户领域的常见 OpenAlex concept ID（可扩充）
# 参考: https://api.openalex.org/concepts?search=...
#
# 状态：预留，当前无调用方（全仓库仅此一处定义，无测试）。
# 9 个键里 8 个是空串，唯一有值的 "non-hermitian" 自己也注着「需要核实」。保留的理由：
# OpenAlex 按 concept 过滤比 keyword 检索准得多，真要收窄检索时这张表就是落点；而那些
# 空串**不是没填完的坑**，它们记录的是「OpenAlex 没有直接对应这个主题的 concept」（见
# "exceptional-point" 那行的注释），删掉就丢了这条已查证过的结论，下次得重新查一遍。
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
# 缓存工具（delegating shim）
# ---------------------------------------------------------------------------
# 下面三个函数已提级为 :mod:`cache_manager` 的公开 API（``cache_key`` / ``read_cache`` /
# ``write_cache``）。保留这些**私有名**作为一行委托，是为了不扰动本模块内部的调用点
# 与现有测试的 patch 目标（``monkeypatch.setattr(oa, "_read_cache", ...)``）；而跳模块的
# 调用方（``arxiv_client`` / ``citation_verify``）已改为直接用公开 API，不再以私有名
# 深入别的模块。新增代码请直接用 :mod:`cache_manager` 的名字。


def _cache_key(url: str, params: dict[str, Any] | None = None) -> Path:
    """根据 URL + params 生成稳定的缓存文件名（委托 :func:`cache_manager.cache_key`）。"""
    return cache_key(url, params, prefix="openalex")


def _read_cache(
    path: Path, max_age_seconds: int = API_CACHE_MAX_AGE_SECONDS
) -> Any | None:
    """读缓存；不存在或过期则返回 None（委托 :func:`cache_manager.read_cache`）。

    签名故意**不**做成 keyword-only：与提级前的形态一致，既有调用点两种写法都能用。
    """
    return read_cache(path, max_age_seconds=max_age_seconds)


def _write_cache(path: Path, data: Any) -> None:
    """写缓存（委托 :func:`cache_manager.write_cache`）。"""
    write_cache(path, data)


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
    """底层 GET，自动附加 API key（若配置）与 ``mailto``、自动缓存。

    OpenAlex 自 2026-02-13 起改为「免费注册 + 按用量计费」：配额由 key 决定，不再是
    polite pool 模型（``mailto`` 照旧附加——无害，但别再指望它抬配额）。无 key 时每天
    只有 **$0.10** 的用量预算（官方定位 testing/demos），免费 key 提到 **$1/天**（10×）。
    计价按调用形态而非请求数：按 ID/DOI 取单条**免费且不限量**，list/filter 每千次
    $0.10，search 每千次 $1——故 $0.10 约等于 1000 次 list 或 100 次 search。
    若 .env 配置了 ``OPENALEX_API_KEY``，以 ``Authorization: Bearer <key>`` 头注入
    ——放在 header 而非 query，避免 key 进入缓存文件名与 URL（_cache_key 只对
    url+params 取哈希）。
    """
    params = dict(params or {})
    if settings.openalex_email and "mailto" not in params:
        params["mailto"] = settings.openalex_email

    headers: dict[str, str] = {}
    if settings.openalex_api_key:
        headers["Authorization"] = f"Bearer {settings.openalex_api_key}"

    cache_path = _cache_key(url, params)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    with http_session() as s:
        r = s.get(
            url, params=params, headers=headers or None, timeout=settings.http_timeout
        )
        r.raise_for_status()
        data = r.json()

    _write_cache(cache_path, data)
    return data


# ---------------------------------------------------------------------------
# Work（论文）相关
# ---------------------------------------------------------------------------
def _extract_work_summary(w: dict[str, Any], *, max_refs: int = 100) -> dict[str, Any]:
    """把 OpenAlex 原始 work JSON 精简为本项目使用的统一结构。

    ``max_refs`` 控制 ``referenced_works`` 保留的条数（``referenced_works_count`` 始终是
    **未截断**的真实总数）。默认 100 而非旧值 20：滚雪球（``research citegraph
    --direction backward``）要拿这些 id 去批量取回被引工作，而一篇 PRL 级论文的参考
    文献表常有 40-60 条，20 条上限会静默丢掉一半以上的引文网络。``_raw`` 本就保留了
    完整原始 JSON，因此提高这个上限**不增加缓存体积**，只多占一点内存里的短字符串。

    同时就地派生 ``listed_in`` / ``journal_tier`` / ``journal_tier_basis``：OpenAlex 把
    ``primary_location.source`` **内联**在 work 响应里，其中就带专家评议名单，因此
    这一步是**零额外请求**的（实测 2024 Nature Physics 的内联 source 给出
    ``cwts-core,jufo-3,ki-jl-2,norway-2``，与单独 ``GET /sources/S…`` 的结果一致）。
    于是 ``search`` / ``citegraph`` / ``get`` 这三条**不查 source** 的路径也能带上期刊
    档次——否则展示层的 tier 尾注会恒空，等于没做。
    """
    # 负数会让切片变成「去掉末尾 N 条」而非「不保留」，故先归一。
    max_refs = 100 if max_refs is None else max(0, int(max_refs))
    primary_loc = w.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    best_oa = w.get("best_oa_location") or {}

    listed_in = source.get("listed_in") or []
    journal_tier, journal_tier_basis = derive_journal_tier(listed_in)

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
        "journal_issn": source.get("issn") or [],
        "listed_in": listed_in,
        "journal_tier": journal_tier,
        "journal_tier_basis": journal_tier_basis,
        # 两个键取自**同一个** source 对象（函数开头已安全地取成 ``{}``）。原实现绕过它
        # 重新挖一遍原始 JSON，而 ``.get("source", {})`` 在 ``primary_location`` 存在但
        # ``source`` 为 **null** 时返回的是 None 而不是 ``{}``（键存在，默认值不生效），
        # 于是紧跟着的 ``.get("publisher")`` 抛 AttributeError。无期刊的纯预印本正是
        # 这种形态（``"primary_location": {"source": null}``），也就是说这条崩溃路径
        # 会在最普通的预印本上触发，而不是在罕见畸形数据上。
        "publisher": source.get("host_organization_name", "")
        or source.get("publisher", ""),
        "volume": (w.get("biblio") or {}).get("volume", ""),
        "issue": (w.get("biblio") or {}).get("issue", ""),
        "first_page": (w.get("biblio") or {}).get("first_page", ""),
        "last_page": (w.get("biblio") or {}).get("last_page", ""),
        "authors": authors,
        "first_author_last_name": normalize_last_name(authors[0]["name"])
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
            for x in (w.get("referenced_works") or [])[:max_refs]
        ],
        "related_works": [
            x.replace("https://openalex.org/", "")
            for x in (w.get("related_works") or [])[:10]
        ],
        "counts_by_year": w.get("counts_by_year", []),
        "_raw": w,  # 保留原始 JSON，供特殊需求
    }


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
        # 必须是 stderr：``research get --json`` / ``citegraph --json`` 把 JSON 写在
        # stdout，一行提示混进去会让调用方的 ``json.loads(stdout)`` 直接失败。
        print(
            f"[openalex] get_work failed for {doi or openalex_id}: {e}",
            file=sys.stderr,
        )
        return None
    return _extract_work_summary(data)


# ---------------------------------------------------------------------------
# 引文图谱（滚雪球）：批量取回 + 前向引用
# ---------------------------------------------------------------------------
#: 合法的 OpenAlex work id 形态。只接受这一种形态是有意的：``works_by_ids`` 把 id 用
#: ``|`` 拼进 filter，一个畸形 id（比如误传的 DOI）会让**整批**请求返回 400，而不是只
#: 丢掉它自己。
_OPENALEX_ID_RE = re.compile(r"^W\d{5,}$")

#: OpenAlex 对单个 filter 值里 ``|`` 分隔项数量的上限（超过会 400）。
MAX_IDS_PER_REQUEST = 50


def _norm_openalex_id(raw: Any) -> str:
    """把 ``https://openalex.org/W123`` / ``w123`` / ``123`` 统一成 ``W123``。

    归一化后仍不匹配 :data:`_OPENALEX_ID_RE` 的一律返回空串，由调用方跳过。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"^https?://(?:www\.)?openalex\.org/", "", s).strip("/").upper()
    if not s.startswith("W"):
        s = f"W{s}"
    return s if _OPENALEX_ID_RE.match(s) else ""


def works_by_ids(
    ids: Iterable[str], *, batch_size: int = MAX_IDS_PER_REQUEST, use_cache: bool = True
) -> list[dict[str, Any]]:
    """按 OpenAlex id 批量取回 work_summary（滚雪球的 **backward** 方向）。

    用 ``filter=openalex_id:W1|W2|...`` 一次取多篇，而不是逐篇 :func:`get_work`。
    这么做的理由在 2026-02-13 计价改版后**变了，但结论没变**：按 ID 取单条现在免费且
    不限量，而 list/filter 每千次 $0.10，故批量在配额上反而略贵（1 次 list ≈ $0.0001
    对 50 次单条 = $0）；真正省下的是 50 次 HTTP 往返的延迟与超时风险。
    ``batch_size`` 硬上限 :data:`MAX_IDS_PER_REQUEST`。

    降级与契约（调用方**不得**假定一一对应）：

    - 非法/重复 id 直接跳过（不污染整批请求）；
    - 某一批网络失败只跳过该批并留一行 stderr，其余批照常返回；
    - OpenAlex 查不到的 id（已合并/删除的记录）不会出现在结果里，
      因此**返回列表长度通常小于输入**。
    """
    clean: list[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        oid = _norm_openalex_id(raw)
        if oid and oid not in seen:
            seen.add(oid)
            clean.append(oid)
    if not clean:
        return []

    size = min(max(1, int(batch_size)), MAX_IDS_PER_REQUEST)
    out: list[dict[str, Any]] = []
    for i in range(0, len(clean), size):
        chunk = clean[i : i + size]
        params: dict[str, Any] = {
            "filter": "openalex_id:" + "|".join(chunk),
            "per-page": len(chunk),
        }
        try:
            data = _get(f"{OPENALEX_BASE}/works", params=params, use_cache=use_cache)
        except Exception as e:  # 一批失败不该拖垮整个图谱
            print(
                f"[openalex] works_by_ids 第 {i // size + 1} 批（{len(chunk)} 个 id）失败："
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        out.extend(_extract_work_summary(w) for w in (data.get("results") or []))
    return out


def works_citing(
    openalex_id: str,
    *,
    per_page: int = 25,
    year_from: int | None = None,
    year_to: int | None = None,
    sort: str = "cited_by_count:desc",
    use_cache: bool = True,
) -> dict[str, Any]:
    """取「引用了某篇论文」的工作（滚雪球的 **forward** 方向）。

    用 ``filter=cites:{id}``。返回结构与 :func:`search_works` 完全一致
    （``{"results", "meta", "next_cursor"}``），因此上层能复用同一套去重 / 展示 / 落盘逻辑。

    **不**按 ``type:article`` 过滤：前向引用里预印本、综述、书章都是真实信号，过滤掉会
    让「这篇论文有没有被跟进」的判断失真（这正是 forward 方向要回答的问题）。

    与降级路径不同，id 非法时**抛 ValueError**：这是调用方的输入错误而非环境故障，
    静默返回空集会被误读成「无人引用这篇论文」。
    """
    oid = _norm_openalex_id(openalex_id)
    if not oid:
        raise ValueError(
            "works_citing 需要 OpenAlex work id（形如 W2789790776）；"
            "只有 DOI / arXiv id 时请先用 get_work 解析。"
        )
    filters = [f"cites:{oid}"]
    if year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    if year_to:
        filters.append(f"publication_year:<{year_to + 1}")
    params: dict[str, Any] = {
        "filter": ",".join(filters),
        "per-page": min(max(int(per_page), 1), 200),
        "sort": sort,
    }
    try:
        data = _get(f"{OPENALEX_BASE}/works", params=params, use_cache=use_cache)
    except Exception as e:
        print(
            f"[openalex] works_citing({oid}) 失败：{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {"results": [], "meta": {"count": 0}, "next_cursor": None}

    meta = data.get("meta") or {}
    return {
        "results": [_extract_work_summary(w) for w in (data.get("results") or [])],
        "meta": {
            "count": meta.get("count", 0),
            "db_response_time_ms": meta.get("db_response_time_ms"),
            "page": meta.get("page"),
            "per_page": meta.get("per_page"),
            "next_cursor": meta.get("next_cursor"),
        },
        "next_cursor": meta.get("next_cursor"),
    }


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
        print(f"[openalex] get_source failed: {e}", file=sys.stderr)
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
        # 专家评议名单（JUFO / Norway / KI-JL / CWTS / ERIH+ / MEDLINE …）。
        # 旧实现把它整个丢掉，于是期刊档次只剩引用类指标可用——对声学这类低引用
        # 密度领域严重失真（JASA 的 2yr_mean_citedness 只有 0.82，但 JUFO 判它顶级）。
        # 零额外请求：本字段就在同一份 source 响应里。
        "listed_in": s.get("listed_in") or [],
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
    # OpenAlex 不提供官方 JCR 分区；jcr_quartile 留空待 WoS Journals API，
    # scimago_quartile 由 journal_metrics 按 ISSN 查本地 SCImago 索引填。
    # journal_tier 则由 listed_in 的专家评议名单派生（见 notes.derive_journal_tier）。
    #
    # 两处都**回落到 work dict 自带的内联字段**：``get_source`` 要一次额外请求，无 key
    # 降级态下预算极小（$0.10/天）、限流更紧，多一次往返就多一分失败概率；而内联的
    # ``listed_in`` / ``journal_issn`` 总在且零请求。
    # 不回落的话，降级态下笔记的档次与 SCImago 分区会静默全空——而这恰恰是最需要
    # 免费指标层的时刻。tier 与 basis 在这里**重算**而不是直接取 ``w["journal_tier"]``：
    # 纯函数重算的成本可忽略，但能保证 tier / basis / listed_in 三者永远自洽（不会一个
    # 来自 source_stats、另一个来自内联而对不上）。
    listed_in = source_stats.get("listed_in") or w.get("listed_in") or []
    tier, tier_basis = derive_journal_tier(listed_in)
    scimago_quartile = journal_metrics.quartile_for(
        source_stats.get("issn") or w.get("journal_issn") or []
    )
    return {
        "title": w.get("title", ""),
        "short_title": _make_short_title(w.get("title", "")),
        "authors": authors_names,
        "first_author_last_name": w.get("first_author_last_name", ""),
        "corresponding_author": corresponding,
        "year": w.get("publication_year"),
        # 统一为 YYYY-MM-DD：OpenAlex 本就给 date-only，但防御性截断使它与 arXiv 的
        # ISO-8601 带时分秒形态归一，同一篇论文不会因源不同而在笔记里写出两种值。
        "publication_date": (w.get("publication_date") or "")[:10],
        "journal": w.get("journal", ""),
        # OpenAlex 不提供引文串；本键只为与 arXiv 源的 frontmatter 契约对齐（见 arxiv_client）。
        "journal_ref": "",
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
        "scimago_quartile": scimago_quartile,
        "citescore": None,
        # None = **未知**；False = 「已确认不是」。ESI 高被引/热点名单只能由 WoS Journals
        # API 给出（Starter API 不提供），写 False 等于在数据里断言一件我们无从知道的事。
        "esi_highly_cited": None,
        "esi_hot_paper": None,
        "journal_h_index": source_stats.get("h_index"),
        "journal_tier": tier,
        "journal_tier_basis": tier_basis,
        "listed_in": listed_in,
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
    """把 OpenAlex ``biblio`` 的 first/last page 合成 frontmatter 的 ``pages``。

    ``first == last`` 时必须只写一次：电子刊（APS 全系、Nature 系）用的是**文章号**而不是
    页码区间，OpenAlex 把同一个号同时填进两个字段，于是旧实现产出
    ``124501-124501``——一个看着像页码区间、实则把同一个文章号写了两遍的值。它比空着
    更糟：空值是可辨识的「没数据」，而 ``124501-124501`` 看着像个真区间，会一路传下去：
    ``research add`` 经 :func:`.zotero_cli.create_item_from_metadata` 把它写进 Zotero 条目的
    ``pages``，document_writing 的 ``refs_bridge.item_to_bibtex`` 再把**那个**字段映射成
    BibTeX 的 ``pages``，最终印在参考文献里。实测 10.1103/PhysRevLett.121.124501 正是这个形态。
    """
    if first and last and first != last:
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

    print(
        "[openalex_client] 调试后门——日常检索请走 "
        "`research search '<query>' --source openalex`，取单篇用 `research get <id>`。"
        "本入口只在单独排查 OpenAlex 本身时用（filter 语法、credits 配额与 429、"
        "字段映射对不对）；它绕过 research 的 _row() 投影，故看到的字段与 CLI 不同。",
        file=sys.stderr,
    )

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
