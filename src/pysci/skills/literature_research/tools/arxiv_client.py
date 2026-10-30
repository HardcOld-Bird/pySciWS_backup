"""arXiv 预印本检索客户端。

arXiv 是物理学前沿的必经之路——凝聚态、量子光学、声学超材料等领域的最新工作
90% 先在 arXiv 挂出，比期刊早 6–18 个月。

官方 API 文档: https://info.arxiv.org/help/api/index.html
- 无需 API key
- 需要在 User-Agent 中提供联系方式
- 单次最多 2000 条，推荐 per_page ≤ 100
- 速率限制：每 3 秒一次请求（本模块通过缓存 + 会话重用遵守）

本模块提供：
- :func:`search_arxiv`    : 按 query 检索
- :func:`get_paper`       : 按 arXiv ID 获取
- :func:`list_recent`     : 按 category 列出最新提交（追踪前沿）
- :func:`download_pdf`    : 下载 PDF 到 cache/pdfs/
- :func:`download_source` : 下载 LaTeX 源码（对物理论文极有价值，比 PDF 解析更准）
- :func:`parse_journal_ref` : 把自由文本 ``journal_ref`` 拆成刊名/卷/页/年
- :func:`arxiv_to_note_frontmatter` : 转成 paper_note 的 frontmatter
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .cache_manager import bump_mtime, cache_key, read_cache, write_cache
from .config import http_session, settings
from .notes import normalize_last_name

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_ABS_BASE = "https://arxiv.org/abs"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_SRC_BASE = "https://arxiv.org/e-print"

#: 尚未正式发表的 arXiv 预印本，其 ``journal`` 字段用的占位刊名。
#:
#: 下游（``research._attach_journal_metrics``）靠它识别「这不是一本真期刊」而跳过期刊指标
#: 查询，因此两边必须引用同一常量，不得各写一份字面量。
ARXIV_PREPRINT_JOURNAL = "arXiv preprint"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"

# 用户领域的 arXiv category（订阅列表）
#
# 状态：预留，当前无调用方。设计意图是给 :func:`list_recent` 当默认订阅表，但
# ``list_recent`` 自己也没有调用方（``research`` CLI 未暴露「追踪前沿」这个动作），
# 两者是一对。保留的理由：arXiv 的 category 词表本身稳定，而这张表的价值在于它
# 已经按用户领域（声学 / 非厄米 / 拓扑 / BIC）从全量 taxonomy 里筛过一遍。
KNOWN_CATEGORIES: dict[str, str] = {
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "physics.app-ph": "Applied Physics",
    "physics.optics": "Optics",
    "physics.class-ph": "Classical Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "quant-ph": "Quantum Physics",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
}


# ---------------------------------------------------------------------------
# 底层：XML → dict
# ---------------------------------------------------------------------------
def _parse_entry(e: ET.Element) -> dict[str, Any]:
    """将 Atom <entry> 解析为统一 dict。"""

    def _text(tag: str, ns: str = ATOM_NS) -> str:
        el = e.find(f"{ns}{tag}")
        return (el.text or "").strip() if el is not None else ""

    arxiv_id_url = _text("id")
    # e.g. http://arxiv.org/abs/2606.12345v2
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?", arxiv_id_url)
    arxiv_id = m.group(1) if m else arxiv_id_url.rsplit("/", 1)[-1]
    version = (m.group(2) or "v1") if m else "v1"

    authors = []
    for a in e.findall(f"{ATOM_NS}author"):
        name_el = a.find(f"{ATOM_NS}name")
        aff_el = a.find(f"{ATOM_NS}affiliation")
        authors.append(
            {
                "name": (name_el.text or "").strip() if name_el is not None else "",
                "affiliation": (aff_el.text or "").strip()
                if aff_el is not None
                else "",
            }
        )

    categories = [c.get("term", "") for c in e.findall(f"{ATOM_NS}category")]
    primary_cat_el = e.find(f"{ARXIV_NS}primary_category")
    primary_cat = (
        primary_cat_el.get("term", "")
        if primary_cat_el is not None
        else (categories[0] if categories else "")
    )

    # 期刊引用与 DOI（若已发表）
    journal_ref = _text("journal_ref", ns=ARXIV_NS)
    doi = _text("doi", ns=ARXIV_NS)
    comment = _text("comment", ns=ARXIV_NS)

    # 链接
    pdf_url = ""
    for link in e.findall(f"{ATOM_NS}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}{version}"

    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": re.sub(r"\s+", " ", _text("title")),
        "abstract": re.sub(r"\s+", " ", _text("summary")),
        "authors": authors,
        "first_author_last_name": normalize_last_name(authors[0]["name"])
        if authors
        else "",
        "published": _text("published"),
        "updated": _text("updated"),
        "primary_category": primary_cat,
        "categories": categories,
        "journal_ref": journal_ref,
        "doi": doi,
        "comment": comment,
        "abs_url": f"{ARXIV_ABS_BASE}/{arxiv_id}{version}",
        "pdf_url": pdf_url,
        "src_url": f"{ARXIV_SRC_BASE}/{arxiv_id}{version}",
    }


# ---------------------------------------------------------------------------
# 核心：查询
# ---------------------------------------------------------------------------
def search_arxiv(
    query: str,
    *,
    categories: Iterable[str] | None = None,
    max_results: int = 50,
    start: int = 0,
    sort_by: str = "relevance",  # relevance | lastUpdatedDate | submittedDate
    sort_order: str = "descending",  # ascending | descending
    use_cache: bool = True,
) -> dict[str, Any]:
    """按 query 检索 arXiv。

    query 支持 arXiv 的字段语法，例如::

        search_arxiv('all:"exceptional point" AND cat:cond-mat.mes-hall')
        search_arxiv('ti:nonhermitian AND abs:acoustic')
        search_arxiv('au:Zhang AND cat:physics.app-ph')

    常用字段：ti(标题) / abs(摘要) / au(作者) / cat(分类) / rn(报告号) / all(全部)
    """
    params: dict[str, Any] = {
        "search_query": query,
        "start": start,
        "max_results": min(max_results, 2000),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    if categories:
        # 追加 cat: 过滤（若 query 里没有）
        cats = list(categories)
        if cats and "cat:" not in query:
            cat_expr = " OR ".join(f"cat:{c}" for c in cats)
            params["search_query"] = f"({query}) AND ({cat_expr})"

    cache_path = cache_key("search", params, prefix="arxiv")
    if use_cache:
        # arXiv 每日更新，故用 1 天而不是 Tier B 的默认 7 天
        cached = read_cache(cache_path, max_age_seconds=86400)
        if cached is not None:
            return cached

    with http_session() as s:
        # arXiv 要求 User-Agent 含联系方式
        headers = {}
        if settings.arxiv_user_agent:
            headers["User-Agent"] = settings.arxiv_user_agent
        r = s.get(
            ARXIV_API_BASE,
            params=params,
            headers=headers,
            timeout=settings.http_timeout,
        )
        r.raise_for_status()
        raw_xml = r.text

    root = ET.fromstring(raw_xml)
    entries = [_parse_entry(e) for e in root.findall(f"{ATOM_NS}entry")]
    total_el = root.find(f"{OPENSEARCH_NS}totalResults")
    total = int(total_el.text or 0) if total_el is not None else len(entries)

    result = {
        "query": params["search_query"],
        "total_results": total,
        "start": start,
        "entries": entries,
        "_raw_xml_len": len(raw_xml),
    }

    write_cache(cache_path, {k: v for k, v in result.items() if k != "_raw_xml_len"})
    return result


def get_paper(arxiv_id: str, use_cache: bool = True) -> dict[str, Any] | None:
    """按 arXiv ID 获取单篇论文（支持 'id' 或 'idv2' 形式）。"""
    arxiv_id = arxiv_id.strip()
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id):
        # 可能是老式 ID，如 cond-mat/0601234
        arxiv_id = quote(arxiv_id, safe="/")
    # 原实现在**条件与分支里各调一次** ``search_arxiv()``。有 Tier B 缓存兜底时那只是
    # 两次函数调用 + 两次 XML 解析 + 两次 dict 构造；但 ``use_cache=False``（``read --refresh``）
    # 时它是**两次真实的 arXiv 请求**，而 arXiv 对高频请求会返 429。``get_paper`` 正是
    # ``read`` 路径的必经之处，所以这个重复值得消除。
    entries = search_arxiv(f"id:{arxiv_id}", max_results=1, use_cache=use_cache)[
        "entries"
    ]
    return entries[0] if entries else None


def list_recent(
    categories: Iterable[str],
    *,
    max_results: int = 50,
    days_back: int | None = None,
) -> dict[str, Any]:
    """状态：预留，当前无调用方（``research`` CLI 未暴露「追踪前沿」动作）。

    列出指定 category 的最新提交（追踪前沿）。默认订阅表见 :data:`KNOWN_CATEGORIES`，
    它同样处于预留状态——两者应当一起被接进 CLI，或者一起被删除。

    days_back: 若指定，只返回最近 N 天内 submitted 的（客户端过滤）
    """
    cats = list(categories)
    cat_expr = " OR ".join(f"cat:{c}" for c in cats)
    result = search_arxiv(
        cat_expr,
        max_results=max_results,
        sort_by="submittedDate",
        sort_order="descending",
        use_cache=False,  # 追踪最新时不用缓存
    )
    if days_back:
        cutoff = time.time() - days_back * 86400
        filtered = []
        for e in result["entries"]:
            try:
                pub_ts = time.mktime(time.strptime(e["published"][:10], "%Y-%m-%d"))
                if pub_ts >= cutoff:
                    filtered.append(e)
            except (ValueError, OverflowError):
                continue
        result["entries"] = filtered
    return result


# ---------------------------------------------------------------------------
# 下载：PDF / LaTeX 源码
# ---------------------------------------------------------------------------
def download_pdf(
    arxiv_id: str, dest_dir: Path | None = None, overwrite: bool = False
) -> Path:
    """下载 arXiv PDF 到 cache/pdfs/{arxiv_id}.pdf。"""
    dest_dir = dest_dir or settings.cache_pdfs
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    if dest.exists() and not overwrite:
        bump_mtime(dest)  # 命中 touch，使 mtime≈最近访问（供 prune LRU）
        print(f"[arxiv] PDF already cached: {dest}")
        return dest

    url = f"{ARXIV_PDF_BASE}/{arxiv_id}"
    headers = {}
    if settings.arxiv_user_agent:
        headers["User-Agent"] = settings.arxiv_user_agent
    with http_session() as s:
        r = s.get(url, headers=headers, timeout=settings.http_timeout * 2, stream=True)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    print(f"[arxiv] Downloaded PDF: {dest}  ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def download_source(
    arxiv_id: str, dest_dir: Path | None = None, overwrite: bool = False
) -> Path | None:
    """下载 arXiv LaTeX 源码 tar.gz（若有）到 cache/pdfs/{arxiv_id}_src.tar.gz。

    LaTeX 源码对物理论文极有价值：公式、图表 caption、参考文献都比 PDF 解析更准。
    """
    dest_dir = dest_dir or settings.cache_pdfs
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{arxiv_id.replace('/', '_')}_src.tar.gz"
    if dest.exists() and not overwrite:
        return dest

    url = f"{ARXIV_SRC_BASE}/{arxiv_id}"
    headers = {}
    if settings.arxiv_user_agent:
        headers["User-Agent"] = settings.arxiv_user_agent
    try:
        with http_session() as s:
            r = s.get(
                url, headers=headers, timeout=settings.http_timeout * 2, stream=True
            )
            if r.status_code != 200:
                print(
                    f"[arxiv] Source not available for {arxiv_id} (HTTP {r.status_code})"
                )
                return None
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(
            f"[arxiv] Downloaded source: {dest}  ({dest.stat().st_size / 1024:.1f} KB)"
        )
        return dest
    except Exception as e:
        print(f"[arxiv] Failed to download source for {arxiv_id}: {e}")
        return None


def extract_tex_from_source(tar_path: Path) -> str | None:
    """从 tar.gz 中提取主 .tex 文件内容（启发式：找含 \\documentclass 的 .tex）。"""
    import tarfile

    if not tar_path.exists():
        return None
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            candidates = []
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if not m.name.lower().endswith(".tex"):
                    continue
                try:
                    content = tf.extractfile(m).read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                # 主文件的启发式判断
                score = 0
                if r"\documentclass" in content:
                    score += 10
                if r"\begin{document}" in content:
                    score += 10
                if r"\bibliography" in content or r"\begin{thebibliography}" in content:
                    score += 3
                score += len(content) / 10000.0
                candidates.append((score, m.name, content))
            if not candidates:
                return None
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][2]
    except (tarfile.TarError, OSError) as e:
        print(f"[arxiv] Failed to extract {tar_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# journal_ref 自由文本 → 结构化书目字段
# ---------------------------------------------------------------------------
#: arXiv ``journal_ref`` 的主模式：``刊名 卷, 页 (年)``。
#:
#: ``journal`` 用惰性 ``.+?``，使正则优先拿**最短**的前缀当刊名、把第一个数字串当卷号
#: （刊名本身可含数字，如 ``2D Materials 5, 031001 (2018)``）。``page`` 的字符类含
#: ``()``，故缺空格的 ``085117(2018)`` 也能靠回溯正确拆开。
_JOURNAL_REF_FULL_RE = re.compile(
    r"^\s*(?P<journal>.+?)\s*(?P<volume>\d+)\s*,\s*(?P<page>[\w\-.()]+)?\s*\((?P<year>\d{4})\)"
)

#: 退化模式：``刊名 卷, 页``（无年份括号），如 ``"Phys. Rev. B 96, 085117"``。
_JOURNAL_REF_NOYEAR_RE = re.compile(
    r"^\s*(?P<journal>.+?)\s*(?P<volume>\d+)\s*,\s*(?P<page>[\w\-.()]+)?\s*$"
)


def parse_journal_ref(ref: str | None) -> dict[str, Any]:
    """把 arXiv 的自由文本 ``journal_ref`` 拆成结构化书目字段。

    arXiv 的 ``journal_ref`` 是作者自填的一整串引文（``"Phys. Rev. Lett. 121, 124501
    (2018)"``）。旧实现把它整个塞进 frontmatter 的 ``journal`` 键，于是 ``INDEX.md`` 的
    Journal 列、笔记的 Journal-tier 段落都变成一串引文，而所有按**刊名**做的期刊指标
    查询（``get_source(name=...)``）全部落空。

    Args:
        ref: arXiv 返回的原始 ``journal_ref``，可为 ``None`` / 空串。

    Returns:
        ``{"journal", "volume", "pages", "year", "journal_ref"}``：

        * ``journal`` —— 纯刊名；拆不出结构时**回落为原串**（宁可粗糙也不丢信息）
        * ``volume`` / ``pages`` —— 字符串或 ``None``（保持源数据的文本形态，不数值化：
          页码可以是 ``eabn7905``、``44-48`` 这类非数字串）
        * ``year`` —— ``int`` 或 ``None``
        * ``journal_ref`` —— 原串（空白折叠后）。拆解是**有损**的，留着原文才能审计。

    空输入 → 四个字段全空、``journal_ref`` 为 ``""``，**不抛异常**。
    """
    raw = re.sub(r"\s+", " ", str(ref or "")).strip()
    out: dict[str, Any] = {
        "journal": "",
        "volume": None,
        "pages": None,
        "year": None,
        "journal_ref": raw,
    }
    if not raw:
        return out
    m = _JOURNAL_REF_FULL_RE.match(raw) or _JOURNAL_REF_NOYEAR_RE.match(raw)
    if m:
        groups = m.groupdict()
        out["journal"] = (groups.get("journal") or "").strip()
        out["volume"] = groups.get("volume")
        out["pages"] = (groups.get("page") or "").strip() or None
        # 退化模式无 year 组，用 groupdict 取值避开 IndexError
        year = groups.get("year")
        out["year"] = int(year) if year else None
    out["journal"] = out["journal"] or raw
    return out


# ---------------------------------------------------------------------------
# 与 openalex_client 的对接
# ---------------------------------------------------------------------------
def arxiv_to_note_frontmatter(e: dict[str, Any]) -> dict[str, Any]:
    """把 arXiv entry 转换为可直接写入 paper_note.md 的 frontmatter 字段。"""
    year = None
    if e.get("published"):
        try:
            year = int(e["published"][:4])
        except ValueError:
            pass
    jr = parse_journal_ref(e.get("journal_ref"))
    return {
        "title": e.get("title", ""),
        "short_title": " ".join(re.split(r"\W+", e.get("title", ""))[:6]),
        "authors": [a["name"] for a in e.get("authors", [])],
        "first_author_last_name": e.get("first_author_last_name", ""),
        "corresponding_author": "",
        # arXiv 的 submitted 年份优先：它决定文件名 {year}_{last}_{slug}.md，改用出版年会
        # 让跨年发表（如 2017-12 投稿、2018-01 见刊）的笔记改名，破坏既有 wiki 链接。
        # 只在 published 缺失时才回落 journal_ref 里的出版年。
        "year": year if year is not None else jr["year"],
        # arXiv 给 ISO-8601 带时分秒（"2018-03-12T04:08:19Z"），OpenAlex 给 date-only。
        # 统一截断为 YYYY-MM-DD，否则同一篇论文在两个源下写出两种值，merge 时无法对齐。
        "publication_date": (e.get("published") or "")[:10],
        "journal": jr["journal"] or ARXIV_PREPRINT_JOURNAL,
        "journal_ref": jr["journal_ref"],
        "publisher": "arXiv",
        "volume": jr["volume"] or "",
        "issue": "",
        "pages": jr["pages"] or "",
        "doi": e.get("doi", ""),
        "arxiv_id": e.get("arxiv_id", ""),
        "openalex_id": "",
        "wos_id": "",
        "zotero_key": "",
        "zotero_uri": "",
        "local_pdf_path": "",
        "oa_url": e.get("pdf_url", ""),
        "oa_status": "green",  # arXiv 属绿色 OA
        "cited_by_count": None,
        "cited_by_count_normalized": None,
        "jif": None,
        "jif_5yr": None,
        "jcr_quartile": "",
        "scimago_quartile": "",
        "citescore": None,
        # None = **未知**；False = 「已确认不是」。旧值 False 对真正的高被引论文是数据里的
        # 谎言（ESI 需 WoS Journals API，当前无程序化来源）。
        "esi_highly_cited": None,
        "esi_hot_paper": None,
        "journal_h_index": None,
        # 期刊档次与 SCImago 分区都靠期刊记录（OpenAlex source）才能得出，而 arXiv 的
        # Atom 响应里根本没有。故这里一律留空，由 research._attach_journal_metrics 在
        # 拿到 source 后回填——两个转换器的**键集**必须一致，否则 merge 时会出现单源独有键。
        "journal_tier": "",
        "journal_tier_basis": [],
        "listed_in": [],
        # topics 是留给 AI 填的**研究主题**标签（如 non-hermitian / exceptional-point），
        # 旧实现把它与 keywords_auto 都等于 arXiv categories，两键完全重复；现在只留后者。
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
        "keywords_auto": e.get("categories", [])[:5],
    }


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    print(
        "[arxiv_client] 调试后门——日常检索请走 "
        "`research search '<query>' --source arxiv`。"
        "本入口只在单独排查 arXiv API 本身时用（字段语法、限速/429、Atom 解析）。",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="arXiv search CLI")
    parser.add_argument(
        "query", nargs="?", default='all:"exceptional point" AND all:acoustic'
    )
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--sort", default="submittedDate")
    args = parser.parse_args()

    res = search_arxiv(args.query, max_results=args.max_results, sort_by=args.sort)
    print(f"[arxiv] total: {res['total_results']}   returned: {len(res['entries'])}")
    for i, e in enumerate(res["entries"], 1):
        print(f"\n#{i} [{e['published'][:10]}] {e['title']}")
        print(
            f"   arXiv   : {e['arxiv_id']}v{e['version'][-1] if e['version'] else '1'}  ({e['primary_category']})"
        )
        print(f"   Authors : {', '.join(a['name'] for a in e['authors'][:5])}")
        print(
            f"   Journal : {e['journal_ref'] or '(preprint)'}   DOI: {e['doi'] or '-'}"
        )
        print(f"   Abs     : {e['abs_url']}")
