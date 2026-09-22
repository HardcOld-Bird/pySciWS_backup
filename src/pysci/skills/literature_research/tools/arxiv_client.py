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
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .config import http_session, settings
from .cache_manager import bump_mtime

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_ABS_BASE = "https://arxiv.org/abs"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_SRC_BASE = "https://arxiv.org/e-print"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"

# 用户领域的 arXiv category（订阅列表）
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
        authors.append({
            "name": (name_el.text or "").strip() if name_el is not None else "",
            "affiliation": (aff_el.text or "").strip() if aff_el is not None else "",
        })

    categories = [c.get("term", "") for c in e.findall(f"{ATOM_NS}category")]
    primary_cat_el = e.find(f"{ARXIV_NS}primary_category")
    primary_cat = primary_cat_el.get("term", "") if primary_cat_el is not None else (categories[0] if categories else "")

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
        "first_author_last_name": _guess_last_name(authors[0]["name"]) if authors else "",
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


def _guess_last_name(full_name: str) -> str:
    if not full_name:
        return ""
    parts = re.split(r"\s+", full_name.strip())
    last = parts[-1] if parts else ""
    last = re.sub(r"[^a-zA-Z]", "", last)
    return last.lower() or (parts[-1].lower() if parts else "")


# ---------------------------------------------------------------------------
# 核心：查询
# ---------------------------------------------------------------------------
def search_arxiv(
    query: str,
    *,
    categories: Iterable[str] | None = None,
    max_results: int = 50,
    start: int = 0,
    sort_by: str = "relevance",       # relevance | lastUpdatedDate | submittedDate
    sort_order: str = "descending",   # ascending | descending
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

    from .openalex_client import _cache_key, _read_cache, _write_cache
    cache_path = _cache_key("arxiv_search", params)
    if use_cache:
        cached = _read_cache(cache_path, max_age_seconds=86400)   # arXiv 每日更新，缓存 1 天
        if cached is not None:
            return cached

    with http_session() as s:
        # arXiv 要求 User-Agent 含联系方式
        headers = {}
        if settings.arxiv_user_agent:
            headers["User-Agent"] = settings.arxiv_user_agent
        r = s.get(ARXIV_API_BASE, params=params, headers=headers, timeout=settings.http_timeout)
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

    from .openalex_client import _write_cache as _wc
    _wc(cache_path, {k: v for k, v in result.items() if k != "_raw_xml_len"})
    return result


def get_paper(arxiv_id: str, use_cache: bool = True) -> dict[str, Any] | None:
    """按 arXiv ID 获取单篇论文（支持 'id' 或 'idv2' 形式）。"""
    arxiv_id = arxiv_id.strip()
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id):
        # 可能是老式 ID，如 cond-mat/0601234
        arxiv_id = quote(arxiv_id, safe="/")
    return search_arxiv(f"id:{arxiv_id}", max_results=1, use_cache=use_cache)["entries"][0] \
        if search_arxiv(f"id:{arxiv_id}", max_results=1, use_cache=use_cache)["entries"] else None


def list_recent(
    categories: Iterable[str],
    *,
    max_results: int = 50,
    days_back: int | None = None,
) -> dict[str, Any]:
    """列出指定 category 的最新提交（追踪前沿）。

    days_back: 若指定，只返回最近 N 天内 submitted 的（客户端过滤）
    """
    cats = list(categories)
    cat_expr = " OR ".join(f"cat:{c}" for c in cats)
    result = search_arxiv(
        cat_expr,
        max_results=max_results,
        sort_by="submittedDate",
        sort_order="descending",
        use_cache=False,   # 追踪最新时不用缓存
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
def download_pdf(arxiv_id: str, dest_dir: Path | None = None, overwrite: bool = False) -> Path:
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


def download_source(arxiv_id: str, dest_dir: Path | None = None, overwrite: bool = False) -> Path | None:
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
            r = s.get(url, headers=headers, timeout=settings.http_timeout * 2, stream=True)
            if r.status_code != 200:
                print(f"[arxiv] Source not available for {arxiv_id} (HTTP {r.status_code})")
                return None
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(f"[arxiv] Downloaded source: {dest}  ({dest.stat().st_size / 1024:.1f} KB)")
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
    return {
        "title": e.get("title", ""),
        "short_title": " ".join(re.split(r"\W+", e.get("title", ""))[:6]),
        "authors": [a["name"] for a in e.get("authors", [])],
        "first_author_last_name": e.get("first_author_last_name", ""),
        "corresponding_author": "",
        "year": year,
        "publication_date": e.get("published", ""),
        "journal": e.get("journal_ref", "") or "arXiv preprint",
        "publisher": "arXiv",
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": e.get("doi", ""),
        "arxiv_id": e.get("arxiv_id", ""),
        "openalex_id": "",
        "wos_id": "",
        "zotero_key": "",
        "zotero_uri": "",
        "local_pdf_path": "",
        "oa_url": e.get("pdf_url", ""),
        "oa_status": "green",   # arXiv 属绿色 OA

        "cited_by_count": None,
        "cited_by_count_normalized": None,
        "jif": None,
        "jif_5yr": None,
        "jcr_quartile": "",
        "scimago_quartile": "",
        "citescore": None,
        "esi_highly_cited": False,
        "esi_hot_paper": False,
        "journal_h_index": None,

        "topics": e.get("categories", []),
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
    parser = argparse.ArgumentParser(description="arXiv search CLI")
    parser.add_argument("query", nargs="?", default='all:"exceptional point" AND all:acoustic')
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--sort", default="submittedDate")
    args = parser.parse_args()

    res = search_arxiv(args.query, max_results=args.max_results, sort_by=args.sort)
    print(f"[arxiv] total: {res['total_results']}   returned: {len(res['entries'])}")
    for i, e in enumerate(res["entries"], 1):
        print(f"\n#{i} [{e['published'][:10]}] {e['title']}")
        print(f"   arXiv   : {e['arxiv_id']}v{e['version'][-1] if e['version'] else '1'}  ({e['primary_category']})")
        print(f"   Authors : {', '.join(a['name'] for a in e['authors'][:5])}")
        print(f"   Journal : {e['journal_ref'] or '(preprint)'}   DOI: {e['doi'] or '-'}")
        print(f"   Abs     : {e['abs_url']}")
