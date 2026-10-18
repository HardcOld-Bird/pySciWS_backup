"""OA 副本回退：Semantic Scholar 元数据 → arXiv PDF / Europe PMC 全文 XML。

`pysci-research read` 在出版商页面失败（Cloudflare / 付费墙）时，用本模块再试一次
**开放获取副本**——这两条通道都不受出版商 Cloudflare 影响：

- **Semantic Scholar**（公开 API，无需 key）按 DOI 给出 ``externalIds.ArXiv`` /
  ``externalIds.PubMedCentral`` 与 ``openAccessPdf``；
- **arXiv** 预印本 → 直接下载 PDF（回到既有 LaTeX/PDF 抽取链）；
- **Europe PMC** → ``fullTextXML`` → JATS 正文转 Markdown（OA 综述/PMC 收录文章）。

失败一律 fail-soft（返回空结果 + 一行原因），不阻塞 `read` 的失败诊断与退出码。
S2 无 key 时限流较严（429 常见）：本模块每篇只查一次，拿到 429 就跳过并留痕。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests

from .config import settings

#: Semantic Scholar 单篇查询（DOI 精确）。
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
S2_FIELDS = "title,externalIds,openAccessPdf"
#: Europe PMC 全文 XML（PMC OA 子集；不受 NCBI 网页 403 影响）。
EUROPEPMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_PDF_MAGIC = b"%PDF-"


def semantic_scholar(doi: str, *, timeout: float | None = None) -> dict[str, Any]:
    """按 DOI 查 S2 的 OA 读数；429/网络错误/无记录 → ``{}``（fail-soft）。"""
    doi = str(doi or "").strip()
    if not doi:
        return {}
    try:
        r = requests.get(
            S2_URL.format(doi=doi),
            params={"fields": S2_FIELDS},
            headers={"User-Agent": _UA},
            timeout=timeout or settings.http_timeout,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
    except (requests.RequestException, ValueError):
        return {}
    ids = data.get("externalIds") or {}
    arxiv = str(ids.get("ArXiv") or "").strip()
    pmcid = str(ids.get("PubMedCentral") or "").strip()
    oa = data.get("openAccessPdf") or {}
    return {
        "title": str(data.get("title") or ""),
        "arxiv_id": arxiv,
        "pmcid": f"PMC{pmcid}" if pmcid and not pmcid.startswith("PMC") else pmcid,
        "oa_url": str(oa.get("url") or ""),
        "oa_status": str(oa.get("status") or ""),
    }


def _download_pdf(url: str, dest: Path, *, timeout: float | None = None) -> Path | None:
    """普通 GET 取 PDF（验魔数），失败 → ``None``。"""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"},
            timeout=timeout or settings.http_timeout * 2,
            stream=True,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return None
        chunks = r.iter_content(chunk_size=65536)
        head = b""
        for chunk in chunks:
            head += chunk or b""
            if len(head) >= len(_PDF_MAGIC):
                break
        if not head.startswith(_PDF_MAGIC):
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("wb") as fh:
            fh.write(head)
            for chunk in chunks:
                fh.write(chunk or b"")
        tmp.replace(dest)
        return dest
    except (requests.RequestException, OSError):
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def jats_to_markdown(xml_text: str) -> str:
    """JATS（Europe PMC 全文 XML）→ Markdown（标题/摘要/分节正文/图注）。

    只取正文语义块，不保留原始标签；解析失败返回空串（由调用方 fail-soft）。
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    def first(name: str) -> ET.Element | None:
        for el in root.iter():
            if _local(el.tag) == name:
                return el
        return None

    title = _text(first("article-title")) or _text(first("title"))
    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    abstract = first("abstract")
    if abstract is not None:
        lines.append("## Abstract\n")
        for p in abstract.iter():
            if _local(p.tag) == "p":
                text = _text(p)
                if text:
                    lines.append(text + "\n")
    body = first("body")
    if body is not None:

        def walk(el: ET.Element, level: int) -> None:
            for child in el:
                tag = _local(child.tag)
                if tag == "sec":
                    st = None
                    for sub in child:
                        if _local(sub.tag) == "title":
                            st = _text(sub)
                            break
                    if st:
                        lines.append(f"{'#' * min(level, 6)} {st}\n")
                    walk(child, level + 1)
                elif tag == "p":
                    text = _text(child)
                    if text:
                        lines.append(text + "\n")
                elif tag == "fig":
                    cap = None
                    for sub in child:
                        if _local(sub.tag) == "caption":
                            cap = _text(sub)
                            break
                    if cap:
                        lines.append(f"**Figure**: {cap}\n")
                elif tag in ("table-wrap",):
                    cap = None
                    for sub in child:
                        if _local(sub.tag) == "caption":
                            cap = _text(sub)
                            break
                    if cap:
                        lines.append(f"**Table**: {cap}\n")

        walk(body, 2)
    return "\n".join(lines).strip() + ("\n" if lines else "")


def europepmc_markdown(pmcid: str, *, timeout: float | None = None) -> str:
    """Europe PMC 全文 XML → Markdown；非 200/解析失败 → 空串。"""
    pmcid = str(pmcid or "").strip()
    if not pmcid:
        return ""
    try:
        r = requests.get(
            EUROPEPMC_XML.format(pmcid=pmcid),
            headers={"User-Agent": _UA},
            timeout=timeout or settings.http_timeout,
        )
        if r.status_code != 200:
            return ""
        return jats_to_markdown(r.text)
    except (requests.RequestException, ValueError):
        return ""


def try_oa(
    doi: str,
    *,
    title: str = "",
    pdf_dest: Path,
    timeout: float | None = None,
) -> tuple[str, Path | None, str]:
    """尝试 OA 副本；返回 ``(markdown, pdf_path, 说明)``。

    顺序：arXiv 预印本 → Europe PMC 全文 XML → S2 的 openAccessPdf 直链。
    S2 查询一次；429/无记录时只给说明（不重试、不阻塞）。
    """
    meta = semantic_scholar(doi, timeout=timeout)
    if not meta:
        return "", None, f"OA 回退：Semantic Scholar 未给出副本（DOI={doi}）"
    basis = f"OA 回退（S2：{meta.get('oa_status') or '?'}）"

    arxiv_id = str(meta.get("arxiv_id") or "").strip()
    if arxiv_id:
        pdf = _download_pdf(
            f"https://arxiv.org/pdf/{arxiv_id}", pdf_dest, timeout=timeout
        )
        if pdf is not None:
            return "", pdf, f"{basis}：arXiv {arxiv_id} 预印本 PDF"

    pmcid = str(meta.get("pmcid") or "").strip()
    if pmcid:
        md = europepmc_markdown(pmcid, timeout=timeout)
        if md:
            return md, None, f"{basis}：Europe PMC {pmcid} 全文 XML"

    oa_url = str(meta.get("oa_url") or "").strip()
    if oa_url:
        pdf = _download_pdf(oa_url, pdf_dest, timeout=timeout)
        if pdf is not None:
            return "", pdf, f"{basis}：openAccessPdf 直链"
        return "", None, f"{basis}：OA 直链不可达（{oa_url}）"
    return "", None, f"{basis}：无 arXiv/PMC/OA 直链"
