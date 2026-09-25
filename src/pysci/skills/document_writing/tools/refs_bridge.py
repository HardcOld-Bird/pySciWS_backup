"""refs_bridge —— 把 Zotero 文献库导出为 LaTeX 可用的 refs.bib。

复用 literature_research 的 ZoteroBridge 读取条目（本地 API 优先、Web API 兜底），
再确定性地转成 BibTeX。citekey 采用 Better BibTeX 风格（首作者姓+年+标题首词），
与文献调研 skill 入库的条目无缝衔接——调研时存进 Zotero，写作时一键拉成 .bib。

说明：本模块跨 skill 复用 ``pysci.skills.literature_research.tools.zotero_bridge``
（延迟导入 + 清晰报错），避免重复实现 Zotero 客户端。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Zotero itemType → BibTeX 条目类型
_TYPE_MAP = {
    "journalArticle": "article",
    "conferencePaper": "inproceedings",
    "book": "book",
    "bookSection": "incollection",
    "thesis": "phdthesis",
    "report": "techreport",
    "preprint": "misc",
    "manuscript": "unpublished",
    "patent": "patent",
    "webpage": "misc",
    "document": "misc",
}

_STOPWORDS = {
    "the", "a", "an", "on", "of", "in", "for", "and", "to", "with", "via",
    "from", "by", "at", "toward", "towards", "non", "nonhermitian",
}


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class BibExportResult:
    bibtex: str
    count: int
    out_path: Path | None = None
    citekeys: list[str] = field(default_factory=list)
    backend: str = ""


# ---------------------------------------------------------------------------
# citekey 与字段清洗
# ---------------------------------------------------------------------------
def _strip_latex_unsafe(s: str) -> str:
    """转义 BibTeX 中易出问题的字符（保守处理，不动数学符号 $）。"""
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _year_of(data: dict[str, Any]) -> str:
    date = str(data.get("date") or data.get("issued") or "")
    m = re.search(r"(\d{4})", date)
    return m.group(1) if m else ""


def make_citekey(data: dict[str, Any]) -> str:
    """Better BibTeX 风格 citekey：首作者姓 + 年 + 标题首个实词（全小写）。"""
    creators = data.get("creators") or []
    last = ""
    for c in creators:
        if not isinstance(c, dict):
            continue
        last = c.get("lastName") or c.get("name") or ""
        if last:
            break
    last = re.sub(r"[^\w]", "", last).lower() or "anon"

    year = _year_of(data) or "nd"

    title = str(data.get("title") or "")
    word = ""
    for w in re.split(r"[^\w]+", title.lower()):
        if w and w not in _STOPWORDS:
            word = w
            break
    word = word or "untitled"

    return f"{last}{year}{word}"


def _authors_field(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for c in data.get("creators") or []:
        if not isinstance(c, dict):
            continue
        if c.get("lastName"):
            first = c.get("firstName") or ""
            parts.append(f"{c['lastName']}, {first}".strip().rstrip(","))
        elif c.get("name"):
            parts.append(c["name"])
    return " and ".join(parts)


def _extra_field(data: dict[str, Any], key: str) -> str:
    """从 extra（形如 'arxiv_id: 2301.x\\njif: 3.2'）里取某键的值。"""
    extra = str(data.get("extra") or "")
    m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", extra, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# 单条目 → BibTeX
# ---------------------------------------------------------------------------
def item_to_bibtex(item: dict[str, Any], *, citekey: str | None = None) -> str:
    """把一个 Zotero 条目（{key, data} 或扁平 data）转成 BibTeX 字符串。"""
    data = item.get("data", item) if isinstance(item, dict) else {}
    key = citekey or make_citekey(data)
    itype = data.get("itemType", "")
    bibtype = _TYPE_MAP.get(itype, "misc")

    fields: dict[str, str] = {}
    author = _authors_field(data)
    if author:
        fields["author"] = author
    if data.get("title"):
        fields["title"] = _strip_latex_unsafe(str(data["title"]))
    year = _year_of(data)
    if year:
        fields["year"] = year

    if bibtype == "article":
        if data.get("publicationTitle"):
            fields["journal"] = str(data["publicationTitle"])
        for src, dst in (("volume", "volume"), ("issue", "number"), ("pages", "pages")):
            if data.get(src):
                fields[dst] = str(data[src])
    elif bibtype == "inproceedings":
        if data.get("proceedingsTitle"):
            fields["booktitle"] = str(data["proceedingsTitle"])
        if data.get("place"):
            fields["address"] = str(data["place"])
    elif bibtype == "book":
        if data.get("publisher"):
            fields["publisher"] = str(data["publisher"])
    elif bibtype == "phdthesis":
        if data.get("university"):
            fields["school"] = str(data["university"])
    elif bibtype == "techreport":
        if data.get("institution"):
            fields["institution"] = str(data["institution"])

    # 预印本 / arXiv
    if itype == "preprint" or _extra_field(data, "arxiv_id"):
        arxiv = _extra_field(data, "arxiv_id") or str(data.get("archiveID") or "")
        arxiv = arxiv.replace("arXiv:", "").strip()
        if arxiv:
            fields["eprint"] = arxiv
            fields["archivePrefix"] = "arXiv"
            fields["primaryClass"] = _extra_field(data, "primary_category") or ""
        if data.get("repository"):
            fields["howpublished"] = str(data["repository"])

    if data.get("DOI"):
        fields["doi"] = str(data["DOI"])
    if data.get("url"):
        fields["url"] = str(data["url"])
    if data.get("abstractNote"):
        abstract = str(data["abstractNote"]).replace("\n", " ").strip()
        if len(abstract) > 1200:
            abstract = abstract[:1200].rstrip() + "..."
        fields["abstract"] = _strip_latex_unsafe(abstract)

    # 去掉空值
    fields = {k: v for k, v in fields.items() if v not in (None, "")}

    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@{bibtype}{{{key},\n{body}\n}}"


# ---------------------------------------------------------------------------
# 批量导出
# ---------------------------------------------------------------------------
def _get_bridge() -> Any:
    try:
        from pysci.skills.literature_research.tools import zotero_bridge
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "无法导入 literature_research.zotero_bridge；refs 导出依赖文献调研 skill。"
        ) from e
    return zotero_bridge.ZoteroBridge()


def export_bib(
    *,
    collection: str | None = None,
    tag: str | None = None,
    query: str | None = None,
    keys: list[str] | None = None,
    limit: int = 200,
    out_path: str | Path | None = None,
    dedupe: bool = True,
) -> BibExportResult:
    """按 collection / tag / query / keys 拉取 Zotero 条目并导出 BibTeX。

    至少提供其中一个筛选条件；keys 优先级最高（精确导出若干条）。
    """
    bridge = _get_bridge()

    items: list[dict[str, Any]] = []
    if keys:
        for k in keys:
            it = bridge.get_item(k)
            if it:
                items.append(it)
    elif query:
        items = bridge.search_items(query, limit=limit)
    else:
        items = bridge.list_items(
            limit=limit, tag=tag, collection=collection
        )

    seen: set[str] = set()
    entries: list[str] = []
    citekeys: list[str] = []
    for i, it in enumerate(items):
        data = it.get("data", it) if isinstance(it, dict) else {}
        if data.get("itemType") in ("attachment", "note"):
            continue
        key = make_citekey(data)
        if dedupe:
            # 同 key 追加序号避免冲突
            base, n = key, 2
            while key in seen:
                key = f"{base}{chr(96 + n)}"  # a/b/c...
                n += 1
            seen.add(key)
        entries.append(item_to_bibtex(it, citekey=key))
        citekeys.append(key)

    header = (
        "% 由 pysci document_writing refs_bridge 从 Zotero 自动生成。\n"
        "% 请勿手改；改 Zotero 后重新运行 `compose tex refs` 刷新。\n\n"
    )
    bibtex = header + "\n\n".join(entries) + ("\n" if entries else "")

    outp: Path | None = None
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(bibtex, encoding="utf-8")

    return BibExportResult(
        bibtex=bibtex,
        count=len(entries),
        out_path=outp,
        citekeys=citekeys,
        backend=getattr(bridge, "backend", ""),
    )


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zotero → refs.bib 导出")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--keys", default=None, help="逗号分隔的条目 key")
    parser.add_argument("--out", default=None, help="输出 .bib 路径")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    res = export_bib(
        collection=args.collection,
        tag=args.tag,
        query=args.query,
        keys=[k.strip() for k in args.keys.split(",")] if args.keys else None,
        limit=args.limit,
        out_path=args.out,
    )
    print(f"[refs] backend={res.backend} 条目={res.count}")
    if res.out_path:
        print(f"[refs] 写入：{res.out_path}")
    print(res.bibtex[:2000])
