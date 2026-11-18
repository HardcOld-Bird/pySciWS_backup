"""refs_bridge：zotero-mcp 导出 JSON → refs.bib 归一化（确定性转换）。

ADR（2026-11-17）：zotero-mcp 只管文献库读写、**不产 .bib**，故需要这段薄胶水
（选型报告估算 100-200 行）。两种输入等价接受，自动识别：

- **Zotero JSON**（``zotero-cli --json`` / zotero-mcp 导出）：``itemType`` / ``creators``
  / ``publicationTitle`` 等 Zotero 字段名；
- **CSL JSON**（Zotero 导出格式 "CSL JSON"）：``type`` / ``author`` / ``container-title``
  / ``issued.date-parts`` 等 CSL 字段名。

设计要点（极简口径）：

- **确定性**：同输入必得同输出（字段顺序固定、key 生成稳定、输出内不写时间戳）；
- **全量重写**：每次调用整体重写目标 refs.bib，不做增量合并（无合并语义即无冲突语义）；
- **key 冲突**：按出现顺序追加 ``a``/``b``/… 后缀去重；
- **转义**：TeX 特殊字符按 BibTeX 字段值惯例转义；标题保大写（值整体花括号包裹）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 条目类型映射：Zotero itemType / CSL type → BibTeX entry type。
_ZOTERO_TYPE_MAP = {
    "journalArticle": "article",
    "conferencePaper": "inproceedings",
    "book": "book",
    "bookSection": "incollection",
    "thesis": "phdthesis",
    "report": "techreport",
    "webpage": "misc",
    "preprint": "misc",
    "manuscript": "misc",
    "presentation": "misc",
    "document": "misc",
}
_CSL_TYPE_MAP = {
    "article-journal": "article",
    "paper-conference": "inproceedings",
    "book": "book",
    "chapter": "incollection",
    "thesis": "phdthesis",
    "report": "techreport",
    "webpage": "misc",
    "article": "misc",
    "manuscript": "misc",
    "speech": "misc",
}

# 输出字段顺序（决定 refs.bib 的稳定形态）；不在表内的字段不输出。
_FIELD_ORDER = (
    "author", "editor", "title", "journal", "booktitle", "publisher", "school",
    "year", "volume", "number", "pages", "doi", "url", "isbn",
)

_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


class RefsError(ValueError):
    """输入 JSON 无法识别（非 Zotero/CSL 条目结构）。"""


def escape_tex(value: str) -> str:
    """转义 BibTeX 字段值里的 TeX 特殊字符（先反斜杠后其余，避免二次转义）。

    Args:
        value: 原始字段值。

    Returns:
        转义后的安全值（换行折叠为空格）。
    """
    text = re.sub(r"\s+", " ", value).strip()
    out = []
    for ch in text:
        out.append(_TEX_ESCAPES.get(ch, ch))
    return "".join(out)


def load_items(path: str | Path) -> list[dict]:
    """读入导出 JSON，返回条目字典列表。

    兼容两种容器：顶层数组，或 ``{"items": [...]}``。

    Args:
        path: JSON 文件路径。

    Returns:
        条目字典列表（不修改原结构）。

    Raises:
        RefsError: JSON 顶层数组之外的结构。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise RefsError(f"无法识别的导出结构（期待条目数组）：{type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def _is_csl(item: dict) -> bool:
    return "itemType" not in item and "type" in item


def _year_of(item: dict, csl: bool) -> str:
    if csl:
        parts = (item.get("issued") or {}).get("date-parts") or []
        if parts and parts[0]:
            return str(parts[0][0])
        return ""
    date = str(item.get("date") or "")
    match = re.search(r"\d{4}", date)
    return match.group(0) if match else ""


def _names_of(item: dict, csl: bool, role: str) -> list[str]:
    """拼装 ``Family, Given`` 作者串；role 为 ``author``/``editor``。"""
    if csl:
        people = item.get(role) or []
        return [
            ", ".join(p for p in (p.get("family"), p.get("given")) if p)
            for p in people
            if isinstance(p, dict) and (p.get("family") or p.get("given"))
        ]
    wanted = "author" if role == "author" else "editor"
    out = []
    for creator in item.get("creators") or []:
        if not isinstance(creator, dict):
            continue
        if creator.get("creatorType", "author") != wanted:
            continue
        if creator.get("name"):  # 机构作者
            out.append(str(creator["name"]))
            continue
        family, given = creator.get("lastName"), creator.get("firstName")
        if family or given:
            out.append(", ".join(p for p in (family, given) if p))
    return out


def normalize_item(item: dict) -> tuple[str, str, dict[str, str]]:
    """把 Zotero/CSL 条目归一为 ``(entry_type, key_hint, fields)``。

    Args:
        item: 单个导出条目（两种格式之一，自动识别）。

    Returns:
        三元组：BibTeX 条目类型、key 生成提示（姓氏+年+标题词）、字段字典
        （键在 :data:`_FIELD_ORDER` 内；值已转义）。

    Raises:
        RefsError: 条目既非 Zotero 也非 CSL 结构。
    """
    csl = _is_csl(item)
    if csl:
        src_type = item.get("type")
    elif "itemType" in item:
        src_type = item.get("itemType")
    else:
        raise RefsError(f"条目既非 Zotero 也非 CSL 结构：{sorted(item)[:6]}")
    type_map = _CSL_TYPE_MAP if csl else _ZOTERO_TYPE_MAP
    entry_type = type_map.get(str(src_type), "misc")

    fields: dict[str, str] = {}
    container = item.get("container-title") if csl else (
        item.get("publicationTitle") or item.get("conferenceName")
    )
    field_source = {
        "title": item.get("title"),
        "journal": container if entry_type == "article" else None,
        "booktitle": container if entry_type in ("inproceedings", "incollection") else None,
        "publisher": item.get("publisher"),
        "school": item.get("publisher") if entry_type == "phdthesis" else None,
        "volume": item.get("volume"),
        "number": str(item["issue"]) if item.get("issue") is not None else None,
        "pages": item.get("page") if csl else item.get("pages"),
        "doi": item.get("DOI") or item.get("doi"),
        "url": item.get("URL") or item.get("url"),
        "isbn": item.get("ISBN") or item.get("isbn"),
    }
    for key, value in field_source.items():
        if value:
            fields[key] = escape_tex(str(value))
    year = _year_of(item, csl)
    if year:
        fields["year"] = year
    authors = _names_of(item, csl, "author")
    if authors:
        fields["author"] = " and ".join(authors)
    editors = _names_of(item, csl, "editor")
    if editors:
        fields["editor"] = " and ".join(editors)

    surname = ""
    if authors:
        surname = authors[0].split(",")[0]
    title_words = re.findall(r"[A-Za-z]+", str(item.get("title") or ""))
    key_hint = f"{surname}|{year}|{title_words[0] if title_words else 'item'}"
    return entry_type, key_hint, fields


def _make_key(hint: str, used: set[str]) -> str:
    """由 ``姓|年|标题词`` 生成 BibTeX key，冲突追加 a/b/… 后缀（确定性）。"""
    parts = hint.split("|")
    base = re.sub(r"[^a-z0-9]", "", "_".join(parts).lower())
    if not base:
        base = "item"
    key, suffix = base, ""
    while key in used:
        suffix = chr(ord(suffix) + 1) if suffix else "a"
        key = f"{base}{suffix}"
    used.add(key)
    return key


def item_to_bibtex(entry_type: str, key: str, fields: dict[str, str]) -> str:
    """渲染单条 BibTeX 条目文本（字段顺序固定）。"""
    lines = [f"@{entry_type}{{{key},"]
    for name in _FIELD_ORDER:
        if name in fields:
            lines.append(f"  {name} = {{{fields[name]}}},")
    lines.append("}")
    return "\n".join(lines)


def write_refs_bib(items: list[dict], out_path: str | Path) -> Path:
    """把条目列表整体重写为目标 refs.bib（排序按 key，确定性输出）。

    Args:
        items: Zotero/CSL 条目列表。
        out_path: 输出路径（父目录自动创建）。

    Returns:
        输出文件路径。

    Raises:
        RefsError: 任一条目无法识别。
    """
    used: set[str] = set()
    entries: list[tuple[str, str]] = []
    for item in items:
        entry_type, hint, fields = normalize_item(item)
        key = _make_key(hint, used)
        entries.append((key, item_to_bibtex(entry_type, key, fields)))
    entries.sort(key=lambda pair: pair[0])
    header = "% generated by pysci-doc refs - do not hand-edit\n\n"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(header + "\n".join(text for _, text in entries) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
