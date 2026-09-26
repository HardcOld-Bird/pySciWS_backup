"""COMSOL 手册知识管线：MinerU 批量转换 + SQLite FTS5 章节索引 + 检索/按节读取。

把本机 COMSOL 安装附带的 PDF 手册（``config.PRIORITY_MANUALS``）转成保真 Markdown
（公式→LaTeX），落到 ``data/skills/comsol_simulation/docs/*.md``，再按标题切章建
**FTS5** 全文索引（Python 内置 ``sqlite3``，零新依赖）到 ``cache/doc_index.db``。

工作流::

    docs.convert_all()          # MinerU 批量转换核心手册（大文件自动分块，可后台跑）
    docs.build_index()          # 切章 + 建 FTS5 索引
    docs.search("perfectly matched layer")   # 命中：手册/章节/页码/摘录
    docs.read("COMSOL_ProgrammingReferenceManual", heading="...")  # 直接读保真 Markdown

转换复用 :mod:`pysci.skills.literature_research.tools.pdf_extract` 的 ``extract_pdf``
（已内置 MinerU 云端 + >200 页自动分块重拼 + 缓存）。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

# ---------------------------------------------------------------------------
# 转换（MinerU 批量）
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_CHUNK_RE = re.compile(r"<!--\s*={3,}.*?pages\s+([0-9]+)-([0-9]+).*?={3,}\s*-->")


def _doc_stem(rel_or_path: str | Path) -> str:
    """手册的文档名（索引主键）= PDF 文件名去后缀。"""
    return Path(rel_or_path).stem


def _resolve_manual(rel_or_path: str | Path) -> Path | None:
    """把 PRIORITY_MANUALS 的相对路径或绝对路径解析为实际存在的 PDF。"""
    p = Path(rel_or_path)
    if p.is_absolute():
        return p if p.exists() else None
    return settings.manual_path(str(rel_or_path))


def convert(
    rel_or_path: str | Path,
    *,
    force: bool = False,
    backend: str | None = None,
) -> Path:
    """转换单本手册为 Markdown 并写入 ``docs/<stem>.md``，返回输出路径。

    Args:
        rel_or_path: ``PRIORITY_MANUALS`` 中的相对路径，或绝对 PDF 路径。
        force: 已存在输出且非 force 时跳过（增量转换）。
        backend: 透传给 extract_pdf（None → settings 默认，MinerU 云端优先）。
    """
    from pysci.skills.literature_research.tools.pdf_extract import extract_pdf

    pdf = _resolve_manual(rel_or_path)
    if pdf is None:
        raise FileNotFoundError(f"手册 PDF 不存在：{rel_or_path}")
    out = settings.docs_dir / f"{_doc_stem(pdf)}.md"
    if out.exists() and not force:
        return out

    t0 = time.time()
    md = extract_pdf(pdf, backend=backend)
    out.write_text(md, encoding="utf-8")
    print(f"[comsol.docs] converted {pdf.name} -> {out} ({len(md)} chars, {time.time()-t0:.1f}s)")
    return out


def convert_all(
    *,
    force: bool = False,
    backend: str | None = None,
    manuals: list[str] | None = None,
) -> dict[str, Path | Exception]:
    """批量转换手册集（默认 ``PRIORITY_MANUALS`` 中本机存在的）。单本失败不影响其他。"""
    rels = manuals if manuals is not None else list(_priority_rel_paths())
    out: dict[str, Path | Exception] = {}
    for rel in rels:
        key = _doc_stem(rel)
        try:
            out[key] = convert(rel, force=force, backend=backend)
        except Exception as e:  # noqa: BLE001
            print(f"[comsol.docs] WARNING: 转换 {rel} 失败：{e}")
            out[key] = e
    return out


def _priority_rel_paths() -> list[str]:
    from .config import PRIORITY_MANUALS

    return [rel for rel in PRIORITY_MANUALS if _resolve_manual(rel) is not None]


# ---------------------------------------------------------------------------
# 切章
# ---------------------------------------------------------------------------
@dataclass
class Section:
    doc: str
    heading: str
    level: int
    pages: str
    body: str


def split_sections(md_text: str, doc: str) -> list[Section]:
    """按 Markdown 标题切章：每个标题到下一个标题（任意层级）为一个 section。

    同时跟踪 MinerU 分块注释里的页范围，作为 section 的近似页码上下文。
    """
    sections: list[Section] = []
    cur: Section | None = None
    pages = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal cur, buf
        if cur is not None:
            cur.body = "\n".join(buf).strip()
            sections.append(cur)
        cur = None
        buf = []

    for line in md_text.splitlines():
        m = _CHUNK_RE.search(line)
        if m:
            pages = f"{m.group(1)}-{m.group(2)}"
            continue
        h = _HEADING_RE.match(line)
        if h:
            flush()
            cur = Section(doc=doc, heading=h.group(2), level=len(h.group(1)), pages=pages, body="")
        elif cur is not None:
            buf.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# FTS5 索引
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS sections USING fts5(
    doc UNINDEXED,
    pages UNINDEXED,
    level UNINDEXED,
    heading,
    body
);
CREATE TABLE IF NOT EXISTS docmeta(
    doc TEXT PRIMARY KEY,
    n_sections INTEGER,
    built_at TEXT
);
"""


def _connect() -> sqlite3.Connection:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.doc_index_db))
    conn.executescript(_SCHEMA)
    return conn


def build_index(docs: list[str] | None = None, *, rebuild: bool = True) -> int:
    """对 ``docs/*.md`` 切章并写入 FTS5 索引，返回写入的 section 总数。

    Args:
        docs: 仅重建这些文档名（stem）；None → 全部 docs/*.md。
        rebuild: True → 先删除目标文档的旧条目（幂等重建）。
    """
    md_files = sorted(settings.docs_dir.glob("*.md"))
    targets = [p for p in md_files if docs is None or p.stem in docs]
    conn = _connect()
    total = 0
    try:
        for p in targets:
            doc = p.stem
            sections = split_sections(p.read_text(encoding="utf-8", errors="replace"), doc)
            if rebuild:
                conn.execute("DELETE FROM sections WHERE doc = ?", (doc,))
                conn.execute("DELETE FROM docmeta WHERE doc = ?", (doc,))
            conn.executemany(
                "INSERT INTO sections(doc, pages, level, heading, body) VALUES (?,?,?,?,?)",
                [(s.doc, s.pages, str(s.level), s.heading, s.body) for s in sections],
            )
            conn.execute(
                "INSERT OR REPLACE INTO docmeta(doc, n_sections, built_at) VALUES (?,?,datetime('now'))",
                (doc, len(sections)),
            )
            total += len(sections)
        conn.commit()
    finally:
        conn.close()
    return total


@dataclass
class Hit:
    doc: str
    heading: str
    level: int
    pages: str
    snippet: str

    def report(self) -> str:
        loc = f" [{self.pages}]" if self.pages else ""
        return f"- {self.doc} § {self.heading}{loc}\n    {self.snippet}"


def search(query: str, *, limit: int = 10, doc: str | None = None) -> list[Hit]:
    """FTS5 全文检索，返回按相关度排序的命中（手册/章节/页码/摘录）。"""
    conn = _connect()
    try:
        sql = (
            "SELECT doc, heading, level, pages, snippet(sections, 4, '>>', '<<', ' … ', 24) "
            "FROM sections WHERE sections MATCH ?"
        )
        params: list[Any] = [query]
        if doc:
            sql += " AND doc = ?"
            params.append(doc)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        raise ValueError(f"FTS5 查询语法错误：{query!r} ({e})") from e
    finally:
        conn.close()
    return [
        Hit(doc=r[0], heading=r[1], level=int(r[2]), pages=r[3], snippet=r[4])
        for r in rows
    ]


def read(doc: str, heading: str | None = None) -> str:
    """读取某手册的 Markdown：heading=None → 整本（从 docs/*.md 原文）；否则返回该章节。"""
    if heading is None:
        p = settings.docs_dir / f"{doc}.md"
        if not p.exists():
            raise FileNotFoundError(f"未找到手册 Markdown：{p}")
        return p.read_text(encoding="utf-8", errors="replace")
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT body FROM sections WHERE doc = ? AND heading = ?", (doc, heading)
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise KeyError(f"手册 {doc} 中未找到章节 {heading!r}")
    return rows[0][0]


def list_docs() -> list[dict[str, Any]]:
    """列出已建索引的手册及其 section 数/构建时间。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT doc, n_sections, built_at FROM docmeta ORDER BY doc"
        ).fetchall()
    finally:
        conn.close()
    return [{"doc": r[0], "n_sections": r[1], "built_at": r[2]} for r in rows]
