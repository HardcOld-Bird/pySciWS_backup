"""docx_io —— 用 python-docx 结构化读写 .docx。

读：按文档顺序保留 标题层级 / 正文 / 项目符号 / 表格 → Markdown。
写：从 Block 规格或约定式 Markdown 生成 .docx（标题/正文/项目符号/表格）。

用于「必须从 docx 数据源取信息」或「必须交付 docx」的场景。仅支持 .docx
（OOXML），不支持旧版 .doc。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# python-docx 是 [writing] 可选依赖；缺失时给出清晰指引而非裸 ImportError。
try:
    from docx import Document
except ImportError as e:  # pragma: no cover
    raise ImportError("python-docx 未安装。请运行：uv sync --extra writing") from e


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class DocxBlock:
    """文档中的一个块（读/写共用）。"""

    kind: str  # "heading" | "paragraph" | "bullet" | "table" | "quote"
    text: str = ""
    level: int = 0  # heading 级别(1..)；bullet 缩进级别(0..)
    rows: list[list[str]] | None = None  # table：行→列


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def _heading_level(style_name: str) -> int | None:
    """把 Word 样式名映射为标题级别；非标题返回 None。"""
    name = (style_name or "").strip()
    if name == "Title":
        return 1
    if name.startswith("Heading"):
        tail = name[len("Heading") :].strip()
        try:
            return max(1, int(tail))
        except ValueError:
            return 1
    return None


def _is_bullet(style_name: str) -> bool:
    return (style_name or "").strip().startswith("List Bullet")


def _bullet_level_from_style(style_name: str) -> int:
    tail = (style_name or "").strip()[len("List Bullet") :].strip()
    try:
        return max(0, int(tail) - 1)
    except ValueError:
        return 0


def _table_to_rows(table: Any) -> list[list[str]]:
    return [
        [(c.text or "").strip().replace("\n", " ") for c in row.cells]
        for row in table.rows
    ]


def read_docx(path: str | Path) -> list[DocxBlock]:
    """按文档顺序读取 .docx 的块序列。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"docx 不存在：{path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"仅支持 .docx（不支持旧版 .doc）：{path.name}")

    doc = Document(str(path))
    blocks: list[DocxBlock] = []

    # iter_inner_content 保持段落/表格交错顺序（python-docx >= 1.1）
    items = list(doc.iter_inner_content()) if hasattr(doc, "iter_inner_content") else [
        *doc.paragraphs,
        *doc.tables,
    ]
    for item in items:
        if item.__class__.__name__ == "Table":
            rows = _table_to_rows(item)
            if rows:
                blocks.append(DocxBlock(kind="table", rows=rows))
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        style = getattr(item.style, "name", "") or ""
        lvl = _heading_level(style)
        if lvl is not None:
            blocks.append(DocxBlock(kind="heading", text=text, level=lvl))
        elif _is_bullet(style):
            blocks.append(DocxBlock(kind="bullet", text=text, level=_bullet_level_from_style(style)))
        else:
            blocks.append(DocxBlock(kind="paragraph", text=text))
    return blocks


# ---------------------------------------------------------------------------
# 渲染为 Markdown
# ---------------------------------------------------------------------------
def _rows_to_md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(norm[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    for r in norm[1:]:
        out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in r) + " |")
    return "\n".join(out)


def blocks_to_markdown(blocks: list[DocxBlock], *, source_name: str = "") -> str:
    parts: list[str] = []
    if source_name:
        parts.append(f"# DOCX 提取：{source_name}\n")
    for b in blocks:
        if b.kind == "heading":
            parts.append(f"{'#' * min(b.level, 6)} {b.text}\n")
        elif b.kind == "bullet":
            parts.append(f"{'  ' * b.level}- {b.text}\n")
        elif b.kind == "table":
            parts.append(_rows_to_md_table(b.rows or []) + "\n")
        elif b.kind == "quote":
            parts.append(f"> {b.text}\n")
        else:
            parts.append(f"{b.text}\n")
    return "\n".join(parts)


def docx_to_markdown(path: str | Path) -> str:
    """一步到位：读取 .docx → Markdown。"""
    path = Path(path)
    return blocks_to_markdown(read_docx(path), source_name=path.name)


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------
_BULLET_STYLES = ("List Bullet", "List Bullet 2", "List Bullet 3", "List Bullet 4", "List Bullet 5")


def _add_block(doc: Any, b: DocxBlock) -> None:
    if b.kind == "heading":
        doc.add_heading(b.text, level=max(1, min(b.level, 9)))
    elif b.kind == "bullet":
        doc.add_paragraph(b.text, style=_BULLET_STYLES[min(b.level, len(_BULLET_STYLES) - 1)])
    elif b.kind == "table":
        rows = b.rows or []
        if not rows:
            return
        n_c = max(len(r) for r in rows)
        tbl = doc.add_table(rows=len(rows), cols=n_c)
        tbl.style = "Table Grid"
        for r, row in enumerate(rows):
            for c in range(n_c):
                tbl.cell(r, c).text = row[c] if c < len(row) else ""
    elif b.kind == "quote":
        para = doc.add_paragraph()
        para.add_run(b.text).italic = True
    else:
        doc.add_paragraph(b.text)


def build_docx(blocks: list[DocxBlock], out_path: str | Path, *, title: str | None = None) -> Path:
    """从 Block 列表新建 .docx；title 给定时加一个 Title 级标题。"""
    doc = Document()
    if title:
        doc.add_heading(title, level=0)
    for b in blocks:
        _add_block(doc, b)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return out


def add_block_to(path: str | Path, block: DocxBlock) -> Path:
    """向已有 .docx 追加一个块并保存。"""
    path = Path(path)
    doc = Document(str(path))
    _add_block(doc, block)
    doc.save(str(path))
    return path


def markdown_to_docx(md_text: str, out_path: str | Path) -> Path:
    """把约定式 Markdown 转成 .docx（与 docx_to_markdown 构成读写闭环）。

    约定：`#`..`######` = 标题层级；`- `/`* `（可缩进）= 项目符号；管道表 = 表格；
    其余非空行 = 正文段落。首个 `# ` 同时作为文档 Title。
    """
    md_text = md_text.lstrip("\ufeff")  # 剥离 BOM（Windows 工具常写入）
    blocks: list[DocxBlock] = []
    table_buf: list[list[str]] = []
    first_h1 = True

    def flush_table() -> None:
        nonlocal table_buf
        if table_buf:
            blocks.append(DocxBlock(kind="table", rows=table_buf))
        table_buf = []

    for raw in md_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(c and set(c) <= set("-: ") for c in cells):
                continue  # 表分隔行
            table_buf.append(cells)
            continue
        flush_table()
        if s.startswith("#"):
            depth = len(s) - len(s.lstrip("#"))
            text = s.lstrip("#").strip()
            if depth == 1 and first_h1:
                blocks.append(DocxBlock(kind="heading", text=text, level=1))
                first_h1 = False
            else:
                blocks.append(DocxBlock(kind="heading", text=text, level=min(depth, 6)))
            continue
        stripped = raw.lstrip(" ")
        if stripped[:1] in ("-", "*"):
            level = (len(raw) - len(stripped)) // 2
            blocks.append(
                DocxBlock(kind="bullet", text=stripped.lstrip("-* ").strip(), level=min(level, 4))
            )
            continue
        if s.startswith(">"):
            blocks.append(DocxBlock(kind="quote", text=s.lstrip("> ").strip()))
            continue
        blocks.append(DocxBlock(kind="paragraph", text=s))
    flush_table()
    return build_docx(blocks, out_path)
