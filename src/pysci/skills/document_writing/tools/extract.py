"""extract —— 统一文档提取（→ Markdown），带缓存。

用 markitdown 把 pptx/docx/xlsx/html/csv 等转成 Markdown；PDF 走已装的核心依赖
pymupdf4llm（如需公式级精读，请改用 literature_research 的 MinerU 后端）。

这是 `compose read <file>` 的后端：给一个文件，得到干净、可被 LLM 直接阅读/编辑的
Markdown，并缓存到 cache/extracted/。对 .pptx 若 markitdown 失败，自动回退到
pptx_io（python-pptx 结构化提取）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass
class ExtractResult:
    markdown: str
    cache_path: Path | None
    backend: str  # markitdown / pymupdf4llm / pptx_io
    from_cache: bool
    source: Path

    @property
    def char_count(self) -> int:
        return len(self.markdown)


# 后缀 → 处理方式
_PPTX_EXT = {".pptx", ".pptm"}
_PDF_EXT = {".pdf"}
_MARKITDOWN_EXT = {
    ".pptx",
    ".pptm",
    ".docx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".txt",
    ".md",
    ".epub",
}


def _cache_path_for(source: Path, backend: str) -> Path:
    return settings.cache_extracted / f"{source.stem}__{backend}.md"


def _via_markitdown(path: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "markitdown 未安装。请运行：uv sync --extra writing"
        ) from e
    md = MarkItDown()
    # convert_local 是面向本地文件的推荐入口（0.1.x）
    convert = getattr(md, "convert_local", None) or md.convert
    result = convert(str(path))
    return getattr(result, "text_content", "") or ""


def _via_pymupdf4llm(path: Path) -> str:
    try:
        import pymupdf4llm
    except ImportError as e:  # pragma: no cover
        raise ImportError("pymupdf4llm 未安装（应为核心依赖）。") from e
    return pymupdf4llm.to_markdown(str(path))


def _via_pptx_io(path: Path) -> str:
    from . import pptx_io

    return pptx_io.pptx_to_markdown(path)


def to_markdown(
    path: str | Path,
    *,
    use_cache: bool = True,
    force: bool = False,
    backend: str = "auto",
) -> ExtractResult:
    """把文件提取为 Markdown（带缓存）。

    Args:
        path: 源文件。
        use_cache: 命中缓存则直接返回（force=True 忽略缓存）。
        force: 强制重新提取并覆盖缓存。
        backend: "auto" 按后缀选择；或显式 "markitdown" / "pymupdf4llm" / "pptx_io"。
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在：{src}")
    ext = src.suffix.lower()

    # 选后端
    if backend == "auto":
        if ext in _PDF_EXT:
            chosen = "pymupdf4llm"
        elif ext in _PPTX_EXT:
            chosen = "markitdown"
        elif ext in _MARKITDOWN_EXT:
            chosen = "markitdown"
        else:
            # 未知后缀：仍尝试 markitdown（它会自行判定），失败则按纯文本读
            chosen = "markitdown"
    else:
        chosen = backend

    cache_path = _cache_path_for(src, chosen)
    if use_cache and not force and cache_path.exists():
        return ExtractResult(
            markdown=cache_path.read_text(encoding="utf-8"),
            cache_path=cache_path,
            backend=chosen,
            from_cache=True,
            source=src,
        )

    md_text = ""
    used = chosen
    if chosen == "pymupdf4llm":
        md_text = _via_pymupdf4llm(src)
    elif chosen == "pptx_io":
        md_text = _via_pptx_io(src)
    else:  # markitdown
        try:
            md_text = _via_markitdown(src)
        except Exception:
            if ext in _PPTX_EXT:
                # 回退到 python-pptx 结构化提取
                md_text = _via_pptx_io(src)
                used = "pptx_io"
                cache_path = _cache_path_for(src, used)
            elif ext == ".docx":
                raise RuntimeError(
                    "docx 提取需要 markitdown[docx]（Phase 2）。临时可运行："
                    'uv add --optional writing "markitdown[docx]"'
                ) from None
            else:
                # 最后兜底：按纯文本读
                try:
                    md_text = src.read_text(encoding="utf-8", errors="replace")
                    used = "plaintext"
                except Exception as e:
                    raise RuntimeError(f"无法提取 {src.name}：{type(e).__name__}: {e}") from e

    if not md_text.strip():
        md_text = f"（{src.name} 提取结果为空）"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(md_text, encoding="utf-8")
    return ExtractResult(
        markdown=md_text,
        cache_path=cache_path,
        backend=used,
        from_cache=False,
        source=src,
    )


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="统一文档 → Markdown 提取")
    parser.add_argument("file", help="源文件（pptx/docx/pdf/xlsx/html/...）")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新提取")
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()

    res = to_markdown(args.file, force=args.force, backend=args.backend)
    print(f"[extract] backend={res.backend}  from_cache={res.from_cache}  "
          f"chars={res.char_count}\n[extract] cache={res.cache_path}\n")
    print(res.markdown)
