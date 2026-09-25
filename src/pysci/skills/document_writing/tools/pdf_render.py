"""pdf_render —— 把 PDF 页面渲染成 PNG，供 LLM「看图」校对版式。

这是产出「高质量终稿」的关键闭环：编译出 PDF 后，渲染成图片，LLM 用 Read 工具
直接查看每一页的真实排版（标题/图/表/公式/参考文献/overfull 溢出），据此迭代。

依赖 pymupdf（核心依赖 pymupdf4llm 已带入）。
"""

from __future__ import annotations

from pathlib import Path

from .config import settings

try:  # pymupdf 新旧命名兼容
    import pymupdf as _mupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as _mupdf  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pymupdf 未安装（应随核心依赖 pymupdf4llm 提供）。"
        ) from e


def render_pdf_pages(
    pdf_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    dpi: int = 140,
    pages: list[int] | None = None,
    max_pages: int | None = None,
    prefix: str = "page",
) -> list[Path]:
    """把 PDF 渲染为逐页 PNG。

    Args:
        pdf_path: 源 PDF。
        out_dir: 输出目录；默认 cache/renders/<stem>/。
        dpi: 渲染分辨率（版式校对 120–160 足够；越大越清晰越慢）。
        pages: 指定 1-based 页码列表；None 表示全部。
        max_pages: 最多渲染前 N 页（防止超长文档刷屏）。
        prefix: 输出文件名前缀。

    Returns:
        渲染出的 PNG 路径列表（按页序）。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    if out_dir is None:
        out_dir = settings.cache_renders / pdf_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = _mupdf.open(str(pdf_path))
    try:
        total = doc.page_count
        if pages:
            idxs = [p - 1 for p in pages if 1 <= p <= total]
        else:
            idxs = list(range(total))
        if max_pages is not None:
            idxs = idxs[:max_pages]

        zoom = dpi / 72.0
        matrix = _mupdf.Matrix(zoom, zoom)
        out: list[Path] = []
        for i in idxs:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = out_dir / f"{prefix}-{i + 1:03d}.png"
            pix.save(str(png))
            out.append(png)
        return out
    finally:
        doc.close()


def page_count(pdf_path: str | Path) -> int:
    doc = _mupdf.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF → PNG 渲染")
    parser.add_argument("pdf")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--pages", default=None, help="逗号分隔的 1-based 页码")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    pg = [int(x) for x in args.pages.split(",")] if args.pages else None
    pngs = render_pdf_pages(
        args.pdf,
        out_dir=args.out_dir,
        dpi=args.dpi,
        pages=pg,
        max_pages=args.max_pages,
    )
    print(f"[render] {len(pngs)} 页 → ")
    for p in pngs:
        print(f"  {p}")
