"""PDF→PNG 视觉校对（pymupdf 薄封装）。

ADR（2026-11-17）：DSH office-render 只渲染 Office 文件、**不渲染 PDF**，LaTeX 产物
与 office-export-pdf 产物的逐页目检需要这条薄封装（≈50 行预算）。pymupdf 是
pymupdf4llm 的同源底层（选型已接受 AGPL-3.0，个人内部使用）；依赖由并行实施者装入
项目 venv，本模块懒导入、缺失时报可操作的错。
"""

from __future__ import annotations

from pathlib import Path


class PdfRenderError(RuntimeError):
    """PDF 渲染失败（依赖缺失或文件不可读）。"""


def render_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 200,
    prefix: str | None = None,
) -> list[Path]:
    """把 PDF 逐页渲染为 PNG（``<前缀>_001.png``…）。

    Args:
        pdf_path: PDF 文件路径。
        out_dir: PNG 输出目录（自动创建）。
        dpi: 渲染分辨率（视觉校对 200dpi 足够；公式出图走 ``formula_png`` 的 600dpi）。
        prefix: 输出文件名前缀；None 时用 PDF 文件 stem。

    Returns:
        按页序的 PNG 路径列表（空 PDF 返回空列表）。

    Raises:
        PdfRenderError: pymupdf 未安装或 PDF 打不开。
    """
    try:
        import pymupdf  # 懒导入（依赖由 pymupdf4llm 同源提供）
    except ImportError as exc:  # pragma: no cover - 依赖就位后不走
        raise PdfRenderError("缺 pymupdf（uv sync 安装 pymupdf4llm 后可用）") from exc
    src = Path(pdf_path)
    if not src.is_file():
        raise PdfRenderError(f"PDF 不存在：{src}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = prefix or src.stem
    paths: list[Path] = []
    try:
        with pymupdf.open(src) as doc:
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=dpi)
                target = out / f"{stem}_{index:03d}.png"
                pix.save(target)
                paths.append(target)
    except Exception as exc:  # pymupdf 自身异常统一包装
        raise PdfRenderError(f"PDF 渲染失败：{src}（{exc}）") from exc
    return paths
