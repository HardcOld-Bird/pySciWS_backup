"""公式 → 透明底 RGBA PNG（ink 裁剪 + 渲染管线编排）。

用途（技能文档"公式渲染约定"）：pptx/docx 图内公式没有逐字公式能力，走图片——
LaTeX ``article+amsmath`` standalone 编译 → PDF 600dpi 渲染（RGBA，透明底）→
ink bbox 裁剪 → 原子落盘。

ADR（2026-11-17）：ink 裁剪无社区对应（Typst 不引入，pandoc 不产图），自研 ≈100 行
（选型报告预算项 5）。裁剪核心 :func:`ink_bbox` 是**纯函数**（alpha 缓冲进、bbox 出），
PNG 编解码交给 Pillow（python-pptx 的传递依赖，DSH office-pptx 锚定件）；渲染用 pymupdf。
两者懒导入，缺失时报可操作的错。源串自检：ink bbox 为空 → 报错而不是交付空图。
"""

from __future__ import annotations

import os
from pathlib import Path

from pysci.skills.document_writing import latex_workflow


class FormulaError(RuntimeError):
    """公式渲染失败（依赖缺失、编译失败或空输出）。"""


def ink_bbox(
    alpha: bytes,
    width: int,
    height: int,
    *,
    threshold: int = 0,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """求 alpha 通道的 ink 包围盒（纯函数，机械可测）。

    Args:
        alpha: 行优先 alpha 缓冲，长度 ``width * height``（0=全透明，255=不透明）。
        width: 图像宽（像素）。
        height: 图像高（像素）。
        threshold: alpha **大于**该值算 ink（默认 0：任何非全透明都算）。
        padding: 包围盒向外扩的像素数（按图像边界截断）。

    Returns:
        ``(left, top, right, bottom)``（右/下**开区间**，可直接给 crop）；
        全透明返回 None。

    Raises:
        ValueError: 缓冲长度与 ``width * height`` 不符，或尺寸非正。
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"尺寸非法：{width}x{height}")
    if len(alpha) != width * height:
        raise ValueError(f"alpha 缓冲长度 {len(alpha)} != {width * height}")
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(height):
        row = y * width
        for x in range(width):
            if alpha[row + x] > threshold:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if max_x < 0:
        return None
    left = max(0, min_x - padding)
    top = max(0, min_y - padding)
    right = min(width, max_x + 1 + padding)
    bottom = min(height, max_y + 1 + padding)
    return (left, top, right, bottom)


def crop_png(
    src: str | Path,
    dst: str | Path,
    *,
    threshold: int = 0,
    padding: int = 2,
) -> Path:
    """按 ink bbox 裁剪透明边距，原子落盘（PIL 编解码）。

    Args:
        src: 源 PNG（RGBA；带 alpha 的 RGB 会被转成 RGBA）。
        dst: 目标 PNG；先写 ``<dst>.tmp`` 再 ``os.replace``（原子落盘）。
        threshold: 传给 :func:`ink_bbox` 的 ink 阈值。
        padding: 传给 :func:`ink_bbox` 的留白像素。

    Returns:
        目标路径。

    Raises:
        FormulaError: Pillow 未安装、源图不可读或全透明（空输出自检）。
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - 依赖就位后不走
        raise FormulaError("缺 Pillow（python-pptx 的传递依赖，uv sync 后可用）") from exc
    source = Path(src)
    try:
        with Image.open(source) as img:
            rgba = img.convert("RGBA")
            width, height = rgba.size
            bbox = ink_bbox(
                rgba.getchannel("A").tobytes(), width, height,
                threshold=threshold, padding=padding,
            )
            if bbox is None:
                raise FormulaError(f"空输出（全透明，视为编译失败）：{source}")
            cropped = rgba.crop(bbox)
            target = Path(dst)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            cropped.save(tmp, format="PNG")
    except FormulaError:
        raise
    except OSError as exc:
        raise FormulaError(f"PNG 读写失败：{source}（{exc}）") from exc
    os.replace(tmp, target)
    return target


def render_formula(
    tex: str,
    out_png: str | Path,
    *,
    dpi: int = 600,
    padding: int = 2,
    inline: bool = False,
) -> Path:
    """把 LaTeX 公式源串渲染为透明底裁剪 PNG（整条公式管线）。

    管线：``article+amsmath`` standalone（``\\[ ... \\]`` 显示模式，``inline=True``
    时 ``$ ... $``）→ latexmk 编译（见 :mod:`latex_workflow`）→ pymupdf 600dpi
    RGBA 渲染 → :func:`crop_png` ink 裁剪 → 原子落盘。

    Args:
        tex: 公式源串（不含 ``$``/``\\[\\]`` 包裹）。
        out_png: 输出 PNG 路径。
        dpi: 渲染分辨率。
        padding: 裁剪留白像素。
        inline: True 用行内模式 ``$...$``，默认显示模式。

    Returns:
        输出 PNG 路径。

    Raises:
        FormulaError: 依赖缺失、编译失败（含错误列表）或空输出。
    """
    import tempfile

    body = f"${tex}$" if inline else f"\\[{tex}\\]"
    document = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n"
    )
    with tempfile.TemporaryDirectory(prefix="dw_formula_") as work:
        tex_path = Path(work) / "formula.tex"
        tex_path.write_text(document, encoding="utf-8")
        result = latex_workflow.build(tex_path, run_lint=False)
        if not result.ok or result.pdf is None:
            detail = "; ".join(f"{i.file}:{i.line}: {i.message}" for i in result.issues)
            raise FormulaError(f"公式编译失败：{detail or result.log_text[-500:]}")
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - 依赖就位后不走
            raise FormulaError("缺 pymupdf（uv sync 安装 pymupdf4llm 后可用）") from exc
        raw = Path(work) / "formula_raw.png"
        with pymupdf.open(result.pdf) as doc:
            pix = doc[0].get_pixmap(dpi=dpi, alpha=True)
            pix.save(raw)
        return crop_png(raw, out_png, padding=padding)
