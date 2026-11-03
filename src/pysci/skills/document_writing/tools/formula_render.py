"""公式渲染管线：LaTeX（article+amsmath）→ PDF → **ink 裁剪** RGBA PNG。

deck 公式图的唯一渲染入口（backlog 20261103-134217-writing）：

背景：TeXLive 2026 的 standalone 类（v1.5a）使数学定界符失效——``equation*``/``\\[``
不进入数学模式，编译必报「Missing $ inserted」等错却仍产出 PDF；旧脚本只断言
「PDF 存在」，误差恢复产物（如空分子分数）于是混进成品 deck。本管线三条铁律：

1. 模板固定 **article + amsmath**（数学模式可靠；**禁 standalone**）；
2. 产物验收 = **编译零 error**（``latex_build`` 的 ``ok`` ＝ PDF 存在 ∧ rc==0 ∧ log 无
   error），不满足即 :class:`FormulaCompileError`——PDF 即便存在也不可用；
3. 失败**不落盘**：PNG 最终原子写出（``tmp`` + ``os.replace``），编译/裁剪任一失败即
   零产物，误差恢复产物不可进 deck。

PDF→PNG 按 **ink 裁剪**（仅墨迹 bbox，白边剔除），RGBA 透明底、零描边；dpi 默认 600
（deck 口径 ≥300）。公式源串（``equation*``/``\\[\\]``/多式同页均可）**原样**进模板。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .latex_build import build

#: 公式模板：article + amsmath + ctex（公式里的中文如「其他区域」）；pagestyle 空。
#: **禁 standalone**——TeXLive 2026 standalone v1.5a 破坏 ``equation*``/``\\[`` 数学定界符。
FORMULA_TEMPLATE = """\\documentclass[12pt]{article}
\\usepackage{amsmath}
\\usepackage{ctex}
\\pagestyle{empty}
\\begin{document}
%s
\\end{document}
"""

#: ink 判定阈值（灰度 < 阈值算墨迹；255=纯白空页边）。
_INK_GRAY: int = 250


class FormulaRenderError(RuntimeError):
    """公式渲染失败（编译/结构/裁剪任一环节）。"""


class FormulaCompileError(FormulaRenderError):
    """编译未过**零 error** 验收——产物（即便存在 PDF）一律不可用。"""


def formula_tex(body: str) -> str:
    """公式源串 → 完整 .tex（article+amsmath 模板）。

    Args:
        body: 公式源串（``\\begin{equation*}``/``\\[..\\]`` 等，可多式同页；原样嵌入）。

    Returns:
        可直接编译的 .tex 全文。
    """
    return FORMULA_TEMPLATE % body


def build_formula_pdf(
    body: str,
    out_pdf: str | Path,
    *,
    engine: str = "xelatex",
    timeout: int = 120,
) -> Path:
    """编译公式源串 → 单页 PDF（**零 error 验收**后才落盘）。

    Args:
        body: 公式源串（见 :func:`formula_tex`）。
        out_pdf: 输出 PDF 路径。
        engine: TeX 引擎（ctex 需 xelatex/lualatex，缺省 xelatex）。
        timeout: 编译超时秒数。

    Returns:
        ``out_pdf`` 路径。

    Raises:
        FormulaCompileError: 编译有 error / PDF 未生成——**不产出任何文件**。
    """
    out_pdf = Path(out_pdf)
    with tempfile.TemporaryDirectory(prefix="formula_") as td:
        tdir = Path(td)
        main_tex = tdir / "formula.tex"
        main_tex.write_text(formula_tex(body), encoding="utf-8", newline="\n")
        res = build(
            main_tex, engine=engine, out_dir=tdir / "build", bib=False, timeout=timeout
        )
        if not res.ok or res.pdf_path is None:
            detail = (
                "；".join(e.fmt() for e in res.errors[:5]) or f"rc={res.returncode}"
            )
            raise FormulaCompileError(
                f"公式编译未过零 error 验收（{detail}）——产物不落盘"
            )
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.pdf_path, out_pdf)
    return out_pdf


def _pdf_page_to_ink_rgba(pdf: Path, *, dpi: int):
    """单页 PDF → ink 裁剪后的 RGBA（透明底、墨迹黑）。"""
    import numpy as np
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(str(pdf))
    try:
        if doc.page_count != 1:
            raise FormulaRenderError(
                f"公式图须单页（一页一图），实际 {doc.page_count} 页——拆组或缩排后重渲"
            )
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    finally:
        doc.close()
    gray = np.asarray(
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    )
    ink = gray < _INK_GRAY
    if not ink.any():
        raise FormulaRenderError("公式渲染无墨迹（空白产物）——不落盘")
    ys, xs = np.nonzero(ink)
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    rgba = np.zeros(crop.shape + (4,), dtype=np.uint8)
    rgba[..., 3] = 255 - crop
    return Image.fromarray(rgba, "RGBA")


def render_formula_png(
    body: str,
    out_png: str | Path,
    *,
    dpi: int = 600,
    engine: str = "xelatex",
    timeout: int = 120,
) -> Path:
    """公式源串 → ink 裁剪 RGBA PNG（deck 公式图成品）。

    全链验收后再原子落盘：编译零 error（:func:`build_formula_pdf`）→ 单页 → 有墨迹
    → ink bbox 裁剪；任一失败**零产物**（误差恢复产物不可进 deck）。

    Args:
        body: 公式源串（见 :func:`formula_tex`）。
        out_png: 输出 PNG 路径（透明底、零描边）。
        dpi: 渲染分辨率（deck 口径 ≥300，缺省 600）。
        engine: TeX 引擎。
        timeout: 编译超时秒数。

    Returns:
        ``out_png`` 路径。
    """
    out_png = Path(out_png)
    with tempfile.TemporaryDirectory(prefix="formula_") as td:
        pdf = build_formula_pdf(
            body, Path(td) / "formula.pdf", engine=engine, timeout=timeout
        )
        img = _pdf_page_to_ink_rgba(pdf, dpi=dpi)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_png.with_name(out_png.name + ".tmp")
        img.save(tmp, format="PNG")
        os.replace(tmp, out_png)
    return out_png
