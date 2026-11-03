"""公式渲染管线：LaTeX（article+amsmath）→ PDF → **ink 裁剪** RGBA PNG。

deck 公式图的唯一渲染入口（backlog 20261103-134217-writing）：

背景：TeXLive 2026 的 standalone 类（v1.5a）使数学定界符失效——``equation*``/``\\[``
不进入数学模式，编译必报「Missing $ inserted」等错却仍产出 PDF；旧脚本只断言
「PDF 存在」，误差恢复产物（如空分子分数）于是混进成品 deck。本管线三条铁律：

1. 模板固定 **article + amsmath**（数学模式可靠；**禁 standalone**）；
2. 产物验收 = **编译零 error**（``latex_build`` 的 ``ok`` ＝ PDF 存在 ∧ rc==0 ∧ log 无
   error），不满足即 :class:`FormulaCompileError`——PDF 即便存在也不可用；
3. 失败**不落盘**：PNG 最终原子写出（``tmp`` + ``os.replace``），编译/裁剪任一失败即
   零产物，误差恢复产物不可进 deck；
4. **渲染图↔源串机械结构自检**（backlog 20261103-131952-reviewer，:func:
   `verify_formula_structure`）：字形层（内容字符不缺不截）＋分数层（分数杠上下
   墨迹、单杠 A/B 绑定）——空分子/错位分组/截断在**产出时**即拦（:
   :class:`FormulaStructureError`），不再依赖终检像素取证。

PDF→PNG 按 **ink 裁剪**（仅墨迹 bbox，白边剔除），RGBA 透明底、零描边；dpi 默认 600
（deck 口径 ≥300）。公式源串（``equation*``/``\\[\\]``/多式同页均可）**原样**进模板。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections import Counter
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


class FormulaStructureError(FormulaRenderError):
    """渲染产物与源串**结构不符**（空分子/错位分组/截断丢字）——不落盘。

    backlog 20261103-131952-reviewer：p14_f4_t.png 曾与源串 ``\\frac{1}{n_{0}}k_{x}``
    结构不符（空分子、分母吞并 k_x）却混进成品——渲染产出时即须机械拦截。
    """


#: 源串里的 LaTeX 命令（``\frac`` 等名）与转义符（``\%`` 等）——其中字母不算内容字符。
_CMD_RE = re.compile(r"\\(?:[A-Za-z]+|.)")

#: 环境定界与不可见参数（``\begin{equation*}``/``\label{...}`` 的名字不渲染）。
_ENV_RE = re.compile(r"\\(?:begin|end)\{[^}]*\}")
_LABEL_RE = re.compile(r"\\label\{[^}]*\}")

#: 会在版面里画**横线**的命令族（分数杠的几何混淆源）：源串含它们时分数结构层整体
#: 跳过（降级到字形层）——否则 \sqrt 上划线 / \overline 会被误判成「空分子分数杠」。
_RULE_COMMANDS_RE = re.compile(
    r"\\(?:sqrt|overline|underline|widehat|widetilde|widecheck|utilde|widebar"
    r"|overrightarrow|overleftarrow|overleftrightarrow|overbrace|underbrace)\b"
)

#: 分数命令族（每实例画一条分数杠）。
_FRAC_COMMANDS_RE = re.compile(r"\\(?:d|t|c)?frac\b")

#: 分数杠窗口：杠上方/下方该范围内才算分子/分母区（12 pt 基准字号的堆叠极限）。
_BAR_WINDOW_PT = 24.0


def _required_chars(body: str) -> dict[str, int]:
    """源串 → **内容字符**计数（字母数字与 CJK；命令名/转义符/结构符不算）。

    ``\\frac{1}{n_{0}}k_{x}`` → ``{1:1, n:1, 0:1, k:1, x:1}``；``\alpha`` 的
    ``alpha`` 是命令名不计（渲染为 α 字形，抽取侧同样放行）。
    """
    s = _ENV_RE.sub(" ", body)
    s = _LABEL_RE.sub(" ", s)
    s = _CMD_RE.sub(" ", s)
    s = re.sub(r"[{}$ \t\r\n_^&]", "", s)
    return dict(Counter(c for c in s if c.isalnum()))


#: 数学表达式开界（多式同页时严格绑定层跳过——防跨式串扰误报）。
_CHUNK_OPEN_RE = re.compile(r"\\\[|\\\(|\\begin\{equation\*?\}|\$\$")


def _frac_arg_chars(body: str) -> tuple[frozenset, frozenset] | None:
    """源串恰含**一个** ``\\frac{A}{B}``（单式页）时返回 (A, B) 内容字符集。

    有第二个 ``\\frac``、含横线命令、或多式同页（跨式串扰不可控）→ ``None``
    （降级宽松层）。frac 之后的后继记号（如 ``…k_{x}``）**留在宇宙里**——它若被
    吞进分母区，正是「错位分组」要抓的形态。
    """
    if len(_CHUNK_OPEN_RE.findall(body)) > 1:
        return None
    m = _FRAC_COMMANDS_RE.search(body)
    if not m:
        return None
    if _FRAC_COMMANDS_RE.search(body, m.end()) or _RULE_COMMANDS_RE.search(body):
        return None
    i = m.end()
    args: list[str] = []
    for _ in range(2):
        while i < len(body) and body[i] in " \t\r\n":
            i += 1
        if i >= len(body) or body[i] != "{":
            return None
        depth, j = 0, i
        while j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return None
        args.append(body[i + 1 : j])
        i = j + 1
    a = frozenset(c for c in _required_chars(args[0]))
    b = frozenset(c for c in _required_chars(args[1]))
    return (a, b) if (a or b) else None


def verify_formula_structure(body: str, pdf: str | Path) -> None:
    """渲染产物 ↔ 源串**机械结构自检**（产出即校验、失败即报）。

    两层判据（backlog 20261103-131952-reviewer）：

    1. **字形层（总是）**：源串的内容字符在 PDF 文本层的出现次数不得少于源串——
       空分子/截断丢字即 :class:`FormulaStructureError`；
    2. **分数层**（源串含 ``\\frac`` 族且无 ``\\sqrt``/``\\overline`` 族横线命令时）：
       分数杠数＝``\\frac`` 数（错位分组丢杠即 FAIL）；每杠上下 24 pt 窗内皆有字形
       （分子区墨迹空洞即 FAIL）；单杠时按源串 ``\\frac{A}{B}`` 严格绑定上下区字符集
       （分母吞并后继记号即 FAIL）。

    Args:
        body: 公式源串（与渲染时同一份）。
        pdf: 渲染产物 PDF 路径。

    Raises:
        FormulaStructureError: 结构不符（附层名与差集）；PDF 读不开同判。
    """
    import pymupdf

    try:
        doc = pymupdf.open(str(pdf))
    except Exception as exc:  # noqa: BLE001 - 产物读不开＝结构不可信
        raise FormulaStructureError(f"渲染产物 PDF 读不开（{exc}）——不落盘") from exc
    try:
        word_boxes = [(w[4], w[:4]) for page in doc for w in page.get_text("words")]
        bars = [
            d["rect"]
            for page in doc
            for d in page.get_drawings()
            if d["rect"].y1 - d["rect"].y0 <= 2 and d["rect"].x1 - d["rect"].x0 >= 3
        ]
    finally:
        doc.close()

    have = Counter(c for w, _ in word_boxes for c in w if c.isalnum())
    need = _required_chars(body)
    lack = {c: (n - have.get(c, 0)) for c, n in need.items() if have.get(c, 0) < n}
    if lack:
        raise FormulaStructureError(
            f"字形层：源串内容字符在产物中缺失/不足 {lack}（截断或空分子丢字）——不落盘"
        )

    if not _FRAC_COMMANDS_RE.search(body) or _RULE_COMMANDS_RE.search(body):
        return  # 无分数 / 含横线命令（杠不可辨）→ 分数层跳过
    n_frac = len(_FRAC_COMMANDS_RE.findall(body))
    if len(bars) != n_frac:
        raise FormulaStructureError(
            f"分数层：分数杠 {len(bars)} 条 ≠ \\frac 族 {n_frac} 个（错位分组/丢杠）——不落盘"
        )
    # 每杠上下窗内皆有字形（分子/分母区墨迹空洞检测）
    for r in bars:
        above = below = False
        for w, (x0, y0, x1, y1) in word_boxes:
            if x1 < r.x0 - 1 or x0 > r.x1 + 1:
                continue
            if r.y0 - _BAR_WINDOW_PT <= y1 <= r.y0 + 0.5:
                above = True
            if r.y1 - 0.5 <= y0 <= r.y1 + _BAR_WINDOW_PT:
                below = True
        if not (above and below):
            which = "分子" if not above else "分母"
            raise FormulaStructureError(
                f"分数层：分数杠 {r} 的{which}区窗内无字形（墨迹空洞）——不落盘"
            )
    # 单杠严格绑定：上下区内容字符须恰为 \\frac{A}/{B} 的字符集（后继记号不得被吞进分母）
    if n_frac == 1 and len(bars) == 1:
        bound = _frac_arg_chars(body)
        if bound is not None:
            a_chars, b_chars = bound
            r = bars[0]
            above: set[str] = set()
            below: set[str] = set()
            for w, (x0, y0, x1, y1) in word_boxes:
                if x1 < r.x0 - 1 or x0 > r.x1 + 1:
                    continue
                cs = {c for c in w if c.isalnum()}
                if r.y0 - _BAR_WINDOW_PT <= y1 <= r.y0 + 0.5:
                    above |= cs
                if r.y1 - 0.5 <= y0 <= r.y1 + _BAR_WINDOW_PT:
                    below |= cs
            universe = set(need)
            if not a_chars <= above or (above & universe) - a_chars:
                raise FormulaStructureError(
                    f"分数层：分子区字符 {sorted(above)} 与源串 A={sorted(a_chars)} 不符"
                    "（错位分组）——不落盘"
                )
            if not b_chars <= below or (below & universe) - b_chars:
                raise FormulaStructureError(
                    f"分数层：分母区字符 {sorted(below)} 与源串 B={sorted(b_chars)} 不符"
                    "（分母吞并后继记号）——不落盘"
                )


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
        verify_formula_structure(body, res.pdf_path)
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
