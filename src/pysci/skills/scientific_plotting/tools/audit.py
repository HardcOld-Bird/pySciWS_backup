"""出版规范自检：宽度、字号、字体嵌入、面板标号、色盲可读性。

发表级插图最常见的翻车点——图宽与设计宽度不符、某处字号过小、EPS 文字被转曲、
配色在红绿色盲下不可分——都能被静态检查提前抓到。``audit_figure`` 在**已绘制的
Figure 对象**上做这些检查（无需导出），``audit_figure_dir`` 则复用 runner 在
style_context 下构建图再自检，一条 CLI 命令 ``figures audit <figdir>`` 即可跑完整报告。

色盲模拟采用 Viénot–Brettel–Mollon (1999) 的线性 RGB 变换近似 deuteranopia / protanopia，
对图中出现的主色两两做变换后距离检查，过近者告警。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
from matplotlib import text as mtext
from matplotlib.figure import Figure

from .style import StyleInfo, preset

# 级别：ERROR 必须修，WARN 建议修，INFO 仅提示。
ERROR, WARN, INFO = "ERROR", "WARN", "INFO"

_PANEL_LABEL_RE = re.compile(r"^\(?[a-zA-Z]\)?$")


@dataclass
class AuditIssue:
    level: str
    code: str
    message: str

    def line(self) -> str:
        return f"[{self.level:<5}] {self.code}: {self.message}"


@dataclass
class AuditReport:
    figdir: Path | None = None
    style: str | None = None
    issues: list[AuditIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(AuditIssue(level, code, message))

    def report(self) -> str:
        head = f"=== figure audit: {self.figdir or '(in-memory)'} ==="
        lines = [head]
        if self.style:
            lines.append(f"style preset: {self.style}")
        if self.metrics:
            lines.append("--- metrics ---")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        lines.append("--- issues ---")
        if not self.issues:
            lines.append("  (none) ✓")
        else:
            lines.extend("  " + i.line() for i in self.issues)
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(f"verdict: {verdict} ({len(self.errors)} error, "
                     f"{len([i for i in self.issues if i.level == WARN])} warn)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 色盲模拟（Viénot–Brettel–Mollon 1999，线性 RGB 近似）
# ---------------------------------------------------------------------------
_DEUTAN = np.array([
    [0.29275, 0.70725, 0.0],
    [0.29275, 0.70725, 0.0],
    [-0.02234, 0.02234, 1.0],
])
_PROTAN = np.array([
    [0.11238, 0.88762, 0.0],
    [0.11238, 0.88762, 0.0],
    [-0.00401, 0.00401, 1.0],
])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(rgb: np.ndarray, kind: str = "deutan") -> np.ndarray:
    """把 (N,3) sRGB[0,1] 数组变换为色觉障碍者所见（近似）。"""
    m = _PROTAN if kind == "protan" else _DEUTAN
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float).reshape(-1, 3))
    sim = lin @ m.T
    return _linear_to_srgb(sim)


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _collect_colors(fig: Figure) -> list[tuple[float, float, float]]:
    """收集图中主要数据颜色（线条 + 散点/集合 facecolor），去重。"""
    colors: set[tuple[int, int, int]] = set()
    for line in fig.findobj(matplotlib.lines.Line2D):
        c = line.get_color()
        try:
            rgb = matplotlib.colors.to_rgb(c)
        except (ValueError, TypeError):
            continue
        colors.add(tuple(int(round(v * 255)) for v in rgb))  # type: ignore[arg-type]
    for coll in fig.findobj(matplotlib.collections.PathCollection):
        for fc in coll.get_facecolors():
            colors.add(tuple(int(round(v * 255)) for v in fc[:3]))  # type: ignore[arg-type]
    return [(r / 255, g / 255, b / 255) for r, g, b in colors]


def _check_colorblind(fig: Figure, rep: AuditReport, *, thresh: float = 0.12) -> None:
    """检查数据色在 deuteranopia 下是否两两可分。"""
    cols = _collect_colors(fig)
    # 剔除接近黑/白/灰的颜色（它们靠明度区分，色盲下通常仍可辨）
    data_cols = [
        c for c in cols
        if not (max(c) - min(c) < 0.08)  # 非灰
    ]
    if len(data_cols) < 2:
        rep.metrics["colorblind_pairs"] = 0
        return
    arr = np.array(data_cols)
    sim = simulate_cvd(arr, "deutan")
    bad = 0
    n = len(sim)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((sim[i] - sim[j]) ** 2)))
            if d < thresh:
                bad += 1
    rep.metrics["colorblind_pairs"] = n * (n - 1) // 2
    rep.metrics["colorblind_confusable_pairs"] = bad
    if bad:
        rep.add(
            WARN,
            "colorblind",
            f"{bad} 对数据色在红绿色盲(deuteranopia)模拟下过于接近(<{thresh})，"
            f"建议改用 palette.py 的色盲安全调色板或叠加不同线型/marker",
        )


# ---------------------------------------------------------------------------
# 主检查
# ---------------------------------------------------------------------------
def _iter_texts(fig: Figure) -> Iterable[mtext.Text]:
    for t in fig.findobj(mtext.Text):
        s = (t.get_text() or "").strip()
        if s:
            yield t


def audit_figure(
    fig: Figure,
    *,
    style_info: StyleInfo | None = None,
    style_name: str | None = None,
    target_width_mm: float | None = None,
    width_tol_mm: float = 2.0,
    min_fontsize: float | None = None,
    expect_panel_labels: bool = False,
    check_colorblind: bool = True,
) -> AuditReport:
    """对一个已绘制的 Figure 做规范自检。

    Args:
        fig: 目标 Figure。
        style_info: 由 style_context 产出的 StyleInfo（提供设计宽度/字体链）。
        style_name: 若无 style_info，可按预设名推断目标宽度。
        target_width_mm: 显式指定目标宽度（优先级最高）。
        width_tol_mm: 宽度容差（mm）。constrained/tight 布局会带来轻微偏差。
        min_fontsize: 最小允许字号（pt）；None 时取预设 base 字号的 0.75 倍。
        expect_panel_labels: 是否要求多子图带 (a)(b)(c) 角标。
        check_colorblind: 是否做色盲可读性检查。
    """
    rep = AuditReport(style=(style_info.preset if style_info else style_name))

    # --- 目标宽度 ---
    if target_width_mm is None:
        if style_info is not None:
            target_width_mm = style_info.width_mm
        elif style_name:
            p = preset(style_name)
            target_width_mm = p.double_width_mm
    if target_width_mm is not None:
        actual_mm = fig.get_size_inches()[0] * 25.4
        rep.metrics["width_mm"] = round(actual_mm, 2)
        rep.metrics["target_width_mm"] = target_width_mm
        if abs(actual_mm - target_width_mm) > width_tol_mm:
            rep.add(
                ERROR,
                "width",
                f"图宽 {actual_mm:.1f}mm 偏离设计宽度 {target_width_mm:.1f}mm "
                f"(>±{width_tol_mm}mm)；请在 style_context 里用正确的 width 参数",
            )

    # --- 最小字号 ---
    if min_fontsize is None and style_info is not None:
        # 从预设推断：base 字号的 0.75 倍作为下限（刻度通常略小）
        p = preset(style_info.preset)
        min_fontsize = p.tick_fontsize * 0.9
    if min_fontsize is None:
        min_fontsize = 5.0
    smallest = None
    smallest_txt = ""
    for t in _iter_texts(fig):
        fs = t.get_fontsize()
        if smallest is None or fs < smallest:
            smallest = fs
            smallest_txt = t.get_text()[:20]
    if smallest is not None:
        rep.metrics["min_fontsize_pt"] = round(smallest, 2)
        rep.metrics["min_fontsize_text"] = smallest_txt
        if smallest < min_fontsize:
            rep.add(
                ERROR,
                "fontsize",
                f"存在字号 {smallest:.1f}pt < 下限 {min_fontsize:.1f}pt 的文本"
                f"（'{smallest_txt}'）；期刊缩印后将难以辨认",
            )

    # --- 字体嵌入 / 可编辑文字 ---
    rc = matplotlib.rcParams
    pdf_ft = rc.get("pdf.fonttype")
    ps_ft = rc.get("ps.fonttype")
    svg_ft = rc.get("svg.fonttype")
    rep.metrics["pdf.fonttype"] = pdf_ft
    rep.metrics["ps.fonttype"] = ps_ft
    rep.metrics["svg.fonttype"] = svg_ft
    if pdf_ft != 42 or ps_ft != 42:
        rep.add(
            WARN,
            "font-embed",
            f"pdf/ps.fonttype={pdf_ft}/{ps_ft}（应为 42）；否则 EPS/PDF 文字会转曲，无法编辑",
        )
    if svg_ft not in ("none", None):
        rep.add(
            WARN,
            "svg-text",
            f"svg.fonttype={svg_ft!r}（建议 'none'）；否则 SVG 文字转路径，不利人工微调",
        )

    # --- 字体链缺失 ---
    # 仅当解析结果既非首选、也非预设认可的度量兼容替代（accepted_substitutes）时才告警；
    # 解析到认可替代（如 nature 用 TeX Gyre Heros 代替 Helvetica）属预期，不报。
    if style_info is not None and style_info.missing_fonts:
        p = preset(style_info.preset)
        acceptable = (p.font_chain[0], *p.accepted_substitutes)
        if style_info.resolved_font not in acceptable:
            rep.add(
                WARN,
                "font-missing",
                f"首选字体 '{p.font_chain[0]}' 未安装，已回退到 '{style_info.resolved_font}'；"
                f"缺失：{', '.join(style_info.missing_fonts)}",
            )

    # --- 面板标号 ---
    axes = fig.get_axes()
    rep.metrics["n_axes"] = len(axes)
    if expect_panel_labels and len(axes) > 1:
        labels = [
            t.get_text().strip() for t in _iter_texts(fig) if _PANEL_LABEL_RE.match(t.get_text().strip())
        ]
        rep.metrics["panel_labels_found"] = len(labels)
        if len(labels) < len(axes):
            rep.add(
                WARN,
                "panel-labels",
                f"检测到 {len(axes)} 个子图但仅 {len(labels)} 个 (a)/(b) 角标；"
                f"可用 layout.label_panels(axes) 批量补齐",
            )

    # --- 色盲可读性 ---
    if check_colorblind:
        _check_colorblind(fig, rep)

    return rep


def audit_figure_dir(
    figdir: Path | str,
    *,
    style: str | None = None,
    width: str | None = None,
    expect_panel_labels: bool = False,
    min_fontsize: float | None = None,
) -> AuditReport:
    """在 style_context 下构建某图目录的管线并自检（不导出）。"""
    from . import runner as _runner

    figdir = Path(figdir)
    try:
        with _runner.built_figure(figdir, style=style, width=width) as (fig, info):
            rep = audit_figure(
                fig,
                style_info=info,
                expect_panel_labels=expect_panel_labels,
                min_fontsize=min_fontsize,
            )
        rep.figdir = figdir
        return rep
    except Exception as e:  # noqa: BLE001
        rep = AuditReport(figdir=figdir, style=style)
        rep.add(ERROR, "build", f"构建管线失败：{e!r}")
        return rep
