r"""出版规范层：期刊风格预设、尺寸、字体链与 mathtext 约定。

这是科研绘图技能的可复用核心——把"发表级插图"的排版契约固化为一组 matplotlib rcParams
预设，任何一幅图的管线只需 ``with style_context("aps", width="double"):`` 即可合规。

固化的规范要点：
- **设计宽度**：单栏 / 双栏（APS: 85 / 170 mm；Nature: 89 / 183 mm）。
- **字体**：APS 用 Arial；Nature 用 Helvetica（带回退链 TeX Gyre Heros → Arial，
  未装 Helvetica 时自动降级，不报错）。
- **字号**：默认 8pt（Nature 7pt），刻度略小。
- **正斜体约定**：物理变量/几何参数用 mathtext 斜体（``$x$``、``$L_\mathrm{c}$``），
  单位与物理常数正体（``$\mathrm{mm}$``、``$\mathrm{k}$``）；矢量/矩阵加粗
  （``$\mathbf{M}$``、``$\boldsymbol{v}$``）。mathtext.fontset 设为 dejavusans 以匹配无衬线正文。
- **可编辑文字**：``pdf.fonttype=42`` / ``ps.fonttype=42``（TrueType 嵌入，EPS/PDF 文字不转曲）；
  ``svg.fonttype="none"``（SVG 保留真实文字，便于人工微调）。
- **色盲友好**：默认套用 Okabe-Ito 调色板（见 palette.py）。

用法::

    from pysci.skills.scientific_plotting.tools.style import style_context, apply_style

    with style_context("aps", width="double") as info:
        fig, ax = plt.subplots()
        ...
        # info.resolved_font -> 实际生效的字体名
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# 尺寸常量（毫米）——出版设计宽度
# ---------------------------------------------------------------------------
#: 1 mm 对应的英寸（matplotlib figsize 以英寸计）。
MM_PER_INCH: float = 25.4


def mm_to_inch(mm: float) -> float:
    """毫米 → 英寸（matplotlib 的 figsize 单位）。"""
    return mm / MM_PER_INCH


# ---------------------------------------------------------------------------
# 字体可用性发现
# ---------------------------------------------------------------------------
def available_font_names() -> set[str]:
    """返回 matplotlib 当前可发现的全部字体族名集合。"""
    return {f.name for f in font_manager.fontManager.ttflist}


def font_available(name: str) -> bool:
    """某字体族名是否可被 matplotlib 使用。"""
    return name in available_font_names()


def resolve_font(chain: list[str]) -> tuple[str | None, list[str]]:
    """在字体链中挑第一个可用字体。

    Returns:
        (resolved, missing)：resolved 为实际生效字体名（都不可用时为 None，matplotlib
        会回退到 rcParams 默认）；missing 为链中本机缺失的字体名，供 doctor/audit 报告。
    """
    avail = available_font_names()
    missing = [n for n in chain if n not in avail]
    for n in chain:
        if n in avail:
            return n, missing
    return None, missing


# ---------------------------------------------------------------------------
# 期刊风格预设
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    """一套期刊排版规范。"""

    name: str                     # 预设键（aps / nature）
    label: str                    # 人类可读描述
    font_chain: list[str]         # 字体回退链（首个可用者生效）
    base_fontsize: float          # 正文字号（pt）
    tick_fontsize: float          # 刻度字号（pt）
    single_width_mm: float        # 单栏设计宽度
    double_width_mm: float        # 双栏设计宽度
    #: 与首选字体度量兼容、可接受的替代字体（解析到它们不算"降级"，audit 不告警）。
    accepted_substitutes: tuple[str, ...] = ()
    extra_rc: dict[str, object] = field(default_factory=dict)  # 预设专属 rcParams 覆盖

    def width_mm(self, width: str) -> float:
        """按 ``single`` / ``double`` 解析设计宽度（mm）。也接受直接的数值字符串。"""
        key = width.lower()
        if key in ("single", "1", "s"):
            return self.single_width_mm
        if key in ("double", "2", "d", "full"):
            return self.double_width_mm
        # 允许显式毫米数，如 "120"
        try:
            return float(key)
        except ValueError as e:
            raise ValueError(
                f"width 必须是 'single'/'double' 或毫米数值，收到 {width!r}"
            ) from e


# APS / PRL（revtex 系）：Arial 8pt，单栏 85mm / 双栏 170mm。
_APS = Preset(
    name="aps",
    label="APS / PRL (revtex): Arial 8pt, 85/170 mm",
    font_chain=["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    base_fontsize=8.0,
    tick_fontsize=7.0,
    single_width_mm=85.0,
    double_width_mm=170.0,
    accepted_substitutes=("Helvetica", "Liberation Sans"),
    extra_rc={
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    },
)

# Nature 系：Helvetica 7pt，单栏 89mm / 双栏 183mm。
# Helvetica 为商业字体；未安装时回退链自动降级到 TeX Gyre Heros（度量兼容克隆）或 Arial。
_NATURE = Preset(
    name="nature",
    label="Nature / Science: Helvetica 7pt, 89/183 mm",
    font_chain=["Helvetica", "TeX Gyre Heros", "Nimbus Sans", "Arial", "DejaVu Sans"],
    base_fontsize=7.0,
    tick_fontsize=6.0,
    single_width_mm=89.0,
    double_width_mm=183.0,
    accepted_substitutes=("TeX Gyre Heros", "Nimbus Sans"),
    extra_rc={
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    },
)

_PRESETS: dict[str, Preset] = {p.name: p for p in (_APS, _NATURE)}

#: 所有预设共享的基础 rcParams（与字体无关的排版契约）。
_BASE_RC: dict[str, object] = {
    # 文字保持可编辑（不转曲）——便于人工在 Illustrator/Inkscape 里微调
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # mathtext 匹配无衬线正文；变量斜体/单位正体由 $...$ 里的 \mathrm{} 控制
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "axes.unicode_minus": False,
    # 版面
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.family": "sans-serif",
}


def preset(name: str) -> Preset:
    """按名取预设；未知名报错并列出可选项。"""
    key = (name or "").lower()
    if key not in _PRESETS:
        avail = ", ".join(sorted(_PRESETS))
        raise KeyError(f"未知风格预设 {name!r}；可选：{avail}")
    return _PRESETS[key]


def list_presets() -> list[Preset]:
    """返回全部预设（供 CLI ``styles`` 列出）。"""
    return list(_PRESETS.values())


# ---------------------------------------------------------------------------
# rcParams 组装 / 应用
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StyleInfo:
    """一次风格应用的结果摘要（供 doctor/audit/日志使用）。"""

    preset: str
    width: str
    width_mm: float
    figsize_in: tuple[float, float]
    resolved_font: str | None
    missing_fonts: list[str]
    palette: str

    def report(self) -> str:
        lines = [
            f"style preset    : {self.preset}",
            f"design width    : {self.width} ({self.width_mm:g} mm)",
            f"figsize         : {self.figsize_in[0]:.3f} x {self.figsize_in[1]:.3f} in",
            f"resolved font   : {self.resolved_font or '(matplotlib default)'}",
            f"missing fonts   : {', '.join(self.missing_fonts) or '(none)'}",
            f"palette         : {self.palette}",
        ]
        return "\n".join(lines)


def build_rcparams(
    preset_name: str = "aps",
    *,
    width: str = "double",
    aspect: float = 0.618,
    fontsize: float | None = None,
    palette: str = "okabe-ito",
) -> tuple[dict[str, object], StyleInfo]:
    """组装一套完整 rcParams，并返回其 StyleInfo 摘要。

    Args:
        preset_name: 期刊预设名（aps / nature）。
        width: 'single' / 'double' 或显式毫米数。
        aspect: 高/宽比（默认黄金比 0.618）；决定 figsize 高度。
        fontsize: 覆盖预设正文字号。
        palette: 色盲友好调色板名（见 palette.py）。
    """
    from . import palette as _palette  # 延迟导入：避免与 palette 形成环

    p = preset(preset_name)
    width_mm = p.width_mm(width)
    w_in = mm_to_inch(width_mm)
    h_in = w_in * aspect
    base_fs = fontsize if fontsize is not None else p.base_fontsize

    resolved, missing = resolve_font(p.font_chain)

    rc: dict[str, object] = dict(_BASE_RC)
    rc.update(p.extra_rc)
    rc["font.sans-serif"] = list(p.font_chain)
    rc["font.size"] = base_fs
    rc["axes.labelsize"] = base_fs
    rc["axes.titlesize"] = base_fs
    rc["xtick.labelsize"] = p.tick_fontsize
    rc["ytick.labelsize"] = p.tick_fontsize
    rc["figure.figsize"] = [w_in, h_in]
    rc["figure.dpi"] = 100
    # 色盲友好颜色循环
    colors = _palette.get_palette(palette)
    rc["axes.prop_cycle"] = f"cycler('color', {list(colors)!r})"

    info = StyleInfo(
        preset=p.name,
        width=width,
        width_mm=width_mm,
        figsize_in=(w_in, h_in),
        resolved_font=resolved,
        missing_fonts=missing,
        palette=palette,
    )
    return rc, info


def apply_style(
    preset_name: str = "aps",
    *,
    width: str = "double",
    aspect: float = 0.618,
    fontsize: float | None = None,
    palette: str = "okabe-ito",
) -> StyleInfo:
    """就地更新全局 ``plt.rcParams`` 并返回 StyleInfo。

    多数场景更推荐用 ``style_context``（临时生效、退出即还原），避免污染全局状态。
    """
    rc, info = build_rcparams(
        preset_name,
        width=width,
        aspect=aspect,
        fontsize=fontsize,
        palette=palette,
    )
    plt.rcParams.update(rc)
    return info


@contextmanager
def style_context(
    preset_name: str = "aps",
    *,
    width: str = "double",
    aspect: float = 0.618,
    fontsize: float | None = None,
    palette: str = "okabe-ito",
) -> Iterator[StyleInfo]:
    """上下文管理器：临时套用某期刊风格，退出后还原 rcParams。

    这是图管线里推荐的用法::

        with style_context("aps", width="double") as info:
            fig, ax = plt.subplots()   # figsize 已按设计宽度设好
            ...
    """
    rc, info = build_rcparams(
        preset_name,
        width=width,
        aspect=aspect,
        fontsize=fontsize,
        palette=palette,
    )
    with plt.rc_context(rc):
        yield info
