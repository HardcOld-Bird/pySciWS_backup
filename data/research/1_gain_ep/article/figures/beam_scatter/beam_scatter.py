"""beam_scatter 综合图（返工第 3 轮）：3 情形列 x (行1 L 场图 / 行2 R 场图 / 行3 雷达)。

几何语义（任务书硬性条款 1，机械判据见同目录 ``ANNOTATIONS.yaml``）：

- 每侧**仅 2 个框**。内框 = 实验扫描场区（正方形边长 31 cm、近边距超表面中心 30 cm），
  与场图**同一比例尺**（数据坐标即米），完整位于半径 R = 70 cm 的半圆域内并留边距；
- 外框 = 同区放大插图（边长 38 cm，> 内框），置于半圆外、完整落在面板边界内；
  内框远侧两顶点 -> 外框近侧两顶点以灰色细虚线连接（对应角）；
- ``Exp.``（带 † 代填角标）/``Sim.`` 标签一律在框外空白区，仅第一列标注。

数据源（唯一）：``raw/far_<cfg>_<inc>.csv``（181 点 / 1 deg）与
``raw/field_<cfg>_<inc>.csv``（x, y, Re p_t；R = 7 lambda 扩域后重算）。

用法（复跑入口在 scripts/，见 ``finalize_beam_scatter_figure.py``）::

    pysci-figures build data/research/1_gain_ep/article/figures/beam_scatter
"""

from __future__ import annotations

import math

import numpy as np

from pysci.paths import research_fig_dir
from pysci.skills.scientific_plotting.tools import field as _field

FIGDIR = research_fig_dir("gain_ep", slug="beam_scatter")
RAW_DIR = FIGDIR / "raw"

# --- 模型几何（与仿真模型一致，单位 m）---
LAMBDA = 343.0 / 3430.0
D = LAMBDA / math.sqrt(2.0)
X0 = 4.0 * D  # 超表面中心 / 半圆域圆心
R_DOMAIN = 0.70  # 扩域后半径 7 lambda

CASES = (
    ("full", "(i) groove + gain"),
    ("nogain", "(ii) groove, no gain"),
    ("nostruct", "(iii) gain, no groove"),
)
INCS = (("L", "L-incidence"), ("R", "R-incidence"))
CH_ANG = (135.0, 45.0)  # 左 / 右输出通道

# --- 双框：内框 = 31 cm 扫描区（真尺度）；外框 = 放大插图（> 内框，半圆外）---
SCAN_SIDE = 0.31
SCAN_NEAR_EDGE = 0.30
SCAN_HALF = SCAN_SIDE / 2.0
SCAN_CENTER_R = SCAN_NEAR_EDGE + SCAN_HALF  # 45.5 cm
OUTER_SIDE = 0.38
OUTER_HALF = OUTER_SIDE / 2.0
OUTER_NEAR_EDGE = R_DOMAIN + 0.06  # 近边距弧 6 cm（框整体在半圆外，肉眼可辨间隙）
OUTER_CENTER_R = OUTER_NEAR_EDGE + OUTER_HALF  # 95 cm
MAG = OUTER_HALF / SCAN_HALF  # 放大倍率（= 38/31）

# --- 面板数据窗口（宽 2*PANEL_HALF、高 PANEL_TOP；宽高比约 1.74:1）---
PANEL_HALF = 1.06
PANEL_TOP = 1.22

INK, GRAY, GRID = "#222222", "#666666", "#B3B3B3"
C_L, C_R = "#0072B2", "#D55E00"

# COMSOL Rainbow：0 值落黄绿（非白）；负向深蓝/青、正向黄/红。
CMAP_NAME = "comsol_rainbow"
_CMAP_STOPS = [
    (0.00, "#1b3f9e"),  # 深蓝 −vmax
    (0.25, "#35c6d9"),  # 青
    (0.50, "#b5e02a"),  # 黄绿（零）
    (0.75, "#ffd21e"),  # 黄
    (1.00, "#d81e2a"),  # 红 +vmax
]
VMAX = 50.0  # 共享对称量程（数据 max 40.94 -> 50；symlog 保小量可见）
LINTHRESH = 1.0
CBAR_TICKS = (-50.0, -5.0, 0.0, 5.0, 50.0)

_MANTISSA = (1.0, 1.2, 1.5, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def _nice_ceil(x: float) -> float:
    """取 >= x 的最小整洁值（mantissa x 10^n），供雷达径向上界。"""
    if x <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(x))
    for v in sorted(m * ee for ee in (e / 100, e / 10, e, e * 10) for m in _MANTISSA):
        if v >= x * (1.0 - 1e-12):
            return v
    return 10.0 * e


def _nice_floor(x: float) -> float:
    """取 <= x 的最大整洁值，供雷达环刻度。"""
    if x <= 0:
        return 0.0
    e = 10.0 ** math.floor(math.log10(x))
    ok = [
        v
        for v in sorted(
            m * ee for ee in (e / 100, e / 10, e, e * 10) for m in _MANTISSA
        )
        if v <= x * (1.0 + 1e-12)
    ]
    return ok[-1] if ok else 0.0


def _fmt(v: float) -> str:
    return f"{v:g}"


def field_norm():
    """共享色标 norm：对称 symlog（0 落色标中位；|v|<=1 Pa 线性、以上对数）。"""
    from matplotlib.colors import SymLogNorm  # noqa: PLC0415

    return SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=-VMAX, vmax=VMAX, base=10)


def field_cmap():
    """共享色标 cmap：COMSOL Rainbow，0 值呈黄绿（非白）。"""
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    return LinearSegmentedColormap.from_list(CMAP_NAME, _CMAP_STOPS)


def load_far(cfg: str, inc: str) -> tuple[np.ndarray, np.ndarray]:
    """读远场 CSV -> (alpha deg, |p_ext| Pa)，1 deg 网格不降采样。"""
    arr = np.loadtxt(RAW_DIR / f"far_{cfg}_{inc}.csv", delimiter=",", comments="%")
    return arr[:, 2], arr[:, 3]


def load_field(cfg: str, inc: str) -> tuple[np.ndarray, np.ndarray]:
    """读场图 CSV -> (xy Nx2 m, Re p_t N Pa)。"""
    return _field.load_field_points(RAW_DIR / f"field_{cfg}_{inc}.csv")


def _frame(ang_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(ang_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _square_uv(dist: float, half: float, ang_deg: float) -> np.ndarray:
    """局部 (u,v) 系中、中心距 X0 为 dist、半边长 half 的正方形 4 顶点。"""
    u, v = _frame(ang_deg)
    c = np.array([X0, 0.0]) + dist * u
    return np.array(
        [
            c + su * half * u + sv * half * v
            for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def _in_half_disc(xy: np.ndarray) -> np.ndarray:
    """半圆流体域判据（y>=0 且 r<=R）；管槽（y<0）按禁回退清单隐去。"""
    return (xy[:, 1] >= 0.0) & (
        (xy[:, 0] - X0) ** 2 + xy[:, 1] ** 2 <= R_DOMAIN**2 + 1e-12
    )


def draw_side_annotations(
    ax, xy, vals, cmap, norm, ang: float, side: str, view: str, *, labels: bool
):
    """每侧：内框（31 cm 扫描区）+ 外框（放大插图）+ 两角虚线 + 框外标签。

    内外框同为 (u,v) 朝向的正方形 —— 外框是内框同向放大（对应角可直连）。
    """
    from matplotlib.patches import Polygon  # noqa: PLC0415

    u, v = _frame(ang)
    off = np.array([X0, 0.0])
    inner = _square_uv(SCAN_CENTER_R, SCAN_HALF, ang)
    outer = _square_uv(OUTER_CENTER_R, OUTER_HALF, ang)

    # 外框内容 = 内框区域的场，同向放大 MAG 倍
    duv = np.stack([(xy - off) @ u, (xy - off) @ v], axis=1)
    m = (
        (duv[:, 0] >= SCAN_NEAR_EDGE - 1e-9)
        & (duv[:, 0] <= SCAN_NEAR_EDGE + SCAN_SIDE + 1e-9)
        & (np.abs(duv[:, 1]) <= SCAN_HALF + 1e-9)
    )
    if m.sum() >= 3:
        uu = OUTER_CENTER_R + (duv[m, 0] - SCAN_CENTER_R) * MAG
        vv = duv[m, 1] * MAG
        mapped = off + uu[:, None] * u + vv[:, None] * v
        tri = _field.add_field_panel(
            ax, mapped, vals[m], cmap=cmap, norm=norm, keep_triangle=None
        )
        tri.set_clip_path(Polygon(outer, closed=True, transform=ax.transData))

    # 两角细虚线：内框远侧顶点 -> 外框近侧顶点（同 sign(v) 为对应角）
    far_a = off + (SCAN_NEAR_EDGE + SCAN_SIDE) * u + SCAN_HALF * v
    far_b = off + (SCAN_NEAR_EDGE + SCAN_SIDE) * u - SCAN_HALF * v
    near_a = off + OUTER_NEAR_EDGE * u + OUTER_HALF * v
    near_b = off + OUTER_NEAR_EDGE * u - OUTER_HALF * v
    for a, b in ((far_a, near_a), (far_b, near_b)):
        (line,) = ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            ls=(0, (3, 2)),
            lw=0.6,
            color=GRAY,
            zorder=5,
            solid_capstyle="butt",
        )
        line.set_gid(f"annot:{view}_{side}_connector;role=connector")

    inner_patch = Polygon(inner, closed=True, fill=False, ec=GRAY, lw=0.6, zorder=6)
    inner_patch.set_gid(f"annot:{view}_{side}_inner;role={side}_inner;kind=box")
    ax.add_patch(inner_patch)
    outer_patch = Polygon(outer, closed=True, fill=False, ec=GRAY, lw=0.6, zorder=6)
    outer_patch.set_gid(f"annot:{view}_{side}_outer;role={side}_outer;kind=box")
    ax.add_patch(outer_patch)

    if labels:
        cxo = float(outer[:, 0].mean())
        ax.text(
            cxo,
            PANEL_TOP - 0.105,
            "Exp.$^{\\dagger}$",
            color=GRAY,
            fontsize=6.5,
            ha="center",
            va="center",
            zorder=8,
        )
        ax.text(
            X0,
            0.82,
            "Sim.",
            color=GRAY,
            fontsize=6.5,
            ha="center",
            va="center",
            zorder=8,
        )


def draw_field_panel(
    ax, cfg: str, inc: str, title: str, data, cmap, norm, *, labels: bool
):
    """场图面板：半圆域 + 双框 + 平直直径边。"""
    from matplotlib.patches import Polygon, Rectangle  # noqa: PLC0415

    xy, vals = data
    th = np.linspace(0.0, math.pi, 361)
    arc = np.column_stack((X0 + R_DOMAIN * np.cos(th), R_DOMAIN * np.sin(th)))
    tri = _field.add_field_panel(
        ax, xy, vals, cmap=cmap, norm=norm, keep_triangle=_in_half_disc
    )
    tri.set_clip_path(Polygon(arc, closed=True, transform=ax.transData))

    view = f"{cfg}_{inc}"
    domain = Polygon(arc, closed=True, fill=False, ec="#333333", lw=0.6, zorder=6)
    domain.set_gid(f"annot:{view}_domain;role=domain;kind=polygon")
    ax.add_patch(domain)
    # 直径边：一条平直直线盖住管槽轮廓（下方不绘制任何内容）
    ax.plot(
        [X0 - R_DOMAIN, X0 + R_DOMAIN],
        [0.0, 0.0],
        color=INK,
        lw=0.9,
        zorder=7,
        solid_capstyle="butt",
    )

    panel = Rectangle(
        (X0 - PANEL_HALF, 0.0),
        2 * PANEL_HALF,
        PANEL_TOP,
        fill=False,
        ec="#dddddd",
        lw=0.4,
        zorder=0.5,
    )
    panel.set_gid(f"annot:{view}_panel;role=panel")
    ax.add_patch(panel)

    for ang, side in zip(CH_ANG, ("left", "right"), strict=True):
        draw_side_annotations(ax, xy, vals, cmap, norm, ang, side, view, labels=labels)

    ax.set_xlim(X0 - PANEL_HALF, X0 + PANEL_HALF)
    ax.set_ylim(0.0, PANEL_TOP)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    # y 显式给定以关闭自动标题定位：inset colorbar 子轴会把 auto 标题挪到父轴顶部。
    ax.set_title(title, color=INK, pad=1.5, fontsize=7, y=1.0)
    ax.set_gid(f"view:{cfg}_{inc}")


def draw_radar(ax, cfg: str) -> None:
    """上半极坐标雷达（0-180 deg）：L/R 双色、1 deg 原始点、线性径标。"""
    far = {inc: load_far(cfg, inc) for inc, _ in INCS}
    rmax = _nice_ceil(max(float(a.max()) for _al, a in far.values()))
    rings = sorted(
        {
            r
            for r in (_nice_floor(rmax / 3.0), _nice_floor(2.0 * rmax / 3.0), rmax)
            if r > 0
        }
    )

    th = np.linspace(0.0, math.pi, 241)
    for rc in rings:
        ax.plot(rc * np.cos(th), rc * np.sin(th), color=GRID, lw=0.45, zorder=1)
    for ang in range(0, 181, 30):
        a = math.radians(ang)
        ax.plot(
            [0.0, rmax * math.cos(a)],
            [0.0, rmax * math.sin(a)],
            color=GRID,
            lw=0.45,
            zorder=1,
        )
    # 角度标注：30..150 在外弧上方；0/180 置于直径边端点内侧上方（不溢出轴框）
    for ang in (30, 60, 90, 120, 150):
        a = math.radians(ang)
        ax.text(
            rmax * 1.06 * math.cos(a),
            rmax * 1.06 * math.sin(a),
            f"{ang}$^\\circ$",
            color=GRAY,
            fontsize=6.5,
            ha="center",
            va="bottom",
            zorder=4,
        )
    for ang, ha in ((0, "left"), (180, "right")):
        a = math.radians(ang)
        ax.text(
            rmax * 0.99 * math.cos(a),
            rmax * 0.035,
            f"{ang}$^\\circ$",
            color=GRAY,
            fontsize=6.5,
            ha=ha,
            va="bottom",
            zorder=4,
        )

    for inc, inc_label in INCS:
        alpha, amp = far[inc]
        a = np.radians(alpha)
        ax.plot(
            amp * np.cos(a),
            amp * np.sin(a),
            color=C_L if inc == "L" else C_R,
            lw=1.1,
            zorder=3,
            label=inc_label,
        )

    # 径向刻度在直径边下方（该处无图线 -> 零遮挡）
    for rc in rings:
        ax.text(
            rc,
            -0.045 * rmax,
            _fmt(rc),
            color=GRAY,
            fontsize=6.5,
            ha="center",
            va="top",
            zorder=4,
        )
    ax.text(
        -rmax * 1.12,
        -0.055 * rmax,
        "$|p_\\mathrm{ext}|$ (Pa)",
        color=GRAY,
        fontsize=6.5,
        ha="left",
        va="top",
        zorder=4,
    )
    ax.set_xlim(-rmax * 1.14, rmax * 1.14)
    ax.set_ylim(-0.16 * rmax, rmax * 1.22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.0,
    )


# ---------------------------------------------------------------------------
# 版面（mm）
# ---------------------------------------------------------------------------
FIG_W_MM = 170.0
_LEFT_MM = 5.0
_CBAR_W_MM = 3.0
_CBAR_GAP_MM = 3.0
_CBAR_LABEL_MM = 4.5
_RIGHT_MM = 4.0
_COL_GAP_MM = 4.0
_TOP_MM = 9.0
_TITLE_MM = 4.4
_TITLE_GAP_MM = 3.0
_HEADER_MM = 3.0
_HEADER_GAP_MM = 4.5
_PANEL_TITLE_MM = 3.4
_ROW_GAP_MM = 1.4  # 行1->行2：面板标题占 3.4，留给标题
_ROW2_GAP_MM = 5.0
_LEGEND_GAP_MM = 3.0
_LEGEND_MM = 3.0
_FOOT_GAP_MM = 6.0
_FOOT_LINES = 4
_FOOT_LINE_MM = 2.6
_BOTTOM_MM = 3.0


def layout() -> dict[str, float]:
    """由常量反解各行位置（mm，原点在图底）。"""
    cols_right = FIG_W_MM - _RIGHT_MM - _CBAR_LABEL_MM - _CBAR_W_MM - _CBAR_GAP_MM
    panel_w = (cols_right - _LEFT_MM - 2 * _COL_GAP_MM) / 3.0
    field_h = panel_w * (PANEL_TOP / (2.0 * PANEL_HALF))
    radar_h = panel_w * ((1.22 + 0.16) / (2.28))
    h = (
        _TOP_MM
        + _TITLE_MM
        + _TITLE_GAP_MM
        + _HEADER_MM
        + _HEADER_GAP_MM
        + field_h
        + _ROW_GAP_MM
        + _PANEL_TITLE_MM
        + field_h
        + _ROW2_GAP_MM
        + radar_h
        + _LEGEND_GAP_MM
        + _LEGEND_MM
        + _FOOT_GAP_MM
        + _FOOT_LINES * _FOOT_LINE_MM
        + _BOTTOM_MM
    )
    y = h
    y_top = y
    y -= _TOP_MM + _TITLE_MM
    y_title = y
    y -= _TITLE_GAP_MM + _HEADER_MM
    y_header = y
    y -= _HEADER_GAP_MM + field_h
    y_f1 = y
    y -= _ROW_GAP_MM + _PANEL_TITLE_MM + field_h
    y_f2 = y
    y -= _ROW2_GAP_MM + radar_h
    y_radar = y
    y_foot = _BOTTOM_MM
    return {
        "h": h,
        "panel_w": panel_w,
        "field_h": field_h,
        "radar_h": radar_h,
        "y_title": y_title,
        "y_header": y_header,
        "y_f1": y_f1,
        "y_f2": y_f2,
        "y_radar": y_radar,
        "y_foot": y_foot,
        "cols_right": cols_right,
        "y_top": y_top,
    }


def _place_letters(fig, panels: list[tuple[str, list]]) -> None:
    """按整体 bbox（含标题/图例/标注）左上角锚定子图字母。"""
    from matplotlib.axes import Axes  # noqa: PLC0415

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for letter, artists in panels:
        boxes = []
        for art in artists:
            if art is None:
                continue
            try:
                # 轴用 window_extent（不含 inset 子轴/图例），标题单独计入
                b = (
                    art.get_window_extent(rend)
                    if isinstance(art, Axes) or not hasattr(art, "get_tightbbox")
                    else art.get_tightbbox(rend)
                )
            except Exception:  # noqa: BLE001 - 不可测量图元跳过
                b = None
            if b is not None:
                boxes.append(b)
        if not boxes:
            continue
        bb = boxes[0] if len(boxes) == 1 else type(boxes[0]).union(boxes)
        x0, y1 = inv.transform((bb.x0, bb.y1))  # display -> figure coords
        fig.text(
            x0 - 0.004,
            y1 + 0.002,
            letter,
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )


def compose_figure():
    """构建整图并返回 Figure（运行器在 style_context 内调用本函数）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.cm import ScalarMappable  # noqa: PLC0415

    lay = layout()
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, lay["h"] / 25.4), facecolor="white")
    cmap = field_cmap()
    norm = field_norm()

    fields = {(cfg, inc): load_field(cfg, inc) for cfg, _ in CASES for inc, _ in INCS}

    fig.text(
        (_LEFT_MM + lay["cols_right"]) / (2.0 * FIG_W_MM),
        (lay["y_title"] + _TITLE_MM / 2.0) / lay["h"],
        "Beam scattering: groove and gain",
        color=INK,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
    )

    panels: list[tuple[str, list]] = []
    letters = iter("abcdefghi")
    for j, (cfg, header) in enumerate(CASES):
        x_mm = _LEFT_MM + j * (lay["panel_w"] + _COL_GAP_MM)
        x = x_mm / FIG_W_MM
        pw = lay["panel_w"] / FIG_W_MM
        fh = lay["field_h"] / lay["h"]
        rh = lay["radar_h"] / lay["h"]

        fig.text(
            x + pw / 2.0,
            lay["y_header"] / lay["h"],
            header,
            color=INK,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        for k, (inc, inc_label) in enumerate(INCS):
            y = (lay["y_f1"], lay["y_f2"])[k] / lay["h"]
            ax = fig.add_axes([x, y, pw, fh])
            draw_field_panel(
                ax,
                cfg,
                inc,
                f"{inc_label} $\\mathrm{{Re}}\\,p_t$",
                fields[(cfg, inc)],
                cmap,
                norm,
                labels=(j == 0),
            )
            panels.append(
                (
                    next(letters),
                    [ax, ax.title, *ax.patches, *ax.lines, *ax.texts],
                )
            )

        ax_r = fig.add_axes([x, lay["y_radar"] / lay["h"], pw, rh])
        draw_radar(ax_r, cfg)
        leg = ax_r.get_legend()
        panels.append((next(letters), [ax_r, *ax_r.lines, *ax_r.texts, leg]))

    # 唯一共享 colorbar：跨场图两行，雷达行右列留空。
    # 用 inset axes（图坐标）而非 fig.add_axes：不计入 fig.axes 的子图计数。
    cb_x = (lay["cols_right"] + _CBAR_GAP_MM) / FIG_W_MM
    cb_y = lay["y_f2"] / lay["h"]
    cb_h = (lay["y_f1"] + lay["field_h"] - lay["y_f2"]) / lay["h"]
    cax = ax.inset_axes(  # 末个场图面板仅作宿主；bounds 用图坐标
        [cb_x, cb_y, _CBAR_W_MM / FIG_W_MM, cb_h], transform=fig.transFigure
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical"
    )
    cbar.set_ticks(list(CBAR_TICKS))
    cbar.set_ticklabels([_fmt(t) for t in CBAR_TICKS])
    cbar.set_label("$\\mathrm{Re}\\,p_t$ (Pa)", rotation=270, labelpad=8.0, fontsize=7)
    cbar.outline.set_linewidth(0.5)
    cax.minorticks_off()
    cax.tick_params(
        which="both",
        direction="out",
        left=False,
        right=True,
        top=False,
        bottom=False,
        length=2.0,
        width=0.5,
        pad=2.0,
        labelsize=6.5,
    )

    _place_letters(fig, panels)

    fig.text(
        _LEFT_MM / FIG_W_MM,
        lay["y_foot"] / lay["h"],
        "$\\dagger$ Exp. diamonds show simulated data (experimental data unavailable)."
        " Diamonds mark the experiment's scan region\n"
        "(side 31 cm; near edge 30 cm from the metasurface centre) at the same scale as the field map;"
        " the enlarged diamond\n"
        "shows it magnified by 1.23. Colour is $\\mathrm{Re}\\,p_t$ on one shared symmetric scale"
        " (symlog: linear for $|\\mathrm{Re}\\,p_t|\\leq 1$ Pa,\n"
        "logarithmic above) so all six panels, including the $\\sim$13$\\times$ weaker cases (ii)/(iii),"
        " remain readable.",
        color=GRAY,
        fontsize=6.5,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    return fig


def build_figure(style=None, research_dir=None, **kwargs):  # noqa: ARG001
    """管线入口：运行器在 style_context 内调用。"""
    return compose_figure()
