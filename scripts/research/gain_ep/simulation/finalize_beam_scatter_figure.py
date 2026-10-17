"""beam-scatter 综合图（返工版）：3 情形列 × (行1 二合一雷达 / 行2 L 场图 / 行3 R 场图)。

排版（任务书第 10 项）：
  3 列 = 3 组对照情形（①groove+gain ②groove,no gain ③gain,no groove），列内自上而下
  行1 半极坐标雷达（L/R 双曲线，同一坐标系 + 小图例）、行2 L 入射半圆场图、行3 R 入射
  半圆场图；每列**右缘一条色标**，高度 = 行2+行3，三列各自独立标度。

数据源与落实点：
  - 远场：``raw/far_<cfg>_<inc>.csv``（1° 网格 181 点，角度列 + |pext| 列），**不降采样**；
    禁用旧 ``raw/far_*.png`` 栅格（COMSOL 原生渲染，带品牌 logo，且主瓣方向不可靠）；
  - 场图：``raw/field_<cfg>_<inc>.csv``（x, y, real(acpr.p_t)）；
  - 标题/色标/数据列三方一致：**Re p_t**（场图小标题、色标标签、CSV 列头同一量）；
  - 场图配色：COMSOL Rainbow 式发散映射（蓝→青→黄绿→黄→红），零值落**黄绿**（非白）；
  - 色标：刻度 5 条（±vmax、半值、0），上界 = 该情形**可见区** max|Re p_t|（不 clamp、
    不饱和），无 ± 浮动端标；色标与子图/轴标签之间留足边距；
  - 直径边：隐去管槽截面轮廓，以一条平直直线遮盖 y<0 内容；
  - 扫描场区菱形（内框，**实线**）：物理边长 31 cm、近边距超表面中心 30 cm，按模型尺度
    （R=0.4 m）换算，尖角向上，中心落在对应通道方向（135°/45°）射线，标 ``Sim.``；
    其远侧两顶点以细虚线连至外框（``Exp.†`` 插图）近侧两顶点——换算记录见
    ``返工交付说明.md``；
  - 零遮挡：雷达幅值刻度移至直径边下方、量名+单位置于直径边下方左侧，场图标注均落在
    无数据、无图线的空白区，一律不使用底色遮罩；
  - Exp. 插图：以仿真数据代填（用户既定做法），标 ``Exp.†``，脚注注明「实验数据暂缺」；
  - 设计用色：限于 Rainbow 同族（雷达双曲线取彩虹两端深蓝/深红）与中性灰阶；
  - 风格：APS 契约层 + scienceplots（science + no-latex）审美基底，契约层优先。

用法（一条命令复现主图，数据集已在 raw/ 内）::

    uv run --no-sync python scripts/research/gain_ep/simulation/finalize_beam_scatter_figure.py

可选 ``--export``：从 COMSOL 模型重新导出场 CSV（需活体 COMSOL，属 sim 环节产物）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from pysci.skills.scientific_plotting.tools import field as _field  # noqa: E402
from pysci.skills.scientific_plotting.tools.style import (  # noqa: E402
    mm_to_inch,
    style_context,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIG_BASE = (
    ROOT / "data" / "research" / "1_gain_ep" / "article" / "figures" / "beam_scatter"
)
RAW_DIR = FIG_BASE / "raw"
OUT_PNG = FIG_BASE / "beam_scatter_combined.png"

# --- 模型几何（与 build_beam_scatter_mph.PARAMS 一致，单位 m）---
LAMBDA = 343.0 / 3430.0
D = LAMBDA / math.sqrt(2.0)
X0 = 4.0 * D  # 超表面几何中心 x（半圆域圆心）
R = 4.0 * LAMBDA  # 域半径 0.4 m

CASES = [
    ("full", "(i) groove + gain"),
    ("nogain", "(ii) groove, no gain"),
    ("nostruct", "(iii) gain, no groove"),
]
INCS = (
    ("L", "L-incidence"),
    ("R", "R-incidence"),
)

# --- 菱形扫描框（外框，任务书第 5 项）：物理边长 31 cm、近边距中心 30 cm ---
SCAN_SIDE = 0.31
SCAN_NEAR_EDGE = 0.30
SCAN_HALF = SCAN_SIDE / 2.0  # 内切半径（中心 → 近边）= 15.5 cm
SCAN_CENTER_R = SCAN_NEAR_EDGE + SCAN_HALF  # 菱形中心径向距离 = 45.5 cm
CH_ANG = (135.0, 45.0)  # 左/右输出通道方位角

# --- 仿真窗口（内框）：扫描场区中落在仿真域内的部分，取最大内接正方形（留边距）---
# 约束：四顶点半径 ≤ R − GEO_MARGIN（GEO_MARGIN = 2.5 cm 边距，禁贴边）
GEO_MARGIN = 0.025
SIM_WIN_HALF = (
    0.030  # 半边长 3 cm → 边长 6 cm（r_max = √(0.36²+0.03²) = 36.1 cm，留 3.9 cm）
)
SIM_WIN_R = SCAN_NEAR_EDGE + SIM_WIN_HALF  # 窗口中心径向 = 33 cm

# 面板半宽（m）：容纳扫描框远顶点（(SCAN_CENTER_R+SCAN_HALF)/√2 = 44.5 cm）、缩略图与框外标注
PANEL_HALF = 0.80

# 缩略图（外框）：仿真窗口整区放大 2.17×（0.13 m > 窗口 0.06 m ⇒ 外框大于内框），置半圆外顶角
INSET_HALF = 0.065  # 半边长（边长 13 cm）
INSET_DIST = 0.76  # 中心径向距离（在扫描框远边之外，与框间留 8.5 cm）
#: 框外标注落位（数据坐标，域外白区；仅第一组绘制）：Sim. 在上方两框之间、Exp.† 在缩略图外
SIM_LABEL_XY = (X0, 0.50)
EXP_LABEL_DXY = (0.50, 0.70)

# --- 配色：设计用色限于 Rainbow 同族 + 中性灰阶 ---
CMAP_NAME = "comsol_rainbow"
_CMAP_STOPS = [
    (0.00, "#1b3f9e"),  # 深蓝 −vmax
    (0.25, "#35c6d9"),  # 青
    (0.50, "#b5e02a"),  # 黄绿（零）
    (0.75, "#ffd21e"),  # 黄
    (1.00, "#d81e2a"),  # 红 +vmax
]
CMAP = LinearSegmentedColormap.from_list(CMAP_NAME, _CMAP_STOPS)
C_L, C_R = "#1b3f9e", "#d81e2a"  # 雷达双曲线：彩虹两端
GRID, INK, GRAY, FAINT = "#b9b9b9", "#222222", "#666666", "#9a9a9a"

_MANTISSA_CEIL = (1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
_MANTISSA_FLOOR = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
# 雷达径向上界更贴近数据 max（任务书第 4 项），额外允许 1.25 档
_MANTISSA_RADAR = (1.0, 1.2, 1.25, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def _nice_ceil(x: float, mantissas: tuple[float, ...] = _MANTISSA_CEIL) -> float:
    """取 ≥ x 的最小「整洁」值（mantissa ×10ⁿ）。"""
    if x <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(x))
    cands = sorted(
        m * ee for ee in (e / 100, e / 10, e, e * 10, e * 100) for m in mantissas
    )
    for v in cands:
        if v >= x * (1.0 - 1e-12):
            return v
    return cands[-1]


def _nice_floor(x: float, mantissas: tuple[float, ...] = _MANTISSA_FLOOR) -> float:
    """取 ≤ x 的最大「整洁」值。"""
    if x <= 0:
        return 0.0
    e = 10.0 ** math.floor(math.log10(x))
    cands = sorted(m * ee for ee in (e / 100, e / 10, e, e * 10) for m in mantissas)
    ok = [v for v in cands if v <= x * (1.0 + 1e-12)]
    return ok[-1] if ok else 0.0


def _fmt(v: float) -> str:
    return f"{v:g}"


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_far(cfg: str, inc: str) -> tuple[np.ndarray, np.ndarray]:
    """读远场 CSV → (方位角 deg 数组, |pext| Pa 数组)。1° 网格，不降采样。"""
    arr = np.loadtxt(RAW_DIR / f"far_{cfg}_{inc}.csv", delimiter=",", comments="%")
    return arr[:, 2], arr[:, 3]


def load_field(cfg: str, inc: str) -> tuple[np.ndarray, np.ndarray]:
    """读场图 CSV → (xy Nx2, Re p_t N)。"""
    return _field.load_field_points(RAW_DIR / f"field_{cfg}_{inc}.csv")


# ---------------------------------------------------------------------------
# 菱形扫描框 / Exp 插图版式（数据坐标，米）
# ---------------------------------------------------------------------------
def _frame(ang_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """通道方向单位矢量 u（射线）与 v（横向），构成菱形的局部正交系。"""
    a = math.radians(ang_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _square_uv(dist: float, half: float, ang_deg: float) -> np.ndarray:
    """局部系 (u,v) 内以超表面中心 + dist·u 为中心、半边长 half 的正方形 4 顶点（逆时针）。"""
    u, v = _frame(ang_deg)
    c = np.array([X0, 0.0]) + dist * u
    order = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    return np.array([c + su * half * u + sv * half * v for su, sv in order])


def _at(dist: float, ang_deg: float) -> tuple[float, float]:
    """通道射线上距超表面中心 dist 的点（直角坐标）。"""
    u, _ = _frame(ang_deg)
    return X0 + dist * u[0], dist * u[1]


def _uv_point(uu: float, vv: float, ang_deg: float) -> tuple[float, float]:
    """局部系 (u,v) 中的点 → 直角坐标（原点 = 超表面中心）。"""
    u, v = _frame(ang_deg)
    return X0 + uu * u[0] + vv * v[0], uu * u[1] + vv * v[1]


def draw_exp_insets(
    ax, xy: np.ndarray, vals: np.ndarray, vmax: float, *, labels: bool
) -> None:
    """每通道：扫描场区外框 + 域内仿真窗口内框 + 两角细虚线 + 半圆外顶角 Exp.† 缩略图。

    - **外框**（扫描场区，实线）：实验扫描范围本身——边长 31 cm、近边距超表面中心 30 cm，
      尖角向上、中心落在该通道射线上；其远侧顶点（r = 62.9 cm）**如实落在仿真域（R = 40 cm）
      之外**（实验扫描区大于仿真域，这是物理事实，不是绘制错误）。
    - **内框**（仿真窗口，实线）：扫描场区中落在仿真域内的部分，取**最大内接正方形**并以
      `GEO_MARGIN` 留边距——即「仿真-实验可比区」；几何可行性已核对：整区放不下（31 cm 扫描框
      远顶点 62.9 cm > R），扩域方案已实测否决（见交付说明 W3 记录），故内框取域内可比区。
    - **缩略图**（外框放大）：内框整区放大 `INSET_HALF / SIM_WIN_HALF` ≈ 2.17×（0.13 m > 0.06 m
      ⇒ 外 > 内），置半圆外顶角，以细虚线连回内框远侧两顶点。
    - **标注**（`labels=False` 时省略）：``Sim.``/``Exp.`` 一律落在**框外**的域外白区，
      满足零遮挡；二三组省略（4ref2.png 实践）。
    """
    norm = Normalize(-vmax, vmax)
    off = np.array([X0, 0.0])
    for ang in CH_ANG:
        u, v = _frame(ang)
        scan = _square_uv(SCAN_CENTER_R, SCAN_HALF, ang)
        win = _square_uv(SIM_WIN_R, SIM_WIN_HALF, ang)
        ins = _square_uv(INSET_DIST, INSET_HALF, ang)
        # 两角细虚线：内框远侧两顶点 → 缩略图近侧两顶点
        for a, b in ((win[0], ins[3]), (win[1], ins[2])):
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                ls=(0, (2.2, 1.8)),
                lw=0.5,
                color=FAINT,
                zorder=4,
            )
        # 缩略图内容 = 内框（仿真窗口）整区的场，按比例放大到插图
        d_uv = np.stack([(xy - off) @ u, (xy - off) @ v], axis=1)
        m = (np.abs(d_uv[:, 0] - SIM_WIN_R) <= SIM_WIN_HALF) & (
            np.abs(d_uv[:, 1]) <= SIM_WIN_HALF
        )
        if m.sum() >= 3:
            w = (
                (d_uv[m] - np.array([SIM_WIN_R, 0.0])) * (INSET_HALF / SIM_WIN_HALF)
            ) @ np.stack([u, v])
            w = w + off + INSET_DIST * u
            tri = _field.add_field_panel(ax, w, vals[m], cmap=CMAP, norm=norm)
            tri.set_clip_path(Polygon(ins, closed=True, transform=ax.transData))
        ax.add_patch(Polygon(ins, closed=True, fill=False, ec=GRAY, lw=0.7, zorder=7))
        ax.add_patch(Polygon(scan, closed=True, fill=False, ec=GRAY, lw=0.8, zorder=6))
        ax.add_patch(Polygon(win, closed=True, fill=False, ec=GRAY, lw=0.8, zorder=6))
    if not labels:
        return
    # 标注：一律在框外（域外白区）；Sim. 落上方两框之间（该处无数据、无框线），Exp.† 落缩略图外
    ax.text(
        SIM_LABEL_XY[0],
        SIM_LABEL_XY[1],
        "Sim.",
        color=GRAY,
        fontsize=6,
        ha="center",
        va="center",
        zorder=8,
    )
    for sx in (-1.0, 1.0):
        ax.text(
            X0 + sx * EXP_LABEL_DXY[0],
            EXP_LABEL_DXY[1],
            "Exp.$\\dagger$",
            color=GRAY,
            fontsize=6,
            ha="center",
            va="center",
            zorder=8,
        )


def _in_half_disc(cxy: np.ndarray) -> np.ndarray:
    """可见区判据：半圆流体域（y≥0 且 r≤R）。管槽（y<0）按任务书第 8 项隐去。"""
    return (cxy[:, 1] >= 0.0) & ((cxy[:, 0] - X0) ** 2 + cxy[:, 1] ** 2 <= R**2)


# ---------------------------------------------------------------------------
# 场图面板
# ---------------------------------------------------------------------------
def draw_field_panel(
    ax,
    cfg: str,
    inc: str,
    label: str,
    vmax: float,
    data: tuple[np.ndarray, np.ndarray],
    *,
    labels: bool,
) -> None:
    xy, vals = data
    _field.add_field_panel(
        ax, xy, vals, cmap=CMAP, vmin=-vmax, vmax=vmax, keep_triangle=_in_half_disc
    )
    # 半圆外弧 + 平直直径边（隐去管槽轮廓，遮盖 y<0 内容）
    th = np.linspace(0.0, math.pi, 361)
    ax.plot(X0 + R * np.cos(th), R * np.sin(th), color=INK, lw=0.7, zorder=6)
    ax.plot(
        [X0 - R, X0 + R], [0.0, 0.0], color=INK, lw=1.0, zorder=6, solid_capstyle="butt"
    )
    draw_exp_insets(ax, xy, vals, vmax, labels=labels)
    ax.set_xlim(X0 - PANEL_HALF, X0 + PANEL_HALF)
    ax.set_ylim(0.0, PANEL_HALF)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (X0 - PANEL_HALF, 0.0),
            2 * PANEL_HALF,
            PANEL_HALF,
            fill=False,
            ec="#dddddd",
            lw=0.4,
            zorder=1,
        )
    )
    ax.set_title(label, color=INK, pad=6.0)


# ---------------------------------------------------------------------------
# 半极坐标雷达（行 1）：L/R 双曲线同坐标系 + 小图例；径向上界 = 该图数据 max
# ---------------------------------------------------------------------------
def draw_radar(ax, cfg: str, header: str) -> None:
    far = {inc: load_far(cfg, inc) for inc, _ in INCS}
    rmax = _nice_ceil(
        max(float(amp.max()) for _alpha, amp in far.values()), _MANTISSA_RADAR
    )
    rings = [_nice_floor(rmax / 3.0), _nice_floor(2.0 * rmax / 3.0), rmax]
    rings = sorted({r for r in rings if 0 < r <= rmax * (1.0 + 1e-9)})

    th = np.linspace(0.0, math.pi, 241)
    for rc in rings:
        ax.plot(rc * np.cos(th), rc * np.sin(th), color=GRID, lw=0.4, zorder=1)
    for ang in (0, 45, 90, 135, 180):
        a = math.radians(ang)
        ax.plot(
            [0.0, rmax * math.cos(a)],
            [0.0, rmax * math.sin(a)],
            color=GRID,
            lw=0.4,
            zorder=1,
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
            label=f"{inc_label} (peak {alpha[int(np.argmax(amp))]:.0f}$^\\circ$)",
        )

    ax.set_xlim(-rmax * 1.14, rmax * 1.14)
    ax.set_ylim(-0.16 * rmax, rmax * 1.22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    for ang in (0, 45, 90, 135, 180):
        a = math.radians(ang)
        ax.text(
            rmax * 1.05 * math.cos(a),
            rmax * 1.05 * math.sin(a),
            f"{ang}$^\\circ$",
            color=GRAY,
            fontsize=6,
            ha="left" if ang == 0 else ("right" if ang in (135, 180) else "center"),
            va="center"
            if ang in (0, 180)
            else ("bottom" if ang in (45, 90) else "top"),
            zorder=4,
        )
    # 幅值刻度移至直径边下方（该处无图线，满足零遮挡）
    for rc in rings:
        ax.text(
            rc,
            -0.045 * rmax,
            _fmt(rc),
            color=GRAY,
            fontsize=6,
            ha="center",
            va="top",
            zorder=4,
        )
    # 量名+单位：置于直径边下方左侧（该处无图线，且与右侧幅值刻度错开 → 零遮挡）
    ax.text(
        -rmax * 1.12,
        -0.055 * rmax,
        "$|p_\\mathrm{ext}|$ (Pa)",
        color=GRAY,
        fontsize=6,
        ha="left",
        va="top",
        zorder=4,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=1,
        fontsize=6,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.5,
        borderpad=0.1,
        labelspacing=0.25,
    )


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
_FIG_W_MM = 178.0
_LEFT, _RIGHT = 0.045, 0.945
_G_PANEL_CBAR, _G_CBAR_COL, _CB_W = 0.012, 0.066, 0.013
_TITLE_MM, _G_ROW_MM = 13.0, 11.5
_MARGIN_TOP_MM, _MARGIN_BOT_MM = 8.0, 22.0
_RADAR_ASPECT = 1.38 / 2.28  # ylim 跨度 / xlim 跨度（半极坐标雷达）


def _layout() -> dict[str, float]:
    """由列宽反解各行的图面比例（保证场图面板 2:1、雷达充满列宽）。

    行序（任务书批语 3）：行 1 = L 入射场图、行 2 = R 入射场图、行 3 = 二合一雷达；
    每列色标与行 1+2（两幅场图）等高、置列右缘。
    """
    panel_w = (_RIGHT - _LEFT - 3 * _G_PANEL_CBAR - 3 * _CB_W - 2 * _G_CBAR_COL) / 3.0
    panel_mm = panel_w * _FIG_W_MM
    radar_h_mm = panel_mm * _RADAR_ASPECT
    field_h_mm = panel_mm / 2.0
    fig_h_mm = (
        _MARGIN_TOP_MM
        + _TITLE_MM
        + field_h_mm
        + _G_ROW_MM
        + field_h_mm
        + _G_ROW_MM
        + radar_h_mm
        + _MARGIN_BOT_MM
    )
    top = 1.0 - _MARGIN_TOP_MM / fig_h_mm
    row0 = (field_h_mm + _TITLE_MM) / fig_h_mm
    gap = _G_ROW_MM / fig_h_mm
    field = field_h_mm / fig_h_mm
    return {
        "panel_w": panel_w,
        "fig_h_mm": fig_h_mm,
        "top": top,
        "row0": row0,
        "gap": gap,
        "field": field,
        "radar": radar_h_mm / fig_h_mm,
        "y_header": top + 0.30 * _TITLE_MM / fig_h_mm,
        "y_f1": top - row0,
        "y_f2": top - row0 - gap - field,
        "y_radar": top - row0 - 2 * (gap + field) - gap,
    }


def compose_combined(*, occlusion_dir: Path | None = None) -> Path:
    """组装整图。

    Args:
        occlusion_dir: 非空时，在同一渲染进程内追加导出**删文字版**（同版式、所有
            ``Text`` 图元整体移除）到该目录，供 ``pysci-figures overlap`` 的零遮挡
            像素判据做真值面（先写受审件、后写删文字版，满足其 mtime 新鲜度纪律）。
    """
    L = _layout()
    fig_h_in = mm_to_inch(L["fig_h_mm"])
    fig_w_in = mm_to_inch(_FIG_W_MM)
    panel_w, y_f1, y_f2 = L["panel_w"], L["y_f1"], L["y_f2"]
    fields = {(cfg, inc): load_field(cfg, inc) for cfg, _ in CASES for inc, _ in INCS}
    vmax = {
        cfg: _nice_ceil(
            max(
                float(np.abs(vals[_in_half_disc(xy)]).max())
                for (c, _), (xy, vals) in fields.items()
                if c == cfg
            )
        )
        for cfg, _ in CASES
    }

    with style_context("aps", width=str(int(_FIG_W_MM)), scienceplots=True):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in))
        letters = iter("abcdefghi")
        for j, (cfg, header) in enumerate(CASES):
            x = _LEFT + j * (panel_w + _G_PANEL_CBAR + _CB_W + _G_CBAR_COL)
            # 列头（放在该列最上方，行 1 之上）
            fig.text(
                x + panel_w / 2.0,
                L["y_header"],
                header,
                color=INK,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
            # 行 1 / 行 2：L、R 入射场图；行 3：二合一雷达
            for k, (inc, inc_label) in enumerate(INCS):
                ax_f = fig.add_axes([x, (y_f1, y_f2)[k], panel_w, L["field"]])
                draw_field_panel(
                    ax_f,
                    cfg,
                    inc,
                    f"{inc_label} $\\mathrm{{Re}}\\,p_t$",
                    vmax[cfg],
                    fields[(cfg, inc)],
                    labels=(j == 0),
                )
                ax_f.text(
                    -0.05,
                    1.02,
                    next(letters),
                    transform=ax_f.transAxes,
                    va="bottom",
                    ha="left",
                    color=INK,
                    fontweight="bold",
                )
            ax_r = fig.add_axes([x, L["y_radar"], panel_w, L["radar"]])
            draw_radar(ax_r, cfg, header)
            ax_r.text(
                -0.05,
                1.0,
                next(letters),
                transform=ax_r.transAxes,
                va="top",
                ha="left",
                color=INK,
                fontweight="bold",
            )
            cax = fig.add_axes(
                [x + panel_w + _G_PANEL_CBAR, y_f2, _CB_W, y_f1 + L["field"] - y_f2]
            )
            cbar = fig.colorbar(
                ScalarMappable(norm=Normalize(-vmax[cfg], vmax[cfg]), cmap=CMAP),
                cax=cax,
            )
            ticks = [-vmax[cfg], -vmax[cfg] / 2.0, 0.0, vmax[cfg] / 2.0, vmax[cfg]]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([_fmt(t) for t in ticks])
            cbar.set_label("$\\mathrm{Re}\\,p_t$ (Pa)", rotation=270, labelpad=7.0)
            cbar.outline.set_linewidth(0.5)
            # 刻度只留右侧主刻度（3~5 条裁决）；关掉 scienceplots 带入的次要刻度与双侧刻度
            cax.minorticks_off()
            cax.tick_params(
                which="both",
                left=False,
                right=True,
                top=False,
                bottom=False,
                length=2.0,
                width=0.5,
                pad=1.5,
            )

        fig.text(
            _LEFT,
            0.018,
            "$\\dagger$ Exp. insets show simulated data (experimental data unavailable).\n"
            "Diamond = the experiment's scan field region (side 31 cm; near edge 30 cm from the metasurface centre): its far side\n"
            "lies outside the model domain ($R$ = 40 cm), as measured. Square ``Sim.'' = the part of that region inside the domain\n"
            "(side 6 cm, corners within 37.5 cm $\\leq$ $R$, i.e. with a margin), shown magnified 2.2$\\times$ at each corner (Exp.$\\dagger$).",
            color=GRAY,
            fontsize=6,
            ha="left",
            va="bottom",
            linespacing=1.55,
        )
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(OUT_PNG, dpi=600, facecolor="white")
            if occlusion_dir is not None:
                from matplotlib.text import Text as _MplText  # noqa: PLC0415

                for t in list(fig.findobj(_MplText)):
                    try:
                        t.remove()
                    except (NotImplementedError, ValueError):
                        t.set_visible(
                            False
                        )  # legend/colorbar 的 Text 只能隐去（不参与渲染）
                occlusion_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    occlusion_dir / f"{OUT_PNG.stem}_notext.png",
                    dpi=600,
                    facecolor="white",
                )
        plt.close(fig)
    return OUT_PNG


# ---------------------------------------------------------------------------
# 可选：COMSOL 重导场 CSV（属 sim 环节，需活体 COMSOL）
# ---------------------------------------------------------------------------
def export_cases(cases: list[str]) -> None:
    import mph  # noqa: PLC0415

    from pysci.skills.comsol_simulation.tools import export as _export  # noqa: PLC0415

    mph_path = ROOT / "data/research/1_gain_ep/simulation/6 半圆波束散射对照.mph"
    client = mph.start(cores=4)
    model = client.load(str(mph_path))
    jm = model.java
    overrides = {
        "full": {"ci": "-0.0730324"},
        "nogain": {"ci": "0"},
        "nostruct": {"ci": "-0.0730324", "h1e": "lambda/200", "h3e": "lambda/200"},
    }
    for cfg in cases:
        for inc, pl, pr in (("L", "1", "0"), ("R", "0", "1")):
            jm.param().set("pampL", pl)
            jm.param().set("pampR", pr)
            for key, val in overrides[cfg].items():
                jm.param().set(key, val)
            jm.component("comp1").geom("geom1").run()
            jm.component("comp1").mesh("mesh1").run()
            jm.study("std1").run()
            dsets = [str(t) for t in (jm.result().dataset().tags() or [])]
            ds = dsets[0] if dsets else "dset1"
            print(
                _export.export_data(
                    model, ds, RAW_DIR / f"field_{cfg}_{inc}.csv", expr="real(acpr.p_t)"
                ).report()
            )
    client.disconnect()


def main() -> None:
    if "--export" in sys.argv:
        only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
        export_cases(only or [c for c, _ in CASES])
    occ = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--occlusion-dir=")]
    print("combined ->", compose_combined(occlusion_dir=Path(occ[0]) if occ else None))


if __name__ == "__main__":
    main()
