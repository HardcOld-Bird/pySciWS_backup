"""beam-scatter 定稿大图：3 组对照 × 左右入射 = 6 case × (半圆场图 + 极坐标散射图) = 12 子图。

产物组织（全部集中在 figures/beam_scatter/ 一个子文件夹）：
  raw/      COMSOL 原生渲染（极坐标 far_*.png，按行 rmax）+ 数值场 field_*.csv
  panels/   逐 case 半圆合成图（含实验占位框），由 compose_beam_scatter_proto.py 生成
  beam_scatter_combined.png   本脚本输出的 12 子图定稿大图

版式（对齐参考图 4ref3 右上，并按"分组标度"需求）：
  1. **分组标度（场）**：每行（对照组）独立色标，vmax = 该行 |Re p_t| 的 99 分位（稳健，
     避免管槽谐振尖峰抬高上界使条纹发白）；场图用 scientific-plotting 的 field 原语
     （tripcolor）以该行 cmap+norm 渲染 → 面板颜色与该行 colorbar 严格一致。
  2. **分组标度（极坐标）**：每行独立极径上界 POLAR_RMAX_ROW（由统一 rmax=60 的旧渲染
     估得各行远场最大值后取整）；COMSOL 以该行 rmax 重导 far_*.png → 弱组不被强组压没。
  3. **菱形通道版式**：每个半圆内两个**尖角向上的正方形（菱形）**通道框（= 仿真 Sim.），
     各以**两条虚线（自左右两顶点）**连到半圆外顶部角落的**菱形**实验插图（Exp.，
     实验位暂以仿真数据放大代填）；标注置于菱形**侧边**而非中心，避免遮挡内部细节；
     外菱形置于半圆竖直中心轴更外侧（INSET_ANG/INSET_R），不遮挡半圆本体。
  4. **超表面轮廓**：每幅场图叠加黑色管槽截面折线（直径边 y=0 下凹各管槽），使管槽
     形状在饱和色块下仍清晰可辨；对照③仅画管槽 2（管槽 1/3 已在几何中整根移除）。
  5. **colorbar 置最右**：每行一条 colorbar，放在整图最右列（与该行对齐）。

用法：
  python finalize_beam_scatter_figure.py                 # 仅离线重排
  python finalize_beam_scatter_figure.py --export        # 全 6 case COMSOL 导出
  python finalize_beam_scatter_figure.py --export --only=nostruct   # 仅重导某对照
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from pysci.skills.scientific_plotting.tools import field as _field  # noqa: E402
from pysci.skills.scientific_plotting.tools import raster  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_beam_scatter_mph import OUT_MPH  # noqa: E402

FIG_BASE = HERE.parents[3] / "data" / "research" / "1_gain_ep" / "article" / "figures" / "beam_scatter"
RAW_DIR = FIG_BASE / "raw"

# --- 几何常量（与 build 一致，单位 m）---
LAMBDA = 343.0 / 3430.0
D = LAMBDA / math.sqrt(2.0)
X0 = 8 * D / 2.0
R = 4 * LAMBDA
NPER = 8
# 单周期管槽（宽/深/周期内偏移），与 build PARAMS 同
W1, H1 = 0.227 * D, 0.569 * LAMBDA
W2, H2, O2 = 0.115 * D, 0.195 * LAMBDA, 0.503 * D
W3, H3, O3 = 0.153 * D, 0.232 * LAMBDA, 0.688 * D

POLAR_RMAX_ROW = {"full": 50.0, "nogain": 5.0, "nostruct": 10.0}  # 各行远场极径上界（Pa，实测取整）
CMAP = "bwr"            # 发散色标（signed Re p_t）

CI_EP = "-0.0730324"
SLIVER = "lambda/200"   # 管槽 1/3 薄片深度 = 等效整根移除（对照③）
# (对照名, 参数覆盖)；h1e/h3e 缺省 = h1/h3（结构存在）
CASES: list[tuple[str, dict[str, str]]] = [
    ("full", {"ci": CI_EP}),
    ("nogain", {"ci": "0"}),
    ("nostruct", {"ci": CI_EP, "h1e": SLIVER, "h3e": SLIVER}),
]
INCS = [("L", "1", "0"), ("R", "0", "1")]
GROUP_LABEL = {"full": "(i) groove + gain", "nogain": "(ii) groove, no gain", "nostruct": "(iii) gain, no groove"}

# --- 菱形通道版式（数据坐标，米）---
CH_ANG = (135.0, 45.0)     # 左/右输出通道方位角（内菱形中心）
CH_R = 0.26                # 内菱形中心半径
S_IN = 0.12                # 内菱形（尖角向上正方形）边长
INSET_ANG = (125.0, 55.0)  # 左/右实验插图方位角（外菱形中心，偏外侧以免遮挡半圆）
INSET_R = 0.55             # 外菱形中心半径（半圆外顶部角落）
S_OUT = 0.17               # 外菱形边长（> S_IN → 放大插图）
_LBL = dict(fontsize=7, va="center", bbox=dict(fc="w", alpha=0.6, ec="none", pad=0.5))


def polar(phi_deg: float, r: float) -> tuple[float, float]:
    a = math.radians(phi_deg)
    return (X0 + r * math.cos(a), r * math.sin(a))


def _rect_anchor(cx: float, cy: float, w: float, h: float, angle: float) -> tuple[float, float]:
    """Rectangle 绕未旋转左下角旋转；反解锚点使 (cx,cy) 为旋转后真中心（同 raster.apply_overlays）。"""
    a = math.radians(angle)
    return (cx - (w / 2 * math.cos(a) - h / 2 * math.sin(a)),
            cy - (w / 2 * math.sin(a) + h / 2 * math.cos(a)))


def _nice_ceil(x: float) -> float:
    """把上界规整到 1/2/5×10^n，色标刻度更整洁。"""
    if x <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


# ---------------------------------------------------------------------------
# COMSOL 导出：逐 case 数值场 CSV（signed real p_t）+ 按行 rmax 的远场 PNG
# （COMSOL Data 导出的 VTK exporttype 不生效，落盘为文本；文本含 x,y,real(p_t) 三列，
#   field 原语的 CSV 读取器可直接解析，故统一用 .csv）
# ---------------------------------------------------------------------------
def export_cases(cases: list[tuple[str, dict[str, str]]]) -> None:
    import mph  # noqa: PLC0415 - 仅导出需要活体 COMSOL

    from pysci.skills.comsol_simulation.tools import export as _export  # noqa: PLC0415

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = mph.start(cores=4)
    model = client.load(str(OUT_MPH))
    jm = model.java
    for cfg, ov in cases:
        for inc, pl, pr in INCS:
            tag = f"{cfg}_{inc}"
            jm.param().set("pampL", pl)
            jm.param().set("pampR", pr)
            for key, val in ov.items():
                jm.param().set(key, val)
            jm.component("comp1").geom("geom1").run()
            jm.component("comp1").mesh("mesh1").run()
            jm.study("std1").run()
            dsets = [str(t) for t in (jm.result().dataset().tags() or [])]
            ds = dsets[0] if dsets else None
            if ds:
                for pg in ("pg_field", "pg_far"):
                    jm.result(pg).set("data", ds)
            print(f"== export field csv {tag} ==")
            print(_export.export_data(model, ds or "dset1", RAW_DIR / f"field_{tag}.csv",
                                      expr="real(acpr.p_t)").report())
            # 远场以该行独立极径重导（弱组不被强组压没）
            print(f"== export far png {tag} (rmax={POLAR_RMAX_ROW[cfg]}) ==")
            print(_export.export_image(model, "pg_far", RAW_DIR / f"far_{tag}.png",
                                       polar_rmax=(0.0, POLAR_RMAX_ROW[cfg]), clean=True,
                                       sidecar=False).report())
    client.disconnect()


# ---------------------------------------------------------------------------
# 超表面截面轮廓：直径边 y=0 下凹各管槽的黑色折线（对照③仅管槽 2）
# ---------------------------------------------------------------------------
def _grooves(cfg: str) -> list[tuple[float, float, float]]:
    """该对照下实际存在的管槽 (x, 宽, 深) 列表（对照③仅管槽 2）。"""
    out: list[tuple[float, float, float]] = []
    for p in range(NPER):
        b = p * D
        if cfg != "nostruct":
            out.append((b, W1, H1))
            out.append((b + O3, W3, H3))
        out.append((b + O2, W2, H2))
    out.sort()
    return out


def _in_fluid(xy2d: np.ndarray, cfg: str) -> np.ndarray:
    """物理流体域判据（半圆 y>=0 ∪ 各管槽矩形）；用于遮罩 Delaunay 在固体壁内的架桥伪影。"""
    x, y = xy2d[:, 0], xy2d[:, 1]
    inside = (y >= 0) & ((x - X0) ** 2 + y ** 2 <= R ** 2)
    for gx, gw, gd in _grooves(cfg):
        inside |= (x >= gx) & (x <= gx + gw) & (y >= -gd) & (y < 0)
    return inside


def draw_metasurface(ax, cfg: str) -> None:
    xs: list[float] = [X0 - R]
    ys: list[float] = [0.0]
    for gx, gw, gd in _grooves(cfg):
        xs += [gx, gx, gx + gw, gx + gw]
        ys += [0.0, -gd, -gd, 0.0]
    xs.append(X0 + R)
    ys.append(0.0)
    ax.plot(xs, ys, color="k", lw=0.9, zorder=6, solid_joinstyle="miter")


# ---------------------------------------------------------------------------
# 菱形通道版式：内菱形(Sim.) + 两顶点虚线 + 外菱形实验插图(Exp., 仿真代填)
# ---------------------------------------------------------------------------
def add_channel_insets(ax, xy: np.ndarray, vals: np.ndarray, vmax: float) -> None:
    norm = Normalize(-vmax, vmax)
    a = math.radians(45.0)
    rot45 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rotm45 = rot45.T
    h_in = S_IN * math.sqrt(2) / 2
    h_out = S_OUT * math.sqrt(2) / 2
    for k in range(2):
        icx, icy = polar(CH_ANG[k], CH_R)
        ox, oy = polar(INSET_ANG[k], INSET_R)
        side = -1 if k == 0 else 1          # 左通道标注放左侧，右通道放右侧
        ha = "right" if side < 0 else "left"
        # 内菱形（尖角向上正方形）= 仿真通道框
        ax.add_patch(Rectangle(_rect_anchor(icx, icy, S_IN, S_IN, 45.0), S_IN, S_IN,
                               angle=45.0, fill=False, ec="k", lw=1.2))
        # 裁剪内菱形自身方域（45° 坐标系下的正方形）→ 放大 → 平移到外菱形
        local = (xy - np.array([icx, icy])) @ rotm45.T
        m = (np.abs(local[:, 0]) <= S_IN / 2) & (np.abs(local[:, 1]) <= S_IN / 2)
        if m.sum() >= 3:
            zoom = S_OUT / S_IN
            w = (local[m] * zoom) @ rot45.T + np.array([ox, oy])
            diamond = Rectangle(_rect_anchor(ox, oy, S_OUT, S_OUT, 45.0), S_OUT, S_OUT,
                                angle=45.0, fill=False, ec="k", lw=1.2)
            ax.add_patch(diamond)
            coll = ax.tripcolor(Triangulation(w[:, 0], w[:, 1]), vals[m],
                                cmap=CMAP, norm=norm, shading="gouraud")
            coll.set_clip_path(diamond)
        # 两条虚线：自内菱形左/右两顶点 → 外菱形左/右两顶点
        ax.plot([icx - h_in, ox - h_out], [icy, oy], ls="--", lw=0.8, color="k")
        ax.plot([icx + h_in, ox + h_out], [icy, oy], ls="--", lw=0.8, color="k")
        # 标注置于侧边：内=Sim.，外=Exp.
        ax.text(icx + side * (h_in + 0.012), icy, "Sim.", ha=ha, color="k", **_LBL)
        ax.text(ox + side * (h_out + 0.012), oy, "Exp.", ha=ha, color="k", **_LBL)


# ---------------------------------------------------------------------------
# 定稿大图组装
# ---------------------------------------------------------------------------
def compose_combined() -> Path:
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cfg, _ov in CASES:
        for inc, _, _ in INCS:
            tag = f"{cfg}_{inc}"
            data[tag] = _field.load_field_points(RAW_DIR / f"field_{tag}.csv")

    # 每行独立场色标上界（99 分位，稳健）
    vmax_row = {
        cfg: _nice_ceil(np.percentile(
            np.concatenate([np.abs(data[f"{cfg}_{i}"][1]) for i in ("L", "R")]), 99))
        for cfg, _ov in CASES
    }

    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.10, hspace=0.22)

    letter = ord("a")
    for ri, (cfg, _ov) in enumerate(CASES):
        vmax = vmax_row[cfg]
        row_first_ax = None
        for ci, inc in enumerate(("L", "R")):
            tag = f"{cfg}_{inc}"
            xy, vals = data[tag]
            # 场图子图（数据驱动，行内统一 norm → 与该行 colorbar 严格一致）
            ax_f = fig.add_subplot(gs[ri, 2 * ci])
            if row_first_ax is None:
                row_first_ax = ax_f
            _field.add_field_panel(ax_f, xy, vals, cmap=CMAP, vmin=-vmax, vmax=vmax,
                                   keep_triangle=lambda c: _in_fluid(c, cfg))
            ax_f.set_aspect("equal")
            ax_f.set_axis_off()
            draw_metasurface(ax_f, cfg)
            add_channel_insets(ax_f, xy, vals, vmax)
            # 极坐标子图（COMSOL 以该行 rmax 重导的远场 PNG）
            ax_p = fig.add_subplot(gs[ri, 2 * ci + 1])
            ax_p.imshow(raster.read_raster(RAW_DIR / f"far_{tag}.png"), aspect="equal")
            ax_p.set_axis_off()
            for ax in (ax_f, ax_p):
                ax.text(0.02, 0.98, f"({chr(letter)})", transform=ax.transAxes,
                        va="top", ha="left", fontsize=11, fontweight="bold")
                letter += 1
            if ri == 0:
                ax_f.set_title(f"{inc}-incidence  field  $|p_t|$", fontsize=11)
                ax_p.set_title(f"{inc}-incidence  far-field  $|p_{{ext}}|$", fontsize=11)
        # 该行 colorbar（最右列，与行对齐）
        cax = fig.add_subplot(gs[ri, 4])
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap=CMAP), cax=cax)
        cbar.set_label(r"Re $p_t$ (Pa)", fontsize=9)
        cax.set_title(f"±{vmax:.0f}", fontsize=8)
        # 行标签置于该行第一个场图左侧
        row_first_ax.text(-0.06, 0.5, GROUP_LABEL[cfg], transform=row_first_ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=11)

    fig.suptitle(
        "Beam scattering at the gain EP: 3 controls × L/R incidence\n"
        "(per-control field scale & polar radius)",
        fontsize=13,
    )
    out = FIG_BASE / "beam_scatter_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def main() -> None:
    if "--export" in sys.argv:
        cases = CASES
        for arg in sys.argv:
            if arg.startswith("--only="):
                want = arg.split("=", 1)[1]
                cases = [c for c in CASES if c[0] == want]
        export_cases(cases)
    print("combined ->", compose_combined())


if __name__ == "__main__":
    main()
