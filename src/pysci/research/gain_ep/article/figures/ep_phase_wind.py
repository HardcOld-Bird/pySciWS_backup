"""ep_phase_wind — S 矩阵元相角色图 + 绕 EP 环路相位累积（3×4=12 子图）。

参考 `data/research/1_gain_ep/article/figures/绘图参考/2ref1d.png` 右块（arg(S) 四矩阵元
色图）并升级：三行（而非两行）× 四列 = 12 幅相角色图：

  (a) 损耗 EP 区域   —— 原始 S 四矩阵元相角，中心 = 损耗 EP。
  (b) gain EP 区域   —— 原始 S 四矩阵元相角，中心 = 增益极点。
  (c) gain EP 区域   —— 归一化 S' 四矩阵元相角（对角元恒 0，相角无定义，留空标注）。

底面两轴 = 第二管槽归一化复声速 (cr, ci)；颜色 = 散射系数相角 arg（循环 colormap）。
每幅色图叠加绕中心一周的白色虚线环路，并以环路积分（unwrap 相角总增量）标注相位
累积 ∮dφ = n·2π（n 为绕数：0 / ±1 / ±2 …，即用户预期的 0 / 2π / 4π 情形）。

数据由 CMT（cmt_reflection_s_matrix）在 (cr, ci) 网格与环路上计算，缓存 npz。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import root

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt

# --- 物理/数值常量 -------------------------------------------------------
CR = 1.0081  # 第二管槽归一化复声速实部（参考）
CI_LOSS_EP = 0.0745  # 损耗 EP 位置（参考）
CI_GAIN_SS = -0.0740  # 增益谱奇点（极点）位置（参考）
W_CR, W_CI = 0.02, 0.025  # (cr, ci) 窗口半宽
N_UNIFORM = 60  # 损耗区均匀网格边长
N_FAR, N_NEAR = 30, 20  # 极点区：远场/近心加密每侧点数
D_MIN = 1e-4  # 距极点最近采样距离
N_LOOP = 720  # 环路采样点数（环路积分用）
LOOP_FRAC = 0.6  # 环路半径 = 窗口半宽 × LOOP_FRAC
N_ORDERS, K_MODES = 8, 25  # CMT 截断（形状图够用；精算见理论脚本）
CMAP = "hsv"  # 循环 colormap（相角专用）

SLUG = "ep_phase_wind"
ELEMS = ("11", "12", "21", "22")


def _s_matrix(cr: float, ci: float):
    return cmt.compute_s_matrix(
        cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    )


def _refine_loss_ep(cr0, ci0):
    """2D 精定位损耗 EP：S21(cr,ci)=0。"""

    def F(x):
        S = _s_matrix(x[0], x[1])
        return [S[1, 0].real, S[1, 0].imag]

    r = root(F, [cr0, ci0], method="hybr")
    return float(r.x[0]), float(r.x[1])


def _refine_gain_pole(cr0, ci0):
    """2D 精定位增益极点：1/S21(cr,ci)=0。"""

    def G(x):
        inv = 1.0 / _s_matrix(x[0], x[1])[1, 0]
        return [inv.real, inv.imag]

    r = root(G, [cr0, ci0], method="hybr")
    return float(r.x[0]), float(r.x[1])


def _axis(center, half, refined):
    """单参数轴：光滑区均匀；极点区中心加密。"""
    if not refined:
        return np.linspace(center - half, center + half, N_UNIFORM)
    far = np.linspace(center - half, center + half, N_FAR)
    d = np.geomspace(D_MIN, half, N_NEAR)
    return np.unique(np.concatenate([far, center - d, center + d]))


def _grid_elements(cr0, ci0, refined):
    """网格上四矩阵元（2D 复数组）。"""
    crs = _axis(cr0, W_CR, refined)
    cis = _axis(ci0, W_CI, refined)
    CR2, CI2 = np.meshgrid(crs, cis)
    out = {f"s{e}": np.empty(CR2.shape, dtype=complex) for e in ELEMS}
    for i in range(CR2.shape[0]):
        for j in range(CR2.shape[1]):
            S = _s_matrix(CR2[i, j], CI2[i, j])
            for e, (a, b) in zip(ELEMS, ((0, 0), (0, 1), (1, 0), (1, 1))):
                out[f"s{e}"][i, j] = S[a, b]
    out["cr"], out["ci"] = CR2, CI2
    return out


def _loop_elements(cr0, ci0):
    """绕中心一周的环路上四矩阵元（1D 复数组，首尾闭合）。"""
    th = np.linspace(0.0, 2 * np.pi, N_LOOP + 1)
    crs = cr0 + LOOP_FRAC * W_CR * np.cos(th)
    cis = ci0 + LOOP_FRAC * W_CI * np.sin(th)
    out = {f"s{e}": np.empty(th.size, dtype=complex) for e in ELEMS}
    for k in range(th.size):
        S = _s_matrix(crs[k], cis[k])
        for e, (a, b) in zip(ELEMS, ((0, 0), (0, 1), (1, 0), (1, 1))):
            out[f"s{e}"][k] = S[a, b]
    out["cr"], out["ci"] = crs, cis
    return out


def _winding(v):
    """环路积分：unwrap 相角总增量 / 2π → 整数绕数。"""
    phi = np.unwrap(np.angle(v))
    return int(np.round((phi[-1] - phi[0]) / (2 * np.pi)))


def _norm_elems(g):
    """由原始矩阵元得归一化 N = (S − tI)/dom 的非零元 N12, N21（对角恒 0）。"""
    dom = np.where(np.abs(g["s12"]) >= np.abs(g["s21"]), g["s12"], g["s21"])
    return g["s12"] / dom, g["s21"] / dom


def _compute_all():
    cr_l, ci_l = _refine_loss_ep(CR, CI_LOSS_EP)
    cr_g, ci_g = _refine_gain_pole(CR, CI_GAIN_SS)
    d = {}
    for tag, (cr0, ci0, ref) in (("l", (cr_l, ci_l, False)), ("g", (cr_g, ci_g, True))):
        g = _grid_elements(cr0, ci0, ref)
        lp = _loop_elements(cr0, ci0)
        for k, v in g.items():
            d[f"{k}_{tag}"] = v
        for k, v in lp.items():
            d[f"loop{k}_{tag}"] = v
        # 绕数：原始四元 + 归一化非零元
        for e in ELEMS:
            d[f"wind_s{e}_{tag}"] = np.array([_winding(lp[f"s{e}"])])
        n12, n21 = _norm_elems(lp)
        d[f"wind_n12_{tag}"] = np.array([_winding(n12)])
        d[f"wind_n21_{tag}"] = np.array([_winding(n21)])
    d["ep_l"] = np.array([cr_l, ci_l])
    d["ep_g"] = np.array([cr_g, ci_g])
    return d


def _cache_path(research_dir):
    """缓存 npz 路径。runner 传入的 research_dir 实为 figures 根目录，兼容两种语义。"""
    if research_dir is None:
        research_dir = (
            Path(__file__).resolve().parents[6] / "data" / "research" / "1_gain_ep"
        )
    rd = Path(research_dir)
    figures_root = rd if rd.name == "figures" else rd / "article" / "figures"
    return figures_root / SLUG / "cache_wind.npz"


def _load(research_dir):
    """优先读 npz 缓存；无缓存则计算并落盘。"""
    cache = _cache_path(research_dir)
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    data = _compute_all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def _wind_label(n):
    """环路相位累积的 mathtext 标注（避免 ∮ 字符 tofu）。"""
    if n == 0:
        return r"$\oint d\varphi=0$"
    return rf"$\oint d\varphi={n:+d}\cdot2\pi$"


def build_figure(style=None, research_dir=None, **kwargs):
    """构建 3×4 相角色图 + 环路绕数标注并返回 Figure。"""
    d = _load(research_dir)
    dom_g = np.where(np.abs(d["s12_g"]) >= np.abs(d["s21_g"]), d["s12_g"], d["s21_g"])
    n12_g, n21_g = d["s12_g"] / dom_g, d["s21_g"] / dom_g

    fig = plt.figure()
    fig.set_size_inches(6.693, 5.4)
    cols_x = [0.055 + i * 0.215 for i in range(4)]
    rows_y = [0.72, 0.42, 0.12]
    ax_w, ax_h = 0.185, 0.24

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)

    row_specs = [
        (
            "l",
            "(a) Loss EP: arg($S$)",
            [d[f"s{e}_l"] for e in ELEMS],
            [int(d[f"wind_s{e}_l"][0]) for e in ELEMS],
        ),
        (
            "g",
            "(b) Gain EP: arg($S$)",
            [d[f"s{e}_g"] for e in ELEMS],
            [int(d[f"wind_s{e}_g"][0]) for e in ELEMS],
        ),
        (
            "c",
            "(c) Gain EP: arg($S'$)",
            [None, n12_g, n21_g, None],
            [None, int(d["wind_n12_g"][0]), int(d["wind_n21_g"][0]), None],
        ),
    ]
    for r, (tag, rowlabel, fields, winds) in enumerate(row_specs):
        fig.text(
            0.012,
            rows_y[r] + ax_h / 2,
            rowlabel,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7,
        )
        for c in range(4):
            ax = fig.add_axes([cols_x[c], rows_y[r], ax_w, ax_h])
            if fields[c] is None:
                # 归一化对角元恒 0：相角无定义
                ax.set_facecolor("0.92")
                ax.text(
                    0.5,
                    0.5,
                    r"$\equiv 0$",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.4",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                Z = np.angle(fields[c])
                ax.pcolormesh(
                    d[f"cr_{tag if tag != 'c' else 'g'}"],
                    d[f"ci_{tag if tag != 'c' else 'g'}"],
                    Z,
                    cmap=CMAP,
                    norm=norm,
                    shading="auto",
                )
                lc, ls = (
                    d[f"loopcr_{tag if tag != 'c' else 'g'}"],
                    d[f"loopci_{tag if tag != 'c' else 'g'}"],
                )
                ax.plot(lc, ls, "w--", lw=0.8)
                ep = d["ep_l"] if tag == "l" else d["ep_g"]
                ax.plot(ep[0], ep[1], "w.", ms=2.5)
                ax.text(
                    0.97,
                    0.96,
                    _wind_label(winds[c]),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                    bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.0),
                )
                ax.tick_params(labelsize=6, pad=0)
                if c > 0:
                    ax.tick_params(labelleft=False)
                if r < 2:
                    ax.tick_params(labelbottom=False)
            if r == 0:
                ax.set_title(
                    f"$S_{{{ELEMS[c]}}}$" if tag != "c" else f"$N_{{{ELEMS[c]}}}$",
                    fontsize=8,
                )
            if r == 2:
                ax.set_xlabel(r"$c_r$", fontsize=7)
            if c == 0:
                ax.set_ylabel(r"$c_i$", fontsize=7)

    cbar = fig.colorbar(mappable, cax=fig.add_axes([0.925, 0.12, 0.012, 0.84]))
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    cbar.set_label("arg", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig
