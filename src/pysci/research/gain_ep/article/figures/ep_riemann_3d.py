"""ep_riemann_3d — 本征值双叶黎曼面 over (cr, ci) 三面板对照。

参考 `data/research/1_gain_ep/article/figures/绘图参考/2ref1b.png` 的形式并升级：
原参考图为两面板（Original S 平坦面+极点尖峰 / Normalized S' 双叶鞍形交叉），
本图扩为三面板以对照三种谱结构：

  (a) 镜像损耗 EP 区域 —— 原始 S 的双叶黎曼面，分支点 = 损耗 EP（平方根枝点鞍形）。
  (b) gain EP 区域 —— 原始 S 的双叶面，极点处 ± 叶各自发散成上/下尖峰。
  (c) gain EP 区域 —— 归一化 S' 的双叶黎曼面，分支点 = 伪损耗 EP（合并于 0）。

三维空间与参考图一致：竖轴 = 本征值实部 Re(λ)，曲面颜色 = 本征值虚部 Im(λ)，
底面两轴 = 第二管槽归一化复声速的实部 cr 与虚部 ci（用户假设：与参考图的几何
参数 (d1, w1) 同构——分支点/极点均为二维参数空间中的孤立点）。

数据由 CMT（cmt_reflection_s_matrix）在 (cr, ci) 网格上计算，缓存 npz 加速迭代。
叶标签取主支 sqrt（λ± = t ± √(S12·S21)）：枝割缝在面上表现为参考图同款的接缝，
属黎曼面可视化的固有特征而非伪影。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 '3d' 投影)
from pathlib import Path
from scipy.optimize import root

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt

# --- 物理/数值常量 -------------------------------------------------------
CR = 1.0081            # 第二管槽归一化复声速实部（参考）
CI_LOSS_EP = 0.0745    # 损耗 EP 位置（参考）
CI_GAIN_SS = -0.0740   # 增益谱奇点（极点）位置（参考）
W_CR, W_CI = 0.02, 0.025          # (cr, ci) 窗口半宽
N_UNIFORM = 60                    # 光滑区域（损耗）均匀网格边长
N_FAR, N_NEAR = 30, 20            # 极点区域：远场/近心加密每侧点数
D_MIN = 1e-4           # 距极点最近采样距离（尖峰 |λ|~几十，落在框内）
N_ORDERS, K_MODES = 8, 25         # CMT 截断（形状图够用；精算见理论脚本）
ZLIM_B = 60.0          # (b) 竖轴限：越界置 nan 隐藏（不饱和裁剪）
CMAP = "viridis"       # 色盲安全 colormap（曲面颜色 = Im λ）

SLUG = "ep_riemann_3d"


def _s_matrix(cr: float, ci: float):
    return cmt.compute_s_matrix(
        cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    )


def _t_s_mu(S):
    """由 S 得 t、s=√(S12·S21)（主支）与归一化本征值 μ = s/dom。"""
    t = 0.5 * (S[0, 0] + S[1, 1])
    s = np.sqrt(S[0, 1] * S[1, 0])
    dom = S[0, 1] if abs(S[0, 1]) >= abs(S[1, 0]) else S[1, 0]
    return t, s, s / dom


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
    """单参数轴：光滑区均匀；极点区中心加密（linspace 远场 + geomspace 近心）。"""
    if not refined:
        return np.linspace(center - half, center + half, N_UNIFORM)
    far = np.linspace(center - half, center + half, N_FAR)
    d = np.geomspace(D_MIN, half, N_NEAR)
    return np.unique(np.concatenate([far, center - d, center + d]))


def _region(cr0, ci0, refined):
    """在 (cr, ci) 网格上算双叶本征值与归一化本征值（2D 复数组）。"""
    crs = _axis(cr0, W_CR, refined)
    cis = _axis(ci0, W_CI, refined)
    CR2, CI2 = np.meshgrid(crs, cis)          # (ni, nj)
    LP = np.empty(CR2.shape, dtype=complex)
    LM = np.empty_like(LP)
    MP = np.empty_like(LP)
    for i in range(CR2.shape[0]):
        for j in range(CR2.shape[1]):
            t, s, mu = _t_s_mu(_s_matrix(CR2[i, j], CI2[i, j]))
            LP[i, j], LM[i, j] = t + s, t - s
            MP[i, j] = mu
    return {"cr": CR2, "ci": CI2, "lp": LP, "lm": LM, "mp": MP}


def _compute_surfaces():
    """精定位 EP/极点后分别建损耗区（均匀）与增益区（加密）网格。"""
    cr_l, ci_l = _refine_loss_ep(CR, CI_LOSS_EP)
    cr_g, ci_g = _refine_gain_pole(CR, CI_GAIN_SS)
    loss = _region(cr_l, ci_l, refined=False)
    gain = _region(cr_g, ci_g, refined=True)
    out = {}
    for tag, rg in (("l", loss), ("g", gain)):
        for k, v in rg.items():
            out[f"{k}_{tag}"] = v
    out["ep_l"] = np.array([cr_l, ci_l])
    out["ep_g"] = np.array([cr_g, ci_g])
    return out


def _cache_path(research_dir):
    """缓存 npz 路径。runner 传入的 research_dir 实为 figures 根目录，兼容两种语义。"""
    if research_dir is None:
        research_dir = (
            Path(__file__).resolve().parents[6] / "data" / "research" / "1_gain_ep"
        )
    rd = Path(research_dir)
    figures_root = rd if rd.name == "figures" else rd / "article" / "figures"
    return figures_root / SLUG / "cache_surfaces.npz"


def _load_surfaces(research_dir):
    """优先读 npz 缓存；无缓存则计算并落盘。"""
    cache = _cache_path(research_dir)
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    data = _compute_surfaces()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def _cut_mask(s2):
    """枝割线遮罩：主支 sqrt 跨枝割时 s 变号（s_new ≈ −s_old），plot_surface 会把
    跳变连成竖直鳍片墙。检测相邻网格点变号处置 nan 切开曲面（黎曼面固有接缝，
    参考图 b2 中心亦可见），仅在 |s| 足够大处生效以免抹掉 EP 合并点。"""
    flip = np.zeros(s2.shape, dtype=bool)
    thr = 0.02 * np.nanmax(np.abs(s2))
    big = np.abs(s2) > thr
    for di, dj in ((1, 0), (0, 1)):
        prev = np.roll(s2, shift=(di, dj), axis=(0, 1))
        f = (np.real(np.conj(prev) * s2) < 0) & big
        if di:
            f[0, :] = False      # 排除 roll 环绕边界
        if dj:
            f[:, 0] = False
        flip |= f
    return flip


def _panel(ax, d, tag, use_mu, zlim=None):
    """画一个双叶黎曼面面板；返回该面板的 ScalarMappable（供 colorbar）。"""
    X, Y = d[f"cr_{tag}"], d[f"ci_{tag}"]
    if use_mu:
        Zp, Zm = d[f"mp_{tag}"].real, -d[f"mp_{tag}"].real
        Cp, Cm = d[f"mp_{tag}"].imag, -d[f"mp_{tag}"].imag
    else:
        Zp, Zm = d[f"lp_{tag}"].real, d[f"lm_{tag}"].real
        Cp, Cm = d[f"lp_{tag}"].imag, d[f"lm_{tag}"].imag
    Zp, Zm = Zp.copy(), Zm.copy()
    Cp, Cm = Cp.copy(), Cm.copy()
    if zlim is not None:
        # 越界（极点发散冲出画幅）置 nan 隐藏，不做饱和裁剪
        Zp[np.abs(Zp) > zlim] = np.nan
        Zm[np.abs(Zm) > zlim] = np.nan
    # 枝割处切开（对 μ 叶用同一 S 的 s2 = λ+ − λ− 检测）
    cut = _cut_mask(d[f"lp_{tag}"] - d[f"lm_{tag}"])
    for A in (Zp, Zm, Cp, Cm):
        A[cut] = np.nan
    vmax = np.nanmax(np.abs(np.concatenate([
        Cp[np.isfinite(Zp)].ravel(), Cm[np.isfinite(Zm)].ravel()])))
    mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=-vmax, vmax=vmax), cmap=CMAP)
    for Z, C in ((Zp, Cp), (Zm, Cm)):
        ax.plot_surface(X, Y, Z, facecolors=mappable.to_rgba(C),
                        shade=False, rstride=1, cstride=1)
    if zlim is not None:
        ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(labelsize=6, pad=0)
    return mappable


def build_figure(style=None, research_dir=None, **kwargs):
    """构建三面板黎曼面图并返回 Figure。"""
    d = _load_surfaces(research_dir)

    fig = plt.figure()
    fig.set_size_inches(6.693, 6.0)
    rects = [(0.00, 0.08, 0.27, 0.84), (0.335, 0.08, 0.27, 0.84),
             (0.67, 0.08, 0.27, 0.84)]
    cbrects = [(0.275, 0.15, 0.012, 0.60), (0.610, 0.15, 0.012, 0.60),
               (0.945, 0.15, 0.012, 0.60)]

    specs = [
        ("l", False, None, "(a) Loss EP: original $S$", r"Re($\lambda$)"),
        ("g", False, ZLIM_B, "(b) Gain EP: original $S$", r"Re($\lambda$)"),
        ("g", True, None, r"(c) Gain EP: normalized $S'$", r"Re($\lambda'$)"),
    ]
    for (rect, cbrect, (tag, use_mu, zlim, title, zlab)) in zip(rects, cbrects, specs):
        ax = fig.add_axes(rect, projection="3d")
        mappable = _panel(ax, d, tag, use_mu, zlim=zlim)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$c_r$", fontsize=7)
        ax.set_ylabel(r"$c_i$", fontsize=7)
        ax.set_zlabel(zlab, fontsize=7)
        cbar = fig.colorbar(mappable, cax=fig.add_axes(cbrect))
        cbar.set_label(r"Im($\lambda'$)" if use_mu else r"Im($\lambda$)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    return fig
