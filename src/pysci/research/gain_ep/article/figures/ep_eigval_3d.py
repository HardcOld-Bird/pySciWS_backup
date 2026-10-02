"""ep_eigval_3d — 增益/损耗 EP 本征值三维空间曲线（三面板对照）。

参考 `data/research/1_gain_ep/article/figures/绘图参考/2ref1a.png` 的形式并升级：
原参考图为两面板（gain EP 的 Original S / Normalized S'），本图扩为三面板以验证
"真实损耗 EP" 与 "gain EP 经 blow-up 归一化得到的伪损耗 EP" 是否等同：

  (a) 镜像位置的损耗 EP —— 损耗侧 (ci>0) 原始 S 的本征值 λ± = t ± √(S12·S21)，
      在 EP 处合并（有限值），竖轴 ci 反向绘制以便与 (c) 精确比较。
  (b) 未经变换的 gain EP —— 增益侧 (ci<0) 原始 S 的本征值，在极点处发散劈裂。
  (c) gain EP 的归一化 S' —— N=(S−tI)/dom 的本征值 μ± = ±√(S12·S21)/dom，
      在极点处合并于 0（伪损耗 EP）。

每面板为三维空间曲线：竖轴 = 第二管槽等效复声速虚部 ci，底面 = 本征值实部/虚部。
数据由 CMT（cmt_reflection_s_matrix）沿 ci 扫描计算，缓存为 npz 以加速迭代。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 '3d' 投影)
from scipy.optimize import root

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt
from pysci.skills.scientific_plotting.tools import palette

# --- 物理/数值常量 -------------------------------------------------------
CR = 1.0081            # 第二管槽归一化复声速实部
CI_LOSS_EP = 0.0745    # 损耗 EP 位置（参考）
CI_GAIN_SS = -0.0740   # 增益谱奇点（极点）位置（参考）
HALF = 0.025           # EP/极点两侧的 |ci| 半扫描宽度
N_PTS = 200            # 远场 linspace 采样点数（验证正确性优先，取轻量）
N_NEAR = 100           # 中心附近 geomspace 加密点数（每侧）
D_MIN = 1e-4           # 距中心最近采样距离：使发散回转点 |λ|~几十落在框内，
                       # 复现参考图闭合双曲钩形；过密会冲穿轴框留下断臂
N_ORDERS, K_MODES = 8, 25         # CMT 截断（形状图够用；精算见理论脚本）
Z_STRETCH = 1.8        # 3D box 竖向拉伸（竖轴显示 = 水平 × Z_STRETCH）

SLUG = "ep_eigval_3d"


def _s_matrix(cr: float, ci: float):
    return cmt.compute_s_matrix(
        cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    )


def _s_eigs_from_S(S):
    """由 S 得原始本征值 (λ+, λ-) 与归一化 N 本征值 (μ+, μ-)。"""
    t = 0.5 * (S[0, 0] + S[1, 1])
    s = np.sqrt(S[0, 1] * S[1, 0])          # 主支 sqrt，保证分支连续
    dom = S[0, 1] if abs(S[0, 1]) >= abs(S[1, 0]) else S[1, 0]
    mu = s / dom
    return t + s, t - s, mu, -mu


def _refine_loss_ep(cr0, ci0):
    """2D 精定位损耗 EP：S21(cr,ci)=0（复方程 → 2 实方程）。"""
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


def _sweep(center, include_center):
    """EP/极点中心加密的 ci 扫描（geomspace 近中心 + linspace 远场）。"""
    far = np.linspace(center - HALF, center + HALF, N_PTS)
    d = np.geomspace(D_MIN, HALF, N_NEAR)
    parts = [far, center - d, center + d]
    if include_center:
        parts.append(np.array([center]))
    return np.unique(np.concatenate(parts))


def _compute_curves():
    """精定位 EP/极点后沿 ci 扫描（扫描线穿过真 EP ⇒ 严格简并/发散）。"""
    cr_l, ci_l_ep = _refine_loss_ep(CR, CI_LOSS_EP)
    cr_g, ci_g_ep = _refine_gain_pole(CR, CI_GAIN_SS)

    ci_loss = _sweep(ci_l_ep, include_center=True)   # 精确命中 EP
    ci_gain = _sweep(ci_g_ep, include_center=False)  # 极点奇异，不取精确点

    def _track(prev, p, m):
        """连续分支追踪：选择与上一步最接近的配对，消除 sqrt 主支割线跳变。"""
        if prev is not None and (
            abs(m - prev[0]) + abs(p - prev[1]) < abs(p - prev[0]) + abs(m - prev[1])
        ):
            return m, p
        return p, m

    lp_l = np.empty_like(ci_loss, dtype=complex)
    lm_l = np.empty_like(ci_loss, dtype=complex)
    prev = None
    for i, ci in enumerate(ci_loss):
        p, m, _, _ = _s_eigs_from_S(_s_matrix(cr_l, ci))
        p, m = _track(prev, p, m)
        prev = (p, m)
        lp_l[i], lm_l[i] = p, m

    n_g = ci_gain.size
    lp_g = np.empty(n_g, dtype=complex)
    lm_g = np.empty(n_g, dtype=complex)
    mp_g = np.empty(n_g, dtype=complex)
    mm_g = np.empty(n_g, dtype=complex)
    prev_l = prev_m = None
    for i, ci in enumerate(ci_gain):
        p, m, mp, mm = _s_eigs_from_S(_s_matrix(cr_g, ci))
        p, m = _track(prev_l, p, m)
        mp, mm = _track(prev_m, mp, mm)
        prev_l, prev_m = (p, m), (mp, mm)
        lp_g[i], lm_g[i], mp_g[i], mm_g[i] = p, m, mp, mm

    return {
        "ci_loss": ci_loss, "ci_gain": ci_gain,
        "lam_p_loss": lp_l, "lam_m_loss": lm_l,
        "lam_p_gain": lp_g, "lam_m_gain": lm_g,
        "mu_p_gain": mp_g, "mu_m_gain": mm_g,
        "cr_l": np.array([cr_l]), "ci_l_ep": np.array([ci_l_ep]),
        "cr_g": np.array([cr_g]), "ci_g_ep": np.array([ci_g_ep]),
    }


def _cache_path(research_dir):
    """缓存 npz 路径。runner 传入的 research_dir 实为 figures 根目录，兼容两种语义。"""
    if research_dir is None:
        research_dir = (
            Path(__file__).resolve().parents[6] / "data" / "research" / "1_gain_ep"
        )
    rd = Path(research_dir)
    figures_root = rd if rd.name == "figures" else rd / "article" / "figures"
    return figures_root / SLUG / "cache_curves.npz"


def _load_curves(research_dir):
    """优先读 npz 缓存；无缓存则计算并落盘。"""
    cache = _cache_path(research_dir)
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    data = _compute_curves()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def _panel(ax, xp, yp, zp, xm, ym, zm, xlim=None, ylim=None, invert_z=False):
    """在 3D 轴上画两条本征值分支空间曲线（+ 支 / − 支）。"""
    c_p, c_m = palette.color(1), palette.color(2)
    if xlim is not None or ylim is not None:
        # 3D 不裁剪：越界点置 nan 断开曲线（表现发散超出显示范围）。
        # 不可用 np.clip 饱和——那会把复平面上各向飞出的远点钉在方框周长上、
        # 连线后描出方形轮廓伪影。
        xp, yp = xp.astype(float).copy(), yp.astype(float).copy()
        xm, ym = xm.astype(float).copy(), ym.astype(float).copy()
        for X, Y in ((xp, yp), (xm, ym)):
            m = np.zeros(X.shape, dtype=bool)
            if xlim is not None:
                m |= (X < xlim[0]) | (X > xlim[1])
            if ylim is not None:
                m |= (Y < ylim[0]) | (Y > ylim[1])
            X[m] = np.nan
            Y[m] = np.nan
    ax.plot(xp, yp, zp, color=c_p, lw=1.6)
    ax.plot(xm, ym, zm, color=c_m, lw=1.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_z:
        ax.invert_zaxis()
    ax.set_box_aspect((1, 1, Z_STRETCH))   # 竖向拉伸，便于观察简并/劈裂
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(labelsize=6, pad=0)


def build_figure(style=None, research_dir=None, **kwargs):
    """构建三面板 3D 本征值空间曲线图并返回 Figure。"""
    d = _load_curves(research_dir)
    ci_l, ci_g = d["ci_loss"], d["ci_gain"]

    fig = plt.figure()
    fig.set_size_inches(6.693, 6.0)  # 三幅 3D 面板需更高画幅
    # 显式轴位（导出为 bbox 紧裁，tight_layout 对 3D 不可靠）
    rects = [(0.01, 0.10, 0.30, 0.80), (0.35, 0.10, 0.30, 0.80), (0.69, 0.10, 0.30, 0.80)]

    # (a) 镜像损耗 EP：原始 S，本征值合并；竖轴反向以便与 (c) 比较
    ax1 = fig.add_axes(rects[0], projection="3d")
    _panel(ax1,
           d["lam_p_loss"].real, d["lam_p_loss"].imag, ci_l,
           d["lam_m_loss"].real, d["lam_m_loss"].imag, ci_l,
           invert_z=True)
    ax1.set_title("(a) Loss EP: original $S$", fontsize=8)
    ax1.set_xlabel(r"Re($\lambda$)", fontsize=7)
    ax1.set_ylabel(r"Im($\lambda$)", fontsize=7)
    ax1.set_zlabel(r"$c_i$", fontsize=7)

    # (b) 未经变换的 gain EP：原始 S，本征值发散劈裂。
    # 中等轴限使发散回转点落在框内（复现参考图钩形）；越界仍 nan 遮断。
    # 极点处插入 nan 断开：λ 经极点穿过无穷，两侧钩形相向但不连成闭合三角帽。
    ax2 = fig.add_axes(rects[1], projection="3d")
    k = int(np.searchsorted(ci_g, float(d["ci_g_ep"][0])))

    def _brk(a):
        return np.insert(np.asarray(a, dtype=float), k, np.nan)

    ci_gb = np.insert(ci_g, k, np.nan)
    _panel(ax2,
           _brk(d["lam_p_gain"].real), _brk(d["lam_p_gain"].imag), ci_gb,
           _brk(d["lam_m_gain"].real), _brk(d["lam_m_gain"].imag), ci_gb,
           xlim=(-60, 60), ylim=(-60, 60))
    ax2.set_title("(b) Gain EP: original $S$", fontsize=8)
    ax2.set_xlabel(r"Re($\lambda$)", fontsize=7)
    ax2.set_ylabel(r"Im($\lambda$)", fontsize=7)
    ax2.set_zlabel(r"$c_i$", fontsize=7)

    # (c) gain EP 归一化 S'：伪损耗 EP，本征值合并于 0
    ax3 = fig.add_axes(rects[2], projection="3d")
    _panel(ax3,
           d["mu_p_gain"].real, d["mu_p_gain"].imag, ci_g,
           d["mu_m_gain"].real, d["mu_m_gain"].imag, ci_g)
    ax3.set_title(r"(c) Gain EP: normalized $S'$ (pseudo-loss)", fontsize=8)
    ax3.set_xlabel(r"Re($\lambda'$)", fontsize=7)
    ax3.set_ylabel(r"Im($\lambda'$)", fontsize=7)
    ax3.set_zlabel(r"$c_i$", fontsize=7)

    return fig
