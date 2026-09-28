"""CMT 反射型管槽超表面散射矩阵计算。

物理模型：
    基于耦合模理论 (Coupled-Mode Theory) 计算一维周期性管槽超表面的反射散射矩阵。
    参考：Fang et al., Phys. Rev. Applied 19, 054003 (2023), Appendix A。

    结构：J 个矩形凹槽构成一个周期，凹槽内可填充等效复声速介质（模拟损耗/增益）。
    入射角 ±45°，周期 D = √2λ/2 确保仅 n=0, ±1 三阶传播模。
    2×2 散射矩阵 S = (r_0^L, r_{+1}^R; r_{-1}^L, r_0^R)。

    通过调节凹槽内复声速 c_s = c_0(c_r + i·c_i) 可实现 EP（例外点）或 DP（diabatic 点）。

计算方法（严格遵循论文 Appendix A, Eqs. A1–A11）：
    1. 自由空间总场展开为 Floquet-Bloch 级数（Eq. A1，N 阶）
    2. 凹槽内场展开为闭端波导模式（Eq. A2，K 阶，含往返因子 U=exp(-2jβl)）
    3. 在 y=0 界面施加压力连续 + 法向速度连续（Eq. A5）
    4. 组装线性方程组 [(-P2,P3);(-V2,V3)]·(A^-;A)=(P1;V1)，求解反射振幅 A^-
    5. 分别对左入射 (θ=+45°) 和右入射 (θ=-45°) 求解，提取 S 矩阵元

参考 MATLAB 代码 reflect_cmt.m / liman.m 的物理逻辑，但矩阵元的符号与指数约定
以论文 Appendix A 为准（详见 solve_reflection_cmt 内注释）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pysci.skills.theoretical_computation.tools import visualize
from pysci.skills.theoretical_computation.tools.session import ensure_session


# ===========================================================================
# 积分核函数（对应 MATLAB 的 N1jifen / fenbujifen / M22jifen）
# ===========================================================================


def _n1_integral(a1: complex, a2: complex, x1: float, x2: float) -> complex:
    """∫_{x1}^{x2} e^{j(a1-a2)x} dx 的解析结果。

    由于 a1, a2 是 Floquet 模式波数（差为 G 的整数倍），
    在整周期上积分时退化为正交性：(x2-x1)·δ_{a1,a2}。
    """
    if np.isclose(a1, a2):
        return x2 - x1
    return 0.0


def _fenbu_integral(t1: complex, s1: float, a1: float, x1: float, x2: float) -> complex:
    """∫_{x1}^{x2} e^{t1·x} · cos(s1·(x - a1)) dx 的解析结果。

    这是自由空间平面波与凹槽波导模式之间的重叠积分。
    三种情况：
      1. t1=0, s1=0 → x2-x1
      2. t1²+s1²=0, t1≠0 → 共振情况（特殊公式）
      3. 一般情况 → 标准解析式
    """
    denom = t1**2 + s1**2

    if np.isclose(t1, 0) and np.isclose(s1, 0):
        return x2 - x1
    elif np.isclose(denom, 0) and not np.isclose(t1, 0):
        # 共振情况：t1² + s1² = 0
        return (
            np.exp(t1 * a1) / 2 * (x2 - x1)
            + (np.exp(t1 * x2 + t1 * (x2 - a1)) - np.exp(t1 * x1 + t1 * (x1 - a1)))
            / 4
            / t1
        )
    else:
        # 一般情况
        val2 = np.exp(t1 * x2) * (
            t1 * np.cos(s1 * (x2 - a1)) + s1 * np.sin(s1 * (x2 - a1))
        )
        val1 = np.exp(t1 * x1) * (
            t1 * np.cos(s1 * (x1 - a1)) + s1 * np.sin(s1 * (x1 - a1))
        )
        return (val2 - val1) / denom


def _m22_integral(a1: float, a2: float, x1: float, x2: float) -> float:
    """∫_{x1}^{x2} cos(a1·(x-x1))·cos(a2·(x-x1)) dx 的解析结果。

    波导模式正交性积分。a1 = k1·π/w, a2 = k2·π/w。
    """
    width = x2 - x1
    if np.isclose(a1, 0) and np.isclose(a2, 0):
        return width
    elif np.isclose(a1, a2):
        return width / 2
    else:
        return 0.0


# ===========================================================================
# 几何参数数据类
# ===========================================================================


@dataclass
class MetagratingParams:
    """反射型管槽超表面几何与物理参数。

    Attributes:
        f0: 工作频率 (Hz)
        c0: 自由空间声速 (m/s)
        theta_i: 入射角 (度)
        J: 凹槽数量（每周期）
        h: 各凹槽深度 (m)，形状 (J,)
        w: 各凹槽宽度 (m)，形状 (J,)
        d: 各凹槽起始偏移/硬边界厚度 (m)，形状 (J,)
        c_complex: 各凹槽内等效复声速 (m/s)，形状 (J,)
        n_orders: 自由空间 Floquet 阶数范围 (-n_orders 到 +n_orders)
        k_modes: 波导模式截断阶数 (0 到 k_modes-1)
    """

    f0: float = 3430.0
    c0: float = 343.0
    theta_i: float = 45.0  # degrees
    J: int = 3
    h: NDArray = None  # (J,) groove depths
    w: NDArray = None  # (J,) groove widths
    d: NDArray = None  # (J,) groove offsets (hard boundary thickness)
    c_complex: NDArray = None  # (J,) complex sound speeds in grooves
    n_orders: int = 10  # Floquet orders: -n_orders to +n_orders
    k_modes: int = 31  # waveguide mode truncation

    def __post_init__(self):
        if self.h is None or self.w is None or self.d is None or self.c_complex is None:
            # 默认使用论文 EP 参数
            lam = self.c0 / self.f0
            a = self.period
            self.h = np.array([0.569, 0.195, 0.232]) * lam
            self.w = np.array([0.227, 0.115, 0.153]) * a
            self.d = np.array([0.0, 0.276, 0.070]) * a
            self.c_complex = np.array([343.0, 343.0 * (1.0081 + 0.0745j), 343.0])

    @property
    def k0(self) -> float:
        """自由空间波数。"""
        return 2 * np.pi * self.f0 / self.c0

    @property
    def wavelength(self) -> float:
        """自由空间波长。"""
        return self.c0 / self.f0

    @property
    def period(self) -> float:
        """光栅周期 a = λ/|sin θ_r - sin θ_i|。"""
        lam = self.wavelength
        theta_r = -self.theta_i
        return abs(lam / (np.sin(np.radians(theta_r)) - np.sin(np.radians(self.theta_i))))

    @property
    def N(self) -> int:
        """Floquet 阶数总数 (2*n_orders + 1)。"""
        return 2 * self.n_orders + 1

    @property
    def K(self) -> int:
        """波导模式数。"""
        return self.k_modes


# ===========================================================================
# CMT 核心求解器
# ===========================================================================


def solve_reflection_cmt(params: MetagratingParams, incidence: str = "left") -> NDArray:
    """求解反射型超表面的 CMT 线性方程组。

    严格遵循论文 Appendix A, Eqs. (A5)–(A11)：
        [(-P2, P3); (-V2, V3)] · (A^-; A)^T = (P1; V1) · A_0^+
    未知量前 N 个为反射振幅 A^-（索引 n_orders ↔ n=0），后 J*K 个为凹槽模振幅。

    关键约定（与 MATLAB 参考代码的差异，已修正）：
      * 往返因子 U = exp(-2j·β·l)（论文 A2/A8/A11），而非 MATLAB 的 exp(+2j·β·h)。
        MATLAB 的符号会使有损槽高阶模 exp(+452)≈1e196 溢出、矩阵病态 cond~1e198。
      * 分支选择 Im(k_y)≤0、Im(β)≤0，保证倏逝波沿传播/深度方向衰减。

    Args:
        params: 超表面参数。
        incidence: "left" (θ_i=+45°) 或 "right" (θ_i=-45°)。

    Returns:
        rn: 反射振幅向量 A^-，形状 (N,)，索引 n_orders 对应 n=0。
    """
    k0 = params.k0
    D = params.period          # 光栅周期（论文记号 D）
    G = 2 * np.pi / D
    J = params.J               # 凹槽数（论文记号 S）
    N = params.N
    K = params.K
    n_orders = params.n_orders

    theta = np.radians(params.theta_i)
    if incidence == "right":
        theta = -theta

    n = np.arange(-n_orders, n_orders + 1)

    # 自由空间横向/纵向波数 (Eq. A1)
    k_x = k0 * np.sin(theta) + n * G                    # (N,)
    k_y = np.sqrt(k0**2 - k_x.astype(complex) ** 2)     # (N,)
    k_y = np.where(k_y.imag > 0, -k_y, k_y)             # Im(k_y) ≤ 0：e^{-j k_y y}(y>0) 衰减

    # 凹槽内等效复波数 (Eq. A2)：k' = 2πf/c_s（复声速 → 复波数）
    kc = 2 * np.pi * params.f0 / params.c_complex       # (J,)
    k_idx = np.arange(K)
    alpha = (k_idx * np.pi)[np.newaxis, :] / params.w[:, np.newaxis]  # α_ks = kπ/t_s (J,K)
    beta = np.sqrt(kc[:, np.newaxis] ** 2 - alpha**2)                 # β_ks (J,K)
    beta = np.where(beta.imag > 0, -beta, beta)         # Im(β) ≤ 0：凹槽内衰减、|U|≤1

    # 凹槽起始位置 x_s
    xj = np.zeros(J)
    xj[0] = params.d[0]
    for j in range(1, J):
        xj[j] = xj[j - 1] + params.w[j - 1] + params.d[j]

    # 往返相位因子 U[s,k] = exp(-2j·β_ks·l_s)  (Eq. A2/A8/A11)
    U_diag = np.exp(-2j * beta * params.h[:, np.newaxis])   # (J,K)

    # --- P_1 (Eq. A6): (J*K,) 入射压力投影 ---
    P1 = np.zeros(J * K, dtype=complex)
    for j in range(J):
        for k in range(K):
            P1[j * K + k] = (1.0 / params.w[j]) * _fenbu_integral(
                -1j * k0 * np.sin(theta),
                alpha[j, k],
                xj[j],
                xj[j],
                xj[j] + params.w[j],
            )

    # --- P_2 (Eq. A7): (J*K, N) 反射压力耦合 ---
    P2 = np.zeros((J * K, N), dtype=complex)
    for j in range(J):
        for k in range(K):
            for nn in range(N):
                P2[j * K + k, nn] = (1.0 / params.w[j]) * _fenbu_integral(
                    -1j * k_x[nn],
                    alpha[j, k],
                    xj[j],
                    xj[j],
                    xj[j] + params.w[j],
                )

    # --- P_3 (Eq. A8): (J*K, J*K) 块对角 = M22_s · (I + U_s) ---
    P3_blocks = []
    for j in range(J):
        Mj = np.zeros((K, K), dtype=complex)
        for k2 in range(K):
            for k1 in range(K):
                Mj[k2, k1] = (1.0 / params.w[j]) * _m22_integral(
                    alpha[j, k2], alpha[j, k1], xj[j], xj[j] + params.w[j]
                )
        Mj = Mj @ np.diag(1.0 + U_diag[j])
        P3_blocks.append(Mj)
    P3 = _block_diag(P3_blocks)

    # --- V_1 (Eq. A9): (N,) = -k0·cosθ_i·δ_{m0} ---
    V1 = np.zeros(N, dtype=complex)
    V1[n_orders] = -k0 * np.cos(theta)

    # --- V_2 (Eq. A10): (N,N) 对角 = k_y[n]·δ_{mn} ---
    V2 = np.diag(k_y)

    # --- V_3 (Eq. A11): (N, J*K) = -(β_ks/D)(1-U)∫cos·e^{+j k_x,m x}dx ---
    V3 = np.zeros((N, J * K), dtype=complex)
    for mm in range(N):
        for j in range(J):
            for k in range(K):
                integral = _fenbu_integral(
                    1j * k_x[mm],
                    alpha[j, k],
                    xj[j],
                    xj[j],
                    xj[j] + params.w[j],
                )
                V3[mm, j * K + k] = (
                    -(beta[j, k] / D) * (1.0 - U_diag[j, k]) * integral
                )

    # --- 组装并求解 (Eq. A5) ---
    M = np.block([[-P2, P3], [-V2, V3]])
    rhs = np.concatenate([P1, V1])
    sol = np.linalg.solve(M, rhs)

    # 前 N 个未知量为反射振幅 A^-（索引 n_orders ↔ n=0）
    rn = sol[:N]
    return rn


def compute_s_matrix(params: MetagratingParams) -> NDArray:
    """计算 2×2 散射矩阵。

    S = [[r_0^L, r_{+1}^R],
         [r_{-1}^L, r_0^R]]

    Returns:
        2×2 复数 S 矩阵。
    """
    n_orders = params.n_orders

    # 左入射 (θ_i = +45°)
    rn_L = solve_reflection_cmt(params, incidence="left")
    # 右入射 (θ_i = -45°)
    rn_R = solve_reflection_cmt(params, incidence="right")

    # 提取 S 矩阵元
    # 左入射：r_0^L = rn_L[n=0], r_{-1}^L = rn_L[n=-1]
    r0_L = rn_L[n_orders]  # n=0
    rm1_L = rn_L[n_orders - 1]  # n=-1

    # 右入射：r_0^R = rn_R[n=0], r_{+1}^R = rn_R[n=+1]
    r0_R = rn_R[n_orders]  # n=0
    rp1_R = rn_R[n_orders + 1]  # n=+1

    S = np.array([[r0_L, rp1_R], [rm1_L, r0_R]], dtype=complex)
    return S


def _block_diag(blocks: list[NDArray]) -> NDArray:
    """构建块对角矩阵（替代 scipy.linalg.block_diag 以保持自包含）。"""
    from scipy.linalg import block_diag

    return block_diag(*blocks)


# ===========================================================================
# 参数化便捷接口
# ===========================================================================


def make_ep_params(
    cr: float = 1.0081,
    ci: float = 0.0745,
    n_orders: int = 10,
    k_modes: int = 31,
) -> MetagratingParams:
    """创建 EP 构型的超表面参数。

    Args:
        cr: 第 2 槽归一化复声速实部。
        ci: 第 2 槽归一化复声速虚部。
        n_orders: Floquet 截断阶数。
        k_modes: 波导模式截断数。
    """
    lam = 343.0 / 3430.0  # 0.1 m
    a = lam / np.sqrt(2)  # period
    return MetagratingParams(
        f0=3430.0,
        c0=343.0,
        theta_i=45.0,
        J=3,
        h=np.array([0.569, 0.195, 0.232]) * lam,
        w=np.array([0.227, 0.115, 0.153]) * a,
        d=np.array([0.0, 0.276, 0.070]) * a,
        c_complex=np.array([343.0, 343.0 * (cr + ci * 1j), 343.0]),
        n_orders=n_orders,
        k_modes=k_modes,
    )


def make_dp_params(
    cr1: float = 1.0,
    ci1: float = 0.240,
    cr3: float = 1.0,
    ci3: float = 0.251,
    n_orders: int = 10,
    k_modes: int = 31,
) -> MetagratingParams:
    """创建 DP 构型的超表面参数。

    DP 构型中第 1、3 槽有损耗，第 2 槽无损耗。
    """
    lam = 343.0 / 3430.0
    a = lam / np.sqrt(2)
    return MetagratingParams(
        f0=3430.0,
        c0=343.0,
        theta_i=45.0,
        J=3,
        h=np.array([0.211, 0.491, 0.210]) * lam,
        w=np.array([0.137, 0.187, 0.145]) * a,
        d=np.array([0.0, 0.096, 0.081]) * a,
        c_complex=np.array(
            [343.0 * (cr1 + ci1 * 1j), 343.0, 343.0 * (cr3 + ci3 * 1j)]
        ),
        n_orders=n_orders,
        k_modes=k_modes,
    )


# ===========================================================================
# 主计算入口
# ===========================================================================


def main(session_dir: Path | None = None) -> None:
    """计算入口：验证收敛 → 定位 EP → 参数空间扫描 → 黎曼面可视化。"""
    if session_dir is None:
        session_dir = ensure_session("gain_ep", "cmt_reflection_s_matrix")
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    print(f"[session] {session_dir}")

    # === 0. 收敛性检查：不同截断阶数对比 ===
    print("\n" + "=" * 60)
    print("Step 0: 收敛性检查 (EP 名义参数, 不同截断)")
    print("=" * 60)
    for n_ord, k_mod in [(4, 11), (6, 21), (8, 31), (10, 31), (12, 41)]:
        p_test = make_ep_params(cr=1.0081, ci=0.0745, n_orders=n_ord, k_modes=k_mod)
        S_test = compute_s_matrix(p_test)
        eigs_test = np.linalg.eigvals(S_test)
        print(
            f"  n_orders={n_ord:2d}, k_modes={k_mod:2d} -> "
            f"|S11|={abs(S_test[0, 0]):.4f}, |S12|={abs(S_test[0, 1]):.4f}, "
            f"|S21|={abs(S_test[1, 0]):.4f}, |S22|={abs(S_test[1, 1]):.4f}, "
            f"|λ1-λ2|={abs(eigs_test[0] - eigs_test[1]):.4e}"
        )

    # === 1. 在 (cr, ci) 参数空间中定位 EP：|r_-1^L|=|S21| → 0 ===
    print("\n" + "=" * 60)
    print("Step 1: 定位 EP（|S21|=|r_-1^L| 最小处, n_orders=8, k_modes=21）")
    print("=" * 60)
    cr_scan = np.arange(0.99, 1.05, 0.005)
    ci_scan = np.arange(0.03, 0.16, 0.005)
    best = (cr_scan[0], ci_scan[0], np.inf, None)
    for cr in cr_scan:
        for ci in ci_scan:
            p = make_ep_params(cr=cr, ci=ci, n_orders=8, k_modes=21)
            S = compute_s_matrix(p)
            s21 = abs(S[1, 0])
            if s21 < best[2]:
                best = (cr, ci, s21, S)
    cr_ep, ci_ep, s21_ep, S_ep = best
    print(f"  EP 候选: cr={cr_ep:.4f}, ci={ci_ep:.4f}, |S21|={s21_ep:.4e}")
    print(f"  S = [[{S_ep[0, 0]:+.5f}, {S_ep[0, 1]:+.5f}],")
    print(f"       [{S_ep[1, 0]:+.5f}, {S_ep[1, 1]:+.5f}]]")
    eigenvalues, eigenvectors = np.linalg.eig(S_ep)
    print(f"  本征值: λ₁={eigenvalues[0]:+.6f}, λ₂={eigenvalues[1]:+.6f}")
    print(f"  |λ₁-λ₂| = {abs(eigenvalues[0] - eigenvalues[1]):.4e}（EP 处应→0）")
    print(f"  Jordan 判据: S11≈S22? {np.isclose(S_ep[0, 0], S_ep[1, 1], atol=1e-3)}, "
          f"|S21|≈0? {s21_ep < 0.05}, |S12|≠0? {abs(S_ep[0, 1]) > 0.3}")

    # === 2. 参数空间扫描（黎曼面），以定位到的 EP 为中心 ===
    print("\n" + "=" * 60)
    print("Step 2: (cr, ci) 参数空间扫描 — 黎曼面")
    print("=" * 60)

    n_scan_orders = 6
    k_scan_modes = 21

    cr_range = np.linspace(cr_ep - 0.02, cr_ep + 0.02, 25)
    ci_range = np.linspace(ci_ep - 0.04, ci_ep + 0.04, 25)

    n_cr, n_ci = len(cr_range), len(ci_range)
    eig1_real = np.zeros((n_cr, n_ci))
    eig1_imag = np.zeros((n_cr, n_ci))
    eig2_real = np.zeros((n_cr, n_ci))
    eig2_imag = np.zeros((n_cr, n_ci))
    s11_abs = np.zeros((n_cr, n_ci))
    s12_abs = np.zeros((n_cr, n_ci))
    s21_abs = np.zeros((n_cr, n_ci))
    s22_abs = np.zeros((n_cr, n_ci))

    total = n_cr * n_ci
    count = 0
    for i, cr in enumerate(cr_range):
        for j, ci in enumerate(ci_range):
            count += 1
            if count % 100 == 0:
                print(f"  进度: {count}/{total}")
            p = make_ep_params(cr=cr, ci=ci, n_orders=n_scan_orders, k_modes=k_scan_modes)
            S = compute_s_matrix(p)
            eigs = np.linalg.eigvals(S)
            eigs = eigs[np.argsort(eigs.real)]
            eig1_real[i, j] = eigs[0].real
            eig1_imag[i, j] = eigs[0].imag
            eig2_real[i, j] = eigs[1].real
            eig2_imag[i, j] = eigs[1].imag
            s11_abs[i, j] = abs(S[0, 0])
            s12_abs[i, j] = abs(S[0, 1])
            s21_abs[i, j] = abs(S[1, 0])
            s22_abs[i, j] = abs(S[1, 1])

    print(f"  扫描完成: {total} 点")

    # === 3. 可视化 ===
    print("\n" + "=" * 60)
    print("Step 3: 可视化")
    print("=" * 60)

    import matplotlib.pyplot as plt

    visualize.setup_style(chinese_fonts=True)

    CR, CI = np.meshgrid(cr_range, ci_range, indexing="ij")

    # --- 图 1: 黎曼面 (3D) ---
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(CI, CR, eig1_real, alpha=0.8, cmap="coolwarm")
    ax1.plot_surface(CI, CR, eig2_real, alpha=0.8, cmap="coolwarm")
    ax1.set_xlabel(r"Im($c_2$)/$c_0$")
    ax1.set_ylabel(r"Re($c_2$)/$c_0$")
    ax1.set_zlabel(r"Re($\lambda$)")
    ax1.set_title("黎曼面: 本征值实部")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(CI, CR, eig1_real, facecolors=plt.cm.viridis(
        (eig1_imag - eig1_imag.min()) / (eig1_imag.max() - eig1_imag.min() + 1e-15)
    ), alpha=0.9)
    ax2.plot_surface(CI, CR, eig2_real, facecolors=plt.cm.viridis(
        (eig2_imag - eig2_imag.min()) / (eig2_imag.max() - eig2_imag.min() + 1e-15)
    ), alpha=0.9)
    ax2.set_xlabel(r"Im($c_2$)/$c_0$")
    ax2.set_ylabel(r"Re($c_2$)/$c_0$")
    ax2.set_zlabel(r"Re($\lambda$)")
    ax2.set_title("黎曼面: 颜色=Im(λ)")

    fig.tight_layout()
    visualize.export_exploration(fig, plots_dir / "riemann_surface_3d.png", close=False)

    # --- 图 2: 本征值劈裂 + S 矩阵元 ---
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        rf"CMT 反射型超表面: EP 参数空间探索 (EP$\approx$cr={cr_ep:.3f}, ci={ci_ep:.3f})",
        fontsize=14,
        fontweight="bold",
    )

    im = axes[0, 0].pcolormesh(CI, CR, np.abs(eig1_real - eig2_real), cmap="hot_r", shading="auto")
    axes[0, 0].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 0].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 0].set_title(r"|Re($\lambda_1$) - Re($\lambda_2$)|")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].pcolormesh(CI, CR, np.abs(eig1_imag - eig2_imag), cmap="hot_r", shading="auto")
    axes[0, 1].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 1].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 1].set_title(r"|Im($\lambda_1$) - Im($\lambda_2$)|")
    plt.colorbar(im, ax=axes[0, 1])

    eig_gap = np.sqrt((eig1_real - eig2_real) ** 2 + (eig1_imag - eig2_imag) ** 2)
    im = axes[0, 2].pcolormesh(CI, CR, eig_gap, cmap="magma_r", shading="auto")
    axes[0, 2].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 2].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 2].set_title(r"|$\lambda_1 - \lambda_2$| (本征值间距)")
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].pcolormesh(CI, CR, s11_abs, cmap="viridis", shading="auto")
    axes[1, 0].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 0].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 0].set_title(r"$|S_{11}| = |r_0^L|$")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].pcolormesh(CI, CR, s12_abs, cmap="viridis", shading="auto")
    axes[1, 1].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 1].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 1].set_title(r"$|S_{12}| = |r_{+1}^R|$")
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].pcolormesh(CI, CR, s21_abs, cmap="viridis", shading="auto")
    axes[1, 2].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 2].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 2].set_title(r"$|S_{21}| = |r_{-1}^L|$")
    plt.colorbar(im, ax=axes[1, 2])

    fig2.tight_layout()
    visualize.export_exploration(fig2, plots_dir / "ep_parameter_space.png")

    # --- 图 3: 沿 ci 方向的 1D 切片（穿过 EP 点），验证 √δ 劈裂 ---
    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle(f"穿过 EP 点的 1D 切片 (cr = {cr_ep:.4f})", fontsize=13)

    cr_idx = np.argmin(np.abs(cr_range - cr_ep))
    ax_a.plot(ci_range, eig1_real[cr_idx, :], "b-o", markersize=3, label=r"Re($\lambda_1$)")
    ax_a.plot(ci_range, eig2_real[cr_idx, :], "r-s", markersize=3, label=r"Re($\lambda_2$)")
    ax_a.set_xlabel(r"Im($c_2$)/$c_0$")
    ax_a.set_ylabel(r"Re($\lambda$)")
    ax_a.set_title("本征值实部")
    ax_a.legend()
    ax_a.axvline(ci_ep, color="gray", linestyle="--", alpha=0.5, label="EP")

    ax_b.plot(ci_range, eig1_imag[cr_idx, :], "b-o", markersize=3, label=r"Im($\lambda_1$)")
    ax_b.plot(ci_range, eig2_imag[cr_idx, :], "r-s", markersize=3, label=r"Im($\lambda_2$)")
    ax_b.set_xlabel(r"Im($c_2$)/$c_0$")
    ax_b.set_ylabel(r"Im($\lambda$)")
    ax_b.set_title("本征值虚部")
    ax_b.legend()
    ax_b.axvline(ci_ep, color="gray", linestyle="--", alpha=0.5)

    fig3.tight_layout()
    visualize.export_exploration(fig3, plots_dir / "ep_slice_1d.png")

    # === 4. 保存数值结果 ===
    np.savez(
        session_dir / "results" / "riemann_scan.npz",
        cr_range=cr_range,
        ci_range=ci_range,
        eig1_real=eig1_real,
        eig1_imag=eig1_imag,
        eig2_real=eig2_real,
        eig2_imag=eig2_imag,
        s11_abs=s11_abs,
        s12_abs=s12_abs,
        s21_abs=s21_abs,
        s22_abs=s22_abs,
        cr_ep=cr_ep,
        ci_ep=ci_ep,
    )
    print(f"\n[done] 数值结果已保存至 {session_dir / 'results' / 'riemann_scan.npz'}")
    print(f"[done] 图表已保存至 {plots_dir}/")


if __name__ == "__main__":
    main()
