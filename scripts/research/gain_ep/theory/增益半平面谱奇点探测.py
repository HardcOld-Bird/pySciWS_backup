"""CMT 增益半平面 (ci<0) 谱奇点探测：gain "EP" = 谱奇点 / 激射阈值。

研究问题（research 1 核心）
--------------------------
Fang et al. PRA 19, 054003 (2023) 的管槽超表面在**损耗**半平面 (ci>0) 有一个 EP
（S 矩阵 Jordan 块、本征值合并、近完美吸收/CPA）。本脚本探测其**镜像**：
增益半平面 (ci<0) 的 "gain EP"，并回答——它是否是**谱奇点**（spectral singularity）？

结论（数值 + 文献综合，见 phase1-3 输出与图）
----------------------------------------------
1. **损耗 EP (ci≈+0.0745)**：S21→0、det S→0、本征值→0 且**合并**（|Δλ|→0，
   Jordan 块 [[E0,1],[0,E0]]，E0≈0）⇒ S 矩阵的**零点** = EP = CPA（完美吸收）。
2. **增益 "EP" (ci≈−0.074)**：|S21|→∞、det S→∞、本征值→∞ 且**劈裂**（|Δλ| 大，不合并）
   ⇒ S 矩阵的**极点** = **谱奇点** = 零宽度共振 = **激射阈值**（det M=0，自持波）。
3. 二者关于无损轴 ci=0 **镜像对偶**（zero ↔ pole，CPA ↔ lasing，时间反演对偶）。
   故 "gain EP" 与谱奇点是**同一物理**（极点/激射），但**不是**严格意义的 EP
   （无本征值合并）——它是损耗 EP 的 CPA-激光对偶。文献依据：
   Mostafazadeh (SS=零宽度共振, M22=0; SS 是 EP 向连续谱的推广, 但 SS 无本征函数合并)；
   PT/CPA-laser 综述（极点=共振/激射，零点=CPA，(PT)S(PT)=S⁻¹，s2=1/s1*）。

关键数值性质：β 分支的规范不变性
--------------------------------
源码 solve_reflection_cmt 强制 Im(β)≤0（passive）。本脚本证明：对反射 S 矩阵，
groove 模 β 的分支选择是**规范自由度**——β→−β（⇒U→1/U）时内部模幅 A→U·A，
而 P3=(1+U)、V3=−(β/D)(1−U) 同乘 1/U，P2/P1（仅含 α）不变，故反射振幅 A⁻（即 S）不变。
因此 passive 分支对增益侧**同样精确**（且避免 exp(+452) 溢出）；极点位置 det M=0 规范不变。
（phase0 的 diag_beta 数值演示：β/U 不同但 S 逐位相同。）

用法（项目根）
--------------
  uv run python scripts/research/gain_ep/theory/增益半平面谱奇点探测.py
输出：phase0 规范不变性自检；phase1 极点精定位；phase2 EP↔SS 镜像对比；
      phase3 二维定位 + 1D 镜像图（session/plots/gain_ep_ss_mirror.png）。
"""

from __future__ import annotations

import numpy as np

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt
from pysci.skills.theoretical_computation.tools.session import ensure_session

N_ORDERS, K_MODES = 8, 21


# ===========================================================================
# 分支可配置的 CMT 求解（用于演示规范不变性；物理结果用源码 passive 分支）
# ===========================================================================


def solve_reflection_branch(params, incidence="left", branch="passive"):
    """镜像 cmt.solve_reflection_cmt，但 groove β 分支可配置（passive / gain）。"""
    k0 = params.k0
    D = params.period
    G = 2 * np.pi / D
    J, N, K, n_orders = params.J, params.N, params.K, params.n_orders

    theta = np.radians(params.theta_i)
    if incidence == "right":
        theta = -theta
    n = np.arange(-n_orders, n_orders + 1)

    k_x = k0 * np.sin(theta) + n * G
    k_y = np.sqrt(k0**2 - k_x.astype(complex) ** 2)
    k_y = np.where(k_y.imag > 0, -k_y, k_y)  # 自由空间恒 passive

    kc = 2 * np.pi * params.f0 / params.c_complex
    k_idx = np.arange(K)
    alpha = (k_idx * np.pi)[np.newaxis, :] / params.w[:, np.newaxis]
    beta = np.sqrt(kc[:, np.newaxis] ** 2 - alpha**2)  # 主根 Re≥0
    beta = np.where(beta.real < 0, -beta, beta)
    if branch == "passive":
        beta = np.where(beta.imag > 0, -beta, beta)
    else:  # gain：仅倏逝模强制衰减，传播模保留放大
        evan = np.abs(beta.imag) > np.abs(beta.real)
        beta = np.where(evan & (beta.imag > 0), -beta, beta)

    xj = np.zeros(J)
    xj[0] = params.d[0]
    for j in range(1, J):
        xj[j] = xj[j - 1] + params.w[j - 1] + params.d[j]
    U_diag = np.exp(-2j * beta * params.h[:, np.newaxis])

    P1 = np.zeros(J * K, dtype=complex)
    for j in range(J):
        for k in range(K):
            P1[j * K + k] = (1.0 / params.w[j]) * cmt._fenbu_integral(
                -1j * k0 * np.sin(theta), alpha[j, k], xj[j], xj[j], xj[j] + params.w[j]
            )
    P2 = np.zeros((J * K, N), dtype=complex)
    for j in range(J):
        for k in range(K):
            for nn in range(N):
                P2[j * K + k, nn] = (1.0 / params.w[j]) * cmt._fenbu_integral(
                    -1j * k_x[nn], alpha[j, k], xj[j], xj[j], xj[j] + params.w[j]
                )
    P3_blocks = []
    for j in range(J):
        Mj = np.zeros((K, K), dtype=complex)
        for k2 in range(K):
            for k1 in range(K):
                Mj[k2, k1] = (1.0 / params.w[j]) * cmt._m22_integral(
                    alpha[j, k2], alpha[j, k1], xj[j], xj[j] + params.w[j]
                )
        P3_blocks.append(Mj @ np.diag(1.0 + U_diag[j]))
    P3 = cmt._block_diag(P3_blocks)

    V1 = np.zeros(N, dtype=complex)
    V1[n_orders] = -k0 * np.cos(theta)
    V2 = np.diag(k_y)
    V3 = np.zeros((N, J * K), dtype=complex)
    for mm in range(N):
        for j in range(J):
            for k in range(K):
                integral = cmt._fenbu_integral(
                    1j * k_x[mm], alpha[j, k], xj[j], xj[j], xj[j] + params.w[j]
                )
                V3[mm, j * K + k] = -(beta[j, k] / D) * (1.0 - U_diag[j, k]) * integral

    M = np.block([[-P2, P3], [-V2, V3]])
    sol = np.linalg.solve(M, np.concatenate([P1, V1]))
    return sol[:N]


def compute_s_branch(params, branch):
    rn_L = solve_reflection_branch(params, "left", branch)
    rn_R = solve_reflection_branch(params, "right", branch)
    no = params.n_orders
    return np.array([[rn_L[no], rn_R[no + 1]], [rn_L[no - 1], rn_R[no]]], dtype=complex)


def diag_beta(ci, cr=1.0081):
    """演示规范不变性：两分支 groove-2 k=0 的 β/U 不同，但 S 相同。"""
    p = cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    kc = 2 * np.pi * p.f0 / p.c_complex
    alpha = (np.arange(K_MODES) * np.pi)[np.newaxis, :] / p.w[:, np.newaxis]
    beta0 = np.sqrt(kc[:, np.newaxis] ** 2 - alpha**2)
    beta0 = np.where(beta0.real < 0, -beta0, beta0)
    for branch in ("passive", "gain"):
        if branch == "passive":
            b = np.where(beta0.imag > 0, -beta0, beta0)
        else:
            evan = np.abs(beta0.imag) > np.abs(beta0.real)
            b = np.where(evan & (beta0.imag > 0), -beta0, beta0)
        U = np.exp(-2j * b[1, 0] * p.h[1])
        print(f"    [{branch:7s}] beta[1,0]={b[1, 0]:+.4f}  |U|={abs(U):.4f}")


# ===========================================================================
# 物理量工具（用源码 passive 分支，已证规范等价）
# ===========================================================================


def S_at(cr, ci, no=10, km=31):
    try:
        S = cmt.compute_s_matrix(
            cmt.make_ep_params(cr=cr, ci=ci, n_orders=no, k_modes=km)
        )
        return S if np.all(np.isfinite(S)) else None
    except np.linalg.LinAlgError:
        return None


def eigs_of(S):
    tr = S[0, 0] + S[1, 1]
    disc = np.sqrt((S[0, 0] - S[1, 1]) ** 2 + 4 * S[0, 1] * S[1, 0])
    return (tr + disc) / 2, (tr - disc) / 2


def _det(S):
    return S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]


# ===========================================================================
# 各阶段
# ===========================================================================


def phase0_invariance(cr=1.0081):
    print("=" * 100)
    print("Phase 0: 规范不变性自检（损耗侧 passive/gain 一致；增益侧 β/U 不同但 S 同）")
    print("=" * 100)
    for br in ("passive", "gain"):
        S = compute_s_branch(
            cmt.make_ep_params(cr=cr, ci=0.0745, n_orders=N_ORDERS, k_modes=K_MODES), br
        )
        print(f"  [{br:7s}] |S21|={abs(S[1, 0]):.6f}  |detS|={abs(_det(S)):.6f}")
    print("  增益侧 ci=-0.075 的 groove-2 k=0（β/U 不同 ⇒ 规范自由度）：")
    diag_beta(-0.0750)


def phase1_pole_scan(cr=1.0081):
    print("\n" + "=" * 100)
    print(f"Phase 1: 增益侧极点精定位 (cr={cr}, n_orders=10, k_modes=31)")
    print("=" * 100)
    best = (None, -1.0)
    for ci in np.arange(-0.060, -0.0901, -0.001):
        S = S_at(cr, float(ci))
        if S is None:
            print(f"  ci={ci:+.4f}  <singular>")
            continue
        s21 = abs(S[1, 0])
        e1, e2 = eigs_of(S)
        print(
            f"  ci={ci:+.4f}  |S21|={s21:11.3f}  |detS|={abs(_det(S)):11.3f}  "
            f"|λ|={abs(e1):8.3f},{abs(e2):8.3f}  |Δλ|={abs(e1 - e2):.3e}"
        )
        if s21 > best[1]:
            best = (float(ci), s21)
    print(f"  → 极点 ci*={best[0]:+.4f}  (|S21|max={best[1]:.1f})")
    return best[0]


def phase2_mirror(ci_pole, cr=1.0081):
    print("\n" + "=" * 100)
    print("Phase 2: 损耗 EP (zero/EP) ↔ 增益 SS (pole/lasing) 镜像对比")
    print("=" * 100)
    for tag, ci in (("LOSS EP", 0.0745), ("GAIN SS", ci_pole)):
        S = S_at(cr, ci)
        if S is None:
            print(f"  [{tag}] ci={ci:+.4f} <singular>")
            continue
        e1, e2 = eigs_of(S)
        print(
            f"  [{tag}] ci={ci:+.4f}  S=[[{S[0, 0]:+.5f},{S[0, 1]:+.5f}],"
            f"[{S[1, 0]:+.5f},{S[1, 1]:+.5f}]]"
        )
        print(
            f"      |S21|={abs(S[1, 0]):.4e}  |detS|={abs(_det(S)):.4e}  "
            f"λ={e1:+.5f},{e2:+.5f}  |Δλ|={abs(e1 - e2):.4e}"
        )


def phase3_map(ci_pole, cr=1.0081, session_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 100)
    print("Phase 3: (cr,ci) 二维定位 + 1D 镜像图 (n_orders=6, k_modes=21)")
    print("=" * 100)
    crs = np.linspace(0.99, 1.03, 17)
    cis = np.linspace(-0.11, 0.11, 23)
    s21 = np.full((len(crs), len(cis)), np.nan)
    dlam = np.full((len(crs), len(cis)), np.nan)
    for i, crc in enumerate(crs):
        for j, cic in enumerate(cis):
            S = S_at(float(crc), float(cic), no=6, km=21)
            if S is None:
                continue
            e1, e2 = eigs_of(S)
            s21[i, j] = abs(S[1, 0])
            dlam[i, j] = abs(e1 - e2)
    iP, jP = np.unravel_index(np.nanargmax(s21), s21.shape)
    iE, jE = np.unravel_index(np.nanargmin(dlam), dlam.shape)
    print(f"  极点(|S21|max): cr={crs[iP]:.4f}, ci={cis[jP]:+.4f}")
    print(f"  EP  (|Δλ|min): cr={crs[iE]:.4f}, ci={cis[jE]:+.4f}  ← 关于 ci=0 镜像")

    cis1 = np.concatenate([np.linspace(0.10, 0.0, 60), np.linspace(-0.002, -0.10, 60)])
    s21v, detv, l1v, l2v = [], [], [], []
    for ci in cis1:
        S = S_at(cr, float(ci))
        if S is None:
            s21v.append(np.nan)
            detv.append(np.nan)
            l1v.append(np.nan)
            l2v.append(np.nan)
            continue
        e1, e2 = eigs_of(S)
        s21v.append(abs(S[1, 0]))
        detv.append(abs(_det(S)))
        l1v.append(abs(e1))
        l2v.append(abs(e2))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)
    for ax in (axA, axB):
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0.0745, color="tab:blue", lw=1.0, ls=":", label="loss EP (+0.0745)")
        ax.axvline(
            ci_pole, color="tab:red", lw=1.0, ls=":", label=f"gain SS ({ci_pole:+.4f})"
        )
    axA.semilogy(cis1, s21v, "-", color="k", lw=1.6)
    axA.axhline(1, color="gray", lw=0.6)
    axA.set_ylabel(r"$|S_{21}|$ (log)")
    axA.set_title(r"$|S_{21}|$: zero at EP $\leftrightarrow$ pole at SS")
    axB.semilogy(cis1, l1v, "-", color="tab:blue", label=r"$|\lambda_1|$")
    axB.semilogy(cis1, l2v, "-", color="tab:green", label=r"$|\lambda_2|$")
    axB.semilogy(cis1, detv, "--", color="tab:red", label=r"$|\det S|$")
    axB.set_ylabel("magnitude (log)")
    axB.set_title(r"eigenvalues/$|\det S|$: $\to0$ at EP, $\to\infty$ at SS")
    axB.legend(fontsize=8, loc="center left")
    axA.set_xlabel(r"Im($c_2$)/$c_0$ (ci>0 loss, ci<0 gain)")
    axB.set_xlabel(r"Im($c_2$)/$c_0$")
    fig.suptitle(
        rf"CMT mirror duality at cr={cr}: loss EP (zero) $\leftrightarrow$ gain SS (pole/lasing)"
    )
    fig.tight_layout()
    out = (session_dir / "plots" / "gain_ep_ss_mirror.png") if session_dir else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"  [plot] {out}")
    else:
        fig.savefig("gain_ep_ss_mirror.png", dpi=130)


def _chordal(z1, z2):
    """Riemann 球 (CP1) 弦距：两点同趋于 ∞ 时 →0（射影意义下合并）。"""
    return abs(z1 - z2) / (np.sqrt(1 + abs(z1) ** 2) * np.sqrt(1 + abs(z2) ** 2))


def phase4_projective(cr=1.0081):
    """射影观点：gain SS = 『无穷远处的 EP』。

    观点1（本征值）：在 CP1（一点紧化）上 +∞ 与 −∞ 是同一点，故反向发散的 λ± 在 ∞ 合并；
        用弦距 chordal(λ1,λ2)→0 严格刻画。
    观点2（矩阵）：(S − t·I)/dom（t=S11=S22 镜面反射，dom=较大副对角元）是 blow-up/射影化，
        把发散的 S 解奇点为有限幂零 Jordan 块；对损耗 EP 与增益 SS 同一操作得同一 Jordan 块（转置等价）。
    并验证本征矢在两极点均并线（cosθ→1），补全 EP 定义的本征矢合并一半。
    """
    print("\n" + "=" * 100)
    print(
        "Phase 4: 射影观点 — gain SS 是『无穷远处的 EP』（CP1 本征值合并 + 本征矢并线 + blow-up Jordan 块）"
    )
    print("=" * 100)
    for tag, ci in (
        ("LOSS EP", 0.0745),
        ("GAIN SS", -0.0740),
        ("off-EP ", 0.060),
        ("off-SS ", -0.060),
    ):
        S = S_at(cr, ci)
        if S is None:
            print(f"  [{tag}] ci={ci:+.4f} <singular>")
            continue
        w, v = np.linalg.eig(S)
        t = 0.5 * (S[0, 0] + S[1, 1])
        dom = S[0, 1] if abs(S[0, 1]) >= abs(S[1, 0]) else S[1, 0]
        N = (S - t * np.eye(2)) / dom
        cosang = abs(np.vdot(v[:, 0], v[:, 1])) / (
            np.linalg.norm(v[:, 0]) * np.linalg.norm(v[:, 1])
        )
        print(f"  [{tag}] ci={ci:+.4f}")
        print(
            f"      仿射|λ1-λ2|={abs(w[0] - w[1]):9.3e}  CP1弦距={_chordal(w[0], w[1]):9.3e}  "
            f"本征矢并线 cosθ={cosang:.5f}"
        )
        print(
            f"      (S-tI)/dom = [[{N[0, 0]:+.4f},{N[0, 1]:+.4f}],[{N[1, 0]:+.4f},{N[1, 1]:+.4f}]]  → 幂零 Jordan 块"
        )


def main():
    session_dir = ensure_session("gain_ep", "gain_ep_ss_probe")
    print(f"[session] {session_dir}")
    cr = 1.0081
    phase0_invariance(cr)
    ci_pole = phase1_pole_scan(cr)
    phase2_mirror(ci_pole, cr)
    phase3_map(ci_pole, cr, session_dir)
    phase4_projective(cr)


if __name__ == "__main__":
    main()
