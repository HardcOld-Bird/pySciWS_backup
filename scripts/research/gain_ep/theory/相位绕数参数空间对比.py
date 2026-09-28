"""相位绕数的参数空间对比验证：(cr, ci) 解析坐标 vs (h1, w1) 几何坐标。

背景：
    参考图 2ref1d 的相角色图以管槽 1 的 depth/width (d1, w1) 为底面轴，且文献中
    环绕相位累积出现 4π；而我们在 (cr, ci) 空间测得 0 / ±2π。本脚本验证二者是否
    真的不同，并诊断机制。

理论：
    绕数 = 环路内该矩阵元零点/极点指数之和（指数 = 阶数 × 参数化 Jacobian 符号）。
    (cr, ci) 是 S 的解析坐标 ⇒ 简单零点/极点指数必为 ±1（∮dφ=±2π）。
    (h1, w1) 是实几何坐标 ⇒ 指数幅值为 2（4π）仅当二阶零/极点、或环路同时包围
    两个奇点。故对一系列环路半径测绕数：若大半径突变为 ±2 而小半径为 ±1，则为
    "包围第二个奇点"；若任意小半径即为 ±2，则为二阶零/极点。

测量对象：S11, S12, S21, S22 与 det(S)（对照：极点处 det 为二阶极点 ⇒ −4π）。
中心：损耗 EP（S21=0）与增益极点（1/S21=0），均在名义几何下于 (cr, ci) 精定位；
(h1, w1) 空间的中心即名义几何值（该处 S21=0 / 极点条件已满足）。
"""

import numpy as np
from scipy.optimize import root

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt

N_ORDERS, K_MODES = 8, 25
N_LOOP = 360
CR0, CI0 = 1.0081, 0.0745

# 名义几何（make_ep_params 默认）
LAM = 343.0 / 3430.0
A_PER = LAM / np.sqrt(2)
H_NOM = np.array([0.569, 0.195, 0.232]) * LAM
W_NOM = np.array([0.227, 0.115, 0.153]) * A_PER

# 环路尺度：(cr,ci) 用绝对半宽；(h1,w1) 用名义值的相对偏差
SCALE_CRCI = (0.02, 0.025)
SCALE_HW = (0.10, 0.10)      # 最大 ±10% × frac
FRACS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)


def _s_at(cr, ci, h=None, w=None):
    p = cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    if h is not None:
        p.h = np.asarray(h, dtype=float).copy()
    if w is not None:
        p.w = np.asarray(w, dtype=float).copy()
    return cmt.compute_s_matrix(p)


def refine_loss_ep():
    def F(x):
        S = _s_at(x[0], x[1])
        return [S[1, 0].real, S[1, 0].imag]
    r = root(F, [CR0, CI0], method="hybr")
    return float(r.x[0]), float(r.x[1])


def refine_gain_pole():
    def G(x):
        inv = 1.0 / _s_at(x[0], x[1])[1, 0]
        return [inv.real, inv.imag]
    r = root(G, [CR0, -0.0740], method="hybr")
    return float(r.x[0]), float(r.x[1])


def _winding(v):
    phi = np.unwrap(np.angle(v))
    return (phi[-1] - phi[0]) / (2 * np.pi)


def _loop_vals(getter, frac):
    th = np.linspace(0.0, 2 * np.pi, N_LOOP + 1)
    out = {k: np.empty(th.size, dtype=complex)
           for k in ("s11", "s12", "s21", "s22", "det")}
    for i, t in enumerate(th):
        S = getter(frac, np.cos(t), np.sin(t))
        out["s11"][i], out["s12"][i] = S[0, 0], S[0, 1]
        out["s21"][i], out["s22"][i] = S[1, 0], S[1, 1]
        out["det"][i] = np.linalg.det(S)
    return out


def main():
    cr_l, ci_l = refine_loss_ep()
    cr_g, ci_g = refine_gain_pole()
    print(f"loss EP  (cr,ci) = ({cr_l:.6f}, {ci_l:.6f})")
    print(f"gain pole(cr,ci) = ({cr_g:.6f}, {ci_g:.6f})")
    print(f"名义几何 h1={H_NOM[0]:.6f}, w1={W_NOM[0]:.6f}")

    centers = (("LOSS EP", (cr_l, ci_l)), ("GAIN pole", (cr_g, ci_g)))

    for cname, (crc, cic) in centers:
        # --- 空间 A: (cr, ci) 解析坐标 ---
        def getter_crci(frac, c, s, crc=crc, cic=cic):
            return _s_at(crc + frac * SCALE_CRCI[0] * c,
                         cic + frac * SCALE_CRCI[1] * s)
        # --- 空间 B: (h1, w1) 几何坐标（(cr,ci) 固定在中心值）---
        def getter_hw(frac, c, s, crc=crc, cic=cic):
            h = H_NOM.copy()
            w = W_NOM.copy()
            h[0] *= 1.0 + frac * SCALE_HW[0] * c
            w[0] *= 1.0 + frac * SCALE_HW[1] * s
            return _s_at(crc, cic, h=h, w=w)

        for space, getter in (("cr,ci", getter_crci), ("h1,w1", getter_hw)):
            print(f"\n=== {cname} | 参数空间 ({space}) ===")
            print("  frac |  W(S11)  W(S12)  W(S21)  W(S22)  W(det)")
            for frac in FRACS:
                vals = _loop_vals(getter, frac)
                ws = [_winding(vals[k]) for k in ("s11", "s12", "s21", "s22", "det")]
                print(f"  {frac:4.2f} | " + "  ".join(f"{w:+7.2f}" for w in ws))


if __name__ == "__main__":
    main()
