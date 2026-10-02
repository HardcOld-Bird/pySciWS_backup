"""CMT 反射型管槽超表面 S 矩阵的 COMSOL 单点验证（损耗 EP）。

一次 JVM 会话完成全部散点求解（COMSOL 比 CMT 慢得多且占用 license）。
验证对象：Fang 型**损耗** EP（ci>0），即第二根管槽等效声速虚部为正（吸收）。

注意：research 1 的核心是**增益**半平面（ci<0）的镜像 "gain EP"，其散射会发散
（谱奇点 / spectral singularity），不在本脚本范围内——本脚本 ci 取正以对齐 CMT/Fang。

关键约定（物理推导 + 无损锚点经验标定）：
- ci 符号 COMSOL 与 CMT 同号（损耗 EP 用 ci>0）。
- 单侧入射用背景场幅值：左=bpf1.pamp=1,bpf2.pamp=0；右=反之。周期条件保持 as-shipped
  （切换 pc1-4 的 active 会破坏周期一致性使散射场 p_s=0）。
- 探针→S 元映射（tbl2 形状 (1,5)；由无损锚点 |S21|=0.9997,|S11|=0.0226 标定）：
    col0 = freq(3430)
    左入射(bpf1.pamp=1): col1 = S11 = r₀ᴸ,   col2 = S21 = r₋₁ᴸ   (col3,col4 恒为 0)
    右入射(bpf2.pamp=1): col3 = S12 = r₊₁ᴿ, col4 = S22 = r₀ᴿ   (col1,col2 恒为 0)

用法（从项目根运行）：
  uv run python scripts/research/gain_ep/simulation/CMT-COMSOL单点验证.py --plan  # 仅 CMT 预测
  uv run python scripts/research/gain_ep/simulation/CMT-COMSOL单点验证.py         # 完整验证（启动 COMSOL）

验证结论：无损锚点 COMSOL 与 CMT 吻合到 4 位有效数字；EP 处单向隐身（左入射双反射抑制、
右入射 |S12|≈1）被复现；唯 S21@EP 因 EP 零点的 √δ 敏感性 + CMT/FEM 位置微偏而离群。
"""

from __future__ import annotations

import sys

import numpy as np

from pysci.paths import research_asset_dir
from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt

_GAIN_EP = research_asset_dir("gain_ep")
MPH = _GAIN_EP / "simulation" / "refs" / "1 周期化-傅里叶变换法-上下对调.mph"

N_ORDERS, K_MODES = 10, 31


def refine_ep(center=(1.0081, 0.0745), half=0.003, step=0.001) -> tuple[float, float]:
    """在名义 EP 附近精细定位（最小化 |S21|=|r₋₁ᴸ|），与验证同截断。

    npz 里的 cr_ep/ci_ep 来自 n_orders=8 粗网格（step 0.005），偏离真实 EP 较远
    （|S21|≈0.021, |λ1-λ2|≈0.29）；此处在 n_orders=10 下精定位到 |S21|≈5e-4 的真 EP。
    """
    cr0, ci0 = center
    best = (cr0, ci0, np.inf)
    for cr in np.arange(cr0 - half, cr0 + half + 1e-12, step):
        for ci in np.arange(ci0 - half, ci0 + half + 1e-12, step):
            S = cmt.compute_s_matrix(
                cmt.make_ep_params(cr=float(cr), ci=float(ci),
                                   n_orders=N_ORDERS, k_modes=K_MODES)
            )
            s21 = abs(S[1, 0])
            if s21 < best[2]:
                best = (float(cr), float(ci), s21)
    print(f"[refine] EP: cr={best[0]:.4f}, ci={best[1]:.4f}, |S21|={best[2]:.2e}")
    return best[0], best[1]


def cmt_S(cr: float, ci: float) -> np.ndarray:
    return cmt.compute_s_matrix(
        cmt.make_ep_params(cr=cr, ci=ci, n_orders=N_ORDERS, k_modes=K_MODES)
    )


def build_points() -> list[tuple[str, float, float, str]]:
    cr_ep, ci_ep = refine_ep()
    return [
        ("lossless", 1.0, 0.0, "LR"),          # 锚点：映射 + 能量守恒
        ("EP", cr_ep, ci_ep, "LR"),            # 关键：单向隐身（非互易反射）
        ("offEP_dci", cr_ep, ci_ep + 0.02, "LR"),  # 偏离 EP：改变损耗
        ("offEP_dcr", cr_ep + 0.02, ci_ep, "LR"),  # 偏离 EP：改变实部
    ]


def map_comsol(side: str, probes: list[complex]) -> dict[str, complex]:
    """按入射方向把 tbl2 探针列映射到 S 矩阵元（无损锚点经验标定）。"""
    if side == "L":
        return {"S11": probes[1], "S21": probes[2]}
    return {"S12": probes[3], "S22": probes[4]}


def map_cmt(side: str, S: np.ndarray) -> dict[str, complex]:
    if side == "L":
        return {"S11": S[0, 0], "S21": S[1, 0]}
    return {"S22": S[1, 1], "S12": S[0, 1]}


def print_row(name, cr, ci, side, comsol: dict, ref: dict) -> None:
    print(f"\n[{name}] cr={cr:.4f} ci={ci:.4f}  incidence={side}")
    for key in comsol:
        c, r = comsol[key], ref[key]
        flag = ""
        if abs(r) < 1e-6:
            flag = "  (CMT≈0)"
        print(
            f"  {key}: |COMSOL|={abs(c):.4f}  |CMT|={abs(r):.4f}  "
            f"|diff|={abs(abs(c) - abs(r)):.4f}{flag}"
        )
    if name == "lossless":
        if side == "L":
            tot_c = abs(comsol["S11"]) ** 2 + abs(comsol["S21"]) ** 2
            tot_r = abs(ref["S11"]) ** 2 + abs(ref["S21"]) ** 2
        else:
            tot_c = abs(comsol["S22"]) ** 2 + abs(comsol["S12"]) ** 2
            tot_r = abs(ref["S22"]) ** 2 + abs(ref["S12"]) ** 2
        print(f"  能量守恒 |r₀|²+|r₋₁|²:  COMSOL={tot_c:.4f}  CMT={tot_r:.4f}  (无损应≈1)")


def plan() -> None:
    pts = build_points()
    print("=" * 78)
    print(f"PLAN: 纯 CMT 预测（无 COMSOL）  n_orders={N_ORDERS} k_modes={K_MODES}")
    print("=" * 78)
    for name, cr, ci, inc in pts:
        S = cmt_S(cr, ci)
        eigs = np.linalg.eigvals(S)
        print(f"\n[{name}] cr={cr:.4f} ci={ci:.4f}")
        print(f"  S = [[{S[0,0]:+.5f}, {S[0,1]:+.5f}],")
        print(f"       [{S[1,0]:+.5f}, {S[1,1]:+.5f}]]")
        print(f"  |S11|={abs(S[0,0]):.4f} |S12|={abs(S[0,1]):.4f} "
              f"|S21|={abs(S[1,0]):.4f} |S22|={abs(S[1,1]):.4f}")
        print(f"  本征值 λ={eigs[0]:+.5f},{eigs[1]:+.5f}  |λ1-λ2|={abs(eigs[0]-eigs[1]):.2e}")
        for side in inc:
            print_row(name, cr, ci, side, map_cmt(side, S), map_cmt(side, S))


def full_run() -> None:
    import mph

    pts = build_points()
    # 预计算 CMT（纯 Python，快）
    cmt_cache = {(n, s): map_cmt(s, cmt_S(cr, ci))
                 for n, cr, ci, inc in pts for s in inc}

    print("=" * 78)
    print("COMSOL vs CMT 单点验证（损耗 EP：ci 同号, pamp 单侧入射, 单会话）")
    print("=" * 78)

    client = mph.start()
    java = client.load(str(MPH)).java
    phys = java.component("comp1").physics("acpr")

    for name, cr, ci, inc in pts:
        java.param().set("cr", str(cr))
        java.param().set("ci", str(ci))
        for side in inc:
            left = side == "L"
            phys.feature("bpf1").set("pamp", "1" if left else "0")
            phys.feature("bpf2").set("pamp", "0" if left else "1")
            try:
                java.study("std1").run()
                tbl = java.result().table("tbl2")
                re = np.asarray(tbl.getReal()).ravel()
                im = np.asarray(tbl.getImag()).ravel()
                if re.size < 5:
                    print(f"\n[{name} {side}] ERROR: tbl2 列数不足 size={re.size}")
                    continue
                probes = [re[k] + 1j * im[k] for k in range(re.size)]
                comsol = map_comsol(side, probes)
                print_row(name, cr, ci, side, comsol, cmt_cache[(name, side)])
            except Exception as e:  # noqa: BLE001
                print(f"\n[{name} {side}] COMSOL ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("验证完成。")
    print("=" * 78)


def main() -> None:
    if "--plan" in sys.argv:
        plan()
    else:
        full_run()


if __name__ == "__main__":
    sys.exit(main())
