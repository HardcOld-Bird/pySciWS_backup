"""validate_s_matrix 的四个对照用例（单 JVM 会话，结果落 simulation/cmt_validation.json）。

用法（项目根，需 COMSOL Server）：
    uv run --no-sync python scripts/research/gain_ep/theory/validate_s_matrix_cases.py

用例：
1. 无损锚点（J=3, D=λ/√2, θ=45°, c=343）——沿用 t4 4 位有效数字口径；
2. J=4（前 3 槽同锚点，追加第 4 槽）——闭合 t4 的「J>3 未对照」缺口；
3. 含损（第 2 槽 cr=1.005, ci=0.05）——非平凡复声速；
4. 3 端口（J=3, D=1.5λ, θ=30°）——端口维度/语义（[0,−1,−2]）。
"""

from __future__ import annotations

import json

import numpy as np

from pysci.paths import assert_within_data, research_asset_dir
from pysci.research.gain_ep.theory.cmt_comsol_validation import (
    compare_s_matrix,
    validate_s_matrix_batch,
)
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParams,
    compute_s_matrix,
    solve_s_matrix,
)

LAM = 343.0 / 3430.0
C0 = 343.0
D0 = LAM / np.sqrt(2.0)


def _anchor_geom(J: int):
    h = [0.569, 0.195, 0.232]
    w = [0.227, 0.115, 0.153]
    x = [0.0, 0.503, 0.688]
    if J == 4:
        h.append(0.30)
        w.append(0.08)
        x.append(0.86)
    return (
        np.array(h[:J]) * LAM,
        np.array(w[:J]) * D0,
        np.array(x[:J]) * D0,
    )


def _params(J=3, theta=45.0, D=D0, c=None) -> MetagratingParams:
    h, w, x = _anchor_geom(J)
    if c is None:
        c = np.full(J, C0, dtype=complex)
    return MetagratingParams(
        f0=C0 / LAM,
        c0=C0,
        theta_i=theta,
        period=D,
        J=J,
        h=h,
        w=w,
        x=x,
        c_complex=np.asarray(c, dtype=complex),
        n_orders=10,
        k_modes=31,
    )


def main() -> None:
    cases = {
        "lossless_anchor_J3": _params(),
        "J4_appended": _params(J=4),
        "lossy_ci2_0p05": _params(c=[C0, C0 * (1.005 + 0.05j), C0]),
        "three_port_D1p5lam_th30": _params(theta=30.0, D=1.5 * LAM),
    }
    names = list(cases)
    comsol = validate_s_matrix_batch([cases[name] for name in names], cores=4)
    report = {}
    for name, S_comsol in zip(names, comsol):
        S_cmt = compute_s_matrix(cases[name])
        cmp = compare_s_matrix(S_cmt, S_comsol)
        report[name] = {
            "theta_i": cases[name].theta_i,
            "period": cases[name].period,
            "J": cases[name].J,
            "ports": [p.order for p in solve_s_matrix(cases[name]).ports],
            **cmp,
        }
        print(
            f"{name}: shape={S_cmt.shape} max|Δ|={cmp['max_abs_diff']:.3e} "
            f"max_rel={cmp['max_rel_diff']:.3e} rms={cmp['rms_diff']:.3e}"
        )
        print("  CMT   |S| =", np.round(cmp["cmt_abs"], 5).tolist())
        print("  FEM   |S| =", np.round(cmp["comsol_abs"], 5).tolist())

    out = research_asset_dir("gain_ep") / "simulation" / "cmt_validation.json"
    assert_within_data(out)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("saved ->", out)


if __name__ == "__main__":
    main()
