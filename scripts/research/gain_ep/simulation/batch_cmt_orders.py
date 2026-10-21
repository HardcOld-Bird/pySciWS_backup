"""CMT 反射模板批量求解与全传播阶次提取（单 JVM 会话）。

用法（项目根）::

    uv run python scripts/research/gain_ep/simulation/batch_cmt_orders.py \
        --template "data/research/1_gain_ep/simulation/12 反射管槽_J3模板.mph" \
        --point 0.070710678,45 --point 0.15,30 \
        --out data/research/1_gain_ep/simulation/cmt_batch.json

每个设计点用 **参考端口栅格**（`solve_s_matrix(MetagratingParams(f0, theta, D)).ports`）
确定传播阶次与 kx，沿 scan1（L 入射）/scan2（R 入射）导出散射场 p_s，做空间 Fourier
积分 ``r_m = (1/2D) ∫ p_s exp(+i kx_m x) dx``（扫描场 x 相位为 e^{-i kx x}，故提取
用正号；旧版用入射栅格 + 负号，3 端口时 L 侧漏掉传播阶）。输出 JSON 保存复数
``re/im/abs/phase_rad``，不硬编码 0/±1。
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import mph
import numpy as np

from pysci.paths import research_asset_dir
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParams,
    solve_s_matrix,
)
from pysci.skills.comsol_simulation.tools.export import export_data


def reference_ports(f: float, theta_deg: float, D: float, c0: float = 343.0):
    """参考端口栅格（端口集合/顺序/ kx 与 CMT ``S`` 完全一致）。

    端口集合只依赖 (f, θi, D)，与槽几何无关；``MetagratingParams`` 的几何缺省
    （EP 名义参数）足够定端口。
    """
    params = MetagratingParams(f0=f, c0=c0, theta_i=theta_deg, period=D)
    return solve_s_matrix(params).ports


def extract_coefficients(x, ps, ports, D: float, *, k: float) -> dict:
    """按参考端口栅格提取复反射系数 ``r_m = (1/2D)∫ p_s e^{+i kx_m x} dx``。

    扫描场 x 相位是 e^{-i kx x}，故提取用正号；返回每阶的
    ``kx_over_k/re/im/abs/phase_rad``（JSON 友好）。
    """
    coeffs = {}
    for port in ports:
        r = np.trapezoid(ps * np.exp(1j * port.kx * x), x) / (2.0 * D)
        coeffs[str(port.order)] = {
            "kx_over_k": port.kx / k,
            "re": float(r.real),
            "im": float(r.imag),
            "abs": float(abs(r)),
            "phase_rad": float(np.angle(r)),
        }
    return coeffs


def main() -> int:
    ap = argparse.ArgumentParser(description="CMT 模板批量 S 阶次提取（单 JVM）")
    ap.add_argument(
        "--template",
        default=str(
            research_asset_dir("gain_ep") / "simulation" / "12 反射管槽_J3模板.mph"
        ),
    )
    ap.add_argument(
        "--point", action="append", required=True, help="D[,theta_deg]，可重复"
    )
    ap.add_argument(
        "--out",
        default=str(research_asset_dir("gain_ep") / "simulation" / "cmt_batch.json"),
    )
    ap.add_argument("--sides", default="L,R")
    ap.add_argument(
        "--set-param", action="append", default=[], help="逐槽/材料参数 k=v，可重复"
    )
    args = ap.parse_args()
    extra = {}
    for kv in args.set_param:
        k, v = kv.split("=", 1)
        extra[k.strip()] = v.strip()

    points = []
    for spec in args.point:
        parts = [p for p in spec.split(",") if p]
        D = float(parts[0])
        theta = float(parts[1]) if len(parts) > 1 else 45.0
        points.append((D, theta))

    client = mph.start(cores=4)
    model = client.load(str(args.template))
    jm = model.java
    comp = jm.component("comp1")
    ph = comp.physics("acpr")
    tmp = Path(tempfile.gettempdir()) / "cmt_batch_scan.csv"
    result = []

    for D, theta in points:
        jm.param().set("D", f"{D:.9g}[m]")
        jm.param().set("theta", f"{theta:.9g}*pi/180")
        for k, v in extra.items():
            jm.param().set(k, v)
        comp.geom("geom1").run()
        comp.mesh("mesh1").run()
        ports = reference_ports(3430.0, theta, D)
        k = 2.0 * math.pi * 3430.0 / 343.0
        entry = {
            "D_m": D,
            "theta_deg": theta,
            "orders": [p.order for p in ports],
            "sides": {},
        }
        for side in [s for s in args.sides.split(",") if s]:
            ph.feature("bpf1").set("pamp", "1" if side == "L" else "0")
            ph.feature("bpf2").set("pamp", "0" if side == "L" else "1")
            jm.study("std1").run()
            line = "scan1" if side == "L" else "scan2"
            paths = []
            for expr, suf in (("real(acpr.p_s)", "re"), ("imag(acpr.p_s)", "im")):
                p = tmp.with_name(f"cmt_batch_{side}_{suf}.csv")
                if not export_data(model, line, p, expr=expr).ok:
                    raise RuntimeError(f"scan export failed: {side}/{suf}")
                paths.append(p)
            a = np.loadtxt(paths[0], delimiter=",", comments="%")
            b = np.loadtxt(paths[1], delimiter=",", comments="%")
            x = a[:, 0]
            ps = a[:, -1] + 1j * b[:, -1]
            coeffs = extract_coefficients(x, ps, ports, D, k=k)
            entry["sides"][side] = coeffs
            print(
                f"D={D:.6g} theta={theta:.3g} side={side} "
                f"orders={[p.order for p in ports]} "
                f"|r|={[round(v['abs'], 5) for v in coeffs.values()]}"
            )
        result.append(entry)

    client.disconnect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
