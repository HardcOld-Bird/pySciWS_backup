"""CMT 反射模板批量求解与全传播阶次提取（单 JVM 会话）。

用法（项目根）::

    uv run python scripts/research/gain_ep/simulation/batch_cmt_orders.py \
        --template "data/research/1_gain_ep/simulation/12 反射管槽_J3模板.mph" \
        --point 0.070710678,45 --point 0.15,30 \
        --out data/research/1_gain_ep/simulation/cmt_batch.json

每个设计点按 (f, theta, D) 计算传播阶次集合 n: |kx0+n*2*pi/D| <= k，沿
scan1（L 入射）/scan2（R 入射）导出散射场 p_s，做空间 Fourier 积分
r_n = (1/L)∫ p_s exp(-i kx_n x) dx；不硬编码 0/±1。
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
from pysci.skills.comsol_simulation.tools.export import export_data


def order_set(
    f: float, theta_deg: float, D: float, c0: float = 343.0
) -> list[tuple[int, float]]:
    k = 2.0 * math.pi * f / c0
    kx0 = math.sin(math.radians(theta_deg)) * k
    step = 2.0 * math.pi / D
    n_min = int(math.ceil((-k - kx0) / step))
    n_max = int(math.floor((k - kx0) / step))
    return [(n, kx0 + n * step) for n in range(n_min, n_max + 1)]


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
        orders = order_set(3430.0, theta, D)
        entry = {
            "D_m": D,
            "theta_deg": theta,
            "orders": [n for n, _ in orders],
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
            coeffs = {}
            for n, kxn in orders:
                r = np.trapezoid(ps * np.exp(-1j * kxn * x), x) / (2.0 * D)
                coeffs[str(n)] = {
                    "kx_over_k": kxn / (2 * math.pi * 3430.0 / 343.0),
                    "abs": float(abs(r)),
                }
            entry["sides"][side] = coeffs
            print(
                f"D={D:.6g} theta={theta:.3g} side={side} orders={[n for n, _ in orders]} "
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
