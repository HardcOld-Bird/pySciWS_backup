"""增益工作点扫描（ci 扫描）远场导出：补充材料图（twin 裁决 C）的数据面。

对照③（gain, no groove：`h1e=h3e=λ/200`，`ci` 递变）在 ci ∈ [−0.03, −0.073] 上的
远场角谱逐档重算并导出 polar CSV（1° 网格 181 点），供补充图同时产出：
①主瓣角/幅值随 ci 的演化曲线；②ci=−0.05 与 ci=−0.073 两幅极坐标快照。

模型与域半径**不动**（R = 4λ，与主图/sim B.b2_pre 同一模型）——扩域实验已证会破坏
远场（见 `export_expanded_domain.py` 的黄金判据对照），故扫描必须在原域上做。

用法::

    uv run --no-sync python scripts/research/gain_ep/simulation/export_ci_scan.py \
        --outdir data/research/1_gain_ep/article/figures/beam_scatter/raw/ci_scan
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mph
import numpy as np

from pysci.skills.comsol_simulation.tools.export import export_polar_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT_MPH = (
    ROOT / "data" / "research" / "1_gain_ep" / "simulation" / "6 半圆波束散射对照.mph"
)

CI_EP = "-0.0730324"
SLIVER = "lambda/200"
LAMBDA = 343.0 / 3430.0
X0 = 4 * (LAMBDA / math.sqrt(2.0))
R_EVAL = 10 * LAMBDA
#: 扫描档位（含端点：−0.03 与交付值 −0.0730324）；−0.05 供极坐标快照
CI_VALUES = ("-0.03", "-0.04", "-0.05", "-0.06", "-0.07", CI_EP)
INCS = (("L", "1", "0"), ("R", "0", "1"))


def _tag(ci: str) -> str:
    """ci 值 → 文件名标签（小数点保留，负号作 m，如 -0.05 → m0.05）。"""
    return ("m" if ci.startswith("-") else "p") + ci.lstrip("+-")


def main() -> int:
    ap = argparse.ArgumentParser(description="ci 扫描远场导出（对照③）")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    client = mph.start(cores=4)
    model = client.load(str(OUT_MPH))
    jm = model.java
    comp = jm.component("comp1")
    print(f"== ci 扫描（对照③：h1e=h3e={SLIVER}）档位 {len(CI_VALUES)} × 入射 2 ==")
    for ci in CI_VALUES:
        for inc, pl, pr in INCS:
            jm.param().set("ci", ci)
            jm.param().set("h1e", SLIVER)
            jm.param().set("h3e", SLIVER)
            jm.param().set("pampL", pl)
            jm.param().set("pampR", pr)
            try:
                comp.geom("geom1").run()
                comp.mesh("mesh1").run()
                jm.study("std1").run()
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] ci={ci} {inc}: {type(exc).__name__}: {exc}")
                continue
            out = outdir / f"far_nostruct_{inc}_ci{_tag(ci)}.csv"
            res = export_polar_csv(
                model,
                out,
                expr="abs(acpr.efc1.pext)",
                radius=R_EVAL,
                center=(X0, 0.0),
                angle_min=0.0,
                angle_max=180.0,
                angle_step=1.0,
                source="dset1",
            )
            if res.ok and res.out_path is not None:
                a = np.loadtxt(res.out_path, delimiter=",", comments="%")
                ang, amp = a[:, 2], a[:, 3]
                print(
                    f"  ci={ci:>10s} {inc}: argmax={ang[int(np.argmax(amp))]:.0f}° "
                    f"max={amp.max():.4f} v45={amp[45]:.4f} v135={amp[135]:.4f} -> {out.name}"
                )
            else:
                print(f"  [ERR] ci={ci} {inc}: {res.error}")

    client.disconnect()
    print("done ->", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
