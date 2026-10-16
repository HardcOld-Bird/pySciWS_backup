"""6 case 求解 + 远场数值导出 runner：3 组对照 × 左右入射。

对照开关（与 build_beam_scatter_mph.PARAMS 一致，全部为全局参数）：
  - full     : ci=CI_EP,  h1e=h1,     h3e=h3      （管槽结构 + 增益介质，完整 gain EP）
  - nogain   : ci=0,      h1e=h1,     h3e=h3      （有管槽、无增益介质）
  - nostruct : ci=CI_EP,  h1e=SLIVER, h3e=SLIVER  （有增益介质、抹平管槽 1/3 → 仅剩管槽 2）
入射：L (pampL=1,pampR=0) / R (pampL=0,pampR=1)。

每个 case 输出一份远场**数值** CSV（`far_<cfg>_<inc>.csv`，1° 网格，角度列 + |acpr.efc1.pext|
幅值列）到 `article/figures/beam_scatter/raw/`。远场「评估圆」显式给出：
半径 `R_EVAL = 10*lambda`（= build PARAMS['r_eval']，远大于域半径 R=4λ，落在远场区，
排除弧边界处的近场奇异）、圆心 `(x0, 0)`（超表面几何中心）、角度 0→180 步 1°。
PNG 由 figure 侧 `finalize_beam_scatter_figure.py` 负责，本脚本不产出 PNG。
"""

from __future__ import annotations

import math
from pathlib import Path

import mph

from pysci.skills.comsol_simulation.tools.export import export_polar_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

CI_EP = "-0.0730324"
SLIVER = "lambda/200"  # 管槽 1/3 薄片深度 = 等效整根移除（对照③）

# 几何/评估圆常量（与 build_beam_scatter_mph.PARAMS 同源）
LAMBDA = 343.0 / 3430.0
X0 = 8 * (LAMBDA / math.sqrt(2.0)) / 2.0  # 阵列中心 = PARAMS['x0']
R_EVAL = 10 * LAMBDA  # = PARAMS['r_eval'] = 10*lambda（远场评估圆半径）
ANGLE_MIN, ANGLE_MAX, ANGLE_STEP = 0.0, 180.0, 1.0
EXPR = "abs(acpr.efc1.pext)"  # 外场计算（散射场）声压模

RAW_DIR = (
    ROOT
    / "data"
    / "research"
    / "1_gain_ep"
    / "article"
    / "figures"
    / "beam_scatter"
    / "raw"
)

CASES = [
    # (cfg, ci, h1e, h3e)
    ("full", CI_EP, "h1", "h3"),
    ("nogain", "0", "h1", "h3"),
    ("nostruct", CI_EP, SLIVER, SLIVER),
]
INCS = [("L", "1", "0"), ("R", "0", "1")]  # (inc, pampL, pampR)


def _annotate_csv(path: Path) -> None:
    """在 COMSOL Data 导出的 CSV 头（'%' 注释块内）注明评估圆参数与表达式。

    保持 COMSOL 原生分列/注释格式不变，仅在列头行之前插入元数据行，使文件自描述。
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.startswith("% x,y")), len(lines))
    meta = [
        f"% Expr,{EXPR}",
        f"% EvaluationCircle,radius={R_EVAL:.9f},center=({X0:.9f},0)",
        "% EvaluationCircleNote,center=metasurface geometric center (x0,0);"
        " radius=10*lambda >> domain radius R=4*lambda (far zone);"
        " alpha in degrees, 0..180 step 1",
    ]
    lines[idx:idx] = meta
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from build_beam_scatter_mph import (
        OUT_MPH,  # noqa: PLC0415 - 同目录脚本，路径单一真源
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = mph.start(cores=4)
    model = client.load(str(OUT_MPH))
    jm = model.java
    comp = jm.component("comp1")

    for cfg, ci, h1e, h3e in CASES:
        for inc, pl, pr in INCS:
            tag = f"{cfg}_{inc}"
            print(
                f"== case {tag} : ci={ci} h1e={h1e} h3e={h3e} pampL={pl} pampR={pr} =="
            )
            jm.param().set("ci", ci)
            jm.param().set("h1e", h1e)
            jm.param().set("h3e", h3e)
            jm.param().set("pampL", pl)
            jm.param().set("pampR", pr)
            try:
                comp.geom("geom1").run()
                comp.mesh("mesh1").run()
                jm.study("std1").run()
                print(f"   solve ok ({tag})")
            except Exception as exc:  # noqa: BLE001
                print(f"   [ERROR] solve {tag}: {type(exc).__name__}: {exc}")
                continue
            dsets = [str(t) for t in (jm.result().dataset().tags() or [])]
            print(f"   datasets = {dsets}")
            out = RAW_DIR / f"far_{tag}.csv"
            res = export_polar_csv(
                model,
                out,
                expr=EXPR,
                radius=R_EVAL,
                center=(X0, 0.0),
                angle_min=ANGLE_MIN,
                angle_max=ANGLE_MAX,
                angle_step=ANGLE_STEP,
                source=dsets[0] if dsets else None,
            )
            print("   " + res.report().replace("\n", "\n   "))
            if res.ok and res.out_path is not None:
                _annotate_csv(res.out_path)
                rows = [
                    ln
                    for ln in res.out_path.read_text(encoding="utf-8").splitlines()
                    if ln and not ln.startswith("%")
                ]
                print(
                    f"   {res.out_path.name} = {len(rows)} 数据行, "
                    f"{res.out_path.stat().st_size} bytes"
                )

    client.disconnect()
    print("done ->", RAW_DIR)


if __name__ == "__main__":
    main()
