"""6 case 求解 + 导出 runner：3 组对照 × 左右入射。

对照开关（与 build_beam_scatter_mph.PARAMS 一致）：
  - full     : ci=-0.0730324, g13=0        （管槽结构 + 增益介质，完整 gain EP）
  - nogain   : ci=0,          g13=0        （有管槽、无增益介质）
  - nostruct : ci=-0.0730324, g13=0.05λ    （有增益介质、抹平管槽1/3）
入射：L (pampL=1,pampR=0) / R (pampL=0,pampR=1)。

每个 case 输出 field_<cfg>_<inc>.png 与 far_<cfg>_<inc>.png。
"""
from __future__ import annotations

import sys
from pathlib import Path

import mph

from pysci.skills.comsol_simulation.tools.export import export_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_beam_scatter_mph import FIG_DIR, OUT_MPH  # noqa: E402

CI_EP = "-0.0730324"
# 固定场图轴限（数据坐标，单位 m）：6 case 几何相同 → 一个 extent 通用，pixel↔data 映射精确。
# 推导：λ=c0/f=0.1, D=λ/√2, x0=Nper·D/2≈0.283, R=4λ=0.4 → x∈[x0-R,x0+R]=[-0.117,0.683]；
#       管槽下探 ≈ -h1-g13 ≈ -0.062。取整并留边距 → 下列 extent。
FIELD_EXTENT = (-0.13, 0.70, -0.07, 0.42)
FIELD_SIZE = (1000, 590)  # 宽:高 ≈ extent 长宽比，减少空白边距
CASES = [
    # (cfg, ci, g13)
    ("full", CI_EP, "0"),
    ("nogain", "0", "0"),
    ("nostruct", CI_EP, "0.05*lambda"),
]
INCS = [("L", "1", "0"), ("R", "0", "1")]  # (inc, pampL, pampR)


def main() -> None:
    client = mph.start(cores=4)
    model = client.load(str(OUT_MPH))
    jm = model.java
    comp = jm.component("comp1")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for cfg, ci, g13 in CASES:
        for inc, pl, pr in INCS:
            tag = f"{cfg}_{inc}"
            print(f"== case {tag} : ci={ci} g13={g13} pampL={pl} pampR={pr} ==")
            jm.param().set("ci", ci)
            jm.param().set("g13", g13)
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
            # 重绑定绘图组到当前求解数据集（保存的 mph 在 solve 前落盘，plot group 无绑定）
            dsets = [str(t) for t in (jm.result().dataset().tags() or [])]
            print(f"   datasets = {dsets}")
            if dsets:
                for pg in ("pg_field", "pg_far"):
                    try:
                        jm.result(pg).set("data", dsets[0])
                    except Exception as exc:  # noqa: BLE001
                        print(f"   [warn] bind {pg}: {type(exc).__name__}")
            # 场图：固定 extent + clean（隐 colorbar/标题）+ sidecar（供下游 raster-panel 精确叠图）
            r_field = export_image(
                model, "pg_field", FIG_DIR / f"field_{tag}.png",
                size=FIELD_SIZE, extent=FIELD_EXTENT, clean=True, sidecar=True,
            )
            print(r_field.report())
            # 远场极坐标图：PolarGroup 轴系不同，不设 extent；仍写 sidecar 记录空白自检
            r_far = export_image(
                model, "pg_far", FIG_DIR / f"far_{tag}.png", sidecar=True,
            )
            print(r_far.report())
            for pg in ("field", "far"):
                p = FIG_DIR / f"{pg}_{tag}.png"
                print(f"   {p.name} = {p.stat().st_size if p.exists() else 'MISSING'} bytes")

    client.disconnect()
    print("done ->", FIG_DIR)


if __name__ == "__main__":
    main()
