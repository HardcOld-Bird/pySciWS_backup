"""扩域重算：域半径 R = 4λ → 7λ，6 组场图/远场全量重算导出（内框几何可行的前提）。

**为什么扩域**（用户批语 1 / 任务书硬性条款 1）：实验扫描框边长 31 cm、近边距超表面中心
30 cm，取尖角向上（中心沿 135°/45° 通道射线）时其径向跨度为 30 → 61 cm、远侧顶点
r = 62.94 cm；而原模型域半径 `R = 4λ = 40 cm`，故该框**放不下**——「内侧方框完整位于
半圆区域内」不成立（近而不交亦无从谈起）。用户裁决的处置＝**扩大仿真区域、图面整体缩小
呈现**；本脚本即该处置的数据面：域半径改为 `R = 7λ = 70 cm`（远侧顶点距弧 7.06 cm，
约 10% R 的边距），物理设置（完美匹配边界 PMB、背景平面波 bpf、增益 ci、管槽 h1e/h3e、
远场评估圆 r_eval = 10λ 与圆心 (x0,0)）**一律不动**，只放大域。

与 `runner_beam_scatter.py` 的关系：后者是 R = 4λ 下的远场交付复现件（B.b2_pre 已交付，
留档不动）；本脚本是**扩域版**，一次重算场图 + 远场两套数据，并打印新旧对照（重叠区内
场图逐点对照 + 远场主瓣角逐项对照），供「扩域是否改变物理结论」的现场判读。

产物先写 `--staging`（默认 pod 外不可写，须显式指定暂存目录），**人工核对后再提升**到
`data/research/1_gain_ep/article/figures/beam_scatter/raw/`（提升由调用方决定，本脚本只
写 staging）。

用法::

    uv run --no-sync python scripts/research/gain_ep/simulation/export_expanded_domain.py \
        --staging orchestration/pods/figure/bench/expand_r7
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import mph
import numpy as np

from pysci.skills.comsol_simulation.tools.export import export_data, export_polar_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIM_DIR = ROOT / "data" / "research" / "1_gain_ep" / "simulation"
OUT_MPH = SIM_DIR / "6 半圆波束散射对照.mph"
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

CI_EP = "-0.0730324"
SLIVER = "lambda/200"
LAMBDA = 343.0 / 3430.0
X0 = 4 * (LAMBDA / math.sqrt(2.0))
R_EVAL = 10 * LAMBDA
R_OLD = "4*lambda"
R_NEW = "7*lambda"

CASES = [
    ("full", CI_EP, "h1", "h3"),
    ("nogain", "0", "h1", "h3"),
    ("nostruct", CI_EP, SLIVER, SLIVER),
]
INCS = [("L", "1", "0"), ("R", "0", "1")]


def _diag(jm) -> None:
    """打印关键设置的选择实体数（几何重建后选择是否仍指向正确边界）。"""
    comp = jm.component("comp1")
    phys = comp.physics("acpr")
    for tag in ("pmb1", "bpf1", "bpf2", "efc1"):
        try:
            f = phys.feature(tag)
            b = list(f.selection().entities(1) or [])
            d = list(f.selection().entities(3) or [])
            print(f"  [diag] {tag}: bnds={len(b)} {b[:6]} | doms={len(d)} {d[:6]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [diag] {tag}: {type(exc).__name__}: {exc}")
    for tag in ("sel_arc", "sel_dom"):
        try:
            sel = comp.selection(tag)
            print(f"  [diag] {tag}: bnds={len(list(sel.entities(1) or []))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [diag] {tag}: {type(exc).__name__}: {exc}")
    try:
        m = comp.mesh("mesh1")
        print(f"  [diag] mesh stat = {m.stat().getNumElem()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [diag] mesh stat 不可读: {type(exc).__name__}: {exc}")


def _compare_far(old: Path, new: Path) -> str:
    """远场新旧对照：主瓣角逐项 + 幅值最大相对差。"""
    if not old.exists():
        return f"    (旧件缺失，跳过对照：{old.name})"
    a = np.loadtxt(old, delimiter=",", comments="%")
    b = np.loadtxt(new, delimiter=",", comments="%")
    if a.shape != b.shape:
        return f"    (形状不同 {a.shape} vs {b.shape})"
    ang_o, amp_o, ang_n, amp_n = a[:, 2], a[:, 3], b[:, 2], b[:, 3]
    pk_o, pk_n = ang_o[int(np.argmax(amp_o))], ang_n[int(np.argmax(amp_n))]
    rel = float(np.max(np.abs(amp_n - amp_o)) / max(amp_o.max(), 1e-30))
    return (
        f"    peak {pk_o:.0f}° -> {pk_n:.0f}° | max|Δ|/max = {rel * 100:.2f}% | "
        f"max {amp_o.max():.4g} -> {amp_n.max():.4g}"
    )


def _compare_field(old: Path, new: Path, tol: float = 1e-6) -> str:
    """场图新旧对照：把新网格插值到旧网格点上，报重叠区内相对偏差。"""
    if not old.exists():
        return f"    (旧件缺失，跳过对照：{old.name})"
    a = np.loadtxt(old, delimiter=",", comments="%")
    b = np.loadtxt(new, delimiter=",", comments="%")
    xa, ya, va = a[:, 0], a[:, 1], a[:, 2]
    xb, yb, vb = b[:, 0], b[:, 1], b[:, 2]
    keep = (ya >= 0.0) & (np.hypot(xa - X0, ya) <= 4 * LAMBDA)
    xa, ya, va = xa[keep], ya[keep], va[keep]
    from scipy.spatial import cKDTree  # noqa: PLC0415 - 仅本诊断用

    tree = cKDTree(np.column_stack([xb, yb]))
    dist, pos = tree.query(np.column_stack([xa, ya]), k=1)
    near = dist <= 0.5e-3
    if near.sum() < 10:
        return f"    (重叠采样点不足 {int(near.sum())}，跳过)"
    ref, got = va[near], vb[pos[near]]
    rel = np.abs(got - ref) / max(float(np.abs(ref).max()), 1e-30)
    return (
        f"    重叠点 {int(near.sum())}/{len(va)}（阈值 {tol} m）| "
        f"max|Δ|/max|ref| = {rel.max() * 100:.2f}% | 均值 = {rel.mean() * 100:.2f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="扩域重算并导出（R 4λ→7λ）")
    ap.add_argument("--r", default=R_NEW, help="新域半径表达式（默认 7*lambda）")
    ap.add_argument(
        "--staging", type=Path, required=True, help="暂存目录（不直接写 raw/）"
    )
    ap.add_argument("--cases", default="", help="只跑指定 case（逗号分隔，默认全部）")
    args = ap.parse_args()
    only = {c for c in args.cases.split(",") if c}
    staging = args.staging.resolve()  # COMSOL 按模型目录解析相对路径 → 必须绝对路径
    staging.mkdir(parents=True, exist_ok=True)

    client = mph.start(cores=4)
    model = client.load(str(OUT_MPH))
    jm = model.java
    comp = jm.component("comp1")
    print(f"== 扩域：R {jm.param().get('R')} -> {args.r} ==")
    jm.param().set("R", args.r)
    comp.geom("geom1").run()
    comp.mesh("mesh1").run()
    _diag(jm)

    for cfg, ci, h1e, h3e in CASES:
        if only and cfg not in only:
            continue
        for inc, pl, pr in INCS:
            tag = f"{cfg}_{inc}"
            print(
                f"== case {tag}: ci={ci} h1e={h1e} h3e={h3e} pampL={pl} pampR={pr} =="
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
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] solve {tag}: {type(exc).__name__}: {exc}")
                continue
            dsets = [str(t) for t in (jm.result().dataset().tags() or [])]
            ds = dsets[0] if dsets else "dset1"
            f_new = staging / f"field_{tag}.csv"
            res = export_data(model, ds, f_new, expr="real(acpr.p_t)")
            print(f"  field: {res.ok} {res.out_path}")
            far_new = staging / f"far_{tag}.csv"
            res2 = export_polar_csv(
                model,
                far_new,
                expr="abs(acpr.efc1.pext)",
                radius=R_EVAL,
                center=(X0, 0.0),
                angle_min=0.0,
                angle_max=180.0,
                angle_step=1.0,
                source=ds,
            )
            print(f"  far:   {res2.ok} {res2.out_path}")
            if res.ok and res.out_path is not None:
                print("  场图对照（新 vs 旧，R=4λ 重叠区）:")
                print(_compare_field(RAW_DIR / f"field_{tag}.csv", res.out_path))
            if res2.ok and res2.out_path is not None:
                print("  远场对照（新 vs 旧）:")
                print(_compare_far(RAW_DIR / f"far_{tag}.csv", res2.out_path))

    client.disconnect()
    print("done ->", staging)
    return 0


def promote(staging: Path, names: list[str]) -> None:
    """把核对通过的 staging 产物提升到 raw/（先备份旧件到 raw/_before_r7/）。"""
    backup = RAW_DIR / "_before_r7"
    backup.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = staging / n
        dst = RAW_DIR / n
        if dst.exists():
            shutil.copy2(dst, backup / n)
        shutil.copy2(src, dst)
        print(f"[promote] {n} -> {dst}")


if __name__ == "__main__":
    raise SystemExit(main())
