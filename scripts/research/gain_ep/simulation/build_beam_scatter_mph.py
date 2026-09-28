"""构建专用 mph：半圆域 8 周期管槽超表面 + 有限波束 ±45° 入射散射（6 case 参数化）。

用途（绘图需求）：3 组对照 × 左右入射 = 6 个半圆仿真场图 + 6 个极坐标散射能量图。
对照开关（全部为全局参数，CLI `run solve --set-param` 或 runner 循环切换）：
  - ci   : 管槽 2 增益等效介质虚部（ci<0 增益；=0 → 无增益介质，对照②）
  - h1e / h3e : 管槽 1/3 的有效深度（=h1/h3 → 结构存在；→ lambda/200 薄片 → 等效整根移除，对照③）
  - pampL / pampR : 左/右 45° 入射背景场幅值开关（继承参考模型约定，勿切 active）

建模要点（依据参考模型 refs/2 8周期有限波束入射.mph 的深 dump + PRM/Acoustics 手册核对）：
  - 半圆空气域（半径 R，直径边 = 超表面平面 y=0），8 周期 × 3 管槽凹入 y<0；
    半圆用 整圆 c1 减去下半矩形 rbot（Difference）得到，避免扇形角度朝向歧义；
  - Array：selection("input") + type=rectangular + fullsize=[Nper,1] + displ=[D,0]；
  - 弧边：Plane Wave Radiation（散射场吸收）+ Exterior Field Calculation（远场，
    y=0 无限硬墙对称面）；其余边界默认硬墙（= 刚性板 + 管槽壁）；
  - 管槽 2 域材料 mat2：复声速 c*(cr+ci*i)（ci 符号约定与 CMT 相同：ci<0 增益）；
    域选择用 CumulativeSelection（arr2 contributeto）→ 跨 g13 配置稳定；
  - 结果：pg_field（总场 Surface 图）/ pg_far（极坐标 RadiationPattern 远场图）。
"""

from __future__ import annotations

from pathlib import Path

import jpype
import mph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIM_DIR = ROOT / "data" / "research" / "1_gain_ep" / "simulation"
OUT_MPH = SIM_DIR / "6 半圆波束散射对照.mph"
FIG_DIR = ROOT / "data" / "research" / "1_gain_ep" / "article" / "figures" / "_beam_scatter_proto"

PARAMS = {
    "c0": "343[m/s]",
    "f": "3430[1/s]",
    "lambda": "c0/f",
    "k": "2*pi/lambda",
    "theta": "45*pi/180",
    "D": "lambda/sqrt(2)",
    "Nper": "8",
    "x0": "Nper*D/2",
    # 管槽几何（单周期内偏移/宽/深，与 CMT make_ep_params / 参考模型一致）
    "w1": "0.227*D", "h1": "0.569*lambda",
    "w2": "0.115*D", "h2": "0.195*lambda", "o2": "0.503*D",
    "w3": "0.153*D", "h3": "0.232*lambda", "o3": "0.688*D",
    # 管槽 1/3 有效深度（对照③置为薄片 = 整根移除）
    "h1e": "h1", "h3e": "h3",
    # 物理开关
    "cr": "1.0037965",
    "ci": "-0.0730324",
    "R": "4*lambda",
    "pampL": "1",
    "pampR": "0",
}


def _set(node, props: dict) -> None:
    """逐个 set，单点失败打印警告但不中断（属性名差异可事后排查）。"""
    if node is None:
        return
    for key, val in props.items():
        try:
            node.set(key, val)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] set {key}={val!r} failed: {type(exc).__name__}: {exc}")


def _create(seq, tag: str, ntype: str, *extra):
    """在特征序列上 create，失败返回 None 并打印（类型名/上下文错误可诊断）。"""
    try:
        return seq.feature().create(tag, ntype, *extra) if hasattr(seq, "feature") else seq.create(tag, ntype, *extra)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] create {tag}::{ntype} failed: {type(exc).__name__}: {exc}")
        return None


def _safe(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return default


def build(*, solve: bool = True, export: bool = True) -> None:
    client = mph.start(cores=4)
    model = client.create("beamsc")
    jm = model.java

    print("== parameters ==")
    for name, val in PARAMS.items():
        jm.param().set(name, val)

    print("== component / geometry ==")
    jm.component().create("comp1")
    comp = jm.component("comp1")
    comp.geom().create("geom1", 2)
    g = comp.geom("geom1")

    # 半圆域：整圆 c1 减去下半矩形 rbot（差集）→ 上半圆盘，直径边落在 y=0
    c1 = _create(g, "c1", "Circle")
    _set(c1, {"r": "R", "pos": ["x0", "0"], "base": "center"})
    rbot = _create(g, "rbot", "Rectangle")
    _set(rbot, {"size": ["2*R+2[mm]", "R+1[mm]"], "pos": ["x0-R-1[mm]", "-R-1[mm]"], "base": "corner"})
    dif1 = _create(g, "dif1", "Difference")
    if dif1 is not None:
        try:
            dif1.selection("input").set(["c1"])
            dif1.selection("input2").set(["rbot"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] dif1 selection: {type(exc).__name__}: {exc}")

    # 单周期的 3 根管槽矩形（凹入 y<0）；h1e/h3e 置薄片时管槽 1/3 等效整根移除（对照③）
    r1 = _create(g, "r1", "Rectangle")
    _set(r1, {"size": ["w1", "h1e"], "pos": ["0", "-h1e"], "base": "corner"})
    r2 = _create(g, "r2", "Rectangle")
    _set(r2, {"size": ["w2", "h2"], "pos": ["o2", "-h2"], "base": "corner"})
    r3 = _create(g, "r3", "Rectangle")
    _set(r3, {"size": ["w3", "h3e"], "pos": ["o3", "-h3e"], "base": "corner"})

    # 8 周期阵列（x 方向，间距 D）；arr2 开 selresult → comp 级域选择 geom1_arr2_dom（管槽2增益域）
    for tag, src in (("arr1", "r1"), ("arr2", "r2"), ("arr3", "r3")):
        arr = _create(g, tag, "Array")
        if arr is None:
            continue
        try:
            arr.selection("input").set([src])
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {tag} input: {type(exc).__name__}: {exc}")
        _set(arr, {"type": "rectangular", "fullsize": ["Nper", "1"], "displ": ["D", "0"]})
        if tag == "arr2":
            _set(arr, {"selresult": "on", "selresultshow": "dom"})

    # fin(FormUnion) 为几何序列默认终端节点，无需 create；直接 run
    try:
        g.run()
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] geom run failed: {type(exc).__name__}: {exc}")

    # 结构自检：域数 + 命名选择
    ndom = _safe(g.getNDomains, default="?")
    print(f"  geom domains = {ndom}")
    for lvl, seq in (("comp", comp.selection()), ("geom", _safe(comp.geom("geom1").selection, default=None))):
        tags = [str(t) for t in (_safe(seq.tags, default=[]) or [])] if seq is not None else []
        print(f"  {lvl} selections = {tags}")
        for t in tags:
            sn = _safe(seq.get, t, default=None)
            doms = _safe(lambda: list(sn.domains()), default="?") if sn is not None else "?"
            lbl = _safe(sn.label, default="") if sn is not None else ""
            print(f"      {t} [{lbl}] domains={doms}")

    # 动态发现管槽2增益域的 comp 级选择 tag（arr2 selresult 自动创建，形如 geom1_arr2_dom）
    _comp_sel_tags = [str(t) for t in (_safe(comp.selection().tags, default=[]) or [])]
    g2_tag = next((t for t in _comp_sel_tags if "arr2_dom" in t), None)
    print(f"  groove2 domain selection tag = {g2_tag}")

    # 弧边选择（y>0 的边界）：辐射条件 + 远场积分
    sel_arc = _safe(comp.selection().create, "sel_arc", "Box", default=None)
    _set(sel_arc, {
        "entitydim": jpype.JInt(1),
        "condition": "intersects",
        "xmin": "x0-R-1[mm]", "xmax": "x0+R+1[mm]",
        "ymin": "1[um]", "ymax": "R+1[mm]",
    })
    arc_bnds = _safe(lambda: list(sel_arc.boundaries()), default="?") if sel_arc is not None else "?"
    print(f"  sel_arc boundaries = {arc_bnds}")

    print("== materials ==")
    mat1 = _safe(comp.material().create, "mat1", "Common", default=None)
    if mat1 is not None:
        mat1.label("Air")
        _set(mat1.propertyGroup("def"), {"density": "1.2[kg/m^3]", "soundspeed": "c0"})
        _safe(mat1.selection().all)
    mat2 = _safe(comp.material().create, "mat2", "Common", default=None)
    if mat2 is not None:
        mat2.label("GainAir groove2")
        _set(mat2.propertyGroup("def"), {"density": "1.2[kg/m^3]", "soundspeed": "c0*(cr+ci*i)"})
        try:
            mat2.selection().named(g2_tag)
            print(f"  mat2 -> named({g2_tag}) ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] mat2 named({g2_tag}): {type(exc).__name__}: {exc}")
        print(f"  mat2 domains = {_safe(lambda: list(mat2.selection().domains()), default='?')}")

    print("== physics ==")
    try:
        comp.physics().create("acpr", "PressureAcoustics", "geom1")
        print("  physics acpr created")
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] physics create failed: {type(exc).__name__}: {exc}")
    ph = comp.physics("acpr")

    bpf1 = _create(ph, "bpf1", "BackgroundPressureField")
    if bpf1 is not None:
        bpf1.label("Background L (+45)")
        _set(bpf1, {"PressureFieldType": "PlaneWave", "pamp": "pampL", "phi": "0",
                    "dir": ["sin(theta)", "-cos(theta)", "0"], "c_mat": "from_mat"})
        _safe(bpf1.selection().all)
    bpf2 = _create(ph, "bpf2", "BackgroundPressureField")
    if bpf2 is not None:
        bpf2.label("Background R (-45)")
        _set(bpf2, {"PressureFieldType": "PlaneWave", "pamp": "pampR", "phi": "0",
                    "dir": ["-sin(theta)", "-cos(theta)", "0"], "c_mat": "from_mat"})
        _safe(bpf2.selection().all)

    pwrad = _create(ph, "pwrad1", "PlaneWaveRadiation")
    if pwrad is not None:
        _safe(pwrad.selection().named, "sel_arc")
        print(f"  pwrad1 sel bnds = {len(_safe(lambda: list(pwrad.selection().entities(1)), default=[]) or [])}")
        print(f"  pwrad1 props = {_safe(lambda: list(pwrad.properties()), default='?')}")

    efc = _create(ph, "efc1", "ExteriorFieldCalculation")
    if efc is not None:
        _safe(efc.selection().named, "sel_arc")
        print(f"  efc1 sel bnds = {len(_safe(lambda: list(efc.selection().entities(1)), default=[]) or [])}")
        props = _safe(lambda: list(efc.properties()), default=[]) or []
        print(f"  efc1 props = {props}")
        # 发现对称面相关属性名（供 y=0 无限硬墙设置）
        for p in props:
            lp = str(p).lower()
            if any(kw in lp for kw in ("sym", "plane", "baffle", "integral", "pext", "name")):
                allowed = _safe(efc.getAllowedPropertyValues, p, default=None)
                cur = _safe(efc.getString, p, default=None)
                print(f"      efc.{p} = {cur!r}  allowed={list(allowed) if allowed else allowed}")

    print("== mesh ==")
    _safe(comp.mesh().create, "mesh1")
    m = comp.mesh("mesh1")
    size = _safe(m.feature, "size", default=None) or _create(m, "size1", "Size")
    _set(size, {"custom": True, "hmax": "lambda/25", "hmin": "lambda/200"})
    _create(m, "ftri1", "FreeTri")
    try:
        m.run()
        print(f"  mesh vertices = {_safe(lambda: m.getNVertices(), default='?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] mesh run failed: {type(exc).__name__}: {exc}")

    print("== study ==")
    _safe(jm.study().create, "std1")
    std = jm.study("std1")
    _safe(std.create, "freq", "Frequency")
    freq = _safe(std.feature, "freq", default=None)
    if freq is not None:
        for cand in ("plist", "flist"):
            try:
                freq.set(cand, ["f"])
                _safe(freq.set, "punit", ["Hz"])
                print(f"  freq set via {cand}=['f'] punit=['Hz']")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    [warn] freq set {cand}: {type(exc).__name__}")

    print("== results ==")
    pgf = _safe(jm.result().create, "pg_field", "PlotGroup2D", default=None)
    surf = _create(pgf, "surf1", "Surface") if pgf is not None else None
    _set(surf, {"expr": "acpr.p_t"})
    pgp = _safe(jm.result().create, "pg_far", "PolarGroup", default=None)
    if pgp is None:
        for alt in ("PolarPlotGroup", "PlotGroupPolar"):
            pgp = _safe(jm.result().create, "pg_far", alt, default=None)
            if pgp is not None:
                print(f"  pg_far created via type {alt}")
                break
    print(f"  result tags = {[str(t) for t in (_safe(jm.result().tags, default=[]) or [])]}")
    rad = _create(pgp, "rad1", "RadiationPattern") if pgp is not None else None
    _set(rad, {"expr": "abs(acpr.efc1.pext)", "refdir": ["1", "0"],
               "anglerestr": "on", "phimin": "0", "phirange": "180"})
    if rad is not None:
        print(f"  rad1 props = {_safe(lambda: list(rad.properties()), default='?')}")

    print("== save ==")
    OUT_MPH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_MPH))
    print(f"  saved -> {OUT_MPH}")

    if solve:
        print("== solve (default: L-incidence, gain EP, structure present) ==")
        try:
            jm.study("std1").run()
            print("  solve ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] solve failed: {type(exc).__name__}: {exc}")
        _diag_field_stats(jm, "dset1")

    if export and solve:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        dsets = [str(t) for t in (_safe(jm.result().dataset().tags, default=[]) or [])]
        print(f"  datasets = {dsets}")
        ds = dsets[0] if dsets else None
        if ds:
            for pg in ("pg_field", "pg_far"):
                node = _safe(jm.result, pg, default=None)
                if node is not None:
                    _safe(node.set, "data", ds)
        _export_png(jm, "pg_field", FIG_DIR / "verify_field_L.png")
        _export_png(jm, "pg_far", FIG_DIR / "verify_far_L.png")

    client.disconnect()


def _diag_field_stats(jm, ds: str) -> None:
    """导出 p_t/p/p_b 到临时 CSV 并打印 min/max，诊断哪个场非零。"""
    import numpy as np
    tmp = Path(__import__("tempfile").gettempdir()) / "beamsc_diag.csv"
    try:
        explist = jm.result().export()
        try:
            ex = explist.create("diag1", "Data")
        except Exception:  # noqa: BLE001
            ex = _safe(explist.feature, "diag1", default=None)
        if ex is None:
            print("  [diag] no Data export node")
            return
        ex.set("data", ds)
        ex.set("filename", str(tmp))
        for expr in ("abs(acpr.p_t)", "abs(acpr.p_b)"):
            try:
                ex.set("expr", [expr])
                ex.run()
                lines = [ln for ln in tmp.read_text().splitlines() if ln and not ln.startswith("%")]
                vals = [float(r.split(",")[-1]) for r in lines[1:]]
                import numpy as np
                a = np.array(vals)
                print(f"  [diag] {expr}: min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")
            except Exception as exc2:  # noqa: BLE001
                print(f"  [diag] {expr}: FAILED {type(exc2).__name__}: {exc2}")
        # BPF 实际幅值/激活状态
        try:
            b1 = jm.component("comp1").physics("acpr").feature("bpf1")
            print(f"  [diag] bpf1 pamp={b1.getString('pamp')} active={b1.isActive()} type={b1.getString('PressureFieldType')}")
            print(f"  [diag] bpf1 dir={list(b1.getStringArray('dir'))} phi={b1.getString('phi')} k_src={b1.getString('k_src')}")
            for pn in ("pampL", "pampR", "theta", "k", "lambda", "c0"):
                print(f"  [diag] param {pn} = {jm.param().get(pn)}")
        except Exception as exc3:  # noqa: BLE001
            print(f"  [diag] bpf1 read failed: {type(exc3).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [diag] failed: {type(exc).__name__}: {exc}")


def _export_png(jm, pgtag: str, out: Path) -> None:
    """用 Image 导出特征把某绘图组渲染成 PNG（COMSOL 原生渲染）。"""
    try:
        explist = jm.result().export()
        tag = f"img_{out.stem}"  # 每个输出文件唯一 tag，避免重复 create 冲突
        try:
            e = explist.create(tag, "Image")
        except Exception:  # noqa: BLE001
            e = None
        if e is None:
            print(f"  [warn] export {pgtag}: no export node")
            return
        e.set("plotgroup", pgtag)
        e.set("pngfilename", str(out))
        _set(e, {"size": "manualweb", "unit": "px", "height": "900", "width": "900", "resolution": "96"})
        _safe(jm.result(pgtag).run)
        e.run()
        print(f"  exported {pgtag} -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] export {pgtag}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import sys
    if "--nosolve" in sys.argv:
        build(solve=False, export=False)
    else:
        build()
