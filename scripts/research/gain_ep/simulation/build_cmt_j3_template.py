"""构建 3 槽参数化反射模板（基于参考周期模型，逐槽 w/h/x/cr/ci）。"""

from __future__ import annotations

import shutil
from pathlib import Path

import jpype
import mph

ROOT = Path(r"D:\XXXIIIGGG\projects\pySci\pySciWS")
SIM = ROOT / "data" / "research" / "1_gain_ep" / "simulation"
REF = SIM / "refs" / "1 周期化-傅里叶变换法-上下对调.mph"
OUT = SIM / "12 反射管槽_J3模板.mph"


def main() -> None:
    shutil.copy2(REF, OUT)
    client = mph.start(cores=4)
    model = client.load(str(OUT))
    jm = model.java
    comp = jm.component("comp1")
    g = comp.geom("geom1")

    params = {
        "w1": "0.227*D",
        "h1": "0.569*lambda-dstb",
        "x1": "0",
        "w2": "0.115*D",
        "h2": "0.195*lambda",
        "x2": "0.503*D",
        "w3": "0.153*D",
        "h3": "0.232*lambda",
        "x3": "0.688*D",
        "cr1": "1",
        "ci1": "0",
        "cr2": "cr",
        "ci2": "ci",
        "cr3": "1",
        "ci3": "0",
    }
    for k, v in params.items():
        jm.param().set(k, v)

    for tag, w, h, x in (
        ("r1", "w1", "h1", "x1"),
        ("r2", "w2", "h2", "x2"),
        ("r3", "w3", "h3", "x3"),
    ):
        f = g.feature(tag)
        f.set("size", [w, h])
        f.set("pos", [x, f"-{h}"])
    g.feature("pt1").set("p", ["0", "1.5*D"])
    g.feature("pt2").set("p", ["2*D", "1.5*D"])
    g.run()
    print("domains =", g.getNDomains())

    # 逐槽材料：mat2 改 cr2/ci2；槽 1/3 新建材料 + Box 选择（两周期、上下两排）
    mat2 = comp.material("mat2")
    mat2.propertyGroup("def").set("soundspeed", "cs(T)*(cr2+ci2*i)")
    print("mat2 doms =", list(mat2.selection().entities(2) or []))

    def groove_domains(tag: str, w: str, h: str, x: str) -> list[int]:
        ids: list[int] = []
        for k, xoff in enumerate(("0", "D")):
            s = comp.selection().create(f"{tag}_{'ab'[k]}", "Box")
            s.set("entitydim", jpype.JInt(2))
            s.set("condition", "inside")
            s.set("xmin", f"({x})+({xoff})-1[um]")
            s.set("xmax", f"({x})+({xoff})+({w})+1[um]")
            s.set("ymin", f"-5*D-({h})-1[um]")
            s.set("ymax", "1[um]")
            d = list(s.entities(2) or [])
            print(f"  {tag}_{'ab'[k]} doms = {d}")
            ids.extend(int(v) for v in d)
        return sorted(set(ids))

    for tag, w, h, x, cr, ci, mtag in (
        ("g1", "w1", "h1", "x1", "cr1", "ci1", "mat3"),
        ("g3", "w3", "h3", "x3", "cr3", "ci3", "mat4"),
    ):
        ids = groove_domains(tag, w, h, x)
        mat = comp.material(mtag)
        mat.label(f"Groove {tag}")
        mat.propertyGroup("def").set("soundspeed", f"cs(T)*({cr}+{ci}*i)")
        mat.selection().set(ids)
        print(f"  {mtag} doms = {ids}")

    # 扫描线 ParCurve2D（y=1.5D，x 0..2D）
    ds = jm.result().dataset()
    if "scan1" not in [str(t) for t in (ds.tags() or [])]:
        s = ds.create("scan1", "ParCurve2D")
        for pr, val in (
            ("data", "dset1"),
            ("par1", "s"),
            ("parmin1", 0.0),
            ("parmax1", 1.0),
            ("res", jpype.JInt(256)),
            ("exprx", "s*2*D"),
            ("expry", "1.5*D"),
            ("global", False),
        ):
            try:
                s.set(pr, val)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] scan1 {pr}: {type(exc).__name__}: {exc}")
    if "scan2" not in [str(t) for t in (ds.tags() or [])]:
        s2 = ds.create("scan2", "ParCurve2D")
        for pr, val in (
            ("data", "dset1"),
            ("par1", "s"),
            ("parmin1", 0.0),
            ("parmax1", 1.0),
            ("res", jpype.JInt(256)),
            ("exprx", "s*2*D"),
            ("expry", "-3.5*D"),
            ("global", False),
        ):
            try:
                s2.set(pr, val)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] scan2 {pr}: {type(exc).__name__}: {exc}")
    print("dataset tags =", [str(t) for t in (ds.tags() or [])])

    # 物理选择复核
    ph = comp.physics("acpr")
    for tag in ("bpf1", "bpf2", "pc1", "pc2", "pc3", "pc4"):
        f = ph.feature(tag)
        print(
            f"  {tag} bnds/domains = {list(f.selection().entities(1) or [])} / {list(f.selection().entities(2) or [])}"
        )
    pml = comp.coordSystem("pml1")
    print("  pml doms =", list(pml.selection().entities(2) or []))

    model.save(str(OUT))
    print("saved ->", OUT)
    client.disconnect()


if __name__ == "__main__":
    main()
