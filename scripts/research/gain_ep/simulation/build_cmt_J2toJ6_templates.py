"""J6 基模板 + J2/4/5/6 副本（同一 34 域拓扑，槽位 sliver 切换）。"""

from __future__ import annotations

from pathlib import Path

import jpype
import mph

ROOT = Path(r"D:\XXXIIIGGG\projects\pySci\pySciWS")
SIM = ROOT / "data" / "research" / "1_gain_ep" / "simulation"
BASE = SIM / "12 反射管槽_J3模板.mph"
J6 = SIM / "12 反射管槽_J6模板.mph"


def box(comp, tag, x, w, h):
    ids = []
    for k, off in enumerate(("0", "D")):
        s = comp.selection().create(f"{tag}_{'ab'[k]}", "Box")
        s.set("entitydim", jpype.JInt(2))
        s.set("condition", "inside")
        s.set("xmin", f"({x})+({off})-1[um]")
        s.set("xmax", f"({x})+({off})+({w})+1[um]")
        s.set("ymin", f"-5*D-({h})-1[um]")
        s.set("ymax", "1[um]")
        ids += [int(v) for v in (s.entities(2) or [])]
    return sorted(set(ids))


def main() -> None:
    client = mph.start(cores=4)
    model = client.load(str(BASE))
    jm = model.java
    comp = jm.component("comp1")
    g = comp.geom("geom1")
    params = {
        "w4": "0.10*D",
        "h4": "0.20*lambda",
        "x4": "0.25*D",
        "w5": "0.10*D",
        "h5": "0.25*lambda",
        "x5": "0.40*D",
        "w6": "0.10*D",
        "h6": "0.30*lambda",
        "x6": "0.87*D",
        "cr4": "1",
        "ci4": "0",
        "cr5": "1",
        "ci5": "0",
        "cr6": "1",
        "ci6": "0",
    }
    for k, v in params.items():
        jm.param().set(k, v)
    for tag, w, h, x in (
        ("r4", "w4", "h4", "x4"),
        ("r5", "w5", "h5", "x5"),
        ("r6", "w6", "h6", "x6"),
    ):
        f = g.feature(tag)
        f.active(True)
        f.set("size", [w, h])
        f.set("pos", [x, f"-{h}"])
    arr = g.create("arr4", "Array")
    arr.selection("input").set(["r4", "r5", "r6"])
    arr.set("type", "rectangular")
    arr.set("fullsize", ["2", "2"])
    arr.set("displ", ["D", "-5*D"])
    g.run()
    for tag, w, h, x, cr, ci, mat_tag in (
        ("g4", "w4", "h4", "x4", "cr4", "ci4", "mat_g4"),
        ("g5", "w5", "h5", "x5", "cr5", "ci5", "mat_g5"),
        ("g6", "w6", "h6", "x6", "cr6", "ci6", "mat_g6"),
    ):
        ids = box(comp, tag, x, w, h)
        if mat_tag not in [str(t) for t in comp.material().tags()]:
            mat = comp.material().create(mat_tag, "Common")
        else:
            mat = comp.material(mat_tag)
        mat.propertyGroup("def").set("density", "1.2[kg/m^3]")
        mat.propertyGroup("def").set("soundspeed", f"343[m/s]*({cr}+{ci}*i)")
        mat.selection().set(ids)
        print(tag, ids)
    model.save(str(J6))
    print("saved", J6, "domains", g.getNDomains())

    # 副本：按 J 设 sliver（w=h=1um）
    defaults = {**params, "w3": "0.153*D", "h3": "0.232*lambda"}
    for J, sliver_keys in (
        (2, ["w3", "h3", "w4", "h4", "w5", "h5", "w6", "h6"]),
        (4, ["w5", "h5", "w6", "h6"]),
        (5, ["w6", "h6"]),
    ):
        for k, v in defaults.items():
            jm.param().set(k, v)
        for k in sliver_keys:
            jm.param().set(k, "1[um]")
        g.run()
        out = SIM / f"12 反射管槽_J{J}模板.mph"
        model.save(str(out))
        print("saved", out)
    for k, v in defaults.items():
        jm.param().set(k, v)
    g.run()
    client.disconnect()


if __name__ == "__main__":
    main()
