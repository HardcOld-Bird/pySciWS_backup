"""只读深 dump：把 .mph 各序列的子特征（tag/类型/标签/关键属性）打印出来。

技能 CLI 的 `inspect tree` 只到序列级；本脚本用 mph+Java 反射下钻一层，
供理解参考模型内部构造（几何特征/背景场/PML/远场/绘图组配置）时使用。
用法：uv run python dump_mph_structure.py <path.mph>
"""

from __future__ import annotations

import sys
from pathlib import Path

import mph


def _safe(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return default


def _feat_lines(seq, indent="    ") -> list[str]:
    """dump 一个特征序列的子节点：tag :: type [label] + 关键属性。"""
    fseq = _safe(seq.feature, default=None)
    if fseq is None:
        return []
    tags = [str(t) for t in (_safe(fseq.tags, default=[]) or [])]
    get_type = getattr(seq, "getType", None)
    lines = []
    for t in tags:
        typ = str(_safe(get_type, t, default="") or "") if callable(get_type) else ""
        node = _safe(fseq.get, t, default=None)
        label = str(_safe(node.label, default="") or "") if node is not None else ""
        head = f"{indent}- {t}  ::  {typ}" + (
            f" [{label}]" if label and label != t else ""
        )
        lines.append(head)
        if node is None:
            continue
        for prop in (
            "size",
            "pos",
            "r",
            "base",
            "rot",
            "expr",
            "pamp",
            "kx",
            "ky",
            "c",
            "rho",
            "hmax",
            "hmin",
            "selection",
            "domain",
            "boundary",
            "k",
            "alpha",
            "phi",
            "pb",
            "type",
            "k1",
            "k2",
            "k0",
            "wavevector",
            "kvec",
            "backgroundtype",
            "dx",
            "dy",
            "x",
            "y",
            # BPF plane-wave direction candidates
            "e_k",
            "ek",
            "ekx",
            "eky",
            "ekz",
            "kz",
            "beta",
            "gamma",
            "angle",
            "q0",
            "p0",
            "T0",
            "level",
            "outofplane",
            # Array / finalize candidates
            "selresult",
            "selresultshow",
            "linearsize",
            "fullsize",
            "displ",
            "input",
            "propagatesel",
        ):
            val = _safe(node.getString, prop, default=None)
            if val not in (None, ""):
                lines.append(f"{indent}    {prop} = {val}")
        # 向量属性（getString 返回空）：用 getStringArray 再探一次
        for prop in (
            "k",
            "ek",
            "e_k",
            "kx",
            "ky",
            "kz",
            "direction",
            "wavevec",
            "q",
            "pos",
            "displ",
            "size",
            "fullsize",
        ):
            arr = _safe(node.getStringArray, prop, default=None)
            if arr is not None and len(list(arr)) > 0:
                lines.append(f"{indent}    {prop}[] = {list(arr)}")
    return lines


def _resolve(arg: str) -> str:
    """把命令行参数解析成真实存在的 .mph 路径。

    PowerShell 向本地进程传参时会在空格/中文处截断，因此这里支持传入
    一个无空格片段（如 ``refs2``），在 simulation 目录下递归 glob 匹配。
    """
    p = Path(arg)
    if p.exists():
        return str(p)
    root = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "research"
        / "1_gain_ep"
        / "simulation"
    )
    # 支持 'refs2' -> 子目录 refs 下以 '2' 开头的 mph；也支持直接文件名前缀
    frag = arg.replace("\\", "/")
    candidates: list[Path] = []
    if frag.startswith("refs"):
        candidates = sorted((root / "refs").glob(frag[4:] + "*.mph"))
    if not candidates:
        candidates = sorted(root.rglob("*" + frag.split("/")[-1] + "*.mph"))
    if not candidates:
        raise SystemExit(f"cannot resolve mph for fragment {arg!r} under {root}")
    return str(candidates[0])


def _deep_props(node, indent="      ") -> list[str]:
    """对一个物理节点做深度自省：方法列表 + properties() + 向量取值。"""
    lines: list[str] = []
    # 1) 方法名（找属性枚举器 / 向量取值器）
    cls = _safe(node.getClass, default=None)
    if cls is not None:
        methods = _safe(cls.getMethods, default=[]) or []
        names = sorted({str(m).split("(")[0].split()[-1] for m in methods})
        interesting = [
            n
            for n in names
            if any(
                kw in n.lower()
                for kw in (
                    "propert",
                    "vector",
                    "array",
                    "getstring",
                    "getdouble",
                    "names",
                )
            )
        ]
        lines.append(f"{indent}methods~ {interesting}")
    # 2) properties() / getPropertyNames()（仅在需要时打开，输出较长）
    # for mname in ("properties", "getPropertyNames"):
    #     fn = getattr(node, mname, None)
    #     if callable(fn):
    #         val = _safe(fn, default=None)
    #         if val is not None:
    #             lines.append(f"{indent}{mname}() = {list(val)}")
    # 3) 候选方向属性的多种取值方式
    cand = (
        "dir",
        "PressureFieldType",
        "pamp",
        "phi",
        "k_src",
        "ek",
        "e_k",
        "ekx",
        "eky",
        "kx",
        "ky",
        "k",
        "direction",
        "wavevec",
        "backgroundtype",
        "type",
        "cs",
        "alpha",
        "beta",
        "theta",
        "q",
        "n",
    )
    for prop in cand:
        for acc in (
            "getString",
            "getVector",
            "getDoubleArray",
            "getStringArray",
            "get",
        ):
            fn = getattr(node, acc, None)
            if not callable(fn):
                continue
            val = _safe(fn, prop, default=None)
            if val is None:
                continue
            sval = (
                str(list(val))
                if hasattr(val, "__len__") and not isinstance(val, str)
                else str(val)
            )
            if sval not in ("", "0", "0.0", "[]", "None"):
                lines.append(f"{indent}{acc}({prop!r}) = {sval}")
    return lines


def main(mph_path: str) -> None:
    mph_path = _resolve(mph_path)
    print(f"== loading {mph_path} ==")
    client = mph.start(cores=4)
    model = client.load(mph_path)
    jm = model.java

    print("== parameters ==")
    raw = model.parameters
    params = raw() if callable(raw) else raw
    for k, v in (params or {}).items():
        print(f"  {k} = {v}")

    comp = jm.component("comp1")

    print("== geom1 features ==")
    print("\n".join(_feat_lines(comp.geom("geom1"))) or "  (none)")

    print("== component named selections ==")
    sseq = _safe(comp.selection, default=None)
    for stag in [str(t) for t in (_safe(sseq.tags, default=[]) or [])] if sseq else []:
        sn = _safe(sseq.get, stag, default=None)
        gt = getattr(sseq, "getType", None)
        styp = str(_safe(gt, stag, default="") or "") if callable(gt) else ""
        lbl = str(_safe(sn.label, default="") or "") if sn is not None else ""
        doms = _safe(lambda: sn.domains, default="?") if sn is not None else "?"
        print(f"  {stag} :: {styp} [{lbl}] domains={doms}")

    print("== physics features ==")
    pseq = _safe(comp.physics, default=None)
    for ptag in [str(t) for t in (_safe(pseq.tags, default=[]) or [])]:
        print(f"  physics {ptag}:")
        print("\n".join(_feat_lines(comp.physics(ptag), indent="    ")) or "    (none)")
        # 对 bpf 节点做深度自省
        fseq = _safe(comp.physics(ptag).feature, default=None)
        for ft in (
            [str(t) for t in (_safe(fseq.tags, default=[]) or [])] if fseq else []
        ):
            if ft.startswith("bpf"):
                node = _safe(fseq.get, ft, default=None)
                if node is not None:
                    print(f"    [deep {ft}]")
                    print("\n".join(_deep_props(node)) or "      (nothing)")

    print("== materials ==")
    mseq = _safe(comp.material, default=None)
    for mtag in [str(t) for t in (_safe(mseq.tags, default=[]) or [])]:
        mat = comp.material(mtag)
        sel = _safe(mat.selection, default=None)
        doms = _safe(lambda: sel.domains, default="?") if sel is not None else "?"
        print(f"  {mtag} [{_safe(mat.label, default='')}] domains={doms}")
        if sel is not None:
            for m in ("named", "getNamed", "getType", "getSelectionType"):
                fn = getattr(sel, m, None)
                if callable(fn):
                    print(f"      sel.{m}() = {_safe(fn, default='?')}")
            gs = getattr(sel, "getString", None)
            if callable(gs):
                print(
                    f"      sel.getString('named') = {_safe(gs, 'named', default='?')}"
                )
            print(f"      sel.toString() = {_safe(sel.toString, default='?')}")
        pg = _safe(mat.propertyGroup, "def", default=None)
        if pg is not None:
            for vn in [str(x) for x in (_safe(lambda: pg.varnames, default=[]) or [])]:
                print(f"      {vn} = {_safe(pg.get, vn, default='?')}")

    print("== mesh1 features ==")
    print("\n".join(_feat_lines(comp.mesh("mesh1"))) or "  (none)")

    print("== study/sol ==")
    sseq = _safe(jm.study, default=None)
    for stag in [str(t) for t in (_safe(sseq.tags, default=[]) or [])]:
        std = jm.study(stag)
        fseq = _safe(std.feature, default=None)
        ftags = (
            [str(t) for t in (_safe(fseq.tags, default=[]) or [])]
            if fseq is not None
            else []
        )
        print(f"  study {stag} [{_safe(std.label, default='')}] features={ftags}")

    print("== results ==")
    rseq = _safe(jm.result, default=None)
    for rtag in [str(t) for t in (_safe(rseq.tags, default=[]) or [])]:
        res = jm.result(rtag)
        gtype = getattr(res, "getType", None)
        rtyp = str(_safe(gtype, default="") or "") if callable(gtype) else ""
        print(f"  result {rtag} [{_safe(res.label, default='')}] :: {rtyp}")
        print("\n".join(_feat_lines(res, indent="    ")) or "    (none)")

    client.disconnect()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
