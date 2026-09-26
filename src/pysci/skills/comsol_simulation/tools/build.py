"""建模 recipe 引擎：几何/材料/物理场/网格/研究原语 + 外部几何导入。

设计取向（对齐计划）：
- **recipe = 纯函数** ``build(model, params) -> None`` + 元数据，用 :func:`recipe` 装饰器注册；
  :func:`run_recipe` 按名调用。迭代 = 改 ``params`` + 重跑（参数写入 COMSOL parameters 节点）。
- **原语**是 COMSOL Java API 的薄封装（防御式：create→set→run，单点失败降级不中断）。
  几何/网格/物理场用 ``<seq>.feature().create(tag, type)``，study/material 用 ``<seq>.create``。
- **外部几何/网格导入** :func:`import_external_geomesh`（STL/PLY/VTK/STEP），为未来 Blender 铺路。
- 从零建模建议以 ``templates/`` 的种子 .mph 为起点（recipe 在其上叠加），避免纯 API 空手起模型。

所有原语接受 mph.Model 或裸 Java model（经 :func:`_jmodel` 归一）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inspect import _call_if, _jmodel, _safe

# ---------------------------------------------------------------------------
# 通用：create + set + (run)
# ---------------------------------------------------------------------------


def _create_node(seq: Any, tag: str, ntype: str) -> Any:
    """在序列上创建节点：优先 ``seq.feature().create``，回退 ``seq.create``。"""
    feat_seq = _call_if(seq, "feature", default=None)
    if feat_seq is not None:
        node = _safe(feat_seq.create, tag, ntype, default=None)
        if node is not None:
            return node
    return _safe(seq.create, tag, ntype, default=None)


def _set_props(node: Any, props: Mapping[str, Any] | None) -> None:
    """逐个 set 属性（单点失败跳过，不中断整体建模）。"""
    if not props or node is None:
        return
    for k, v in props.items():
        _safe(node.set, k, v)


def _comp(model: Any, component: str) -> Any:
    jm = _jmodel(model)
    return _safe(jm.component, component, default=None) if component else jm


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def set_parameters(model: Any, mapping: Mapping[str, str]) -> None:
    """把 ``name -> "value[unit]"`` 写入全局 parameters 节点（参数化迭代的基础）。"""
    jm = _jmodel(model)
    p = _call_if(jm, "param", default=None)
    if p is None:
        return
    for name, value in mapping.items():
        _safe(p.set, name, str(value))


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
def geom_seq(model: Any, component: str = "comp1", geom: str = "geom1") -> Any:
    return _safe(_call_if(_comp(model, component), "geom", default=None), geom, default=None)


def geom_feature(
    model: Any,
    tag: str,
    ftype: str,
    props: Mapping[str, Any] | None = None,
    *,
    component: str = "comp1",
    geom: str = "geom1",
    run: bool = False,
) -> Any:
    """通用几何特征原语（Block/Rectangle/Circle/Polygon/Array/...）。"""
    g = geom_seq(model, component, geom)
    node = _create_node(g, tag, ftype)
    _set_props(node, props)
    if run:
        _safe(g.run)
    return node


def add_block(model: Any, tag: str, size: tuple[float, ...], pos: tuple[float, ...] = (), **kw: Any) -> Any:
    props: dict[str, Any] = {"size": list(size)}
    if pos:
        props["pos"] = list(pos)
    props.update(kw)
    return geom_feature(model, tag, "Block", props)


def add_rectangle(model: Any, tag: str, size: tuple[float, float], pos: tuple[float, float] = (0, 0), **kw: Any) -> Any:
    props: dict[str, Any] = {"size": list(size), "pos": list(pos)}
    props.update(kw)
    return geom_feature(model, tag, "Rectangle", props)


def add_circle(model: Any, tag: str, r: float, pos: tuple[float, float] = (0, 0), **kw: Any) -> Any:
    props: dict[str, Any] = {"r": r, "pos": list(pos)}
    props.update(kw)
    return geom_feature(model, tag, "Circle", props)


def add_polygon(model: Any, tag: str, coords: list[tuple[float, float]], **kw: Any) -> Any:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    props: dict[str, Any] = {"source": "table", "tablex": xs, "tabley": ys}
    props.update(kw)
    return geom_feature(model, tag, "Polygon", props)


def add_array(model: Any, tag: str, input_tags: list[str], counts: list[int], displ: list[list[float]], **kw: Any) -> Any:
    props: dict[str, Any] = {"input": input_tags, "xsize": counts[0] if counts else 1}
    if len(counts) > 1:
        props["ysize"] = counts[1]
    if displ:
        props["xdispl"] = displ[0]
        if len(displ) > 1:
            props["ydispl"] = displ[1]
    props.update(kw)
    return geom_feature(model, tag, "Array", props)


def finalize_geom(model: Any, component: str = "comp1", geom: str = "geom1") -> None:
    """运行几何序列（build selected / form union）。"""
    g = geom_seq(model, component, geom)
    _safe(g.run)


# ---------------------------------------------------------------------------
# 材料
# ---------------------------------------------------------------------------
def add_material(
    model: Any,
    tag: str,
    props: Mapping[str, Any] | None = None,
    *,
    component: str | None = "comp1",
) -> Any:
    """添加材料（component=None → 全局材料）。props 如 {"density":"1.2","sound_speed":"343"}。"""
    if component:
        seq = _call_if(_comp(model, component), "material", default=None)
    else:
        seq = _call_if(_jmodel(model), "material", default=None)
    node = _create_node(seq, tag, "Common")
    if node is not None and props:
        pset = _call_if(node, "propertyGroup", default=None)
        target = pset if pset is not None else node
        _set_props(target, props)
    return node


# ---------------------------------------------------------------------------
# 物理场
# ---------------------------------------------------------------------------
def add_physics(
    model: Any,
    tag: str,
    ptype: str,
    geom: str = "geom1",
    props: Mapping[str, Any] | None = None,
    *,
    component: str = "comp1",
) -> Any:
    """添加物理场接口（如 ``("acpr","PressureAcousticsFrequency","geom1")``）。"""
    seq = _call_if(_comp(model, component), "physics", default=None)
    node = _safe(seq.create, tag, ptype, geom, default=None) or _create_node(seq, tag, ptype)
    _set_props(node, props)
    return node


def add_physics_feature(
    model: Any,
    physics: str,
    tag: str,
    ftype: str,
    props: Mapping[str, Any] | None = None,
    *,
    component: str = "comp1",
    selection: Mapping[str, list[int]] | None = None,
) -> Any:
    """在物理场下加边界/域条件特征（如 PML、BackgroundPressureField、NormalDisplacement）。"""
    ph = _safe(_call_if(_comp(model, component), "physics", default=None), physics, default=None)
    node = _create_node(ph, tag, ftype)
    _set_props(node, props)
    if node is not None and selection:
        for sel_kind, idx in selection.items():
            _safe(node.selection().set, sel_kind, list(idx))
    return node


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------
def mesh_seq(model: Any, component: str = "comp1", mesh: str = "mesh1") -> Any:
    return _safe(_call_if(_comp(model, component), "mesh", default=None), mesh, default=None)


def add_mesh(model: Any, tag: str = "mesh1", *, component: str = "comp1") -> Any:
    seq = _call_if(_comp(model, component), "mesh", default=None)
    return _safe(seq.create, tag, default=None) or _create_node(seq, tag, "Mesh")


def mesh_feature(
    model: Any,
    tag: str,
    ftype: str,
    props: Mapping[str, Any] | None = None,
    *,
    component: str = "comp1",
    mesh: str = "mesh1",
) -> Any:
    """网格特征原语（Size/FreeTri/FreeTet/Edge/...）。"""
    m = mesh_seq(model, component, mesh)
    node = _create_node(m, tag, ftype)
    _set_props(node, props)
    return node


def set_mesh_size(model: Any, hmax: float, hmin: float | None = None, *, component: str = "comp1", mesh: str = "mesh1", tag: str = "size") -> Any:
    props: dict[str, Any] = {"hmax": hmax, "custom": True}
    if hmin is not None:
        props["hmin"] = hmin
    return mesh_feature(model, tag, "Size", props, component=component, mesh=mesh)


def run_mesh(model: Any, component: str = "comp1", mesh: str = "mesh1") -> None:
    _safe(mesh_seq(model, component, mesh).run)


# ---------------------------------------------------------------------------
# 研究
# ---------------------------------------------------------------------------
def add_study(model: Any, tag: str = "std1", steps: list[tuple[str, str]] | None = None) -> Any:
    """创建研究；steps = [(stepTag, stepType), ...] 如 [("freq","Frequency")]。"""
    jm = _jmodel(model)
    study = _safe(jm.study().create, tag, default=None)
    for stag, stype in steps or []:
        _safe(study.create, stag, stype)
    return study


def add_parametric(model: Any, study: str, param: str, values: list[str], tag: str = "param") -> Any:
    """给研究加参数扫描步：``param`` 取 ``values``（COMSOL 表达式字符串列表）。"""
    jm = _jmodel(model)
    std = _safe(jm.study, study, default=None)
    node = _safe(std.create, tag, "Parametric", default=None)
    if node is not None:
        _safe(node.set, "pname", [param])
        _safe(node.set, "plistarr", [list(values)])
    return node


# ---------------------------------------------------------------------------
# 外部几何/网格导入（Blender 接口预留）
# ---------------------------------------------------------------------------
_IMPORT_TYPE_BY_SUFFIX = {
    ".stl": "stl",
    ".ply": "ply",
    ".vtk": "vtk",
    ".vtu": "vtk",
    ".step": "step",
    ".stp": "step",
    ".iges": "iges",
    ".igs": "iges",
}


def import_external_geomesh(
    model: Any,
    path: str | Path,
    *,
    tag: str = "imp1",
    component: str = "comp1",
    geom: str = "geom1",
    fmt: str | None = None,
) -> Any:
    """把外部几何/网格（STL/PLY/VTK/STEP/IGES）作为 Import 特征挂到几何序列。

    这是未来 Blender 建模能力的接入点：Blender 导出 STL/PLY → 本函数导入 COMSOL。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"外部几何/网格文件不存在：{p}")
    fmt = (fmt or _IMPORT_TYPE_BY_SUFFIX.get(p.suffix.lower(), "")).lower()
    g = geom_seq(model, component, geom)
    node = _create_node(g, tag, "Import")
    if node is None:
        return None
    _safe(node.set, "filename", str(p))
    if fmt:
        _safe(node.set, "format", fmt)
    return node


# ---------------------------------------------------------------------------
# Recipe 注册表
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    name: str
    fn: Callable[[Any, Mapping[str, Any]], None]
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # 默认参数
    tags: tuple[str, ...] = ()


_REGISTRY: dict[str, Recipe] = {}


def recipe(
    name: str,
    *,
    description: str = "",
    params: Mapping[str, Any] | None = None,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable[[Any, Mapping[str, Any]], None]], Callable[[Any, Mapping[str, Any]], None]]:
    """装饰器：把 ``build(model, params)`` 纯函数注册为命名 recipe。"""

    def deco(fn: Callable[[Any, Mapping[str, Any]], None]):
        _REGISTRY[name] = Recipe(
            name=name, fn=fn, description=description, params=dict(params or {}), tags=tags
        )
        return fn

    return deco


def list_recipes() -> list[Recipe]:
    return list(_REGISTRY.values())


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的 recipe：{name}（可用：{sorted(_REGISTRY)}）")
    return _REGISTRY[name]


def run_recipe(model: Any, name: str, params: Mapping[str, Any] | None = None) -> None:
    """按名运行 recipe：默认参数被 ``params`` 覆盖后交给 build 纯函数。"""
    r = get_recipe(name)
    merged = {**r.params, **(params or {})}
    r.fn(model, merged)


# ---------------------------------------------------------------------------
# 内置示例 recipe：2D 矩形波导 + 压力声学频域（gain_ep 类问题的最小骨架）
# ---------------------------------------------------------------------------
@recipe(
    "waveguide2d_acpr",
    description="2D 矩形波导 + 压力声学频域 + 自由三角网格 + 频域研究（最小可跑骨架）",
    params={
        "L": 0.5,
        "H": 0.1,
        "freq": 3430.0,
        "hmax": 0.01,
    },
    tags=("acoustics", "2d"),
)
def _waveguide2d_acpr(model: Any, p: Mapping[str, Any]) -> None:
    set_parameters(model, {"f": f"{p['freq']}[1/s]", "c0": "343[m/s]"})
    add_rectangle(model, "r1", (float(p["L"]), float(p["H"])))
    finalize_geom(model)
    add_physics(model, "acpr", "PressureAcousticsFrequency", "geom1")
    add_mesh(model, "mesh1")
    set_mesh_size(model, float(p["hmax"]))
    run_mesh(model)
    add_study(model, "std1", [("freq", "Frequency")])
