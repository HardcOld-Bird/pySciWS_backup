"""结果与几何/网格导出：Agent「观察」仿真的两条通道。

- **视觉通道**：:func:`export_image` 导出绘图组为 PNG（COMSOL 原生渲染）；若 headless 图形
  栈不可用，则由 postprocess.py 的 pyvista 离屏渲染兜底（导出 VTK 后在 Python 侧渲染）。
- **数值通道**：:func:`export_data`（场数据 → CSV/TXT/VTK）、:func:`export_table`（探针表 →
  CSV/TXT）、:func:`export_mesh` / :func:`export_geometry`（网格/几何 → VTK/STL/PLY，供 pyvista）。

COMSOL 导出节点的属性名因版本/类型而异，故本模块对关键 set 采用**候选名逐个尝试**的防御式写法，
并把失败原因结构化返回（见 :class:`ExportResult`），便于在实机验证中定位并微调。
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .inspect import _call_if, _jmodel, _safe, _tags
from .postprocess import (
    comsol_auto_window,
    detect_frame_box_array,
    interior_blank_metrics,
    read_gray_png,
)

# 导出节点 tag 计数器（COMSOL tag 须为合法标识符，用前缀 + 递增序号）
_tag_counter = itertools.count(1)


def _new_tag(prefix: str) -> str:
    return f"{prefix}{next(_tag_counter)}"


@dataclass
class ExportResult:
    ok: bool
    kind: str
    out_path: Path | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        if self.ok:
            base = f"[{self.kind}] OK -> {self.out_path}"
        else:
            base = f"[{self.kind}] FAILED: {self.error}"
        if self.warnings:
            base += "\n" + "\n".join(f"  [warn] {w}" for w in self.warnings)
        return base


def _result_export(model: Any) -> Any:
    """返回 ``model.java.result().export()`` 序列。"""
    jm = _jmodel(model)
    return jm.result().export()


def _set_first_ok(node: Any, candidates: tuple[str, ...], value: Any) -> str | None:
    """依次尝试候选属性名 set，返回首个成功的属性名；全失败返回 None。"""
    for prop in candidates:
        try:
            node.set(prop, value)
            return prop
        except Exception:  # noqa: BLE001
            continue
    return None


def _create_export(model: Any, tag: str, kind: str) -> Any:
    exp = _result_export(model)
    exp.create(tag, kind)
    return exp.get(tag) if hasattr(exp, "get") else _safe(exp, tag, default=None)


def _apply_color_range(pg: Any, color_range: tuple[float, float]) -> list[str]:
    """对绘图组内全部 Surface 类特征统一色标（rangecoloractive/min/max）。

    多 case 对比时“共享 colorbar 标度”的源头保障：不靠自动标度（逐 case 漂移），
    而是显式钉死颜色范围。返回实际生效的 ``tag.prop`` 列表（防御式，全失败返空）。
    """
    applied: list[str] = []
    lo, hi = color_range
    for ftag in _tags(_call_if(pg, "feature", default=None)):
        feat = _safe(pg.feature, ftag, default=None)
        if feat is None:
            continue
        ftype = str(_call_if(feat, "getType", default="") or _call_if(feat, "type", default="") or "")
        if "Surface" not in ftype:
            continue
        if _set_first_ok(feat, ("rangecoloractive",), "on") is None:
            continue
        for prop, val in (("rangecolormin", str(lo)), ("rangecolormax", str(hi))):
            if _set_first_ok(feat, (prop,), val) is not None:
                applied.append(f"{ftag}.{prop}={val}")
    return applied


def _apply_polar_rmax(pg: Any, polar_rmax: tuple[float, float]) -> list[str]:
    """对极坐标绘图组（PolarGroup）统一极径范围（axislimits/rmin/rmax）。"""
    applied: list[str] = []
    rmin, rmax = polar_rmax
    for prop, val in (("axislimits", "on"), ("rmin", str(rmin)), ("rmax", str(rmax))):
        if _set_first_ok(pg, (prop,), val) is not None:
            applied.append(f"{prop}={val}")
    return applied


# ---------------------------------------------------------------------------
# 图像（PNG）
# ---------------------------------------------------------------------------
def export_image(
    model: Any,
    plotgroup: str,
    out_png: str | Path,
    *,
    size: tuple[int, int] | None = None,
    tag: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    clean: bool = False,
    sidecar: bool = True,
    color_range: tuple[float, float] | None = None,
    polar_rmax: tuple[float, float] | None = None,
    geom_bbox: tuple[float, float, float, float] | None = None,
) -> ExportResult:
    """把一个绘图组导出为 PNG（COMSOL 原生渲染）。

    Args:
        plotgroup: 绘图组 tag（如 "pg10"）。
        out_png: 输出 PNG 路径。
        size: (宽, 高) 像素；None → 用 COMSOL 默认。
        tag: 导出节点 tag（默认自动生成）。
        extent: (x0, x1, y0, y1) 显式轴限（数据坐标）；设置后渲染窗口已知，
            配合 sidecar 的 crop_box_px 可得精确 pixel↔data 映射（下游叠图用）。
            注意：2D 绘图组（PlotGroup2D）不暴露轴限属性，此参数对其无效——
            改用 ``geom_bbox`` 走 auto-zoom 反演（见下）。
        clean: 隐藏 colorbar/图例/标题（仅留轴框与内容，减少下游裁剪干扰）。
        sidecar: 写 ``<out>.sidecar.json``（extent/clean 生效情况、像素尺寸、
            轴框像素框、内部空白度量）。
        color_range: (min, max) 统一 Surface 色标（rangecoloractive=on +
            rangecolormin/max），使多 case 共享同一 colorbar 标度。
        polar_rmax: (rmin, rmax) 统一极坐标绘图组极径（axislimits=on）。
        geom_bbox: (x0, x1, y0, y1) 几何包围盒（数据坐标）。提供时用
            :func:`comsol_auto_window` 反演 COMSOL 2D auto-zoom 窗口，结果写入
            sidecar 的 ``extent_recovered``，下游 raster 直接可用。

    导出后自动做**空白自校验**：检测轴框内部是否近纯色；若是则附 warning
    （常见原因：重求解后绘图组未重绑数据集 ``result(pg).set('data', dset)``）。
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    etag = tag or _new_tag("img")
    warnings: list[str] = []
    applied_extent: dict[str, str] = {}
    applied_clean: list[str] = []
    applied_scale: list[str] = []
    try:
        node = _create_export(model, etag, "Image")
        if node is None:
            return ExportResult(False, "image", None, "无法创建 Image 导出节点")
        # Image 节点用 sourcetype/sourceobject 指定来源（不是 plotgroup 属性）
        _safe(node.set, "sourcetype", "plotgroup")
        if _set_first_ok(node, ("sourceobject",), plotgroup) is None:
            return ExportResult(False, "image", None, f"无法设置 Image 源对象 sourceobject={plotgroup}")
        if _set_first_ok(node, ("pngfilename", "filename"), str(out)) is None:
            return ExportResult(False, "image", None, "无法设置 PNG 输出文件名属性")
        if size:
            _safe(node.set, "size", "manual")
            _set_first_ok(node, ("unit",), "pixel")
            _safe(node.set, "height", int(size[1]))
            _safe(node.set, "width", int(size[0]))
        pg = _safe(_jmodel(model).result, plotgroup, default=None)
        if extent is not None:
            if pg is None:
                warnings.append(f"绘图组 {plotgroup} 不存在，--extent 未生效")
            else:
                for prop, val in zip(("xmin", "xmax", "ymin", "ymax"), extent, strict=True):
                    if _set_first_ok(pg, (prop,), str(val)) is not None:
                        applied_extent[prop] = str(_call_if(pg, "getString", prop, default=""))
                if len(applied_extent) < 4:
                    warnings.append(f"extent 仅部分生效：{applied_extent or '无'}（该绘图组类型可能不支持手动轴限）")
        if clean and pg is not None:
            for props, val in ((("colorlegend", "legend", "showcolorbar"), "off"), (("titletype",), "none")):
                hit = _set_first_ok(pg, props, val)
                if hit:
                    applied_clean.append(f"{hit}={val}")
        if color_range is not None:
            if pg is None:
                warnings.append(f"绘图组 {plotgroup} 不存在，--color-range 未生效")
            else:
                applied_scale += _apply_color_range(pg, color_range)
                if not applied_scale:
                    warnings.append("color_range 未生效：绘图组内无 Surface 类特征或属性名不匹配")
        if polar_rmax is not None:
            if pg is None:
                warnings.append(f"绘图组 {plotgroup} 不存在，--polar-rmax 未生效")
            else:
                hit = _apply_polar_rmax(pg, polar_rmax)
                applied_scale += hit
                if not hit:
                    warnings.append("polar_rmax 未生效：该绘图组可能不是 PolarGroup")
        node.run()
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "image", None, f"{type(e).__name__}: {e}", warnings=warnings)

    if not out.exists():
        return ExportResult(
            False, "image", None,
            f"run() 后未生成文件：{out}（headless 图形栈可能不可用，改用 pyvista 渲染）",
            warnings=warnings,
        )

    # --- 导出后自检：轴框 + 内部空白度量 + sidecar ---
    gray = read_gray_png(out)
    box = detect_frame_box_array(gray) if gray is not None else None
    metrics = interior_blank_metrics(gray, box) if (gray is not None and box is not None) else None
    if metrics and metrics["blank"]:
        warnings.append(
            f"PNG 轴框内部近空白（unique_q={metrics['unique_q']}, std={metrics['std']}）："
            "绘图组可能无数据——重求解后须 result(pg).set('data', dset) 重绑数据集"
        )
    if box is None:
        warnings.append("未检测到轴框像素框（sidecar 的 crop_box_px 为 null）")
    extent_recovered: list[float] | None = None
    if geom_bbox is not None:
        if box is not None:
            extent_recovered = [float(v) for v in comsol_auto_window(tuple(geom_bbox), tuple(box))]
        else:
            warnings.append("提供了 geom_bbox 但未检测到轴框，extent_recovered 无法反演")
    if sidecar:
        sc = out.with_name(out.stem + ".sidecar.json")
        payload = {
            "png": str(out),
            "plotgroup": plotgroup,
            "size_px": ([gray.shape[1], gray.shape[0]] if gray is not None else None),
            "extent_requested": list(extent) if extent else None,
            "extent_applied": applied_extent,
            "extent_recovered": extent_recovered,
            "geom_bbox": list(geom_bbox) if geom_bbox else None,
            "clean_applied": applied_clean,
            "scale_applied": applied_scale,
            "crop_box_px": list(box) if box else None,
            "interior": metrics,
        }
        try:
            sc.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"sidecar 写入失败：{type(e).__name__}")
    return ExportResult(True, "image", out, warnings=warnings)


# ---------------------------------------------------------------------------
# 场数据（CSV/TXT/VTK）
# ---------------------------------------------------------------------------
_DATA_FORMAT_BY_SUFFIX = {".csv": "csv", ".txt": "csv", ".vtk": "vtk", ".vtu": "vtk"}


def _resolve_dataset(model: Any, source: str | None) -> str | None:
    """把 ``source``（dataset tag 或 plot group tag）解析为 dataset tag。

    Data 导出节点作用在**数据集**（``data`` 属性）上而非绘图组。若传入的是绘图组 tag，
    则读它的 ``data`` 属性拿到其底层数据集；若已是 dataset tag 则直接用；都不行则回退
    到首个可用数据集。
    """
    jm = _jmodel(model)
    dset_seq = _call_if(_call_if(jm, "result", default=None), "dataset", default=None)
    dsets = _tags(dset_seq)
    if source and source in dsets:
        return source
    if source:
        pg = _safe(jm.result, source, default=None)
        if pg is not None:
            ds = _safe(pg.getString, "data", default=None)
            if ds:
                return str(ds)
    return dsets[0] if dsets else None


def export_data(
    model: Any,
    source: str,
    out_file: str | Path,
    *,
    fmt: str | None = None,
    expr: str | Sequence[str] | None = None,
    tag: str | None = None,
) -> ExportResult:
    """导出场数据到 CSV/TXT/VTK（VTK 供 pyvista 做程序化后处理与渲染）。

    Args:
        source: dataset tag（如 "dset1"）或 plot group tag（自动解析到底层数据集）。
        out_file: 输出文件路径（后缀决定默认格式）。
        fmt: 显式格式 "csv"/"txt"/"vtk"；None → 由后缀推断。
        expr: 要导出的场表达式（如 "acpr.p_t"）；None → 仅导出网格坐标。
    """
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or _DATA_FORMAT_BY_SUFFIX.get(out.suffix.lower(), "csv")).lower()
    ds = _resolve_dataset(model, source)
    if ds is None:
        return ExportResult(False, "data", None, f"无法解析数据集（source={source}）")
    etag = tag or _new_tag("data")
    try:
        node = _create_export(model, etag, "Data")
        if node is None:
            return ExportResult(False, "data", None, "无法创建 Data 导出节点")
        node.set("data", ds)
        if expr:
            node.set("expr", list(expr) if isinstance(expr, (list, tuple)) else str(expr))
        if fmt in ("vtk", "vtu"):
            _safe(node.set, "exporttype", "vtk")
        if _set_first_ok(node, ("filename",), str(out)) is None:
            return ExportResult(False, "data", None, "无法设置数据输出文件名属性")
        node.run()
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "data", None, f"{type(e).__name__}: {e}")

    if out.exists():
        return ExportResult(True, "data", out)
    return ExportResult(False, "data", None, f"run() 后未生成文件：{out}")


# ---------------------------------------------------------------------------
# 探针表（CSV/TXT）
# ---------------------------------------------------------------------------
def export_table(model: Any, table_tag: str, out_file: str | Path) -> ExportResult:
    """把一个结果表（如探针/全局表）保存到文件。"""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    jm = _jmodel(model)
    try:
        table = jm.result().table(table_tag)
        # 优先用 save(path)；回退到读 getTableData 自行写
        saved = _safe(table.save, str(out), default=None)
        if out.exists():
            return ExportResult(True, "table", out)
        if saved is None:
            rows = _safe(table.getTableData, True, default=None)
            if rows is not None:
                with out.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(",".join(str(c) for c in row) + "\n")
                return ExportResult(True, "table", out)
        return ExportResult(False, "table", None, f"表保存失败：{table_tag}")
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "table", None, f"{type(e).__name__}: {e}")


def read_table(model: Any, table_tag: str) -> list[list[str]]:
    """直接把结果表读成 Python 二维字符串列表（不落盘）。"""
    jm = _jmodel(model)
    table = jm.result().table(table_tag)
    rows = _safe(table.getTableData, True, default=[]) or []
    return [[str(c) for c in row] for row in rows]


# ---------------------------------------------------------------------------
# 网格 / 几何（VTK/STL/PLY，供 pyvista 离屏渲染）
# ---------------------------------------------------------------------------
def export_mesh(
    model: Any,
    mesh_tag: str,
    out_file: str | Path,
    *,
    component: str = "comp1",
) -> ExportResult:
    """导出网格到文件（VTK/STL/PLY）。用于 pyvista 渲染网格 + 质量检查。"""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    jm = _jmodel(model)
    try:
        comp = jm.component(component) if component else jm
        mesh = comp.mesh(mesh_tag)
        exp = mesh.export() if hasattr(mesh, "export") else None
        if exp is None:
            return ExportResult(False, "mesh", None, "mesh 节点无 export()（改用 result Data 的 VTK 导出）")
        # 网格导出 API 形态因版本而异，尽力尝试
        if _safe(getattr(exp, "create", None), default=None) is not None:
            pass
        _set_first_ok(exp, ("filename", "vtkfilename", "stlfilename"), str(out))
        _safe(exp.set, "format", out.suffix.lstrip(".").lower())
        ran = _safe(exp.run, default=None)
        if ran is None:
            _safe(exp, "run")
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "mesh", None, f"{type(e).__name__}: {e}")

    if out.exists():
        return ExportResult(True, "mesh", out)
    return ExportResult(False, "mesh", None, f"run() 后未生成文件：{out}")


def export_geometry(
    model: Any,
    geom_tag: str,
    out_file: str | Path,
    *,
    component: str = "comp1",
) -> ExportResult:
    """导出几何到文件（STL/PLY 等）。GeomSequence 提供 export()（introspect 已确认）。"""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    jm = _jmodel(model)
    try:
        comp = jm.component(component) if component else jm
        geom = comp.geom(geom_tag)
        exp = geom.export()
        _set_first_ok(exp, ("filename",), str(out))
        _safe(exp.set, "format", out.suffix.lstrip(".").lower())
        _safe(exp.run) or _safe(exp, "run")
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "geometry", None, f"{type(e).__name__}: {e}")

    if out.exists():
        return ExportResult(True, "geometry", out)
    return ExportResult(False, "geometry", None, f"run() 后未生成文件：{out}")


# ---------------------------------------------------------------------------
# 便捷：默认输出目录
# ---------------------------------------------------------------------------
def default_out_dir(subdir: str = "exports") -> Path:
    """技能数据区下的默认导出目录 data/skills/comsol_simulation/runs/<subdir>。"""
    d = settings.runs_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
