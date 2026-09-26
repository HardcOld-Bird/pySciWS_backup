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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .inspect import _call_if, _jmodel, _safe, _tags

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

    def report(self) -> str:
        if self.ok:
            return f"[{self.kind}] OK -> {self.out_path}"
        return f"[{self.kind}] FAILED: {self.error}"


def _result_export(model: Any) -> Any:
    """返回 ``model.java.result().export()`` 序列。"""
    jm = _jmodel(model)
    return jm.result().export()


def _set_first_ok(node: Any, candidates: tuple[str, ...], value: Any) -> str | None:
    """依次尝试候选属性名 set，返回首个成功的属性名；全失败返回 None。"""
    last_err = None
    for prop in candidates:
        try:
            node.set(prop, value)
            return prop
        except Exception as e:  # noqa: BLE001
            last_err = f"{prop}: {e}"
    return None


def _create_export(model: Any, tag: str, kind: str) -> Any:
    exp = _result_export(model)
    exp.create(tag, kind)
    return exp.get(tag) if hasattr(exp, "get") else _safe(exp, tag, default=None)


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
) -> ExportResult:
    """把一个绘图组导出为 PNG（COMSOL 原生渲染）。

    Args:
        plotgroup: 绘图组 tag（如 "pg10"）。
        out_png: 输出 PNG 路径。
        size: (宽, 高) 像素；None → 用 COMSOL 默认。
        tag: 导出节点 tag（默认自动生成）。
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    etag = tag or _new_tag("img")
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
        node.run()
    except Exception as e:  # noqa: BLE001
        return ExportResult(False, "image", None, f"{type(e).__name__}: {e}")

    if out.exists():
        return ExportResult(True, "image", out)
    return ExportResult(False, "image", None, f"run() 后未生成文件：{out}（headless 图形栈可能不可用，改用 pyvista 渲染）")


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
