"""结果与几何/网格导出：Agent「观察」仿真的两条通道。

- **视觉通道**：:func:`export_image` 导出绘图组为 PNG（COMSOL 原生渲染）；若 headless 图形
  栈不可用，则由 postprocess.py 的 pyvista 离屏渲染兜底（导出 VTK 后在 Python 侧渲染）。
- **数值通道**：:func:`export_data`（场数据 → CSV/TXT/VTK）、:func:`export_table`（探针表 →
  CSV/TXT）、:func:`export_mesh` / :func:`export_geometry`（网格/几何 → VTK/STL/PLY，供 pyvista）。

COMSOL 导出节点的属性名因版本/类型而异，故本模块对关键 set 采用**候选名逐个尝试**的防御式写法，
并把失败原因结构化返回（见 :class:`ExportResult`），便于在实机验证中定位并微调。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import postprocess as _post
from .config import settings
from .inspect import _call_if, _jmodel, _safe, _tags
from .postprocess import (
    comsol_auto_window,
    detect_frame_box_array,
    interior_blank_metrics,
    read_gray_png,
)


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


def _existing_export_tags(model: Any) -> set[str]:
    """模型中已存在的 ``result().export()`` 节点 tag 集合（防御式，失败返空）。"""
    try:
        return {str(t) for t in _tags(_result_export(model))}
    except Exception:  # noqa: BLE001
        return set()


def _free_export_tag(model: Any, prefix: str) -> str:
    """生成一个在该模型 export 序列中**尚未占用**的 tag（``prefix`` + 最小空闲序号）。

    COMSOL tag 须为合法标识符。不能用模块级递增计数器：加载已含导出节点（如预求解
    模型自带的 img2/img3）的模型时，裸计数器会与之碰撞并抛
    ``FlException: 已存在具有给定名称的对象``。这里查询模型现有 tag 并跳过占用者，
    对「同一会话多次导出」与「模型自带导出节点」两种情况都安全。
    """
    used = _existing_export_tags(model)
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"


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
        ftype = str(
            _call_if(feat, "getType", default="")
            or _call_if(feat, "type", default="")
            or ""
        )
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


def _apply_clean_brand(node: Any, applied: list[str]) -> None:
    """在 Image 导出节点上按 COMSOL 手册关掉品牌 logo（= 「optional components」之一）。

    手册「Image (Export)」属性表：``logo*d`` 仅在 ``options*d`` 为 on 时才受控（默认
    logo*d=on、options*d=off），故须**先开 options 再关 logo**。1d/2d/3d 三档都设，
    免绘图组维数不同而漏。属性名不存在时静默跳过（版本差异），由导出后的像素核验兜底。
    """
    for prop in ("options1d", "options2d", "options3d"):
        if _set_first_ok(node, (prop,), "on") is not None:
            applied.append(f"{prop}=on")
    for prop in ("logo1d", "logo2d", "logo3d"):
        if _set_first_ok(node, (prop,), "off") is not None:
            applied.append(f"{prop}=off")


def _json_default(o: Any) -> Any:
    """``json.dumps(default=...)`` 兜底：把无法原生序列化的对象降级为纯 Python。

    覆盖三类：
    - ``np.ndarray`` → ``tolist()``；``np.generic``（``np.int*``/``np.float*``/``np.bool_``）
      → ``item()``：sidecar 的 ``crop_box_px`` / ``interior`` 度量常含这些类型。
    - 其余未知类型（如**未经 ``str()`` 归一的 JPype ``java.lang.String`` 代理**——
      直接取自 ``result().tags()`` 的 ``plotgroup`` 即属此类）→ ``str(o)`` 兜底。

    sidecar 是 best-effort 元数据，绝不因某个值不可序列化而抛 ``TypeError`` 令整份
    sidecar 丢失；故末档降级为 ``str`` 而非抛错。
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


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
    strip_brand: bool | None = None,
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
        clean: 隐藏 colorbar/图例/标题，并**清除 COMSOL 品牌 logo**（下游排版用）。后者走两手：
            ①导出节点原生开关 ``options*d=on`` + ``logo*d=off``（手册「Image (Export)」）；
            ②导出后像素核验——角区仍有品牌像素就地抹除并附 warning（原生开关在部分
            版本/绘图组类型上不生效，见 backlog 20261015-204911）。
        sidecar: 写 ``<out>.sidecar.json``（extent/clean 生效情况、像素尺寸、
            轴框像素框、内部空白度量、``brand`` 品牌清理读数）。
        strip_brand: 单独的品牌清除开关；``None``（默认）**跟随 clean**——即 ``clean=True``
            必然出无品牌图。显式 ``False`` 可保留品牌（仅当 clean=False 时有意义）。
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
    etag = tag or _free_export_tag(model, "img")
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
            return ExportResult(
                False, "image", None, f"无法设置 Image 源对象 sourceobject={plotgroup}"
            )
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
                for prop, val in zip(
                    ("xmin", "xmax", "ymin", "ymax"), extent, strict=True
                ):
                    if _set_first_ok(pg, (prop,), str(val)) is not None:
                        applied_extent[prop] = str(
                            _call_if(pg, "getString", prop, default="")
                        )
                if len(applied_extent) < 4:
                    warnings.append(
                        f"extent 仅部分生效：{applied_extent or '无'}（该绘图组类型可能不支持手动轴限）"
                    )
        if clean and pg is not None:
            for props, val in (
                (("colorlegend", "legend", "showcolorbar"), "off"),
                (("titletype",), "none"),
            ):
                hit = _set_first_ok(pg, props, val)
                if hit:
                    applied_clean.append(f"{hit}={val}")
        if clean:
            _apply_clean_brand(node, applied_clean)
        if color_range is not None:
            if pg is None:
                warnings.append(f"绘图组 {plotgroup} 不存在，--color-range 未生效")
            else:
                applied_scale += _apply_color_range(pg, color_range)
                if not applied_scale:
                    warnings.append(
                        "color_range 未生效：绘图组内无 Surface 类特征或属性名不匹配"
                    )
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
        return ExportResult(
            False, "image", None, f"{type(e).__name__}: {e}", warnings=warnings
        )

    if not out.exists():
        return ExportResult(
            False,
            "image",
            None,
            f"run() 后未生成文件：{out}（headless 图形栈可能不可用，改用 pyvista 渲染）",
            warnings=warnings,
        )

    # --- 品牌清除：像素级核验兜底（原生 logo*d=off 不保证在各版本/绘图组类型上生效）---
    brand: dict[str, Any] = {}
    do_strip = clean if strip_brand is None else bool(strip_brand)
    if do_strip:
        brand = _post.strip_brand(out)
        if brand.get("found"):
            warnings.append(
                "导出 PNG 仍带品牌标记（原生 logo*d=off 未生效）——已就地抹除："
                f"bbox={brand.get('bbox')} 填充={brand.get('fill')}"
            )
            applied_clean.append("brand_pixels_stripped")

    # --- 导出后自检：轴框 + 内部空白度量 + sidecar ---
    gray = read_gray_png(out)
    box = detect_frame_box_array(gray) if gray is not None else None
    metrics = (
        interior_blank_metrics(gray, box)
        if (gray is not None and box is not None)
        else None
    )
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
            extent_recovered = [
                float(v) for v in comsol_auto_window(tuple(geom_bbox), tuple(box))
            ]
        else:
            warnings.append(
                "提供了 geom_bbox 但未检测到轴框，extent_recovered 无法反演"
            )
    if sidecar:
        sc = out.with_name(out.stem + ".sidecar.json")
        payload = {
            "png": str(out),
            "plotgroup": str(plotgroup),
            "size_px": ([gray.shape[1], gray.shape[0]] if gray is not None else None),
            "extent_requested": list(extent) if extent else None,
            "extent_applied": applied_extent,
            "extent_recovered": extent_recovered,
            "geom_bbox": list(geom_bbox) if geom_bbox else None,
            "clean_applied": applied_clean,
            "scale_applied": applied_scale,
            "crop_box_px": list(box) if box else None,
            "interior": metrics,
            "brand": brand,
        }
        try:
            sc.write_text(
                json.dumps(
                    payload, indent=2, ensure_ascii=False, default=_json_default
                ),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            warnings.append(f"sidecar 写入失败：{type(e).__name__}: {e}")
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
    etag = tag or _free_export_tag(model, "data")
    try:
        node = _create_export(model, etag, "Data")
        if node is None:
            return ExportResult(False, "data", None, "无法创建 Data 导出节点")
        node.set("data", ds)
        if expr:
            node.set(
                "expr", list(expr) if isinstance(expr, (list, tuple)) else str(expr)
            )
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


def _free_dataset_tag(seq: Any, prefix: str) -> str:
    """生成一个在该 dataset 序列中尚未占用的 tag（``prefix`` + 最小空闲序号）。"""
    used = {str(t) for t in _tags(seq)}
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"


def _set_coerced(node: Any, prop: str, value: Any) -> bool:
    """设置一个属性，依次尝试原生类型 → 字符串形态（COMSOL 各版本对 double/int/bool
    属性的 Java 签名敏感，JPype 重载解析未必命中；失败再降级为字符串）。"""
    for v in (value, str(value)):
        try:
            node.set(prop, v)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def export_polar_csv(
    model: Any,
    out_file: str | Path,
    *,
    expr: str | Sequence[str],
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    angle_min: float = 0.0,
    angle_max: float = 180.0,
    angle_step: float = 1.0,
    source: str | None = None,
    dataset_tag: str | None = None,
    tag: str | None = None,
) -> ExportResult:
    """沿**角度序列**把场量导出为数值 CSV（外场/远场极坐标扫描的数值通道）。

    在 ``source`` 底数据集上建一个 ``ParCurve2D``（参数曲线）采样数据集：以 ``center`` 为
    圆心、``radius`` 为半径、``alpha`` 为参数（**单位：度**，``angle_min..angle_max`` 步长
    ``angle_step``）的圆弧；逐点求值 ``expr``，再用 Data 导出节点把「角度 + 场值」写成 CSV。

    外场量（如 ``acpr.efc1.pext``）须在**网格外**求值，故数据集置 ``global=on``（COMSOL
    手册：ParCurve2D 的 global 属性正是为 far-field results processing 在网格外求全局表达式
    而设）；因此 ``radius`` 应落在求解域之外（远场区），否则取值不具外场语义。

    Args:
        expr: 场表达式（如 ``"acpr.efc1.pext"``）；多个则逐列导出。
        radius: 采样圆弧半径（模型长度单位）。
        center: 圆心 ``(x0, y0)``（模型长度单位）。
        angle_min/angle_max/angle_step: 角度序列（**度**）；默认 0→180 步 1 = 181 点。
        source: 底数据集 tag 或绘图组 tag；None → 首个可用数据集。
        dataset_tag/tag: 显式指定临时 ParCurve2D / 导出节点 tag；默认自动生成（不落盘模型，
            故不留污染）。
    """
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    jm = _jmodel(model)
    ds = _resolve_dataset(model, source)
    if ds is None:
        return ExportResult(False, "polar", None, f"无法解析数据集（source={source}）")
    span = (angle_max - angle_min) / angle_step if angle_step else 0
    n = int(round(span)) + 1
    if n < 2:
        return ExportResult(
            False, "polar", None, f"角度序列点数不足：range/step={span}（须 ≥1 段）"
        )
    warnings: list[str] = []
    dtag = dataset_tag or _free_dataset_tag(jm.result().dataset(), "pc2d")
    cx, cy = center
    try:
        dnode = jm.result().dataset().create(dtag, "ParCurve2D")
        if dnode is None:
            return ExportResult(False, "polar", None, "无法创建 ParCurve2D 数据集")
        # 参数以「度」为单位，避免下游把弧度量误当角度量。
        for prop, val in (
            ("data", ds),
            ("par1", "alpha"),
            ("parmin1", float(angle_min)),
            ("parmax1", float(angle_max)),
            ("res", int(n)),
            ("exprx", f"({cx})+({radius})*cos(alpha*pi/180)"),
            ("expry", f"({cy})+({radius})*sin(alpha*pi/180)"),
            ("global", True),
        ):
            if not _set_coerced(dnode, prop, val):
                warnings.append(f"ParCurve2D 属性 {prop} 未能设置")
    except Exception as e:  # noqa: BLE001
        return ExportResult(
            False, "polar", None, f"{type(e).__name__}: {e}", warnings=warnings
        )

    exprs = list(expr) if isinstance(expr, (list, tuple)) else [str(expr)]
    etag = tag or _free_export_tag(model, "data")
    try:
        enode = _create_export(model, etag, "Data")
        if enode is None:
            return ExportResult(False, "polar", None, "无法创建 Data 导出节点")
        enode.set("data", dtag)
        try:
            enode.set("expr", ["alpha", *exprs])  # 首列即角度（度）
        except Exception:  # noqa: BLE001
            enode.set("expr", exprs)
            warnings.append("角度列 alpha 未能加入 expr；角度可由 x/y 列反算")
        if _set_first_ok(enode, ("filename",), str(out)) is None:
            return ExportResult(False, "polar", None, "无法设置数据输出文件名属性")
        enode.run()
    except Exception as e:  # noqa: BLE001
        return ExportResult(
            False, "polar", None, f"{type(e).__name__}: {e}", warnings=warnings
        )

    if out.exists():
        return ExportResult(True, "polar", out, warnings=warnings)
    return ExportResult(
        False, "polar", None, f"run() 后未生成文件：{out}", warnings=warnings
    )


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
            return ExportResult(
                False,
                "mesh",
                None,
                "mesh 节点无 export()（改用 result Data 的 VTK 导出）",
            )
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
