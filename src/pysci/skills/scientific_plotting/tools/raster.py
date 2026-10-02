"""栅格面板：把外部渲染图（COMSOL PNG 等）以**数据坐标**精确叠进出版图。

背景：直接在渲染栅格上以 image-fraction 目视标定叠加标注极其脆弱——渲染含边距/
colorbar、``imshow`` 的 origin 易翻、坐标漂移需多轮目视迭代。本模块提供受支持路径：

1. comsol 技能 ``export image --extent …`` 产出 PNG + ``<png>.sidecar.json``
   （``extent_requested``/``extent_applied``/``crop_box_px``/``interior``）；
   缺 sidecar 时可用 ``post framebox --sidecar …`` 补 ``crop_box_px``。
2. 本模块按 ``crop_box_px`` 裁出轴框内部，``imshow(extent=数据窗口)`` →
   像素↔数据映射精确；叠加原语（旋转框/占位面板/虚线/文字）全部用数据坐标书写。

叠加原语（overlay spec，JSON/YAML 列表）::

    [{"type": "rotbox",  "cx":.., "cy":.., "w":.., "h":.., "angle":.., "ec":"k", "lw":1.5},
     {"type": "panel",   "cx":.., "cy":.., "w":.., "h":.., "angle":.., "text":"Exp. (TBD)"},
     {"type": "dashed",  "x0":.., "y0":.., "x1":.., "y1":..},
     {"type": "text",    "x":.., "y":.., "s":"..", "fontsize":8}]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.image import imread
from matplotlib.patches import Rectangle

#: extent 元组顺序（与 comsol export image --extent 一致）
Extent = tuple[float, float, float, float]
#: 像素框元组顺序（origin 顶左）
CropBox = tuple[int, int, int, int]


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """读 comsol 技能导出的 sidecar JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_overlays(path: str | Path) -> list[dict[str, Any]]:
    """读叠加原语清单（.json 或 .yaml/.yml）。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415 - 仅 YAML spec 需要

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"overlay spec 须为原语列表：{p}")
    return data


def read_raster(image: str | Path, crop_box: CropBox | None = None) -> np.ndarray:
    """读渲染 PNG（0-1 浮点数组）；给 crop_box 则裁出轴框内部。"""
    img = np.asarray(imread(str(image)), dtype=float)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        img = img[y0:y1, x0:x1]
    return img


def resolve_extent_and_crop(
    sidecar: dict[str, Any] | None,
    *,
    extent: Extent | None = None,
    crop_box: CropBox | None = None,
) -> tuple[Extent | None, CropBox | None]:
    """解析数据窗口与像素框：显式参数优先，其次 sidecar。

    sidecar 的 ``extent_requested`` 仅在 ``extent_applied`` 四项齐全时可信
    （否则轴限未真正生效，像素↔数据映射不成立）；若未生效但有 ``extent_recovered``
    （comsol 技能 ``export image --geom-bbox`` 反演 auto-zoom 写入），则采用后者。
    """
    if sidecar:
        if crop_box is None and sidecar.get("crop_box_px"):
            crop_box = tuple(sidecar["crop_box_px"])
        if extent is None:
            applied = sidecar.get("extent_applied") or {}
            req = sidecar.get("extent_requested")
            if req and len(applied) == 4:
                extent = tuple(req)
            elif sidecar.get("extent_recovered"):
                extent = tuple(sidecar["extent_recovered"])
    return extent, crop_box


def add_raster_panel(
    fig: Any,
    rect: list[float],
    image: str | Path,
    extent: Extent,
    *,
    crop_box: CropBox | None = None,
    aspect: str = "equal",
    axis_off: bool = True,
) -> Any:
    """在 fig 的 rect 位置加一个栅格面板轴，像素↔数据映射由 extent 精确给定。"""
    img = read_raster(image, crop_box)
    ax = fig.add_axes(rect)
    ax.imshow(img, extent=list(extent), origin="upper", aspect=aspect)
    if axis_off:
        ax.set_axis_off()
    return ax


def apply_overlays(ax: Any, overlays: list[dict[str, Any]]) -> None:
    """在数据坐标下应用叠加原语（rotbox/panel/dashed/text）。"""
    for ov in overlays:
        kind = ov.get("type")
        if kind in ("rotbox", "panel"):
            cx, cy, w, h = ov["cx"], ov["cy"], ov["w"], ov["h"]
            fill = kind == "panel" or bool(ov.get("fill", False))
            # Rectangle 绕锚点（未旋转左下角）旋转；反解锚点使 (cx,cy) 为旋转后真中心
            a = math.radians(ov.get("angle", 0))
            ax_x = cx - (w / 2 * math.cos(a) - h / 2 * math.sin(a))
            ax_y = cy - (w / 2 * math.sin(a) + h / 2 * math.cos(a))
            ax.add_patch(
                Rectangle(
                    (ax_x, ax_y),
                    w,
                    h,
                    angle=ov.get("angle", 0),
                    fill=fill,
                    ec=ov.get("ec", "k"),
                    fc=ov.get("fc", "white" if kind == "panel" else "none"),
                    lw=ov.get("lw", 1.5),
                    alpha=ov.get("alpha", 0.92 if kind == "panel" else 1.0),
                )
            )
            if kind == "panel" and ov.get("text"):
                ax.text(
                    cx,
                    cy,
                    ov["text"],
                    ha="center",
                    va="center",
                    fontsize=ov.get("fontsize", 7),
                    color=ov.get("color", "0.35"),
                )
        elif kind == "dashed":
            ax.plot(
                [ov["x0"], ov["x1"]],
                [ov["y0"], ov["y1"]],
                ls=ov.get("ls", "--"),
                lw=ov.get("lw", 1.0),
                color=ov.get("color", "k"),
            )
        elif kind == "text":
            ax.text(
                ov["x"],
                ov["y"],
                ov["s"],
                fontsize=ov.get("fontsize", 8),
                ha=ov.get("ha", "left"),
                va=ov.get("va", "baseline"),
                color=ov.get("color", "k"),
            )
        else:
            raise ValueError(
                f"未知叠加原语 type={kind!r}（支持 rotbox/panel/dashed/text）"
            )


def compose_raster_panel(
    image: str | Path,
    out: str | Path,
    *,
    extent: Extent | None = None,
    sidecar: str | Path | None = None,
    crop_box: CropBox | None = None,
    overlays: list[dict[str, Any]] | None = None,
    figsize: tuple[float, float] = (7, 7),
    dpi: int = 150,
    axis_off: bool = True,
) -> Path:
    """单面板合成：栅格底图（精确 extent）+ 数据坐标叠加原语 → PNG。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - 合成才需要 pyplot

    sc = load_sidecar(sidecar) if sidecar else None
    extent, crop_box = resolve_extent_and_crop(sc, extent=extent, crop_box=crop_box)
    if extent is None:
        raise ValueError(
            "缺数据窗口：请显式 --extent，或提供含 extent_applied 的 sidecar"
        )
    fig = plt.figure(figsize=figsize)
    ax = add_raster_panel(
        fig, [0, 0, 1, 1], image, extent, crop_box=crop_box, axis_off=axis_off
    )
    if overlays:
        apply_overlays(ax, overlays)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 多面板组装（compose_grid）：N×M 混合 raster / 纯图像 / matplotlib 轴
# ---------------------------------------------------------------------------
def load_grid_spec(path: str | Path) -> dict[str, Any]:
    """读多面板组装 spec（.json / .yaml/.yml），返回 dict。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415 - 仅 YAML spec 需要

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"grid spec 须为映射（含 rows/cols/panels）：{p}")
    return data


@dataclass
class GridResult:
    """:func:`compose_grid` 的产物：figure + 网格轴，供调用方（尤其 axes 类面板）二次填充后保存。"""

    fig: Any
    axes: list[list[Any]]  # rows×cols，空格为 None
    axes_map: dict[str, Any] = field(default_factory=dict)  # panel name -> Axes
    raster_axes: list[Any] = field(
        default_factory=list
    )  # 栅格场面板轴（COMSOL PNG，colorbar 近似）
    field_axes: list[Any] = field(
        default_factory=list
    )  # 数据驱动场面板轴（colorbar 严格一致）

    def save(self, out: str | Path, *, dpi: int = 150, **kwargs: Any) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=dpi, bbox_inches="tight", **kwargs)
        return out

    def close(self) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close(self.fig)


def compose_grid(spec: dict[str, Any]) -> GridResult:
    """按 spec 组装 N×M 多面板图（混合 raster / 纯图像 / 待填充 matplotlib 轴）。

    领域无关的多面板出版图原语：把“场图（带数据窗口）+ 纯图像 + 自绘轴”统一排布，
    共享 colorbar / 面板字母 / 行列标题 / suptitle。spec 驱动，保持通用。

    spec 键：
      rows, cols         网格尺寸（必填）
      figsize            (w,h) 英寸；默认 (cols*3, rows*3)
      panels             面板列表，按行优先填入空闲格；每项可带 row/col 显式定位
      panel_labels       bool；自动 (a)(b)… 于每格左上（只计非空格）
      col_titles         长 cols 的标题列表（置于第 0 行各列上方）
      row_labels         长 rows 的标签列表（置于各行左侧，竖排）
      colorbar           {cmap,vmin,vmax,label,location,axes:[name…],pad,fraction}
      suptitle           总标题
      rcparams           dict，绘制前 update 到 plt.rcParams
      gridspec_kw        dict，透传 add_gridspec（hspace/wspace 等）

    panel 键：
      kind               raster | image | field | axes | blank（默认 image）
      image              PNG 路径（raster/image）
      sidecar            sidecar JSON 路径（raster，取 extent_applied/extent_recovered/crop_box_px）
      extent             显式数据窗口 [x0,x1,y0,y1]（raster，优先于 sidecar）
      crop_box           显式像素框 [x0,y0,x1,y1]（raster）
      overlays           叠加原语列表（raster，见模块 docstring）
      source/scalars/cols  field 面板：场源（.vtk/.csv）、VTK 标量名、CSV 的 (x,y,v) 列索引
      cmap/vmin/vmax     field 面板色标；缺省继承 colorbar 的 cmap/vmin/vmax → 与共享 colorbar 严格一致
      aspect / axis_off  imshow 纵横比 / 是否关轴
      title / name       格标题 / 轴命名（axes_map 键，供 colorbar.axes 或调用方填充）

    raster/image 面板立即绘制；axes 面板创建空轴并记入 ``GridResult.axes_map[name]``，
    由调用方在 ``save()`` 前自行绘制（CLI 纯 spec 模式下 axes 面板仅留空轴）。
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - 组装才需要 pyplot

    rows = int(spec.get("rows", 1))
    cols = int(spec.get("cols", 1))
    figsize = tuple(spec.get("figsize", (cols * 3.0, rows * 3.0)))
    fig = plt.figure(figsize=figsize)
    if spec.get("rcparams"):
        plt.rcParams.update(spec["rcparams"])
    gs = fig.add_gridspec(rows, cols, **dict(spec.get("gridspec_kw", {})))

    axes: list[list[Any]] = [[None] * cols for _ in range(rows)]
    axes_map: dict[str, Any] = {}
    raster_axes: list[Any] = []
    field_axes: list[Any] = []
    cb = spec.get(
        "colorbar"
    )  # 提前：field 面板默认继承其 cmap/vmin/vmax 以保证严格一致

    panels = spec.get("panels", []) or []
    occupied: set[tuple[int, int]] = {
        (int(p["row"]), int(p["col"])) for p in panels if "row" in p and "col" in p
    }
    auto = 0
    for panel in panels:
        if "row" in panel and "col" in panel:
            r, c = int(panel["row"]), int(panel["col"])
        else:
            while auto < rows * cols and (auto // cols, auto % cols) in occupied:
                auto += 1
            r, c = divmod(auto, cols)
            auto += 1
        occupied.add((r, c))
        ax = fig.add_subplot(gs[r, c])
        axes[r][c] = ax
        name = panel.get("name")
        if name:
            axes_map[name] = ax
        kind = panel.get("kind", "image")
        if kind == "raster":
            sc = load_sidecar(panel["sidecar"]) if panel.get("sidecar") else None
            ext = tuple(panel["extent"]) if panel.get("extent") else None
            crop = tuple(panel["crop_box"]) if panel.get("crop_box") else None
            ext, crop = resolve_extent_and_crop(sc, extent=ext, crop_box=crop)
            if ext is None:
                raise ValueError(
                    f"raster 面板 {name or (r, c)} 缺数据窗口：需 extent 或含 "
                    "extent_applied/extent_recovered 的 sidecar"
                )
            ax.imshow(
                read_raster(panel["image"], crop),
                extent=list(ext),
                origin="upper",
                aspect=panel.get("aspect", "equal"),
            )
            if panel.get("overlays"):
                apply_overlays(ax, panel["overlays"])
            if panel.get("axis_off", True):
                ax.set_axis_off()
            raster_axes.append(ax)
        elif kind == "image":
            ax.imshow(read_raster(panel["image"]), aspect=panel.get("aspect", "equal"))
            if panel.get("axis_off", True):
                ax.set_axis_off()
        elif kind == "axes":
            if panel.get("axis_off", False):
                ax.set_axis_off()
        elif kind == "field":
            from . import field as _field  # noqa: PLC0415 - 仅 field 面板需要

            xy, vals = _field.load_field_points(
                panel["source"],
                scalars=panel.get("scalars"),
                cols=tuple(panel.get("cols", (0, 1, 2))),
            )
            _field.add_field_panel(
                ax,
                xy,
                vals,
                cmap=panel.get("cmap") or (cb or {}).get("cmap", "bwr"),
                vmin=panel.get("vmin", (cb or {}).get("vmin")),
                vmax=panel.get("vmax", (cb or {}).get("vmax")),
                triangles=panel.get("triangles"),
                shading=panel.get("shading", "gouraud"),
            )
            ax.set_aspect(panel.get("aspect", "equal"))
            if panel.get("axis_off", False):
                ax.set_axis_off()
            field_axes.append(ax)
        elif kind in ("blank", "empty"):
            ax.set_axis_off()
            axes[r][c] = None
        else:
            raise ValueError(f"未知面板 kind={kind!r}（支持 raster/image/axes/blank）")
        if axes[r][c] is not None and panel.get("title"):
            ax.set_title(panel["title"], fontsize=panel.get("title_fontsize", 11))

    if spec.get("panel_labels"):
        fmt = spec.get("panel_label_format", "({})")
        fs = spec.get("panel_label_fontsize", 11)
        fw = spec.get("panel_label_fontweight", "bold")
        i = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if ax is None:
                    continue
                ax.text(
                    0.02,
                    0.98,
                    fmt.format(chr(ord("a") + i)),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=fs,
                    fontweight=fw,
                )
                i += 1

    for c, t in enumerate(spec.get("col_titles") or []):
        if c < cols and t and axes[0][c] is not None:
            axes[0][c].set_title(t, fontsize=spec.get("col_title_fontsize", 11))

    for r, lab in enumerate(spec.get("row_labels") or []):
        if r >= rows or not lab:
            continue
        anchor = next((a for a in axes[r] if a is not None), None)
        if anchor is not None:
            anchor.text(
                spec.get("row_label_offset", -0.08),
                0.5,
                lab,
                transform=anchor.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=spec.get("row_label_fontsize", 11),
            )

    if cb:
        target = cb.get("axes")
        if target:
            cb_axes = [axes_map[n] for n in target if n in axes_map]
        else:
            cb_axes = (
                field_axes or raster_axes
            )  # field 面板优先（颜色与 colorbar 严格一致）
        if cb_axes:
            mappable = ScalarMappable(
                norm=Normalize(cb.get("vmin"), cb.get("vmax")),
                cmap=cb.get("cmap", "viridis"),
            )
            cbar = fig.colorbar(
                mappable,
                ax=cb_axes,
                location=cb.get("location", "right"),
                pad=cb.get("pad", 0.02),
                fraction=cb.get("fraction", 0.03),
            )
            if cb.get("label"):
                cbar.set_label(cb["label"], fontsize=cb.get("label_fontsize", 10))

    if spec.get("suptitle"):
        fig.suptitle(spec["suptitle"], fontsize=spec.get("suptitle_fontsize", 13))

    return GridResult(
        fig=fig,
        axes=axes,
        axes_map=axes_map,
        raster_axes=raster_axes,
        field_axes=field_axes,
    )


def compose_grid_to_file(
    spec: dict[str, Any], out: str | Path, *, dpi: int | None = None, close: bool = True
) -> Path:
    """:func:`compose_grid` + savefig 一步到位（CLI / 无需二次填充 axes 面板时用）。"""
    res = compose_grid(spec)
    p = res.save(
        out,
        dpi=dpi or int(spec.get("dpi", 150)),
        pad_inches=spec.get("pad_inches", 0.1),
    )
    if close:
        res.close()
    return p
