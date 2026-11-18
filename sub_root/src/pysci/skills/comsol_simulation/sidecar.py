"""COMSOL 栅格 sidecar：生产端（数据坐标↔像素坐标变换元数据 JSON）+ 消费端（叠加）。

**契约一句话**：出版级导出的 PNG 旁边落一个 ``<name>.sidecar.json``，写清
「哪个绘图组/数据集、图多大、轴框在像素哪里、数据窗口是什么、data→pixel 仿射」，
下游（③域 truthful colorbar / 叠加标注）**不目视迭代**即可精确把数据坐标画回像素图上。

像素约定：**原点左上**、x 向右、y 向下；``crop_box_px = [x0, y0, x1, y1]``（轴框矩形，
顶左→底右）。数据坐标 y 向上——变换里 y 翻转由 ``sy > 0`` + 公式
``py = oy - sy*y`` 显式表达（:class:`Transform`）。

auto-zoom 反演（:func:`comsol_auto_window`）：COMSOL 2D 绘图组**无轴限属性**（Java API
不暴露 xmin/xmax…，旧项目逐属性确证），精确 pixel↔data 映射靠反演——COMSOL auto-zoom
= 等纵横比（每像素数据量 x=y）、以几何包围盒中心为中心、按轴框像素框长宽比在宽/高
受限方向展开。故 extent = f(几何 bbox, crop_box_px)，无需目视迭代。

ADR（2026-11-17，用户裁决）：③域移交的消费端**薄化入②域**（本模块下半），衔接
``pysci.skills.scientific_plotting.field_panel`` 的 truthful colorbar；像素擦除式
imgcheck **不建**（品牌剥离走 COMSOL 原生 ``logo*d=off``，见 export.py）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pysci.paths import assert_within_data

#: sidecar 契约版本（消费端按此校验；变更字段须升版本）。
SIDECAR_SCHEMA = "pysci.comsol.sidecar/1"


@dataclass(frozen=True)
class Transform:
    """数据坐标→像素坐标仿射（纯数学，y 翻转显式）。

    ``px = ox + sx*x``；``py = oy - sy*y``（sy>0 ⇒ 数据 y 向上、像素 y 向下）。

    Attributes:
        sx: x 向比例（像素/数据单位）。
        sy: y 向比例（像素/数据单位，正数）。
        ox: x 平移（像素）。
        oy: y 平移（像素）。
    """

    sx: float
    sy: float
    ox: float
    oy: float

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """数据坐标 → 像素坐标（原点左上）。"""
        return (self.ox + self.sx * x, self.oy - self.sy * y)

    def to_data(self, px: float, py: float) -> tuple[float, float]:
        """像素坐标（原点左上）→ 数据坐标。"""
        return ((px - self.ox) / self.sx, (self.oy - py) / self.sy)

    def to_dict(self) -> dict[str, float]:
        """序列化为 sidecar 的 ``transform`` 字段。"""
        return {"sx": self.sx, "sy": self.sy, "ox": self.ox, "oy": self.oy}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Transform:
        """从 sidecar 的 ``transform`` 字段还原。"""
        return cls(sx=float(d["sx"]), sy=float(d["sy"]),
                   ox=float(d["ox"]), oy=float(d["oy"]))


def comsol_auto_window(
    geom_bbox: tuple[float, float, float, float],
    crop_box_px: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """auto-zoom 反演：几何 bbox + 轴框像素框 → COMSOL 显示的数据窗口。

    规则（见模块 docstring）：窗口等纵横比（xspan/crop_w == yspan/crop_h）、
    以几何 bbox 中心为中心、**最小**地完整包住几何 bbox。

    Args:
        geom_bbox: 几何包围盒 ``(x0, x1, y0, y1)``（数据单位，可由建模参数精确算出）。
        crop_box_px: 轴框像素框 ``(x0, y0, x1, y1)``（原点左上）。

    Returns:
        显示数据窗口 ``(xmin, xmax, ymin, ymax)``。

    Raises:
        ValueError: bbox/crop 退化（零尺寸）。
    """
    gx0, gx1, gy0, gy1 = geom_bbox
    cx0, cy0, cx1, cy1 = crop_box_px
    gw, gh = gx1 - gx0, gy1 - gy0
    cw, ch = cx1 - cx0, cy1 - cy0
    if gw <= 0 or gh <= 0:
        raise ValueError(f"几何 bbox 退化：{geom_bbox}")
    if cw <= 0 or ch <= 0:
        raise ValueError(f"轴框像素框退化：{crop_box_px}")
    aspect = cw / ch  # 像素框长宽比
    if gw / gh >= aspect:
        xspan, yspan = gw, gw / aspect
    else:
        xspan, yspan = gh * aspect, gh
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    return (cx - xspan / 2, cx + xspan / 2, cy - yspan / 2, cy + yspan / 2)


def pixel_transform(
    image_size: tuple[int, int],
    crop_box_px: tuple[float, float, float, float],
    extent: tuple[float, float, float, float],
) -> Transform:
    """由图像尺寸 + 轴框像素框 + 数据窗口构造 data→pixel 仿射（纯函数）。

    Args:
        image_size: ``(宽 px, 高 px)``。
        crop_box_px: 轴框像素框 ``(x0, y0, x1, y1)``（原点左上）。
        extent: 轴框对应的数据窗口 ``(xmin, xmax, ymin, ymax)``。

    Returns:
        :class:`Transform`。

    Raises:
        ValueError: 任何维度退化。
    """
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError(f"图像尺寸退化：{image_size}")
    x0, y0, x1, y1 = crop_box_px
    xmin, xmax, ymin, ymax = extent
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"轴框像素框退化：{crop_box_px}")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"数据窗口退化：{extent}")
    sx = (x1 - x0) / (xmax - xmin)
    sy = (y1 - y0) / (ymax - ymin)
    ox = x0 - sx * xmin
    oy = y0 + sy * ymax
    return Transform(sx=sx, sy=sy, ox=ox, oy=oy)


def build_sidecar(
    *,
    image_path: str | Path,
    image_size: tuple[int, int],
    crop_box_px: tuple[float, float, float, float] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    geom_bbox: tuple[float, float, float, float] | None = None,
    source: dict[str, object] | None = None,
    export: dict[str, object] | None = None,
    interior: dict[str, object] | None = None,
) -> dict[str, object]:
    """组装 sidecar 字典（纯函数；写盘走 :func:`write_sidecar`）。

    缺省规则：``crop_box_px=None`` → 轴框=整幅图（``clean`` 导出无 colorbar/标题时成立）；
    ``extent=None`` 且给了 ``geom_bbox`` → :func:`comsol_auto_window` 反演；
    两者都缺则必须显式给 ``extent``。

    Args:
        image_path: 对应 PNG 路径（记入元数据）。
        image_size: ``(宽 px, 高 px)``。
        crop_box_px: 轴框像素框；None → 整幅图。
        extent: 数据窗口；None → 用 geom_bbox 反演。
        geom_bbox: 几何包围盒（反演用）。
        source: 源信息（``model``/``plot_group``/``dataset``/``geom_bbox``…）。
        export: 导出参数（``width_mm``/``dpi``/``clean``/``brand_stripped``…）。
        interior: 轴框内部自检（``blank``/``flat_frac``，export.blank_check 产出）。

    Returns:
        可 JSON 序列化的 sidecar 字典。

    Raises:
        ValueError: extent 与 geom_bbox 双缺。
    """
    w, h = image_size
    box = tuple(crop_box_px) if crop_box_px is not None else (0.0, 0.0, float(w), float(h))
    if extent is None:
        if geom_bbox is None:
            raise ValueError("需要 extent 或 geom_bbox（反演出数据窗口）")
        extent = comsol_auto_window(geom_bbox, box)
    tr = pixel_transform((w, h), box, extent)
    return {
        "schema": SIDECAR_SCHEMA,
        "image": {"path": str(image_path), "width_px": w, "height_px": h},
        "crop_box_px": list(box),
        "extent": list(extent),
        "transform": tr.to_dict(),
        "source": dict(source or {}),
        "export": dict(export or {}),
        "interior": dict(interior or {}),
    }


def write_sidecar(
    png_path: str | Path, data: dict[str, object], *, guard_data: bool = True
) -> Path:
    """把 sidecar 写到 ``<png 去后缀>.sidecar.json``（与 PNG 同目录）。

    Args:
        png_path: 对应 PNG 路径（``out.png`` → ``out.sidecar.json``）。
        data: :func:`build_sidecar` 的输出。
        guard_data: True 时断言产物落 ``data/`` 根内（防 stray；测试置 False）。

    Returns:
        sidecar JSON 路径。

    Raises:
        ValueError: 产物路径越界（guard_data）。
    """
    png = Path(png_path)
    out = png.with_suffix(".sidecar.json")
    if guard_data:
        assert_within_data(out, what="sidecar")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_sidecar(path: str | Path) -> dict[str, object]:
    """读 sidecar JSON 并做最小契约校验（消费端入口）。

    Args:
        path: sidecar JSON 路径。

    Returns:
        sidecar 字典。

    Raises:
        ValueError: 缺契约关键字段或 schema 不符。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"sidecar schema 不符：{data.get('schema')!r}（期望 {SIDECAR_SCHEMA}）")
    for key in ("image", "crop_box_px", "extent", "transform"):
        if key not in data:
            raise ValueError(f"sidecar 缺关键字段：{key}")
    return data


def sidecar_transform(sidecar: dict[str, object]) -> Transform:
    """从 sidecar 还原 :class:`Transform`（消费端共用）。"""
    return Transform.from_dict(sidecar["transform"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 消费端：matplotlib/ultraplot 图上做数据坐标叠加（衔接③域 truthful colorbar）
# ---------------------------------------------------------------------------

def raster_extent(sidecar: dict[str, object]) -> tuple[float, float, float, float]:
    """``imshow(extent=…)`` 所需数据窗口（``(xmin, xmax, ymin, ymax)``）。"""
    xmin, xmax, ymin, ymax = sidecar["extent"]
    return (float(xmin), float(xmax), float(ymin), float(ymax))


def imshow_sidecar(
    ax: object,
    sidecar: dict[str, object],
    image_path: str | Path | None = None,
    **imshow_kwargs: object,
) -> object:
    """把 sidecar 对应的栅格按数据坐标精确铺回 Axes（``imshow(origin='upper')``）。

    约定：``extent`` 是数据窗口、图像顶边=ymax ⇒ ``origin='upper'`` 即像素 y 向下语义。

    Args:
        ax: Axes（``imshow``）。
        sidecar: :func:`read_sidecar` 的输出。
        image_path: 图像路径；None → 用 sidecar 里记的 path。
        **imshow_kwargs: 透传 ``imshow``（zorder/alpha…）。

    Returns:
        AxesImage。
    """
    import matplotlib.image as mpimg

    src = image_path or sidecar["image"]["path"]
    img = mpimg.imread(str(src))
    return ax.imshow(
        img, extent=raster_extent(sidecar), origin="upper", **imshow_kwargs
    )


def rect_corners_pixel(
    center_px: tuple[float, float],
    width_px: float,
    height_px: float,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """像素空间旋转矩形的四角（绕**中心**旋转——锚点旋转修正的数学核心，纯函数）。

    在像素空间旋转（各向同性），再逐角映射到数据坐标画 Polygon，
    绕开「matplotlib Rectangle 绕角点旋转 + 非等纵横比 ⇒ 形变」的坑。

    Args:
        center_px: 像素中心 ``(px, py)``（原点左上）。
        width_px / height_px: 像素尺寸。
        angle_deg: 逆时针旋转角（像素空间；注意像素 y 向下，画布视觉为顺时针）。

    Returns:
        四角像素坐标列表（左上起逆时针序）。

    Raises:
        ValueError: 尺寸非正。
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"矩形尺寸须为正：{width_px}x{height_px}")
    import math

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = center_px
    corners = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        rx = dx * width_px / 2
        ry = dy * height_px / 2
        corners.append((cx + rx * cos_a - ry * sin_a, cy + rx * sin_a + ry * cos_a))
    return corners


def overlay_rect_pixel(
    ax: object,
    sidecar: dict[str, object],
    center_px: tuple[float, float],
    width_px: float,
    height_px: float,
    angle_deg: float = 0.0,
    **patch_kwargs: object,
) -> object:
    """在叠加图上画「像素空间定义」的旋转矩形（数据坐标 Polygon 落笔）。

    Args:
        ax: Axes（``add_patch``）。
        sidecar: sidecar 字典（提供 pixel→data 变换）。
        center_px / width_px / height_px / angle_deg: 见 :func:`rect_corners_pixel`。
        **patch_kwargs: 透传 ``Polygon``（edgecolor/linewidth/…）。

    Returns:
        Polygon patch。
    """
    from matplotlib.patches import Polygon

    tr = sidecar_transform(sidecar)
    data_corners = [tr.to_data(px, py) for px, py in
                    rect_corners_pixel(center_px, width_px, height_px, angle_deg)]
    patch = Polygon(data_corners, fill=False, **patch_kwargs)
    ax.add_patch(patch)
    return patch


def overlay_point_pixel(
    ax: object,
    sidecar: dict[str, object],
    px: float,
    py: float,
    **scatter_kwargs: object,
) -> object:
    """在叠加图上按**像素坐标**打点（映射到数据坐标 scatter；标注锚定用）。

    Args:
        ax: Axes（``scatter``）。
        sidecar: sidecar 字典。
        px / py: 像素坐标（原点左上）。
        **scatter_kwargs: 透传 ``scatter``。

    Returns:
        PathCollection。
    """
    x, y = sidecar_transform(sidecar).to_data(px, py)
    return ax.scatter([x], [y], **scatter_kwargs)
