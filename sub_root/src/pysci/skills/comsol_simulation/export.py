"""出版级 COMSOL 出图：Image 导出节点像素级参数拼装 + 品牌剥离 + 空白自检 + sidecar。

与③域导出契约同风格（尺寸/DPI/背景/品牌剥离的**契约化**）：本模块是「COMSOL 直出栅格」
的控制面——mph ``Model.export()`` 只裸跑导出节点，像素级控制全部在这里拼装。

**属性名真源**（mph 官方 demo `create_capacitor.py` 的 Image 导出节点全量属性 +
COMSOL 6.4 手册「Image (Export)」+ 旧项目 2026-09 实测；速查表见技能
references/java-mph-pitfalls.md）：

- 节点挂在 ``model/'exports'``，类型 ``"Image"``（mph：``exports.create('Image', name=…)``）；
- ``sourceobject=<绘图组 tag>``、``filename``/``pngfilename``=输出路径；
- 尺寸 ``size="manualweb"``（候选 "manual"）+ ``unit="px"``（候选 "pixel"）+
  ``width``/``height``（像素）+ **``resolution``=DPI** + ``lockratio="off"``；
- 品牌剥离 = **先开** ``options1d/2d/3d="on"``（optional components 总开关，logo*d
  仅在其开启时受控），**再关** ``logo1d/2d/3d="off"``——顺序不可换；
- clean（下游排版用）= ``title1d/2d/3d="off"`` + ``legend1d/2d/3d="off"``（导出节点侧）
  + 子特征 ``colorlegend="off"`` + 绘图组 ``titletype="none"``（候选名防御式首中）；
- ``background="color"`` + ``customcolor=[1,1,1]``（白底）/ ``antialias="on"``。

防御式「候选名首中」（:func:`build_export_plan` 出计划、驱动逐条试设）：不同绘图组
类型/版本属性名有出入，首中即停并记录实际生效名——拼装是纯函数，机械断言恒跑。

ADR（2026-11-17）：品牌剥离只走原生 ``logo*d=off``（imgcheck 像素擦除**不建**，用户裁决）；
空白自检保留（数据集未绑定 → 空白 PNG 的防复发），结果写 sidecar ``interior``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pysci.paths import assert_within_data
from pysci.skills.comsol_simulation import sidecar as sc

#: 出版位图 DPI（期刊印刷口径 ≥600，与③域 export.PUBLICATION_DPI 一致）。
PUBLICATION_DPI = 600

#: 预览/目检 PNG DPI。
PREVIEW_DPI = 150

#: 1 inch = 25.4 mm（mm→px 换算）。
_MM_PER_INCH = 25.4

#: 空白判定阈值：轴框内部与主色一致的像素占比 ≥ 此值即 blank。
BLANK_FLAT_FRACTION = 0.995


@dataclass(frozen=True)
class PropOp:
    """一条属性写入计划：``候选名`` 依次试设，**首中即停**。

    Attributes:
        target: 目标对象（``"export"`` 导出节点 / ``"plot_group"`` / ``"features"`` 子特征）。
        candidates: 候选属性名（不同绘图组类型/版本属性名有出入）。
        value: 目标值（一律字符串，绕开 JPype 数值重载歧义）。
    """

    target: str
    candidates: tuple[str, ...]
    value: str


@dataclass
class ExportPlan:
    """出图计划（纯数据；驱动 :func:`apply_plan` 执行）。

    Attributes:
        ops: 有序属性写入计划（顺序有意义：options*d 先于 logo*d）。
        warnings: 拼装期告警（如 2D extent 不受支持）。
    """

    ops: list[PropOp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pixels_from_size(
    *,
    width_mm: float | None = None,
    height_mm: float | None = None,
    dpi: int = PUBLICATION_DPI,
    aspect: float | None = None,
) -> tuple[int, int]:
    """尺寸契约 → 像素宽高（纯函数；``px = round(mm / 25.4 * dpi)``）。

    Args:
        width_mm: 图宽（mm）。
        height_mm: 图高（mm）。
        dpi: 出版 DPI。
        aspect: 宽/高比——只给单边时补另一边（如 ``aspect=width/height``）。

    Returns:
        ``(宽 px, 高 px)``。

    Raises:
        ValueError: 两边都缺、dpi 非正、换算结果 ≤ 0。
    """
    if width_mm is None and height_mm is None:
        raise ValueError("需要 width_mm 与 height_mm 至少其一")
    if dpi <= 0:
        raise ValueError(f"dpi 须为正：{dpi}")
    if width_mm is None or height_mm is None:
        if aspect is None or aspect <= 0:
            raise ValueError("只给单边尺寸时须提供正 aspect（宽/高比）补另一边")
        if width_mm is None:
            width_mm = height_mm * aspect
        else:
            height_mm = width_mm / aspect
    w_px = round(width_mm / _MM_PER_INCH * dpi)
    h_px = round(height_mm / _MM_PER_INCH * dpi)
    if w_px <= 0 or h_px <= 0:
        raise ValueError(f"换算像素尺寸退化：{width_mm}mm x {height_mm}mm @ {dpi}dpi")
    return (w_px, h_px)


def build_image_ops(
    plot_group: str,
    out_png: str | Path,
    width_px: int,
    height_px: int,
    *,
    dpi: int = PUBLICATION_DPI,
    background: str = "color",
) -> list[PropOp]:
    """Image 导出节点基础属性计划（纯函数；属性名见模块 docstring 真源）。

    Args:
        plot_group: 来源绘图组 tag（如 ``"pg1"``）。
        out_png: 输出 PNG 路径（内部转绝对路径）。
        width_px / height_px: 像素尺寸。
        dpi: 位图分辨率（写入 ``resolution``）。
        background: 背景模式（``"color"`` 白底 / ``"transparent"``）。

    Returns:
        ``PropOp`` 列表。
    """
    return [
        PropOp("export", ("sourcetype",), "plotgroup"),
        PropOp("export", ("sourceobject",), str(plot_group)),
        # mph 1.4 Model.export 用 filename；COMSOL 6.4 手册同列 pngfilename——首中兼容
        PropOp("export", ("filename", "pngfilename"), str(Path(out_png).resolve())),
        # 尺寸枚举真源 = mph 官方 demo（manualweb/px）；宽度/高度即像素
        PropOp("export", ("size",), "manualweb"),
        PropOp("export", ("unit",), "px"),
        PropOp("export", ("width",), str(width_px)),
        PropOp("export", ("height",), str(height_px)),
        PropOp("export", ("resolution",), str(dpi)),
        PropOp("export", ("lockratio",), "off"),
        PropOp("export", ("background",), background),
        PropOp("export", ("antialias",), "on"),
    ]


def build_brand_ops(*, strip: bool = True) -> list[PropOp]:
    """品牌剥离计划（纯函数）——**先开 options*d 再关 logo*d**，顺序由列表顺序保证。

    ``options1d/2d/3d`` 与 ``logo1d/2d/3d`` 是**各自独立**的属性（不是候选名），
    三个维度都要设（mph 官方 demo 全量属性表确证）。

    Args:
        strip: True → ``options*d=on`` + ``logo*d=off``；False → 空计划（保留品牌）。

    Returns:
        ``PropOp`` 列表（strip=False 时为空）。
    """
    if not strip:
        return []
    return [
        PropOp("export", ("options1d",), "on"),
        PropOp("export", ("options2d",), "on"),
        PropOp("export", ("options3d",), "on"),
        PropOp("export", ("logo1d",), "off"),
        PropOp("export", ("logo2d",), "off"),
        PropOp("export", ("logo3d",), "off"),
    ]


def build_clean_ops() -> list[PropOp]:
    """clean 计划（纯函数）：隐藏 title/legend（导出节点侧）+ colorbar（子特征侧）。

    Returns:
        ``PropOp`` 列表。
    """
    return [
        PropOp("export", ("title1d",), "off"),
        PropOp("export", ("title2d",), "off"),
        PropOp("export", ("title3d",), "off"),
        PropOp("export", ("legend1d",), "off"),
        PropOp("export", ("legend2d",), "off"),
        PropOp("export", ("legend3d",), "off"),
        PropOp("features", ("colorlegend", "legend", "showcolorbar"), "off"),
        PropOp("plot_group", ("titletype",), "none"),
    ]


def build_color_range_ops(vmin: float, vmax: float) -> list[PropOp]:
    """手动色标范围计划（纯函数）——钉死色标范围，防逐 case 自动漂移。

    Args:
        vmin / vmax: 色标下/上限（对称发散传 ∓MAX）。

    Returns:
        ``PropOp`` 列表（含 ``rangecoloractive=on``）。
    """
    return [
        PropOp("features", ("rangecoloractive",), "on"),
        PropOp("features", ("rangecolormin",), str(vmin)),
        PropOp("features", ("rangecolormax",), str(vmax)),
    ]


def build_polar_rmax_ops(rmax: float, rmin: float = 0.0) -> list[PropOp]:
    """极坐标极径范围计划（纯函数）——多 case 统一 rmax 才能比相对能量。

    Args:
        rmax: 极径上限。
        rmin: 极径下限（默认 0）。

    Returns:
        ``PropOp`` 列表。
    """
    return [
        PropOp("plot_group", ("axislimits",), "on"),
        PropOp("plot_group", ("rmin",), str(rmin)),
        PropOp("plot_group", ("rmax",), str(rmax)),
    ]


def build_export_plan(
    plot_group: str,
    out_png: str | Path,
    width_px: int,
    height_px: int,
    *,
    dpi: int = PUBLICATION_DPI,
    background: str = "color",
    clean: bool = True,
    strip_brand: bool | None = None,
    color_range: tuple[float, float] | None = None,
    polar_rmax: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> ExportPlan:
    """总装出图计划（纯函数）：基础 + clean + 品牌 + 可选色标/极径。

    Args:
        plot_group / out_png / width_px / height_px: 见 :func:`build_image_ops`。
        dpi / background: 分辨率与背景（见 :func:`build_image_ops`）。
        clean: 隐藏 title/legend/colorbar（下游排版用）。
        strip_brand: 品牌剥离开关；None → 跟随 clean（clean=True 必然出无品牌图）。
        color_range: ``(vmin, vmax)`` 手动色标范围。
        polar_rmax: 极坐标极径上限。
        extent: 数据窗口请求——**2D 绘图组无轴限属性**（Java API 不暴露），仅记入
            sidecar 元数据并给告警；精确映射走 :func:`sidecar.comsol_auto_window`。

    Returns:
        :class:`ExportPlan`。
    """
    plan = ExportPlan(
        ops=build_image_ops(
            plot_group, out_png, width_px, height_px, dpi=dpi, background=background
        )
    )
    if clean:
        plan.ops += build_clean_ops()
    do_strip = clean if strip_brand is None else bool(strip_brand)
    plan.ops += build_brand_ops(strip=do_strip)
    if color_range is not None:
        plan.ops += build_color_range_ops(*color_range)
    if polar_rmax is not None:
        plan.ops += build_polar_rmax_ops(polar_rmax)
    if extent is not None:
        plan.warnings.append(
            "extent 仅部分生效：PlotGroup2D 无 xmin/xmax 轴限属性"
            "（旧项目逐属性确证）——精确 pixel↔data 映射请用 geom_bbox 反演"
        )
    return plan


# ---------------------------------------------------------------------------
# 驱动层（mph Node API：property(name, value) 写属性；鸭子类型 model）
# ---------------------------------------------------------------------------


def _set_first_ok(node: object, candidates: tuple[str, ...], value: str) -> str | None:
    """候选属性名依次试设，**首中即停**（驱动核心）。

    Args:
        node: mph ``Node``（``property(str, value)`` 写属性；属性不存在即抛）。
        candidates: 候选属性名。
        value: 目标值。

    Returns:
        实际生效的属性名；全部失败返回 None。
    """
    for prop in candidates:
        try:
            node.property(prop, value)
            return prop
        except Exception:  # noqa: BLE001 - Java 端属性不存在即异常，属预期探测
            continue
    return None


def apply_plan(
    plan: ExportPlan, *, export_node: object, plot_group_node: object,
    feature_nodes: list[object] | None = None,
) -> list[str]:
    """执行出图计划（驱动；候选名首中，失败汇总不中断）。

    Args:
        plan: :func:`build_export_plan` 的输出。
        export_node: Image 导出节点。
        plot_group_node: 绘图组节点。
        feature_nodes: 绘图组子特征节点（Surface/PolarPlot 等；``features`` 目标遍历）。

    Returns:
        实际生效的 ``"target.prop=value"`` 记录列表。
    """
    applied: list[str] = []
    for op in plan.ops:
        if op.target == "export":
            hit = _set_first_ok(export_node, op.candidates, op.value)
        elif op.target == "plot_group":
            hit = _set_first_ok(plot_group_node, op.candidates, op.value)
        else:  # features
            hit = None
            for feat in feature_nodes or []:
                hit = _set_first_ok(feat, op.candidates, op.value)
                if hit is not None:
                    break
        if hit is not None:
            applied.append(f"{op.target}.{hit}={op.value}")
    return applied


def blank_check(
    png_path: str | Path,
    *,
    crop_box_px: tuple[float, float, float, float] | None = None,
    central: float = 0.6,
    sample_step: int = 4,
) -> dict[str, object]:
    """绘图区空白自检（导出后必跑；防「数据集未绑定 → 空白 PNG」复发）。

    只采样**中心 ``central`` 比例区域**（clean 导出下即绘图区）——轴框/刻度/标签都在
    边缘，混入采样会把「只剩轴框」误判为有内容（2026-11-17 实测教训）。

    Args:
        png_path: 导出的 PNG。
        crop_box_px: 轴框像素框（原点左上）；None → 整幅图。
        central: 中心采样区边长比例（0~1）。
        sample_step: 采样步长（像素；越大越快）。

    Returns:
        ``{"blank": bool, "flat_frac": float}``（flat_frac = 中心区与主色一致的采样占比）。

    Raises:
        FileNotFoundError: PNG 不存在（导出失败）。
    """
    import matplotlib.image as mpimg

    path = Path(png_path)
    if not path.exists():
        raise FileNotFoundError(f"导出 PNG 不存在：{path}")
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop_box_px or (0, 0, w, h)
    mx = (x1 - x0) * (1 - central) / 2
    my = (y1 - y0) * (1 - central) / 2
    samples = [
        tuple(float(c) for c in img[py, px])
        for py in range(int(y0 + my), int(y1 - my), max(1, sample_step))
        for px in range(int(x0 + mx), int(x1 - mx), max(1, sample_step))
    ]
    flat = _flat_fraction(samples)
    return {"blank": flat >= BLANK_FLAT_FRACTION, "flat_frac": flat}


def _flat_fraction(samples: list[tuple[float, ...]], tol: float = 1e-3) -> float:
    """与主色一致的采样占比（纯函数；blank 判定的数学核心）。

    Args:
        samples: 采样像素（各通道 0..1 或 0..255，元组/列表）。
        tol: 同色容差（逐通道）。

    Returns:
        主色占比（0..1）；空采样返回 1.0（视作 blank，宁枉勿纵）。
    """
    if not samples:
        return 1.0
    counts: dict[tuple[float, ...], int] = {}
    for s in samples:
        key = tuple(round(c / tol) * tol for c in s)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(samples)


def export_image(
    model: object,
    plot_group: str,
    out_png: str | Path,
    *,
    width_mm: float | None = None,
    height_mm: float | None = None,
    dpi: int = PUBLICATION_DPI,
    clean: bool = True,
    strip_brand: bool | None = None,
    color_range: tuple[float, float] | None = None,
    polar_rmax: float | None = None,
    geom_bbox: tuple[float, float, float, float] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    guard_data: bool = True,
) -> dict[str, object]:
    """出版级出图一键编排：拼装→设参→导出→空白自检→sidecar（驱动，需活体 model）。

    Args:
        model: mph ``Model``（鸭子类型：``exports()``/``result()``/``export()``）。
        plot_group: 来源绘图组 tag。
        out_png: 输出 PNG 路径。
        width_mm / height_mm / dpi: 尺寸契约（见 :func:`pixels_from_size`；
            两者至少其一，单边须配 aspect——此处按整幅像素宽高直接给全）。
        clean / strip_brand / color_range / polar_rmax / extent: 见 :func:`build_export_plan`。
        geom_bbox: 几何包围盒 → sidecar ``extent`` 反演（auto-zoom 逆模型）。
        extent: 显式数据窗口（优先于 geom_bbox 反演）。
        guard_data: True 时断言产物落 ``data/`` 根内（防 stray；测试置 False）。

    Returns:
        ``{"png": Path, "sidecar": Path, "applied": [...], "warnings": [...]}``。

    Raises:
        ValueError: 尺寸信息缺失或产物路径越界。
        FileNotFoundError: 导出后 PNG 未生成。
    """
    out = Path(out_png)
    if guard_data:
        assert_within_data(out, what="图产物")
    if width_mm is None and height_mm is None:
        raise ValueError("需要 width_mm 与 height_mm 至少其一（尺寸契约）")
    pg_node = _find_child(model / "plots", plot_group)
    if pg_node is None:
        raise ValueError(f"绘图组不存在：{plot_group}（model.plots() 有全部名字）")
    tag_member = getattr(pg_node, "tag", None)
    pg_tag = str(tag_member() if callable(tag_member) else (tag_member or plot_group))
    w_px, h_px = pixels_from_size(width_mm=width_mm, height_mm=height_mm, dpi=dpi)
    # sourceobject 记绘图组 **tag**（COMSOL 6.4 手册口径；显示名经 _find_child 解析）
    plan = build_export_plan(
        pg_tag, out, w_px, h_px, dpi=dpi, clean=clean, strip_brand=strip_brand,
        color_range=color_range, polar_rmax=polar_rmax, extent=extent,
    )

    exports = model / "exports"
    tag = f"img_{out.stem}"
    node = _find_child(exports, tag) or exports.create("Image", name=tag)

    applied = apply_plan(
        plan, export_node=node, plot_group_node=pg_node,
        feature_nodes=_child_features(pg_node),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    model.export(node)
    if not out.exists():
        raise FileNotFoundError(f"导出未产出 PNG：{out}")

    interior = blank_check(out, crop_box_px=None)
    if interior["blank"]:
        plan.warnings.append(
            "导出 PNG 轴框内部空白——先查绘图组数据集绑定"
            "（重求解后须 plotgroup.rebind 重绑，见 java-mph-pitfalls §3）"
        )
    data = sc.build_sidecar(
        image_path=out,
        image_size=(w_px, h_px),
        extent=extent,
        geom_bbox=geom_bbox,
        source={"plot_group": plot_group},
        export={
            "width_mm": width_mm, "height_mm": height_mm, "dpi": dpi,
            "clean": clean, "brand_stripped": (clean if strip_brand is None
                                               else bool(strip_brand)),
        },
        interior=interior,
    )
    side_path = sc.write_sidecar(out, data, guard_data=guard_data)
    return {
        "png": out, "sidecar": side_path,
        "applied": applied, "warnings": plan.warnings,
    }


def _find_child(group: object, name_or_tag: str) -> object | None:
    """在 mph 分组节点（``model/'plots'`` 等）的子节点里按 name/tag 查找（鸭子类型）。

    Args:
        group: 分组节点（可迭代出子节点；子节点有 ``name()``/``tag()``）。
        name_or_tag: 绘图组/导出节点的显示名或 tag。

    Returns:
        匹配的子节点；无匹配返回 None。
    """
    for child in group:
        for getter in ("name", "tag"):
            member = getattr(child, getter, None)
            value = member() if callable(member) else member
            if value is not None and str(value) == name_or_tag:
                return child
    return None


def _child_features(pg_node: object) -> list[object]:
    """绘图组子特征节点列表（Surface/PolarPlot 等；mph ``children()``）。"""
    fn = getattr(pg_node, "children", None)
    return list(fn()) if callable(fn) else []
