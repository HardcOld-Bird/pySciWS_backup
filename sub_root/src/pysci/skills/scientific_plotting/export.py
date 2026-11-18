"""导出封装：尺寸契约 / fonttype=42 / EPS+PDF+SVG+PNG 多格式 / 预览闭环。

工作流位（SKILL.md 第 5 步）：把 Figure 落成**交付件**（矢量）+ ``_preview.png``
（agent read_image 视觉验收）。职责边界：本模块只做"怎么存"，画什么是图脚本的事。

机制要点（references/export-acceptance.md 有完整验收清单）：

- **尺寸契约**：:func:`fit_figure_size` 按 :data:`~styles.JOURNAL_WIDTHS` 把图宽
  施加到 figsize（mm→inch）；audit 以**实测**成品宽度 ±2mm 复核（防 tight bbox 假绿）。
- **可编辑矢量**：保存期间以 rc_context 强制 ``pdf.fonttype=42`` / ``ps.fonttype=42`` /
  ``svg.fonttype=none``——即使调用方忘了铺契约层样式也不会交出位图字体。
- **确定性元数据**：PDF/SVG 去掉时间戳（``CreationDate``/``Date`` 置 None），
  同一图两次导出字节稳定，diff 可读。
- :func:`build_savefig_kwargs` 是**纯函数**（参数拼装），机械断言无需 matplotlib。

ADR（2026-11-17）：薄封装替代旧 figures 全集导出壳；产物落 data/ 的护栏
（:func:`pysci.paths.assert_within_data`）默认开，测试产景用 ``guard_data=False``
落到 ``tests/artifacts/``（项目测试产物区，不入库）。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import assert_within_data

#: 交付格式（矢量三件 + PNG 位图；期刊 EPS 必需、SVG 末修、PDF 投稿、PNG 预览/嵌入）。
EXPORT_FORMATS: tuple[str, ...] = ("pdf", "eps", "svg", "png")

#: 出版位图 DPI（期刊印刷口径 ≥600）。
PUBLICATION_DPI = 600

#: 预览 PNG DPI（agent 视觉验收用，兼顾可读与体积）。
PREVIEW_DPI = 150

#: 可编辑矢量的字体契约（fonttype=42：TrueType 软字体嵌入，Illustrator/Inkscape 可改字）。
_FONTTYPE_RC: dict[str, object] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def build_savefig_kwargs(
    fmt: str,
    *,
    dpi: int = PUBLICATION_DPI,
    transparent: bool = False,
) -> dict[str, object]:
    """拼装 ``savefig`` 关键字（纯函数：格式专属 dpi / 确定性元数据）。

    Args:
        fmt: 格式名（:data:`EXPORT_FORMATS` 之一）。
        dpi: 位图 DPI（仅 png 生效；矢量格式分辨率无关）。
        transparent: PNG 透明底（仅 png 生效；矢量默认自带透明）。

    Returns:
        可直接 ``fig.savefig(path, **kwargs)`` 的字典。

    Raises:
        ValueError: 格式不在支持集内。
    """
    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"不支持的导出格式：{fmt!r}（支持：{', '.join(EXPORT_FORMATS)}）")
    common: dict[str, object] = {"format": fmt}
    if fmt == "png":
        return {**common, "dpi": dpi, "transparent": transparent}
    if fmt == "pdf":
        # CreationDate=None 抑制时间戳 → 字节级确定性（两次导出可 diff）。
        return {
            **common,
            "metadata": {"Creator": "pysci", "Producer": "pysci", "CreationDate": None},
        }
    if fmt == "svg":
        return {**common, "metadata": {"Creator": "pysci", "Date": None}}
    return {**common, "metadata": {"Creator": "pysci"}}  # eps


def fit_figure_size(
    fig: object,
    *,
    width_mm: float | None = None,
    height_mm: float | None = None,
    journal: str | None = None,
    span: str | None = None,
) -> tuple[float, float]:
    """把尺寸契约施加到 Figure（鸭子类型，不 import matplotlib）。

    宽度来源优先级：显式 ``width_mm`` > ``journal``+``span`` 契约解析。
    ``height_mm`` 缺省时保持图当前高度。

    Args:
        fig: 有 ``get_size_inches``/``set_size_inches`` 的 Figure。
        width_mm: 显式图宽（mm）。
        height_mm: 显式图高（mm）；None → 保持现高。
        journal: 期刊族（``"aps"`` / ``"nature"``）。
        span: 栏宽（``"single"`` / ``"double"``）。

    Returns:
        施加后的 ``(宽 mm, 高 mm)``。

    Raises:
        ValueError: 既无 width_mm 又无 journal/span，或契约解析失败。
    """
    from pysci.skills.scientific_plotting import styles

    if width_mm is None:
        if journal is None or span is None:
            raise ValueError("需要 width_mm，或 journal+span 尺寸契约")
        width_mm = styles.resolve_width_mm(journal, span)
    cur_w_in, cur_h_in = fig.get_size_inches()
    w_in = styles.mm_to_inch(width_mm)
    h_in = styles.mm_to_inch(height_mm) if height_mm is not None else cur_h_in
    fig.set_size_inches(w_in, h_in)
    return (w_in * 25.4, h_in * 25.4)


def export_figure(
    fig: object,
    out_base: str | Path,
    *,
    formats: tuple[str, ...] = EXPORT_FORMATS,
    width_mm: float | None = None,
    height_mm: float | None = None,
    journal: str | None = None,
    span: str | None = None,
    dpi: int = PUBLICATION_DPI,
    preview: bool = True,
    preview_dpi: int = PREVIEW_DPI,
    guard_data: bool = True,
) -> tuple[Path, ...]:
    """多格式导出 + ``_preview.png`` 预览（工作流第 5 步的落盘点）。

    保存期间强制可编辑矢量字体契约（fonttype=42 / svg 文字不转路径）。

    Args:
        fig: Figure（鸭子类型：``savefig``/``get_size_inches``/``set_size_inches``）。
        out_base: 产物基路径（不含后缀），如 ``.../article/figures/fig1_demo/out/fig1``。
        formats: 交付格式集（默认矢量三件 + PNG）。
        width_mm / height_mm / journal / span: 尺寸契约（见 :func:`fit_figure_size`）。
        dpi: 出版位图 DPI。
        preview: 是否额外产 ``<base>_preview.png`` 供 agent 视觉验收。
        preview_dpi: 预览 DPI。
        guard_data: True 时断言产物落 ``data/`` 根内（防 stray；测试置 False）。

    Returns:
        写出文件的路径元组（含预览 PNG）。

    Raises:
        ValueError: 无尺寸信息、产物路径越界（guard_data）或格式不支持。
    """
    import matplotlib  # 懒导入

    if width_mm is None and (journal is None or span is None):
        raise ValueError("需要 width_mm，或 journal+span 尺寸契约")
    fit_figure_size(
        fig, width_mm=width_mm, height_mm=height_mm, journal=journal, span=span
    )
    out_base = Path(out_base)
    if guard_data:
        assert_within_data(out_base, what="图产物")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with matplotlib.rc_context(_FONTTYPE_RC):
        for fmt in formats:
            target = out_base.with_suffix(f".{fmt.lower()}")
            fig.savefig(target, **build_savefig_kwargs(fmt, dpi=dpi))
            written.append(target)
        if preview:
            pv = out_base.with_name(out_base.name + "_preview.png")
            fig.savefig(pv, **build_savefig_kwargs("png", dpi=preview_dpi))
            written.append(pv)
    return tuple(written)
