"""多格式导出 + PNG 预览（视觉闭环的关键产物）。

一次 ``save_figure`` 调用同时产出：
- **矢量交付件**：EPS（期刊投稿主格式）、PDF、SVG（供人工在 Illustrator/Inkscape 微调）。
- **位图交付件**：高分辨率 PNG（save_dpi，默认 600）。
- **预览件**：``<stem>_preview.png``（preview_dpi，默认 200）——这是让 Agent 能"看到"图
  的关键：EPS/SVG 无法被 Read 工具直接读取，PNG 预览填补了这一环。

字体嵌入由 style 层的 ``pdf.fonttype=42 / ps.fonttype=42`` 保证（文字不转曲、保持可编辑）；
``svg.fonttype="none"`` 使 SVG 保留真实文字。导出元数据做了确定性处理（可被 SOURCE_DATE_EPOCH 固定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt

#: 支持的矢量交付格式。
VECTOR_FORMATS = ("eps", "pdf", "svg")
#: 支持的位图交付格式。
RASTER_FORMATS = ("png",)
#: 全部可作为交付件的格式。
ALL_FORMATS = VECTOR_FORMATS + RASTER_FORMATS + ("jpg", "tif", "tiff")


@dataclass
class ExportResult:
    """一次导出的结果清单。"""

    stem: str
    out_dir: Path
    deliverables: dict[str, Path] = field(default_factory=dict)  # fmt -> path
    preview: Path | None = None
    errors: dict[str, str] = field(default_factory=dict)        # fmt -> 错误信息

    @property
    def ok(self) -> bool:
        """是否至少产出了预览且无致命错误。"""
        return self.preview is not None and not self.errors

    def report(self) -> str:
        lines = [f"export stem : {self.stem}", f"out dir     : {self.out_dir}"]
        for fmt, p in self.deliverables.items():
            size = p.stat().st_size if p.exists() else -1
            lines.append(f"  [{fmt:>4}] {p.name}  ({size/1024:.1f} KB)")
        if self.preview is not None:
            lines.append(f"  [prev] {self.preview.name}  <- Read this to inspect")
        for fmt, err in self.errors.items():
            lines.append(f"  [FAIL {fmt}] {err}")
        return "\n".join(lines)


def save_preview(
    fig: matplotlib.figure.Figure,
    path: Path,
    dpi: int = 200,
) -> Path:
    """把 Figure 栅格化为 PNG 预览（供 Agent Read）。始终成功或抛异常。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format="png")
    return path


def save_figure(
    fig: matplotlib.figure.Figure,
    out_dir: Path | str,
    stem: str,
    *,
    formats: tuple[str, ...] | list[str] = ("eps", "pdf", "svg", "png"),
    save_dpi: int = 600,
    preview_dpi: int = 200,
    metadata: dict | None = None,
    close: bool = False,
) -> ExportResult:
    """导出 Figure 到多种格式，并额外产出 PNG 预览。

    Args:
        fig: 目标 Figure（应已在 style_context 下绘制完成）。
        out_dir: 输出目录（通常是图管线目录下的 ``out/``），不存在会自动创建。
        stem: 文件主名（不含扩展名），如 ``fig1``。
        formats: 交付格式集合；未知格式会被忽略并记入 errors。
        save_dpi: 位图交付件与矢量中栅格元素的 dpi。
        preview_dpi: 预览 PNG 的 dpi（够 Agent 看清即可，无需太高）。
        metadata: 写入矢量件的元数据（如 ``{"Software": "..."}"``）。
        close: 导出后是否关闭 Figure（CLI 批处理时建议 True 释放内存）。

    Returns:
        ExportResult：含各交付件路径、预览路径、逐格式错误。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(stem=stem, out_dir=out_dir)

    save_kw: dict = {"dpi": save_dpi, "bbox_inches": "tight"}
    if metadata:
        save_kw["metadata"] = metadata

    for fmt in formats:
        f = (fmt or "").lower().strip()
        if not f:
            continue
        if f not in ALL_FORMATS:
            result.errors[f] = f"不支持的格式（可选 {', '.join(ALL_FORMATS)}）"
            continue
        path = out_dir / f"{stem}.{f}"
        try:
            # jpg/tif 用 'jpeg'/'tiff' 的 matplotlib 格式名
            mpl_fmt = {"jpg": "jpeg", "tif": "tiff", "tiff": "tiff"}.get(f, f)
            fig.savefig(path, format=mpl_fmt, **save_kw)
            result.deliverables[f] = path
        except Exception as e:  # noqa: BLE001 - 单格式失败不应中断其余导出
            result.errors[f] = repr(e)

    # 预览件：始终产出（这是视觉闭环的必需品）
    try:
        result.preview = save_preview(fig, out_dir / f"{stem}_preview.png", dpi=preview_dpi)
    except Exception as e:  # noqa: BLE001
        result.errors["preview"] = repr(e)

    if close:
        plt.close(fig)
    return result
