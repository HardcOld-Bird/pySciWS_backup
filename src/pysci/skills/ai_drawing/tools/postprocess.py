"""Pillow 图像后处理：简单调整、拼合、主色板抽取（无需 GPU）。

承载用户所说的"简单调整"能力——对 AI 生成图（或任意位图）做裁剪/缩放/旋转/格式与
色彩模式转换/加边，以及多图拼合（contact sheet，供 Agent 一眼看全画廊）与主色板抽取
（供 :mod:`bridge` 把审美范本的配色迁移到 scientific_plotting 复现管线）。

所有函数以**路径进、路径出**（便于 CLI 编排），返回写出的目标路径。Pillow 已在
主依赖中显式声明，本模块不引入任何重型/联网依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

#: 支持的输出格式（由目标后缀推断）→ Pillow format 名。
_FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _resolve_format(dst: Path, explicit: str | None = None) -> str:
    """由目标后缀（或显式指定）推断 Pillow 保存格式。"""
    if explicit:
        return explicit.upper()
    fmt = _FORMAT_BY_SUFFIX.get(dst.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"无法从后缀 {dst.suffix!r} 推断格式；支持 {sorted(_FORMAT_BY_SUFFIX)}"
            f"，或用 --format 显式指定"
        )
    return fmt


def open_image(src: str | Path, *, mode: str | None = None) -> Image.Image:
    """打开图像（可选转色彩模式）。"""
    img = Image.open(src)
    if mode:
        img = img.convert(mode)
    return img


def adjust_image(
    src: str | Path,
    dst: str | Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
    rotate: float | None = None,
    mode: str | None = None,
    pad: tuple[int, int] | None = None,
    pad_color: tuple[int, ...] = (255, 255, 255),
    quality: int = 95,
    fmt: str | None = None,
) -> Path:
    """对单张图做一串简单调整并写出。

    调整顺序（固定，符合直觉）：crop → rotate → resize → pad → 色彩模式/格式转换。

    Args:
        src: 源图路径。
        dst: 目标路径（后缀决定格式，除非给了 ``fmt``）。
        crop: 像素裁剪框 ``(x0, y0, x1, y1)``。
        resize: 目标尺寸 ``(w, h)``；某一边为 0 时按原图长宽比自动计算。
        rotate: 逆时针旋转角度（``expand=True``，不裁掉旋转后超出的部分）。
        mode: 色彩模式转换（``"RGB"`` / ``"RGBA"`` / ``"L"`` 等）。
        pad: 加边到 ``(w, h)`` 画布并居中，空白处填 ``pad_color``。
        pad_color: 加边填充色（RGB/RGBA 元组）。
        quality: JPEG/WEBP 质量（1-100）。
        fmt: 显式输出格式（覆盖后缀推断）。

    Returns:
        写出的目标路径。
    """
    src = Path(src)
    dst = Path(dst)
    img = Image.open(src)

    if crop is not None:
        x0, y0, x1, y1 = crop
        img = img.crop((x0, y0, x1, y1))
    if rotate:
        img = img.rotate(float(rotate), expand=True)
    if resize is not None:
        w, h = int(resize[0]), int(resize[1])
        if w == 0 or h == 0:
            ow, oh = img.size
            if w == 0 and h == 0:
                raise ValueError("resize 宽高不能同时为 0")
            ratio = ow / oh
            w = int(round(h * ratio)) if w == 0 else w
            h = int(round(w / ratio)) if h == 0 else h
        img = img.resize((w, h), Image.LANCZOS)
    if pad is not None:
        pw, ph = int(pad[0]), int(pad[1])
        canvas = Image.new(
            img.mode if img.mode in ("RGB", "RGBA") else "RGB", (pw, ph), pad_color
        )
        ox = max(0, (pw - img.width) // 2)
        oy = max(0, (ph - img.height) // 2)
        canvas.paste(img, (ox, oy))
        img = canvas
    if mode:
        img = img.convert(mode)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_fmt = _resolve_format(dst, fmt)
    save_kwargs: dict[str, Any] = {}
    if out_fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = int(quality)
        if out_fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")  # JPEG 不支持 alpha
    img.save(dst, format=out_fmt, **save_kwargs)
    return dst


def extract_palette(
    src: str | Path,
    n: int = 6,
    *,
    thumb: int = 160,
) -> list[str]:
    """抽取图像主色板（按出现频率降序），返回 hex 列表（如 ``["#1a2b3c", ...]``）。

    用中位切色彩量化（median-cut）在缩略图上取主色——足以刻画一张审美范本的配色基调，
    供 :mod:`bridge` 迁移到科研绘图。纯 CPU、离线、无需 GPU。

    Args:
        src: 源图路径。
        n: 主色数量（1-16）。
        thumb: 量化前缩略到的最大边长（提速，不影响主色判断）。
    """
    n = max(1, min(int(n), 16))
    img = Image.open(src).convert("RGB")
    img.thumbnail((thumb, thumb))
    quant = img.quantize(colors=n, method=Image.MEDIANCUT)
    palette = quant.getpalette() or []
    counts = quant.getcolors(n) or []
    # getcolors 返回 [(count, palette_index), ...]；按 count 降序即主色优先。
    counts.sort(key=lambda c: c[0], reverse=True)
    hexes: list[str] = []
    for _, idx in counts:
        rgb = palette[idx * 3 : idx * 3 + 3]
        if len(rgb) < 3:
            continue
        hexes.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return hexes[:n]


def contact_sheet(
    images: list[str | Path],
    dst: str | Path,
    *,
    cols: int = 4,
    thumb: int = 256,
    pad: int = 8,
    label: bool = True,
    bg: tuple[int, int, int] = (245, 245, 245),
) -> Path:
    """把多张图拼成一张 contact sheet（缩略图网格 + 文件名标签），供 Agent 一眼看全。

    Args:
        images: 源图路径列表（按序排列）。
        dst: 输出路径（建议 .png）。
        cols: 每行列数。
        thumb: 单张缩略图最大边长。
        pad: 单元格间距/留白（像素）。
        label: 是否在每张下方标注文件名。
        bg: 背景色。

    Returns:
        写出的 contact sheet 路径。空列表时抛 ValueError。
    """
    paths = [Path(p) for p in images if Path(p).is_file()]
    if not paths:
        raise ValueError("contact_sheet：没有可用的源图")

    cols = max(1, min(cols, len(paths)))
    rows = (len(paths) + cols - 1) // cols
    label_h = 18 if label else 0
    cell_w = thumb + pad * 2
    cell_h = thumb + pad * 2 + label_h
    sheet_w = cols * cell_w + pad
    sheet_h = rows * cell_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(sheet)

    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001 - 坏图跳过，不整张失败
            continue
        im.thumbnail((thumb, thumb))
        r, c = divmod(i, cols)
        x0 = pad + c * cell_w + (thumb - im.width) // 2
        y0 = pad + r * cell_h + (thumb - im.height) // 2
        sheet.paste(im, (x0, y0))
        if label:
            draw.text(
                (pad + c * cell_w + 4, pad + r * cell_h + thumb + pad + 2),
                p.name[:28],
                fill=(40, 40, 40),
            )

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, format=_resolve_format(dst, "PNG"))
    return dst
