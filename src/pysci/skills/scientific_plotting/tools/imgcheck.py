"""不依赖管线的图像体检（backlog 20261015-204912 §4.4）。

对**非 scaffold 产物**（散落脚本产出的 PNG——`figures audit` 在那类目录上恒 N/A）给出可执行
的工具核验面：尺寸 / 色域占用 / 饱和像素占比 / **边缘水印·logo 检测**。审查四步里的「步骤 2
工具核验」由它承接（I.i3 校准审查 §4.4 实测：此前该能力只能靠自制 PIL 脚本，不可复用）。

与 comsol-simulation 的分工（**故意不合并**）：
- 那边 ``postprocess.find_brand_box``/``strip_brand`` 是**导出时的精确抹除器**——只认 COMSOL
  那枚蓝黄 logo，命中即**改像素**；
- 本模块是**只读筛查器**——扫边缘带内任何「非底色 + 高饱和」的标记块，另标注「蓝黄成对」
  （外部渲染器 logo 的典型签名）。判据与用途不同：一读一写，合并会让筛查器耦合到导出路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: 边缘带宽度（相对短边比例；下限 4px）——水印/logo 通常落在图边距里。
BAND_FRAC: float = 0.02
BAND_MIN_PX: int = 4

#: 「非中性（有彩色）」与「高饱和」的色度阈值（max-min 通道差，0–255）。
CHROMA_NEUTRAL: float = 24.0
CHROMA_SATURATED: float = 80.0

#: 判定「偏离底色」的通道差阈值。
BG_DIFF: float = 40.0

#: 边缘标记的最小连通面积（像素）——滤掉单像素噪点/抗锯齿毛刺。
MIN_MARK_AREA: int = 4

#: 边缘标记的**最大**面积占比（相对整图）——水印/logo 是小块；铺满画布的数据区若恰好触及
#: 边缘带，会连成一个巨大连通域，这条把它排除（防对「满幅图」误报）。
MAX_MARK_FRAC: float = 0.02

#: 品牌蓝/黄判据（与 comsol 侧同源的经验阈值，探测 COMSOL 类 logo 的签名）。
_BRAND_BLUE = lambda r, g, b: (
    (b > g) & (g > r) & (b >= 140) & ((b - r) >= 55) & ((g - r) >= 15)
)  # noqa: E731
_BRAND_YELLOW = lambda r, g, b: (
    (r >= 175) & (g >= 175) & (b <= 150) & (np.abs(r - g) <= 70)
)  # noqa: E731


def load_rgb(path: str | Path) -> np.ndarray | None:
    """读 PNG/JPG 为 ``(H, W, 3)`` uint8 数组；读不到返回 ``None``（体检面不抛）。"""
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:  # noqa: BLE001 - 只读体检绝不因坏文件抛断
        return None


def chroma(arr: np.ndarray) -> np.ndarray:
    """逐像素色度 = max(R,G,B) − min(R,G,B)（公开：occlusion.py 复用，杜绝第二实现）。"""
    f = arr.astype(np.int16)
    return (f.max(axis=-1) - f.min(axis=-1)).astype(float)


def _hue(arr: np.ndarray) -> np.ndarray:
    """逐像素色相（度，0–360）；无彩色处返回 0（调用方须用色度掩膜过滤）。"""
    r, g, b = (arr[..., i].astype(float) for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = np.where(mx - mn == 0, 1.0, mx - mn)
    hue = np.select(
        [mx == r, mx == g],
        [((g - b) / d) % 6.0, ((b - r) / d) + 2.0],
        default=((r - g) / d) + 4.0,
    )
    return (hue * 60.0) % 360.0


def border_band_marks(
    arr: np.ndarray, *, band: int | None = None
) -> list[dict[str, Any]]:
    """扫**边缘带**里的「非底色 + 高饱和」标记块，返回各块读数（面积/bbox/是否蓝黄成对）。

    两步走，为的是**不把跨带界的标记切碎**：
    1. 全图求「标记与否」（偏离底色 + 有色度），按连通域切分（:func:`scipy.ndimage.label`）；
    2. 只保留**触及边缘带**且面积落在 ``[MIN_MARK_AREA, MAX_MARK_FRAC·H·W]`` 的连通域——
       于是跨带界的 logo 报**完整 bbox 与完整色像素**（成对判据才准），而铺满画布的数据区
       因面积超限被排除。

    底色取边缘带内**低色度像素的中位数**（自适应白底/黑底/浅色底）。
    """
    from scipy import ndimage  # noqa: PLC0415 - 仅体检面需要，延迟到调用点

    h, w = arr.shape[0], arr.shape[1]
    if band is None:
        band = max(BAND_MIN_PX, int(min(w, h) * BAND_FRAC))
    band = int(min(band, max(1, min(h, w) // 2)))
    band_mask = np.zeros((h, w), dtype=bool)
    band_mask[:band, :] = band_mask[-band:, :] = True
    band_mask[:, :band] = band_mask[:, -band:] = True

    ch = chroma(arr)
    neutral = band_mask & (ch <= CHROMA_NEUTRAL)
    bg = (
        np.median(arr[neutral].astype(float), axis=0)
        if int(neutral.sum()) >= 10
        else np.array([255.0, 255.0, 255.0])
    )
    diff = np.abs(arr.astype(float) - bg).max(axis=-1)
    hit = (diff > BG_DIFF) & (ch > BG_DIFF)

    labels, n = ndimage.label(hit)
    touching = set(int(v) for v in np.unique(labels[band_mask])) - {0}
    max_area = int(MAX_MARK_FRAC * h * w)
    blue = _BRAND_BLUE(arr[..., 0], arr[..., 1], arr[..., 2])
    yellow = _BRAND_YELLOW(arr[..., 0], arr[..., 1], arr[..., 2])

    out: list[dict[str, Any]] = []
    for i in sorted(touching):
        sel = labels == i
        area = int(sel.sum())
        if area < MIN_MARK_AREA or area > max_area:
            continue
        ys, xs = np.where(sel)
        out.append(
            {
                "area": area,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "brand_like": bool((blue & sel).any() and (yellow & sel).any()),
                "bg": [int(v) for v in np.round(bg)],
            }
        )
    out.sort(key=lambda d: -d["area"])
    return out


def read_dpi(path: str | Path) -> float | None:
    """读 PNG/JPG 的 **dpi 元数据**（PNG 的 pHYs → ``info["dpi"]``）；读不到回 ``None``。

    用于把像素尺寸换算成**等效物理尺寸**（backlog 20261015-221750）：审查员核验
    「生产者自述 mm 尺寸 vs 成品实测」时不必再反解版面常量。缺元数据/坏文件一律 ``None``
    （体检面不抛）。
    """
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
    except Exception:  # noqa: BLE001 - 只读体检绝不因坏文件抛断
        return None
    if isinstance(dpi, (tuple, list)) and dpi:
        dpi = dpi[0]
    try:
        value = float(dpi)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _mm(px: int, dpi: float | None) -> float | None:
    """像素 → 毫米（``px / dpi * 25.4``）；无 dpi 回 ``None``。"""
    if not dpi:
        return None
    return round(px / float(dpi) * 25.4, 1)


def image_health(
    arr: np.ndarray,
    *,
    band: int | None = None,
    hue_bins: int = 24,
    dpi: float | None = None,
) -> dict[str, Any]:
    """图像体检读数（纯函数）：尺寸（像素 + 等效 mm）/ 色域占用 / 饱和像素占比 / 边缘标记。

    ``dpi`` 缺省 ``None``：给了就一并算 ``size_mm``（换算口径见 :func:`_mm`）。
    """
    h, w = arr.shape[0], arr.shape[1]
    ch = chroma(arr)
    colored = ch > CHROMA_NEUTRAL
    colored_ratio = float(colored.mean())
    saturated_ratio = float((ch > CHROMA_SATURATED).mean())
    if colored.any():
        bins = (_hue(arr)[colored] // (360.0 / hue_bins)).astype(int)
        hue_bins_used = int(np.unique(np.clip(bins, 0, hue_bins - 1)).size)
    else:
        hue_bins_used = 0
    marks = border_band_marks(arr, band=band)
    w_mm, h_mm = _mm(w, dpi), _mm(h, dpi)
    return {
        "size_px": [w, h],
        "dpi": dpi,
        "size_mm": [w_mm, h_mm] if w_mm is not None and h_mm is not None else None,
        "colored_ratio": round(colored_ratio, 4),
        "hue_bins_used": hue_bins_used,
        "hue_bins_total": hue_bins,
        "saturated_ratio": round(saturated_ratio, 4),
        "edge_marks": marks,
    }


def findings(health: dict[str, Any]) -> list[str]:
    """把读数转成**可执行判据**文本（WARN 级；无标记则空）。"""
    out: list[str] = []
    for m in health["edge_marks"]:
        kind = "疑似外部渲染器 logo（蓝黄成对）" if m["brand_like"] else "边缘标记块"
        out.append(
            f"[WARN] edge-mark: {kind} 面积 {m['area']}px bbox={m['bbox']}"
            f"（底色 {m['bg']}）——外部渲染器的水印/UI 元素应清除后入图（rubric H3/D4）"
        )
    return out


def report_lines(health: dict[str, Any], *, tight_bbox: bool = False) -> list[str]:
    """体检报告行（人读）。

    ``tight_bbox=True``（调用方按绘图样式判）时追加一行**裁切提示**——成品尺寸是
    ``savefig.bbox=tight`` 的**裁切后包围盒**，与生产者自述的 ``figsize`` 名义值本就不同
    （backlog 20261015-221750：审查员为判定该差异需反解版面常量 + 核样式源码）。
    """
    lines = [
        "=== figure inspect ===",
        f"  尺寸            : {health['size_px'][0]}×{health['size_px'][1]} px",
        _size_mm_line(health),
        f"  色域占用        : 有彩色像素 {health['colored_ratio']:.1%}"
        f"；色相桶 {health['hue_bins_used']}/{health['hue_bins_total']}",
        f"  饱和像素占比    : {health['saturated_ratio']:.1%}",
        f"  边缘标记        : {len(health['edge_marks'])} 处",
    ]
    if tight_bbox:
        lines.append(
            "  [i] savefig.bbox=tight 已生效：成品尺寸为**裁切后包围盒**，"
            "非 figsize 名义值（差异随内容变化）"
        )
    for m in health["edge_marks"]:
        lines.append(
            f"    - 面积 {m['area']}px bbox={m['bbox']}"
            f"{'  「蓝黄成对」' if m['brand_like'] else ''}"
        )
    lines.extend(findings(health))
    return lines


def _size_mm_line(health: dict[str, Any]) -> str:
    """等效物理尺寸行（由 PNG dpi 元数据换算；缺 dpi 时明说「未知」而非留空）。"""
    size_mm = health.get("size_mm")
    dpi = health.get("dpi")
    if not size_mm:
        return "  等效物理尺寸    : 未知（PNG 无 dpi 元数据 → 无法换算 mm）"
    return (
        f"  等效物理尺寸    : {size_mm[0]}×{size_mm[1]} mm"
        f"（由 PNG dpi 元数据换算，@{round(float(dpi), 1):g} dpi）"
    )


def inspect_image(
    path: str | Path, *, band: int | None = None, dpi: float | None = None
) -> tuple[dict[str, Any], int]:
    """体检一张图：返回 ``(读数, rc)``；rc=1 表示有边缘标记（供脚本/审查门禁）。

    读不到文件时返回 ``({}, 2)``（用法/输入错误，与「体检出问题」区分开）。
    ``dpi`` 缺省时**自读 PNG 元数据**（显式传入可覆盖，供测试/无元数据场景）。
    """
    arr = load_rgb(path)
    if arr is None:
        return {}, 2
    health = image_health(arr, band=band, dpi=read_dpi(path) if dpi is None else dpi)
    return health, (1 if health["edge_marks"] else 0)
