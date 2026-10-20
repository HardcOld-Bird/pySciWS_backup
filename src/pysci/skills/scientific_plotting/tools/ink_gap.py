"""统一 ink-gap 测距：暗墨掩膜 → 2D 连体成分 → cKDTree 成对最小距。

H4「零遮挡/近而不交」的唯一测距入口（reviewer 量表与绘图技能同源引用）。**禁止**用
1D 投影 gap 合并、组件硬切边界或区域取宽来代替本工具的三类读数：

- **pixel**：两组成分的**墨迹像素**间 cKDTree 最小欧氏距（字形↔字形，严判遮挡）；
- **region**：两组成分并集的**包围盒**间距离（文字块↔文字块，判布局间距；y/x 范围
  重叠时退化为另一轴上的空隙）；
- **pairwise**：全图成分两两最近距表（排序输出，px/mm）。

与 H4 实案（`beam_scatter_combined_r5.png`）的对应：cb3 刻度标签列 ↔ 轴题块用
**region** 读 32 px（1.35 mm）；**pixel** 读到的是顶部 "0" 字形到轴题的 157 px——
后者不是布局间距，不能拿来判 H4。硬切标签组件会得到 1 px 假象（禁止）；1D 投影合并
会把上下不相邻的文字行并成一块而给出 15 px 级失真（禁止）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

#: 暗墨判据缺省：三通道 max<130 且通道极差<45（近黑/深灰文字）。
DEFAULT_MAX_CHANNEL = 130
DEFAULT_MAX_DELTA = 45


@dataclass(frozen=True)
class InkRegion:
    """一个 2D 连体墨迹成分（全图标注，不按 ROI 硬切）。"""

    label: int
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1（含端点）
    area_px: int
    pixels_xy: np.ndarray

    def intersects(self, roi: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = self.bbox_px
        rx0, ry0, rx1, ry1 = roi
        return x0 <= rx1 and x1 >= rx0 and y0 <= ry1 and y1 >= ry0


@dataclass(frozen=True)
class GapReading:
    a: tuple[int, int, int, int]
    b: tuple[int, int, int, int]
    distance_px: float
    distance_mm: float | None
    mode: str


def ink_mask(
    img: np.ndarray,
    *,
    max_channel: int = DEFAULT_MAX_CHANNEL,
    max_delta: int = DEFAULT_MAX_DELTA,
) -> np.ndarray:
    """暗墨掩膜：``max(rgb)<max_channel`` 且 ``max-min<max_delta``。"""
    arr = np.asarray(img)
    if arr.dtype == bool:
        if arr.ndim != 2:
            raise ValueError(f"布尔掩膜须为 2D：{arr.shape!r}")
        return arr
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
    else:
        raise ValueError(f"图像须为 HxW 或 HxWx3：{arr.shape!r}")
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return (mx < max_channel) & ((mx - mn) < max_delta)


def components(mask: np.ndarray) -> list[InkRegion]:
    """2D 8-连通成分（全图标注；不按 ROI/投影切分）。"""
    lab, n = ndimage.label(
        np.asarray(mask, dtype=bool), structure=np.ones((3, 3), bool)
    )
    out: list[InkRegion] = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        ys_grid, xs_grid = np.nonzero(lab[sl] == i)
        out.append(
            InkRegion(
                label=i,
                bbox_px=(
                    int(xs.start),
                    int(ys.start),
                    int(xs.stop - 1),
                    int(ys.stop - 1),
                ),
                area_px=int(ys_grid.size),
                pixels_xy=np.column_stack(
                    (xs_grid + xs.start, ys_grid + ys.start)
                ).astype(np.int32),
            )
        )
    return out


def select_components(
    regions: list[InkRegion], roi: tuple[int, int, int, int]
) -> list[InkRegion]:
    """按 ROI 选**整块**成分（bbox 相交即选，不裁剪像素）。"""
    return [r for r in regions if r.intersects(roi)]


def _bbox(regions: list[InkRegion]) -> tuple[int, int, int, int]:
    if not regions:
        raise ValueError("组内无墨迹成分")
    x0 = min(r.bbox_px[0] for r in regions)
    y0 = min(r.bbox_px[1] for r in regions)
    x1 = max(r.bbox_px[2] for r in regions)
    y1 = max(r.bbox_px[3] for r in regions)
    return (x0, y0, x1, y1)


def _pixels(regions: list[InkRegion]) -> np.ndarray:
    if not regions:
        raise ValueError("组内无墨迹成分")
    return np.concatenate([r.pixels_xy for r in regions], axis=0)


def pixel_gap_px(regions_a: list[InkRegion], regions_b: list[InkRegion]) -> float:
    """两组墨迹**像素**间最小欧氏距（cKDTree）。"""
    pa = _pixels(regions_a)
    pb = _pixels(regions_b)
    if pa.size == 0 or pb.size == 0:
        raise ValueError("组内无墨迹像素")
    return float(cKDTree(pa).query(pb, k=1)[0].min())


def region_gap_px(regions_a: list[InkRegion], regions_b: list[InkRegion]) -> float:
    """两组**文字块**（成分并集包围盒）间距离：各轴空隙的正交合成。"""
    xa0, ya0, xa1, ya1 = _bbox(regions_a)
    xb0, yb0, xb1, yb1 = _bbox(regions_b)
    # 边缘坐标差（H4 口径：两文字块外缘之间的距离；相接=0）。
    dx = max(0, xb0 - xa1, xa0 - xb1)
    dy = max(0, yb0 - ya1, ya0 - yb1)
    return float(np.hypot(dx, dy))


def _to_mm(px: float, dpi: float | None) -> float | None:
    return None if not dpi else px * 25.4 / float(dpi)


def gap_between_rois(
    img: np.ndarray,
    roi_a: tuple[int, int, int, int],
    roi_b: tuple[int, int, int, int],
    *,
    mode: str = "region",
    dpi: float | None = None,
    max_channel: int = DEFAULT_MAX_CHANNEL,
    max_delta: int = DEFAULT_MAX_DELTA,
) -> GapReading:
    """两个 ROI 的文字块间距。

    Args:
        mode: ``region``＝包围盒距离（布局间距，H4 口径）；``pixel``＝墨迹像素
            cKDTree 最小距（字形间距，严判遮挡）。
    """
    if mode not in ("region", "pixel"):
        raise ValueError(f"mode 须为 region|pixel：{mode!r}")
    regions = components(ink_mask(img, max_channel=max_channel, max_delta=max_delta))
    ra = select_components(regions, roi_a)
    rb = select_components(regions, roi_b)
    if not ra or not rb:
        raise ValueError(f"ROI 内无墨迹：a={len(ra)}, b={len(rb)}")
    dist = region_gap_px(ra, rb) if mode == "region" else pixel_gap_px(ra, rb)
    return GapReading(
        a=_bbox(ra),
        b=_bbox(rb),
        distance_px=dist,
        distance_mm=_to_mm(dist, dpi),
        mode=mode,
    )


def pairwise_gaps(
    img: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None = None,
    top: int | None = None,
    dpi: float | None = None,
    max_channel: int = DEFAULT_MAX_CHANNEL,
    max_delta: int = DEFAULT_MAX_DELTA,
) -> list[GapReading]:
    """全图（或 ``box`` 内整块成分）两两最近距表，按距离升序。

    用单棵 cKDTree 查询每个墨迹像素的最近异标签像素，取每对成分的最小值；不做
    1D 投影合并，也不按 box 硬切成分。
    """
    mask = ink_mask(img, max_channel=max_channel, max_delta=max_delta)
    if box is not None:
        # box 只做**整块筛选**（不切像素）：保留与 box 相交的成分，再整块重标。
        orig_lab, _n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        keep = {r.label for r in components(mask) if r.intersects(box)}
        mask = np.isin(orig_lab, sorted(keep))
    regions = components(mask)
    if len(regions) < 2:
        return []
    # 只用**边界像素**查询：最近异标签点必在边界上；块内部点会把 k 邻居全耗在同块。
    boundary = mask & ~ndimage.binary_erosion(
        mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    )
    lab, _n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    ys, xs = np.nonzero(boundary)
    bp = np.column_stack((xs, ys)).astype(np.int32)
    bl = lab[ys, xs]
    all_y, all_x = np.nonzero(mask)
    all_pix = np.column_stack((all_x, all_y)).astype(np.int32)
    all_lab = lab[all_y, all_x]
    tree = cKDTree(all_pix)
    best: dict[tuple[int, int], float] = {}
    for p, label in zip(bp, bl, strict=True):
        k = min(64, len(all_pix))
        while True:
            dist, idx = tree.query(p, k=k)
            hit = None
            for d, j in zip(np.atleast_1d(dist), np.atleast_1d(idx), strict=True):
                if all_lab[j] != label:
                    hit = float(d)
                    break
            if hit is not None:
                key = tuple(sorted((int(label), int(all_lab[j]))))
                if hit < best.get(key, float("inf")):
                    best[key] = hit
                break
            if k >= len(all_pix):
                break
            k = min(k * 4, len(all_pix))
    by_label = {r.label: r for r in regions}
    rows = [
        GapReading(
            a=by_label[a].bbox_px,
            b=by_label[b].bbox_px,
            distance_px=d,
            distance_mm=_to_mm(d, dpi),
            mode="pixel",
        )
        for (a, b), d in sorted(best.items(), key=lambda kv: kv[1])
    ]
    return rows[:top] if top else rows


def load_image(path: str | Path) -> np.ndarray:
    """读 PNG/JPG 为 RGB 数组（无 PIL 时抛 ImportError）。"""
    from PIL import Image  # noqa: PLC0415 - 仅图像读取需要

    return np.asarray(Image.open(path).convert("RGB"))
