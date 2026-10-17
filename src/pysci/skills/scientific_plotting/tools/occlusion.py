"""零遮挡像素判据：文字墨迹 ∩ 目标蒙版（彩色数据区 / 图线真身）+ 覆盖与间距读数。

这是 reviewer 量表 H4「零遮挡铁律」的**可执行面**，也是「墨迹∩彩色/框线真身」判据的
**唯一实现**（backlog 20261016-014160：原先 figure/reviewer 两 pod 的 bench 各写一份，
口径会漂移）。pod 的 ``bench/`` 只留草稿与裁剪输出，判据一律调本模块（经 `figures overlap`）。

**三版图像**：受审图 ``A``，以及把标注/刻度文字**置白或删除**后重渲染（版面不变）的 ``W``：

- **墨迹蒙版** = ``|A − W|.max(axis=2) > ink_tol``——差出来的正是**文字字形**本身；
- **彩色数据区蒙版** = ``chroma(A) > chroma_min``——场图色块（非中性、高彩度）；
- **图线真身蒙版**（在 ``W`` 上取）= ``lum(W) < line_lum_max 且 chroma(W) < line_chroma_max``
  ——``W`` 无文字遮挡，故暗而低彩度的像素就是真实弧/框线/网格。

**方法学的坑（照此选材）**：

- **「置白版」假阴性**：只把文字置白而不删，字形抗锯齿边缘仍残留 → 墨迹蒙版偏小、判据偏松；
- **「差分版」假阳性**：两版若有亚像素位移，位移边缘被算进墨迹 → 判据偏严；
- 故**取「删文字版」作真值面**（在其上直接读框线真身），且 ``A`` 与 ``W`` 须**同一次渲染导出**
  （同版式），否则版面漂移污染判据。

**缓存自防护**（本模块的硬纪律）：``W`` 是**派生缓存**，图更新后旧 ``W`` 会给出假读数
（实案：旧缓存把墨迹↔弧线的最短距拉回 1.0px 的假 FAIL）。故入口先比 ``mtime(W) ≥ mtime(A)``：
过期且给了 ``rebuild`` → 先重生成再复检；仍过期或未给 ``rebuild`` → **rc=2 显式报错退出**，
**绝不用过期缓存出读数**（假 FAIL/假 PASS 都杜绝）。
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .imgcheck import chroma, load_rgb

#: 判据缺省阈值（与 bench 现场口径一致，收敛后此处为唯一真源）。
INK_TOL: int = 30  # 墨迹：|A−W| 逐通道最大差
CHROMA_MIN: float = 40.0  # 彩色数据区：色度下限
LINE_LUM_MAX: float = 60.0  # 图线真身：亮度上限（暗）
LINE_CHROMA_MAX: float = 40.0  # 图线真身：色度上限（低彩）

#: 像素 → 毫米的缺省换算（600 dpi：25.4/600）。给 ``--mm-per-px`` 时以之为准。
MM_PER_PX_600DPI: float = 25.4 / 600.0


def luminance(arr: np.ndarray) -> np.ndarray:
    """逐像素亮度（RGB 三通道均值）。"""
    return arr.astype(float).mean(axis=2)


def ink_mask(img: np.ndarray, white: np.ndarray, *, tol: int = INK_TOL) -> np.ndarray:
    """文字墨迹蒙版：``|img − white|`` 逐通道最大差 > ``tol``。

    Args:
        img: 受审图 ``(H, W, 3)``。
        white: 同尺寸的**删文字版**（文字置白后重渲染）。
        tol: 差分阈值（缺省 :data:`INK_TOL`）。

    Raises:
        ValueError: 两图尺寸不一致（版面漂移的信号，宁可报错也不出假读数）。
    """
    if img.shape != white.shape:
        raise ValueError(
            f"受审图与删文字版尺寸不一致：{img.shape} vs {white.shape}"
            "（版面漂移？两版须同一次渲染导出）"
        )
    return np.abs(img.astype(int) - white.astype(int)).max(axis=2) > tol


def colored_mask(img: np.ndarray, *, chroma_min: float = CHROMA_MIN) -> np.ndarray:
    """彩色数据区蒙版：色度 > ``chroma_min``。"""
    return chroma(img) > chroma_min


def line_mask(
    white: np.ndarray,
    *,
    lum_max: float = LINE_LUM_MAX,
    chroma_max: float = LINE_CHROMA_MAX,
) -> np.ndarray:
    """图线真身蒙版：**在删文字版上**取「暗且低彩度」像素（真实弧/框线/网格）。"""
    return (luminance(white) < lum_max) & (chroma(white) < chroma_max)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return mask
    x0, x1, y0, y1 = box
    return mask[y0:y1, x0:x1]


def overlap_count(
    a: np.ndarray, b: np.ndarray, *, box: tuple[int, int, int, int] | None = None
) -> int:
    """两蒙版的交叠像素数（``box`` 为可选 ``(x0,x1,y0,y1)`` 裁切窗）。"""
    return int(np.logical_and(_crop(a, box), _crop(b, box)).sum())


def min_distance_px(
    frm: np.ndarray,
    to: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None = None,
    chunk: int = 300,
) -> float | None:
    """``frm`` 蒙版像素到 ``to`` 蒙版像素的**最短欧氏距（px）**；任一侧为空 → ``None``。

    分块计算（``chunk`` 行一批）避免全对全的 ``O(N·M)`` 内存爆炸。
    """
    ys, xs = np.nonzero(_crop(frm, box))
    ys2, xs2 = np.nonzero(_crop(to, box))
    if len(xs) == 0 or len(xs2) == 0:
        return None
    q = np.column_stack([xs2, ys2]).astype(float)
    best = float("inf")
    p = np.column_stack([xs, ys]).astype(float)
    for i in range(0, len(p), chunk):
        d = np.sqrt(((p[i : i + chunk, None, :] - q[None, :, :]) ** 2).sum(-1))
        best = min(best, float(d.min()))
    return best


# ---------------------------------------------------------------------------
# 缓存自防护
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheCheck:
    """缓存新鲜度判定结果。"""

    ok: bool
    stale: tuple[Path, ...]
    missing: tuple[Path, ...]
    message: str


def _stale_and_missing(img: Path, caches: Sequence[Path]) -> tuple[list, list]:
    ref = img.stat().st_mtime
    missing = [c for c in caches if not c.exists()]
    stale = [c for c in caches if c.exists() and c.stat().st_mtime < ref]
    return stale, missing


def check_caches(
    img: Path,
    caches: Sequence[Path],
    *,
    rebuild: str | None = None,
    runner=subprocess.run,
) -> CacheCheck:
    """删文字版缓存的新鲜度守卫：过期/缺失 → 重生成（若给 ``rebuild``）或显式报错。

    Args:
        img: 受审图；缓存须**不早于**它的 mtime，否则视为过期。
        caches: 派生缓存（删文字版 PNG）。
        rebuild: 可选的重生成命令（shell 串）；给了就先跑一次再复检。
        runner: 子进程执行器（测试注入；缺省 :func:`subprocess.run`）。

    Returns:
        :class:`CacheCheck`；``ok=False`` 时调用方**必须** rc≠0 退出，不得用过期缓存出读数。
    """
    stale, missing = _stale_and_missing(img, caches)
    if (stale or missing) and rebuild:
        runner(rebuild, shell=True, check=False)
        stale, missing = _stale_and_missing(img, caches)
    if missing:
        return CacheCheck(
            False,
            tuple(stale),
            tuple(missing),
            f"缓存缺失：{'、'.join(str(c) for c in missing)}"
            + ("（--rebuild 后仍在）" if rebuild else "；请先生成或给 --rebuild"),
        )
    if stale:
        return CacheCheck(
            False,
            tuple(stale),
            (),
            f"缓存过期（早于受审图 {img.name}）：{'、'.join(str(c) for c in stale)}"
            + ("（--rebuild 后仍过期）" if rebuild else "；请重生成或给 --rebuild"),
        )
    return CacheCheck(True, (), (), "缓存新鲜")


def render_ink_verdict(
    img: np.ndarray,
    white: np.ndarray,
    *,
    ink_tol: int = INK_TOL,
    chroma_min: float = CHROMA_MIN,
    line_lum_max: float = LINE_LUM_MAX,
    line_chroma_max: float = LINE_CHROMA_MAX,
    box: tuple[int, int, int, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """对一层删文字版算出零遮挡读数（覆盖计数 + 最短距）。

    ``progress`` 在每个阶段边界被调用（``蒙版``/``交集``/``最近距``/``汇总``）——供 CLI
    打印可观察的阶段进度（大图上``最近距``可能耗时数分钟，backlog 20261017-101924）。

    Returns:
        ``{ink_px, colored_px, line_px, ink_colored_px, ink_line_px, ink_line_min_px,
        pass}``——``pass`` 要求两个交叠计数均为 0。
    """

    def _p(stage: str) -> None:
        if progress is not None:
            progress(stage)

    _p("蒙版")
    ink = ink_mask(img, white, tol=ink_tol)
    colored = colored_mask(img, chroma_min=chroma_min)
    line = line_mask(white, lum_max=line_lum_max, chroma_max=line_chroma_max)
    _p("交集")
    ic = overlap_count(ink, colored, box=box)
    il = overlap_count(ink, line, box=box)
    _p("最近距")
    dist = min_distance_px(ink, line, box=box)
    _p("汇总")
    return {
        "ink_px": int(_crop(ink, box).sum()),
        "colored_px": int(_crop(colored, box).sum()),
        "line_px": int(_crop(line, box).sum()),
        "ink_colored_px": ic,
        "ink_line_px": il,
        "ink_line_min_px": dist,
        "pass": ic == 0 and il == 0,
    }


def downsample_rgb(arr: np.ndarray, factor: int) -> np.ndarray:
    """**点采样**降采样（``factor`` 整数倍；``1`` 原样返回）——粗判口径。

    大图（≥4000 px 宽）全分辨率判据可达数分钟；``--downsample 2`` 先粗判可把耗时降到
    ~1/16（计数与最近距都是双蒙版对的乘积）。**最终结论仍以全分辨率为准**。
    """
    k = max(1, int(factor))
    return arr if k == 1 else np.ascontiguousarray(arr[::k, ::k])


def load_or_raise(path: Path) -> np.ndarray:
    """读 RGB 数组；读不到抛 ``OSError``（判据面不静默吞坏文件）。"""
    arr = load_rgb(path)
    if arr is None:
        raise OSError(f"无法读取图像：{path}")
    return arr
