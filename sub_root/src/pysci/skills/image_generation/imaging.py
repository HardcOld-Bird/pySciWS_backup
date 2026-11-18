"""像素测量与配准：measure（连通域量化）+ align（相位相关亚像素配准）。

科研用途：**生成图/实验图的像素级量化与对齐**——measure 把二值区域的面积/质心/
长短轴/离心率数出来（配 scale bar 换算物理尺寸、核对生成图比例）；align 求两张图的
亚像素平移量（图层拆分后归位、组图/叠加对齐）。服务 ⑦域生成图后处理；
数据图（③域）不走这里。

ADR（2026-11-17，用户批准）：旧 pySciWS 实现基于 scikit-image
（``measure.regionprops`` / ``registration.phase_cross_correlation``）；本重建改为
**scipy.ndimage + numpy.fft 单轨实现**——scipy 已是核心依赖，为两个算子不值得拉动
skimage+opencv 全家桶（极简原则）。语义与 skimage **兼容**：

- 连通域 8 连通（``skimage.measure.label(connectivity=2)`` 同款）；
- ``axis_major/minor = 4·√λ``（λ 为二阶中心矩协方差特征值，regionprops 的直径口径）；
- ``eccentricity = √(1 - λ2/λ1)``；
- ``align`` 返回的 ``shift`` 是"把 src 平移 shift 后与 ref 重合"
  （``phase_cross_correlation(ref, src)`` 同款约定），亚像素走 Guizar-Sicairos
  上采样 DFT 细化。skimage/opencv 将来到位可无痛切换（输出字段不变）。

用法::

    from pysci.skills.image_generation import imaging

    regions = imaging.measure_regions("figure.png", min_area=50)
    res = imaging.align("layer.png", "base.png", dest="layer_aligned.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


class ImagingError(RuntimeError):
    """测量/配准调用失败（图像缺失、尺寸不一致、空掩膜等）。"""


# ---------------------------------------------------------------------------
# I/O 辅助
# ---------------------------------------------------------------------------
def _load_gray(src: str | Path | np.ndarray, *, mode: str = "L") -> np.ndarray:
    """读图为 numpy 数组（路径走 Pillow；数组原样返回）。"""
    if isinstance(src, np.ndarray):
        return src
    p = Path(src).expanduser()
    if not p.is_file():
        raise ImagingError(f"图像不存在：{p}")
    with Image.open(p) as im:
        return np.asarray(im.convert(mode))


def _otsu(gray: np.ndarray) -> float:
    """Otsu 自动阈值（直方图最大化类间方差；常量图返回中值）。"""
    g = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0 or np.count_nonzero(hist) <= 1:
        return float(g.flat[0])
    prob = hist / total
    omega = np.cumsum(prob)
    levels = np.arange(256, dtype=np.float64)
    mu_total = float((prob * levels).sum())
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_between = np.cumsum(prob * levels)
        sigma_b = np.where(denom > 0, (mu_total * omega - mu_between) ** 2 / denom, -1.0)
    return float(levels[int(np.argmax(sigma_b))])


# ---------------------------------------------------------------------------
# measure：连通域量化
# ---------------------------------------------------------------------------
def measure_regions(
    src: str | Path | np.ndarray,
    *,
    min_area: int = 50,
    thresh: float | None = None,
    mask: str | Path | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """连通域测量：面积、质心、外接框、长轴/短轴、离心率（8 连通）。

    二值化来源（优先级）：``mask``（>127 为前景）> ``thresh`` 固定阈值 > Otsu 自动阈值。
    输出按面积降序；字段与旧 skimage 口径兼容（见模块 docstring 的 ADR）。

    Args:
        src: 源图路径或灰度数组。
        min_area: 最小面积（像素），小于该值的区域忽略（去噪）。
        thresh: 固定阈值（>thresh 为前景）。
        mask: 显式二值掩膜（路径或数组）。

    Returns:
        区域字典列表：``label / area_px / centroid(x,y) / bbox_xyxy /
        axis_major / axis_minor / aspect / eccentricity``。
    """
    if mask is not None:
        binary = _load_gray(mask) > 127
    else:
        gray = _load_gray(src)
        binary = gray > (float(thresh) if thresh is not None else _otsu(gray))

    structure = np.ones((3, 3), dtype=bool)  # 8 连通
    labeled, n = ndi.label(binary, structure=structure)
    if n == 0:
        return []
    objs = ndi.find_objects(labeled)
    ys, xs = np.nonzero(binary)
    labels_at = labeled[ys, xs]

    out: list[dict[str, Any]] = []
    for lab in range(1, n + 1):
        sel = labels_at == lab
        area = int(sel.sum())
        if area < int(min_area):
            continue
        cy = float(ys[sel].mean())
        cx = float(xs[sel].mean())
        dy = ys[sel] - cy
        dx = xs[sel] - cx
        mu20 = float((dx * dx).sum()) / area
        mu02 = float((dy * dy).sum()) / area
        mu11 = float((dx * dy).sum()) / area
        # 协方差特征值：λ = (mu20+mu02)/2 ± sqrt(((mu20-mu02)/2)^2 + mu11^2)
        tmp = np.hypot((mu20 - mu02) / 2.0, mu11)
        lam1 = (mu20 + mu02) / 2.0 + tmp
        lam2 = (mu20 + mu02) / 2.0 - tmp
        major = 4.0 * float(np.sqrt(max(lam1, 0.0)))
        minor = 4.0 * float(np.sqrt(max(lam2, 0.0)))
        ecc = float(np.sqrt(max(0.0, 1.0 - lam2 / lam1))) if lam1 > 0 else 0.0
        sl = objs[lab - 1]
        min_row, max_row = sl[0].start, sl[0].stop
        min_col, max_col = sl[1].start, sl[1].stop
        out.append(
            {
                "label": lab,
                "area_px": area,
                "centroid": (round(cx, 2), round(cy, 2)),
                "bbox_xyxy": (int(min_col), int(min_row), int(max_col), int(max_row)),
                "axis_major": round(major, 2),
                "axis_minor": round(minor, 2),
                "aspect": round(major / minor, 3) if minor > 0 else None,
                "eccentricity": round(ecc, 4),
            }
        )
    return sorted(out, key=lambda d: d["area_px"], reverse=True)


# ---------------------------------------------------------------------------
# align：相位相关亚像素配准
# ---------------------------------------------------------------------------
def align(
    src: str | Path | np.ndarray,
    ref: str | Path | np.ndarray,
    dest: str | Path | None = None,
    *,
    upsample: int = 10,
) -> dict[str, Any]:
    """相位相关配准：求把 ``src`` 平移多少后与 ``ref`` 重合（亚像素）。

    约定与 ``skimage.registration.phase_cross_correlation(ref, src)`` 相同：
    返回的 ``shift`` 应用于 src（``scipy.ndimage.shift(src, shift)``）即与 ref 对齐。
    亚像素精度由 Guizar-Sicairos 上采样 DFT 在粗峰值 ±1 像素邻域细化得到。

    Args:
        src: 待配准图（路径或灰度数组）。
        ref: 参考图（与 src 尺寸一致）。
        dest: 给了则应用平移并把对齐后的 **RGB** 图写出（灰度进 RGB 出）。
        upsample: 亚像素细化倍率（1 = 只给整像素）。

    Returns:
        ``{"shift_yx": [dy, dx], "shift_xy": [dx, dy], "rms_error": float}``
        （``rms_error`` 为对齐后与 ref 的灰度 RMS 差）；写盘时附 ``"out": 路径``。

    Raises:
        ImagingError: 两图尺寸不一致或图像为空。
    """
    a = np.real(_load_gray(src)).astype(np.float64)  # moving（.real 容忍复数谱输入）
    b = np.real(_load_gray(ref)).astype(np.float64)  # reference
    if a.shape != b.shape:
        raise ImagingError(f"配准要求两图尺寸一致：src {a.shape} vs ref {b.shape}")

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)  # ifft 峰值 = 把 src 平移到 ref 所需的 shift
    power = np.abs(cross)
    # 相位门限（coherence gating）：功率低于峰值 1e-3 的 bin 其相位由插值/数值噪声
    # 主导（带限图——平滑渐变/AI 插画——会整段稀释真峰），归一化时置零不参与。
    # 宽带自然图几乎不受影响（被掩的只是低于峰 3 个数量级的 bin）。
    gate = power > power.max() * 1e-3
    cross = np.divide(cross, power, out=np.zeros_like(cross), where=gate)

    cc = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    shifts = [
        float(p if p <= n // 2 else p - n) for p, n in zip(peak, cc.shape, strict=True)
    ]

    up = max(1, int(upsample))
    if up > 1:
        # 粗峰值 ±1 像素邻域、1/up 像素步进的上采样 DFT 细化（Guizar-Sicairos）。
        row_pts = shifts[0] + np.arange(-up, up + 1) / up
        col_pts = shifts[1] + np.arange(-up, up + 1) / up
        fine = _sample_idft(cross, row_pts, col_pts)
        i, j = np.unravel_index(int(np.argmax(np.abs(fine))), fine.shape)
        shifts = [float(row_pts[i]), float(col_pts[j])]

    moved = ndi.shift(a, (shifts[0], shifts[1]), order=1, mode="constant", cval=0.0)
    rms = float(np.sqrt(np.mean((moved - b) ** 2)))
    result: dict[str, Any] = {
        "shift_yx": [round(v, 4) for v in shifts],
        "shift_xy": [round(shifts[1], 4), round(shifts[0], 4)],
        "rms_error": round(rms, 6),
    }
    if dest is not None:
        rgb = np.real(
            np.asarray(
                Image.open(Path(src)).convert("RGB")
                if not isinstance(src, np.ndarray)
                else src,
                dtype=np.float64,
            )
        )
        moved_rgb = ndi.shift(
            rgb, (shifts[0], shifts[1], 0) if rgb.ndim == 3 else (shifts[0], shifts[1]),
            order=1, mode="constant", cval=0.0,
        )
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(moved_rgb, 0, 255).astype(np.uint8)).save(out)
        result["out"] = str(out)
    return result


def _sample_idft(
    spectrum: np.ndarray, row_pts: np.ndarray, col_pts: np.ndarray
) -> np.ndarray:
    """在**分数**坐标上直接求逆 DFT 值（矩阵乘法实现，粗细两档共用）。

    ``out[i, j] = Σ_{y,x} spectrum[y,x] · exp(2πi(νy·row_i + νx·col_j))``
    （与 ``np.fft.ifft2`` 同号约定）。**频率必须用 fftfreq 的有符号栅格**：整数点上
    有符号/未移位索引两口径重合，但分数点上未移位索引会破坏共轭对称（±k 对的
    相位差 e^{2πiτ} ≠ 1），只有有符号频率给出真实连续相位插值（= 连续相关面）。
    """
    ky = np.fft.fftfreq(spectrum.shape[0])
    kx = np.fft.fftfreq(spectrum.shape[1])
    er = np.exp(2j * np.pi * np.outer(ky, row_pts))  # (H, P)
    ec = np.exp(2j * np.pi * np.outer(kx, col_pts))  # (W, Q)
    return er.T @ spectrum @ ec
