"""imaging 的机械断言：measure 固定样例（解析可验）+ align 固定样例（已知平移）。

样例全部合成、确定性；期望值由均匀矩形的二阶矩解析式给出
（Var = (n²-1)/12，轴长 = 4√λ 口径，与 skimage regionprops 兼容）。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from pysci.skills.image_generation import imaging


def _two_blobs_gray() -> np.ndarray:
    """100x100 灰度：10x10 方块 + 20x6 横矩形 + 2x2 小点（噪声）。"""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[10:20, 10:20] = 255  # 方块：area=100
    gray[40:46, 50:70] = 255  # 横矩形：area=120
    gray[90:92, 90:92] = 255  # 小点：area=4（min_area 过滤）
    return gray


def test_measure_regions_fixed_sample():
    """面积/质心/外接框精确、长短轴/离心率解析可验；按面积降序。"""
    res = imaging.measure_regions(_two_blobs_gray(), min_area=10, thresh=127)
    assert [r["area_px"] for r in res] == [120, 100]

    rect, square = res
    # 横矩形（rows 40:46, cols 50:70）
    assert rect["centroid"] == (59.5, 42.5)  # (x, y)
    assert rect["bbox_xyxy"] == (50, 40, 70, 46)
    assert rect["axis_major"] == pytest.approx(23.07, abs=0.02)  # 4*sqrt((400-1)/12)
    assert rect["axis_minor"] == pytest.approx(6.83, abs=0.02)  # 4*sqrt((36-1)/12)
    assert rect["aspect"] == pytest.approx(3.376, abs=0.01)
    assert rect["eccentricity"] == pytest.approx(0.9551, abs=0.001)
    # 方块（rows/cols 10:20）：轴等长、离心率 0
    assert square["centroid"] == (14.5, 14.5)
    assert square["bbox_xyxy"] == (10, 10, 20, 20)
    assert square["axis_major"] == pytest.approx(11.49, abs=0.02)  # 4*sqrt(8.25)
    assert square["axis_minor"] == pytest.approx(11.49, abs=0.02)
    assert square["aspect"] == pytest.approx(1.0, abs=0.01)
    assert square["eccentricity"] == pytest.approx(0.0, abs=1e-6)


def test_measure_min_area_filter_and_empty():
    assert len(imaging.measure_regions(_two_blobs_gray(), min_area=4, thresh=127)) == 3
    empty = np.zeros((20, 20), dtype=np.uint8)
    assert imaging.measure_regions(empty, thresh=127) == []


def test_measure_mask_and_otsu_paths(tmp_path):
    """显式 mask 优先；无 thresh 时 Otsu 自动阈值。"""
    mask_p = tmp_path / "mask.png"
    mask = _two_blobs_gray()
    mask[90:92, 90:92] = 0  # mask 里去掉小点
    Image.fromarray(mask).save(mask_p)
    res = imaging.measure_regions(np.zeros((100, 100), np.uint8), mask=mask_p, min_area=10)
    assert sorted(r["area_px"] for r in res) == [100, 120]

    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:] = 30  # 低对比背景
    gray[10:20, 10:20] = 220
    res2 = imaging.measure_regions(gray, min_area=10)  # Otsu 路径
    assert [r["area_px"] for r in res2] == [100]


# ---------------------------------------------------------------------------
# align：固定样例（已知整数/亚像素平移）
# ---------------------------------------------------------------------------
def _fourier_shift(img: np.ndarray, d: tuple[float, float]) -> np.ndarray:
    """精确分数平移（scipy 约定：``fourier_shift`` 作用在**傅立叶谱**上）。

    配准估计器的标准合成验证输入：连续带限平移、无插值误差。
    """
    return np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(img), d)).real


def _pattern() -> np.ndarray:
    """三个高斯斑的平滑纹理图（无随机、远离边界）。"""
    yy, xx = np.mgrid[0:64, 0:64]
    return (
        np.exp(-(((yy - 20) ** 2 + (xx - 25) ** 2) / 50.0))
        + 0.7 * np.exp(-(((yy - 45) ** 2 + (xx - 45) ** 2) / 30.0))
        + 0.5 * np.exp(-(((yy - 30) ** 2 + (xx - 15) ** 2) / 80.0))
    )


def test_align_integer_shift():
    """src = ref 平移 (3, -2)（样条重采样）→ align 返回把它移回的 (-3, +2)。"""
    ref = _pattern()
    src = ndi.shift(ref, (3, -2), order=3, mode="constant")
    res = imaging.align(src, ref, upsample=10)
    assert res["shift_yx"][0] == pytest.approx(-3.0, abs=0.1)
    assert res["shift_yx"][1] == pytest.approx(2.0, abs=0.1)
    assert res["shift_xy"] == [res["shift_yx"][1], res["shift_yx"][0]]
    assert res["rms_error"] < 0.05


def test_align_subpixel_fourier():
    """亚像素平移 (2.5, -1.25)（精确傅立叶平移）→ 恢复 (-2.5, +1.25)，≤0.06px。"""
    ref = _pattern()
    src = _fourier_shift(ref, (2.5, -1.25))
    res = imaging.align(src, ref, upsample=10)
    assert res["shift_yx"][0] == pytest.approx(-2.5, abs=0.06)
    assert res["shift_yx"][1] == pytest.approx(1.25, abs=0.06)


def test_align_subpixel_spline_resampling():
    """亚像素平移（样条重采样=真实重采样路径）→ 精度 ≤0.2px。

    带限图 + 样条重采样下亚像素估计天然欠约束（相位信息集中在低频），0.2px 是
    该口径下的合理容差；精确验证走 :func:`test_align_subpixel_fourier`。
    """
    ref = _pattern()
    src = ndi.shift(ref, (2.5, -1.25), order=3, mode="constant")
    res = imaging.align(src, ref, upsample=10)
    assert res["shift_yx"][0] == pytest.approx(-2.5, abs=0.2)
    assert res["shift_yx"][1] == pytest.approx(1.25, abs=0.2)


def test_align_integer_shift_without_upsample():
    """upsample=1 只给整像素。"""
    ref = _pattern()
    src = ndi.shift(ref, (0, 5), order=3, mode="constant")
    res = imaging.align(src, ref, upsample=1)
    assert res["shift_yx"] == [0.0, -5.0]


def test_align_dest_writes_aligned_rgb(tmp_path):
    """给了 dest：应用平移写出图（像素级核对对齐结果）。"""
    ref = _pattern() * 255.0  # 0–255 量纲（落盘 uint8 不失真）
    src = ndi.shift(ref, (2, -3), order=3, mode="constant")
    dest = tmp_path / "aligned.png"
    res = imaging.align(src, ref, dest=dest, upsample=10)
    assert dest.is_file()
    assert "out" in res
    with np.errstate(all="ignore"):
        aligned = np.asarray(Image.open(dest).convert("L"), dtype=np.float64)
    core = (slice(8, 56), slice(8, 56))  # 避开平移引入的黑边
    assert np.sqrt(np.mean((aligned[core] - ref[core]) ** 2)) < 15.0


def test_align_shape_mismatch_raises():
    with pytest.raises(imaging.ImagingError):
        imaging.align(np.zeros((10, 10)), np.zeros((10, 12)))
